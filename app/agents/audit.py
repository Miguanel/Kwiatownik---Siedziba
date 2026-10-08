"""Audytor wiedzy: skanuje baze wiedzy Kwiatownika i pisze raport oceny z zaleceniami.

Co czyta: pliki roslin Kwiatownika2 (data/plants/*.json - recznie wpisane rozdzialy, blok "wiedza" z sieci ze zrodlami,
"scalone", zdjecia), przepisy (data/przepisy/*.json) i stan bazy Siedziby (informacje, zapytania, kraje).

Ocena = kod (metryki, 60%) + ekspert LLM (40%). Metryki kodem dla kazdej rosliny (0-100):
  - kompletnosc      - pokrycie schematu strony (app/knowledge/schema.coverage),
  - trudna_wiedza    - informacje ze zrodel spoza PL/EN, ze stron innych niz Wikipedia, z Chin/Japonii,
                       z sekcji trudno dostepnych (medycyna Wschodu, barwienie, kosmetyka, wierzenia, nazwy ludowe),
  - opowiesci        - ciekawostki, legendy, wierzenia, historia, nazwy ludowe (reczne + z sieci, zwlaszcza zagraniczne),
  - przepisy         - przepisy niekulinarne z ta roslina,
  - zrodla           - liczba zrodel z adresem, zdjecia, scalenie z rozdzialami.
Zalecenia: reguly (zawsze) + ekspert LLM (priorytety, nowe zalecenia ze slownika, kontrola kodem).
"""
import json
import logging
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path

from sqlmodel import Session, col, func, select

from app.agents import store
from app.agents.tasks import EXPERT_KINDS, KINDS, describe
from app.config import settings
from app.db import engine
from app.knowledge import schema
from app.knowledge.sections import SECTION_TITLES_PL, norm_text
from app.models import AgentRun, AgentTask, Item, ItemKind, ItemStatus, Plant, PlantFact, PlantQuery, SearchQuery

log = logging.getLogger(__name__)

COMMON_LANGS = {"pl", "en"}
ASIA_LANGS = {"zh", "ja", "ko"}
HARD_SECTIONS = {"medycyna_wschodu", "barwienie", "kosmetyka", "nazwy_ludowe", "kultura"}
STORY_SECTIONS = {"historia", "kultura", "nazwy_ludowe", "ciekawostki"}
CRITERIA = {"kompletnosc": "Kompletnosc schematu", "trudna_wiedza": "Wiedza trudno dostepna",
            "opowiesci": "Opowiesci i wierzenia", "przepisy": "Przepisy niekulinarne", "zrodla": "Zrodla i wiarygodnosc"}
WEIGHTS = {"kompletnosc": 0.20, "trudna_wiedza": 0.30, "opowiesci": 0.25, "przepisy": 0.10, "zrodla": 0.15}
FLAG_LABELS = {
    "bez_wiedzy_z_sieci": "brak wiedzy z sieci", "malo_opowiesci": "malo opowiesci", "brak_azji": "brak zrodel z Azji",
    "brak_wschodu": "brak medycyny Wschodu", "brak_barwienia": "brak barwienia", "brak_kosmetyki": "brak kosmetyki",
    "brak_bezpieczenstwa": "brak przeciwwskazan", "do_scalenia": "wiedza nie scalona", "bez_zdjec": "bez zdjec",
    "male_pokrycie": "pokrycie < 60%", "bez_przepisow": "bez przepisow niekulinarnych",
}
EXPERT_WEIGHT = 0.4
SKIP_RECIPE_FILES = ("wzorzec_przepisu.json",)


# ------------------------------------------------------------------ odczyt bazy wiedzy Kwiatownika
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def read_plants(plants_dir: Path) -> dict[str, dict]:
    """Pliki roslin w nowym formacie (glowny folder; archiwum w podfolderach nie jest publikowane)."""
    out = {}
    d = Path(plants_dir)
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        data = _read_json(f)
        if isinstance(data, dict):
            out[f.stem] = data
    return out


def read_recipes(przepisy_dir: Path) -> list[dict]:
    """Wszystkie przepisy z plikow Kwiatownika (bez wzorca), bez powtorzen (ten sam id albo tytul)."""
    out, seen = [], set()
    d = Path(przepisy_dir)
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        if f.name in SKIP_RECIPE_FILES:
            continue
        data = _read_json(f)
        rows = data if isinstance(data, list) else next((v for v in (data or {}).values() if isinstance(v, list)), []) \
            if isinstance(data, dict) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            key = str(r.get("id") or "") or norm_text(str(r.get("tytul") or ""))
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({**r, "_plik": f.name})
    return out


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def recipe_category(r: dict) -> str:
    typ = str(r.get("typ") or "").lower()
    text = norm_text(" ".join([typ, str(r.get("tytul") or ""), str(r.get("metoda") or "")]))
    if typ.startswith("kulinar"):
        return "kulinarne"
    if any(w in text for w in ("barwi", "farbow", "barwnik", "dye")):
        return "barwienie"
    if "kosmet" in text or "rzemiosl" in typ:
        return "kosmetyka i rzemioslo"
    if "nalew" in text or "likier" in text or "miod pitny" in text:
        return "nalewki i trunki"
    if typ.startswith("medyczne"):
        return "ziololecznictwo"
    return (typ.split("_")[0] or "inne") if typ else "inne"


def from_abroad(r: dict) -> bool:
    """Przepis z zagranicznej strony (zebrany przez Siedzibe, ze zrodlem)."""
    return bool(r.get("siedziba") or r.get("zrodla") or "http" in str(r.get("pochodzenie") or ""))


def _stem(word: str) -> str:
    """Rdzen slowa odporny na odmiane (tarnina/tarniny, krwawnik/krwawnika): bez 2 ostatnich liter, 4-6 znakow."""
    return word[:min(6, max(4, len(word) - 2))]


class PlantIndex:
    """Ktore rosliny wystepuja w przepisie (slug / nazwa rosliny / skladniki)."""

    def __init__(self, plants: dict[str, dict]):
        self.stems: dict[str, list[str]] = {}
        self.by_name: dict[str, str] = {}
        counts: Counter = Counter()
        for pid, data in plants.items():
            name = norm_text(str(data.get("nazwa_pl") or pid.replace("_", " ")))
            self.by_name[name] = pid
            stems = [_stem(w) for w in name.split() if len(w) >= 3]
            self.stems[pid] = stems
            counts.update(set(stems))
        # rdzen wystepujacy tylko w jednej roslinie (np. "tarni") wystarczy; "pospol", "zwycz" - nie
        self.unique = {s for s, n in counts.items() if n == 1 and len(s) >= 5}

    def match(self, r: dict) -> set[str]:
        out = set()
        slug = str(r.get("slug") or "")
        if slug in self.stems:
            out.add(slug)
        name = norm_text(str(r.get("roslina") or ""))
        if name in self.by_name:
            out.add(self.by_name[name])
        words = norm_text(" ".join(_strings([r.get("tytul"), r.get("skladniki"), r.get("roslina")]))).split()
        for pid, stems in self.stems.items():
            if not stems or pid in out:
                continue
            hits = [s for s in stems if any(w.startswith(s) for w in words)]
            if len(hits) == len(stems) or any(s in self.unique for s in hits):
                out.add(pid)
        return out


def lang2(code) -> str:
    return (str(code or "?").lower().replace("_", "-").split("-")[0]) or "?"


# ------------------------------------------------------------------ metryki jednej rosliny
def plant_metrics(pid: str, data: dict, recipes: int = 0, noncul: int = 0,
                  web_counts: dict[str, int] | None = None) -> dict:
    cov = schema.coverage(data, web_counts or {})
    status = {r["path"]: r["status"] for r in cov["sloty"]}
    wiedza = data.get("wiedza") if isinstance(data.get("wiedza"), dict) else {}
    zr = {z.get("nr"): z for z in wiedza.get("zrodla") or [] if isinstance(z, dict)}
    points = hard = foreign = asia = web_pages = stories_web = stories_foreign = 0
    langs: set[str] = set()
    lang_points: Counter = Counter()
    secs: Counter = Counter()
    for sec in wiedza.get("sekcje") or []:
        key = sec.get("klucz")
        for pt in sec.get("punkty") or []:
            points += 1
            secs[key] += 1
            srcs = [zr[n] for n in pt.get("zrodla") or [] if n in zr]
            ls = {lang2(z.get("jezyk")) for z in srcs}
            langs |= ls
            lang_points.update(ls - COMMON_LANGS - {"?"})
            is_foreign = bool(ls - COMMON_LANGS - {"?"})
            is_web = any(z.get("typ") not in (None, "wikipedia") for z in srcs)
            foreign += is_foreign
            asia += bool(ls & ASIA_LANGS)
            web_pages += is_web
            hard += bool(is_foreign or is_web or key in HARD_SECTIONS)
            if key in STORY_SECTIONS:
                stories_web += 1
                stories_foreign += is_foreign
    if not points and wiedza.get("fakty"):                 # starszy zapis: luzna lista informacji
        for f in wiedza["fakty"]:
            points += 1
            lg = lang2(f.get("jezyk"))
            langs.add(lg)
            if lg not in COMMON_LANGS | {"?"}:
                lang_points[lg] += 1
            secs[f.get("sekcja")] += 1
            is_foreign = lg not in COMMON_LANGS | {"?"}
            foreign += is_foreign
            asia += lg in ASIA_LANGS
            hard += bool(is_foreign or f.get("sekcja") in HARD_SECTIONS)
            if f.get("sekcja") in STORY_SECTIONS:
                stories_web += 1
                stories_foreign += is_foreign
    manual = data.get("ciekawostki")
    stories_manual = len(manual) if isinstance(manual, list) else (1 if schema.filled(manual) else 0)
    kult = data.get("ciekawostki_kulturowe")
    stories_manual += len(kult) if isinstance(kult, list) else (1 if schema.filled(kult) else 0)
    foreign_langs = sorted(langs - COMMON_LANGS - {"?"})
    photos = bool(data.get("url") or (data.get("zdjecia_wiki") or {}).get("zdjecia") or data.get("zdjecia"))
    merged = bool(data.get("scalone"))
    n_sources = len(zr) or len({(f.get("zrodlo") or {}).get("url") for f in wiedza.get("fakty") or []} - {None})

    scores = {
        "kompletnosc": cov["procent"],
        "trudna_wiedza": min(100, hard * 5 + len(foreign_langs) * 10 + (20 if asia else 0)),
        "opowiesci": min(100, stories_manual * 6 + stories_web * 6 + stories_foreign * 6),
        "przepisy": min(100, noncul * 15 + (10 if recipes else 0)),
        "zrodla": min(100, n_sources * 10 + (20 if photos else 0) + (10 if merged else 0)),
    }
    score = round(sum(WEIGHTS[k] * v for k, v in scores.items()), 1)
    flags = []
    if not points:
        flags.append("bez_wiedzy_z_sieci")
    if stories_manual + stories_web < 4 or not stories_foreign:
        flags.append("malo_opowiesci")
    if not asia:
        flags.append("brak_azji")
    if not secs.get("medycyna_wschodu") and status.get("profil_energetyczny.opis") in ("brak", "czeka"):
        flags.append("brak_wschodu")
    if not secs.get("barwienie") and status.get("zastosowanie.rzemieslnicze") in ("brak", "czeka"):
        flags.append("brak_barwienia")
    if not secs.get("kosmetyka") and status.get("zastosowanie.kosmetyczne") in ("brak", "czeka"):
        flags.append("brak_kosmetyki")
    if status.get("ostrzezenia") in ("brak", "czeka"):
        flags.append("brak_bezpieczenstwa")
    if wiedza.get("sekcje") and not merged:
        flags.append("do_scalenia")
    if not photos:
        flags.append("bez_zdjec")
    if cov["procent"] < 60:
        flags.append("male_pokrycie")
    if not noncul:
        flags.append("bez_przepisow")
    return {"id": pid, "nazwa": data.get("nazwa_pl") or data.get("gatunek") or pid,
            "lacina": data.get("nazwa_lat") or data.get("nazwa_lacinska"),
            "wynik": score, "oceny": scores, "pokrycie": cov["procent"], "puste": cov["puste"],
            "punkty": points, "trudne": hard, "zagraniczne": foreign, "azja": asia, "strony": web_pages,
            "jezyki": foreign_langs, "jezyki_punkty": dict(lang_points), "sekcje": dict(secs),
            "opowiesci": {"reczne": stories_manual, "z_sieci": stories_web, "zagraniczne": stories_foreign},
            "przepisy": recipes, "przepisy_niekulinarne": noncul, "zrodla": n_sources, "zdjecia": photos,
            "scalone": merged, "braki": flags, "_status": status}


# ------------------------------------------------------------------ cala baza
def data_problems(plants: dict[str, dict]) -> list[str]:
    """Bledy w plikach roslin Kwiatownika, ktore psuja ocene i zbieranie wiedzy (np. plik jednej rosliny
    z trescia innej): id w pliku inne niz nazwa pliku, ta sama nazwa lacinska w kilku plikach."""
    out = []
    by_latin: dict[str, list[str]] = {}
    for pid, d in plants.items():
        fid = str(d.get("id") or "")
        if fid and fid != pid:
            out.append(f"{pid}.json: id w pliku to '{fid}' ({d.get('nazwa_pl') or '?'})")
        lat = norm_text(str(d.get("nazwa_lat") or d.get("nazwa_lacinska") or ""))
        if lat:
            by_latin.setdefault(lat, []).append(pid)
    for lat, ids in sorted(by_latin.items()):
        if len(ids) > 1:
            out.append(f"ta sama nazwa lacinska '{lat}' w plikach: " + ", ".join(f"{i}.json" for i in ids))
    return out


def _web_counts(pid: str) -> dict[str, int]:
    try:
        from app.worker.knowledge import web_section_counts
        return web_section_counts(pid)
    except Exception:   # noqa: BLE001 - audyt dziala tez bez bazy Siedziby
        return {}


def _site_views() -> dict[str, int]:
    try:
        from app.agents.backend_sync import site_views
        return site_views(30)
    except Exception:  # noqa: BLE001 - audyt dziala tez bez statystyk strony
        return {}


def siedziba_stats(days: int = 30) -> dict:
    """Tempo zdobywania wiedzy w Siedzibie (ostatnie `days` dni) i stan potoku."""
    since = store.now() - timedelta(days=days)
    out: dict = {}
    with Session(engine) as s:
        out["informacje"] = dict(s.exec(select(PlantFact.status, func.count()).group_by(PlantFact.status)).all())
        recent = s.exec(select(PlantFact.language, PlantFact.status).where(PlantFact.created_at >= since)).all()
        out["nowe_informacje"] = len(recent)
        out["nowe_zweryfikowane"] = sum(1 for _, st in recent if st in ("verified", "applied"))
        out["nowe_wg_jezyka"] = dict(Counter(lang2(lg) for lg, st in recent if st in ("verified", "applied"))
                                     .most_common(12))
        q = s.exec(select(PlantQuery.facts, PlantQuery.used_url).where(PlantQuery.created_at >= since)).all()
        out["zapytania"] = len(q)
        out["zapytania_udane"] = sum(1 for f, u in q if u)
        out["zapytania_z_informacjami"] = sum(1 for f, u in q if (f or 0) > 0)
        out["rejestr_nigdy_badane"] = s.exec(select(func.count()).select_from(Plant).where(
            col(Plant.last_research_at).is_(None))).one()
        out["przepisy_siedziby"] = dict(s.exec(select(Item.status, func.count()).where(
            Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None)).group_by(Item.status)).all())
        out["przepisy_siedziby"] = {(k.value if isinstance(k, ItemStatus) else str(k)): v
                                    for k, v in out["przepisy_siedziby"].items()}
        out["kraje_ostatnio"] = {c: str(t)[:10] for c, t in s.exec(
            select(SearchQuery.country, func.max(SearchQuery.created_at)).group_by(SearchQuery.country)).all()}
    return out


def compute(plants_dir: Path | None = None, przepisy_dir: Path | None = None, web_counts=_web_counts,
            with_db: bool = True) -> dict:
    plants = read_plants(plants_dir or settings.kwiatownik_plants_dir)
    recipes = read_recipes(przepisy_dir or settings.kwiatownik_przepisy_dir)
    index = PlantIndex(plants)
    per_recipes: Counter = Counter()
    per_noncul: Counter = Counter()
    cats: Counter = Counter()
    cats_noncul_abroad: Counter = Counter()
    abroad = 0
    for r in recipes:
        cat = recipe_category(r)
        cats[cat] += 1
        ab = from_abroad(r)
        abroad += ab
        if cat != "kulinarne" and ab:
            cats_noncul_abroad[cat] += 1
        for pid in index.match(r):
            per_recipes[pid] += 1
            if cat != "kulinarne":
                per_noncul[pid] += 1
    rows = [plant_metrics(pid, d, per_recipes[pid], per_noncul[pid], web_counts(pid)) for pid, d in plants.items()]
    views = _site_views() if with_db else {}
    for r in rows:
        r["odslony"] = views.get(r["id"], 0)          # odslony na stronie z 30 dni (backend Kwiatownika)
    problems = data_problems(plants)
    rows.sort(key=lambda x: x["wynik"])
    n = len(rows) or 1
    gaps = []
    for slot in schema.SLOTS:
        brak = sum(1 for r in rows if r["_status"].get(slot.path) == "brak")
        czeka = sum(1 for r in rows if r["_status"].get(slot.path) == "czeka")
        gaps.append({"path": slot.path, "rozdzial": slot.chapter, "tytul": slot.title, "brak": brak, "czeka": czeka,
                     "procent": round(100 * brak / n)})
    gaps.sort(key=lambda g: -g["brak"])
    for r in rows:
        r.pop("_status", None)
    langs: Counter = Counter()
    secs: Counter = Counter()
    for r in rows:
        langs.update(r.get("jezyki_punkty") or {})
        secs.update(r["sekcje"])
    flags = Counter(f for r in rows for f in r["braki"])
    points = sum(r["punkty"] for r in rows)
    kryteria = {k: round(sum(r["oceny"][k] for r in rows) / n, 1) for k in CRITERIA}
    kpi = {
        "rosliny": len(rows), "wynik": round(sum(r["wynik"] for r in rows) / n, 1),
        "pokrycie": round(sum(r["pokrycie"] for r in rows) / n, 1),
        "z_wiedza_z_sieci": sum(1 for r in rows if r["punkty"]),
        "z_opowiesciami": sum(1 for r in rows if "malo_opowiesci" not in r["braki"]),
        "z_azja": sum(1 for r in rows if r["azja"]),
        "punkty": points, "trudne": sum(r["trudne"] for r in rows),
        "udzial_trudnych": round(100 * sum(r["trudne"] for r in rows) / points) if points else 0,
        "opowiesci": sum(r["opowiesci"]["reczne"] + r["opowiesci"]["z_sieci"] for r in rows),
        "opowiesci_zagraniczne": sum(r["opowiesci"]["zagraniczne"] for r in rows),
        "przepisy": len(recipes), "przepisy_niekulinarne": sum(v for k, v in cats.items() if k != "kulinarne"),
        "przepisy_z_zagranicy": abroad,
        "rosliny_bez_przepisow": sum(1 for r in rows if not r["przepisy_niekulinarne"]),
        "odslony_30d": sum(r.get("odslony", 0) for r in rows),
    }
    return {"kpi": kpi, "kryteria": kryteria, "rosliny": rows, "luki": gaps, "problemy_danych": problems,
            "jezyki": dict(langs.most_common()), "sekcje": {SECTION_TITLES_PL.get(k, k): v for k, v in secs.most_common()},
            "braki": {FLAG_LABELS.get(k, k): v for k, v in flags.most_common()}, "braki_klucze": dict(flags),
            "przepisy_kategorie": dict(cats.most_common()),
            "przepisy_zagraniczne_kategorie": dict(cats_noncul_abroad.most_common()),
            "siedziba": siedziba_stats() if with_db else {}}


# ------------------------------------------------------------------ zalecenia (reguly)
def _chunks(ids: list[str], size: int, limit: int) -> list[list[str]]:
    size = max(1, size)
    return [ids[i:i + size] for i in range(0, len(ids), size)][:limit]


def recommend(m: dict, per_task: int | None = None, max_per_kind: int = 2) -> list[dict]:
    per_task = per_task or settings.agents_plants_per_task
    rows = m["rosliny"]                                   # od najslabszych
    names = {r["id"]: r["nazwa"] for r in rows}
    recs: list[dict] = []

    def add(kind: str, ids: list[str], prio: int, why: str, size: int | None = None, limit: int | None = None):
        for chunk in _chunks(ids, size or per_task, limit or max_per_kind):
            p = {"rosliny": chunk, "nazwy": [names.get(i, i) for i in chunk]}
            recs.append({"typ": kind, "tytul": describe(kind, p), **p, "sloty": list(KINDS[kind].slots),
                         "priorytet": prio, "uzasadnienie": why, "zrodlo": "reguly"})

    def with_flag(flag: str, key=None) -> list[str]:
        """Rosliny z brakiem - najpierw te, ktore czytelnicy najczesciej ogladaja (odslony z backendu strony)."""
        sel = [r for r in rows if flag in r["braki"]]
        base = key or (lambda r: r["wynik"])
        return [r["id"] for r in sorted(sel, key=lambda r: (-r.get("odslony", 0), base(r)))]

    k = m["kpi"]
    add("nowa_wiedza", with_flag("bez_wiedzy_z_sieci"), 1,
        "Rosliny bez zadnej wiedzy z sieci - pierwszy przebieg: Wikipedia w wielu jezykach (takze zh/ja) i strony z innych krajow.")
    add("opowiesci", [i for i in with_flag("malo_opowiesci",
                                           key=lambda r: r["opowiesci"]["zagraniczne"] * 2 + r["opowiesci"]["z_sieci"])
                      if i not in with_flag("bez_wiedzy_z_sieci")], 1,
        f"Malo opowiesci i wierzen (w calej bazie {k['opowiesci_zagraniczne']} z zagranicznych zrodel) - legendy, obrzedy, "
        "historia i nazwy ludowe w zrodlach ukrainskich, rosyjskich, japonskich i chinskich.")
    add("wschod", with_flag("brak_wschodu", key=lambda r: r["wynik"]), 2,
        "Brak opisu w medycynie chinskiej / kampo (natura, smak, meridiany) - wiedza prawie niedostepna po polsku.")
    add("barwienie", with_flag("brak_barwienia", key=lambda r: r["wynik"]), 2,
        "Brak informacji o barwieniu i rzemiosle - japonskie 草木染め i niemieckie Pflanzenfarben to dobre zrodla.")
    add("bezpieczenstwo", with_flag("brak_bezpieczenstwa", key=lambda r: r["wynik"]), 2,
        "Brak przeciwwskazan i toksycznosci - wazne przy przepisach leczniczych.", limit=1)
    add("kosmetyka", with_flag("brak_kosmetyki", key=lambda r: r["wynik"]), 3,
        "Brak zastosowan kosmetycznych (pielegnacja skory i wlosow).", limit=1)
    add("luki", [r["id"] for r in rows if "male_pokrycie" in r["braki"] and "bez_wiedzy_z_sieci" not in r["braki"]], 3,
        "Pokrycie schematu ponizej 60% - Siedziba sama dobierze najwazniejsze puste podrozdzialy i jezyki.")
    add("scal", with_flag("do_scalenia"), 2,
        "Wiedza z sieci jest w pliku, ale nie trafila do rozdzialow strony (czytelnik jej nie widzi w Medycynie, Rzemiosle...).",
        size=per_task * 2, limit=1)
    add("zdjecia", with_flag("bez_zdjec"), 4, "Rosliny bez galerii - zdjecia z Wikimedia Commons (autor + licencja), bez LLM.",
        size=10, limit=1)

    # przepisy: za malo przepisow niekulinarnych z zagranicy -> nowe strony w krajach dawno nieprzeszukanych
    cats = m.get("przepisy_zagraniczne_kategorie", {})
    weak = [c for c in ("barwienie", "kosmetyka i rzemioslo", "nalewki i trunki") if cats.get(c, 0) < 15]
    if weak:
        from app.discovery.countries import COUNTRIES
        last = (m.get("siedziba") or {}).get("kraje_ostatnio", {})
        order = sorted((c for c in COUNTRIES if c != "pl"), key=lambda c: (last.get(c) or "", list(COUNTRIES).index(c)))
        for code in order[:2]:
            p = {"kraj": code, "kraj_nazwa": COUNTRIES[code].name}
            recs.append({"typ": "nowe_zrodla", "tytul": describe("nowe_zrodla", p), **p, "rosliny": [], "sloty": [],
                         "priorytet": 3, "zrodlo": "reguly",
                         "uzasadnienie": "Malo przepisow niekulinarnych z zagranicy w kategoriach: " + ", ".join(weak)
                                         + f". Kraj {'jeszcze nieprzeszukany' if not last.get(code) else 'przeszukany ' + last[code]}."})
    for n, r in enumerate(recs, 1):
        r["nr"] = n
    return recs


# ------------------------------------------------------------------ ekspert LLM
SYSTEM = ("Jestes ekspertem etnobotaniki i ziololecznictwa (medycyna ludowa Europy, medycyna chinska i kampo, "
          "barwierstwo roslinne) oraz redaktorem bazy wiedzy Kwiatownik. Oceniasz stan bazy rzetelnie i krytycznie, "
          "WYLACZNIE na podstawie podanych danych. Odpowiadasz po polsku, tylko JSON.")

PROMPT = """CEL KWIATOWNIKA: wiedza o roslinach TRUDNA DO ZDOBYCIA w Polsce (zrodla zagraniczne - Chiny, Japonia, Ukraina, \
Rosja, Niemcy, Francja...; medycyna Wschodu; barwienie; dawne zastosowania) oraz OPOWIESCI zwiazane z roslinami \
(legendy, wierzenia, obrzedy, historia, etymologia, nazwy ludowe). Do tego przepisy niekulinarne (ziololecznictwo, \
nalewki, barwienie, kosmetyka). Kazda informacja ma zrodlo (URL).

OCENA KODEM (0-100, srednia roslin): {kryteria}
KPI: {kpi}
JEZYKI ZRODEL (bez pl/en, liczba informacji z sieci): {jezyki}
SEKCJE WIEDZY Z SIECI: {sekcje}
NAJCZESTSZE BRAKI (liczba roslin): {braki}
PUSTE PODROZDZIALY (liczba roslin bez tresci): {luki}
PRZEPISY wg kategorii: {przepisy}
BLEDY W PLIKACH ROSLIN: {problemy}
SIEDZIBA - ostatnie 30 dni: {siedziba}

NAJSLABSZE ROSLINY (id | nazwa | wynik | oceny | braki):
{slabe}
NAJLEPSZE ROSLINY: {mocne}
NAJCZESCIEJ CZYTANE NA STRONIE (odslony 30 dni): {popularne}

ZALECENIA Z REGUL (nr | typ | rosliny | uzasadnienie):
{zalecenia}

SLOWNIK TYPOW ZALECEN (tylko te): {slownik}

Zadanie:
1. Ocen baze w 5 kryteriach 1-10: kompletnosc, trudna_wiedza, opowiesci, przepisy, zrodla.
2. Podsumuj stan w 2-4 zdaniach; wypisz mocne i slabe strony (konkretnie, z liczbami z danych).
3. Ustal priorytety zalecen z regul (1 = najpilniejsze, 5 = moze poczekac) - patrzac na CEL (wiedza trudno dostepna \
i opowiesci najwazniejsze).
4. Zaproponuj do 5 NOWYCH zalecen: typ ze slownika + rosliny (tylko id z listy powyzej) + uzasadnienie.
5. Podaj do 5 pomyslow, gdzie szukac trudnej wiedzy i opowiesci (kraje, tradycje, rodzaje zrodel) dla tych roslin.

Zwroc TYLKO JSON:
{{"oceny": {{"kompletnosc": 6, "trudna_wiedza": 4, "opowiesci": 5, "przepisy": 6, "zrodla": 7}},
"podsumowanie": "...", "mocne": ["..."], "slabe": ["..."],
"priorytety": [{{"nr": 1, "priorytet": 1, "uzasadnienie": "..."}}],
"nowe_zalecenia": [{{"typ": "opowiesci", "rosliny": ["id"], "priorytet": 2, "uzasadnienie": "..."}}],
"pomysly": ["..."]}}"""


def _compact(d: dict, limit: int = 12) -> str:
    return json.dumps(dict(list(d.items())[:limit]), ensure_ascii=False)


def build_prompt(m: dict, recs: list[dict]) -> str:
    rows = m["rosliny"]
    slabe = "\n".join(f"{r['id']} | {r['nazwa']} | {r['wynik']} | "
                      + ", ".join(f"{k[:6]}={v}" for k, v in r["oceny"].items()) + " | "
                      + ", ".join(FLAG_LABELS.get(f, f) for f in r["braki"]) for r in rows[:15])
    mocne = ", ".join(f"{r['id']} ({r['wynik']})" for r in rows[-5:][::-1])
    zal = "\n".join(f"{r['nr']} | {r['typ']} | {', '.join(r.get('rosliny') or [r.get('kraj', '')])} | {r['uzasadnienie']}"
                    for r in recs)
    luki = {g["tytul"]: g["brak"] for g in m["luki"][:10]}
    sied = {k: v for k, v in (m.get("siedziba") or {}).items() if k != "kraje_ostatnio"}
    return PROMPT.format(kryteria=json.dumps(m["kryteria"], ensure_ascii=False), kpi=json.dumps(m["kpi"], ensure_ascii=False),
                         jezyki=_compact(m["jezyki"], 15), sekcje=_compact(m["sekcje"], 15),
                         braki=_compact(m["braki"]), luki=json.dumps(luki, ensure_ascii=False),
                         przepisy=_compact(m["przepisy_kategorie"]),
                         problemy="; ".join(m.get("problemy_danych") or []) or "brak", siedziba=json.dumps(sied, ensure_ascii=False),
                         slabe=slabe, mocne=mocne, zalecenia=zal or "(brak)",
                         popularne=", ".join(f"{r['id']} ({r['odslony']})" for r in sorted(
                             rows, key=lambda r: -r.get("odslony", 0))[:10] if r.get("odslony")) or "brak danych",
                         slownik="; ".join(f"{k} = {KINDS[k].hint or KINDS[k].label}" for k in EXPERT_KINDS))


def parse_json(text: str) -> dict:
    t = (text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(t)
    except ValueError:
        mt = re.search(r"\{.*\}", t, re.S)
        if not mt:
            raise ValueError("odpowiedz bez JSON")
        data = json.loads(mt.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON nie jest obiektem")
    return data


def check_expert(data: dict, m: dict, recs: list[dict]) -> dict:
    """Kontrola kodem odpowiedzi eksperta: oceny 1-10, priorytety tylko dla istniejacych zalecen,
    nowe zalecenia tylko z typem ze slownika i roslinami z bazy. Odrzucone - zapisane z powodem."""
    known = {r["id"]: r["nazwa"] for r in m["rosliny"]}
    rejected = []
    scores = {}
    for k in CRITERIA:
        try:
            scores[k] = max(1, min(10, round(float((data.get("oceny") or {}).get(k)))))
        except (TypeError, ValueError):
            rejected.append(f"brak oceny: {k}")
    nrs = {r["nr"] for r in recs}
    prios = []
    for p in data.get("priorytety") or []:
        try:
            nr, pr = int(p.get("nr")), max(1, min(5, int(p.get("priorytet"))))
        except (TypeError, ValueError, AttributeError):
            continue
        if nr in nrs:
            prios.append({"nr": nr, "priorytet": pr, "uzasadnienie": str(p.get("uzasadnienie") or "")[:300]})
        else:
            rejected.append(f"priorytet dla nieistniejacego zalecenia {nr}")
    new = []
    for z in (data.get("nowe_zalecenia") or [])[:8]:
        if not isinstance(z, dict):
            continue
        typ = str(z.get("typ") or "")
        ids = [str(i) for i in z.get("rosliny") or [] if str(i) in known]
        bad = [str(i) for i in z.get("rosliny") or [] if str(i) not in known]
        if typ not in EXPERT_KINDS:
            rejected.append(f"nowe zalecenie z nieznanym typem: {typ}")
            continue
        if bad:
            rejected.append(f"pominieto nieznane rosliny: {', '.join(bad[:5])}")
        if not ids:
            continue
        try:
            pr = max(1, min(5, int(z.get("priorytet") or 3)))
        except (TypeError, ValueError):
            pr = 3
        new.append({"typ": typ, "rosliny": ids[:settings.agents_plants_per_task * 2], "priorytet": pr,
                    "uzasadnienie": str(z.get("uzasadnienie") or "")[:400]})
        if len(new) >= 5:
            break

    def strings(key: str, n: int = 6) -> list[str]:
        return [str(x)[:300] for x in (data.get(key) or []) if isinstance(x, (str, int, float))][:n]

    return {"oceny": scores, "podsumowanie": str(data.get("podsumowanie") or "")[:1200],
            "mocne": strings("mocne"), "slabe": strings("slabe"), "priorytety": prios, "nowe_zalecenia": new,
            "pomysly": strings("pomysly"), "odrzucone": rejected}


async def expert_review(llm, m: dict, recs: list[dict], logf=None) -> dict | None:
    if llm is None:
        return None
    logf = logf or (lambda x: None)
    for attempt in (1, 2):
        try:
            res = await llm.complete(build_prompt(m, recs), system=SYSTEM, json_mode=True)
            out = check_expert(parse_json(res.text), m, recs)
            if not out["oceny"]:
                raise ValueError("brak ocen w odpowiedzi")
            out["model"] = f"{res.provider}/{res.model}"
            return out
        except Exception as exc:  # noqa: BLE001 - audyt bez eksperta jest nadal uzyteczny
            logf(f"Ekspert LLM - proba {attempt} nieudana: {type(exc).__name__}: {str(exc)[:160]}")
    return None


def apply_expert(recs: list[dict], expert: dict | None, m: dict) -> list[dict]:
    if not expert:
        return recs
    by_nr = {r["nr"]: r for r in recs}
    for p in expert["priorytety"]:
        r = by_nr[p["nr"]]
        r["priorytet_regul"] = r["priorytet"]
        r["priorytet"] = p["priorytet"]
        if p["uzasadnienie"]:
            r["ekspert"] = p["uzasadnienie"]
    names = {r["id"]: r["nazwa"] for r in m["rosliny"]}
    nr = max(by_nr, default=0)
    for z in expert["nowe_zalecenia"]:
        nr += 1
        p = {"rosliny": z["rosliny"], "nazwy": [names.get(i, i) for i in z["rosliny"]]}
        recs.append({"nr": nr, "typ": z["typ"], "tytul": describe(z["typ"], p), **p, "sloty": list(KINDS[z["typ"]].slots),
                     "priorytet": z["priorytet"], "uzasadnienie": z["uzasadnienie"], "zrodlo": "ekspert"})
    return recs


# ------------------------------------------------------------------ porownanie z poprzednim audytem
def compare(prev: dict, m: dict, score: float, prev_score: float | None) -> dict:
    old = {r["id"]: r for r in (prev.get("metryki") or {}).get("rosliny") or []}
    if not old:
        return {}
    deltas = []
    for r in m["rosliny"]:
        o = old.get(r["id"])
        if o is None:
            continue
        d = round(r["wynik"] - o["wynik"], 1)
        if d:
            deltas.append({"id": r["id"], "nazwa": r["nazwa"], "przed": o["wynik"], "po": r["wynik"], "zmiana": d,
                           "punkty": r["punkty"] - o.get("punkty", 0),
                           "opowiesci": (r["opowiesci"]["z_sieci"] - (o.get("opowiesci") or {}).get("z_sieci", 0))})
    deltas.sort(key=lambda x: -x["zmiana"])
    okpi = (prev.get("metryki") or {}).get("kpi") or {}
    kpi_delta = {k: round(v - okpi[k], 1) for k, v in m["kpi"].items() if isinstance(v, (int, float)) and k in okpi
                 and v != okpi[k]}
    return {"ocena": round(score - prev_score, 1) if prev_score is not None else None,
            "lepsze": deltas[:10], "gorsze": [x for x in deltas if x["zmiana"] < 0][-10:][::-1],
            "nowe_rosliny": [r["id"] for r in m["rosliny"] if r["id"] not in old], "kpi": kpi_delta}


# ------------------------------------------------------------------ przebieg audytu
def save_tasks(run_id: int, recs: list[dict]) -> int:
    """Zalecenia -> AgentTask 'proposed'. Niezlecone zalecenia poprzednich audytow -> 'expired' (zastapione)."""
    with Session(engine) as s:
        for t in s.exec(select(AgentTask).where(AgentTask.agent == "audytor", AgentTask.status == "proposed")).all():
            t.status, t.note = "expired", f"zastapione przez audyt #{run_id}"
            s.add(t)
        for r in recs:
            p = {k: r[k] for k in ("rosliny", "nazwy", "sloty", "kraj", "kraj_nazwa") if r.get(k)}
            p["nr"] = r["nr"]
            reason = r["uzasadnienie"] + (f" | Ekspert: {r['ekspert']}" if r.get("ekspert") else "")
            s.add(AgentTask(run_id=run_id, agent="audytor", kind=r["typ"], title=r["tytul"], reason=reason[:1000],
                            priority=int(r["priorytet"]), params_json=json.dumps(p, ensure_ascii=False)))
        s.commit()
    return len(recs)


async def run_audit(llm=None, trigger: str = "reczny", job_id: int | None = None, logf=None,
                    plants_dir: Path | None = None, przepisy_dir: Path | None = None) -> int:
    """Pelny audyt. Zwraca id przebiegu (AgentRun)."""
    run_id = store.start_run("audytor", trigger, job_id)

    def say(msg: str) -> None:
        store.run_log(run_id, msg)
        if logf:
            logf(msg)

    try:
        say("Skanuje baze wiedzy Kwiatownika (pliki roslin, przepisy) i stan Siedziby...")
        m = compute(plants_dir, przepisy_dir)
        k = m["kpi"]
        say(f"Rosliny: {k['rosliny']}, srednia ocena kodu {k['wynik']}/100, pokrycie {k['pokrycie']}%, "
            f"informacji z sieci {k['punkty']} (trudno dostepne {k['udzial_trudnych']}%), "
            f"opowiesci {k['opowiesci']} (z zagranicy {k['opowiesci_zagraniczne']}), przepisy {k['przepisy']} "
            f"(niekulinarne {k['przepisy_niekulinarne']})")
        recs = recommend(m)
        say(f"Zalecenia z regul: {len(recs)}")
        expert = None
        if settings.agents_expert_review and llm is not None:
            say("Ocena eksperta LLM...")
            expert = await expert_review(llm, m, recs, say)
            if expert:
                say(f"Ekspert ({expert['model']}): " + ", ".join(f"{k}={v}" for k, v in expert["oceny"].items())
                    + (f"; odrzucone przez kontrole: {len(expert['odrzucone'])}" if expert["odrzucone"] else ""))
        elif llm is None:
            say("Brak LLM - raport tylko z metryk")
        recs = apply_expert(recs, expert, m)
        recs.sort(key=lambda r: (r["priorytet"], r["nr"]))
        code_score = k["wynik"]
        expert_score = round(10 * sum(expert["oceny"].values()) / len(expert["oceny"]), 1) if expert else None
        score = round((1 - EXPERT_WEIGHT) * code_score + EXPERT_WEIGHT * expert_score, 1) if expert_score is not None \
            else code_score
        prev_run = store.last_run("audytor")
        prev_run = prev_run if prev_run and prev_run.id != run_id else None
        changes = compare(store.report(prev_run), m, score, prev_run.score if prev_run else None)
        save_tasks(run_id, recs)
        rep = {"metryki": m, "zalecenia": recs, "ekspert": expert, "zmiany": changes, "wagi": WEIGHTS,
               "kryteria_nazwy": CRITERIA, "poprzedni": prev_run.id if prev_run else None}
        delta = f" ({changes['ocena']:+.1f})" if changes.get("ocena") is not None else ""
        summary = (f"Ocena {score:.0f}/100{delta}; {k['rosliny']} roslin, trudna wiedza {m['kryteria']['trudna_wiedza']:.0f}, "
                   f"opowiesci {m['kryteria']['opowiesci']:.0f}; zalecen: {len(recs)}")
        store.finish_run(run_id, "done", rep, score=score, code_score=code_score, expert_score=expert_score,
                         model=expert["model"] if expert else None, summary=summary)
        say(summary)
    except Exception as exc:
        say(f"BLAD: {type(exc).__name__}: {exc}")
        store.finish_run(run_id, "failed", summary=f"Audyt nieudany: {str(exc)[:200]}")
        raise
    return run_id


def plant_history(plant_id: str, limit: int = 30) -> list[dict]:
    """Ocena rosliny w kolejnych audytach (od najstarszego)."""
    out = []
    with Session(engine) as s:
        runs = s.exec(select(AgentRun).where(AgentRun.agent == "audytor", AgentRun.status == "done")
                      .order_by(col(AgentRun.id).desc()).limit(limit)).all()
    for run in reversed(runs):
        rep = store.report(run)
        row = next((r for r in (rep.get("metryki") or {}).get("rosliny") or [] if r["id"] == plant_id), None)
        if row:
            out.append({"run": run.id, "at": run.created_at, "wynik": row["wynik"], "oceny": row["oceny"],
                        "punkty": row["punkty"], "braki": row["braki"]})
    return out
