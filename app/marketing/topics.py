"""Tematy postow: Siedziba (pierwsze posty), pora roku, ciekawostka z zagranicznej strony, nietypowy przepis.

Kazdy builder zwraca Material (dane dla modelu + link/zrodlo) albo None, gdy nie ma z czego pisac."""
import json
import random
from urllib.parse import urlparse

from sqlmodel import Session, col, select

from app.marketing import picker
from app.marketing.season import Season
from app.marketing.writer import Material, plant_block
from app.models import FbIdea, FbPost, Item, ItemKind, ItemStatus, PlantFact

SITE_URL = "https://kwiatownik.onrender.com"
KINDS = {"siedziba": "Siedziba Kwiatownika", "pora_roku": "Ziele na te pore roku",
         "ciekawostka": "Ciekawostka z zagranicy", "przepis": "Nietypowy przepis"}
AUTO_ROTATION = ("ciekawostka", "przepis", "pora_roku")
INTRO_POSTS = 2   # tyle pierwszych postow opowiada o Siedzibie

LANG_PL = {"de": "niemieckiej", "en": "angielskiej", "uk": "ukraińskiej", "ru": "rosyjskiej", "zh": "chińskiej",
           "ja": "japońskiej", "fr": "francuskiej", "it": "włoskiej", "es": "hiszpańskiej", "cs": "czeskiej",
           "sk": "słowackiej", "lt": "litewskiej", "hu": "węgierskiej", "ro": "rumuńskiej", "be": "białoruskiej",
           "ko": "koreańskiej", "hi": "indyjskiej (hindi)", "tr": "tureckiej", "pt": "portugalskiej",
           "nl": "holenderskiej", "sv": "szwedzkiej", "fi": "fińskiej", "lv": "łotewskiej", "et": "estońskiej"}
FACT_SECTIONS = ("ciekawostki", "kultura", "historia", "barwienie", "nazwy_ludowe", "medycyna_wschodu",
                 "kosmetyka", "zastosowanie_lecznicze")
SECTION_PL = {"ciekawostki": "ciekawostka", "kultura": "kultura i obyczaje", "historia": "historia",
              "barwienie": "barwienie", "nazwy_ludowe": "nazwy ludowe", "medycyna_wschodu": "medycyna Wschodu",
              "kosmetyka": "kosmetyka", "zastosowanie_lecznicze": "dawne lecznictwo"}

# Co robi Siedziba - TYLKO prawdziwe funkcje (model nie moze nic dopisac)
SIEDZIBA_FEATURES = [
    "Siedziba Kwiatownika to program, który działa na komputerze twórcy Kwiatownika i sam szuka po świecie "
    "stron o ziołach i roślinach – ukraińskich, niemieckich, angielskich, a w wiedzy o roślinach także "
    "chińskich i japońskich. Wszystko, co znajdzie, trafia do Kwiatownika (kwiatownik.onrender.com) – "
    "cyfrowego zielnika.",
    "Do każdej strony uczy się sama dopasować: rozpoznaje, gdzie na stronie jest przepis, a gdzie reklama, "
    "i potem pobiera przepisy z całej strony.",
    "Zbiera przepisy, w których roślina jest najważniejsza: nalewki, napary, syropy, maści, barwienie "
    "tkanin i włosów, naturalną kosmetykę. Zwykłe przepisy kuchenne odsiewa.",
    "Tłumaczy przepisy na polski (własny tłumacz i modele sztucznej inteligencji) i zamienia obce miary "
    "(funty, uncje, kubki) na gramy i mililitry.",
    "Pilnuje, żeby ten sam przepis nie trafił dwa razy, i opisuje przepisy według dawnych systemów, "
    "np. pięciu przemian.",
    "Zbiera wiedzę o każdej roślinie z Wikipedii w wielu językach i ze stron z innych krajów. Każda informacja "
    "ma źródło (link) i cytat; najpierw sprawdza ją program (czy cytat naprawdę jest na stronie), a potem "
    "drugi, inny model sztucznej inteligencji.",
    "Dobiera zdjęcia roślin z Wikimedia Commons – tylko na wolnej licencji, z podpisanym autorem.",
    "Sterowana jest z panelu w przeglądarce, a pracuje na zmianę kilkoma modelami sztucznej inteligencji – "
    "gdy jeden się zmęczy, robotę bierze następny.",
]


def _used_refs(s: Session, prefix: str) -> set[str]:
    out = set()
    for ref in s.exec(select(FbPost.ref).where(col(FbPost.ref).startswith(prefix),
                                               FbPost.status != "rejected")).all():
        out |= {f"{prefix}{x}" for x in (ref or "")[len(prefix):].split(",") if x}
    return out


def count_kind(s: Session, kind: str) -> int:
    return len(s.exec(select(FbPost.id).where(FbPost.kind == kind, FbPost.status != "rejected")).all())


def _ideas_stats(s: Session) -> str:
    """Prawdziwe liczby z puli ciekawostek (dziadek moze sie pochwalic) - tylko gdy pula juz jest."""
    from app.marketing import ideas
    c = ideas.counts(s)
    if not c["total"]:
        return ""
    langs = [LANG_PL.get(lg, lg).replace("iej", "ich").replace("kiej", "kich") for lg in c["foreign_langs"]
             if lg in LANG_PL]
    return (f"\n- Siedziba zebrała już {c['total']} kandydatów na ciekawostki"
            + (f", między innymi ze stron {', '.join(langs[:5])}" if langs else "") + ".")


def siedziba(s: Session, season: Season) -> Material:
    n = count_kind(s, "siedziba")
    if n == 0:
        task = ("To PIERWSZY post na fanpage'u. Przedstaw się: kim jesteś i że pracujesz w Siedzibie Kwiatownika – "
                "jesteś jej gawędziarzem: kiedy Siedziba wygrzebie coś ciekawego, to ty to opowiadasz ludziom. "
                "W 2–3 zdaniach powiedz po swojemu, co Siedziba robi (wybierz najciekawsze rzeczy z DANYCH). "
                "Zapowiedz, że odtąd będziesz tu przedstawiał najnowsze ciekawostki, które Siedziba znajdzie na "
                "zagranicznych stronach, i dziwne przepisy z różnych stron świata. Zakończ pytaniem do czytelników.")
        data = "\n".join(f"- {f}" for f in SIEDZIBA_FEATURES) + _ideas_stats(s)
        label = "powitanie i ogloszenie Siedziby"
    else:
        feats = SIEDZIBA_FEATURES[1:]
        a = feats[(2 * (n - 1)) % len(feats)]
        b = feats[(2 * (n - 1) + 1) % len(feats)]
        task = ("Kolejny post o Siedzibie Kwiatownika. Opowiedz bliżej, jak dziadek to rozumie, o tych dwóch "
                "rzeczach, które Siedziba robi (porównaj do dawnych, wiejskich spraw). Na koniec zapowiedz "
                "ciekawostki i dziwne przepisy z zagranicy.")
        data = f"- {SIEDZIBA_FEATURES[0]}\n- {a}\n- {b}" + _ideas_stats(s)
        label = f"o Siedzibie (cz. {n + 1})"
    return Material(kind="siedziba", label=label, task=task, data=data, season=season,
                    main_name="Siedziba", link_text="Zajrzyjcie do Kwiatownika 👉", link_url=SITE_URL + "/",
                    ref=f"siedziba:{n + 1}")


def pora_roku(s: Session, season: Season, plants: dict[str, dict], plant_id: str | None,
              recent: set[str]) -> Material | None:
    if not plants:
        return None
    chosen = picker.choose(plants, season, SITE_URL, plant_id=plant_id, recent=recent)
    if not chosen:
        return None
    main = chosen[0]
    task = (f"Opowiedz o roślinie {main.nazwa_pl} w odniesieniu do obecnej pory roku: co się z nią teraz dzieje "
            "w lesie i na łące, co teraz zbierać albo robić, jak to downiej bywało. Pozostałe rośliny z danych "
            "możesz wspomnieć mimochodem albo wcale.")
    return Material(kind="pora_roku", label=main.nazwa_pl, task=task,
                    data="\n".join(plant_block(p) for p in chosen), season=season, main_name=main.nazwa_pl,
                    main_latin=main.nazwa_lat, link_url=main.url, plants=[p.brief() for p in chosen],
                    ref=f"plant:{main.id}")


def _idea_material(x: FbIdea, sc: float, n_pool: int, season: Season, plants: dict[str, dict]) -> Material:
    from app.marketing.ideas import is_foreign
    data = plants.get(x.plant_id) if x.plant_id else None
    lang = (x.language or "").split("-")[0].lower()
    foreign = is_foreign(x.language, x.original)
    if foreign and lang in LANG_PL:
        where = f"ze strony {LANG_PL[lang]}"
    elif foreign:
        where = "z zagranicznej strony"
    else:
        where = "z polskiej strony" if lang.startswith("pl") else "ze strony"
    domain = urlparse(x.source_url or "").netloc
    lines = [f"CIEKAWOSTKA (Siedziba wybrała ją z puli {n_pool} kandydatów, ocena {sc:.0f}/100): {x.text}"]
    if x.teaser:
        lines.append(f"Zajawka redaktora: {x.teaser}")
    if x.original:
        lines.append(f"Dosłowny fragment źródła{' (' + x.language + ')' if x.language else ''}: {x.original}")
    lines.append(f"Skąd: {where} – {x.source_name or domain} ({domain})")
    if data:
        lines.insert(0, f"Roślina: {data['nazwa_pl']}" + (f" ({data.get('nazwa_lat')})" if data.get("nazwa_lat") else ""))
        warning = data.get("ostrzezenia")
        if isinstance(warning, str) and warning.strip():
            lines.append(f"OSTRZEŻENIE: {warning[:300]}")
    elif x.plant_name:
        lines.insert(0, f"Roślina: {x.plant_name}")
    name = data["nazwa_pl"] if data else (x.plant_name or None)
    task = (f"Opowiedz tę ciekawostkę, którą Siedziba wygrzebała {where}. Dziadek się dziwi albo porównuje "
            "z tym, jak u nas na wsi to robili. Powiedz, z jakiego kraju albo języka to pochodzi. "
            "Trzymaj się dokładnie treści ciekawostki – nie dopowiadaj faktów.")
    url = f"{SITE_URL}/plant/{x.plant_id}/" if data else SITE_URL + "/"
    return Material(kind="ciekawostka", label=f"{name or 'ciekawostka'} ({x.language or '?'}, {sc:.0f})", task=task,
                    data="\n".join(lines), season=season, main_name=name,
                    main_latin=data.get("nazwa_lat") if data else None, link_url=url,
                    link_text=None if data else "Zajrzyjcie do Kwiatownika 👉",
                    sources=[x.source_url] if x.source_url else [], ref=f"idea:{x.id}",
                    plants=[{"id": x.plant_id, "nazwa_pl": data["nazwa_pl"], "nazwa_lat": data.get("nazwa_lat"),
                             "url": url}] if data else [])


def ciekawostka(s: Session, season: Season, plants: dict[str, dict], plant_id: str | None = None,
                rnd: random.Random | None = None, idea_id: int | None = None) -> Material | None:
    """Najpierw pula ciekawostek zebranych i ocenionych przez Siedzibe (app/marketing/ideas.py, z premia za
    pore roku); gdy pula pusta - zweryfikowane informacje z zagranicznych stron (starsza droga)."""
    from app.marketing import ideas
    if idea_id:
        x = s.get(FbIdea, idea_id)
        if x and x.text:
            return _idea_material(x, x.score, len(ideas.pool(s, season.month)), season, plants)
    best = ideas.pool(s, season.month, plant_id, limit=500)
    if best:
        x, sc = best[0]
        return _idea_material(x, sc, len(best), season, plants)
    rnd = rnd or random.Random()
    used = _used_refs(s, "fact:")
    stmt = select(PlantFact).where(col(PlantFact.status).in_(["verified", "applied"]),
                                   col(PlantFact.language).is_not(None), PlantFact.language != "pl",
                                   col(PlantFact.section).in_(FACT_SECTIONS))
    if plant_id:
        stmt = stmt.where(PlantFact.plant_id == plant_id)
    facts = [f for f in s.exec(stmt).all() if f"fact:{f.id}" not in used and f.plant_id in plants]
    if not facts:
        return None
    by_src: dict[tuple, list[PlantFact]] = {}
    for f in facts:
        by_src.setdefault((f.plant_id, f.source_url), []).append(f)
    # najlepiej ta sama strona i roslina, kilka informacji; ciekawostki/kultura przed lecznictwem
    weight = {sec: len(FACT_SECTIONS) - i for i, sec in enumerate(FACT_SECTIONS)}
    groups = sorted(by_src.values(), key=lambda g: -(min(len(g), 3) * 3 + max(weight[f.section] for f in g)
                                                      + rnd.random() * 6))
    group = sorted(groups[0], key=lambda f: -weight[f.section])[:3]
    pid = group[0].plant_id
    data = plants[pid]
    lang = group[0].language or ""
    where = f"strony {LANG_PL.get(lang, lang)}" if lang in LANG_PL else f"zagranicznej strony ({lang})"
    src = group[0].source_name or urlparse(group[0].source_url).netloc
    lines = [f"Roślina: {data['nazwa_pl']}" + (f" ({data.get('nazwa_lat')})" if data.get("nazwa_lat") else ""),
             f"Skąd: z {where} – {src}"]
    lines += [f"- [{SECTION_PL.get(f.section, f.section)}] {f.text}" for f in group]
    warning = data.get("ostrzezenia")
    if isinstance(warning, str) and warning.strip():
        lines.append(f"OSTRZEŻENIE: {warning[:300]}")
    task = (f"Opowiedz ciekawostkę o roślinie {data['nazwa_pl']}, którą Siedziba wygrzebała ze {where}. "
            "Dziadek się dziwi albo porównuje z tym, jak u nas na wsi to robili. Powiedz, z jakiego kraju "
            "albo języka to pochodzi.")
    url = f"{SITE_URL}/plant/{pid}/"
    return Material(kind="ciekawostka", label=f"{data['nazwa_pl']} ({lang})", task=task, data="\n".join(lines),
                    season=season, main_name=data["nazwa_pl"], main_latin=data.get("nazwa_lat"), link_url=url,
                    sources=[group[0].source_url], ref="fact:" + ",".join(str(f.id) for f in group),
                    plants=[{"id": pid, "nazwa_pl": data["nazwa_pl"], "nazwa_lat": data.get("nazwa_lat"),
                             "url": url}])


def _recipe_score(item: Item, d: dict, rnd: random.Random) -> float:
    lang = (item.language or "").lower()
    score = 0.0
    if lang and not lang.startswith("pl"):
        score += 3                                   # z zagranicy
    if any("polsk" not in str(x).lower() for x in d.get("pochodzenie") or []):
        score += 1
    typ = f"{d.get('typ', '')} {d.get('metoda', '')}".lower()
    if not any(w in typ for w in ("napar", "herbat", "kulinarne")):
        score += 1.5                                 # cos innego niz zwykla herbatka
    if d.get("wymogi_szamanskie_i_czasowe") or d.get("wskazowki_tradycyjne"):
        score += 1
    return score + rnd.random() * 2


def przepis(s: Session, season: Season, rnd: random.Random | None = None) -> Material | None:
    """Nietypowy przepis zebrany z zagranicznej strony (bez archiwum Kwiatownika 1)."""
    rnd = rnd or random.Random()
    used = _used_refs(s, "item:")
    items = s.exec(select(Item).where(Item.kind == ItemKind.recipe, col(Item.status).in_(
        [ItemStatus.approved, ItemStatus.exported, ItemStatus.translated]),
        col(Item.duplicate_of).is_(None), col(Item.data_json).is_not(None))).all()
    best = None
    for it in items:
        if f"item:{it.id}" in used or not (it.url or "").startswith("http"):
            continue
        try:
            d = json.loads(it.data_json)
        except ValueError:
            continue
        if not d.get("tytul"):
            continue
        sc = _recipe_score(it, d, rnd)
        if best is None or sc > best[0]:
            best = (sc, it, d)
    if not best:
        return None
    _, it, d = best
    skl = [x.get("nazwa") if isinstance(x, dict) else str(x) for x in d.get("skladniki") or []]
    kroki = [str(x) for x in d.get("sposob_przygotowania") or []]
    lines = [f"Przepis: {d['tytul']}", f"Pochodzenie: {', '.join(map(str, d.get('pochodzenie') or [])) or '-'}"
             f" (strona: {urlparse(it.url).netloc})",
             f"Rodzaj: {d.get('typ') or '-'}, metoda: {d.get('metoda') or '-'}"]
    if d.get("opis"):
        lines.append(f"Opis: {str(d['opis'])[:400]}")
    if skl:
        lines.append("Składniki: " + "; ".join(x for x in skl[:10] if x))
    if kroki:
        lines.append("Jak się robi (skrót): " + " ".join(kroki[:5])[:600])
    for key in ("wskazowki_tradycyjne", "wymogi_szamanskie_i_czasowe", "bezpieczenstwo_i_interakcje", "uwagi"):
        val = d.get(key)
        if val:
            lines.append(f"{key.replace('_', ' ')}: {json.dumps(val, ensure_ascii=False)[:300]}")
    task = ("Opowiedz o tym nietypowym przepisie, który Siedziba przyniosła z zagranicznej strony. Dziadek się "
            "dziwi, kręci nosem albo przyznaje, że coś w tym jest; porównaj z tym, co robiło się u nas. Podaj, "
            "skąd przepis pochodzi. Nie przepisuj całego przepisu – zachęć, żeby szczegóły sprawdzić w Kwiatowniku.")
    return Material(kind="przepis", label=d["tytul"], task=task, data="\n".join(lines), season=season,
                    link_text="Przepisy w Kwiatowniku 👉", link_url=f"{SITE_URL}/przepisy/",
                    sources=[it.url], ref=f"item:{it.id}")


def auto_kind(s: Session) -> list[str]:
    """Kolejnosc prob dla trybu 'auto': najpierw posty o Siedzibie, potem rotacja tematow."""
    if count_kind(s, "siedziba") < INTRO_POSTS:
        return ["siedziba"]
    last: dict[str, int] = {}
    for kind, pid in s.exec(select(FbPost.kind, FbPost.id).where(FbPost.status != "rejected")).all():
        if kind:
            last[kind] = max(last.get(kind, 0), pid)
    return sorted(AUTO_ROTATION, key=lambda k: last.get(k, 0))   # najdawniej uzywany temat pierwszy
