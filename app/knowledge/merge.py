"""Scalanie wiedzy z sieci z rozdzialami pliku rosliny Kwiatownika.

Wiedza z sieci (blok "wiedza": sekcje -> punkty ze zrodlami) jest wmontowywana w istniejace rozdzialy
i podrozdzialy strony (app/knowledge/schema.py), np. punkt o dzialaniu lisci -> "Surowce i zbiory -> Liscie:
Dzialanie", punkt o barwieniu -> "Praktyczne zastosowanie -> Rzemioslo".

Kroki:
1. przydzial punktow do podrozdzialow: regula (sekcja + czesc rosliny + slowa kluczowe), poprawiana przez LLM
   (lista dozwolonych podrozdzialow; zly przydzial -> regula);
2. dla kazdego podrozdzialu z nowymi punktami LLM pisze go na nowo: tekst Kwiatownika + punkty z sieci,
   podobne informacje polaczone w jedno zdanie, kazde zdanie z numerami informacji, na ktorych sie opiera;
3. kontrola kodem: tresc tekstu Kwiatownika zachowana, zdania wierne zrodlom (rdzenie slow, liczby tylko
   z oryginalow), kazdy punkt uzyty albo odrzucony jako "nie pasuje", bez HTML. Gdy LLM zawiedzie albo
   kontrola nie przejdzie - scalenie bez LLM: tekst Kwiatownika + nowe punkty, punkty podobne do zdania
   Kwiatownika dopisane do niego jako zrodlo potwierdzajace.

Wynik trafia do bloku "scalone" pliku rosliny - reczne pola zostaja nietkniete:
  "scalone": {"wersja": 1, "zaktualizowano": "2026-10-01",
              "pola": {"zastosowanie.medyczne": {"tytul", "typ": "tekst"|"lista", "oryginal": <reczna wartosc>,
                        "tresc": [{"tekst", "zrodla": [nr z wiedza.zrodla], "fakty": ["s12"]}],
                        "model": "ollama/SpeakLeash/..." | "bez LLM", "wejscie": "<odcisk>"}},
              "uzyte_fakty": ["s12", ...]}
Strona pokazuje wersje scalona tylko, gdy "oryginal" == obecna wartosc pola (inaczej: tekst reczny, bo
ktos go w miedzyczasie zmienil); oryginal mozna zawsze rozwinac.
"""
import contextvars
import hashlib
import json
import re
import time
from datetime import date

from app.knowledge.extract import numbers_supported
from app.knowledge.schema import (CORE_PART_FIELDS, Slot, filled, web_parts, get_path, merge_slots, part_field_key, part_key_for, part_slots,
                                  slot_for)
from app.knowledge.sections import norm_part, norm_text, similar

HTML_RE = re.compile(r"<[a-z/!]", re.I)
MAX_SENTENCE = 700
MAX_LLM_POINTS = 18          # wiecej punktow w jednym podrozdziale -> scalanie bez LLM (male modele lokalne)
ROUTE_BATCH = 30
KEEP_ORIGINAL = 0.8          # tyle rdzeni slow tekstu Kwiatownika musi zostac po scaleniu
FAITHFUL = 0.6
SAME_INFO = 0.45             # podobienstwo, od ktorego punkt "potwierdza" zdanie Kwiatownika (bez LLM)


# ------------------------------------------------------------------ pomocnicze
# slowa "lacznikowe" parafrazy (wystepuje, zawiera, takze...) - nie niosa nowej informacji
_STOP = {"wyste", "zawie", "obejm", "posta", "posia", "pojaw", "tworz", "stosu", "uzywa", "takze", "rowni",
         "ktore", "ktory", "ktora", "ktorych", "jest", "maja", "moze", "moga", "tego", "tych", "oraz", "bardz",
         "rosli", "gatun", "charak", "wyroz", "czest", "bywa", "steze", "ilosc", "zawar", "takic", "jako",
         "przez", "wsrod", "ponad", "okolo", "wiele", "liczn", "rowne", "sklad", "zwlas", "glown", "dzial",
         "wplyw", "posia", "nalez", "sluzy", "wykaz", "zarow", "sposo", "rowni", "ktorz", "jedna", "inne",
         "innyc", "takze", "dodat", "przyp", "znany", "znane", "zwyk", "tradyc", "dawni", "obecn"}


def _stems(text: str) -> set[str]:
    return {w[:5] for w in norm_text(text).split() if len(w) > 3} - _STOP


def faithful(new: str, originals: list[str], threshold: float = FAITHFUL, slack: int = 1) -> bool:
    """Zdanie po redakcji wierne zrodlom: liczby tylko z oryginalow, a nowych rdzeni slow (spoza oryginalow
    i spoza slow lacznikowych) najwyzej 40% albo `slack` (krotkie zdania: jedno slowo parafrazy)."""
    joined = " ".join(originals)
    if not numbers_supported(new, joined):
        return False
    ns = _stems(new)
    novel = ns - _stems(joined)
    return len(novel) <= max(slack, (1 - threshold) * len(ns))


def covers(sentence: str, text: str) -> float:
    """Jaka czesc rdzeni slow tekstu (punktu / zdania Kwiatownika) jest w zdaniu."""
    ts = _stems(text)
    return 1.0 if not ts else len(ts & _stems(sentence)) / len(ts)


def kept(original: str, out: str) -> float:
    os_ = _stems(original)
    return 1.0 if not os_ else len(os_ & _stems(out)) / len(os_)


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}: {as_text(v)}" for k, v in value.items())
    return str(value)


def as_items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [as_text(v) for v in value if as_text(v)]
    return [as_text(value)]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def fingerprint(original, points: list[dict]) -> str:
    raw = json.dumps([original, sorted(f for p in points for f in p["fakty"])], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _parse(raw: str) -> dict:
    data = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if not isinstance(data, dict):
        raise ValueError("odpowiedz nie jest obiektem JSON")
    return data


# ------------------------------------------------------------------ punkty z sieci
def collect_points(data: dict) -> list[dict]:
    """Punkty z uporzadkowanej wiedzy (wiedza.sekcje) -> lista z numerami pid (1..n)."""
    out = []
    for sec in ((data.get("wiedza") or {}).get("sekcje") or []):
        for p in sec.get("punkty") or []:
            if not p.get("tekst") or not p.get("zrodla"):
                continue
            out.append({"pid": len(out) + 1, "sekcja": sec.get("klucz"), "czesc": p.get("czesc"),
                        "tekst": p["tekst"], "zrodla": list(p.get("zrodla") or []), "fakty": list(p.get("fakty") or [])})
    return out


_KW = {
    "interakcje": ("interakc", "lekami", "leków", "lekow", "leki ", "antykoagul", "cyp", "warfaryn", "metaboli"),
    "wymagania.gleba": ("gleb", "podłoż", "podloz", "piaszcz", "glin"),
    "wymagania.ph_gleby": ("ph ", "odczyn", "kwaśn", "kwasn", "zasadow", "wapien"),
    "wymagania.mrozoodpornosc": ("mróz", "mroz", "mrozo", "strefa"),
    "wymagania.woda": ("wilgotn", "podlew", "susz", "wod"),
    "wymagania.stanowisko": ("słońc", "slonc", "cień", "cien", "stanowisk", "półcie", "polcie", "naslonecz", "nasłonecz"),
    "profil_energetyczny.smak": ("smak",),
    "profil_energetyczny.termika": ("chłodz", "chlodz", "chłodn", "chlodn", "ciepł", "ciepl", "zimn", "natur"),
    "profil_energetyczny.zywiol": ("meridian", "kanał", "kanal", "żywioł", "zywiol", "przemian"),
    "czas_zbioru": ("zbier", "zbiór", "zbior", "susz", "kwitnieni", "jesieni", "wiosn", "zbioru"),
}


def _has(text: str, key: str) -> bool:
    t = f" {text.lower()} "
    return any(k in t for k in _KW[key])


def default_route(data: dict, point: dict) -> str | None:
    """Przydzial punktu do podrozdzialu regulami (bez LLM). None = zostaje tylko w "Wiedzy z sieci"."""
    sec, text = point.get("sekcja"), point.get("tekst") or ""
    pkey = part_key_for(data, point.get("czesc"))
    parts = data.get("czesci_rosliny") if isinstance(data.get("czesci_rosliny"), dict) else {}
    if not pkey and not norm_part(point.get("czesc")) and sec == "sklad" and len(parts) == 1:
        pkey = next(iter(parts))
    if pkey and sec in ("zastosowanie_lecznicze", "sklad", "czesci_rosliny", "bezpieczenstwo"):
        field = {"zastosowanie_lecznicze": "wlasciwości", "sklad": "skladniki_aktywne",
                 "bezpieczenstwo": "ostrzezenia"}.get(sec)
        if sec == "czesci_rosliny":
            field = "czas_zbioru" if _has(text, "czas_zbioru") else "opis_botaniczny"
        fk = part_field_key(parts.get(pkey) or {}, field)
        if any(s.path == f"czesci_rosliny.{pkey}.{fk}" for s in part_slots(data)):
            return f"czesci_rosliny.{pkey}.{fk}"
    part = norm_part(point.get("czesc"))
    if not pkey and part and sec in ("zastosowanie_lecznicze", "sklad", "czesci_rosliny", "bezpieczenstwo"):
        field = {"zastosowanie_lecznicze": "wlasciwości", "sklad": "skladniki_aktywne",
                 "bezpieczenstwo": "ostrzezenia"}.get(sec)
        if sec == "czesci_rosliny":
            field = "czas_zbioru" if _has(text, "czas_zbioru") else "opis_botaniczny"
        return f"czesci_rosliny.{part}.{field}"           # nowa czesc rosliny (tylko z sieci)
    if sec == "zastosowanie_lecznicze":
        return "zastosowanie.medyczne"
    if sec == "barwienie":
        return "zastosowanie.rzemieslnicze"
    if sec == "kosmetyka":
        return "zastosowanie.kosmetyczne"
    if sec == "bezpieczenstwo":
        return "interakcje" if _has(text, "interakcje") else "ostrzezenia"
    if sec == "opis":
        return "identyfikacja.cechy_kluczowe"
    if sec == "medycyna_wschodu":
        for key in ("profil_energetyczny.smak", "profil_energetyczny.termika", "profil_energetyczny.zywiol"):
            if _has(text, key):
                return key
        return "profil_energetyczny.opis"
    if sec in ("uprawa", "wystepowanie"):
        for key in ("wymagania.ph_gleby", "wymagania.mrozoodpornosc", "wymagania.gleba", "wymagania.stanowisko",
                    "wymagania.woda"):
            if _has(text, key):
                return key
        return "zastosowanie.ogrodowe" if sec == "uprawa" else None
    if sec in ("ciekawostki", "historia", "kultura", "nazwy_ludowe"):
        return "ciekawostki"
    return None


# ------------------------------------------------------------------ przydzial (LLM)
ROUTE_SYSTEM = """Jestes redaktorem polskiego zielnika "Kwiatownik". Strona rosliny ma rozdzialy i podrozdzialy \
(lista S1, S2...). Przydziel KAZDA informacje z sieci do JEDNEGO podrozdzialu, do ktorego najlepiej pasuje \
tresciowo, albo "brak", gdy zaden nie pasuje. Informacja o konkretnej czesci rosliny (np. korzeniu) -> \
podrozdzial tej czesci, jesli jest na liscie. Nie zmieniaj tekstu informacji.
Zwroc TYLKO JSON: {"przydzial": [{"id": 1, "podrozdzial": "S3"}, {"id": 2, "podrozdzial": "brak"}]}"""


def route_prompt(plant_pl: str, slots: list[Slot], points: list[dict]) -> str:
    s_lines = [f"S{i} | {s.chapter} -> {s.title} | {s.hint}" for i, s in enumerate(slots, 1)]
    p_lines = [f"{p['pid']} | {p['sekcja']} | {p.get('czesc') or '-'} | {p['tekst']}" for p in points]
    return (f"ROSLINA: {plant_pl}\n\nPODROZDZIALY:\n" + "\n".join(s_lines) +
            "\n\nINFORMACJE (id | sekcja | czesc | tekst):\n" + "\n".join(p_lines))


def apply_routing(raw: str, slots: list[Slot], points: list[dict]) -> dict[int, str | None]:
    """Odpowiedz LLM -> {pid: sciezka|None}; tylko poprawne wpisy (reszta zostaje przy regule)."""
    data = _parse(raw)
    ids = {p["pid"] for p in points}
    out: dict[int, str | None] = {}
    for row in data.get("przydzial") or []:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        label = str(row.get("podrozdzial") or "").strip().upper()
        if pid not in ids:
            continue
        if label in ("BRAK", "NULL", "NONE", ""):
            out[pid] = None
            continue
        m = re.fullmatch(r"S?(\d+)", label)
        if m and 1 <= int(m.group(1)) <= len(slots):
            out[pid] = slots[int(m.group(1)) - 1].path
    return out


async def route(llm, data: dict, plant_pl: str, points: list[dict], prefer=None, log=None) -> dict[int, str | None]:
    say = log or (lambda m: None)
    routes = {p["pid"]: default_route(data, p) for p in points}
    if llm is None or not points:
        return require_core_parts(data, routes, points, say)
    slots = merge_slots(data)
    changed = 0
    for i in range(0, len(points), ROUTE_BATCH):
        batch = points[i:i + ROUTE_BATCH]
        try:
            res = await _complete(llm, route_prompt(plant_pl, slots, batch), ROUTE_SYSTEM, prefer)
            got = apply_routing(res.text, slots, batch)
        except Exception as exc:                       # przydzial regulami wystarczy
            say(f"  przydzial do rozdzialow: LLM niedostepny ({str(exc)[:80]}) - reguly")
            continue
        for pid, path in got.items():
            if routes.get(pid) != path:
                changed += 1
            routes[pid] = path
    if changed:
        say(f"  przydzial do rozdzialow: LLM poprawil {changed} z {len(points)} punktow")
    return require_core_parts(data, routes, points, say)


PART_KEY_SECTIONS = {"sklad": "skladniki_aktywne", "zastosowanie_lecznicze": "wlasciwości"}


def _part_of_path(path: str | None) -> str | None:
    bits = (path or "").split(".")
    return bits[1] if len(bits) == 3 and bits[0] == "czesci_rosliny" else None


def require_core_parts(data: dict, routes: dict[int, str | None], points: list[dict], say=None) -> dict[int, str | None]:
    """Czesci rosliny w generatorze:
    1. informacja o SKLADZIE albo DZIALANIU konkretnej czesci (sekcja sklad / zastosowanie_lecznicze + czesc)
       zawsze trafia do podrozdzialu tej czesci - takze gdy LLM przydzielil ja do rozdzialu calej rosliny;
    2. nowa czesc (tylko z sieci) powstaje, gdy ma substancje czynne ALBO dzialanie (wystarczy jedno z nich);
       czesc bez obu (np. tylko opis lub ostrzezenie) nie powstaje - jej punkty ida do rozdzialow calej rosliny."""
    out = dict(routes)
    fixed = 0
    for p in points:
        if p.get("sekcja") not in PART_KEY_SECTIONS or not norm_part(p.get("czesc")):
            continue
        rule = default_route(data, p)                     # podrozdzial tej czesci wg reguly
        want = _part_of_path(rule)
        if want and _part_of_path(out.get(p["pid"])) != want:
            out[p["pid"]] = rule
            fixed += 1
    new = set(web_parts(data))
    by_part: dict[str, set[str]] = {}
    for path in out.values():
        part = _part_of_path(path)
        if part in new:
            by_part.setdefault(part, set()).add(path.split(".")[2])
    weak = {part for part, fields in by_part.items() if not fields & set(CORE_PART_FIELDS)}
    for p in points:
        if _part_of_path(out.get(p["pid"])) in weak:
            out[p["pid"]] = default_route(data, dict(p, czesc=None))
    if say:
        strong = sorted(part for part in by_part if part not in weak)
        if strong:
            say(f"  nowe czesci rosliny z sieci (do generatora): {', '.join(strong)}")
        if fixed:
            say(f"  sklad/dzialanie czesci przywrocone do ich podrozdzialow: {fixed}")
        if weak:
            say(f"  czesci bez skladu i dzialania (nie trafia do generatora): {', '.join(sorted(weak))}")
    return out


PREFER_TIMEOUT_S = 120.0     # limit jednego zapytania do modelu preferowanego (Bielik) - potem chmura


# stan chmury w jednym scalaniu rosliny: gdy Gemini/Groq raz nie odpowiedza (limity, kwota dzienna),
# kolejne podrozdzialy nie czekaja na nie od nowa (do 90 s na zapytanie) - tylko Bielik albo reguly
_CLOUD = contextvars.ContextVar("merge_cloud_state", default=None)


class CloudDown(Exception):
    pass


async def _complete(llm, prompt: str, system: str, prefer, timeout: float | None = None):
    """Najpierw modele preferowane (Bielik, z limitem czasu), potem TYLKO modele z chmury: lokalny model
    zapasowy (np. qwen3 liczacy na CPU) potrafi odpowiadac kilkanascie minut - wtedy lepiej scalic regulami."""
    state = _CLOUD.get()
    down = bool(state and state.get("down"))
    if down and not prefer:
        raise CloudDown("modele z chmury niedostepne w tym scalaniu")
    try:
        return await llm.complete(prompt, system=system, json_mode=True, prefer=prefer or None,
                                  prefer_timeout=timeout or PREFER_TIMEOUT_S, cloud_only=True, prefer_only=down)
    except TypeError:                                  # atrapa / stary router bez tych opcji
        pass
    except CloudDown:
        raise
    except Exception:
        if state is not None:
            state["down"] = True
        raise
    if prefer:
        try:
            return await llm.complete(prompt, system=system, json_mode=True, prefer=prefer)
        except TypeError:
            pass
    return await llm.complete(prompt, system=system, json_mode=True)


# ------------------------------------------------------------------ scalanie podrozdzialu
MERGE_SYSTEM = """Jestes redaktorem polskiego zielnika "Kwiatownik". Piszesz na nowo JEDEN podrozdzial strony \
rosliny: laczysz tekst Kwiatownika z nowymi informacjami z sieci (P1, P2...).
ZASADY:
- Zachowaj CALA tresc tekstu Kwiatownika - mozesz ja przeredagowac i polaczyc z informacjami z sieci, ale nie
  usuwaj zadnego faktu.
- Dodaj informacje z sieci pasujace do tego podrozdzialu. Informacje mowiace to samo (albo prawie to samo)
  polacz w jedno zdanie i podaj wszystkie ich numery.
- NIE dodawaj niczego spoza podanych tekstow: zadnych nowych faktow, nazw, liczb ani dawek.
- "z" = numery informacji z sieci, na ktorych opiera sie zdanie (np. [1, 3]); zdanie wylacznie z tekstu
  Kwiatownika -> "z": [].
- Informacje z sieci, ktore nie pasuja do tego podrozdzialu, wpisz do "nie_pasuje".
- Pisz poprawna polszczyzna, rzeczowo i opisowo (bez rozkazow typu "stosuj", "pij").
FORMAT_PLACEHOLDER
Zwroc TYLKO JSON: {"zdania": [{"tekst": "...", "z": [1]}], "nie_pasuje": []}"""
FORMAT_TEXT = "- Podrozdzial to akapit tekstu: zwroc kolejne zdania akapitu w logicznej kolejnosci."
FORMAT_LIST = ("- Podrozdzial to LISTA: kazdy element listy (krotki, 1-2 zdania) to osobny wpis w \"zdania\"; "
               "kazdy element listy Kwiatownika musi zostac (moze byc uzupelniony).")


def merge_prompt(plant_pl: str, latin: str | None, slot: Slot, original, points: list[dict]) -> str:
    if slot.kind == "lista":
        orig = "\n".join(f"- {x}" for x in as_items(original)) or "(pusto)"
    else:
        orig = as_text(original) or "(pusto - podrozdzial jeszcze nie istnieje)"
    lines = [f"P{i} | {p['tekst']}" for i, p in enumerate(points, 1)]
    return (f"ROSLINA: {plant_pl}" + (f" ({latin})" if latin else "") +
            f"\nPODROZDZIAL: {slot.chapter} -> {slot.title} ({slot.hint})\n\nTEKST KWIATOWNIKA:\n{orig}\n\n"
            "INFORMACJE Z SIECI:\n" + "\n".join(lines))


def _sources(points: list[dict]) -> tuple[list[int], list[str]]:
    return (sorted({n for p in points for n in p["zrodla"]}),
            [f for p in points for f in p["fakty"]])


def apply_merge(raw: str, slot: Slot, original, points: list[dict]) -> tuple[list[dict] | None, list[int], list[str]]:
    """Odpowiedz LLM -> (tresc podrozdzialu, pid-y "nie pasuje", uwagi). tresc=None -> odpowiedz bezuzyteczna.

    Kod nie odrzuca calej odpowiedzi za drobne bledy - naprawia ja:
    - zdanie niewierne zrodlom (nowe fakty, liczby spoza zrodel, HTML) -> usuwane,
    - zrodla zdania: numery podane przez LLM (jesli zdanie naprawde z nich korzysta) + punkty, ktorych tresc
      zdanie zawiera (LLM czesto myli numery),
    - zgubione zdania / elementy tekstu Kwiatownika -> dopisane z oryginalu,
    - punkty z sieci pominiete (i nie "nie_pasuje") -> dopisane jako osobne zdania ze zrodlem.
    Calosc odrzucana tylko gdy: zly JSON, brak zdan albo wiecej niz polowa zdan zmyslona."""
    notes: list[str] = []
    try:
        data = _parse(raw)
    except ValueError:
        return None, [], ["niepoprawny JSON"]
    orig_units = as_items(original) if slot.kind == "lista" else sentences(as_text(original))
    orig_text = " ".join(orig_units)
    by_num = {i: p for i, p in enumerate(points, 1)}
    base = ([orig_text] if orig_text else []) + [p["tekst"] for p in points]
    rows = [r for r in (data.get("zdania") or []) if isinstance(r, dict)]
    out: list[dict] = []
    used: set[int] = set()
    dropped = 0
    for row in rows:
        text = re.sub(r"\s+", " ", str(row.get("tekst") or "")).strip()
        nums_raw = row.get("z") or []
        nums_raw = nums_raw if isinstance(nums_raw, list) else [nums_raw]
        nums = set()
        for n in nums_raw:
            try:
                nums.add(int(str(n).strip().lstrip("Pp")))
            except ValueError:
                pass
        if not (3 <= len(text) <= MAX_SENTENCE) or HTML_RE.search(text) or not faithful(text, base):
            dropped += 1
            notes.append(f"usunieto zdanie spoza zrodel: {text[:50]}")
            continue
        cited = {n for n in nums if n in by_num and covers(text, by_num[n]["tekst"]) >= 0.25}
        cited |= {n for n, p in by_num.items() if covers(text, p["tekst"]) >= 0.5}
        used |= cited
        pts = [by_num[n] for n in sorted(cited)]
        zr, fk = _sources(pts)
        out.append({"tekst": text, "zrodla": zr, "fakty": fk})
    if not out or dropped > len(rows) / 2:
        return None, [], notes + ["za duzo zdan spoza zrodel"]
    joined = " ".join(t["tekst"] for t in out)
    lost = [u for u in orig_units if covers(joined, u) < 0.6]          # zgubione zdania Kwiatownika
    for u in lost:
        out.append({"tekst": u, "zrodla": [], "fakty": []})
    if lost:
        notes.append(f"dopisano zgubiony tekst Kwiatownika ({len(lost)})")
    rejected = set()
    for n in data.get("nie_pasuje") or []:
        try:
            rejected.add(int(str(n).strip().lstrip("Pp")))
        except ValueError:
            pass
    missing = [n for n in by_num if n not in used and n not in rejected]
    for n in missing:                                                  # pominiete punkty z sieci
        p = by_num[n]
        out.append({"tekst": p["tekst"], "zrodla": sorted(set(p["zrodla"])), "fakty": list(p["fakty"])})
    if missing:
        notes.append(f"dopisano pominiete informacje ({len(missing)})")
    return out, [by_num[n]["pid"] for n in sorted(rejected & set(by_num) - used)], notes


def deterministic(slot: Slot, original, points: list[dict]) -> list[dict]:
    """Scalenie bez LLM: tekst/elementy Kwiatownika + nowe punkty. Punkt podobny do zdania Kwiatownika
    (albo do innego punktu) nie jest powtarzany - jego zrodla dopisuja sie do tamtego zdania."""
    units = as_items(original) if slot.kind == "lista" else sentences(as_text(original))
    out = [{"tekst": u, "zrodla": [], "fakty": []} for u in units]
    for p in points:
        best = max(out, key=lambda o: similar(o["tekst"], p["tekst"]), default=None)
        if best is not None and similar(best["tekst"], p["tekst"]) >= SAME_INFO:
            best["zrodla"] = sorted(set(best["zrodla"]) | set(p["zrodla"]))
            best["fakty"] = best["fakty"] + [f for f in p["fakty"] if f not in best["fakty"]]
        else:
            out.append({"tekst": p["tekst"], "zrodla": sorted(set(p["zrodla"])), "fakty": list(p["fakty"])})
    return out


async def merge_slot(llm, plant_pl: str, latin: str | None, slot: Slot, original, points: list[dict],
                     prefer=None, log=None, timeout: float | None = None) -> tuple[list[dict], list[int], str]:
    """(tresc, pid-y "nie pasuje", jak: 'llm:provider/model' | 'bez LLM: powod')."""
    say = log or (lambda m: None)
    if llm is None:
        return deterministic(slot, original, points), [], "bez LLM"
    if len(points) > MAX_LLM_POINTS:
        return deterministic(slot, original, points), [], f"bez LLM: {len(points)} punktow"
    system = MERGE_SYSTEM.replace("FORMAT_PLACEHOLDER", FORMAT_LIST if slot.kind == "lista" else FORMAT_TEXT)
    notes: list[str] = []
    for attempt in range(2):                           # druga proba - zwykle trafia na inny model
        try:
            res = await _complete(llm, merge_prompt(plant_pl, latin, slot, original, points), system,
                                  prefer if attempt == 0 else None, timeout)
        except Exception as exc:
            say(f"    {slot.title}: LLM niedostepny ({str(exc)[:80]}) - scalenie bez LLM")
            return deterministic(slot, original, points), [], "bez LLM: brak modelu"
        tresc, rejected, notes = apply_merge(res.text, slot, original, points)
        if tresc is not None:
            secs = getattr(res, "seconds", None)
            say(f"    {slot.title}: {res.provider}/{res.model}" + (f" {secs:.0f} s" if isinstance(secs, (int, float)) else "")
                + (" (" + "; ".join(notes[:3]) + ")" if notes else ""))
            return tresc, rejected, f"llm:{res.provider}/{res.model}"
        say(f"    {slot.title}: odpowiedz {res.provider}/{res.model} odrzucona - " + "; ".join(notes[:3]))
    return deterministic(slot, original, points), [], "bez LLM: " + (notes[-1] if notes else "kontrola")


# ------------------------------------------------------------------ calosc
async def build_merged(llm, data: dict, plant_pl: str, latin: str | None = None, prefer=None, log=None,
                       today: str | None = None, redo_other_models: bool = False,
                       budget_s: float | None = None, prefer_timeout: float | None = None) -> dict | None:
    """Blok "scalone" dla pliku rosliny (None, gdy nie ma czego scalac).
    redo_other_models: podrozdzialy scalone wczesniej innym modelem niz preferowane (np. Gemini, gdy Bielik
    jeszcze sie pobieral) sa scalane ponownie - teraz preferowanym (Bielik).
    budget_s: limit czasu na rosline dla modeli preferowanych (potem chmura); prefer_timeout: limit jednego
    zapytania do modelu preferowanego (model za wolny -> pomijany przez 30 min)."""
    say = log or (lambda m: None)
    points = collect_points(data)
    if not points:
        return None
    previous = ((data.get("scalone") or {}).get("pola") or {}) if isinstance(data.get("scalone"), dict) else {}
    # przydzial = dluga klasyfikacja (wszystkie punkty i podrozdzialy) - szybciej i lepiej modelami z listy;
    # Bielik pisze tylko teksty podrozdzialow (krotkie prompty)
    _CLOUD.set({"down": False})
    routes = await route(llm, data, plant_pl, points, None, log)
    by_slot: dict[str, list[dict]] = {}
    for p in points:
        path = routes.get(p["pid"])
        if path and slot_for(data, path):
            by_slot.setdefault(path, []).append(p)
    pola: dict[str, dict] = {}
    stats = {"llm": 0, "bez_llm": 0, "z_pamieci": 0}
    started = time.monotonic()
    use_prefer = prefer
    for path, pts in by_slot.items():
        if use_prefer and budget_s and time.monotonic() - started > budget_s:
            use_prefer = None                          # limit czasu rosliny: reszta podrozdzialow w chmurze
            say(f"  limit czasu scalania z modelem lokalnym ({int(budget_s)} s) - pozostale podrozdzialy: modele z listy")
        slot = slot_for(data, path)
        original = get_path(data, path)
        original = original if filled(original) else None
        key = fingerprint(original, pts)
        old = previous.get(path)
        preferred = {m.split(":", 1)[1] if ":" in m else m for m in (prefer or [])}
        by_other = redo_other_models and preferred and not any(
            str(old.get("model") or "").endswith(m) for m in preferred) if old else False
        if old and old.get("wejscie") == key and old.get("tresc") and not by_other:
            pola[path] = old
            stats["z_pamieci"] += 1
            continue
        tresc, _rejected, how = await merge_slot(llm, plant_pl, latin, slot, original, pts, use_prefer, log,
                                                 timeout=prefer_timeout)
        if not any(t["zrodla"] for t in tresc):
            continue                                   # LLM uznal, ze nic z sieci tu nie pasuje
        stats["llm" if how.startswith("llm:") else "bez_llm"] += 1
        pola[path] = {"tytul": f"{slot.chapter} - {slot.title}", "typ": "lista" if slot.kind == "lista" else "tekst",
                      "oryginal": original, "tresc": tresc, "model": how.removeprefix("llm:"), "wejscie": key}
    if not pola:
        return None
    used = sorted({f for e in pola.values() for t in e["tresc"] for f in t["fakty"]})
    outside = sum(1 for p in points if not set(p["fakty"]) <= set(used))
    say(f"  scalanie z rozdzialami: {len(pola)} podrozdzialow (LLM {stats['llm']}, bez LLM {stats['bez_llm']}, "
        f"bez zmian {stats['z_pamieci']}); punkty tylko w 'Wiedzy z sieci': {outside}")
    return {"wersja": 1, "zaktualizowano": today or date.today().isoformat(), "pola": pola, "uzyte_fakty": used}


def with_merged(cand: dict, scalone: dict | None) -> dict:
    if scalone:
        cand["scalone"] = scalone
    else:
        cand.pop("scalone", None)
    return cand


def validate_merged(cand: dict) -> list[str]:
    """Kontrola bloku "scalone" w kopii pliku (uzywana przez plantfile.validate_candidate)."""
    sc = cand.get("scalone")
    if sc is None:
        return []
    if not isinstance(sc, dict) or not isinstance(sc.get("pola"), dict):
        return ["zly blok scalone"]
    errors: list[str] = []
    wiedza = cand.get("wiedza") or {}
    nrs = {z.get("nr") for z in wiedza.get("zrodla") or [] if isinstance(z, dict)}
    fact_ids = {f.get("id") for f in wiedza.get("fakty") or [] if isinstance(f, dict)}
    for path, entry in sc["pola"].items():
        if slot_for(cand, path) is None:
            errors.append(f"scalone: nieznany podrozdzial {path}")
            continue
        current = get_path(cand, path)
        if entry.get("oryginal") is not None and entry.get("oryginal") != current:
            errors.append(f"scalone: oryginal {path} rozny od pola w pliku")
        if entry.get("typ") not in ("tekst", "lista"):
            errors.append(f"scalone: zly typ {path}")
        tresc = entry.get("tresc")
        if not isinstance(tresc, list) or not tresc:
            errors.append(f"scalone: pusta tresc {path}")
            continue
        for t in tresc:
            text = str(t.get("tekst") or "")
            if not (3 <= len(text) <= MAX_SENTENCE) or HTML_RE.search(text):
                errors.append(f"scalone: zly tekst w {path}")
            if not set(t.get("zrodla") or []) <= nrs:
                errors.append(f"scalone: zrodlo spoza listy w {path}")
            if not set(t.get("fakty") or []) <= fact_ids:
                errors.append(f"scalone: nieistniejaca informacja w {path}")
    if not set(sc.get("uzyte_fakty") or []) <= fact_ids:
        errors.append("scalone: uzyte_fakty spoza wiedzy")
    return errors
