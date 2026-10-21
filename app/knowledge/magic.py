"""Magia roslin: zespol algorytmow szukajacych dawnej wiedzy magicznej, zabobonow, obrzedow i praktyk czarownic,
w ktorych uzywano roslin - takze tam, gdzie praktyka ma inna nazwe w innym kraju.

Algorytmy (zadanie "plant_magic", app/worker/magic.py):
1. Generator zapytan dla rosliny (plant_queries): nazwa rosliny w jezyku (Wikidata, nazwy ludowe znalezione
   wczesniej) + pojecia z tezaurusa (magic_lexicon.CONCEPTS) w jezykach, w ktorych dany temat jest najlepiej
   opisany; rotacja pojec i jezykow, bez zapytan zadanych niedawno.
2. Azymuty (tropy, MagicLead): z kazdej strony LLM wypisuje lokalne nazwy rosliny, nazwy praktyk i odpowiedniki
   w innych kulturach. Kazdy trop staje sie nowym zapytaniem (w jezyku tropu) - wyszukiwanie "idzie za sladem"
   do glebokosci MAGIC_MAX_DEPTH, a tropy, ktore daly wiedze, prowadza dalej.
3. Glebokie archiwa (DEEP_SOURCES): zapytania site: do bibliotek cyfrowych i starych ksiag (Zeno, Wikisource,
   ctext, Aozora, LacusCurtius, CyberLeninka...). Z dlugich tekstow brane sa fragmenty wokol nazwy rosliny.
4. Przeszukiwanie "od praktyki" (SWEEPS, sweep_prompt): strona o praktyce (np. ziola na Matke Boska Zielna,
   Hexensalbe) -> LLM wypisuje informacje o WIELU roslinach naraz -> rozpoznanie roslin Kwiatownika (PlantIndex).
5. Kontrola kodem: cytat musi byc w zrodle, liczby z cytatu, bez ilosci i dawek (DOSAGE_RE), wspolczesna
   magia "new age" odrzucana (MAGIC_INCLUDE_MODERN=false), rodzaj praktyki ze slownika; potem weryfikacja drugim
   modelem jak cala wiedza o roslinach.
Informacje maja sekcje "magia" i metadane (rodzaj, kraj, epoka, tradycja, toksyczne) - rozmieszczenie
(placement.py) ustawia je w rozdziale "Magia i zabobony" w podrozdziale rodzaju praktyki.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field

from app.knowledge.extract import numbers_supported, quote_in_text
from app.knowledge.magic_lexicon import (CONCEPT_BY_KEY, CONCEPTS, DEEP_SOURCES, RODZAJ_KEYS, RODZAJE, STEMS,
                                         concept_order, magic_score)
from app.knowledge.sections import PARTS, norm_part, norm_text

MAX_FACTS = 12
MAX_LEADS = 6
LEAD_TYPES = ("nazwa", "praktyka", "analogia")
# ilosci i dawki - opis dawnej praktyki nie moze byc przepisem do powtorzenia
DOSAGE_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:g\b|gram|mg\b|ml\b|l\b|litr|kropl|łyżk|lyzk|szklan|garś|gars|"
                       r"uncj|drachm|łut|lut\b|szczypt|%)", re.I)
MODERN_RE = re.compile(r"wsp[oó][lł]czes|xx\s*w|xxi|wicca|wikka|new\s*age|neopog|neopag", re.I)

MAGIC_SYSTEM = """Jestes badaczem etnobotaniki i folkloru dla polskiego zielnika "Kwiatownik". Z tekstu zrodla \
wybierasz informacje o MAGICZNYCH i obrzedowych zastosowaniach JEDNEJ rosliny: czary, zabobony i przesady, \
amulety i ochrona przed zlem, uroki i ich odczynianie, magia milosna, wrozby i sny, obrzedy doroczne, rosliny \
czarownic, klatwy (czarna magia), magiczne leczenie i zamawianie, duchy i demony.
ZASADY:
- Tylko to, co JEST w tekscie i dotyczy TEJ rosliny (albo jej czesci). Nie dopisuj nic od siebie.
- "tekst": 1-2 zdania po polsku, wlasnymi slowami. Jesli zrodlo mowi GDZIE, KIEDY i KTO (kraj/region, epoka,
  grupa ludzi, autor dawnej ksiegi) - zawrzyj to w zdaniu, np. "Na Podlasiu w XIX w. wierzono, ze...".
- OPISUJ, NIE INSTRUUJ: zadnych ilosci, proporcji, dawek, krokow przygotowania ani sposobu uzycia mieszanin,
  ktore sie je, pije, pali, wdycha albo wciera w skore. Rosliny trujace (pokrzyk, lulek, bielun, mandragora, tojad,
  szalej, cis...) - tylko historyczny opis i "toksyczne": true.
- "cytat": dokladny fragment zrodla w oryginalnym jezyku (znak w znak, max 250 znakow), ktory to potwierdza.
- "rodzaj": jeden z: """ + ", ".join(f"{k} ({v[2][:70]})" for k, v in RODZAJE.items()) + """
- "kraj": kraj lub region tradycji (po polsku) albo null; "epoka": np. "starożytność", "średniowiecze",
  "XVI w.", "XIX w.", "współczesność" albo null; "tradycja": krotko skad wiedza, np. "wierzenia ludowe",
  "zielnik Mattioliego", "procesy czarownic", "zapis etnograficzny", albo null.
- Wspolczesne "korespondencje" wicca / new age bez podstaw w dawnych zrodlach -> epoka "współczesność".
- "czesc": czesc rosliny TYLKO z listy: """ + ", ".join(PARTS) + """ - albo null.
- "tropy" (azymuty do dalszych poszukiwan, max 5): slowa z TEGO tekstu albo wiedza, ktora pomoze znalezc wiecej
  o magii tej rosliny w innych zrodlach:
  * typ "nazwa": ludowa/lokalna nazwa rosliny zwiazana z magia (np. "Hexenkraut", "czarcie ziele", "одолень-трава"),
  * typ "praktyka": nazwa obrzedu, zaklecia, amuletu albo dawnej ksiegi w oryginalnym jezyku (np. "Kräuterbuschen",
    "купальские травы", "malocchio"),
  * typ "analogia": ta sama lub podobna praktyka w INNEJ kulturze, w jezyku tej kultury (np. dla wianka
    swietojanskiego: "Johanniskranz" de, "вінок на Купала" uk).
  Kazdy trop: {"termin": "w oryginalnym pismie, max 60 znakow", "jezyk": "kod ISO 639-1", "typ": "...",
  "opis": "krotko po polsku, czego szukac"}.
Zwroc TYLKO JSON: {"fakty": [{"rodzaj": "...", "czesc": null, "tekst": "...", "cytat": "...", "kraj": null,
"epoka": null, "tradycja": null, "toksyczne": false}], "tropy": [...]}
Brak magicznych informacji o tej roslinie -> {"fakty": [], "tropy": []}."""

SWEEP_SYSTEM = MAGIC_SYSTEM.replace(
    "wybierasz informacje o MAGICZNYCH i obrzedowych zastosowaniach JEDNEJ rosliny",
    "wybierasz informacje o MAGICZNYCH i obrzedowych zastosowaniach ROSLIN wymienionych w tekscie (moze ich byc "
    "wiele)").replace(
    "- Tylko to, co JEST w tekscie i dotyczy TEJ rosliny (albo jej czesci). Nie dopisuj nic od siebie.",
    "- Tylko to, co JEST w tekscie. Kazda informacja dotyczy JEDNEJ rosliny: \"roslina\" = nazwa jak w tekscie, "
    "\"lacina\" = nazwa lacinska gatunku (z tekstu albo z wiedzy botanicznej; null, gdy niepewna).").replace(
    '{"fakty": [{"rodzaj": "...",', '{"fakty": [{"roslina": "...", "lacina": "...", "rodzaj": "...",')


@dataclass
class MagicFact:
    section: str
    part: str | None
    text: str
    quote: str
    meta: dict = field(default_factory=dict)       # rodzaj, kraj, epoka, tradycja, toksyczne
    plant: str | None = None                       # tylko przeszukiwanie "od praktyki": nazwa z tekstu
    latin: str | None = None


@dataclass
class MQuery:
    kind: str           # koncept | trop | archiwum
    lang: str
    query: str
    key: str            # magia:<koncept>:<jezyk> | magia:trop:<id> | magia:arch:<domena>
    rodzaj: str | None = None
    lead_id: int | None = None
    depth: int = 0


# ------------------------------------------------------------------ prompty
def build_prompt(plant_pl: str, latin: str | None, names: dict[str, str], source_name: str, lang: str, text: str,
                 known: list[str], hint: str | None = None) -> str:
    other = ", ".join(f"{k}: {v}" for k, v in list(names.items())[:12] if v and k not in ("pl",))
    known_txt = "\n".join(f"- {k}" for k in known[:30])
    return (f"ROSLINA: {plant_pl}" + (f" ({latin})" if latin else "") +
            (f"\nINNE NAZWY: {other}" if other else "") +
            f"\nZRODLO: {source_name} (jezyk: {lang})" +
            (f"\nTROP, ktorym tu trafilismy: {hint}" if hint else "") +
            (f"\n\nJUZ WIEMY (nie powtarzaj):\n{known_txt}" if known else "") +
            f"\n\nTEKST ZRODLA:\n{text}")


def sweep_prompt(source_name: str, lang: str, text: str, topic: str) -> str:
    return f"ZRODLO: {source_name} (jezyk: {lang})\nTEMAT STRONY: {topic}\n\nTEKST ZRODLA:\n{text}"


# ------------------------------------------------------------------ kontrola odpowiedzi
def _clean(v, limit: int = 80) -> str | None:
    s = re.sub(r"\s+", " ", str(v or "")).strip()
    if not s or s.lower() in ("null", "none", "-", "brak", "nieznany", "nieznana"):
        return None
    return s[:limit]


def norm_rodzaj(v) -> str:
    r = norm_text(str(v or "")).replace(" ", "_")
    if r in RODZAJE:
        return r
    for k in RODZAJ_KEYS:                       # "magia milosna" -> milosna, "czarna magia" -> czarna
        if k in r or r in k and len(r) >= 4:
            return k
    return "zabobony"


def is_modern(meta: dict) -> bool:
    return bool(MODERN_RE.search(f"{meta.get('epoka') or ''} {meta.get('tradycja') or ''}"))


def parse_facts(raw: str, source_text: str, include_modern: bool = False,
                sweep: bool = False) -> tuple[list[MagicFact], list[dict], list[str]]:
    """Odpowiedz LLM -> (informacje po kontroli, tropy, powody odrzucenia)."""
    data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
    if isinstance(data, list):
        data = {"fakty": data}
    if not isinstance(data, dict):
        return [], [], ["odpowiedz modelu w zlym formacie"]
    facts, rejected = [], []
    for f in (data.get("fakty") or [])[:MAX_FACTS * (3 if sweep else 1)]:
        if not isinstance(f, dict):
            continue
        text = re.sub(r"\s+", " ", str(f.get("tekst") or "")).strip()
        quote = re.sub(r"\s+", " ", str(f.get("cytat") or "")).strip()
        if not (20 <= len(text) <= 600):
            rejected.append(f"dlugosc: {text[:60]}")
            continue
        if not quote_in_text(quote, source_text):
            rejected.append(f"cytatu nie ma w zrodle: {text[:60]}")
            continue
        if not numbers_supported(text, quote):
            rejected.append(f"liczby spoza cytatu: {text[:60]}")
            continue
        if DOSAGE_RE.search(text):
            rejected.append(f"ilosci/dawki (opis, nie przepis): {text[:60]}")
            continue
        meta = {"rodzaj": norm_rodzaj(f.get("rodzaj")), "kraj": _clean(f.get("kraj"), 60),
                "epoka": _clean(f.get("epoka"), 40), "tradycja": _clean(f.get("tradycja"), 80),
                "toksyczne": f.get("toksyczne") is True}
        meta = {k: v for k, v in meta.items() if v not in (None, False)}
        if not include_modern and is_modern(meta):
            rejected.append(f"wspolczesna magia bez dawnego zrodla: {text[:60]}")
            continue
        mf = MagicFact("magia", norm_part(f.get("czesc")), text, quote[:300], meta)
        if sweep:
            mf.plant, mf.latin = _clean(f.get("roslina"), 80), _clean(f.get("lacina"), 80)
            if not mf.plant and not mf.latin:
                rejected.append(f"bez nazwy rosliny: {text[:60]}")
                continue
        facts.append(mf)
    return facts, parse_leads(data.get("tropy")), rejected


def parse_leads(raw) -> list[dict]:
    out, seen = [], set()
    for t in (raw or [])[:MAX_LEADS] if isinstance(raw, list) else []:
        if not isinstance(t, dict):
            continue
        term = re.sub(r"\s+", " ", str(t.get("termin") or "")).strip().strip('"„”')
        lang = str(t.get("jezyk") or "").strip().lower()
        typ = str(t.get("typ") or "").strip().lower()
        if not (2 <= len(term) <= 60) or not re.fullmatch(r"[a-z]{2,3}", lang) or re.search(r"[<>{}]", term):
            continue
        typ = typ if typ in LEAD_TYPES else "praktyka"
        key = (term.lower(), lang)
        if key in seen:
            continue
        seen.add(key)
        out.append({"termin": term, "jezyk": lang, "typ": typ, "opis": _clean(t.get("opis"), 200)})
    return out


def lead_fingerprint(plant_id: str | None, term: str, lang: str) -> str:
    return hashlib.sha1(f"{plant_id or '*'}|{norm_text(term)}|{lang}".encode()).hexdigest()[:16]


# ------------------------------------------------------------------ generator zapytan
def name_variants(names: dict[str, str], folk: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Nazwy rosliny w kazdym jezyku: Wikidata/Wikipedia + nazwy ludowe z tropow (najpierw oficjalna)."""
    out: dict[str, list[str]] = {}
    for lang, name in names.items():
        if name:
            out.setdefault(lang, []).append(str(name))
    for lang, items in (folk or {}).items():
        for n in items:
            if n and n not in out.get(lang, []):
                out.setdefault(lang, []).append(n)
    return out


def plant_queries(names: dict[str, str], langs: list[str], done: set[str], leads: list = (),
                  folk: dict[str, list[str]] | None = None, used_domains: set[str] | None = None,
                  limit: int = 8, archive_every: int = 3) -> list[MQuery]:
    """Zapytania dla jednej rosliny: najpierw tropy (azymuty), potem pojecia z tezaurusa w jezykach, w ktorych
    temat jest najlepiej opisany (rotacja - bez par pojecie+jezyk zadanych niedawno), co kilka - glebokie archiwa.
    `leads` - otwarte tropy rosliny (obiekty z polami id, term, language, kind, rodzaj, depth)."""
    variants = name_variants(names, folk)
    latin = names.get("la")
    trop_q: list[MQuery] = []
    for ld in leads:
        q = lead_query(ld, variants, latin)
        if q and q.query not in done:
            trop_q.append(q)
    concept_q: list[MQuery] = []
    for key in concept_order():
        c = CONCEPT_BY_KEY[key]
        order = [x for x in c.langs if x in langs] + [x for x in langs if x not in c.langs]
        for lang in order:
            words = c.terms.get(lang)
            name = (variants.get(lang) or [None])[0] or (latin if lang in ("la", "en") else None)
            if not words or not name:
                continue
            qkey = f"magia:{c.key}:{lang}"
            q = f"\"{name}\" {words}"
            if qkey in done or q in done:
                continue
            concept_q.append(MQuery("koncept", lang, q, qkey, c.rodzaj))
            break                                      # jedno pojecie = jeden jezyk na przebieg
    arch_q = archive_queries(variants, latin, done, used_domains or set())
    out: list[MQuery] = []
    ti = ci = ai = 0
    while len(out) < limit and (ti < len(trop_q) or ci < len(concept_q) or ai < len(arch_q)):
        n = len(out)
        if n % archive_every == archive_every - 1 and ai < len(arch_q):
            out.append(arch_q[ai]); ai += 1
        elif n % 2 == 0 and ti < len(trop_q):
            out.append(trop_q[ti]); ti += 1
        elif ci < len(concept_q):
            out.append(concept_q[ci]); ci += 1
        elif ti < len(trop_q):
            out.append(trop_q[ti]); ti += 1
        elif ai < len(arch_q):
            out.append(arch_q[ai]); ai += 1
    return out


def lead_query(ld, variants: dict[str, list[str]], latin: str | None) -> MQuery | None:
    """Trop -> zapytanie w jezyku tropu. nazwa: "<nazwa ludowa>" + pojecie rodzaju; praktyka/analogia: "<termin>"
    + nazwa rosliny w tym jezyku (albo rodzaj lacinski)."""
    term, lang, kind = getattr(ld, "term", None), getattr(ld, "language", None), getattr(ld, "kind", None)
    if not term or not lang:
        return None
    rodzaj = getattr(ld, "rodzaj", None)
    if kind == "nazwa":
        c = next((c for c in CONCEPTS if c.rodzaj == (rodzaj or "zabobony") and lang in c.terms), None) \
            or CONCEPT_BY_KEY["zabobony"]
        words = c.terms.get(lang) or c.terms.get("en", "")
        q = f"\"{term}\" {words}".strip()
    else:
        name = (variants.get(lang) or [None])[0] or (latin.split()[0] if latin else None)
        q = f"\"{term}\"" + (f" \"{name}\"" if name and name.lower() not in term.lower() else "")
    return MQuery("trop", lang, q, f"magia:trop:{getattr(ld, 'id', 0)}", rodzaj, getattr(ld, "id", None),
                  int(getattr(ld, "depth", 0) or 0))


def archive_queries(variants: dict[str, list[str]], latin: str | None, done: set[str],
                    used_domains: set[str]) -> list[MQuery]:
    """site: do bibliotek cyfrowych i dawnych ksiag - w jezyku archiwum (lacina: nazwa lacinska)."""
    out = []
    for src in DEEP_SOURCES:
        key = f"magia:arch:{src.domain}"
        if key in done:
            continue
        if src.lang == "la":
            name = latin.split()[0] if latin else None
        else:
            name = (variants.get(src.lang) or [None])[0] or (latin if src.lang == "en" else None)
        if not name:
            continue
        out.append(MQuery("archiwum", src.lang, f"site:{src.domain} \"{name}\" {src.word}", key))
    # najpierw archiwa, z ktorych ta roslina jeszcze nic nie ma
    return sorted(out, key=lambda q: q.key.split(":", 2)[2] in used_domains)


def sweep_queries(done: set[str], limit: int = 2) -> list[MQuery]:
    from app.knowledge.magic_lexicon import SWEEPS
    out = []
    for lang, q, rodzaj in SWEEPS:
        key = f"magia:przeglad:{norm_text(q)[:60]}"
        if key in done or q in done:
            continue
        out.append(MQuery("przeglad", lang, q, key, rodzaj))
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ strony
def page_relevant(text: str, name_keys: list[str], min_magic: int = 2) -> tuple[bool, str]:
    """Strona mowi o tej roslinie (jedna z nazw) i o magii (>= min_magic roznych rdzeni). (czy, powod)."""
    low = (text or "").lower()
    if name_keys and not any(k in low for k in name_keys):
        return False, "brak nazwy rosliny"
    score = magic_score(low)
    if score < min_magic:
        return False, f"za malo o magii ({score})"
    return True, f"magia: {score}"


def focus_text(text: str, name_keys: list[str], window: int = 1600, limit: int = 10000) -> str:
    """Dlugie teksty (stare ksiegi, archiwa): fragmenty wokol nazw rosliny zamiast poczatku strony."""
    if len(text) <= limit:
        return text
    low = text.lower()
    spans: list[list[int]] = []
    for k in name_keys:
        start = 0
        while True:
            i = low.find(k, start)
            if i < 0:
                break
            spans.append([max(0, i - window // 2), min(len(text), i + window // 2)])
            start = i + len(k)
    if not spans:
        return text[:limit]
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, total = [], 0
    for a, b in merged:
        if total >= limit:
            break
        piece = text[a:min(b, a + limit - total)]
        out.append(piece)
        total += len(piece)
    return "\n[...]\n".join(out)


def lang_stems(lang: str) -> tuple[str, ...]:
    return STEMS.get(lang, ())


# ------------------------------------------------------------------ rozpoznawanie roslin ("od praktyki")
class PlantIndex:
    """Nazwa rosliny z tekstu (dowolny jezyk) albo nazwa lacinska -> id rosliny Kwiatownika."""

    def __init__(self, plants: list[tuple[str, str | None, str | None, dict]]):
        # plants: (id, nazwa_pl, nazwa_lat, {jezyk: nazwa})
        self.by_latin: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        for pid, pl, lat, names in plants:
            if lat:
                self.by_latin.setdefault(" ".join(lat.lower().split()[:2]), pid)
            for n in [pl, *(names or {}).values()]:
                if n and len(str(n)) >= 3:
                    self.by_name.setdefault(norm_text(str(n)), pid)

    def find(self, name: str | None, latin: str | None) -> str | None:
        if latin:
            key = " ".join(latin.lower().replace("×", "").split()[:2])
            if key in self.by_latin:
                return self.by_latin[key]
        if name:
            n = norm_text(name)
            if n in self.by_name:
                return self.by_name[n]
            if latin is None and len(n) >= 6:          # odmiana: "dziurawca" -> "dziurawiec"
                for k, pid in self.by_name.items():
                    if len(k) >= 6 and (k[:-2] == n[:-2] or k.startswith(n[:-2])) and abs(len(k) - len(n)) <= 3:
                        return pid
        return None


# ------------------------------------------------------------------ priorytet tropow
def lead_order(leads: list, covered_langs: set[str]) -> list:
    """Tropy do sprawdzenia: najplytsze najpierw, jezyki bez wiedzy o magii tej rosliny wyzej, potem te,
    ktorych "rodzic" dal najwiecej informacji, potem najstarsze."""
    return sorted(leads, key=lambda ld: (int(getattr(ld, "depth", 0) or 0),
                                         getattr(ld, "language", "") in covered_langs,
                                         -int(getattr(ld, "parent_yield", 0) or 0),
                                         getattr(ld, "id", 0) or 0))
