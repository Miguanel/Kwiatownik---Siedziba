"""Kontrole kodem tlumaczenia maszynowego (bez LLM).

Kazdy fragment (tytul, skladnik, krok...) porownywany jest z oryginalem:
- liczby: tlumacz nie moze zgubic ani dopisac zadnej liczby (ilosci, czasy, temperatury),
- obcy alfabet (cyrylica, greka, CJK...) w polskim tekscie = fragment nieprzetlumaczony,
- zdanie identyczne z oryginalem (4+ slowa) = nieprzetlumaczone,
- zapetlenia ("i i i i"), powtorzone zdania, tekst uciety albo kilka razy dluzszy,
- rosliny: gdy oryginal wymienia rosline z katalogu (nazwa lacinska albo nazwa w danym jezyku z Wikidata),
  polskie tlumaczenie musi ja wymieniac (np. 'chamomile' -> 'rumianek', a nie 'plytki').

Blad = przepis nie idzie dalej (status "blad", powod "kontrola: ..."); uwaga = zapis w rekord["siedziba"]["kontrola"].
"""
import re
from dataclasses import dataclass, field

from app.knowledge.sections import norm_text

FOREIGN_SCRIPT = re.compile(
    "[Ͱ-ϿЀ-ӿ԰-֏֐-׿؀-ۿऀ-ॿႠ-ჿ"
    "぀-ヿ㐀-鿿가-힯]")
_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
NUM_RE = re.compile(r"(\d+)\s*/\s*(\d+)|(\d+(?:[.,]\d+)?)|([½¼¾⅓⅔⅛])")
PL_NUMBER_WORDS = {
    0.5: ("pół", "połow"), 1: ("jed", "raz"), 2: ("dwa", "dwie", "dwóch", "dwu", "oba", "obie"),
    3: ("trzy", "trzech", "trzema"), 4: ("czter",), 5: ("pięć", "pięciu"), 6: ("sześć", "sześciu"),
    7: ("siedem", "siedmiu"), 8: ("osiem", "ośmiu"), 9: ("dziewięć", "dziewięciu"), 10: ("dziesięć", "dziesięciu"),
    12: ("tuzin", "dwanaście"), 15: ("kwadrans",), 30: ("pół godziny",), 60: ("godzin",),
}
MIN_LEN_RATIO, WARN_LOW, WARN_HIGH = 0.3, 0.5, 2.2


@dataclass
class Problem:
    level: str            # "blad" albo "uwaga"
    where: str            # np. "skladnik 2", "krok 3", "tytul"
    what: str

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "blad"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "uwaga"]

    def ok(self) -> bool:
        return not self.errors

    def summary(self, limit: int = 3) -> str:
        errs = self.errors
        more = f" (+{len(errs) - limit})" if len(errs) > limit else ""
        return "kontrola: " + "; ".join(str(p) for p in errs[:limit]) + more


# ------------------------------------------------------------ liczby
# liczby rzymskie stulecia (tylko wielkie litery - "i w lesie" to nie 1): "XIX w.", "XIXe siècle", "XIX. Jh."
ROMAN_RE = re.compile(r"\b([IVXL]{2,7}|[IVX](?=\.?\s*(?:w\.|wiek|Jh|Jahrh|si[eè]cle|cent)))(?:e|er|\.)?\s*"
                      r"(?=[wW]\.|[wW]\b|[wW]iek|[sS]tul|[cC]ent|J[hH]|[jJ]ahrh|si[eè]cle|sec|век|ст)")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def _roman(s: str) -> int | None:
    s = s.upper()
    if not s or any(c not in _ROMAN for c in s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b in _ROMAN and _ROMAN[b] > v else v
    return total if 0 < total <= 30 else None

def numbers(text: str) -> set[float]:
    """Wszystkie liczby tekstu (ulamki, 1,5 / 1.5, 1 1/2 -> tez 1.5, 1.000 -> 1 i 1000)."""
    vals: list[tuple[float, int, int]] = []
    for m in NUM_RE.finditer(text or ""):
        if m.group(1):
            den = int(m.group(2))
            if den:
                vals.append((int(m.group(1)) / den, m.start(), m.end()))
        elif m.group(3):
            raw = m.group(3)
            vals.append((float(raw.replace(",", ".")), m.start(), m.end()))
            if re.fullmatch(r"\d{1,3}[.,]\d{3}", raw):                # 1.000 (de) albo 1,000 (en) = tysiac
                vals.append((float(raw.replace(",", "").replace(".", "")), m.start(), m.end()))
        else:
            vals.append((_FRAC[m.group(4)], m.start(), m.end()))
    for m in ROMAN_RE.finditer(text or ""):                         # "XIX w.", "XIXe siècle" = 19
        v = _roman(m.group(1))
        if v:
            vals.append((float(v), m.start(), m.end()))
    out = {round(v, 3) for v, _, _ in vals}
    for (a, _, ea), (b, sb, _) in zip(vals, vals[1:]):              # "1 1/2", "1½" -> 1.5
        if a == int(a) and 0 < b < 1 and sb - ea <= 1:
            out.add(round(a + b, 3))
    return out


def _said_in_words(value: float, text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in PL_NUMBER_WORDS.get(value, ()))


def number_problems(src: str, out: str) -> list[str]:
    s, o = numbers(src), numbers(out)

    def near(x: float, pool: set[float]) -> bool:
        return any(abs(x - y) < 0.011 for y in pool)

    def part_of_sum(x: float, mine: set[float], other: set[float]) -> bool:
        """'1 1/2' po jednej stronie i '1,5' po drugiej: skladowe 1 i 0,5 sa w porzadku."""
        return any(near(x + y, other) for y in mine if y != x and min(x, y) < 1)

    added = sorted(x for x in o if not near(x, s) and not part_of_sum(x, o, s))
    lost = sorted(x for x in s if not near(x, o) and not part_of_sum(x, s, o) and not _said_in_words(x, out))
    msgs = []
    if added:
        msgs.append("dopisana liczba " + ", ".join(_fmt(x) for x in added[:3]))
    if lost:
        msgs.append("zgubiona liczba " + ", ".join(_fmt(x) for x in lost[:3]))
    return msgs


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x).replace(".", ",")


# ------------------------------------------------------------ jakosc fragmentu
def _loop(out: str) -> bool:
    words = out.lower().split()
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    sents = [s.strip().lower() for s in re.split(r"[.!?]+", out) if len(s.strip()) > 8]
    return any(sents.count(s) >= 3 for s in set(sents))


def check_segment(where: str, src: str, out: str, lang: str | None) -> list[Problem]:
    src, out = (src or "").strip(), (out or "").strip()
    if not src:
        return []
    probs: list[Problem] = []
    foreign = (lang or "")[:2].lower() not in ("", "pl")
    if not out:
        return [Problem("blad", where, "puste tlumaczenie")]
    if foreign and FOREIGN_SCRIPT.search(out):
        probs.append(Problem("blad", where, "obcy alfabet - fragment nieprzetlumaczony"))
    elif foreign and len(src.split()) >= 4 and norm_text(src) == norm_text(out):
        probs.append(Problem("blad", where, "fragment nieprzetlumaczony"))
    if _loop(out):
        probs.append(Problem("blad", where, "zapetlenie tlumacza"))
    for msg in number_problems(src, out):
        probs.append(Problem("blad", where, msg))
    if len(src) >= 40:
        ratio = len(out) / len(src)
        if ratio < MIN_LEN_RATIO:
            probs.append(Problem("blad", where, f"tlumaczenie uciete ({len(out)} z {len(src)} znakow)"))
        elif ratio > 3:
            probs.append(Problem("blad", where, "tlumaczenie kilka razy dluzsze od oryginalu"))
        elif ratio < WARN_LOW or ratio > WARN_HIGH:
            probs.append(Problem("uwaga", where, f"nietypowa dlugosc tlumaczenia ({ratio:.1f}x)"))
    return probs


# ------------------------------------------------------------ rosliny (nazwy w jezykach z Wikidata)
def _word_root(w: str) -> str:
    return w if len(w) <= 5 else w[:max(5, len(w) - 2)]


# jezyki, w ktorych rzeczownik stoi na poczatku nazwy rosliny (ромашка лікарська, plantain lancéolé);
# w pozostalych na koncu (Echte Kamille, pot marigold, paprastoji ramunė)
HEAD_FIRST = {"uk", "ru", "be", "bg", "mk", "sr", "hr", "bs", "sl", "cs", "sk", "pl", "fr", "it", "es", "pt", "ro",
              "ca", "gl"}


AMBIGUOUS = {"plantain", "lime", "rocket", "dock", "orange", "poppy seed", "linden tea"}   # wieloznaczne slowa


def _word_level(code: str, word: str) -> str:
    """Pojedyncze slowo-nazwa: pewne ("blad") od 5 liter, po angielsku od 7 (lime, sage, elder - niepewne)."""
    return "blad" if len(word) >= 5 and (code != "en" or len(word) >= 7) and word not in AMBIGUOUS else "uwaga"


def _names(r: dict, lang: str) -> list[tuple[str, bool]]:
    """[(nazwa, kanoniczna?)] rosliny w jezyku: etykieta i tytul artykulu (kanoniczne) + aliasy Wikidata."""
    out = [((r.get("names") or {}).get(lang), True), ((r.get("titles") or {}).get(lang), True)]
    out += [(a, False) for a in (r.get("aliases") or {}).get(lang, [])]
    seen, res = set(), []
    for name, canon in out:
        name = re.sub(r"\s*\(.*?\)\s*", " ", name or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            res.append((name, canon))
    return res


class ForeignNames:
    """Nazwy roslin katalogu w obcych jezykach (Wikidata: etykiety, tytuly artykulow, aliasy) + lacinskie.

    Dopasowanie ostrozne: cala nazwa (slowa 3+ litery jako poczatki slow tekstu) albo sam rzeczownik nazwy
    (5+ liter: "Echte Kamille" -> Kamille pasuje do "Kamillenblüten"). Poziom: "blad" - roslina na pewno jest
    w oryginale (zgubiona w tlumaczeniu = przepis zatrzymany), "uwaga" - slowo moze znaczyc cos innego
    (angielskie pojedyncze slowa: lime, sage...). Nazwa wspolna dla dwoch roslin - pomijana."""

    def __init__(self, rows: dict[str, dict]):
        # rows: {plant_id: {"lat", "pl", "names": {jezyk: nazwa}, "aliases": {jezyk: [..]}, "titles": {jezyk: tytul}}}
        self.latin: list[tuple[str, list[str]]] = []
        self.polish: dict[str, list[str]] = {}
        entries: dict[str, dict[tuple, tuple[set, str]]] = {}
        for pid, r in rows.items():
            lat = norm_text(r.get("lat") or "").split()
            if len(lat) >= 2:
                self.latin.append((pid, lat[:2]))
            pl_main = norm_text(re.sub(r"\(.*?\)", " ", r.get("pl") or "")).split()
            pl_alt = norm_text(" ".join(re.findall(r"\((.*?)\)", r.get("pl") or ""))).split()
            self.polish[pid] = [_word_root(w) for w in pl_main if len(w) >= 4 and w not in _PL_ADJ] + \
                               [_word_root(w) for w in pl_alt if len(w) >= 4]
            langs = set(r.get("names") or {}) | set(r.get("aliases") or {}) | set(r.get("titles") or {})
            for lang in langs:
                code = lang[:2].lower()
                if code == "pl":
                    continue
                for name, _canon in _names(r, lang):
                    words = norm_text(name).split()
                    if not words or words[:2] == lat[:2] or name.lower() in AMBIGUOUS:
                        continue
                    full = tuple(w for w in words if len(w) >= 3)
                    if not full:
                        continue
                    single = len(full) == 1
                    self._add(entries, code, full, pid, _word_level(code, full[0]) if single else "blad")
                    if not single:
                        head = words[0] if code in HEAD_FIRST else words[-1]
                        if len(head) >= 5 and head not in AMBIGUOUS:
                            self._add(entries, code, (head,), pid, _word_level(code, head))
        # nazwa wspolna kilku roslinom - nie wiadomo ktora, pomijamy
        self.by_lang: dict[str, list[tuple[str, tuple, str]]] = {
            code: [(next(iter(pids)), words, level) for words, (pids, level) in e.items() if len(pids) == 1]
            for code, e in entries.items()}

    @staticmethod
    def _add(entries: dict, code: str, words: tuple, pid: str, level: str) -> None:
        pids, old = entries.setdefault(code, {}).get(words, (set(), "uwaga"))
        pids.add(pid)
        entries[code][words] = (pids, "blad" if "blad" in (old, level) else level)

    def __bool__(self) -> bool:
        return bool(self.latin or self.by_lang)

    def found_levels(self, text: str, lang: str | None) -> dict[str, str]:
        t = " " + norm_text(text) + " "
        hits = {pid: "blad" for pid, (g, s) in self.latin if f" {g} {s}" in t}
        for pid, words, level in self.by_lang.get((lang or "")[:2].lower(), []):
            if hits.get(pid) != "blad" and all(f" {_word_root(w)}" in t for w in words):
                hits[pid] = level if pid not in hits else ("blad" if "blad" in (level, hits[pid]) else level)
        return hits

    def found(self, text: str, lang: str | None) -> set[str]:
        """Rosliny na pewno wymienione w tekscie (poziom "blad")."""
        return {pid for pid, lv in self.found_levels(text, lang).items() if lv == "blad"}

    def mentioned_pl(self, pid: str, text: str) -> bool:
        t = " " + norm_text(text) + " "
        if any(f" {g} {s}" in t for p, (g, s) in self.latin if p == pid):
            return True
        return any(f" {root}" in t for root in self.polish.get(pid, []))


def glossary_entries(rows: dict[str, dict], lang: str) -> dict[str, str]:
    """Slownik dla tlumacza (DeepL glossary): nazwa rosliny w jezyku zrodla -> polska nazwa z katalogu.
    Tylko nazwy jednoznaczne; pojedyncze slowa od 5 liter (angielskie od 7, bez wieloznacznych)."""
    code = (lang or "")[:2].lower()
    out: dict[str, str] = {}
    clash: set[str] = set()
    for pid, r in rows.items():
        target = re.sub(r"\s*\(.*?\)\s*", " ", r.get("pl") or "").strip().lower()
        lat = (r.get("lat") or "").strip().lower()
        if not target:
            continue
        for name, _canon in _names(r, code):
            if "\t" in name or name.lower() == lat or name.lower() in AMBIGUOUS:
                continue
            words = name.split()
            if len(words) == 1 and (len(name) < 5 or (code == "en" and len(name) < 7)):
                continue
            if len(name) > 80:
                continue
            variants = {name: target} if code == "de" else {
                name.lower(): target, name[:1].upper() + name[1:].lower(): target[:1].upper() + target[1:]}
            for term, tgt in variants.items():
                if term in out and out[term] != tgt:
                    clash.add(term)
                out.setdefault(term, tgt)
    return {k: v for k, v in out.items() if k not in clash}


_PL_ADJ = {"zwyczajny", "zwyczajna", "pospolity", "pospolita", "lekarski", "lekarska", "czarny", "czarna", "bialy",
           "biala", "polny", "polna", "wlasciwy", "wlasciwa", "wonna", "wonny", "zwyczajne", "lancetowata",
           "pieprzowa", "zielona", "czerwona", "czerwony", "zolta", "zolty", "drobny", "drobna", "wielki", "wielka",
           "wiosenny", "wiosenna", "ogrodowa", "ogrodowy", "dzika", "dziki"}


def plant_problems(pairs: list[tuple[str, str, str]], lang: str | None, names: ForeignNames | None,
                   plants: dict[str, str]) -> list[Problem]:
    """pairs: (gdzie, oryginal, tlumaczenie). Roslina wymieniona w oryginale musi byc w tlumaczeniu."""
    if not names:
        return []
    probs = []
    for where, src, out in pairs:
        for pid, level in sorted(names.found_levels(src, lang).items()):
            if not names.mentioned_pl(pid, out):
                label = (plants.get(pid) or pid).split(" (")[0]
                probs.append(Problem(level, where, f"roslina '{label}' z oryginalu zgubiona w tlumaczeniu "
                                                   f"('{src[:40]}' -> '{out[:40]}')"))
    return probs


def check_translation(segments: list[tuple[str, str, str]], lang: str | None, names: ForeignNames | None = None,
                      plants: dict[str, str] | None = None) -> Report:
    """segments: (gdzie, oryginal, tlumaczenie) - wszystkie fragmenty jednego przepisu."""
    rep = Report()
    for where, src, out in segments:
        rep.problems += check_segment(where, src, out, lang)
    rep.problems += plant_problems(segments, lang, names, plants or {})
    return rep


_ROWS: tuple[float, int, dict] | None = None
ROWS_TTL_S = 600


def plant_rows(db_engine=None) -> dict[str, dict]:
    """Nazwy roslin z bazy Siedziby (Wikidata: names_json, aliases_json, wiki_titles_json), pamietane 10 min.
    Blad bazy -> pusty slownik (kontrola roslin i slownik tlumacza wylaczone, tlumaczenie dziala dalej)."""
    import json
    import time

    from sqlmodel import Session, select

    from app.models import Plant
    global _ROWS
    if db_engine is None:
        from app.db import engine as db_engine
    now = time.time()
    if _ROWS and _ROWS[1] == id(db_engine) and now - _ROWS[0] < ROWS_TTL_S:
        return _ROWS[2]

    def js(v):
        try:
            return json.loads(v) if v else {}
        except ValueError:
            return {}
    try:
        with Session(db_engine) as s:
            rows = {p.id: {"lat": p.nazwa_lat, "pl": p.nazwa_pl, "names": js(p.names_json),
                           "aliases": js(p.aliases_json), "titles": js(p.wiki_titles_json)}
                    for p in s.exec(select(Plant)).all()}
    except Exception:  # noqa: BLE001 - kontrola nigdy nie zatrzymuje tlumaczenia z powodu bazy
        return {}
    _ROWS = (now, id(db_engine), rows)
    return rows


def forget_plant_rows() -> None:
    global _ROWS
    _ROWS = None


def load_foreign_names(db_engine=None) -> ForeignNames:
    return ForeignNames(plant_rows(db_engine))
