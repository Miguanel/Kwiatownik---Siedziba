"""LLM wskazuje w tekscie zrodla (Wikipedia albo inna strona) fragmenty z informacjami o roslinie.

Od 2026-11-03 LLM NIE pisze tekstu informacji: zwraca tylko sekcje, czesc rosliny i "cytat" (doslowny fragment
zrodla). Kod sprawdza cytat i bierze dokladny fragment zrodla (app/knowledge/quotes.py), a tekst po polsku
to tlumaczenie maszynowe tego fragmentu (DeepL / NLLB) po kontroli - albo sam fragment, gdy zrodlo jest po polsku.
"""
import json
import re
from dataclasses import dataclass

from app.knowledge.sections import PARTS, SECTIONS, norm_part, norm_text

MAX_FACTS = 14

SYSTEM_PROMPT = """Jestes redaktorem polskiego zielnika "Kwiatownik". W podanym tekscie zrodla WSKAZUJESZ fragmenty \
z konkretnymi, ciekawymi i sprawdzalnymi informacjami o JEDNEJ roslinie. NIE piszesz wlasnych zdan i NIE tlumaczysz.
ZASADY:
- Tylko fragmenty, ktore dotycza TEJ rosliny (nie innych gatunkow, nie calego rodzaju).
- "cytat": 1-2 PELNE zdania skopiowane z tekstu zrodla znak w znak, w oryginalnym jezyku (takze po chinsku czy
  japonsku), max 350 znakow. Jedna informacja = jeden fragment; nie lacz zdan z roznych miejsc tekstu.
- Zrodla chinskie i japonskie: natura (性/寒/温), smak (味), meridiany (归经/帰経) i wskazania -> sekcja
  "medycyna_wschodu".
- Pomijaj fragmenty z dawkowaniem albo poleceniami leczniczymi, oczywistosci (np. "jest rosliną"), taksonomie
  i dane bez znaczenia dla czytelnika.
- Szczegolnie cenne: dawne i ludowe zastosowania, wierzenia i obrzedy, nazwy ludowe, barwienie, toksycznosc.
- "czesc": czesc rosliny, ktorej dotyczy fragment, TYLKO z listy: """ + ", ".join(PARTS) + """ - albo null
  (cala roslina, nazwa, toksycznosc itp. -> null).
Zwroc TYLKO JSON: {"fakty": [{"sekcja": "...", "czesc": "kwiaty lub null", "cytat": "..."}]}
Dozwolone sekcje:
""" + "\n".join(f"- {k}: {v}" for k, v in SECTIONS.items())


@dataclass
class Fact:
    section: str
    part: str | None
    text: str                     # po polsku: tlumaczenie maszynowe cytatu (albo cytat ze zrodla po polsku)
    quote: str                    # doslowny fragment zrodla
    translated_by: str | None = None


def build_prompt(plant_pl: str, latin: str | None, source_name: str, lang: str, text: str,
                 known: list[str], focus: list[str] | None = None) -> str:
    known_txt = "\n".join(f"- {k}" for k in known[:40])
    focus_txt = "\n".join(f"- {f}" for f in (focus or [])[:8])
    return (f"ROSLINA: {plant_pl}" + (f" ({latin})" if latin else "") +
            f"\nZRODLO: {source_name} (jezyk: {lang})\n" +
            (f"\nSZCZEGOLNIE SZUKAMY (tego brakuje w zielniku - jesli tekst o tym mowi, wypisz to):\n{focus_txt}\n"
             if focus else "") +
            (f"\nJUZ WIEMY (nie powtarzaj):\n{known_txt}\n" if known else "") +
            f"\nTEKST ZRODLA:\n{text}")


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def is_cjk(text: str) -> bool:
    """Tekst glownie chinski/japonski (znaki CJK, kana)."""
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(1 for c in chars if _CJK.match(c)) >= 0.3 * len(chars)


def quote_in_text(quote: str, text: str) -> bool:
    """Cytat musi naprawde byc w zrodle (dokladnie albo prawie - LLM czasem zmienia interpunkcje)."""
    q, t = norm_text(quote), norm_text(text)
    if is_cjk(q):                                   # chinski / japonski: bez spacji -> n-gramy znakow
        qc, tc = q.replace(" ", ""), t.replace(" ", "")
        if len(qc) < 6:
            return False
        if qc in tc:
            return True
        grams = [qc[i:i + 4] for i in range(len(qc) - 3)]
        return bool(grams) and sum(1 for g in grams if g in tc) >= 0.8 * len(grams)
    if len(q) < 12:
        return False
    if q in t:
        return True
    words = q.split()
    if len(words) < 4:
        return False
    # co najmniej 80% kolejnych trojek slow cytatu wystepuje w tekscie
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return sum(1 for g in grams if g in t) >= 0.8 * len(grams)


def numbers_supported(fact_text: str, quote: str) -> bool:
    """Liczby w informacji (lata, ilosci, wysokosci) musza wystepowac w cytacie - chroni przed zmyslaniem."""
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", fact_text))
    return all(n.replace(",", ".") in quote.replace(",", ".") for n in nums)


def parse_facts(raw: str, source_text: str) -> tuple[list[Fact], list[str]]:
    """Odpowiedz LLM -> (informacje z doslownym cytatem ze zrodla, jeszcze bez polskiego tekstu; powody odrzucen)."""
    from app.knowledge.quotes import snap_quote
    data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
    facts, rejected = [], []
    if isinstance(data, list):            # sama lista informacji zamiast {"fakty": [...]}
        data = {"fakty": data}
    if not isinstance(data, dict) or not isinstance(data.get("fakty") or [], list):
        return [], ["odpowiedz modelu w zlym formacie"]
    seen: set[str] = set()
    for f in (data.get("fakty") or [])[:MAX_FACTS]:
        if not isinstance(f, dict):
            continue
        quote = re.sub(r"\s+", " ", str(f.get("cytat") or "")).strip()
        section = str(f.get("sekcja") or "").strip()
        section = section if section in SECTIONS else "ciekawostki"
        span = snap_quote(quote, source_text)
        if not span:
            rejected.append(f"cytatu nie ma w zrodle: {quote[:60]}")
            continue
        if len(span) < 20:
            rejected.append(f"za krotki fragment: {span[:60]}")
            continue
        key = norm_text(span)
        if key in seen:
            continue
        seen.add(key)
        facts.append(Fact(section, norm_part(f.get("czesc")), "", span))
    return facts, rejected


async def extract_facts(llm, plant_pl: str, latin: str | None, source_name: str, lang: str, text: str,
                        known: list[str], focus: list[str] | None = None) -> tuple[list[Fact], list[str], str]:
    res = await llm.complete(build_prompt(plant_pl, latin, source_name, lang, text, known, focus),
                             system=SYSTEM_PROMPT, json_mode=True)
    facts, rejected = parse_facts(res.text, text)
    return facts, rejected, f"{res.provider}/{res.model}"
