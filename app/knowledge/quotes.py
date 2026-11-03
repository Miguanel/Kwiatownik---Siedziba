"""Wiedza o roslinach bez tekstow pisanych przez LLM (decyzja 2026-11-03).

LLM tylko WSKAZUJE w tekscie zrodla fragment (cytat) z informacja i przypisuje mu sekcje / czesc rosliny.
Dalej wszystko robi kod:
1. snap_quote - cytat musi byc w zrodle; zamiast slow modelu brany jest DOKLADNY fragment zrodla, poszerzony
   do pelnych zdan (najwyzej MAX_QUOTE znakow),
2. tlumacz maszynowy (DeepL / NLLB, cel "wiedza" - z budzetem limitu DeepL) tlumaczy cytat na polski;
   zrodlo po polsku = sam cytat,
3. kontrola tlumaczenia (app/pipeline/checks.py): liczby, obcy alfabet, zapetlenia, dlugosc,
4. brak tlumacza albo wyczerpany budzet -> informacja czeka ("untranslated") i jest tlumaczona pozniej.
"""
import re
from difflib import SequenceMatcher

from app.knowledge.extract import is_cjk, quote_in_text
from app.knowledge.sections import norm_text

MAX_QUOTE = 450                 # znakow fragmentu zrodla (po poszerzeniu do pelnych zdan)
MIN_COVER = 0.7                 # tyle slow cytatu modelu musi sie znalezc w jednym miejscu zrodla
SENT_END = re.compile(r"(?<=[.!?。！？])\s+|\n+")
POLISH = ("pl",)


def _sentence_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    """Poszerza [start, end) do pelnych zdan (najwyzej 250 znakow wstecz i 300 naprzod)."""
    back = source[max(0, start - 250):start]
    cut = None
    for m in SENT_END.finditer(back):
        cut = m.end()
    new_start = max(0, start - 250) + cut if cut is not None else (0 if start <= 250 else start)
    fwd = source[end:end + 300]
    m = re.search(r"[.!?。！？](?=\s|$)|\n", fwd)
    new_end = end + (m.end() if m else 0) if m else end
    return new_start, new_end


def snap_quote(quote: str, source: str, max_len: int = MAX_QUOTE) -> str | None:
    """Dokladny fragment zrodla odpowiadajacy cytatowi modelu (pelne zdania). None = cytatu nie ma w zrodle."""
    quote = re.sub(r"\s+", " ", quote or "").strip()
    if not quote or not source:
        return None
    if is_cjk(quote):
        if not quote_in_text(quote, source):
            return None
        i = source.find(quote)
        if i < 0:
            return quote[:max_len]                     # zgodny w 80% 4-gramow - zostaje cytat modelu
        s, e = _sentence_bounds(source, i, i + len(quote))
        span = source[s:e].strip()
        return span if len(span) <= max_len else quote[:max_len]
    q = norm_text(quote).split()
    toks = [(norm_text(m.group()), m.start(), m.end()) for m in re.finditer(r"\w+", source)]
    toks = [t for t in toks if t[0]]
    if len(q) < 3 or not toks:
        return None
    sm = SequenceMatcher(None, q, [t[0] for t in toks], autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    if not blocks:
        return None
    main = max(blocks, key=lambda b: b.size)
    lo, hi = main.b - len(q) * 2, main.b + main.size + len(q) * 2
    near = [b for b in blocks if lo <= b.b <= hi and b.size >= (2 if len(q) >= 6 else 1)] or [main]
    covered = sum(b.size for b in near)
    if covered < MIN_COVER * len(q):
        return None
    b0 = min(b.b for b in near)
    b1 = max(b.b + b.size for b in near) - 1
    start, end = toks[b0][1], toks[b1][2]
    if end - start > max_len * 1.5:
        return None
    s, e = _sentence_bounds(source, start, end)
    span = re.sub(r"\s+", " ", source[s:e]).strip()
    if len(span) > max_len:
        span = re.sub(r"\s+", " ", source[start:end]).strip()
    return span[:max_len] if span else None


def translate_quotes(mt, items: list[tuple[str, str]], purpose: str = "wiedza") -> list[tuple[str | None, str, str]]:
    """[(jezyk, cytat)] -> [(tlumaczenie albo None, powod, tlumacz)]. Powod "brak tlumacza: ..." = sprobuj pozniej,
    "kontrola: ..." = tlumaczenie zle (odrzucone)."""
    from app.mt.nllb import MTUnavailable
    from app.pipeline.checks import check_segment
    out: list[tuple[str | None, str, str]] = [(None, "", "")] * len(items)
    by_lang: dict[str, list[int]] = {}
    for i, (lang, text) in enumerate(items):
        code = (lang or "")[:2].lower()
        if code in POLISH:
            out[i] = (text, "", "cytat")
        elif not code:
            out[i] = (None, "kontrola: nieznany jezyk zrodla", "")
        else:
            by_lang.setdefault(code, []).append(i)
    for code, idx in by_lang.items():
        if mt is None:
            for i in idx:
                out[i] = (None, "brak tlumacza: tlumacz maszynowy wylaczony", "")
            continue
        texts = [items[i][1] for i in idx]
        try:
            res = _mt_call(mt, texts, code, purpose)
        except MTUnavailable as exc:
            for i in idx:
                out[i] = (None, f"brak tlumacza: {str(exc)[:150]}", "")
            continue
        engine = getattr(mt, "last_name", None) or getattr(mt, "label", None) or "mt"
        for i, src, pl in zip(idx, texts, res):
            errs = [p for p in check_segment("cytat", src, pl, code) if p.level == "blad"]
            out[i] = (None, "kontrola: " + "; ".join(p.what for p in errs[:2]), engine) if errs else (pl.strip(), "", engine)
    return out


def _mt_call(mt, texts: list[str], lang: str, purpose: str) -> list[str]:
    try:
        return mt.translate(texts, lang, "pl", purpose=purpose)
    except TypeError:                                  # tlumacz bez celu (np. sam NLLB, testowy)
        return mt.translate(texts, lang, "pl")


def supported_label(label: str | None, text: str) -> bool:
    """Etykieta (kraj, epoka, tradycja) podana przez LLM zostaje tylko, gdy wynika z przetlumaczonego cytatu:
    liczby i liczby rzymskie z etykiety sa w tekscie, a co najmniej polowa slow (rdzenie 5 liter) tez."""
    if not label:
        return False
    t = " " + norm_text(text) + " "
    nums = re.findall(r"\b[ivxlcdm]+\b|\d+", norm_text(label))
    if any(f" {n} " not in t and n not in t for n in nums):
        return False
    words = [w for w in norm_text(label).split() if len(w) >= 4 and not re.fullmatch(r"[ivxlcdm]+|\d+", w)]
    if not words:
        return bool(nums)
    return sum(1 for w in words if f" {w[:5]}" in t) * 2 >= len(words)


def finish_facts(facts: list, lang: str, mt, dosage_re=None) -> tuple[list, list, list[str]]:
    """Informacje z cytatami -> (gotowe z polskim tekstem, czekajace na tlumacza, powody odrzucenia).
    Informacja o magii (ma .meta): bez ilosci/dawek w tlumaczeniu, etykiety kraj/epoka/tradycja tylko wynikajace
    z tekstu."""
    if not facts:
        return [], [], []
    res = translate_quotes(mt, [(lang, f.quote) for f in facts])
    ready, pending, rejected = [], [], []
    for f, (pl, why, engine) in zip(facts, res):
        if pl is None and why.startswith("brak tlumacza"):
            f.text, f.translated_by = f.quote, None
            pending.append(f)
            continue
        if pl is None:
            rejected.append(f"{why}: {f.quote[:60]}")
            continue
        if not (12 <= len(pl) <= 700):
            rejected.append(f"dlugosc tlumaczenia: {pl[:60]}")
            continue
        if dosage_re is not None and dosage_re.search(pl):
            rejected.append(f"ilosci/dawki (opis, nie przepis): {pl[:60]}")
            continue
        meta = getattr(f, "meta", None)
        if meta:
            for key in ("kraj", "epoka", "tradycja"):
                if key in meta and not supported_label(meta[key], pl):
                    meta.pop(key)
        f.text, f.translated_by = pl, engine
        ready.append(f)
    return ready, pending, rejected


def quote_leads(leads: list[dict], source: str, lang: str | None, mt) -> list[dict]:
    """Wzmianki o przepisach: "co_wiadomo" = przetlumaczony cytat ze zrodla (nie slowa LLM), tytul z tlumaczenia
    nazwy oryginalnej, gdy ta jest w tekscie. Cytat spoza zrodla -> wzmianka bez opisu."""
    if not leads:
        return leads
    items, where = [], []
    for i, w in enumerate(leads):
        q = snap_quote(w.get("cytat") or "", source, 300) if w.get("cytat") else None
        w["cytat"] = q
        w["co_wiadomo"] = None
        if q:
            items.append((lang or "", q))
            where.append((i, "co_wiadomo"))
        orig = (w.get("tytul_oryginalny") or "").strip()
        if orig and norm_text(orig) and norm_text(orig) in norm_text(source):
            items.append((lang or "", orig))
            where.append((i, "tytul"))
    if items:
        for (i, key), (pl, _why, _engine) in zip(where, translate_quotes(mt, items, "wiedza")):
            if pl:
                leads[i][key] = pl[:800] if key == "co_wiadomo" else pl[:150]
    return leads
