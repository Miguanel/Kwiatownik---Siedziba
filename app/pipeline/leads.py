"""Wzmianki o przepisach: strona opisuje przepis (np. "w Japonii z lisci dokudami robi sie herbate na tradzik"),
ale nie podaje jego wykonania. Siedziba zapisuje taka wzmianke i szuka pelnego przepisu w innych miejscach internetu.

Skad wzmianki:
  - tlumaczenie przepisu: LLM stwierdza "brak konkretnego przepisu" albo przepis jest niepelny (export.translate_one),
  - przegladanie ciekawostek (Item.kind == fact) - zadanie "leads_extract".
Szukanie (zadanie "leads_search"): zapytania w jezyku zrodla, po angielsku i po polsku (nazwa przepisu, roslina
lacinska + slowo "przepis") -> SearXNG -> strona musi dotyczyc tej rosliny -> classify() (JSON-LD / reguly / LLM)
musi uznac ja za przepis -> nowa pozycja w Zebranych (status raw), dalej zwykly potok: tlumaczenie, filtr, eksport.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.knowledge.sections import norm_text
from app.models import Item, ItemKind, ItemStatus, RecipeLead

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
RECIPE_WORDS = {"en": "recipe", "de": "Rezept", "fr": "recette", "es": "receta", "it": "ricetta", "uk": "рецепт",
                "ru": "рецепт", "pl": "przepis", "cs": "recept", "sk": "recept", "lt": "receptas", "hu": "recept",
                "ro": "rețetă", "bg": "рецепта", "ja": "作り方", "zh": "做法", "ko": "만드는 법"}
SKIP = ("youtube.com", "facebook.com", "instagram.com", "pinterest.", "amazon.", "allegro.", "ebay.", "aliexpress.",
        "tiktok.com", "x.com", "twitter.com")


def lead_fingerprint(title: str, plant: str | None) -> str:
    words = sorted({w[:6] for w in norm_text(f"{title} {plant or ''}").split() if len(w) > 2})
    return hashlib.sha1(" ".join(words).encode()).hexdigest()[:16]


def save_leads(leads: list[dict], source_url: str, source_name: str | None, language: str | None,
               item_id: int | None, model: str | None) -> list[int]:
    """Zapisuje wzmianki (bez powtorzen: ta sama nazwa + roslina). Zwraca id nowych."""
    new = []
    with Session(engine) as s:
        for w in leads:
            fp = lead_fingerprint(w["tytul"], w.get("roslina"))
            if s.exec(select(RecipeLead.id).where(RecipeLead.fingerprint == fp)).first():
                continue
            row = RecipeLead(title=w["tytul"][:200], title_orig=(w.get("tytul_oryginalny") or None),
                             plant=w.get("roslina"), plant_latin=w.get("roslina_lacinska"), typ=w.get("typ"),
                             known=w.get("co_wiadomo"),
                             ingredients_json=json.dumps(w.get("znane_skladniki") or [], ensure_ascii=False),
                             quote=w.get("cytat"), source_url=source_url, source_name=source_name, language=language,
                             item_id=item_id, fingerprint=fp, model=model)
            s.add(row)
            s.commit()
            new.append(row.id)
    return new


# ------------------------------------------------------------------ wzmianki w ciekawostkach
EXTRACT_PROMPT = """Czytasz artykul o roslinach dla polskiego zielnika "Kwiatownik". Znajdz w nim przepisy, o ktorych
tekst mowi, ale ktorych pelnego wykonania NIE podaje. Zwroc TYLKO JSON: {"wzmianki": [...]}.
"""


async def extract_leads(llm, title: str, text: str) -> tuple[list[dict], str]:
    from app.pipeline.translate import LEAD_FORMAT, TYPES, clean_leads
    prompt = f"TYPY: {', '.join(TYPES)}\nTYTUL: {title}\n\nTEKST:\n{text[:9000]}"
    res = await llm.complete(prompt, system=EXTRACT_PROMPT + LEAD_FORMAT, json_mode=True)
    data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
    return clean_leads(data.get("wzmianki") if isinstance(data, dict) else data), f"{res.provider}/{res.model}"


def facts_to_check(limit: int) -> list[int]:
    with Session(engine) as s:
        return list(s.exec(select(Item.id).where(Item.kind == ItemKind.fact, col(Item.raw_text).is_not(None),
                                                 col(Item.leads_at).is_(None))
                           .order_by(col(Item.id).desc()).limit(limit)).all())


# ------------------------------------------------------------------ szukanie pelnego przepisu
def queries_for(lead: RecipeLead) -> list[tuple[str, str]]:
    """(zapytanie, jezyk): nazwa w jezyku zrodla, angielska z lacinska nazwa rosliny, polska."""
    lang = (lead.language or "en")[:2]
    out: list[tuple[str, str]] = []
    if lead.title_orig:
        out.append((f"\"{lead.title_orig}\" {RECIPE_WORDS.get(lang, '')}".strip(), lang))
    if lead.plant_latin:
        kind = (lead.title_orig or lead.title) if lang == "en" else ""
        out.append((f"{lead.plant_latin} {kind} {RECIPE_WORDS['en']}".replace("  ", " ").strip(), "en"))
        if lang not in ("en", "pl"):
            out.append((f"{lead.plant_latin} {RECIPE_WORDS.get(lang, '')}".strip(), lang))
    out.append((f"{lead.title} {RECIPE_WORDS['pl']}", "pl"))
    seen, uniq = set(), []
    for q, lg in out:
        if q not in seen:
            seen.add(q)
            uniq.append((q, lg))
    return uniq[:4]


def about_lead(text: str, lead: RecipeLead) -> bool:
    """Strona o tej roslinie / tym przepisie: rodzaj lacinski, nazwa rosliny albo slowa nazwy przepisu."""
    low = norm_text(text)
    if lead.plant_latin and norm_text(lead.plant_latin).split()[0] in low:
        return True
    for name in (lead.plant, lead.title_orig):
        if name and norm_text(name) and norm_text(name) in low:
            return True
    words = [w for w in norm_text(lead.title).split() if len(w) > 4]
    return bool(words) and sum(1 for w in words if w[:6] in low) >= max(1, len(words) - 1)


async def search_lead(lead_id: int, llm, fetcher, backends, logf) -> dict:
    """Szuka pelnego przepisu dla jednej wzmianki. Zwraca {"found": item_id | None, "checked": n}."""
    from app.discovery.search import search_all
    from app.scrapers.classify import DEFAULT_KEYWORDS, classify, jsonld_recipe
    from app.scrapers.page import parse_page
    from app.scrapers.recipe_data import normalize_jsonld_recipe
    with Session(engine) as s:
        lead = s.get(RecipeLead, lead_id)
        s.expunge(lead)
    history = json.loads(lead.search_log or "[]")
    tried = {h.get("url") for h in history if h.get("url")} | {lead.source_url}
    found_item, checked = None, 0
    for q, lang in queries_for(lead):
        if found_item:
            break
        results = await search_all(backends, q, lang, "", log_fn=logf) if backends else []
        history.append({"q": q, "lang": lang, "wyniki": len(results)})
        for r in results[:6]:
            host = urlparse(r.url).netloc
            if r.url in tried or any(d in host for d in SKIP):
                continue
            tried.add(r.url)
            checked += 1
            res = await fetcher.get(r.url)
            if not res.ok:
                history.append({"url": r.url, "wynik": f"blad {res.error or res.status}"})
                continue
            page = await asyncio.to_thread(parse_page, res.text, res.url)
            if not about_lead(f"{page.title} {page.text}", lead):
                history.append({"url": res.url, "wynik": "nie o tej roslinie"})
                continue
            c = await classify(page, DEFAULT_KEYWORDS, llm)
            if c.kind != "recipe":
                history.append({"url": res.url, "wynik": f"to nie przepis ({c.reason[:80]})"})
                continue
            node = jsonld_recipe(page)
            found_item = _save_found(lead, res.url, c, page, normalize_jsonld_recipe(node) if node else None)
            history.append({"url": res.url, "wynik": f"PRZEPIS ({c.method})", "item": found_item})
            logf(f"[znaleziony] wzmianka #{lead.id} '{lead.title[:50]}': {res.url}")
            break
    with Session(engine) as s:
        row = s.get(RecipeLead, lead_id)
        row.attempts += 1
        row.last_search_at = datetime.now(timezone.utc)
        row.search_log = json.dumps(history[-60:], ensure_ascii=False)
        if found_item:
            row.status, row.found_item_id = "znaleziony", found_item
            with Session(engine) as s2:
                it = s2.get(Item, found_item)
                row.found_url = it.url if it else None
        elif row.attempts >= MAX_ATTEMPTS:
            row.status = "nie_znaleziono"
        s.add(row)
        s.commit()
    if not found_item:
        logf(f"[brak] wzmianka #{lead.id} '{lead.title[:50]}': sprawdzono {checked} stron")
    return {"found": found_item, "checked": checked}


def _save_found(lead: RecipeLead, url: str, c, page, structured: dict | None) -> int:
    """Odnaleziony przepis -> pozycja w Zebranych (albo istniejaca, jesli juz byla)."""
    with Session(engine) as s:
        old = s.exec(select(Item).where(Item.url == url)).first()
        if old:
            if old.kind != ItemKind.recipe and old.status in (ItemStatus.skipped, ItemStatus.raw, ItemStatus.error):
                old.kind, old.status = ItemKind.recipe, ItemStatus.raw
                old.raw_text = old.raw_text or page.text
                old.reason = f"odnaleziony dla wzmianki #{lead.id}"
                s.add(old)
                s.commit()
            return old.id
        it = Item(url=url, kind=ItemKind.recipe, status=ItemStatus.raw, title=(c.title or page.title or url)[:300],
                  raw_text=page.text, structured_json=json.dumps(structured, ensure_ascii=False) if structured else None,
                  confidence=c.confidence, method=f"wzmianka:{c.method}", language=page.lang or lead.language,
                  reason=f"odnaleziony dla wzmianki #{lead.id}: {lead.title}"[:300])
        s.add(it)
        s.commit()
        return it.id


def leads_to_search(limit: int) -> list[int]:
    with Session(engine) as s:
        return list(s.exec(select(RecipeLead.id).where(RecipeLead.status == "szukam",
                                                       RecipeLead.attempts < MAX_ATTEMPTS)
                           .order_by(RecipeLead.attempts, col(RecipeLead.last_search_at).is_not(None),
                                     RecipeLead.last_search_at, RecipeLead.id).limit(limit)).all())
