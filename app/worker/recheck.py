"""Etap 7: ponowna ocena stron odrzuconych bez LLM (zadanie "page_recheck").

Gdy LLM byl niedostepny, siec page_kind decydowala "jako zapas" - i 1425 stron oznaczyla jako "nic ciekawego"
(zgodnosc tej klasy z LLM: 2/14). Takie strony (Item.kind=other, metoda nn:page_kind / odlozone:llm) sa tu
pobierane jeszcze raz i oceniane normalnie: reguly -> LLM (bez sieci jako zapasu). Przepis albo ciekawostka
wracaja do Zebranych (status raw), "nic ciekawego" od LLM zostaje z nowa metoda (bez ponownego sprawdzania).
LLM znow niedostepny -> strona czeka na nastepny raz.
"""
import asyncio
import json
from datetime import datetime, timezone

from sqlmodel import Session, col, func, or_, select

from app.config import settings
from app.db import engine
from app.models import Item, ItemKind, ItemStatus, Job
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner

PICK = or_(col(Item.method).like("nn:page_kind%"), col(Item.method) == "odlozone:llm")


def pending_count() -> int:
    with Session(engine) as s:
        return s.exec(select(func.count()).select_from(Item).where(Item.kind == ItemKind.other, PICK)).one()


def pending_ids(limit: int) -> list[int]:
    with Session(engine) as s:
        return list(s.exec(select(Item.id).where(Item.kind == ItemKind.other, PICK)
                           .order_by(col(Item.id).desc()).limit(limit)).all())


def _progress(job_id: int, done: int, total: int, stats: dict) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job:
            job.progress, job.total = done, total
            job.stats_json = json.dumps(stats)
            s.add(job)
            s.commit()


async def run_page_recheck(job_id: int, runner: JobRunner, llm, limit: int = 50, fetcher=None) -> dict:
    from app.scrapers.classify import DEFAULT_KEYWORDS, DEFERRED, classify, jsonld_recipe
    from app.scrapers.clean import clean_text
    from app.scrapers.fetcher import Fetcher
    from app.scrapers.page import parse_page
    from app.scrapers.recipe_data import normalize_jsonld_recipe
    log = lambda m: job_log(job_id, m)  # noqa: E731
    if llm is None:
        raise RuntimeError("Ponowna ocena stron wymaga LLM (klucze w .env albo Ollama)")
    ids = pending_ids(limit)
    stats = {"sprawdzone": 0, "przepisy": 0, "ciekawostki": 0, "nic": 0, "czeka": 0, "bledy": 0}
    log(f"Strony odrzucone bez LLM do ponownej oceny: {len(ids)} z {pending_count()}")
    own = fetcher is None
    fetcher = fetcher or Fetcher(settings.user_agent, settings.request_delay_s)
    waiting_in_row = 0
    try:
        for n, iid in enumerate(ids, 1):
            if runner.should_stop(job_id):
                log("Zatrzymano")
                break
            if waiting_in_row >= 3:
                log("LLM niedostepny trzy razy z rzedu - przerywam, reszta przy nastepnym uruchomieniu")
                break
            with Session(engine) as s:
                it = s.get(Item, iid)
                url, title = it.url, it.title
            activity.set(job_id, f"ponowna ocena {n}/{len(ids)}", title or url)
            res = await fetcher.get(url)
            if not res.ok:
                stats["bledy"] += 1
                with Session(engine) as s:
                    it = s.get(Item, iid)
                    it.method, it.reason = "heuristic", f"ponowna ocena: strona niedostepna ({res.error or res.status})"
                    s.add(it)
                    s.commit()
                continue
            page = await asyncio.to_thread(parse_page, res.text, res.url)
            c = await classify(page, DEFAULT_KEYWORDS, llm, nn_fallback=False)
            if c.method == DEFERRED:
                stats["czeka"] += 1
                waiting_in_row += 1
                continue
            waiting_in_row = 0
            stats["sprawdzone"] += 1
            with Session(engine) as s:
                it = s.get(Item, iid)
                it.method, it.confidence = c.method, c.confidence
                it.reason = f"ponowna ocena: {c.reason}"[:300]
                if c.kind in ("recipe", "fact"):
                    node = jsonld_recipe(page) if c.kind == "recipe" else None
                    it.kind, it.status = ItemKind(c.kind), ItemStatus.raw
                    it.title = c.title or page.title or it.title
                    it.raw_text = clean_text(page.text)
                    it.language = it.language or page.lang
                    if node:
                        it.structured_json = json.dumps(normalize_jsonld_recipe(node), ensure_ascii=False)
                    stats["przepisy" if c.kind == "recipe" else "ciekawostki"] += 1
                    log(f"[odzyskana] {c.kind}: {(c.title or url)[:70]} ({c.method})")
                else:
                    stats["nic"] += 1
                s.add(it)
                s.commit()
            _progress(job_id, n, len(ids), stats)
    finally:
        if own:
            await fetcher.aclose()
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in stats.items())
        + f"; czeka jeszcze {pending_count()} stron")
    _progress(job_id, len(ids), len(ids), stats)
    return stats
