"""Zadania wzmianek o przepisach: 'leads_extract' (wzmianki w ciekawostkach) i 'leads_search' (szukanie pelnych przepisow)."""
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlmodel import Session

from app.config import settings
from app.db import engine
from app.llm.router import AllModelsFailedError
from app.models import Item, Job, RecipeLead
from app.pipeline import leads as leadmod
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


def _job(job_id: int, done: int, total: int, stats: dict | None = None) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job:
            job.progress, job.total = done, total
            if stats is not None:
                job.stats_json = json.dumps(stats, ensure_ascii=False)
            s.add(job)
            s.commit()


async def run_leads_extract(job_id: int, runner: JobRunner, llm, limit: int = 20) -> dict:
    """Przeglada ciekawostki (artykuly) i zapisuje wzmianki o przepisach bez pelnego wykonania."""
    ids = leadmod.facts_to_check(limit)
    stats = {"artykuly": 0, "wzmianki": 0, "bledy": 0}
    job_log(job_id, f"Ciekawostki do przejrzenia: {len(ids)}")
    _job(job_id, 0, len(ids))
    for i, iid in enumerate(ids, 1):
        if runner.should_stop(job_id):
            break
        with Session(engine) as s:
            it = s.get(Item, iid)
            title, text, url, lang = it.title or "", it.raw_text or "", it.url, it.language
        activity.set(job_id, "szukam wzmianek o przepisach", title[:80])
        try:
            found, model = await leadmod.extract_leads(llm, title, text)
        except AllModelsFailedError as exc:
            job_log(job_id, f"Modele niedostepne - przerywam ({str(exc)[:120]})")
            break
        except Exception as exc:  # noqa: BLE001 - zly JSON jednego artykulu nie zatrzymuje reszty
            stats["bledy"] += 1
            job_log(job_id, f"[blad] #{iid} {title[:50]}: {str(exc)[:120]}")
            found, model = [], None
        new = leadmod.save_leads(found, url, urlparse(url).netloc, lang, iid, model) if found else []
        with Session(engine) as s:
            it = s.get(Item, iid)
            it.leads_at = datetime.now(timezone.utc)
            s.add(it)
            s.commit()
        stats["artykuly"] += 1
        stats["wzmianki"] += len(new)
        if new:
            job_log(job_id, f"#{iid} {title[:50]}: {len(new)} - " + ", ".join(w["tytul"][:40] for w in found[:3]))
        _job(job_id, i, len(ids), stats)
    job_log(job_id, f"Koniec: przejrzano {stats['artykuly']}, nowych wzmianek {stats['wzmianki']}")
    activity.clear(job_id)
    return stats


async def run_leads_search(job_id: int, runner: JobRunner, llm, lead_ids: list[int] | None = None,
                           limit: int = 5) -> dict:
    """Szuka pelnych przepisow (skladniki + kroki) dla wzmianek."""
    from app.scrapers.fetcher import Fetcher
    from app.worker.discover import build_backends
    ids = lead_ids or leadmod.leads_to_search(limit)
    backends = build_backends()
    stats = {"wzmianki": 0, "znalezione": 0, "strony": 0}
    if not backends:
        job_log(job_id, "Brak wyszukiwarki (SEARXNG_URL / BRAVE_API_KEY) - nie ma gdzie szukac")
        return stats
    fetcher = Fetcher(settings.user_agent, settings.request_delay_s,
                      accept_language="pl,en;q=0.8,de;q=0.6,uk;q=0.5,ja;q=0.5,zh;q=0.5,*;q=0.3")
    _job(job_id, 0, len(ids))
    try:
        for i, lid in enumerate(ids, 1):
            if runner.should_stop(job_id):
                break
            with Session(engine) as s:
                lead = s.get(RecipeLead, lid)
                if not lead or lead.status in ("znaleziony", "odrzucony"):
                    continue
                title = lead.title
            activity.set(job_id, "szukam pelnego przepisu", title[:80])
            job_log(job_id, f"Wzmianka #{lid}: {title[:70]}")
            try:
                res = await leadmod.search_lead(lid, llm, fetcher, backends, lambda m: job_log(job_id, m))
            except AllModelsFailedError as exc:
                job_log(job_id, f"Modele niedostepne - przerywam ({str(exc)[:120]})")
                break
            stats["wzmianki"] += 1
            stats["strony"] += res["checked"]
            stats["znalezione"] += 1 if res["found"] else 0
            _job(job_id, i, len(ids), stats)
    finally:
        await fetcher.aclose()
    job_log(job_id, f"Koniec: {stats['znalezione']}/{stats['wzmianki']} odnalezionych, sprawdzono {stats['strony']} stron"
            + (" - odnalezione czekaja w Zebranych na tlumaczenie" if stats["znalezione"] else ""))
    activity.clear(job_id)
    return stats
