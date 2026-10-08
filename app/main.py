import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agents.loop import agents_loop
from app.api.llm import router as llm_router
from app.config import settings
from app.db import init_db
from app.llm.factory import build_router
from app.web.agents_routes import router as agents_router
from app.web.site_routes import router as site_router
from app.web.discover_routes import router as discover_router
from app.web.export_routes import router as export_router
from app.web.routes import router as web_router
from app.web.scrapers_routes import router as scrapers_router
from app.web.knowledge_routes import router as knowledge_router
from app.web.marketing_routes import router as marketing_router
from app.web.monitor_routes import router as monitor_router
from app.web.tasks_routes import router as tasks_router
from app.worker import actions
from app.worker.discover import cleanup_discoveries
from app.worker.jobs import make_runner, mark_interrupted_jobs, retry_depth
from app.worker.ollama_pull import ensure_ollama
from app.worker.export import recheck_filters
from app.worker.scan import backfill_duplicates
from app.worker.scheduler import auto_scan_loop

WEB_DIR = Path(__file__).parent / "web"
log = logging.getLogger(__name__)


def _safe(label: str, fn, default=None):
    """Krok startu, ktory nie jest niezbedny - blad jest logowany, ale nie blokuje uruchomienia strony."""
    try:
        return fn()
    except Exception:
        log.exception("Start: krok '%s' nie powiodl sie - pomijam", label)
        return default


def _sync_plants() -> int:
    from sqlmodel import Session

    from app.db import engine
    from app.knowledge.registry import sync_kwiatownik
    with Session(engine) as s:
        return sync_kwiatownik(s, settings.kwiatownik_plants_dir)


async def _refresh_models(state) -> None:
    """Wykrywanie modeli w tle - strona dziala od razu, nawet gdy API odpowiada wolno."""
    try:
        await state.llm.refresh()
        log.warning("Modele LLM: %d na liscie", len(state.llm.order))
    except Exception as exc:  # brak sieci nie blokuje startu - zostaje zapisany ranking
        log.warning("Odswiezenie modeli nie powiodlo sie: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    interrupted = _safe("wykrycie przerwanych zadan", mark_interrupted_jobs, [])
    cleaned = _safe("porzadkowanie odkrytych stron", cleanup_discoveries, 0)
    if cleaned:
        log.warning("Odkrywanie: uporzadkowano %d wpisow (dodane do zrodel / poprawione domeny)", cleaned)
    rechecked = _safe("filtr kulinarny starszych przepisow", recheck_filters, 0)
    if rechecked:
        log.warning("Filtr kulinarny: %d starszych przepisow oznaczono jako pominiete", rechecked)
    _safe("rejestr roslin z Kwiatownika", _sync_plants, 0)
    merged = _safe("laczenie duplikatow", backfill_duplicates, 0)
    if merged:
        log.warning("Polaczono %d duplikatow przepisow ze starszych skanow", merged)
    app.state.jobs = make_runner(settings.max_concurrent_jobs)  # kolejka zadan z limitem
    app.state.llm = _safe("router LLM", build_router)  # router modeli (None = brak kluczy)
    tasks = []
    if app.state.llm:
        tasks.append(asyncio.create_task(_refresh_models(app.state)))
    tasks.append(asyncio.create_task(ensure_ollama(app.state)))  # lokalny model zapasowy (w tle)
    if settings.resume_interrupted_jobs:  # dokoncz zadania przerwane restartem
        for jid in interrupted or []:
            try:
                if retry_depth(jid) < settings.max_auto_retries:
                    new = actions.retry_job(app.state, jid)
                    log.warning("Wznowiono przerwane zadanie #%s jako #%s", jid, new)
            except Exception:
                log.exception("Nie udalo sie wznowic zadania #%s", jid)
    tasks.append(asyncio.create_task(auto_scan_loop(app.state)))  # codzienny skan zrodel "auto"
    tasks.append(asyncio.create_task(agents_loop(app.state)))     # planista ciaglosci: Siedziba zawsze nad czyms pracuje
    log.warning("Siedziba Kwiatownika gotowa: http://localhost:8000")
    yield
    for t in tasks:
        t.cancel()
    await app.state.jobs.shutdown()
    if app.state.llm:
        await app.state.llm.aclose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
app.include_router(web_router)
app.include_router(llm_router)
app.include_router(discover_router)
app.include_router(export_router)
app.include_router(tasks_router)
app.include_router(monitor_router)
app.include_router(scrapers_router)
app.include_router(knowledge_router)
app.include_router(marketing_router)
app.include_router(agents_router)
app.include_router(site_router)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}