"""Petla planisty ciaglosci w tle (start aplikacji)."""
import asyncio
import logging

from app.config import settings

log = logging.getLogger(__name__)


IDLE_POLL_S = 60     # co tyle sekund szybkie sprawdzenie, czy kolejka nie stoi (samo zapytanie do bazy)


def _idle(state=None) -> bool:
    """Jest wolne miejsce w tle: mniej zadan Siedziby niz limit z panelu (0 = wstrzymane, nigdy wolne)."""
    from sqlmodel import Session

    from app.agents import store
    from app.agents.tasks import production_jobs
    from app.db import engine
    cap = store.job_limit(state)
    if cap <= 0:
        return False
    with Session(engine) as s:
        return len(production_jobs(s)) < cap


async def agents_loop(state, first_delay: float = 90, poll_s: float = IDLE_POLL_S) -> None:
    """Pelne sprawdzenie (straznik + propozycje) co AGENTS_CHECK_MINUTES, a gdy kolejka stoi - najpozniej po
    minucie (zadanie, ktore skonczylo sie po kilku sekundach, nie zostawia Siedziby bezczynnej na 5 min).
    "Stoi" = mniej zadan niz limit zadan w tle z panelu (0-6) - planista doklada po jednym na sprawdzenie.
    Przy okazji: synchronizacja z backendem Kwiatownika (co BACKEND_SYNC_MINUTES) i wdrozeniowiec (co 30 min)."""
    if not settings.agents_enabled:
        return
    import time

    from app.agents import backend_sync, deploy, store
    from app.agents.planner import run_planner
    await asyncio.sleep(first_delay)          # po starcie: najpierw wznowione zadania i modele
    last_full = last_sync = float("-inf")
    while True:
        try:
            due = time.monotonic() - last_full >= max(1, settings.agents_check_minutes) * 60
            if due or (store.runtime().get("auto", True) and _idle(state)):
                run_planner(state, "auto")
                last_full = time.monotonic()
        except Exception:
            log.exception("Planista ciaglosci: sprawdzenie nieudane")
        try:
            await deploy.maybe_start(state)
        except Exception:
            log.exception("Wdrozeniowiec: sprawdzenie nieudane")
        if backend_sync.configured() and time.monotonic() - last_sync >= max(1, settings.backend_sync_minutes) * 60:
            last_sync = time.monotonic()
            try:
                await backend_sync.sync()
            except Exception:
                log.exception("Backend Kwiatownika: synchronizacja nieudana")
        await asyncio.sleep(poll_s)
