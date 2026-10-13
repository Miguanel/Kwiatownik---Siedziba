"""Automatyczny trening sieci co jakis czas (petla w tle, ustawienia w panelu /nn).

Co 10 minut petla sprawdza, czy pora na trening:
  - wlaczony automat i minelo co najmniej `hours` godzin od ostatniego treningu (recznego albo automatycznego),
  - przy interwale >= 24 h i ustawionej godzinie - tylko w oknie 2 h od tej godziny (np. w nocy, gdy komputer wolny),
  - "tylko gdy przybylo danych": trenowane sa tylko sieci z nowymi przykladami (status.readiness);
    inaczej wszystkie sieci, ktore maja dosc danych,
  - nie startuje, gdy trening juz trwa albo czeka w kolejce.
Nowa wersja zastepuje biezaca tylko, gdy nie jest gorsza (mistrz/pretendent w train.py) - automat niczego nie psuje.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import Job, JobStatus, NNModel
from app.nn import store

log = logging.getLogger(__name__)
HOURS_CHOICES = (6, 12, 24, 48, 72, 168)


def auto_settings() -> dict:
    a = store.load_state().get("auto") or {}
    return {"on": bool(a.get("on", settings.nn_auto_train)), "hours": int(a.get("hours", settings.nn_auto_train_hours)),
            "time": a.get("time", settings.nn_auto_train_time) or "", "only_new": bool(a.get("only_new", True)),
            "last": a.get("last"), "history": list(a.get("history") or [])}


def save_auto(**fields) -> dict:
    with store._lock:
        data = store.load_state()
        a = data.setdefault("auto", {})
        a.update(fields)
        store.save_state(data)
    return auto_settings()


def _utc(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt and dt.tzinfo is None else dt


def last_training(s: Session) -> datetime | None:
    """Ostatni trening: najnowsza wersja sieci albo ostatnie zakonczone zadanie nn_train."""
    m = s.exec(select(func.max(NNModel.created_at))).one()
    j = s.exec(select(func.max(Job.finished_at)).where(Job.kind == "nn_train")).one()
    times = [_utc(t) for t in (m, j) if t]
    return max(times) if times else None


def in_window(now_local: datetime, hhmm: str, width_h: float = 2.0) -> bool:
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except ValueError:
        return True
    start = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
    if start > now_local:
        start -= timedelta(days=1)
    return now_local - start < timedelta(hours=width_h)


def due(now: datetime | None = None) -> tuple[list[str], str]:
    """(zadania do treningu, powod). Pusta lista = nie teraz (powod mowi dlaczego)."""
    from app.nn.status import approx_examples, readiness
    from app.nn.tasks import TASKS
    now = now or datetime.now(timezone.utc)
    a = auto_settings()
    if not settings.nn_enabled or not a["on"]:
        return [], "automatyczny trening wylaczony"
    with Session(engine) as s:
        busy = s.exec(select(Job.id).where(Job.kind == "nn_train",
                                           col(Job.status).in_([JobStatus.queued, JobStatus.running]))).first()
        if busy:
            return [], f"trening juz trwa (zadanie #{busy})"
        last = last_training(s)
        if last and now - last < timedelta(hours=a["hours"]):
            nxt = (last + timedelta(hours=a["hours"])).astimezone(ZoneInfo(settings.timezone))
            return [], (f"ostatni trening {last.astimezone(ZoneInfo(settings.timezone)):%Y-%m-%d %H:%M} - "
                        f"nastepny po {a['hours']} h (najwczesniej {nxt:%m-%d %H:%M}"
                        + (f", w oknie od {a['time']}" if a["time"] and a["hours"] >= 24 else "") + ")")
        if a["time"] and a["hours"] >= 24 and not in_window(now.astimezone(ZoneInfo(settings.timezone)), a["time"]):
            return [], f"czeka na godzine {a['time']}"
        if a["only_new"]:
            tasks = [r["task"] for r in readiness(ttl=0)]
            if not tasks:
                return [], "brak nowych przykladow od ostatniego treningu"
            return tasks, "przybylo nowych przykladow"
        tasks = [t for t, sp in TASKS.items() if approx_examples(s, t) >= sp.min_total]
    return tasks, f"co {a['hours']} h" if tasks else "za malo danych"


def run_if_due(state, now: datetime | None = None) -> int | None:
    from app.worker import actions
    tasks, reason = due(now)
    if not tasks:
        return None
    jid = actions.start_nn_train(state, tasks)
    a = auto_settings()
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo(settings.timezone)).strftime("%Y-%m-%d %H:%M")
    hist = ([{"at": local, "job": jid, "tasks": tasks, "reason": reason}] + a["history"])[:15]
    save_auto(last=now.isoformat(timespec="minutes"), history=hist)
    log.warning("Automatyczny trening sieci: zadanie #%s (%s: %s)", jid, reason, ", ".join(tasks))
    return jid


async def nn_schedule_loop(state, every_s: float = 600, first_s: float = 180) -> None:
    await asyncio.sleep(first_s)                      # po starcie najpierw wznowienia i odswiezenie modeli
    while True:
        try:
            run_if_due(state)                         # w petli zdarzen: start zadania wymaga asyncio.create_task
        except Exception:  # noqa: BLE001 - petla w tle nigdy nie umiera
            log.exception("Automatyczny trening sieci: blad sprawdzania")
        await asyncio.sleep(every_s)
