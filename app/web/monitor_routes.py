"""Widok "Na zywo": nad czym system pracuje w tej chwili (odswiezany co 2 s przez HTMX)."""
import json
import re
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request, Response
from sqlmodel import Session, col, func, select

from app.db import engine
from app.models import Job, JobStatus, Source
from app.web.templating import templates
from app.worker import activity, snapshots

router = APIRouter()
LOG_LINE = re.compile(r"^(\d\d:\d\d:\d\d) (.*)$")


def _utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt and dt.tzinfo is None else dt


def _fmt_s(sec: float | None) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def _ctx(request: Request) -> dict:
    now = datetime.now(timezone.utc)
    with Session(engine) as s:
        names = {x.id: x.name for x in s.exec(select(Source)).all()}
        running = s.exec(select(Job).where(Job.status == JobStatus.running).order_by(Job.id)).all()
        queued = s.exec(select(Job).where(Job.status == JobStatus.queued).order_by(Job.id)).all()
        finished = s.exec(select(Job).where(col(Job.finished_at).is_not(None))
                          .order_by(col(Job.finished_at).desc()).limit(10)).all()
        hour = now - timedelta(hours=1)
        done_h = s.exec(select(func.count()).select_from(Job).where(
            Job.status == JobStatus.done, col(Job.finished_at) >= hour)).one()
        failed_h = s.exec(select(func.count()).select_from(Job).where(
            Job.status == JobStatus.failed, col(Job.finished_at) >= hour)).one()

    cards, feed = [], []
    for j in running:
        started = _utc(j.started_at or j.created_at)
        elapsed = (now - started).total_seconds()
        eta = elapsed / j.progress * (j.total - j.progress) if j.progress and j.total and j.total > j.progress else None
        lines = [ln for ln in (j.log or "").splitlines() if ln.strip()]
        for ln in lines[-15:]:
            m = LOG_LINE.match(ln)
            feed.append({"time": m.group(1) if m else "", "job": j.id, "kind": j.kind,
                         "text": m.group(2) if m else ln})
        cards.append({"job": j, "name": names.get(j.source_id, ""), "elapsed": _fmt_s(elapsed), "eta": _fmt_s(eta),
                      "pct": min(100, int(100 * (j.progress or 0) / j.total)) if j.total else None,
                      "act": activity.get(j.id), "last": lines[-4:],
                      "stats": json.loads(j.stats_json) if j.stats_json else None})
    feed.sort(key=lambda x: x["time"], reverse=True)

    router_ = getattr(request.app.state, "llm", None)
    llm_inflight = dict(router_.inflight) if router_ else {}
    llm_recent = [dict(r, ago=round(time.time() - r["ts"])) for r in list(router_.recent)[::-1][:10]] if router_ else []
    runner = request.app.state.jobs
    return {
        "cards": cards, "queued": queued, "names": names, "feed": feed[:30],
        "finished": [{"job": j, "name": names.get(j.source_id, ""),
                      "took": _fmt_s((_utc(j.finished_at) - _utc(j.started_at or j.created_at)).total_seconds())}
                     for j in finished],
        "limit": runner.max_concurrent, "done_h": done_h, "failed_h": failed_h,
        "llm_inflight": llm_inflight, "llm_recent": llm_recent, "events": activity.recent_events(25),
        "snapshots": [_when(m) for m in snapshots.recent(15)],
        "now": now,
    }


@router.get("/monitor")
def monitor_page(request: Request):
    return templates.TemplateResponse(request, "monitor.html", _ctx(request))


@router.get("/monitor/live")
def monitor_live(request: Request):
    return templates.TemplateResponse(request, "monitor_live.html", _ctx(request))


def _when(meta: dict) -> dict:
    try:
        return {**meta, "ts": datetime.fromisoformat(meta["ts"])}
    except (KeyError, TypeError, ValueError):
        return meta


@router.get("/monitor/json/{sid}")
def snapshot_page(request: Request, sid: str):
    """Podglad pliku JSON przed i po operacji (roznice, obie wersje, obok siebie)."""
    s = snapshots.get(sid)
    if not s:
        raise HTTPException(404, "Nie ma takiej migawki (mogla zostac usunieta jako starsza)")
    diff, cut = snapshots.diff_lines(s["before"], s["after"])
    s = _when(s)
    return templates.TemplateResponse(request, "monitor_json.html", {"s": s, "ts": s["ts"], "diff": diff,
                                                                     "diff_cut": cut})


@router.get("/monitor/json/{sid}/{which}")
def snapshot_raw(sid: str, which: str):
    text = snapshots.raw(sid, which)
    if text is None:
        raise HTTPException(404, "Brak tej wersji pliku")
    return Response(text, media_type="application/json; charset=utf-8")
