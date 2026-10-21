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


# czytelne nazwy zadan i grupy (kolor paska karty w widoku "Na zywo")
KIND_LABELS = {
    "scan": "Skan strony", "harvest": "Pełne pobieranie strony", "scan_pages": "Czytanie wybranych stron",
    "fetch_list": "Czytanie wybranych stron", "build_profile": "Uczenie się nowej strony", "discover": "Szukanie nowych stron",
    "plant_research": "Zbieranie wiedzy o roślinach", "plant_apply": "Zapis wiedzy do Kwiatownika",
    "plant_merge": "Scalanie wiedzy z rozdziałami", "plant_place": "Rozmieszczanie wiedzy w rozdziałach",
    "plant_organize": "Porządkowanie wiedzy", "plant_photos": "Zdjęcia z Wikimedia Commons", "plants_sync": "Rejestr roślin",
    "translate": "Tłumaczenie przepisów", "enrich": "Opracowanie przepisów", "export": "Eksport przepisów",
    "import_k1": "Import ze starego Kwiatownika", "leads_extract": "Wzmianki o przepisach", "leads_search": "Szukanie przepisów",
    "plant_magic": "Magia roślin",
    "nn_train": "Trening sieci neuronowych", "nn_init": "Pobieranie enkodera", "agent_audit": "Audyt bazy wiedzy",
    "agent_dispatch": "Zleceniodawca", "agent_deploy": "Wdrożeniowiec", "fb_post": "Post na fanpage",
    "fb_ideas": "Pomysły na posty", "fb_review": "Ocena postów", "ollama_pull": "Pobieranie modelu",
}
GROUPS = (("zrodla", "Źródła", ("scan", "harvest", "scan_pages", "fetch_list", "build_profile", "discover")),
          ("wiedza", "Wiedza o roślinach", ("plant_",)),
          ("przepisy", "Przepisy", ("translate", "enrich", "export", "import_k1", "leads_")),
          ("sieci", "Sieci neuronowe", ("nn_",)),
          ("agenci", "Agenci", ("agent_", "fb_", "ollama_pull")))
STALE_S = 600          # karta ostrzega, gdy zadanie nie zrobilo nowego kroku od 10 min


def job_group(kind: str) -> tuple[str, str]:
    base = kind.split(":", 1)[0]
    for key, label, prefixes in GROUPS:
        if any(base == p or (p.endswith("_") and base.startswith(p)) for p in prefixes):
            return key, label
    return "inne", "Inne"


def job_subject(j: Job, names: dict, plants: dict) -> str:
    """O czym jest zadanie: zrodlo, rosliny (pierwsze 3 nazwy), kraj."""
    if j.source_id and names.get(j.source_id):
        return names[j.source_id]
    try:
        p = json.loads(j.params_json) if j.params_json else {}
    except ValueError:
        p = {}
    ids = p.get("plant_ids") or []
    if ids:
        shown = ", ".join(plants.get(i, i) for i in ids[:3])
        return shown + (f" i {len(ids) - 3} innych" if len(ids) > 3 else "")
    if ":" in j.kind:
        return j.kind.split(":", 1)[1].upper()
    return ""


def _ctx(request: Request) -> dict:
    now = datetime.now(timezone.utc)
    from app.models import Plant
    with Session(engine) as s:
        names = {x.id: x.name for x in s.exec(select(Source)).all()}
        plants = {p.id: p.nazwa_pl for p in s.exec(select(Plant)).all()}
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
        act = activity.get(j.id)
        group, group_label = job_group(j.kind)
        recent = []
        for ln in lines[-5:]:
            m = LOG_LINE.match(ln)
            recent.append((m.group(1), m.group(2)) if m else ("", ln))
        cards.append({"job": j, "name": names.get(j.source_id, ""), "elapsed": _fmt_s(elapsed), "eta": _fmt_s(eta),
                      "pct": min(100, int(100 * (j.progress or 0) / j.total)) if j.total else None,
                      "act": act, "last": lines[-4:], "recent": recent[::-1],
                      "title": KIND_LABELS.get(j.kind.split(":", 1)[0], j.kind), "group": group,
                      "group_label": group_label, "subject": job_subject(j, names, plants),
                      "stale": _fmt_s(act["age"]) if act and act["age"] > STALE_S else None,
                      "stats": json.loads(j.stats_json) if j.stats_json else None})
    feed.sort(key=lambda x: x["time"], reverse=True)

    router_ = getattr(request.app.state, "llm", None)
    llm_inflight = dict(router_.inflight) if router_ else {}
    llm_recent = [dict(r, ago=round(time.time() - r["ts"])) for r in list(router_.recent)[::-1][:10]] if router_ else []
    from app.nn.runtime import brain
    nn_recent = [dict(r, ago=round(time.time() - r["ts"])) for r in list(brain.recent)[::-1][:10]]
    runner = request.app.state.jobs
    return {
        "nn_recent": nn_recent,
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
