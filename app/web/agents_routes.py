"""Widok "Agenci" (/agents): audytor wiedzy, zleceniodawca i planista ciaglosci - stan, raporty i historia."""
import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.agents import store, watchdog
from app.agents.audit import CRITERIA, FLAG_LABELS, plant_history
from app.agents.planner import run_planner
from app.agents.tasks import KINDS, STATUS_LABELS, order, params, production_jobs, sync_tasks
from app.config import settings
from app.db import engine
from app.models import AgentRun, AgentTask, Job, JobStatus
from app.web.templating import templates
from app.worker import actions

router = APIRouter()
PAGE = 30
AGENT_INFO = {
    "wdrozeniowiec": f"Publikuje nowa wersje Kwiatownika: gdy Siedziba nazbiera co najmniej {settings.deploy_min_points} "
                     f"informacji albo {settings.deploy_min_recipes} przepisow (albo cokolwiek po {settings.deploy_max_hours} h), "
                     "robi commit danych Kwiatownika2 z wpisem kroniki i push - Render buduje strone sam. "
                     f"Najczesciej co {settings.deploy_min_hours} h.",
    "audytor": "Skanuje baze wiedzy Kwiatownika (pliki roslin, przepisy) i pisze raport oceny: kompletnosc, wiedza "
               "trudno dostepna, opowiesci, przepisy niekulinarne, zrodla. Metryki liczy kod, ocene dopisuje ekspert LLM. "
               "Konczy sie lista zalecen.",
    "zleceniodawca": "Bierze zalecenia ostatniego audytu i zleca je Siedzibie (kolejka) - od najpilniejszych, "
                     f"najwyzej {settings.agents_dispatch_max} na przebieg, bez powtorek z ostatnich "
                     f"{settings.agents_reorder_days} dni.",
    "planista": f"Co {settings.agents_check_minutes} min sprawdza, czy Siedziba nad czyms pracuje. Zatrzymuje zadania "
                f"zawieszone (bez znaku zycia od {settings.agents_stall_minutes} min), pilnuje obciazenia modeli LLM, "
                "proponuje kolejne zadanie; w trybie automatycznym, gdy kolejka stoi, sam zleca JEDNO.",
}
RESULT_KEYS = ("plants", "facts", "verified", "applied", "rejected", "sources", "found", "new_sites", "recipes",
               "translated", "exported", "merged", "photos", "pages", "saved", "items")


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _result(t: AgentTask) -> str:
    if not t.result_json:
        return ""
    try:
        data = json.loads(t.result_json)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    keys = [k for k in RESULT_KEYS if isinstance(data.get(k), (int, float))] or \
           [k for k, v in data.items() if isinstance(v, (int, float))]
    return ", ".join(f"{k}={data[k]}" for k in keys[:6])


def _task_rows(rows: list[AgentTask], s: Session) -> list[dict]:
    jobs = {j.id: j for j in s.exec(select(Job).where(col(Job.id).in_([t.job_id for t in rows if t.job_id]))).all()}
    out = []
    for t in rows:
        job = jobs.get(t.job_id)
        label = KINDS[t.kind].label if t.kind in KINDS else t.kind
        out.append({"t": t, "status": STATUS_LABELS.get(t.status, t.status), "kind": label,
                    "title": (t.title or "").removeprefix(label + ": ").removeprefix(label),
                    "job": job, "progress": f"{job.progress or 0}/{job.total or '?'}" if job else "",
                    "result": _result(t), "p": params(t)})
    return out


def _trend(points: list[tuple[int, float]]) -> dict | None:
    """Wspolrzedne wykresu oceny kolejnych audytow (SVG 0-100)."""
    if len(points) < 2:
        return None
    n = len(points) - 1
    pts = [(round(i * 100 / n, 2), round(100 - v, 2), rid, v) for i, (rid, v) in enumerate(points)]
    return {"line": " ".join(f"{x},{y}" for x, y, _, _ in pts), "pts": pts}


def _deploy_ready() -> str | None:
    """Powod, dla ktorego wdrozeniowiec nie moze publikowac (None = gotowy). Bez liczenia zmian - to robi przycisk."""
    from app.agents.deploy import readiness
    try:
        return readiness()[1]
    except Exception as exc:  # noqa: BLE001
        return str(exc)[:200]


def _overview(request: Request) -> dict:
    sync_tasks()
    with Session(engine) as s:
        last = {a: s.exec(select(AgentRun).where(AgentRun.agent == a).order_by(col(AgentRun.id).desc())).first()
                for a in store.AGENTS}
        running = {a: bool(s.exec(select(Job).where(Job.kind == k, col(Job.status).in_([JobStatus.queued, JobStatus.running]))).first())
                   for a, k in (("audytor", "agent_audit"), ("zleceniodawca", "agent_dispatch"),
                                ("wdrozeniowiec", "agent_deploy"))}
        busy = production_jobs(s)
        audits = s.exec(select(AgentRun).where(AgentRun.agent == "audytor", AgentRun.status == "done")
                        .order_by(col(AgentRun.id).desc()).limit(20)).all()
        counts = dict(s.exec(select(AgentTask.status, func.count()).group_by(AgentTask.status)).all())
        tasks = _task_rows(list(s.exec(select(AgentTask).where(col(AgentTask.ordered_at).is_not(None))
                                       .order_by(col(AgentTask.ordered_at).desc()).limit(15)).all()), s)
        proposed = _task_rows(list(s.exec(select(AgentTask).where(AgentTask.status == "proposed")
                                          .order_by(AgentTask.priority, col(AgentTask.id)).limit(20)).all()), s)
        n_runs = dict(s.exec(select(AgentRun.agent, func.count()).group_by(AgentRun.agent)).all())
        guard_runs = s.exec(select(AgentRun).where(AgentRun.agent == "planista", col(AgentRun.report_json).contains('"straznik": [{'))
                            .order_by(col(AgentRun.id).desc()).limit(8)).all()
    guard = [{**e, "run": r.id, "at": r.created_at} for r in guard_runs for e in (store.report(r).get("straznik") or [])][:10]
    audit = audits[0] if audits else None
    rep = store.report(audit)
    plan = store.report(last["planista"])
    return {"request": request, "agents": store.AGENTS, "info": AGENT_INFO, "last": last, "running": running,
            "auto": store.runtime().get("auto", True), "busy": busy, "audit": audit, "rep": rep,
            "criteria": CRITERIA, "flag_labels": FLAG_LABELS, "trend": _trend([(a.id, a.score or 0) for a in reversed(audits)]),
            "counts": counts, "status_labels": STATUS_LABELS, "tasks": tasks, "proposed": proposed, "plan": plan,
            "n_runs": n_runs, "has_llm": getattr(request.app.state, "llm", None) is not None,
            "check_minutes": settings.agents_check_minutes, "enabled": settings.agents_enabled,
            "quiet": watchdog.job_silence(busy), "stall": settings.agents_stall_minutes, "guard": guard,
            "load": watchdog.llm_load(request.app.state), "deploy_ready": _deploy_ready(),
            "msg": request.query_params.get("msg", "")}


@router.get("/agents")
async def agents_page(request: Request):
    return templates.TemplateResponse(request, "agents.html", _overview(request))


@router.get("/agents/status")
async def agents_status(request: Request):
    """Pasek stanu (odswiezany co kilka sekund): kolejka, agenci w toku, liczniki zadan."""
    return templates.TemplateResponse(request, "agents_status.html", _overview(request))


@router.post("/agents/auto")
async def agents_auto(value: str = Form("on")):
    store.set_runtime(auto=value == "on")
    return _redirect("/agents?msg=" + ("Tryb automatyczny wlaczony" if value == "on" else "Tryb automatyczny wylaczony"))


@router.post("/agents/run/{agent}")
async def agents_run(request: Request, agent: str, mode: str = Form("check"), pick: str = Form("")):
    state = request.app.state
    if agent == "audytor":
        jid = actions.start_agent_audit(state)
        return _redirect(f"/agents?msg=Audyt+uruchomiony+(zadanie+%23{jid})")
    if agent == "zleceniodawca":
        jid = actions.start_agent_dispatch(state)
        return _redirect(f"/agents?msg=Zleceniodawca+uruchomiony+(zadanie+%23{jid})")
    if agent == "wdrozeniowiec":
        jid = actions.start_agent_deploy(state, force=mode == "force", dry_run=mode != "force")
        return _redirect(f"/agents?msg=Wdrozeniowiec+-+zadanie+%23{jid}")
    if agent == "planista":
        rid = run_planner(state, "reczny", execute=mode == "order", force=mode == "order", pick=pick or None)
        return _redirect(f"/agents/runs/{rid}")
    return _redirect("/agents")


@router.get("/agents/runs")
async def agents_runs(request: Request, agent: str = "", page: int = 1):
    page = max(1, page)
    with Session(engine) as s:
        stmt = select(AgentRun)
        cnt = select(func.count()).select_from(AgentRun)
        if agent:
            stmt, cnt = stmt.where(AgentRun.agent == agent), cnt.where(AgentRun.agent == agent)
        total = s.exec(cnt).one()
        runs = s.exec(stmt.order_by(col(AgentRun.id).desc()).offset((page - 1) * PAGE).limit(PAGE)).all()
        ordered = dict(s.exec(select(AgentTask.dispatch_run_id, func.count()).where(
            col(AgentTask.dispatch_run_id).in_([r.id for r in runs])).group_by(AgentTask.dispatch_run_id)).all())
        proposed = dict(s.exec(select(AgentTask.run_id, func.count()).where(
            col(AgentTask.run_id).in_([r.id for r in runs]), AgentTask.agent == "audytor").group_by(AgentTask.run_id)).all())
    return templates.TemplateResponse(request, "agents_runs.html", {
        "request": request, "runs": runs, "agent": agent, "agents": store.AGENTS, "page": page,
        "pages": max(1, -(-total // PAGE)), "total": total, "ordered": ordered, "proposed": proposed})


@router.get("/agents/runs/{run_id}")
async def agents_run_detail(request: Request, run_id: int):
    sync_tasks()
    with Session(engine) as s:
        run = s.get(AgentRun, run_id)
        if not run:
            return _redirect("/agents/runs")
        proposed = _task_rows(list(s.exec(select(AgentTask).where(AgentTask.run_id == run_id)
                                          .order_by(AgentTask.priority, col(AgentTask.id))).all()), s)
        ordered = _task_rows(list(s.exec(select(AgentTask).where(AgentTask.dispatch_run_id == run_id)
                                         .order_by(col(AgentTask.id))).all()), s)
        prev = s.exec(select(AgentRun).where(AgentRun.agent == run.agent, AgentRun.id < run_id)
                      .order_by(col(AgentRun.id).desc())).first()
        nxt = s.exec(select(AgentRun).where(AgentRun.agent == run.agent, AgentRun.id > run_id)
                     .order_by(AgentRun.id)).first()
    rep = store.report(run)
    return templates.TemplateResponse(request, "agents_run.html", {
        "request": request, "run": run, "rep": rep, "agents": store.AGENTS, "criteria": CRITERIA,
        "flag_labels": FLAG_LABELS, "proposed": proposed, "ordered": ordered, "prev": prev, "next": nxt,
        "kinds": KINDS, "status_labels": STATUS_LABELS, "plants_filter": request.query_params.get("braki", "")})


@router.get("/agents/tasks")
async def agents_tasks(request: Request, status: str = "", kind: str = "", agent: str = "", page: int = 1):
    page = max(1, page)
    sync_tasks()
    with Session(engine) as s:
        stmt = select(AgentTask)
        cnt = select(func.count()).select_from(AgentTask)
        for field, value in ((AgentTask.status, status), (AgentTask.kind, kind), (AgentTask.agent, agent)):
            if value:
                stmt, cnt = stmt.where(field == value), cnt.where(field == value)
        total = s.exec(cnt).one()
        rows = _task_rows(list(s.exec(stmt.order_by(col(AgentTask.id).desc()).offset((page - 1) * PAGE).limit(PAGE)).all()), s)
        counts = dict(s.exec(select(AgentTask.status, func.count()).group_by(AgentTask.status)).all())
        kinds = dict(s.exec(select(AgentTask.kind, func.count()).group_by(AgentTask.kind)).all())
    return templates.TemplateResponse(request, "agents_tasks.html", {
        "request": request, "rows": rows, "status": status, "kind": kind, "agent": agent, "page": page,
        "pages": max(1, -(-total // PAGE)), "total": total, "counts": counts, "kinds_count": kinds,
        "kinds": KINDS, "status_labels": STATUS_LABELS, "agents": store.AGENTS,
        "msg": request.query_params.get("msg", "")})


@router.post("/agents/tasks/{task_id}/order")
async def agents_task_order(request: Request, task_id: int, back: str = Form("/agents/tasks")):
    ok, msg = order(request.app.state, task_id)
    sep = "&" if "?" in back else "?"
    return _redirect(f"{back}{sep}msg=" + ("Zlecono+" + msg.replace(" ", "+").replace("#", "%23") if ok
                                         else "Nie+zlecono:+" + msg.replace(" ", "+").replace("#", "%23")))


@router.post("/agents/tasks/{task_id}/skip")
async def agents_task_skip(task_id: int, back: str = Form("/agents/tasks")):
    with Session(engine) as s:
        t = s.get(AgentTask, task_id)
        if t and t.status == "proposed":
            t.status, t.note = "skipped", "pominiete recznie"
            s.add(t)
            s.commit()
    return _redirect(back)


@router.get("/agents/plants/{plant_id}")
async def agents_plant(request: Request, plant_id: str):
    hist = plant_history(plant_id)
    with Session(engine) as s:
        rows = [t for t in s.exec(select(AgentTask).where(col(AgentTask.params_json).contains(f'"{plant_id}"'))
                                  .order_by(col(AgentTask.id).desc()).limit(60)).all()
                if plant_id in (params(t).get("rosliny") or [])]
        tasks = _task_rows(rows, s)
    name = plant_id
    if hist:
        audit = store.report(store.last_run("audytor"))
        name = next((r["nazwa"] for r in (audit.get("metryki") or {}).get("rosliny") or [] if r["id"] == plant_id), plant_id)
    return templates.TemplateResponse(request, "agents_plant.html", {
        "request": request, "pid": plant_id, "name": name, "hist": hist, "tasks": tasks, "criteria": CRITERIA,
        "flag_labels": FLAG_LABELS, "trend": _trend([(h["run"], h["wynik"]) for h in hist])})
