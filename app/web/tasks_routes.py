from fastapi import APIRouter, Form, Request
from sqlmodel import Session, col, select

from app.db import engine
from app.models import Job, JobStatus
from app.web.templating import templates
from app.worker.suggestions import CATEGORIES, build_groups, run_task

router = APIRouter()


def _ctx(cat: str = "", message: str | None = None) -> dict:
    groups = build_groups()
    counts = {c: sum(g.total for g in groups if g.category == c) for c in CATEGORIES}
    if cat:
        groups = [g for g in groups if g.category == cat]
    return {"groups": groups, "categories": CATEGORIES, "cat": cat, "counts": counts, "message": message}


@router.get("/tasks")
def tasks_page(request: Request, cat: str = ""):
    return templates.TemplateResponse(request, "tasks.html", _ctx(cat))


@router.get("/tasks/panel")
def tasks_panel(request: Request, cat: str = ""):
    return templates.TemplateResponse(request, "tasks_panel.html", _ctx(cat))


@router.post("/tasks/run")
async def tasks_run(request: Request, group: str = Form(...), action: str = Form(...), ids: list[str] = Form([]),
                    all_items: bool = Form(False), cat: str = Form("")):
    if all_items:  # "Wykonaj dla wszystkich" - wszystkie pozycje grupy (takze spoza widocznej listy)
        g = next((g for g in build_groups(limit=10_000) if g.key == group), None)
        ids = [i.id for i in g.items] if g else []
    msg = run_task(request.app.state, group, action, ids)
    return templates.TemplateResponse(request, "tasks_panel.html", _ctx(cat, msg))


@router.get("/tasks/queue")
def tasks_queue(request: Request):
    """Mini-podglad kolejki (odswiezany co kilka sekund)."""
    with Session(engine) as s:
        jobs = s.exec(select(Job).where(col(Job.status).in_([JobStatus.running, JobStatus.queued]))
                      .order_by(Job.id)).all()
    runner = request.app.state.jobs
    return templates.TemplateResponse(request, "tasks_queue.html",
                                      {"jobs": jobs, "limit": runner.max_concurrent})
