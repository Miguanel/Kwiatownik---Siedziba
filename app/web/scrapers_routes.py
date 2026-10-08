"""Widok "Scrapery": zrodla z gotowym profilem scrapera i pelne pobieranie calej strony tym profilem."""
from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import Item, ItemKind, ItemStatus, Job, JobStatus, Source
from app.scrapers.profile import ScraperProfile
from app.web.templating import templates
from app.worker import actions

router = APIRouter()
ACTIVE = (JobStatus.queued, JobStatus.running)


def _rows() -> list[dict]:
    with Session(engine) as s:
        sources = s.exec(select(Source).where(col(Source.profile_status).is_not(None)).order_by(Source.name)).all()
        counts: dict[tuple[int, str], int] = {}
        for sid, kind, n in s.exec(select(Item.source_id, Item.kind, func.count()).where(
                Item.status != ItemStatus.skipped).group_by(Item.source_id, Item.kind)).all():
            counts[(sid, kind.value if hasattr(kind, "value") else kind)] = n
        running = {j.source_id: j for j in s.exec(select(Job).where(
            col(Job.status).in_(ACTIVE), col(Job.kind).in_(["harvest", "scan", "build_profile"]))).all()}
        last_harvest = {}
        for j in s.exec(select(Job).where(Job.kind == "harvest").order_by(col(Job.id))).all():
            last_harvest[j.source_id] = j
    rows = []
    for src in sources:
        prof = ScraperProfile.from_json(src.scraper_profile)
        v = (prof.validation if prof else {}) or {}
        rx = v.get("regex") or {}
        stats = (prof.stats if prof else {}) or {}
        rows.append({
            "src": src, "prof": prof, "ready": src.profile_status in ("ready", "degraded"),
            "coverage": v.get("coverage"), "precision": rx.get("precision"), "recall": rx.get("recall"),
            "facts_ok": bool((v.get("facts") or {}).get("ok")) or bool(prof and prof.fact_url_regex),
            "used": stats.get("used", 0), "failed": stats.get("failed", 0),
            "recipes": counts.get((src.id, ItemKind.recipe.value), 0),
            "facts": counts.get((src.id, ItemKind.fact.value), 0),
            "job": running.get(src.id), "last": last_harvest.get(src.id),
        })
    rows.sort(key=lambda r: (not r["ready"], not r["src"].active, r["src"].name.lower()))
    return rows


@router.get("/scrapers")
def scrapers_page(request: Request, msg: str | None = None):
    return templates.TemplateResponse(request, "scrapers.html", {
        "rows": _rows(), "default_pages": settings.harvest_max_pages, "msg": msg})


@router.post("/scrapers/harvest")
async def scrapers_harvest(request: Request, ids: list[int] = Form([]), max_pages: str = Form("")):
    """Pelne pobieranie zaznaczonych stron (kazda strona = osobne zadanie w kolejce)."""
    return _harvest(request, ids, max_pages)


def _harvest(request: Request, ids: list[int], max_pages: str):
    limit = int(max_pages) if str(max_pages).strip().isdigit() else None
    started = []
    for sid in ids:
        with Session(engine) as s:
            src = s.get(Source, sid)
            ok = src and src.active and src.profile_status in ("ready", "degraded")
        if ok:
            started.append(actions.start_harvest(request.app.state, sid, limit))
    if len(started) == 1:
        return RedirectResponse(f"/jobs/{started[0]}", status_code=303)
    return RedirectResponse(f"/scrapers?msg=Uruchomiono+{len(started)}+zadan+pobierania", status_code=303)


@router.post("/scrapers/{sid}/harvest")
async def scrapers_harvest_one(sid: int, request: Request, max_pages: str = Form("")):
    """Przycisk na karcie strony - tylko ta strona (zaznaczenia innych kart sa ignorowane)."""
    return _harvest(request, [sid], max_pages)


@router.post("/scrapers/{sid}/rebuild")
async def scrapers_rebuild(sid: int, request: Request):
    return RedirectResponse(f"/jobs/{actions.start_build_profile(request.app.state, sid)}", status_code=303)
