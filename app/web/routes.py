import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine, get_session
from app.exporters.kwiatownik import source_entry, with_sources
from app.models import Item, ItemKind, ItemStatus, Job, JobStatus, SitePattern, Source
from app.scrapers.profile import ScraperProfile
from app.web.templating import templates
from app.worker import actions
from app.worker.jobs import job_log

router = APIRouter()


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


# ------------------------------------------------------------------ pulpit
@router.get("/")
def dashboard(request: Request, session: Session = Depends(get_session)):
    rows = session.exec(select(Item.kind, Item.status, func.count()).group_by(Item.kind, Item.status)).all()
    counts = {k.value: {s.value: 0 for s in ItemStatus} for k in (ItemKind.recipe, ItemKind.fact)}
    for kind, status, n in rows:
        if kind.value in counts:
            counts[kind.value][status.value] = n
    jobs = session.exec(select(Job).order_by(Job.id.desc()).limit(8)).all()
    sources = session.exec(select(func.count()).select_from(Source)).one()
    dups = session.exec(select(func.count()).select_from(Item).where(col(Item.duplicate_of).is_not(None))).one()
    runner = request.app.state.jobs
    return templates.TemplateResponse(request, "dashboard.html", {
        "counts": counts, "jobs": jobs, "sources": sources, "dups": dups,
        "running": len(runner.active), "waiting": runner.waiting, "limit": runner.max_concurrent})


# ------------------------------------------------------------------ zrodla
@router.get("/sources")
def sources_list(request: Request, session: Session = Depends(get_session)):
    sources = session.exec(select(Source).order_by(Source.id)).all()
    rows = session.exec(select(Item.source_id, Item.kind, func.count()).group_by(Item.source_id, Item.kind)).all()
    counts: dict[int, dict[str, int]] = {}
    for sid, kind, n in rows:
        counts.setdefault(sid, {})[kind.value] = n
    vpn = {s.country for s in sources if s.country and settings.proxy_for(s.country)}
    return templates.TemplateResponse(request, "sources.html", {"sources": sources, "counts": counts, "vpn": vpn})


@router.post("/sources/scan-all")
def sources_scan_all(request: Request):
    actions.scan_all(request.app.state)
    return _redirect("/jobs")


@router.post("/sources")
def sources_add(name: str = Form(...), base_url: str = Form(...), language: str = Form("en"),
                keywords: str = Form(""), max_pages: int = Form(50), max_depth: int = Form(3),
                country: str = Form(""), session: Session = Depends(get_session)):
    url = base_url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    session.add(Source(name=name.strip(), base_url=url, language=language.strip() or "en",
                       keywords=keywords.strip() or None, max_pages=max_pages, max_depth=max_depth,
                       country=country.strip().lower() or None))
    session.commit()
    return _redirect("/sources")


@router.get("/sources/{sid}")
def source_detail(sid: int, request: Request, session: Session = Depends(get_session)):
    src = session.get(Source, sid) or _404()
    patterns = session.exec(select(SitePattern).where(SitePattern.source_id == sid)
                            .order_by(col(SitePattern.hits).desc(), col(SitePattern.misses).desc())).all()
    jobs = session.exec(select(Job).where(Job.source_id == sid).order_by(Job.id.desc()).limit(10)).all()
    recipes = session.exec(select(func.count()).select_from(Item).where(
        Item.source_id == sid, Item.kind == ItemKind.recipe)).one()
    profile = ScraperProfile.from_json(src.scraper_profile)
    return templates.TemplateResponse(request, "source.html", {
        "src": src, "patterns": patterns, "jobs": jobs, "profile": profile, "recipes": recipes,
        "profile_json": json.dumps(json.loads(src.scraper_profile), ensure_ascii=False, indent=2)
        if src.scraper_profile else None})


@router.post("/sources/{sid}/edit")
def source_edit(sid: int, keywords: str = Form(""), max_pages: int = Form(50), max_depth: int = Form(3),
                active: bool = Form(False), auto_scan: bool = Form(False), country: str = Form(""),
                session: Session = Depends(get_session)):
    src = session.get(Source, sid) or _404()
    src.keywords, src.max_pages, src.max_depth, src.active = keywords.strip() or None, max_pages, max_depth, active
    src.auto_scan = auto_scan
    src.country = country.strip().lower() or None
    session.add(src)
    session.commit()
    return _redirect(f"/sources/{sid}")


@router.post("/sources/{sid}/forget-patterns")
def source_forget_patterns(sid: int, session: Session = Depends(get_session)):
    for p in session.exec(select(SitePattern).where(SitePattern.source_id == sid)).all():
        session.delete(p)
    session.commit()
    return _redirect(f"/sources/{sid}")


@router.post("/sources/{sid}/scan")
async def source_scan(sid: int, request: Request, max_pages: str = Form(""),
                      session: Session = Depends(get_session)):
    # async: zadanie w tle musi powstac w petli zdarzen aplikacji (nie w watku)
    session.get(Source, sid) or _404()
    limit = int(max_pages) if max_pages.strip().isdigit() else None
    return _redirect(f"/jobs/{actions.start_scan(request.app.state, sid, limit)}")


@router.post("/sources/{sid}/build-profile")
async def source_build_profile(sid: int, request: Request, session: Session = Depends(get_session)):
    session.get(Source, sid) or _404()
    return _redirect(f"/jobs/{actions.start_build_profile(request.app.state, sid)}")


@router.post("/sources/{sid}/fetch-list")
async def source_fetch_list(sid: int, request: Request, urls: str = Form(""),
                            session: Session = Depends(get_session)):
    session.get(Source, sid) or _404()
    lst = list(dict.fromkeys(u for u in urls.split() if u.startswith("http")))
    if not lst:
        return _redirect(f"/sources/{sid}")
    return _redirect(f"/jobs/{actions.start_fetch_list(request.app.state, sid, lst)}")


@router.post("/sources/{sid}/profile/delete")
def source_profile_delete(sid: int, session: Session = Depends(get_session)):
    src = session.get(Source, sid) or _404()
    src.scraper_profile, src.profile_status = None, None
    session.add(src)
    session.commit()
    return _redirect(f"/sources/{sid}")


# ------------------------------------------------------------------ zadania
@router.get("/jobs")
def jobs_list(request: Request, session: Session = Depends(get_session)):
    jobs = session.exec(select(Job).order_by(Job.id.desc()).limit(50)).all()
    names = {s.id: s.name for s in session.exec(select(Source)).all()}
    return templates.TemplateResponse(request, "jobs.html", {"jobs": jobs, "names": names})


@router.get("/jobs/{jid}")
def job_detail(jid: int, request: Request, session: Session = Depends(get_session)):
    job = session.get(Job, jid) or _404()
    src = session.get(Source, job.source_id) if job.source_id else None
    return templates.TemplateResponse(request, "job.html", {"job": job, "src": src, **_job_ctx(job)})


@router.get("/jobs/{jid}/panel")
def job_panel(jid: int, request: Request, session: Session = Depends(get_session)):
    job = session.get(Job, jid) or _404()
    return templates.TemplateResponse(request, "job_panel.html", {"job": job, **_job_ctx(job)})


@router.post("/jobs/{jid}/retry")
async def job_retry(jid: int, request: Request):
    new = actions.retry_job(request.app.state, jid)
    return _redirect(f"/jobs/{new or jid}")


@router.post("/jobs/{jid}/stop")
async def job_stop(jid: int, request: Request):
    # async: request_stop planuje wymuszone przerwanie w petli zdarzen aplikacji
    if request.app.state.jobs.request_stop(jid):
        job_log(jid, "Zatrzymywanie... (biezacy krok ma 15 s na zakonczenie, potem przerwanie)")
    else:
        # zadanie "wisi" w bazie, ale nie dziala w tej instancji (np. po awarii) - po prostu je zamknij
        with Session(engine) as s:
            job = s.get(Job, jid)
            if job and job.status in (JobStatus.running, JobStatus.queued):
                job.status = JobStatus.cancelled
                job.finished_at = datetime.now(timezone.utc)
                job.log = (job.log or "") + "Zatrzymano (zadanie nie dzialalo - osierocone po restarcie)\n"
                s.add(job)
                s.commit()
    back = request.headers.get("referer") or f"/jobs/{jid}"
    return _redirect(back if "/monitor" in back else f"/jobs/{jid}")


def _job_ctx(job: Job) -> dict:
    stats = json.loads(job.stats_json) if job.stats_json else None
    pct = int(100 * (job.progress or 0) / job.total) if job.total else 0
    frontier = len(json.loads(job.state_json).get("frontier") or []) if job.state_json else 0
    return {"stats": stats, "pct": min(pct, 100), "live": job.status in (JobStatus.running, JobStatus.queued),
            "frontier": frontier}


# ------------------------------------------------------------------ zebrane strony
def _items_stmt(kind: str, status: str, source_id: int | None, q: str, dups: str):
    stmt = select(Item)
    if not dups:  # domyslnie tylko "glowne" przepisy - duplikaty sa ich dodatkowymi zrodlami
        stmt = stmt.where(col(Item.duplicate_of).is_(None))
    if kind == "useful":
        stmt = stmt.where(col(Item.kind).in_([ItemKind.recipe, ItemKind.fact]))
    elif kind:
        stmt = stmt.where(Item.kind == ItemKind(kind))
    if status:
        stmt = stmt.where(Item.status == ItemStatus(status))
    if source_id:
        stmt = stmt.where(Item.source_id == source_id)
    if q:
        stmt = stmt.where(col(Item.title).contains(q) | col(Item.url).contains(q))
    return stmt


@router.get("/items")
def items_list(request: Request, kind: str = "useful", status: str = "", source_id: str = "",
               q: str = "", dups: str = "", page: int = 1, session: Session = Depends(get_session)):
    source_id = int(source_id) if source_id.isdigit() else None
    stmt = _items_stmt(kind, status, source_id, q, dups)
    total = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    per_page = 50
    items = session.exec(stmt.order_by(col(Item.id).desc()).offset((page - 1) * per_page).limit(per_page + 1)).all()
    sources = session.exec(select(Source).order_by(Source.name)).all()
    ids = [i.id for i in items[:per_page]]
    extra = dict(session.exec(select(Item.duplicate_of, func.count()).where(col(Item.duplicate_of).in_(ids))
                              .group_by(Item.duplicate_of)).all()) if ids else {}
    return templates.TemplateResponse(request, "items.html", {
        "items": items[:per_page], "has_next": len(items) > per_page, "page": page, "extra": extra, "total": total,
        "sources": sources, "names": {s.id: s.name for s in sources},
        "f": {"kind": kind, "status": status, "source_id": source_id or "", "q": q, "dups": dups},
        "kinds": [k.value for k in ItemKind], "statuses": [s.value for s in ItemStatus]})


BULK_LABELS = {"scan": "skanowanie", "translate": "tlumaczenie", "enrich": "opracowanie", "approve": "zatwierdzenie", "reject": "odrzucenie",
               "raw": "przywrocenie"}


@router.post("/items/bulk")
async def items_bulk(request: Request, action: str = Form(...), ids: list[int] = Form([]),
                     all_matching: bool = Form(False), kind: str = Form("useful"), status: str = Form(""),
                     source_id: str = Form(""), q: str = Form(""), dups: str = Form(""), back: str = Form("/items"),
                     session: Session = Depends(get_session)):
    """Akcje na wielu zaznaczonych stronach naraz (albo na wszystkich pasujacych do filtra)."""
    if all_matching:
        sid = int(source_id) if source_id.isdigit() else None
        ids = [it.id for it in session.exec(_items_stmt(kind, status, sid, q, dups)).all()]
    items = session.exec(select(Item).where(col(Item.id).in_(ids))).all() if ids else []
    if not items:
        return _redirect(back)
    state = request.app.state
    if action == "scan":
        by_source: dict[int, list[str]] = {}
        for it in items:
            if it.source_id:
                by_source.setdefault(it.source_id, []).append(it.url)
        jobs = [actions.start_scan_pages(state, sid, urls) for sid, urls in by_source.items()]
        return _redirect(f"/jobs/{jobs[0]}" if len(jobs) == 1 else "/jobs")
    if action == "translate":
        rec = [it.id for it in items if it.kind == ItemKind.recipe]
        return _redirect(f"/jobs/{actions.start_translate(state, item_ids=rec)}" if rec else back)
    if action == "enrich":
        rec = [it.id for it in items if it.kind == ItemKind.recipe and it.data_json]
        return _redirect(f"/jobs/{actions.start_enrich(state, item_ids=rec)}" if rec else back)
    new_status = {"approve": ItemStatus.approved, "reject": ItemStatus.skipped, "raw": ItemStatus.raw}.get(action)
    if new_status:
        for it in items:
            if action == "approve" and not it.data_json:
                continue  # zatwierdzamy tylko przetlumaczone
            it.status = new_status
            session.add(it)
        session.commit()
    return _redirect(back)


@router.get("/items/{iid}")
def item_detail(iid: int, request: Request, session: Session = Depends(get_session)):
    item = session.get(Item, iid) or _404()
    src = session.get(Source, item.source_id) if item.source_id else None
    recipe = json.loads(item.structured_json) if item.structured_json else None
    primary = session.get(Item, item.duplicate_of) if item.duplicate_of else item
    group = [primary] + list(session.exec(select(Item).where(Item.duplicate_of == primary.id)
                                          .order_by(Item.id)).all())
    names = {s.id: s.name for s in session.exec(select(Source)).all()}
    zrodla = with_sources({}, [source_entry(names.get(g.source_id, "?"), g.url, g.title, g.language, g.created_at)
                               for g in group])["zrodla"]
    polish = json.loads(item.data_json) if item.data_json else None
    return templates.TemplateResponse(request, "item.html", {
        "item": item, "src": src, "recipe": recipe, "primary": primary, "group": group, "names": names,
        "polish": polish, "polish_json": json.dumps(polish, ensure_ascii=False, indent=2) if polish else None,
        "zrodla_json": json.dumps({"zrodla": zrodla}, ensure_ascii=False, indent=2)})


@router.post("/items/{iid}/set")
def item_set(iid: int, status: str = Form(""), kind: str = Form(""), duplicate_of: str = Form(""),
             session: Session = Depends(get_session)):
    item = session.get(Item, iid) or _404()
    if status:
        item.status = ItemStatus(status)
    if kind:
        item.kind = ItemKind(kind)
    if duplicate_of == "none":
        item.duplicate_of = None
    elif duplicate_of.strip().lstrip("#").isdigit():
        target = session.get(Item, int(duplicate_of.strip().lstrip("#")))
        if target and target.id != item.id:
            primary_id = target.duplicate_of or target.id
            item.duplicate_of = primary_id
            for child in session.exec(select(Item).where(Item.duplicate_of == item.id)).all():
                child.duplicate_of = primary_id  # przepnij "dzieci" na nowy glowny
                session.add(child)
    session.add(item)
    session.commit()
    return _redirect(f"/items/{iid}")


def _404():
    raise HTTPException(404, "Nie znaleziono")
