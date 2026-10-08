from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, select

from app.config import settings
from app.db import get_session
from app.discovery.countries import COUNTRIES
from app.discovery.finder import DiscoveryConfig
from app.models import DiscoveredSite, Job, SearchQuery
from app.web.templating import templates
from app.worker import actions
from app.worker.discover import add_site_as_source, add_verified_as_sources, discovery_counts, vpn_check

router = APIRouter()
STATUSES = ["verified", "maybe", "new", "unreachable", "rejected"]


@router.get("/discover")
def discover_page(request: Request, country: str = "", status: str = "verified", q: str = "",
                  session: Session = Depends(get_session)):
    stmt = select(DiscoveredSite)
    if country:
        stmt = stmt.where(DiscoveredSite.country == country)
    if status:
        stmt = stmt.where(DiscoveredSite.status == status)
    if q:
        stmt = stmt.where(col(DiscoveredSite.domain).contains(q) | col(DiscoveredSite.title).contains(q))
    sites = session.exec(stmt.order_by(col(DiscoveredSite.score).desc(), col(DiscoveredSite.id).desc()).limit(200)).all()
    queries = session.exec(select(SearchQuery).order_by(col(SearchQuery.id).desc()).limit(30)).all()
    jobs = session.exec(select(Job).where(col(Job.kind).startswith("discover")).order_by(col(Job.id).desc())
                        .limit(8)).all()
    return templates.TemplateResponse(request, "discover.html", {
        "sites": sites, "queries": queries, "jobs": jobs, "countries": COUNTRIES, "counts": discovery_counts(),
        "f": {"country": country, "status": status, "q": q}, "statuses": STATUSES,
        "proxies": {c: settings.proxy_for(c) for c in COUNTRIES},
        "engines": [n for n, on in (("SearXNG", settings.searxng_url.strip()), ("Brave API", settings.brave_api_key))
                    if on]})


@router.post("/discover/run")
async def discover_run(request: Request, countries: list[str] = Form(...), queries: int = Form(5),
                       pages: int = Form(1), max_new: int = Form(30), topic: str = Form(""),
                       own_queries: str = Form(""), auto_add: bool = Form(False)):
    own = [q.strip() for q in own_queries.splitlines() if q.strip()]
    for c in countries:
        if c in COUNTRIES:
            cfg = DiscoveryConfig(queries=queries, pages_per_query=pages, max_new_sites=max_new,
                                  topic=topic.strip() or None, extra_queries=own)
            actions.start_discover(request.app.state, c, cfg, auto_add)
    return RedirectResponse("/discover", status_code=303)


@router.post("/discover/{site_id}/add")
def discover_add(site_id: int):
    add_site_as_source(site_id)
    return RedirectResponse("/discover", status_code=303)


@router.post("/discover/{site_id}/status")
def discover_set_status(site_id: int, status: str = Form(...), session: Session = Depends(get_session)):
    site = session.get(DiscoveredSite, site_id)
    if site and status in STATUSES:
        site.status = status
        session.add(site)
        session.commit()
    return RedirectResponse("/discover", status_code=303)


@router.post("/discover/add-verified")
def discover_add_verified(country: str = Form("")):
    add_verified_as_sources(country or None)
    return RedirectResponse("/sources", status_code=303)


@router.get("/vpn/{country}")
async def vpn_test(request: Request, country: str):
    res = await vpn_check(country)
    return templates.TemplateResponse(request, "vpn_result.html", {"r": res, "country": country})
