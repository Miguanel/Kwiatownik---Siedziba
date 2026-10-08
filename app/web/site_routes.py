"""Widok "Strona" (/site): statystyki Kwiatownika z backendu na Render (kopia w Siedzibie, tabela SiteStat)."""
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, func, select

from app.agents import backend_sync
from app.config import settings
from app.db import engine
from app.models import Plant, SiteStat
from app.web.templating import templates

router = APIRouter()
TZ = ZoneInfo(settings.timezone)
KIND_LABELS = {"page": "Odsłony stron", "plant_view": "Odsłony roślin", "plant_section": "Otwarte rozdziały",
               "plantid_click": "Rozpoznawanie ze zdjęcia (plant.id)", "plantid_result": "Rozpoznane rośliny",
               "plantid_unknown": "Rozpoznane, a brak w Kwiatowniku", "recipe_open": "Otwarte przepisy",
               "source_click": "Kliknięte źródła", "search": "Wyszukiwania", "visitors": "Goście",
               "papyrus": "Kliknięcia w papirusie"}


def _day(n: int) -> str:
    return (datetime.now(TZ) - timedelta(days=n)).strftime("%Y-%m-%d")


def site_context(days: int = 30) -> dict:
    since, week, today = _day(days - 1), _day(6), _day(0)
    with Session(engine) as s:
        rows = s.exec(select(SiteStat).where(SiteStat.day >= since, SiteStat.day != "archiwum")).all()
        all_time = dict(s.exec(select(SiteStat.kind, func.sum(SiteStat.n)).group_by(SiteStat.kind)).all())
        names = {p.id: p.nazwa_pl for p in s.exec(select(Plant)).all()}
    per_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tiles = defaultdict(lambda: {"dzis": 0, "tydzien": 0, "okres": 0})
    for r in rows:
        per_kind[r.kind][r.key] += r.n
        per_day[r.day][r.kind] += r.n
        t = tiles[r.kind]
        t["okres"] += r.n
        if r.day >= week:
            t["tydzien"] += r.n
        if r.day == today:
            t["dzis"] += r.n

    def top(kind: str, n: int = 12) -> list[tuple[str, int]]:
        return sorted(((k, v) for k, v in per_kind.get(kind, {}).items() if k), key=lambda x: -x[1])[:n]

    sections = defaultdict(int)
    for key, v in per_kind.get("plant_section", {}).items():
        sections[key.split(":", 1)[-1]] += v
    days_list = [_day(i) for i in range(days - 1, -1, -1)]
    series = [{"day": d, "views": per_day[d].get("plant_view", 0), "plantid": per_day[d].get("plantid_click", 0),
               "visitors": per_day[d].get("visitors", 0)} for d in days_list]
    peak = max([1] + [x["views"] for x in series])
    return {"days": days, "tiles": {k: dict(v) for k, v in tiles.items()}, "all_time": all_time,
            "top_plants": [(k, names.get(k, k), v) for k, v in top("plant_view", 15)],
            "top_sections": sorted(sections.items(), key=lambda x: -x[1])[:12],
            "unknown": top("plantid_unknown", 15), "recognized": top("plantid_result", 10),
            "sources": top("source_click", 12), "searches": top("search", 15), "recipes": top("recipe_open", 10),
            "pages": top("page", 10), "series": series, "peak": peak, "labels": KIND_LABELS,
            "sync": backend_sync.load_state(), "configured": backend_sync.configured(),
            "backend_url": settings.backend_url, "sync_minutes": settings.backend_sync_minutes, "msg": ""}


@router.get("/site")
async def site_page(request: Request, days: int = 30):
    ctx = site_context(max(7, min(days, 365)))
    ctx["msg"] = request.query_params.get("msg", "")
    return templates.TemplateResponse(request, "site.html", ctx)


@router.post("/site/sync")
async def site_sync():
    out = await backend_sync.sync()
    msg = "Zsynchronizowano" if out.get("ok") else f"Blad: {out.get('blad') or out.get('powod')}"
    return RedirectResponse("/site?msg=" + msg.replace(" ", "+"), status_code=303)
