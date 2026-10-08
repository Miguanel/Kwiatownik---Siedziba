"""Widok "Rosliny": wiedza o roslinach zbierana przez Siedzibe (Wikipedia + inne kraje) ze zrodlami."""
import json
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.knowledge import plantfile, schema
from app.knowledge.sections import SECTION_TITLES_PL
from app.models import Job, JobStatus, Plant, PlantFact, PlantQuery
from app.web.templating import templates
from app.worker import actions

router = APIRouter()
STATUSES = ("new", "verified", "applied", "rejected")


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


_FILE_CACHE: dict[str, tuple[float, dict | None]] = {}


def _file(pid: str) -> dict | None:
    """Plik rosliny z Kwiatownika (pamiec podreczna wg czasu modyfikacji - widok listy czyta wszystkie)."""
    path = Path(settings.kwiatownik_plants_dir) / f"{pid}.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _FILE_CACHE.get(pid)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data, _ = plantfile.read_plant(settings.kwiatownik_plants_dir, pid)
    except (OSError, ValueError):
        data = None
    _FILE_CACHE[pid] = (mtime, data)
    return data


def _web_counts(s: Session) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for pid, sec, n in s.exec(select(PlantFact.plant_id, PlantFact.section, func.count()).where(
            col(PlantFact.status).in_(["verified", "applied"])).group_by(PlantFact.plant_id, PlantFact.section)).all():
        out.setdefault(pid, {})[sec] = n
    return out


STATUS_LABELS = {"reczne": "ręcznie", "reczne+sieci": "ręcznie + z sieci", "z_sieci": "z sieci",
                 "czeka": "czeka na scalenie", "brak": "brak"}


@router.get("/plants")
def plants_page(request: Request, origin: str = "", q: str = "", msg: str | None = None, missing: str = "",
                sort: str = ""):
    with Session(engine) as s:
        stmt = select(Plant)
        if origin:
            stmt = stmt.where(Plant.origin == origin)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(col(Plant.nazwa_pl).ilike(like) | col(Plant.nazwa_lat).ilike(like))
        plants = s.exec(stmt.order_by(Plant.nazwa_pl)).all()
        counts: dict[str, dict[str, int]] = {}
        for pid, st, n in s.exec(select(PlantFact.plant_id, PlantFact.status, func.count())
                                 .group_by(PlantFact.plant_id, PlantFact.status)).all():
            counts.setdefault(pid, {})[st] = n
        origins = dict(s.exec(select(Plant.origin, func.count()).group_by(Plant.origin)).all())
        kinds = ["plants_sync", "plant_research", "plant_apply", "plant_organize", "plant_photos", "plant_merge"]
        jobs = s.exec(select(Job).where(col(Job.kind).in_(kinds))
                      .order_by(col(Job.id).desc()).limit(6)).all()
        running = any(j.status in (JobStatus.queued, JobStatus.running) for j in jobs)
        web = _web_counts(s)
    files = {p.stem for p in Path(settings.kwiatownik_plants_dir).glob("*.json")} \
        if Path(settings.kwiatownik_plants_dir).exists() else set()
    # pokrycie schematu: kazda roslina + podsumowanie podrozdzialow (ile roslin ma / nie ma danej informacji)
    cov: dict[str, dict] = {}
    summary = {s.path: {"slot": s, "reczne": 0, "z_sieci": 0, "czeka": 0, "brak": 0} for s in schema.SLOTS}
    for p in plants:
        c = schema.coverage(_file(p.id) if p.id in files else None, web.get(p.id))
        c["merged"] = sum(1 for r in c["sloty"] if r["status"] in ("z_sieci", "reczne+sieci"))
        cov[p.id] = c
        if p.id not in files:                          # archiwum / nowa bez pliku - nie liczy sie do pokrycia
            continue
        for r in c["sloty"]:
            key = {"reczne+sieci": "z_sieci"}.get(r["status"], r["status"])
            summary[r["path"]][key] += 1
    if missing:
        plants = [p for p in plants if missing in cov[p.id]["braki"]]
    if sort == "luki":
        plants = sorted(plants, key=lambda p: cov[p.id]["procent"])
    return templates.TemplateResponse(request, "plants.html", {
        "plants": plants, "counts": counts, "origins": origins, "origin": origin, "q": q, "jobs": jobs,
        "running": running, "files": files, "msg": msg, "auto_apply": settings.knowledge_auto_apply,
        "langs": settings.knowledge_languages, "wiki_langs": settings.knowledge_wiki_langs,
        "web_pages": settings.knowledge_web_pages, "cov": cov, "summary": list(summary.values()),
        "missing": missing, "sort": sort, "slot_titles": {s.path: f"{s.chapter} - {s.title}" for s in schema.SLOTS},
        "search_langs": settings.knowledge_search_langs, "merge_models": settings.merge_models,
        "auto_merge": settings.knowledge_auto_merge})


@router.get("/plants/{pid}")
def plant_page(pid: str, request: Request, status: str = "", msg: str | None = None):
    with Session(engine) as s:
        plant = s.get(Plant, pid)
        if plant is None:
            return _redirect("/plants?msg=Nie+ma+takiej+rosliny")
        stmt = select(PlantFact).where(PlantFact.plant_id == pid)
        if status:
            stmt = stmt.where(PlantFact.status == status)
        facts = s.exec(stmt.order_by(PlantFact.section, PlantFact.id)).all()
        counts = dict(s.exec(select(PlantFact.status, func.count()).where(PlantFact.plant_id == pid)
                             .group_by(PlantFact.status)).all())
    by_section: dict[str, list] = {}
    for f in facts:
        by_section.setdefault(f.section, []).append(f)
    titles = json.loads(plant.wiki_titles_json) if plant.wiki_titles_json else {}
    has_file = (Path(settings.kwiatownik_plants_dir) / f"{pid}.json").exists()
    layout = None
    data = None
    if has_file:
        try:
            data, _ = plantfile.read_plant(settings.kwiatownik_plants_dir, pid)
            w = (data or {}).get("wiedza") or {}
            if w.get("sekcje"):
                layout = {"sections": len(w["sekcje"]), "points": sum(len(x.get("punkty") or []) for x in w["sekcje"]),
                          "sources": len(w.get("zrodla") or []), "how": w.get("uklad"), "date": w.get("uporzadkowano")}
        except (OSError, ValueError):
            layout = None
    with Session(engine) as s:
        web = _web_counts(s).get(pid, {})
        queries = s.exec(select(PlantQuery).where(PlantQuery.plant_id == pid)
                         .order_by(col(PlantQuery.id).desc()).limit(20)).all()
    cov = schema.coverage(data, web)
    merged = _merged_view(data)
    return templates.TemplateResponse(request, "plant.html", {
        "plant": plant, "by_section": by_section, "counts": counts, "status": status, "titles": titles,
        "section_titles": SECTION_TITLES_PL, "has_file": has_file, "statuses": STATUSES, "layout": layout,
        "photos": _plant_photos(plant), "chapters": schema.chapters(cov), "cov": cov, "merged": merged,
        "status_labels": STATUS_LABELS, "queries": queries, "msg": msg})


def _merged_view(data: dict | None) -> list[dict]:
    """Podrozdzialy scalone z wiedza z sieci: tekst Kwiatownika vs wersja scalona (ze zrodlami)."""
    if not data or not isinstance(data.get("scalone"), dict):
        return []
    zr = {z.get("nr"): z for z in ((data.get("wiedza") or {}).get("zrodla") or []) if isinstance(z, dict)}
    out = []
    for path, e in (data["scalone"].get("pola") or {}).items():
        current = schema.get_path(data, path)
        out.append({"path": path, "tytul": e.get("tytul") or path, "typ": e.get("typ"), "model": e.get("model"),
                    "oryginal": e.get("oryginal"), "aktualny": e.get("oryginal") == current or
                    (e.get("oryginal") is None and not schema.filled(current)),
                    "tresc": [dict(t, linki=[zr[n] for n in t.get("zrodla") or [] if n in zr]) for t in e.get("tresc") or []]})
    return out


def _plant_photos(plant) -> dict:
    try:
        data = json.loads(plant.photos_json) if plant.photos_json else {}
    except ValueError:
        data = {}
    return {"zdjecia": data.get("zdjecia") or [], "autorzy": data.get("autorzy") or [], "at": plant.photos_at}


@router.post("/plants/sync")
async def plants_sync(request: Request):
    return _redirect(f"/jobs/{actions.start_plants_sync(request.app.state)}")


@router.post("/plants/research")
async def plants_research(request: Request, ids: list[str] = Form([]), limit: str = Form("")):
    n = int(limit) if limit.strip().isdigit() else 10
    return _redirect(f"/jobs/{actions.start_plant_research(request.app.state, ids or None, n)}")


@router.post("/plants/{pid}/research")
async def plant_research_one(pid: str, request: Request):
    return _redirect(f"/jobs/{actions.start_plant_research(request.app.state, [pid])}")


@router.post("/plants/{pid}/apply")
async def plant_apply_one(pid: str, request: Request):
    return _redirect(f"/jobs/{actions.start_plant_apply(request.app.state, [pid])}")


@router.post("/plants/photos")
async def plants_photos(request: Request):
    return _redirect(f"/jobs/{actions.start_plant_photos(request.app.state)}")


@router.post("/plants/{pid}/photos")
async def plant_photos_one(pid: str, request: Request):
    return _redirect(f"/jobs/{actions.start_plant_photos(request.app.state, [pid])}")


@router.post("/plants/organize")
async def plants_organize(request: Request):
    return _redirect(f"/jobs/{actions.start_plant_organize(request.app.state)}")


@router.post("/plants/merge")
async def plants_merge(request: Request):
    return _redirect(f"/jobs/{actions.start_plant_merge(request.app.state)}")


@router.post("/plants/{pid}/merge")
async def plant_merge_one(pid: str, request: Request):
    return _redirect(f"/jobs/{actions.start_plant_merge(request.app.state, [pid])}")


@router.post("/plants/{pid}/unmerge")
def plant_unmerge(pid: str):
    """Usuwa scalenie: strona Kwiatownika wraca do recznych tekstow (kopia zapasowa w data/backups)."""
    from app.worker.knowledge import unmerge_plant
    done = unmerge_plant(pid)
    return _redirect(f"/plants/{pid}?msg=" + ("Usunieto+scalenie+-+strona+pokazuje+teksty+reczne" if done
                                              else "Brak+scalenia"))


@router.post("/plants/research-gap")
async def plants_research_gap(request: Request, slot: str = Form(...), limit: str = Form("10")):
    """Szukanie jednej brakujacej informacji (podrozdzialu) dla roslin, ktorym jej brakuje."""
    with Session(engine) as s:
        rows = s.exec(select(Plant).where(Plant.origin != "archiwum")).all()
        web = _web_counts(s)
    n = int(limit) if limit.strip().isdigit() else 10
    ids = [p.id for p in rows if slot in schema.coverage(_file(p.id), web.get(p.id))["puste"]][:n]
    if not ids:
        return _redirect("/plants?msg=Zadna+roslina+nie+ma+tej+luki")
    return _redirect(f"/jobs/{actions.start_plant_research(request.app.state, ids, len(ids), slots=[slot])}")


@router.post("/plants/{pid}/organize")
async def plant_organize_one(pid: str, request: Request):
    return _redirect(f"/jobs/{actions.start_plant_organize(request.app.state, [pid])}")


@router.post("/plants/facts/{fid}/status")
def fact_status(fid: int, status: str = Form(...)):
    """Reczna korekta: odrzuc informacje albo przywroc ja do weryfikacji."""
    with Session(engine) as s:
        f = s.get(PlantFact, fid)
        if f and status in ("rejected", "new", "verified") and f.status != "applied":
            f.status, f.reason = status, ("odrzucone recznie" if status == "rejected" else None)
            s.add(f)
            s.commit()
        pid = f.plant_id if f else ""
    return _redirect(f"/plants/{pid}")
