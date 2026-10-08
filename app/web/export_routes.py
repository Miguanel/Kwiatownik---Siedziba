import json
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.exporters import store
from app.db import get_session
from app.models import Item, ItemKind, ItemStatus, Job
from app.pipeline.enrich import parse_systems
from app.web.templating import templates
from app.worker import actions
from app.worker.archive_import import archive_files
from app.worker.export import items_without_enrichment

router = APIRouter()


def _back(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


@router.get("/export")
def export_page(request: Request, session: Session = Depends(get_session)):
    rows = session.exec(select(Item.status, func.count()).where(
        Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None)).group_by(Item.status)).all()
    counts = {st.value: n for st, n in rows}
    exists = session.exec(select(Item).where(Item.status == ItemStatus.exists).order_by(col(Item.id).desc())
                          .limit(50)).all()
    errors = session.exec(select(Item).where(Item.status == ItemStatus.error, Item.kind == ItemKind.recipe)
                          .order_by(col(Item.id).desc()).limit(20)).all()
    to_review = session.exec(select(Item).where(Item.status == ItemStatus.translated, col(Item.duplicate_of).is_(None))
                             .order_by(col(Item.id).desc()).limit(50)).all()
    jobs = session.exec(select(Job).where(col(Job.kind).in_(["translate", "enrich", "export", "import_k1"])).order_by(col(Job.id).desc())
                        .limit(8)).all()
    folder = Path(settings.kwiatownik_przepisy_dir)
    series = store.series_of(settings.export_filename)
    target = next((f for f in store.current_files(folder) if store.parse_name(f.name)[0] == series),
                  folder / store.dated_name(series))
    return templates.TemplateResponse(request, "export.html", {
        "counts": counts, "exists": exists, "errors": errors, "to_review": to_review, "jobs": jobs,
        "target": target, "target_exists": target.exists(), "store_files": store.all_files(folder), "require_approval": settings.export_require_approval,
        "batch": settings.translate_batch, "enrich_on": settings.enrich_recipes,
        "systems": parse_systems(settings.enrich_systems), "skip_culinary": settings.skip_culinary,
        "to_enrich": len(items_without_enrichment(session=session)),
        "k1_files": [f.name for f in archive_files(settings.kwiatownik1_przepisy_dir)], "titles": {i.id: json.loads(i.data_json).get("tytul") for i in
                                                      exists + to_review if i.data_json}})


@router.post("/export/translate")
async def export_translate(request: Request, limit: int = Form(20)):
    return _back(f"/jobs/{actions.start_translate(request.app.state, limit=limit)}")


@router.post("/export/import-k1")
async def export_import_k1(request: Request):
    return _back(f"/jobs/{actions.start_import_k1(request.app.state)}")


@router.post("/export/enrich")
async def export_enrich(request: Request, limit: int = Form(20)):
    return _back(f"/jobs/{actions.start_enrich(request.app.state, limit=limit)}")


@router.post("/items/{iid}/enrich")
async def item_enrich(iid: int, request: Request):
    return _back(f"/jobs/{actions.start_enrich(request.app.state, item_ids=[iid])}")


@router.post("/export/run")
async def export_run(request: Request, translate_first: int = Form(0)):
    return _back(f"/jobs/{actions.start_export(request.app.state, translate_first)}")


@router.post("/export/approve-all")
def export_approve_all(session: Session = Depends(get_session)):
    for it in session.exec(select(Item).where(Item.status == ItemStatus.translated)).all():
        it.status = ItemStatus.approved
        session.add(it)
    session.commit()
    return _back("/export")


@router.post("/items/{iid}/translate")
async def item_translate(iid: int, request: Request):
    return _back(f"/jobs/{actions.start_translate(request.app.state, item_ids=[iid])}")


@router.post("/items/{iid}/not-duplicate")
def item_not_duplicate(iid: int, session: Session = Depends(get_session)):
    """Uzytkownik uznal, ze to jednak inny przepis - zdejmij oznaczenie i nie sprawdzaj ponownie."""
    it = session.get(Item, iid)
    if it:
        it.status = ItemStatus.translated if it.data_json else ItemStatus.raw
        it.match_ref, it.match_info, it.duplicate_of = None, "ignored", None
        session.add(it)
        session.commit()
    return _back(f"/items/{iid}")
