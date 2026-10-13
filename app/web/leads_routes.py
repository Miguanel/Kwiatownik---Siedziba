"""Panel /wzmianki: przepisy, o ktorych strony mowia, ale bez pelnego wykonania - i szukanie ich gdzie indziej."""
import json
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.db import engine
from app.models import Item, RecipeLead
from app.pipeline import leads as leadmod
from app.web.templating import templates

router = APIRouter()
STATUS_LABELS = {"szukam": "do odnalezienia", "znaleziony": "odnaleziony", "nie_znaleziono": "nie znaleziono",
                 "odrzucony": "odrzucony"}


@router.get("/wzmianki")
def leads_page(request: Request, status: str = "szukam", msg: str = "", page: int = 1):
    per = 50
    with Session(engine) as s:
        counts = dict(s.exec(select(RecipeLead.status, func.count()).group_by(RecipeLead.status)).all())
        q = select(RecipeLead)
        if status in STATUS_LABELS:
            q = q.where(RecipeLead.status == status)
        total = s.exec(select(func.count()).select_from(q.subquery())).one()
        rows = list(s.exec(q.order_by(col(RecipeLead.id).desc()).offset((max(page, 1) - 1) * per).limit(per)).all())
        found = {r.found_item_id: s.get(Item, r.found_item_id) for r in rows if r.found_item_id}
        facts_left = len(leadmod.facts_to_check(100000))
    extra = {r.id: {"ingredients": json.loads(r.ingredients_json or "[]"), "log": json.loads(r.search_log or "[]")}
             for r in rows}
    return templates.TemplateResponse(request, "leads.html", {
        "rows": rows, "extra": extra, "counts": counts, "status": status, "labels": STATUS_LABELS, "msg": msg, "found": found,
        "total": total, "page": max(page, 1), "per": per, "facts_left": facts_left,
        "max_attempts": leadmod.MAX_ATTEMPTS})


@router.post("/wzmianki/szukaj")
async def leads_search(request: Request, ids: list[int] = Form([]), limit: int = Form(5)):
    from app.worker import actions
    jid = actions.start_leads_search(request.app.state, ids or None, max(1, min(limit, 50)))
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/wzmianki/wyciagnij")
async def leads_extract(request: Request, limit: int = Form(20)):
    from app.worker import actions
    jid = actions.start_leads_extract(request.app.state, max(1, min(limit, 200)))
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/wzmianki/{lid}/status")
def lead_status(lid: int, status: str = Form(...), back: str = Form("szukam")):
    with Session(engine) as s:
        row = s.get(RecipeLead, lid)
        if row and status in ("szukam", "odrzucony"):
            row.status = status
            if status == "szukam":
                row.attempts = 0                       # przywrocona = szukamy od nowa
            s.add(row)
            s.commit()
    msg = "Odrzucono wzmianke." if status == "odrzucony" else "Wzmianka wraca na liste do odnalezienia."
    return RedirectResponse(f"/wzmianki?status={quote(back)}&msg={quote(msg)}", status_code=303)
