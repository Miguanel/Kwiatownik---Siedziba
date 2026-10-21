"""Panel /magia: magia roslin - czary, zabobony, obrzedy, rosliny czarownic. Rozklad zebranej wiedzy (rodzaje
praktyk, kraje, epoki, jezyki), tropy-azymuty, ostatnie znaleziska i uruchamianie zadania plant_magic."""
import json
from collections import Counter
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.knowledge.magic_lexicon import CONCEPTS, DEEP_SOURCES, RODZAJE, SWEEPS
from app.models import MagicLead, Plant, PlantFact, PlantQuery
from app.web.templating import templates

router = APIRouter()
LEAD_STATUS = {"nowy": "do sprawdzenia", "sprawdzony": "sprawdzone", "wyczerpany": "bez wyniku", "odrzucony": "odrzucone"}
FACT_STATUS = {"new": "do weryfikacji", "verified": "zweryfikowane", "applied": "w Kwiatowniku", "rejected": "odrzucone"}
KIND_LABELS = {"nazwa": "nazwa ludowa", "praktyka": "praktyka", "analogia": "odpowiednik"}


def _meta(raw: str | None) -> dict:
    try:
        d = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


@router.get("/magia")
def magic_page(request: Request, trop: str = "nowy", msg: str = "", page: int = 1):
    per = 60
    with Session(engine) as s:
        fact_counts = dict(s.exec(select(PlantFact.status, func.count()).where(PlantFact.section == "magia")
                                  .group_by(PlantFact.status)).all())
        rows = s.exec(select(PlantFact.plant_id, PlantFact.language, PlantFact.meta_json)
                      .where(PlantFact.section == "magia", PlantFact.status != "rejected")).all()
        lead_counts = dict(s.exec(select(MagicLead.status, func.count()).group_by(MagicLead.status)).all())
        lq = select(MagicLead)
        if trop in LEAD_STATUS:
            lq = lq.where(MagicLead.status == trop)
        lead_total = s.exec(select(func.count()).select_from(lq.subquery())).one()
        leads = list(s.exec(lq.order_by(MagicLead.depth, col(MagicLead.id).desc())
                            .offset((max(page, 1) - 1) * per).limit(per)).all())
        recent = list(s.exec(select(PlantFact).where(PlantFact.section == "magia")
                             .order_by(col(PlantFact.id).desc()).limit(30)).all())
        names = dict(s.exec(select(Plant.id, Plant.nazwa_pl)).all())
        queries = s.exec(select(func.count()).select_from(PlantQuery)
                         .where(col(PlantQuery.slot).startswith("magia"))).one()
        q_with_page = s.exec(select(func.count()).select_from(PlantQuery)
                             .where(col(PlantQuery.slot).startswith("magia"), col(PlantQuery.used_url).is_not(None))).one()
        archives = dict(s.exec(select(PlantQuery.slot, func.sum(PlantQuery.facts))
                               .where(col(PlantQuery.slot).startswith("magia:arch:")).group_by(PlantQuery.slot)).all())
    by_kind, by_country, by_age, by_lang, by_plant = Counter(), Counter(), Counter(), Counter(), Counter()
    toxic = 0
    for pid, lang, raw in rows:
        m = _meta(raw)
        by_kind[m.get("rodzaj") or "zabobony"] += 1
        if m.get("kraj"):
            by_country[m["kraj"]] += 1
        if m.get("epoka"):
            by_age[m["epoka"]] += 1
        by_lang[(lang or "?")[:2]] += 1
        by_plant[pid] += 1
        toxic += 1 if m.get("toksyczne") else 0
    top = max(by_kind.values(), default=0) or 1
    return templates.TemplateResponse(request, "magic.html", {
        "msg": msg, "fact_counts": fact_counts, "fact_labels": FACT_STATUS, "total_facts": len(rows),
        "rodzaje": RODZAJE, "by_kind": by_kind, "top": top, "by_country": by_country.most_common(14),
        "by_age": by_age.most_common(12), "by_lang": by_lang.most_common(), "by_plant": by_plant.most_common(15),
        "toxic": toxic, "names": names, "leads": leads, "lead_counts": lead_counts, "lead_labels": LEAD_STATUS,
        "kind_labels": KIND_LABELS, "trop": trop, "lead_total": lead_total, "page": max(page, 1), "per": per,
        "recent": [(f, _meta(f.meta_json)) for f in recent], "queries": queries, "q_with_page": q_with_page,
        "concepts": CONCEPTS, "deep": DEEP_SOURCES, "sweeps": SWEEPS,
        "archives": {k.split(":", 2)[2]: int(v or 0) for k, v in archives.items()},
        "plants": sorted(names.items(), key=lambda kv: (kv[1] or kv[0]).lower()), "settings": settings})


@router.post("/magia/szukaj")
async def magic_search(request: Request, limit: int = Form(3), sweeps: int = Form(2), roslina: str = Form("")):
    from app.worker import actions
    pid = roslina.strip()
    if pid:
        jid = actions.start_plant_magic(request.app.state, [pid], sweeps=max(0, min(sweeps, 10)))
    else:
        jid = actions.start_plant_magic(request.app.state, None, max(1, min(limit, 30)), max(0, min(sweeps, 10)))
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/magia/trop/{lid}/status")
def lead_status(lid: int, status: str = Form(...), back: str = Form("nowy")):
    with Session(engine) as s:
        row = s.get(MagicLead, lid)
        if row and status in ("nowy", "odrzucony"):
            row.status = status
            if status == "nowy":
                row.attempts = 0
            s.add(row)
            s.commit()
    msg = "Trop odrzucony - nie bedzie szukany." if status == "odrzucony" else "Trop wraca do sprawdzenia."
    return RedirectResponse(f"/magia?trop={quote(back)}&msg={quote(msg)}", status_code=303)
