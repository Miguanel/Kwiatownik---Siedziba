"""Panel /encyklopedie: zagraniczne encyklopedie i bazy wiedzy o roslinach - wlaczanie, statystyki, test."""
from html import escape
from urllib.parse import urlparse

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.knowledge import encyclopedias as encmod
from app.models import Plant, PlantFact, PlantQuery
from app.web.templating import templates

router = APIRouter()


def _stats() -> dict[str, dict]:
    out = {e.key: {"asked": 0, "found": 0, "facts": 0, "applied": 0, "last": None} for e in encmod.ENCYCLOPEDIAS}
    with Session(engine) as s:
        rows = s.exec(select(PlantQuery.slot, func.count(), func.count(PlantQuery.used_url),
                             func.sum(PlantQuery.facts), func.max(PlantQuery.created_at))
                      .where(col(PlantQuery.slot).startswith("enc:")).group_by(PlantQuery.slot)).all()
        for slot, n, found, facts, last in rows:
            key = slot.split(":", 1)[1]
            if key in out:
                out[key].update(asked=n, found=found, facts=int(facts or 0), last=last)
        for e in encmod.ENCYCLOPEDIAS:
            out[e.key]["applied"] = s.exec(select(func.count()).select_from(PlantFact).where(
                col(PlantFact.source_url).contains(e.domain),
                col(PlantFact.status).in_(["verified", "applied"]))).one()
    return out


@router.get("/encyklopedie")
def encyclopedias_page(request: Request, msg: str = ""):
    with Session(engine) as s:
        plants = s.exec(select(Plant.id, Plant.nazwa_pl, Plant.nazwa_lat).order_by(Plant.nazwa_pl)).all()
    groups: dict[str, list] = {}
    for e in encmod.ENCYCLOPEDIAS:
        groups.setdefault(e.country, []).append(e)
    return templates.TemplateResponse(request, "encyclopedias.html", {
        "groups": groups, "flags": encmod.FLAGS, "on": encmod.enabled_map(), "stats": _stats(), "plants": plants,
        "msg": msg, "enabled": settings.knowledge_encyclopedias, "per_run": settings.knowledge_enc_max,
        "countries": {"jp": "Japonia", "cn": "Chiny", "kr": "Korea", "us": "USA", "gb": "Wielka Brytania"}})


@router.post("/encyklopedie/test")
async def encyclopedia_test(plant_id: str = Form(...), key: str = Form("")):
    """Proba znalezienia strony rosliny w zrodlach - bez LLM i bez zapisu (sprawdzenie, czy adresy dzialaja)."""
    import json

    from app.knowledge.gaps import plant_names
    from app.scrapers.fetcher import Fetcher
    from app.worker.discover import build_backends
    with Session(engine) as s:
        p = s.get(Plant, plant_id)
        if not p:
            return HTMLResponse('<p class="box">Nie ma takiej rosliny.</p>')
        try:
            labels = json.loads(p.names_json) if p.names_json else {}
            titles = json.loads(p.wiki_titles_json) if p.wiki_titles_json else {}
        except ValueError:
            labels, titles = {}, {}
        names = plant_names(p.nazwa_lat, p.nazwa_pl, labels, titles)
    encs = [encmod.BY_KEY[key]] if key in encmod.BY_KEY else list(encmod.ENCYCLOPEDIAS)
    fetcher = Fetcher(settings.user_agent, 0.5, accept_language="ja,zh;q=0.9,ko;q=0.8,en;q=0.7,*;q=0.3")
    backends = build_backends()
    rows = []
    try:
        for e in encs:
            name = encmod.plant_name(e, names)
            if not name:
                rows.append(f"<tr><td>{encmod.FLAGS.get(e.country, '')} {escape(e.name)}</td><td colspan=2 class=muted>"
                            f"brak nazwy gatunku w jezyku '{e.name_lang}' (Wikidata) - zrodlo pomijane</td></tr>")
                continue
            f = await encmod.fetch(e, names, fetcher, backends)
            if f.url:
                rows.append(f"<tr><td>{encmod.FLAGS.get(e.country, '')} {escape(e.name)}<br><span class=muted>"
                            f"{escape(name)}</span></td><td><span class='tag done'>znaleziono</span> {escape(f.how)}<br>"
                            f"<a href='{escape(f.url)}' target=_blank rel=noopener>{escape(urlparse(f.url).netloc)}"
                            f"{escape(urlparse(f.url).path[:60])}</a></td><td class=enc-prev>"
                            f"{escape(f.text[:280])}...</td></tr>")
            else:
                rows.append(f"<tr><td>{encmod.FLAGS.get(e.country, '')} {escape(e.name)}<br><span class=muted>"
                            f"{escape(name)}</span></td><td><span class='tag failed'>brak</span></td>"
                            f"<td class=muted>{escape(f.reason[:240])}</td></tr>")
    finally:
        await fetcher.aclose()
    head = (f"<p class=muted>{escape(p.nazwa_pl or plant_id)} ({escape(p.nazwa_lat or '-')}) - nazwy: "
            + escape(", ".join(f"{k}: {v}" for k, v in names.items() if k in ("la", "ja", "zh", "ko", "en"))) + "</p>")
    return HTMLResponse(head + "<div class=ag-tbl><table><tr><th>zrodlo</th><th>wynik</th><th>poczatek tekstu / powod"
                        "</th></tr>" + "".join(rows) + "</table></div>")


@router.post("/encyklopedie/{key}")
def encyclopedia_toggle(key: str, on: str = Form("1")):
    encmod.set_enabled(key, on == "1")
    return RedirectResponse("/encyklopedie", status_code=303)
