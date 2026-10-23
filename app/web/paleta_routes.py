"""Panel /paleta: Paleta Barw Zielarza w Kwiatowniku2 - ziola palety, dopasowane rosliny Siedziby, ile jest
zweryfikowanej wiedzy o farbowaniu wlosow i co trafilo do static/json/paleta_siedziba.json."""
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from app.web.templating import templates
from app.worker import paleta as wp

router = APIRouter()
# podrozdzialy schematu wiedzy, ktore szuka "Szukaj wiedzy o barwieniu" (app/knowledge/schema.py)
SLOTY_BARWIENIA = ["zastosowanie.rzemieslnicze", "zastosowanie.kosmetyczne"]


@router.get("/paleta")
def paleta_page(request: Request, msg: str = ""):
    ov = wp.overview()
    return templates.TemplateResponse(request, "paleta.html", {"msg": msg, **ov})


@router.post("/paleta/eksport")
def paleta_export(request: Request):
    from app.worker import actions
    jid = actions.start_paleta_export(request.app.state)
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/paleta/szukaj")
def paleta_search(request: Request, rosliny: str = Form("")):
    """Zbieranie wiedzy o barwieniu/kosmetyce dla roslin ziol palety (zwykla droga: wyszukiwarka -> LLM ->
    weryfikacja drugim modelem). Po weryfikacji informacje o wlosach trafia do palety przy kolejnym eksporcie."""
    from app.worker import actions
    ids = [x for x in (r.strip() for r in rosliny.split(",")) if x][:20]
    if not ids:
        return RedirectResponse("/paleta?msg=" + quote("Brak roslin do przeszukania."), status_code=303)
    jid = actions.start_plant_research(request.app.state, ids, len(ids), slots=SLOTY_BARWIENIA)
    return RedirectResponse(f"/jobs/{jid}", status_code=303)
