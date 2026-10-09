"""Reczny commit danych Kwiatownika2 (wdrozeniowiec w trybie DEPLOY_MODE=reczny).

- /commit/pasek (htmx, co 60 s na kazdej stronie): licznik zmian czekajacych na commit w pasku nawigacji
  i komunikat "Commit gotowy" z przyciskiem, gdy progi sa spelnione albo jest nowe rozmieszczenie wiedzy,
- /commit: co wejdzie do commita (liczby, rosliny, przepisy, pliki, wpis kroniki = tresc commita) + przycisk,
- POST /commit: zadanie agent_deploy (force, trigger "przycisk") -> commit + push -> Render buduje strone.
"""
from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from app.agents import deploy
from app.config import settings
from app.web.templating import templates
from app.worker import actions

router = APIRouter()


@router.get("/commit/pasek")
async def commit_bar(request: Request, strona: str = ""):
    st = deploy.pending()
    return templates.TemplateResponse(request, "commit_bar.html", {"st": st, "strona": strona,
                                                                    "enabled": settings.deploy_enabled})


@router.get("/commit/stan.json")
async def commit_state():
    st = deploy.pending()
    z = st.get("zmiany") or {}
    return {"tryb": st["tryb"], "gotowy": st["gotowy"], "w_toku": st.get("w_toku"), "licznik": st["licznik"],
            "blokada": st.get("blokada"), "powod": st.get("powod"), "pliki": len(st.get("pliki") or []),
            "informacje": z.get("informacje", 0), "przepisy": z.get("przepisy", 0),
            "rozmieszczone": z.get("rozmieszczone", 0), "nowe_rosliny": z.get("nowe_rosliny", 0),
            "zdjecia": z.get("zdjecia", 0), "rozdzialy": z.get("rozdzialy", 0)}


@router.get("/commit")
async def commit_page(request: Request):
    st = deploy.pending(refresh=True)
    return templates.TemplateResponse(request, "commit.html", {"st": st, "msg": request.query_params.get("msg", ""),
                                                                "min_points": settings.deploy_min_points,
                                                                "min_recipes": settings.deploy_min_recipes})


@router.post("/commit")
async def commit_now(request: Request, potwierdz: str = Form("")):
    st = deploy.pending(refresh=True)
    if st.get("w_toku"):
        return RedirectResponse("/commit?msg=Commit+juz+trwa", status_code=303)
    if st.get("blokada"):
        return RedirectResponse("/commit?msg=Nie+mozna+commitowac:+" + st["blokada"][:120].replace(" ", "+"), status_code=303)
    if not st.get("pliki"):
        return RedirectResponse("/commit?msg=Nic+nie+czeka+na+commit", status_code=303)
    jid = actions.start_agent_deploy(request.app.state, trigger="przycisk", force=True)
    deploy.invalidate_pending()
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/commit/odswiez")
async def commit_refresh():
    deploy.pending(refresh=True)
    return RedirectResponse("/commit", status_code=303)
