from dataclasses import asdict
from fastapi import APIRouter, Form, HTTPException, Request

from app.config import settings
from app.llm.base import LLMError
from app.llm.router import ModelRouter
from app.web.templating import templates

router = APIRouter()

DEFAULT_PROMPT = 'Podaj jedna ciekawostke o pokrzywie zwyczajnej. Odpowiedz JSON: {"ciekawostka": "..."}'


def _router(request: Request) -> ModelRouter:
    r = getattr(request.app.state, "llm", None)
    if r is None:
        raise HTTPException(503, "Brak providerow LLM - uzupelnij GEMINI_API_KEY / GROQ_API_KEY w .env")
    return r


def _models_ctx(request: Request) -> dict:
    r = getattr(request.app.state, "llm", None)
    return {"models": r.snapshot() if r else [], "errors": r.discovery_errors if r else {},
            "configured": r is not None}


# ---------------- panel (HTML / HTMX) ----------------
def _mt_status() -> dict | None:
    from app.mt.nllb import get_translator
    mt = get_translator()
    return mt.status() if mt else None


@router.get("/llm")
def llm_page(request: Request):
    return templates.TemplateResponse(request, "llm.html", {
        "app_name": settings.app_name, "prompt": DEFAULT_PROMPT, "mt": _mt_status(), **_models_ctx(request)})


@router.post("/mt/test")
async def mt_test_partial(request: Request, text: str = Form(...), lang: str = Form("en")):
    """Probne tlumaczenie NLLB (bez LLM) - wynik i czas."""
    import asyncio
    import time
    from html import escape

    from fastapi.responses import HTMLResponse

    from app.mt.nllb import MTUnavailable, get_translator
    mt = get_translator()
    if mt is None:
        return HTMLResponse('<p class="box">MT_ENGINE=llm - tlumacz NLLB wylaczony.</p>')
    t0 = time.perf_counter()
    try:
        out = (await asyncio.to_thread(mt.translate, [text[:2000]], lang, "pl"))[0]
    except MTUnavailable as exc:
        return HTMLResponse(f'<p class="box">Niedostepny: {escape(str(exc))}</p>')
    return HTMLResponse(f'<p class="box"><b>{escape(out)}</b><br><span class="muted">NLLB {escape(mt.device)}, '
                        f'{time.perf_counter() - t0:.2f} s (pierwsze uzycie wczytuje model)</span></p>')


@router.get("/llm/models")
def llm_models_partial(request: Request):
    return templates.TemplateResponse(request, "llm_models.html", _models_ctx(request))


@router.post("/llm/refresh")
async def llm_refresh_partial(request: Request):
    await _router(request).refresh()
    return templates.TemplateResponse(request, "llm_models.html", _models_ctx(request))


@router.post("/llm/reset")
def llm_reset_partial(request: Request):
    _router(request).reset()
    return templates.TemplateResponse(request, "llm_models.html", _models_ctx(request))


@router.post("/llm/ollama/pull")
async def llm_ollama_pull(request: Request, model: str = Form(...)):
    from fastapi.responses import RedirectResponse

    from app.worker import actions
    model = model.strip()
    if not model:
        return RedirectResponse("/llm", status_code=303)
    return RedirectResponse(f"/jobs/{actions.start_ollama_pull(request.app.state, model)}", status_code=303)


@router.post("/llm/test")
async def llm_test_partial(request: Request, prompt: str = Form(DEFAULT_PROMPT), json_mode: bool = Form(False)):
    try:
        result, error = await _router(request).complete(prompt, json_mode=json_mode), None
    except LLMError as exc:
        result, error = None, str(exc)
    resp = templates.TemplateResponse(request, "llm_results.html", {"result": result, "error": error})
    resp.headers["HX-Trigger"] = "models-changed"  # odswiez tabele rankingu
    return resp


# ---------------- API (JSON) ----------------
@router.get("/api/llm/models")
def api_models(request: Request):
    r = _router(request)
    return {"models": r.snapshot(), "discovery_errors": r.discovery_errors}


@router.post("/api/llm/refresh")
async def api_refresh(request: Request):
    r = _router(request)
    await r.refresh()
    return {"models": r.snapshot(), "discovery_errors": r.discovery_errors}


@router.post("/api/llm/test")
async def api_test(request: Request, prompt: str = DEFAULT_PROMPT, json_mode: bool = True):
    try:
        return asdict(await _router(request).complete(prompt, json_mode=json_mode))
    except LLMError as exc:
        raise HTTPException(502, str(exc))
