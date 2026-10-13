"""Panel /nn: co zrobily sieci neuronowe, ich wersje i metryki, tryby pracy, rozstrzyganie rozbieznosci."""
import json
import re
import time
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import NNModel, NNPrediction
from app.nn import schedule, store
from app.nn.encoders import default_encoder_name, get_encoder
from app.nn.status import overview, task_summary
from app.nn.tasks import TASKS, label_name
from app.web.templating import templates

router = APIRouter()

DECISIONS = {
    "przyjeta": "siec zdecydowala sama (bez LLM)",
    "do_llm": "siec niepewna - zdecydowal LLM",
    "cien": "tryb cienia - zdecydowal LLM, porownanie",
    "kontrola": "kontrola losowa - siec pewna, ale sprawdzil LLM",
    "zapas": "LLM niedostepny - uzyto odpowiedzi sieci",
    "zgodna": "kontrola: siec zgodna z LLM",
    "konflikt": "kontrola: siec pewna, LLM odpowiedzial inaczej - do rozstrzygniecia",
    "uzupelnienie": "kontrola: siec uzupelnila brak (czesc rosliny)",
    "niepewna": "kontrola: rozbieznosc, ale siec niepewna",
}
templates.env.globals["nn_label"] = label_name
templates.env.globals["nn_decisions"] = DECISIONS


def _encoders() -> list[dict]:
    try:
        import onnxruntime  # noqa: F401
        ort = True
    except ImportError:
        ort = False
    e5 = get_encoder("e5").status()
    e5["onnxruntime"] = ort
    return [e5, get_encoder("ngram").status()]


def _jobs_running() -> dict[str, int]:
    from app.models import Job, JobStatus
    with Session(engine) as s:
        rows = s.exec(select(Job).where(col(Job.kind).in_(["nn_train", "nn_init"]),
                                        col(Job.status).in_([JobStatus.queued, JobStatus.running]))).all()
    return {j.kind: j.id for j in rows}


def _ref_link(ref: str) -> str | None:
    if ref.startswith("fact:"):
        return None
    if ref.startswith("item:") and ref.split(":", 1)[1].isdigit():
        return f"/items/{ref.split(':', 1)[1]}"
    if ref.startswith("http"):
        return ref
    return None


def _fact_plants(rows: list[NNPrediction]) -> dict[str, str]:
    from app.models import PlantFact
    ids = [int(r.ref.split(":", 1)[1]) for r in rows if r.ref.startswith("fact:") and r.ref.split(":", 1)[1].isdigit()]
    if not ids:
        return {}
    with Session(engine) as s:
        return {f"fact:{i}": p for i, p in s.exec(select(PlantFact.id, PlantFact.plant_id)
                                                   .where(col(PlantFact.id).in_(ids))).all()}


def _predictions(task: str = "", decision: str = "", page: int = 1, per: int = 40) -> dict:
    with Session(engine) as s:
        q = select(NNPrediction)
        if task:
            q = q.where(NNPrediction.task == task)
        if decision == "rozbieznosc":
            q = q.where(NNPrediction.agree == False)  # noqa: E712
        elif decision:
            q = q.where(NNPrediction.decision == decision)
        total = s.exec(select(func.count()).select_from(q.subquery())).one()
        rows = list(s.exec(q.order_by(col(NNPrediction.id).desc()).offset((page - 1) * per).limit(per)).all())
    links = {r.id: _ref_link(r.ref) for r in rows}
    plants = _fact_plants(rows)
    return {"preds": rows, "links": links, "plants": plants, "total": total, "page": page, "per": per,
            "f_task": task, "f_decision": decision, "tops": {r.id: json.loads(r.top_json or "[]") for r in rows}}


@router.get("/nn")
def nn_page(request: Request, msg: str = ""):
    from app.nn.runtime import brain
    ctx = {"ov": overview(), "encoders": _encoders(), "default_encoder": default_encoder_name(),
           "running": _jobs_running(), "msg": msg, "tasks": TASKS, "modes": store.MODE_LABELS,
           "errors": brain.errors, "target": settings.nn_target_accuracy,
           "job_limit": getattr(getattr(request.app.state, "jobs", None), "max_concurrent", 1),
           "auto": schedule.auto_settings(), "auto_due": schedule.due()[1], "hours_choices": schedule.HOURS_CHOICES}
    ctx.update(_predictions())
    return templates.TemplateResponse(request, "nn.html", ctx)


@router.get("/nn/decyzje")
def nn_predictions_partial(request: Request, task: str = "", decision: str = "", page: int = 1):
    return templates.TemplateResponse(request, "nn_predictions.html", _predictions(task, decision, max(1, page)))


@router.post("/nn/train")
async def nn_train(request: Request, task: str = Form(""), encoder: str = Form("")):
    from app.worker import actions
    jid = actions.start_nn_train(request.app.state, [task] if task else None, encoder or None)
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/nn/init")
async def nn_init(request: Request):
    from app.worker import actions
    jid = actions.start_nn_init(request.app.state)
    return RedirectResponse(f"/jobs/{jid}", status_code=303)


@router.post("/nn/auto")
async def nn_auto(request: Request, on: str = Form(""), hours: int = Form(24), time_: str = Form("", alias="time"),
                  only_new: str = Form(""), now: str = Form("")):
    """Ustawienia automatycznego treningu; "Sprawdz teraz" od razu uruchamia trening, jesli jest pora."""
    hours = hours if hours in schedule.HOURS_CHOICES else 24
    t = time_.strip()
    if t and not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", t):
        t = ""
    schedule.save_auto(on=on == "1", hours=hours, time=t, only_new=only_new == "1")
    msg = "Zapisano ustawienia automatycznego treningu."
    if now == "1":
        jid = schedule.run_if_due(request.app.state)
        msg = f"Uruchomiono trening - zadanie #{jid}." if jid else f"Nie teraz: {schedule.due()[1]}."
    return RedirectResponse("/nn?msg=" + quote(msg), status_code=303)


@router.post("/nn/{task}/mode")
def nn_mode(task: str, mode: str = Form(...), back: str = Form("/nn")):
    if task in TASKS and mode in store.MODES:
        store.set_task_state(task, mode=mode)
    return RedirectResponse(back if back.startswith("/nn") else "/nn", status_code=303)


@router.post("/nn/{task}/version/{version}")
def nn_use_version(task: str, version: int):
    """Reczny wybor wersji (np. powrot do poprzedniej)."""
    from app.nn.runtime import brain
    with Session(engine) as s:
        row = s.exec(select(NNModel).where(NNModel.task == task, NNModel.version == version)).first()
        if row and store.model_dir(task, version).exists():
            for old in s.exec(select(NNModel).where(NNModel.task == task, NNModel.status == "biezaca")).all():
                old.status = "wycofana"
                old.note = (old.note or "") + f" | recznie zastapiona przez v{version}"
                s.add(old)
            row.status = "biezaca"
            row.note = (row.note or "") + " | wybrana recznie"
            s.add(row)
            s.commit()
            st = store.task_state(task)
            store.set_task_state(task, version=version, mode=st["mode"] if st["version"] else "cien")
            brain.invalidate(task)
    return RedirectResponse(f"/nn/zadanie/{task}", status_code=303)


@router.post("/nn/rozstrzygnij/{pred_id}")
def nn_resolve(request: Request, pred_id: int, choice: str = Form(...)):
    from app.nn.hooks import resolve
    msg = resolve(pred_id, choice)
    if request.headers.get("HX-Request"):
        return HTMLResponse(f'<td colspan="8" class="muted">{msg}</td>')
    return RedirectResponse("/nn?msg=" + msg, status_code=303)


@router.post("/nn/test")
def nn_test(request: Request, text: str = Form(...), task: str = Form("page_kind")):
    from html import escape

    from app.nn.runtime import brain
    if task not in TASKS:
        return HTMLResponse('<p class="box">Nieznane zadanie.</p>')
    t0 = time.perf_counter()
    try:
        preds, ms, lm = brain.predict(task, [text[:6000]])
    except Exception as exc:  # noqa: BLE001
        return HTMLResponse(f'<p class="box">Blad: {escape(str(exc))}</p>')
    if not preds:
        return HTMLResponse('<p class="box">Ta siec nie ma jeszcze wytrenowanej wersji - uruchom trening.</p>')
    label, conf, top = preds[0]
    sure = conf >= lm.threshold and label in lm.allowed
    bars = "".join(
        f'<div class="ag-row"><span>{escape(label_name(lab))}</span><div class="ag-track"><i style="width:{p * 100:.0f}%">'
        f'</i></div><b>{p:.0%}</b></div>' for lab, p in top)
    verdict = ("siec zdecydowalaby sama (pewnosc ponad progiem " + f"{lm.threshold:.2f})" if sure
               else f"siec oddalaby to do LLM (prog {lm.threshold:.2f}"
                    + ("" if label in lm.allowed else ", klasa niedopuszczona do samodzielnych decyzji") + ")")
    return HTMLResponse(f'<div class="ag-card">{bars}<p class="muted">v{lm.version} ({lm.encoder}), '
                        f'{(time.perf_counter() - t0) * 1000:.0f} ms &middot; {escape(verdict)}</p></div>')


@router.get("/nn/zadanie/{task}")
def nn_task_page(request: Request, task: str, v: int | None = None):
    if task not in TASKS:
        return RedirectResponse("/nn", status_code=303)
    with Session(engine) as s:
        versions = list(s.exec(select(NNModel).where(NNModel.task == task).order_by(col(NNModel.version).desc())).all())
        summary = task_summary(s, task, days=30)
        per_class_online = s.exec(select(NNPrediction.label, NNPrediction.agree, func.count()).where(
            NNPrediction.task == task, col(NNPrediction.agree).is_not(None))
            .group_by(NNPrediction.label, NNPrediction.agree)).all()
    shown = next((x for x in versions if x.version == v), None) or summary["model"] or (versions[0] if versions else None)
    metrics = json.loads(shown.metrics_json or "{}") if shown else {}
    labels = json.loads(shown.labels_json or "[]") if shown else []
    online: dict[str, list[int]] = {}
    for lab, ag, n in per_class_online:
        online.setdefault(lab, [0, 0])[0 if ag else 1] += n
    vm = {x.id: json.loads(x.metrics_json or "{}") for x in versions}
    ctx = {"task": task, "spec": TASKS[task], "versions": versions, "vm": vm, "shown": shown, "m": metrics,
           "labels": labels, "sm": summary, "online": online,
           "modes": store.MODE_LABELS if TASKS[task].kind == "decyzja" else store.MODE_LABELS_CHECK,
           "running": _jobs_running(), "target": settings.nn_target_accuracy}
    ctx.update(_predictions(task))
    return templates.TemplateResponse(request, "nn_task.html", ctx)
