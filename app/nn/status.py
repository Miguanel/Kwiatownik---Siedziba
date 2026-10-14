"""Podsumowania pracy sieci dla panelu /nn, zakladki Na zywo i listy "Zadania do zrobienia"."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import Item, ItemKind, NNModel, NNPrediction, NNSample, PlantFact
from app.nn import store
from app.nn.tasks import TASKS

SAVED = ("przyjeta", "uzupelnienie")          # decyzje, przy ktorych LLM nie byl potrzebny / siec dodala wartosc
_cache: dict = {}


def approx_examples(s: Session, task: str) -> int:
    """Szybkie oszacowanie liczby przykladow (bez budowania zbioru) - do wykrywania, ze warto trenowac ponownie."""
    human = s.exec(select(func.count()).select_from(NNSample).where(NNSample.task == task,
                                                                   NNSample.origin == "czlowiek")).one()
    if task == "page_kind":
        smp = s.exec(select(func.count()).select_from(NNSample).where(
            NNSample.task == task, NNSample.weight > 0, col(NNSample.origin).not_like("nn:%"))).one()
        boot = s.exec(select(func.count()).select_from(Item).where(
            col(Item.raw_text).is_not(None), col(Item.kind).in_([ItemKind.recipe, ItemKind.fact]))).one()
        return int(smp + boot)
    if task == "recipe_type":
        return int(human + s.exec(select(func.count()).select_from(Item).where(
            Item.kind == ItemKind.recipe, col(Item.data_json).is_not(None), col(Item.duplicate_of).is_(None))).one())
    return int(human + s.exec(select(func.count()).select_from(PlantFact).where(
        col(PlantFact.status).in_(["verified", "applied"]))).one())


def current_model(s: Session, task: str) -> NNModel | None:
    st = store.task_state(task)
    if not st.get("version"):
        return None
    return s.exec(select(NNModel).where(NNModel.task == task, NNModel.version == st["version"])).first()


def human_since(s: Session, task: str, since) -> int:
    """Przyklady oznaczone przez czlowieka (panel /nn) od chwili `since` (od ostatniego treningu)."""
    from app.models import NNSample
    q = select(func.count()).select_from(NNSample).where(NNSample.task == task, NNSample.origin == "czlowiek")
    if since is not None:
        q = q.where(NNSample.created_at >= since)
    return int(s.exec(q).one() or 0)


def task_summary(s: Session, task: str, days: int = 7) -> dict:
    spec = TASKS[task]
    st = store.task_state(task)
    m = current_model(s, task)
    metrics = json.loads(m.metrics_json or "{}") if m else {}
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = s.exec(select(NNPrediction.decision, NNPrediction.agree, func.count()).where(
        NNPrediction.task == task, NNPrediction.created_at >= since)
        .group_by(NNPrediction.decision, NNPrediction.agree)).all()
    by_dec: dict[str, int] = {}
    agree = disagree = 0
    audit_ok = audit_n = 0
    for dec, ag, n in rows:
        by_dec[dec] = by_dec.get(dec, 0) + n
        if ag is True:
            agree += n
        elif ag is False:
            disagree += n
        if dec == "kontrola" and ag is not None:
            audit_n += n
            audit_ok += n if ag else 0
    total = sum(by_dec.values())
    n_now = approx_examples(s, task)
    new = n_now - (m.n_train if m else 0)
    conflicts = s.exec(select(func.count()).select_from(NNPrediction).where(
        NNPrediction.task == task, NNPrediction.decision == "konflikt", col(NNPrediction.human_label).is_(None))).one()
    agreement = agree / (agree + disagree) if agree + disagree else None
    thr = (metrics.get("threshold") or {}).get("threshold", 1.01)
    ready = (st["mode"] == "cien" and m is not None and thr <= 1.0 and agreement is not None
             and agree + disagree >= 50 and agreement >= settings.nn_target_accuracy - 0.1)
    human = human_since(s, task, m.created_at if m else None)
    # R12: ponowny trening ma sens dopiero przy wyraznie wiekszym zbiorze (dawniej co kilkadziesiat przykladow -
    # 12 treningow, jedna przyjeta wersja) albo gdy czlowiek oznaczyl nowe przypadki (najcenniejsze przyklady)
    retrain = n_now >= spec.min_total and (
        m is None or human >= settings.nn_retrain_human
        or (new >= settings.nn_retrain_min_new and m.n_train and new / m.n_train >= settings.nn_retrain_growth))
    return {
        "key": task, "spec": spec, "mode": st["mode"], "model": m, "metrics": metrics, "human_new": human,
        "decisions": by_dec, "total": total, "saved": sum(by_dec.get(k, 0) for k in SAVED),
        "agree": agree, "disagree": disagree, "agreement": agreement,
        "audit": (audit_ok / audit_n) if audit_n else None, "audit_n": audit_n,
        "examples": n_now, "new": new, "conflicts": conflicts, "ready_active": ready, "retrain": retrain,
    }


def overview(days: int = 7) -> dict:
    with Session(engine) as s:
        tasks = [task_summary(s, t, days) for t in TASKS]
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        day = s.exec(select(func.count()).select_from(NNPrediction).where(NNPrediction.created_at >= since)).one()
        day_saved = s.exec(select(func.count()).select_from(NNPrediction).where(
            NNPrediction.created_at >= since, col(NNPrediction.decision).in_(SAVED))).one()
        ms = s.exec(select(func.avg(NNPrediction.ms)).where(NNPrediction.created_at >= since)).one()
    return {"tasks": tasks, "day": day, "day_saved": day_saved, "avg_ms": round(ms, 1) if ms else None,
            "saved": sum(t["saved"] for t in tasks), "conflicts": sum(t["conflicts"] for t in tasks),
            "total": sum(t["total"] for t in tasks), "days": days}


def readiness(ttl: float = 60.0) -> list[dict]:
    """Zadania, ktore warto (ponownie) wytrenowac - z pamiecia na minute (lista zadan odswieza sie czesto)."""
    hit = _cache.get("ready")
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    out = []
    with Session(engine) as s:
        for t in TASKS:
            sm = task_summary(s, t)
            if sm["retrain"]:
                out.append({"task": t, "title": TASKS[t].title, "examples": sm["examples"], "new": sm["new"],
                            "version": sm["model"].version if sm["model"] else None})
    _cache["ready"] = (time.time(), out)
    return out


def conflicts(limit: int = 100) -> tuple[list[NNPrediction], int]:
    with Session(engine) as s:
        base = select(NNPrediction).where(NNPrediction.decision == "konflikt", col(NNPrediction.human_label).is_(None))
        total = s.exec(select(func.count()).select_from(base.subquery())).one()
        rows = list(s.exec(base.order_by(col(NNPrediction.id).desc()).limit(limit)).all())
    return rows, total
