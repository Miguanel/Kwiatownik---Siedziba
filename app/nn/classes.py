"""Etap 7: siec decyduje sama tylko w klasach, w ktorych sprawdzila sie w potoku (zgodnosc z LLM).

Dla zadan-decyzji (page_kind, recipe_type): z dziennika NNPrediction (decyzje z pewnoscia >= progu modelu, dla
ktorych znana jest odpowiedz LLM) liczona jest zgodnosc kazdej klasy i dolna granica przedzialu Wilsona (95%).
Klasa z co najmniej NN_CLASS_MIN_N przypadkami i dolna granica >= NN_CLASS_MIN_LB trafia do "active_classes"
(state.json) - w trybie cienia siec decyduje wtedy sama tylko w tej klasie (z kontrola losowa NN_AUDIT_RATE).
Spadek ponizej NN_CLASS_DROP_LB - klasa wraca do LLM. Klasa "other" (nic ciekawego) nigdy nie jest wlaczana:
jej bledy oznaczaja utracone strony.
"""
import logging
import math

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.models import NNModel, NNPrediction
from app.nn import store

log = logging.getLogger(__name__)
DECISION_TASKS = ("page_kind", "recipe_type")
NEVER = {"page_kind": {"other"}}


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


def class_quality(task: str, db_engine=None) -> dict[str, dict]:
    """{klasa: {n, zgodne, zgodnosc, dolna}} dla decyzji z pewnoscia >= progu biezacego modelu."""
    db_engine = db_engine or engine
    with Session(db_engine) as s:
        model = s.exec(select(NNModel).where(NNModel.task == task, NNModel.status == "biezaca")
                       .order_by(col(NNModel.id).desc())).first()
        thr = model.threshold if model and model.threshold else 0.95
        rows = s.exec(select(NNPrediction.label, NNPrediction.agree).where(
            NNPrediction.task == task, col(NNPrediction.agree).is_not(None), NNPrediction.confidence >= thr)).all()
    out: dict[str, dict] = {}
    for label, agree in rows:
        q = out.setdefault(label, {"n": 0, "zgodne": 0})
        q["n"] += 1
        q["zgodne"] += 1 if agree else 0
    for q in out.values():
        q["zgodnosc"] = round(q["zgodne"] / q["n"], 3)
        q["dolna"] = round(wilson_lower(q["zgodne"], q["n"]), 3)
    return out


def update_active_classes(db_engine=None) -> dict[str, list[str]]:
    """Przelicza klasy aktywne dla zadan-decyzji i zapisuje je w state.json. Zwraca {zadanie: [klasy]}."""
    result = {}
    for task in DECISION_TASKS:
        quality = class_quality(task, db_engine)
        cur = set(store.task_state(task).get("active_classes") or [])
        new = set()
        for label, q in quality.items():
            if label in NEVER.get(task, set()):
                continue
            if q["n"] >= settings.nn_class_min_n and q["dolna"] >= settings.nn_class_min_lb:
                new.add(label)
            elif label in cur and q["dolna"] >= settings.nn_class_drop_lb:
                new.add(label)                        # histereza: raz wlaczona klasa nie miga
        if new != cur:
            log.warning("Siec %s: klasy decydowane samodzielnie %s -> %s", task, sorted(cur), sorted(new))
            store.set_task_state(task, active_classes=sorted(new))
        result[task] = sorted(new)
    return result
