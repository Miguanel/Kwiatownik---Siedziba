"""Siec w potoku: bramka pewnosci i dziennik decyzji.

Tryby zadania (panel /nn):
  wylaczona - potok dziala jak dawniej
  cien      - siec odpowiada przy kazdym pytaniu, ale decyduje LLM; zapisujemy zgodnosc (pomiar przed wlaczeniem)
  aktywna   - siec decyduje sama, gdy pewnosc >= prog z walidacji i klasa jest dopuszczona; reszta idzie do LLM.
              NN_AUDIT_RATE decyzji (domyslnie 5%) i tak sprawdza LLM - biezaca, nieobciazona ocena trafnosci.
Zadania kontrolne (sekcja / czesc informacji) nie zastepuja LLM - porownuja jego odpowiedz z siecia:
zgodna / konflikt (do rozstrzygniecia w panelu) / uzupelnienie (LLM nie podal czesci rosliny, siec jest pewna).
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sqlmodel import Session

from app.config import settings
from app.db import engine
from app.models import NNPrediction
from app.nn import store
from app.nn.encoders import encode_texts, get_encoder
from app.nn.mlp import MLP
from app.nn.tasks import RARE, TASKS

log = logging.getLogger(__name__)


@dataclass
class Loaded:
    task: str
    version: int
    encoder: str
    labels: list[str]
    threshold: float
    allowed: set[str]
    model: MLP


@dataclass
class Decision:
    task: str
    version: int
    label: str
    confidence: float
    top: list[tuple[str, float]]
    mode: str
    decision: str
    use: bool = False             # potok ma uzyc etykiety sieci
    consult: bool = True          # potok ma zapytac LLM
    pred_id: int | None = None
    ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def method(self) -> str:
        return f"nn:{self.task}@v{self.version}"

    @property
    def fallback_ok(self) -> bool:
        """Odpowiedz sieci jako zapas, gdy LLM zawiodl (zamiast slabszych regul)."""
        return self.label != RARE and self.confidence >= settings.nn_fallback_confidence


def _job_id() -> int | None:
    try:
        from app.worker import snapshots
        return snapshots.current_job.get()
    except Exception:  # noqa: BLE001
        return None


class Brain:
    def __init__(self):
        self._loaded: dict[str, Loaded | None] = {}
        self._lock = threading.Lock()
        self.recent: deque = deque(maxlen=40)      # widok "Na zywo"
        self.errors: dict[str, str] = {}

    # ------------------------------------------------------------ modele
    def invalidate(self, task: str | None = None) -> None:
        with self._lock:
            if task:
                self._loaded.pop(task, None)
            else:
                self._loaded.clear()

    def loaded(self, task: str) -> Loaded | None:
        if task in self._loaded:
            return self._loaded[task]
        with self._lock:
            if task in self._loaded:
                return self._loaded[task]
            st = store.task_state(task)
            lm = None
            if st.get("version"):
                d = store.model_dir(task, int(st["version"]))
                try:
                    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                    if not get_encoder(meta["encoder"]).available():
                        raise RuntimeError(f"enkoder {meta['encoder']} niedostepny")
                    lm = Loaded(task, int(meta["version"]), meta["encoder"], list(meta["labels"]),
                                float(meta["threshold"]), set(meta.get("allowed") or []), MLP.load(d / "weights.npz"))
                    self.errors.pop(task, None)
                except Exception as exc:  # noqa: BLE001 - brak/uszkodzony model = potok bez sieci
                    self.errors[task] = str(exc)[:300]
                    log.warning("Siec %s v%s niedostepna: %s", task, st.get("version"), exc)
            self._loaded[task] = lm
            return lm

    def predict(self, task: str, texts: list[str]) -> tuple[list[tuple[str, float, list]], float, Loaded | None]:
        lm = self.loaded(task)
        if lm is None or not texts:
            return [], 0.0, lm
        t0 = time.perf_counter()
        X = encode_texts(get_encoder(lm.encoder), texts)
        proba = lm.model.predict_proba(X)
        ms = (time.perf_counter() - t0) * 1000 / len(texts)
        out = []
        for row in proba:
            order = np.argsort(-row)[:3]
            top = [(lm.labels[i], round(float(row[i]), 4)) for i in order]
            out.append((top[0][0], top[0][1], top))
        return out, round(ms, 1), lm

    # ------------------------------------------------------------ bramka (zadania decyzyjne)
    def gate(self, task: str, text: str, ref: str) -> Decision | None:
        """None = siec nie pracuje (wylaczona / brak modelu). Inaczej decyzja z polami use/consult."""
        if not settings.nn_enabled:
            return None
        mode = store.task_state(task)["mode"]
        if mode == "wylaczona":
            return None
        try:
            preds, ms, lm = self.predict(task, [text])
        except Exception as exc:  # noqa: BLE001 - blad sieci nigdy nie zatrzymuje potoku
            self.errors[task] = str(exc)[:300]
            log.warning("Siec %s: blad predykcji (%s) - potok bez sieci", task, exc)
            return None
        if not preds:
            return None
        label, conf, top = preds[0]
        confident = conf >= lm.threshold and label in lm.allowed
        d = Decision(task, lm.version, label, conf, top, mode, "cien", ms=ms)
        if mode == "aktywna" and confident:
            if random.random() < settings.nn_audit_rate:
                d.decision, d.use, d.consult = "kontrola", False, True
            else:
                d.decision, d.use, d.consult = "przyjeta", True, False
        elif mode == "aktywna":
            d.decision = "do_llm"
        d.pred_id = self._log(d, ref, text)
        return d

    def teach(self, d: Decision | None, teacher_label: str | None, teacher: str | None,
              decision: str | None = None) -> None:
        """Odpowiedz LLM dla decyzji sieci (zgodnosc w cieniu, kontrola losowa, przypadki oddane do LLM)."""
        if d is None or d.pred_id is None:
            return
        try:
            with Session(engine) as s:
                p = s.get(NNPrediction, d.pred_id)
                if not p:
                    return
                if teacher_label is not None:
                    p.teacher_label = teacher_label
                    p.teacher = (teacher or "")[:120]
                    p.agree = teacher_label == p.label
                if decision:
                    p.decision = decision
                s.add(p)
                s.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("Dziennik sieci: nie zapisano odpowiedzi LLM (%s)", exc)

    # ------------------------------------------------------------ kontrola (zadania kontrolne)
    def check(self, task: str, items: list[tuple[str, str, str | None]], teacher: str | None) -> list[Decision]:
        """items: (ref, tekst, etykieta LLM). Zwraca decyzje: zgodna / konflikt / uzupelnienie / cien."""
        if not settings.nn_enabled or not items:
            return []
        mode = store.task_state(task)["mode"]
        if mode == "wylaczona":
            return []
        try:
            preds, ms, lm = self.predict(task, [t for _, t, _ in items])
        except Exception as exc:  # noqa: BLE001
            self.errors[task] = str(exc)[:300]
            log.warning("Siec %s: blad predykcji (%s)", task, exc)
            return []
        out = []
        for (ref, text, given), (label, conf, top) in zip(items, preds):
            confident = conf >= lm.threshold and label in lm.allowed
            d = Decision(task, lm.version, label, conf, top, mode, "cien", consult=False, ms=ms)
            agree = given == label
            if mode == "aktywna":
                if agree:
                    d.decision = "zgodna"
                elif confident and given in (None, "brak") and label != "brak":
                    d.decision, d.use = "uzupelnienie", True
                elif confident:
                    d.decision = "konflikt"
                else:
                    d.decision = "zgodna" if agree else "niepewna"
            d.pred_id = self._log(d, ref, text, teacher_label=given, teacher=teacher)
            out.append(d)
        return out

    # ------------------------------------------------------------ dziennik
    def _log(self, d: Decision, ref: str, text: str, teacher_label: str | None = None,
             teacher: str | None = None) -> int | None:
        self.recent.append({"ts": time.time(), "task": d.task, "version": d.version, "label": d.label,
                            "conf": d.confidence, "decision": d.decision, "ms": d.ms})
        try:
            with Session(engine) as s:
                p = NNPrediction(task=d.task, version=d.version, ref=ref[:500], preview=(text or "")[:240],
                                 label=d.label, confidence=round(d.confidence, 4),
                                 top_json=json.dumps(d.top, ensure_ascii=False), mode=d.mode, decision=d.decision,
                                 teacher_label=teacher_label, teacher=(teacher or None),
                                 agree=(teacher_label == d.label) if teacher_label is not None else None,
                                 ms=d.ms, job_id=_job_id())
                s.add(p)
                s.commit()
                return p.id
        except Exception as exc:  # noqa: BLE001 - dziennik nie moze zatrzymac potoku
            log.warning("Dziennik sieci: nie zapisano decyzji (%s)", exc)
            return None

    def status(self) -> dict:
        return {t: {"state": store.task_state(t), "loaded": bool(self._loaded.get(t)), "error": self.errors.get(t)}
                for t in TASKS}


brain = Brain()


def record_sample(task: str, ref: str, text: str, label: str, origin: str, weight: float = 1.0,
                  group: str | None = None) -> None:
    """Zapamietuje przyklad treningowy (tekst + etykieta nauczyciela). Ten sam ref -> aktualizacja."""
    if not settings.nn_enabled or not text or not label:
        return
    from sqlmodel import select

    from app.models import NNSample
    try:
        with Session(engine) as s:
            row = s.exec(select(NNSample).where(NNSample.task == task, NNSample.ref == ref[:500])).first()
            if row and row.origin == "czlowiek" and origin != "czlowiek":
                return                                   # poprawki czlowieka nie nadpisujemy
            row = row or NNSample(task=task, ref=ref[:500], text="", label="", origin="")
            row.text, row.label, row.origin, row.weight, row.group = text[:6000], label, origin[:120], weight, group
            s.add(row)
            s.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("Nie zapisano przykladu treningowego %s (%s)", task, exc)
