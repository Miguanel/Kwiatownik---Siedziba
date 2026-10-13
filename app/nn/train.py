"""Trening wersji sieci: zbior -> osadzenia -> walidacja krzyzowa z grupami -> model koncowy -> mistrz/pretendent.

Procedura (dla kazdego zadania i enkodera):
  1. Zbior z bazy (tasks.py); rzadkie klasy -> "_inne" (siec ich nie przyznaje, zawsze LLM).
  2. Osadzenia tekstow (e5 z pamiecia podreczna albo n-gramy).
  3. k-krotna walidacja krzyzowa z grupami (roslina / witryna / zrodlo): kazdy przyklad dostaje przewidywanie
     modelu, ktory nie widzial ani jego, ani jego grupy -> uczciwa ocena + dobor progu pewnosci.
  4. Model koncowy na wszystkich danych (z grupowa walidacja 15% do wczesnego zatrzymania i kalibracji).
  5. Mistrz/pretendent: nowa wersja zastepuje aktywna tylko, gdy nie jest gorsza (ocena = srednia z macro-F1
     i pokrycia przy docelowej trafnosci). Przy NN_ENCODER=auto i pobranym e5 trenowane sa oba enkodery
     i wygrywa lepszy.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.models import NNModel
from app.nn import store
from app.nn.encoders import EmbeddingCache, default_encoder_name, encode_texts, get_encoder, take_rows
from app.nn.evaluate import cap_groups, group_kfold, summarize
from app.nn.mlp import MLP
from app.nn.tasks import RARE, TASKS

log = logging.getLogger(__name__)


class NotEnoughData(Exception):
    pass


@dataclass
class Prepared:
    labels: list[str]
    y: np.ndarray
    w: np.ndarray
    groups: list[str]
    texts: list[str]
    merged: dict
    counts: dict
    origins: dict


def prepare(task: str) -> Prepared:
    spec = TASKS[task]
    with Session(engine) as s:
        ds = spec.build(s)
    merged = ds.merge_rare(spec.min_per_class, spec.allow_rare)
    counts = Counter(ds.labels)
    real = [c for c in counts if c != RARE]
    if len(ds) < spec.min_total or len(real) < spec.min_classes:
        raise NotEnoughData(f"{spec.title}: {len(ds)} przykladow, {len(real)} klas "
                            f"(potrzeba min. {spec.min_total} przykladow i {spec.min_classes} klas po "
                            f"{spec.min_per_class}+)")
    labels = sorted(counts)
    return Prepared(labels, np.array([labels.index(x) for x in ds.labels], dtype=np.int64),
                    np.array(ds.weights, dtype=np.float32), cap_groups(ds.groups), ds.texts, merged, dict(counts),
                    dict(Counter(o.split(":")[0] if o.startswith("llm") else o for o in ds.origins)))


def inner_split(rows: np.ndarray, groups: list[str], share: float = 0.15, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Czesc walidacyjna z calych grup (do wczesnego zatrzymania i kalibracji temperatura)."""
    rng = np.random.default_rng(seed)
    by_g: dict[str, list[int]] = {}
    for r in rows.tolist():
        by_g.setdefault(groups[r], []).append(r)
    keys = list(by_g)
    rng.shuffle(keys)
    val: list[int] = []
    target = max(1, int(len(rows) * share))
    for g in keys:
        if len(val) >= target or len(by_g) < 3:
            break
        if len(val) + len(by_g[g]) > 2 * target and val:
            continue
        val += by_g[g]
    if not val or len(val) == len(rows):                  # za malo grup - losowo
        perm = rng.permutation(rows)
        return perm[target:], perm[:target]
    vs = set(val)
    return np.array([r for r in rows.tolist() if r not in vs], dtype=np.int64), np.array(sorted(val), dtype=np.int64)


def _score(m: dict | None) -> float:
    if not m:
        return -1.0
    return round(0.5 * (m.get("macro_f1") or 0) + 0.5 * ((m.get("effective") or {}).get("coverage") or 0), 4)


def fit_and_evaluate(p: Prepared, encoder_name: str, log_fn=None, should_stop=None, progress=None):
    """Zwraca (model koncowy, metryki out-of-fold, raport uczenia, sekundy)."""
    t0 = time.monotonic()
    enc = get_encoder(encoder_name)
    cache = None if enc.sparse else EmbeddingCache(encoder_name)
    if log_fn:
        log_fn(f"[{encoder_name}] osadzenia {len(p.texts)} tekstow...")
    X = encode_texts(enc, p.texts, cache, progress)
    if cache:
        cache.save()
    folds = group_kfold(p.groups, p.y, k=5)
    oof = np.zeros((len(p.y), len(p.labels)), dtype=np.float32)
    all_rows = np.arange(len(p.y))
    for i, test in enumerate(folds):
        train = np.setdiff1d(all_rows, test)
        tr, va = inner_split(train, p.groups)
        m = MLP(enc.dim, len(p.labels), enc.sparse)
        m.fit(take_rows(X, tr), p.y[tr], p.w[tr], take_rows(X, va), p.y[va], p.w[va], should_stop=should_stop)
        oof[test] = m.predict_proba(take_rows(X, test))
        if log_fn:
            acc = float((oof[test].argmax(1) == p.y[test]).mean()) if len(test) else 0
            log_fn(f"[{encoder_name}] czesc {i + 1}/{len(folds)}: {len(test)} przykladow testowych, trafnosc {acc:.1%}")
        if should_stop and should_stop():
            raise InterruptedError("zatrzymano")
    metrics = summarize(p.y, oof, p.labels, settings.nn_target_accuracy, never={RARE})
    metrics["folds"] = len(folds)
    metrics["groups"] = len(set(p.groups))
    tr, va = inner_split(all_rows, p.groups)
    final = MLP(enc.dim, len(p.labels), enc.sparse)
    rep = final.fit(take_rows(X, tr), p.y[tr], p.w[tr], take_rows(X, va), p.y[va], p.w[va], should_stop=should_stop)
    return final, metrics, rep, round(time.monotonic() - t0, 1)


def _save(task: str, encoder_name: str, p: Prepared, model: MLP, metrics: dict, rep, secs: float,
          job_id: int | None) -> NNModel:
    with Session(engine) as s:
        last = s.exec(select(NNModel.version).where(NNModel.task == task).order_by(col(NNModel.version).desc())).first()
        version = (last or 0) + 1
        d = store.model_dir(task, version)
        d.mkdir(parents=True, exist_ok=True)
        model.save(d / "weights.npz")
        meta = {"task": task, "version": version, "encoder": encoder_name, "labels": p.labels,
                "threshold": metrics["threshold"]["threshold"], "allowed": metrics["allowed"],
                "temperature": model.temperature, "n_train": len(p.y), "merged_rare": p.merged,
                "fit": {"epochs": rep.epochs, "best_epoch": rep.best_epoch, "train_loss": rep.train_loss,
                        "val_loss": rep.val_loss}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        metrics = {**metrics, "fit": meta["fit"], "merged_rare": p.merged, "temperature": model.temperature,
                   "origins": p.origins}
        row = NNModel(task=task, version=version, encoder=encoder_name, n_train=len(p.y),
                      labels_json=json.dumps(p.labels, ensure_ascii=False),
                      metrics_json=json.dumps(metrics, ensure_ascii=False),
                      threshold=float(metrics["threshold"]["threshold"]),
                      allowed_json=json.dumps(metrics["allowed"], ensure_ascii=False), job_id=job_id, seconds=secs)
        s.add(row)
        s.commit()
        s.refresh(row)
        return row


def _decide(task: str, rows: list[NNModel]) -> tuple[NNModel, str]:
    """Mistrz/pretendent. Zwraca (zwyciezca, opis decyzji); statusy zapisane w bazie, stan zadania zaktualizowany."""
    st = store.task_state(task)
    with Session(engine) as s:
        champ = s.exec(select(NNModel).where(NNModel.task == task, NNModel.status == "biezaca")).first()
        cands = [s.get(NNModel, r.id) for r in rows]
        cands.sort(key=lambda r: -_score(json.loads(r.metrics_json or "{}")))
        best = cands[0]
        for r in cands[1:]:
            r.status = "odrzucona"
            r.note = (f"slabsza od v{best.version} ({best.encoder}) w tym samym treningu: ocena "
                      f"{_score(json.loads(r.metrics_json))} < {_score(json.loads(best.metrics_json))}")
            s.add(r)
        new_score = _score(json.loads(best.metrics_json))
        champ_ok = champ is not None and (champ.encoder != "e5" or get_encoder("e5").available())
        if champ_ok:
            old_score = _score(json.loads(champ.metrics_json or "{}"))
            if new_score >= old_score - 0.01:
                champ.status = "wycofana"
                champ.note = f"zastapiona przez v{best.version} (ocena {new_score} vs {old_score})"
                best.status = "biezaca"
                best.note = f"zastapila v{champ.version}: ocena {new_score} (byla {old_score})"
                s.add(champ)
            else:
                best.status = "odrzucona"
                best.note = f"gorsza od biezacej v{champ.version}: ocena {new_score} < {old_score}"
        else:
            if champ is not None:
                champ.status = "wycofana"
                champ.note = "enkoder niedostepny"
                s.add(champ)
            best.status = "biezaca"
            best.note = f"pierwsza wersja (ocena {new_score})"
        s.add(best)
        s.commit()
        winner = best if best.status == "biezaca" else champ
        note = best.note
        winner_id = winner.id
        winner_version = winner.version
    if winner_id == best.id:
        store.set_task_state(task, version=winner_version,
                             mode=st["mode"] if st["version"] else "cien")      # nowa siec zaczyna w cieniu
    from app.nn.runtime import brain
    brain.invalidate(task)
    with Session(engine) as s:
        return s.get(NNModel, winner_id), note


def train_task(task: str, encoder: str | None = None, log_fn=None, should_stop=None, progress=None,
               job_id: int | None = None) -> tuple[NNModel, str]:
    p = prepare(task)
    if log_fn:
        log_fn(f"{TASKS[task].title}: {len(p.y)} przykladow, klasy: " +
               ", ".join(f"{k} {v}" for k, v in sorted(p.counts.items(), key=lambda x: -x[1])) +
               (f"; rzadkie polaczone w '{RARE}': {', '.join(p.merged)}" if p.merged else ""))
    names = [encoder] if encoder else [default_encoder_name()]
    if not encoder and names[0] == "e5":
        names.append("ngram")                           # turniej enkoderow: wygrywa lepszy
    rows = []
    for name in names:
        model, metrics, rep, secs = fit_and_evaluate(p, name, log_fn, should_stop, progress)
        row = _save(task, name, p, model, metrics, rep, secs, job_id)
        eff = metrics["effective"]
        if log_fn:
            log_fn(f"[{name}] v{row.version}: trafnosc {metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f}, "
                   f"ECE {metrics['ece']:.3f}; prog {metrics['threshold']['threshold']} -> siec rozstrzyga "
                   f"{eff['coverage']:.0%} przypadkow z trafnoscia "
                   + (f"{eff['accuracy']:.1%} (95% CI {eff['ci'][0]:.1%}-{eff['ci'][1]:.1%})" if eff["accuracy"]
                      else "- (za malo pewnych przypadkow, tylko cien)") + f", {secs} s")
        rows.append(row)
    return _decide(task, rows)
