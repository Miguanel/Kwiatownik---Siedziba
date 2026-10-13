"""Ocena modeli: walidacja krzyzowa z grupowaniem, metryki, kalibracja i dobor progu pewnosci.

Dlaczego grupy: informacje o tej samej roslinie (albo strony z tej samej witryny) sa do siebie podobne.
Gdyby trafialy jednoczesnie do treningu i testu, siec "pamietalaby" nazwe rosliny zamiast uczyc sie tresci,
a wynik bylby zawyzony. Dlatego test zawsze zawiera rosliny / witryny, ktorych model nie widzial.

Klasyfikacja selektywna (Geifman i El-Yaniv 2017): siec decyduje tylko wtedy, gdy jej skalibrowana pewnosc
przekracza prog; reszta idzie do LLM. Prog wybieramy na przewidywaniach spoza treningu (out-of-fold) tak,
zeby trafnosc decyzji podjetych przez siec byla >= NN_TARGET_ACCURACY przy jak najwiekszym pokryciu.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def cap_groups(groups: list[str], max_share: float = 0.08, min_size: int = 10) -> list[str]:
    """Bardzo duze grupy (np. jedna witryna = polowa zbioru) dzielone na podgrupy, zeby jedna czesc walidacji
    nie byla w calosci jedna witryna, a model trenowany bez niej - wtedy ocena mierzylaby tylko przesuniecie
    domeny, a trening tracilby wiekszosc danych. Podgrupy sa ciaglymi blokami (kolejnosc wstawiania)."""
    cap = max(min_size, int(len(groups) * max_share))
    seen: dict[str, int] = {}
    out = []
    for g in groups:
        k = seen.get(g, 0)
        seen[g] = k + 1
        out.append(g if k < cap else f"{g}#{k // cap}")
    return out


def group_kfold(groups: list[str], y: np.ndarray, k: int = 5, seed: int = 7) -> list[np.ndarray]:
    """Podzial na k czesci tak, by kazda grupa byla w jednej czesci; czesci podobnej wielkosci i z podobnym
    rozkladem klas (zachlannie: najwieksze grupy najpierw do czesci, ktorej najbardziej brakuje ich klasy)."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    k = max(2, min(k, len(by_group)))
    keys = list(by_group)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -len(by_group[g]))
    n_classes = int(y.max()) + 1 if len(y) else 1
    fold_cls = np.zeros((k, n_classes))
    fold_n = np.zeros(k)
    folds: list[list[int]] = [[] for _ in range(k)]
    total_cls = np.bincount(y, minlength=n_classes) + 1e-9
    for g in keys:
        rows = by_group[g]
        cls = np.bincount(y[rows], minlength=n_classes)
        # koszt: jak bardzo czesc jest juz "pelna" w klasach tej grupy + wielkosc czesci
        cost = (fold_cls[:, cls > 0] / total_cls[cls > 0]).sum(axis=1) + fold_n / max(len(y), 1)
        f = int(np.argmin(cost))
        folds[f] += rows
        fold_cls[f] += cls
        fold_n[f] += len(rows)
    return [np.array(sorted(f), dtype=np.int64) for f in folds if f]


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% przedzial ufnosci Wilsona dla trafnosci (rzetelny tez przy malych n)."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(max(0.0, (c - r) / d), 4), round(min(1.0, (c + r) / d), 4)


def per_class(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> list[dict]:
    out = []
    for c, name in enumerate(labels):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"label": name, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                    "support": int((y == c).sum())})
    return out


def confusion(y: np.ndarray, pred: np.ndarray, n: int) -> list[list[int]]:
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m.tolist()


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Oczekiwany blad kalibracji: srednia |trafnosc - pewnosc| w przedzialach pewnosci, wazona liczebnoscia."""
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return round(float(total), 4)


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            out.append({"od": round(float(lo), 2), "do": round(float(hi), 2), "n": int(sel.sum()),
                        "pewnosc": round(float(conf[sel].mean()), 3), "trafnosc": round(float(correct[sel].mean()), 3)})
    return out


def coverage_curve(conf: np.ndarray, correct: np.ndarray, points: int = 21) -> list[dict]:
    """Dla progow 0..1: jaka czesc przypadkow siec rozstrzyga (pokrycie) i z jaka trafnoscia."""
    out = []
    for t in np.linspace(0, 1, points):
        sel = conf >= t
        n = int(sel.sum())
        out.append({"prog": round(float(t), 2), "pokrycie": round(n / max(len(conf), 1), 3),
                    "trafnosc": round(float(correct[sel].mean()), 3) if n else None, "n": n})
    return out


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target: float, min_n: int = 20) -> dict:
    """Najnizszy prog (= najwieksze pokrycie), przy ktorym trafnosc przypadkow ponad progiem >= target.
    Brak takiego progu -> 1.01 (siec nie decyduje sama, pracuje tylko w cieniu)."""
    order = np.argsort(-conf)
    c_sorted, ok_sorted = conf[order], correct[order].astype(np.float64)
    cum = np.cumsum(ok_sorted)
    best = None
    for i in range(len(order)):
        n = i + 1
        if i + 1 < len(order) and c_sorted[i + 1] == c_sorted[i]:
            continue                                     # prog musi obejmowac wszystkie rowne pewnosci
        if n >= min_n and cum[i] / n >= target:
            best = i
    if best is None:
        return {"threshold": 1.01, "coverage": 0.0, "accuracy": None, "n": 0, "ci": [0.0, 0.0]}
    n = best + 1
    lo, hi = wilson(int(cum[best]), n)
    return {"threshold": round(float(c_sorted[best]), 4), "coverage": round(n / len(conf), 4),
            "accuracy": round(float(cum[best] / n), 4), "n": n, "ci": [lo, hi]}


def allowed_classes(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, threshold: float, labels: list[str],
                    target: float, never: set[str]) -> list[str]:
    """Klasy, ktore siec moze przyznawac sama: precyzja ponad progiem >= target - 0.05 i min. 5 przypadkow.
    Klasa o slabej precyzji (np. rzadki typ przepisu) zawsze idzie do LLM, nawet przy wysokiej pewnosci."""
    out = []
    sel = conf >= threshold
    for c, name in enumerate(labels):
        if name in never:
            continue
        m = sel & (pred == c)
        if m.sum() >= 5 and (y[m] == c).mean() >= target - 0.05:
            out.append(name)
    return out


def summarize(y: np.ndarray, proba: np.ndarray, labels: list[str], target: float, never: set[str]) -> dict:
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = (pred == y)
    pc = per_class(y, pred, labels)
    present = [x for x in pc if x["support"] > 0]
    thr = choose_threshold(conf, correct, target)
    allowed = allowed_classes(y, pred, conf, thr["threshold"], labels, target, never)
    sel = (conf >= thr["threshold"]) & np.isin(pred, [labels.index(a) for a in allowed])
    eff_n = int(sel.sum())
    eff = {"coverage": round(eff_n / max(len(y), 1), 4),
           "accuracy": round(float(correct[sel].mean()), 4) if eff_n else None,
           "ci": list(wilson(int(correct[sel].sum()), eff_n)), "n": eff_n}
    return {
        "n": int(len(y)), "accuracy": round(float(correct.mean()), 4) if len(y) else 0.0,
        "macro_f1": round(float(np.mean([x["f1"] for x in present])), 4) if present else 0.0,
        "ece": ece(conf, correct), "per_class": pc, "confusion": confusion(y, pred, len(labels)),
        "threshold": thr, "allowed": allowed, "effective": eff,
        "curve": coverage_curve(conf, correct), "reliability": reliability(conf, correct),
        "class_counts": dict(Counter(labels[i] for i in y.tolist())),
    }
