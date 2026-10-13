"""Glowica MLP w czystym numpy (bez PyTorch - obraz Dockera zostaje lekki).

Architektura: wejscie -> warstwa ukryta (ReLU, dropout) -> softmax.
  - wejscie geste (osadzenia e5, 384 wymiary): zwykla warstwa liniowa
  - wejscie rzadkie (n-gramy): warstwa osadzen z sumowaniem (EmbeddingBag, jak w fastText)
Uczenie: AdamW (Loshchilov i Hutter 2019), entropia krzyzowa z wagami klas (zrownowazenie rzadkich klas)
i wagami przykladow (etykieta czlowieka > LLM > reguly), wygladzanie etykiet (Szegedy i in. 2016),
wczesne zatrzymanie na stracie walidacyjnej z powrotem do najlepszych wag.
Kalibracja: skalowanie temperatura na zbiorze walidacyjnym (Guo i in. 2017) - pewnosc 0.9 ma znaczyc
"w 9 przypadkach na 10 trafnie", bo od tego zalezy, kiedy siec moze zastapic LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.nn.encoders import Sparse, n_rows, take_rows


@dataclass
class TrainConfig:
    hidden: int = 128
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    batch: int = 32
    max_epochs: int = 80
    patience: int = 8
    seed: int = 13
    class_weight_clip: tuple[float, float] = (0.3, 6.0)


def sparse_config() -> TrainConfig:
    return TrainConfig(hidden=64, dropout=0.1, lr=8e-3, weight_decay=2e-5, max_epochs=30, patience=4)


@dataclass
class FitReport:
    epochs: int = 0
    best_epoch: int = 0
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    temperature: float = 1.0


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, n_in: int, n_classes: int, sparse: bool, cfg: TrainConfig | None = None):
        self.cfg = cfg or (sparse_config() if sparse else TrainConfig())
        self.n_in, self.n_classes, self.sparse = n_in, n_classes, sparse
        self.temperature = 1.0
        rng = np.random.default_rng(self.cfg.seed)
        h = self.cfg.hidden
        std1 = 0.1 if sparse else np.sqrt(2.0 / n_in)
        self.p = {
            "W1": (rng.standard_normal((n_in, h)) * std1).astype(np.float32),
            "b1": np.zeros(h, np.float32),
            "W2": (rng.standard_normal((h, n_classes)) * np.sqrt(2.0 / (h + n_classes))).astype(np.float32),
            "b2": np.zeros(n_classes, np.float32),
        }

    # ------------------------------------------------------------ przebieg w przod / w tyl
    def _layer1(self, X) -> np.ndarray:
        if isinstance(X, Sparse):
            if len(X) == 0:
                return np.zeros((0, self.cfg.hidden), np.float32) + self.p["b1"]
            rows = self.p["W1"][X.idx] * X.val[:, None]
            return np.add.reduceat(rows, X.off[:-1], axis=0) + self.p["b1"]     # kazdy dokument ma ceche 0
        return X @ self.p["W1"] + self.p["b1"]

    def logits(self, X, train: bool = False, rng=None):
        z1 = self._layer1(X)
        a1 = np.maximum(z1, 0)
        mask = None
        if train and self.cfg.dropout > 0:
            mask = (rng.random(a1.shape) >= self.cfg.dropout).astype(np.float32) / (1 - self.cfg.dropout)
            a1 = a1 * mask
        return a1 @ self.p["W2"] + self.p["b2"], (z1, a1, mask)

    def _grads(self, X, y, w, rng) -> tuple[float, dict]:
        out, (z1, a1, mask) = self.logits(X, train=True, rng=rng)
        prob = softmax(out)
        n, c = prob.shape
        eps = self.cfg.label_smoothing
        target = np.full((n, c), eps / c, np.float32)
        target[np.arange(n), y] += 1 - eps
        wsum = float(w.sum()) or 1.0
        loss = float(-(w * (target * np.log(prob + 1e-9)).sum(axis=1)).sum() / wsum)
        d_out = (prob - target) * (w / wsum)[:, None]
        g = {"W2": a1.T @ d_out, "b2": d_out.sum(axis=0)}
        d_a1 = d_out @ self.p["W2"].T
        if mask is not None:
            d_a1 *= mask
        d_z1 = d_a1 * (z1 > 0)
        g["b1"] = d_z1.sum(axis=0)
        if isinstance(X, Sparse):
            # gradient tylko dla wierszy (n-gramow) obecnych w partii - jak SparseAdam w PyTorch
            rows, inv = np.unique(X.idx, return_inverse=True)
            contrib = d_z1[X.doc_ids()] * X.val[:, None]
            gw = np.stack([np.bincount(inv, weights=contrib[:, j], minlength=len(rows))
                           for j in range(contrib.shape[1])], axis=1).astype(np.float32)   # szybciej niz add.at
            g["W1"] = (rows, gw)
        else:
            g["W1"] = X.T @ d_z1
        return loss, g

    def loss(self, X, y, w=None, temperature: float = 1.0) -> float:
        if n_rows(X) == 0:
            return 0.0
        z, _ = self.logits(X)
        prob = softmax(z / temperature)
        w = np.ones(len(y), np.float32) if w is None else w
        return float(-(w * np.log(prob[np.arange(len(y)), y] + 1e-9)).sum() / (w.sum() or 1.0))

    # ------------------------------------------------------------ uczenie
    def class_weights(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
        cw = np.where(counts > 0, len(y) / (self.n_classes * np.maximum(counts, 1)), 0.0)
        return np.clip(cw, *self.cfg.class_weight_clip).astype(np.float32)

    def fit(self, X, y: np.ndarray, sample_w: np.ndarray | None, X_val=None, y_val=None, w_val=None,
            should_stop=None) -> FitReport:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        y = np.asarray(y, dtype=np.int64)
        sw = np.ones(len(y), np.float32) if sample_w is None else np.asarray(sample_w, np.float32)
        w = sw * self.class_weights(y)[y]
        has_val = X_val is not None and len(y_val)
        if has_val:
            y_val = np.asarray(y_val, dtype=np.int64)
            wv = (np.ones(len(y_val), np.float32) if w_val is None else np.asarray(w_val, np.float32))
            wv = wv * self.class_weights(y)[y_val]
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, step = 0.9, 0.999, 0
        rep = FitReport()
        best, best_p, bad = np.inf, None, 0
        n = len(y)
        for epoch in range(cfg.max_epochs):
            order = rng.permutation(n)
            losses = []
            for s in range(0, n, cfg.batch):
                rows = order[s:s + cfg.batch]
                loss, g = self._grads(take_rows(X, rows), y[rows], w[rows], rng)
                losses.append(loss)
                step += 1
                lr_t = cfg.lr * np.sqrt(1 - b2 ** step) / (1 - b1 ** step)
                for k in self.p:
                    if isinstance(g[k], tuple):                               # rzadki gradient osadzen
                        r, gr = g[k]
                        mr = b1 * m[k][r] + (1 - b1) * gr
                        vr = b2 * v[k][r] + (1 - b2) * gr * gr
                        m[k][r], v[k][r] = mr, vr
                        self.p[k][r] = (self.p[k][r] * (1 - cfg.lr * cfg.weight_decay)
                                        - lr_t * mr / (np.sqrt(vr) + 1e-8)).astype(np.float32)
                        continue
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k]
                    if k.startswith("W"):
                        self.p[k] *= (1 - cfg.lr * cfg.weight_decay)          # AdamW: zanik wag oddzielnie
                    self.p[k] -= (lr_t * m[k] / (np.sqrt(v[k]) + 1e-8)).astype(np.float32)
            rep.train_loss.append(round(float(np.mean(losses)), 4))
            rep.epochs = epoch + 1
            if not has_val:
                continue
            vl = self.loss(X_val, y_val, wv)
            rep.val_loss.append(round(vl, 4))
            if vl < best - 1e-4:
                best, bad, rep.best_epoch = vl, 0, epoch + 1
                best_p = {k: a.copy() for k, a in self.p.items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
            if should_stop and should_stop():
                break
        if best_p is not None:
            self.p = best_p
        if has_val:
            self.temperature = self.fit_temperature(X_val, y_val)
        rep.temperature = self.temperature
        return rep

    def fit_temperature(self, X, y) -> float:
        """Jedna liczba T minimalizujaca NLL na walidacji: p = softmax(z / T). Nie zmienia predykcji, tylko pewnosc."""
        if n_rows(X) < 5:
            return 1.0
        z, _ = self.logits(X)
        best_t, best_l = 1.0, np.inf
        for t in np.exp(np.linspace(np.log(0.25), np.log(6.0), 80)):
            p = softmax(z / t)
            nll = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
            if nll < best_l:
                best_t, best_l = float(t), nll
        return round(best_t, 4)

    # ------------------------------------------------------------ predykcja i zapis
    def predict_proba(self, X) -> np.ndarray:
        if n_rows(X) == 0:
            return np.zeros((0, self.n_classes), np.float32)
        z, _ = self.logits(X)
        return softmax(z / self.temperature)

    def save(self, path) -> None:
        np.savez_compressed(path, **self.p, meta=np.array([self.n_in, self.n_classes, int(self.sparse),
                                                           self.cfg.hidden], dtype=np.int64),
                            temperature=np.array([self.temperature], dtype=np.float32))

    @staticmethod
    def load(path) -> "MLP":
        z = np.load(path, allow_pickle=False)
        n_in, n_classes, sparse, hidden = (int(x) for x in z["meta"])
        cfg = sparse_config() if sparse else TrainConfig()
        cfg.hidden = hidden
        m = MLP.__new__(MLP)
        m.cfg, m.n_in, m.n_classes, m.sparse = cfg, n_in, n_classes, bool(sparse)
        m.p = {k: z[k] for k in ("W1", "b1", "W2", "b2")}
        m.temperature = float(z["temperature"][0])
        return m
