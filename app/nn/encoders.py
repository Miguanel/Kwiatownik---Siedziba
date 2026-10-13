"""Enkodery tekstu: tekst -> wektor cech dla glowicy MLP.

1. "e5"    - intfloat/multilingual-e5-small (12 warstw transformera, 384 wymiary, ~100 jezykow, w tym zh/ja/uk),
             eksport ONNX int8 (~120 MB) liczony przez onnxruntime na CPU. Zamrozony: przy kilkuset przykladach
             na klase dostrajanie calego transformera przeucza sie, a zamrozone osadzenia + mala glowica
             to sprawdzony, stabilny uklad (tzw. linear/MLP probing).
             Pobieranie: zadanie "nn_init" (przycisk w panelu /nn) -> DATA_DIR/nn/encoder/.
2. "ngram" - worek n-gramow znakow i slow z haszowaniem (Joulin i in. 2016, fastText). Nie wymaga pobierania,
             dziala w kazdym jezyku (n-gramy znakow obsluguja tez pismo chinskie i japonskie). Zapas,
             gdy enkodera e5 jeszcze nie ma, i punkt odniesienia przy porownaniu wersji.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

NGRAM_BUCKETS = 1 << 16          # liczba kubelkow haszowania (wiersze warstwy osadzen)
NGRAM_MAX_CHARS = 3000
_WORD = re.compile(r"\w+", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


# ------------------------------------------------------------------ rzadkie wektory (n-gramy)
@dataclass
class Sparse:
    """Partia rzadkich dokumentow w ukladzie CSR: idx/val wszystkich cech po kolei, off = granice dokumentow."""
    idx: np.ndarray      # int64
    val: np.ndarray      # float32
    off: np.ndarray      # int64, len = n + 1

    def __len__(self) -> int:
        return len(self.off) - 1

    @staticmethod
    def from_docs(docs: list[tuple[np.ndarray, np.ndarray]]) -> "Sparse":
        if not docs:
            return Sparse(np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(1, np.int64))
        lens = np.array([len(i) for i, _ in docs], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        return Sparse(np.concatenate([i for i, _ in docs]).astype(np.int64),
                      np.concatenate([v for _, v in docs]).astype(np.float32), off)

    def take(self, rows) -> "Sparse":
        rows = np.asarray(rows, dtype=np.int64)
        return Sparse.from_docs([(self.idx[self.off[r]:self.off[r + 1]], self.val[self.off[r]:self.off[r + 1]])
                                 for r in rows])

    def doc_ids(self) -> np.ndarray:
        return np.repeat(np.arange(len(self), dtype=np.int64), np.diff(self.off))


def take_rows(X, rows):
    """Wiersze macierzy gestej albo partii rzadkiej."""
    return X.take(rows) if isinstance(X, Sparse) else X[np.asarray(rows, dtype=np.int64)]


def n_rows(X) -> int:
    return len(X) if isinstance(X, Sparse) else X.shape[0]


class NgramEncoder:
    name = "ngram"
    dim = NGRAM_BUCKETS
    sparse = True

    def available(self) -> bool:
        return True

    @staticmethod
    def _features(text: str) -> list[str]:
        t = unicodedata.normalize("NFKC", (text or "")[:NGRAM_MAX_CHARS]).lower()
        toks = _WORD.findall(t)
        feats: list[str] = []
        prev = None
        for tok in toks[:600]:
            if _CJK.search(tok):                         # ciagi bez spacji: n-gramy znakow 1-3
                for n in (1, 2, 3):
                    feats += ["c" + tok[i:i + n] for i in range(max(0, len(tok) - n + 1))]
                prev = None
                continue
            feats.append("w" + tok)
            if prev:
                feats.append("b" + prev + " " + tok)
            prev = tok
            if len(tok) <= 30:
                w = f"<{tok}>"
                for n in (3, 4, 5):
                    feats += ["g" + w[i:i + n] for i in range(max(0, len(w) - n + 1))]
        return feats

    def encode_one(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        counts: dict[int, int] = {}
        for f in self._features(text):
            h = zlib.crc32(f.encode("utf-8")) % (NGRAM_BUCKETS - 1) + 1     # 0 = cecha "dokument" (bias)
            counts[h] = counts.get(h, 0) + 1
        idx = np.fromiter([0, *counts.keys()], dtype=np.int64, count=len(counts) + 1)
        val = np.fromiter([1.0, *(1.0 + math.log(c) for c in counts.values())], dtype=np.float32,
                          count=len(counts) + 1)
        val /= float(np.linalg.norm(val)) or 1.0
        return idx, val

    def encode(self, texts: list[str], progress=None) -> Sparse:
        docs = []
        for i, t in enumerate(texts):
            docs.append(self.encode_one(t))
            if progress and i % 200 == 0:
                progress(i, len(texts))
        return Sparse.from_docs(docs)

    def status(self) -> dict:
        return {"name": self.name, "present": True, "loaded": True, "detail": f"{NGRAM_BUCKETS} kubelkow haszowania"}


# ------------------------------------------------------------------ e5 (ONNX)
E5_FILES = {
    # plik docelowy -> kandydaci (repozytorium/sciezka) probowani po kolei
    "model.onnx": ["{repo}/resolve/main/onnx/model_quantized.onnx", "{alt}/resolve/main/onnx/model_quantized.onnx",
                   "{alt}/resolve/main/onnx/model.onnx", "{repo}/resolve/main/onnx/model.onnx"],
    "tokenizer.json": ["{repo}/resolve/main/tokenizer.json", "{alt}/resolve/main/tokenizer.json"],
    "sentencepiece.bpe.model": ["{alt}/resolve/main/sentencepiece.bpe.model",
                                "{repo}/resolve/main/sentencepiece.bpe.model"],
}
E5_OPTIONAL = {"sentencepiece.bpe.model"}        # wystarczy tokenizer.json albo model sentencepiece


def encoder_dir() -> Path:
    return settings.nn_root / "encoder" / "multilingual-e5-small"


class E5Encoder:
    name = "e5"
    dim = 384
    sparse = False
    max_len = 256
    batch = 16

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else encoder_dir()
        self._session = None
        self._tok = None
        self._sp = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def available(self) -> bool:
        if not (self.path / "model.onnx").exists():
            return False
        if (self.path / "tokenizer.json").exists():
            try:
                import tokenizers  # noqa: F401
                return True
            except ImportError:
                pass
        return (self.path / "sentencepiece.bpe.model").exists()

    # ---- ladowanie
    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, settings.nn_threads)
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(self.path / "model.onnx"), opts, providers=["CPUExecutionProvider"])
            tj = self.path / "tokenizer.json"
            if tj.exists():
                try:
                    from tokenizers import Tokenizer
                    tok = Tokenizer.from_file(str(tj))
                    tok.enable_truncation(self.max_len)
                    tok.no_padding()
                    self._tok = tok
                except ImportError:
                    self._tok = None
            if self._tok is None:
                import sentencepiece as spm
                self._sp = spm.SentencePieceProcessor(model_file=str(self.path / "sentencepiece.bpe.model"))
            self._inputs = {i.name for i in sess.get_inputs()}
            self._session = sess

    def _ids(self, text: str) -> list[int]:
        if self._tok is not None:
            return self._tok.encode(text).ids
        # XLM-R w ukladzie fairseq: <s>=0 <pad>=1 </s>=2 <unk>=3, identyfikatory sentencepiece przesuniete o 1
        pieces = [3 if i == 0 else i + 1 for i in self._sp.encode(text, out_type=int)]
        return [0, *pieces[: self.max_len - 2], 2]

    # ---- kodowanie
    def encode(self, texts: list[str], progress=None) -> np.ndarray:
        self._load()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])          # podobne dlugosci razem = mniej wypelnienia
        for b in range(0, len(texts), self.batch):
            rows = order[b:b + self.batch]
            ids = [self._ids("query: " + (texts[r] or "")[:4000]) for r in rows]
            width = max(len(x) for x in ids)
            input_ids = np.ones((len(ids), width), dtype=np.int64)          # 1 = <pad>
            mask = np.zeros((len(ids), width), dtype=np.int64)
            for k, x in enumerate(ids):
                input_ids[k, :len(x)] = x
                mask[k, :len(x)] = 1
            feed = {"input_ids": input_ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            hidden = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            out[rows] = mean_pool(hidden, mask)
            if progress:
                progress(min(b + self.batch, len(texts)), len(texts))
        return out

    def status(self) -> dict:
        present = self.available()
        size = (self.path / "model.onnx").stat().st_size // 1_000_000 if present else 0
        return {"name": self.name, "present": present, "loaded": self._session is not None, "path": str(self.path),
                "detail": f"ONNX {size} MB, 384 wymiary" if present else "brak - pobierz w panelu /nn",
                "error": self.error}


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Usrednienie wektorow tokenow (bez wypelnienia) + normalizacja L2 - jak w e5 / sentence-transformers."""
    m = mask[..., None].astype(np.float32)
    v = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    return (v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)).astype(np.float32)


# ------------------------------------------------------------------ wybor i pamiec podreczna
_ENCODERS: dict[str, object] = {}


def get_encoder(name: str):
    if name not in _ENCODERS:
        _ENCODERS[name] = E5Encoder() if name == "e5" else NgramEncoder()
    return _ENCODERS[name]


def default_encoder_name() -> str:
    """NN_ENCODER=auto -> e5, gdy pobrany i onnxruntime jest zainstalowany; inaczej n-gramy."""
    want = (settings.nn_encoder or "auto").lower()
    if want in ("ngram", "e5"):
        return want
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "ngram"
    return "e5" if get_encoder("e5").available() else "ngram"


def text_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Osadzenia e5 policzone przy poprzednich treningach (klucz: sha1 tekstu) - ponowny trening jest szybki."""

    def __init__(self, encoder_name: str):
        self.path = settings.nn_root / "cache" / f"{encoder_name}.npz"
        self.data: dict[str, np.ndarray] = {}
        if self.path.exists():
            try:
                z = np.load(self.path, allow_pickle=False)
                self.data = dict(zip(z["keys"].tolist(), z["vecs"]))
            except Exception as exc:  # noqa: BLE001 - uszkodzony plik = liczymy od nowa
                log.warning("Pamiec osadzen %s nieczytelna (%s) - licze od nowa", self.path, exc)

    def encode(self, encoder, texts: list[str], progress=None) -> np.ndarray:
        keys = [text_key(t) for t in texts]
        missing = sorted({k: i for i, k in enumerate(keys) if k not in self.data}.values())
        if missing:
            vecs = encoder.encode([texts[i] for i in missing], progress)
            for i, v in zip(missing, vecs):
                self.data[keys[i]] = v
        return np.stack([self.data[k] for k in keys]) if keys else np.zeros((0, encoder.dim), np.float32)

    def save(self, keep: int = 60000) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = list(self.data.items())[-keep:]
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array([k for k, _ in items]), vecs=np.stack([v for _, v in items]) if items
                 else np.zeros((0, 384), np.float32))
        tmp.replace(self.path)


def encode_texts(encoder, texts: list[str], cache: EmbeddingCache | None = None, progress=None):
    if encoder.sparse:
        return encoder.encode(texts, progress)
    return cache.encode(encoder, texts, progress) if cache else encoder.encode(texts, progress)


# ------------------------------------------------------------------ pobieranie e5
def _has_tokenizers() -> bool:
    try:
        import tokenizers  # noqa: F401
        return True
    except ImportError:
        return False


async def _fetch(client, url: str, fname: str, progress, attempts: int = 5) -> tuple[Path | None, str]:
    """Pobiera jeden plik do katalogu tymczasowego KONTENERA (nie na dysk Windows podpiety do Dockera - tam duze
    zapisy potrafia konczyc sie OSError EIO). Przerwane pobieranie wznawia od miejsca przerwania (Range).
    Zwraca (plik tymczasowy, '') albo (None, powod)."""
    import asyncio
    import tempfile

    import httpx
    tmp = Path(tempfile.gettempdir()) / f"siedziba-nn-{text_key(url)[:10]}-{fname}.part"   # osobno dla kazdego adresu
    last = ""
    for attempt in range(attempts):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": settings.user_agent, **({"Range": f"bytes={done}-"} if done else {})}
        try:
            async with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 416 and done:          # juz caly plik
                    return tmp, ""
                if r.status_code not in (200, 206):
                    return None, f"{url}: HTTP {r.status_code}"
                if r.status_code == 200:
                    done = 0                                 # serwer nie wznawia - od poczatku
                total = done + int(r.headers.get("content-length") or 0)
                with open(tmp, "ab" if done else "wb") as fh:
                    async for chunk in r.aiter_bytes(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(fname, done, total)
            if total and done < total:
                raise OSError(f"przerwane po {done // 1_000_000}/{total // 1_000_000} MB")
            return tmp, ""
        except (httpx.HTTPError, OSError) as exc:
            last = f"{url}: {type(exc).__name__}: {exc}"
            log.warning("Pobieranie %s - proba %d/%d nieudana (%s), wznawiam", fname, attempt + 1, attempts, exc)
            await asyncio.sleep(settings.nn_retry_wait_s * (attempt + 1))
    return None, last


def _install(tmp: Path, final: Path, attempts: int = 4) -> None:
    """Kopia z katalogu tymczasowego do DATA_DIR/nn/encoder (z ponowieniem przy chwilowym bledzie dysku)."""
    import shutil
    import time as _time
    part = final.with_name(final.name + ".kopiowanie")
    for attempt in range(attempts):
        try:
            shutil.copyfile(tmp, part)
            if part.stat().st_size != tmp.stat().st_size:
                raise OSError("niepelna kopia")
            part.replace(final)
            tmp.unlink(missing_ok=True)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            log.warning("Zapis %s nieudany (%s) - ponawiam", final.name, exc)
            _time.sleep(2 * (attempt + 1))


async def download_e5(progress=None, repo: str | None = None, alt: str | None = None) -> dict[str, str]:
    """Pobiera pliki enkodera z Hugging Face do DATA_DIR/nn/encoder/. Zwraca {plik: adres, z ktorego pobrano}."""
    import asyncio

    import httpx
    repo = "https://huggingface.co/" + (repo or settings.nn_e5_repo)
    alt = "https://huggingface.co/" + (alt or settings.nn_e5_alt_repo)
    dest = encoder_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.part"):                 # resztki z dawnych prob (zapis prosto na dysk Windows)
        try:
            old.unlink()
        except OSError:
            pass
    got: dict[str, str] = {}
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30, read=60)) as client:
        for fname, cands in E5_FILES.items():
            if (dest / fname).exists():
                got[fname] = "juz bylo"
                continue
            if fname in E5_OPTIONAL and "tokenizer.json" in got and _has_tokenizers():
                continue
            last = ""
            for tpl in cands:
                url = tpl.format(repo=repo, alt=alt)
                tmp, last = await _fetch(client, url, fname, progress)
                if tmp is None:
                    continue
                await asyncio.to_thread(_install, tmp, dest / fname)
                got[fname] = url
                break
            if fname not in got and fname not in E5_OPTIONAL:
                raise RuntimeError(f"Nie udalo sie pobrac {fname} ({last})")
    _ENCODERS.pop("e5", None)          # nowy plik = nowa sesja ONNX
    return got
