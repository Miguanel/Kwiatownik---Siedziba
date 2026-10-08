"""Szybkie tlumaczenie maszynowe bez LLM: NLLB-200 (Meta) w CTranslate2, lokalnie (CPU int8 albo GPU).

Model trzeba raz przygotowac (pobranie z Hugging Face + konwersja do CTranslate2):
    docker compose --profile mt run --rm nllb-init
Trafia do ./models/<nazwa> (w kontenerze /models/<nazwa>): model.bin, config.json, shared_vocabulary.*,
sentencepiece.bpe.model. Bez modelu Siedziba tlumaczy jak wczesniej - przez LLM.

NLLB tlumaczy bezposrednio miedzy 200 jezykami (uk->pl, de->pl... bez posrednictwa angielskiego).
Tokenizacja: sentencepiece; wejscie = [kod_jezyka_zrodla] + tokeny + ["</s>"], wyjscie zaczyna sie od kodu
jezyka docelowego (target_prefix), ktory odcinamy.
"""
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)

# ISO 639-1 -> kod NLLB (FLORES-200)
NLLB_CODES = {
    "pl": "pol_Latn", "en": "eng_Latn", "de": "deu_Latn", "uk": "ukr_Cyrl", "ru": "rus_Cyrl", "fr": "fra_Latn",
    "cs": "ces_Latn", "lt": "lit_Latn", "sk": "slk_Latn", "hu": "hun_Latn", "ro": "ron_Latn", "bg": "bul_Cyrl",
    "it": "ita_Latn", "es": "spa_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl", "pt": "por_Latn", "nl": "nld_Latn",
    "sv": "swe_Latn", "fi": "fin_Latn", "lv": "lvs_Latn", "et": "est_Latn", "sl": "slv_Latn", "be": "bel_Cyrl",
    "el": "ell_Grek", "tr": "tur_Latn", "da": "dan_Latn", "no": "nob_Latn", "nb": "nob_Latn", "ca": "cat_Latn",
    "bs": "bos_Latn", "mk": "mkd_Cyrl", "ka": "kat_Geor", "hy": "hye_Armn", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "ko": "kor_Hang", "ar": "arb_Arab", "he": "heb_Hebr", "fa": "pes_Arab", "hi": "hin_Deva",
}
MAX_SEGMENT_CHARS = 400          # dluzszy tekst dzielony na zdania (NLLB uczony na zdaniach)
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[\"'(«„]?[A-ZÀ-ÝА-ЯЁЇІЄҐ0-9])")


class MTUnavailable(RuntimeError):
    """Brak modelu albo bibliotek - uzyj LLM."""


class MTQualityError(RuntimeError):
    """Tlumaczenie wyglada na zepsute (puste, zapetlone, dziwnie dlugie)."""


def nllb_code(lang: str | None) -> str | None:
    return NLLB_CODES.get((lang or "").strip().lower()[:2]) if lang else None


def split_segments(text: str, limit: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Tekst -> zdania (albo kawalki do `limit` znakow), zeby model tlumaczyl krotkie fragmenty."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    out: list[str] = []
    for sent in SENT_SPLIT.split(text):
        while len(sent) > limit:                   # bardzo dlugie zdanie bez kropek: tniemy po przecinku/spacji
            cut = max(sent.rfind(", ", 0, limit), sent.rfind(" ", 0, limit))
            cut = cut if cut > limit // 3 else limit
            out.append(sent[:cut + 1].strip())
            sent = sent[cut + 1:].strip()
        if sent:
            out.append(sent)
    return out


def looks_broken(src: str, out: str) -> bool:
    """Typowe usterki NLLB: pusta odpowiedz, zapetlenie ("i i i i"), wynik kilka razy dluzszy od zrodla."""
    if not out.strip():
        return bool(src.strip())
    if len(out) > 3 * len(src) + 40:
        return True
    words = out.lower().split()
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    if len(words) >= 12 and len(set(words)) / len(words) < 0.3:
        return True
    return False


class NLLBTranslator:
    """Leniwe ladowanie modelu (przy pierwszym tlumaczeniu), bezpieczne dla watkow, pamiec podreczna zdan."""

    def __init__(self, model_dir: Path, device: str = "cpu", compute_type: str = "int8", threads: int = 4,
                 beam_size: int = 2, cache_size: int = 5000):
        self.model_dir = Path(model_dir)
        self.device, self.compute_type, self.threads, self.beam_size = device, compute_type, threads, beam_size
        self._translator = None
        self._sp = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._cache_size = cache_size
        self.error: str | None = None

    # ------------------------------------------------------------ stan
    def model_present(self) -> bool:
        return (self.model_dir / "model.bin").exists() and (self.model_dir / "sentencepiece.bpe.model").exists()

    def status(self) -> dict:
        return {"model_dir": str(self.model_dir), "present": self.model_present(), "loaded": self._translator is not None,
                "device": self.device, "compute_type": self.compute_type, "error": self.error}

    def available(self) -> bool:
        if not self.model_present():
            return False
        try:
            self._load()
            return True
        except MTUnavailable:
            return False

    def _load(self):
        if self._translator is not None:
            return
        with self._lock:
            if self._translator is not None:
                return
            if not self.model_present():
                raise MTUnavailable(f"brak modelu NLLB w {self.model_dir} (docker compose --profile mt run --rm nllb-init)")
            try:
                import ctranslate2
                import sentencepiece as spm
            except ImportError as exc:
                self.error = f"brak biblioteki: {exc}"
                raise MTUnavailable(self.error) from exc
            device = self.device
            if device == "auto":
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            compute = self.compute_type if device == "cpu" else ("int8_float16" if self.compute_type == "int8"
                                                                  else self.compute_type)
            try:
                self._translator = ctranslate2.Translator(str(self.model_dir), device=device, compute_type=compute,
                                                          intra_threads=self.threads)
                self._sp = spm.SentencePieceProcessor(model_file=str(self.model_dir / "sentencepiece.bpe.model"))
            except Exception as exc:                   # zly model, brak CUDA itd.
                self.error = f"nie udalo sie wczytac modelu: {exc}"
                raise MTUnavailable(self.error) from exc
            self.device, self.error = device, None
            log.info("NLLB wczytany z %s (%s, %s)", self.model_dir, device, compute)

    # ------------------------------------------------------------ tlumaczenie
    def translate(self, texts: list[str], src_lang: str, tgt_lang: str = "pl") -> list[str]:
        """Lista tekstow -> lista tlumaczen (ta sama kolejnosc). Dlugie teksty dzielone na zdania."""
        src, tgt = nllb_code(src_lang), nllb_code(tgt_lang)
        if not src or not tgt:
            raise MTUnavailable(f"jezyk nieobslugiwany przez NLLB: {src_lang}")
        if src == tgt:
            return list(texts)
        self._load()
        pieces: list[list[str]] = [split_segments(t) for t in texts]
        todo = sorted({s for segs in pieces for s in segs if (src, s) not in self._cache})
        if todo:
            with self._lock:                            # jeden model, jedno tlumaczenie naraz (CPU i tak pelne)
                done = self._translate_segments(todo, src, tgt)
            for s, out in zip(todo, done):
                self._cache[(src, s)] = out
                self._cache.move_to_end((src, s))
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return [" ".join(self._cache[(src, s)] for s in segs) for segs in pieces]

    def _translate_segments(self, segments: list[str], src: str, tgt: str) -> list[str]:
        batch = [[src] + self._sp.encode(s, out_type=str)[:400] + ["</s>"] for s in segments]
        results = self._translator.translate_batch(
            batch, target_prefix=[[tgt]] * len(batch), beam_size=self.beam_size, max_batch_size=32,
            max_decoding_length=320)
        out = []
        for seg, res in zip(segments, results):
            toks = [t for t in res.hypotheses[0] if t != tgt]
            text = self._sp.decode(toks).strip()
            if text.endswith(".") and not seg.rstrip().endswith((".", "!", "?")):
                text = text[:-1].rstrip()               # NLLB dopisuje kropke do skladnikow typu "salt"
            out.append(text)
        return out


_INSTANCE: NLLBTranslator | None = None


def get_translator() -> NLLBTranslator | None:
    """Wspolny tlumacz wg ustawien (None, gdy MT_ENGINE nie jest 'nllb')."""
    global _INSTANCE
    from app.config import settings
    if (settings.mt_engine or "").lower() != "nllb":
        return None
    if _INSTANCE is None:
        _INSTANCE = NLLBTranslator(Path(settings.mt_model_dir), settings.mt_device, settings.mt_compute_type,
                                   settings.mt_threads, settings.mt_beam_size)
    return _INSTANCE
