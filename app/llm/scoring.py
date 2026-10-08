"""Heurystyczna ocena "jak dobry jest model" na podstawie jego nazwy.

Zwraca None dla modeli, ktore nie nadaja sie do generowania tekstu (embedding, TTS, obraz...).
Wynik jest porownywalny tylko w obrebie jednego providera - router normalizuje go do 0-100.
"""
import math
import re

GEMINI_EXCLUDE = ("embedding", "tts", "image", "live", "audio", "robotics", "computer-use",
                  "veo", "imagen", "aqa", "learnlm", "nano-banana")
GROQ_EXCLUDE = ("whisper", "tts", "guard", "embed", "playai", "orpheus", "distil", "compound",
                "prompt-guard", "safeguard", "allam")
GROQ_FAMILY_BONUS = {           # sila rodziny modeli (wieksze = lepsze)
    "gpt-oss": 6, "kimi-k2": 6, "llama-4-maverick": 5, "qwen3": 4, "deepseek": 3,
    "llama-4-scout": 3, "llama-3.3": 2, "mistral": 1, "llama-3.1": 0, "gemma": -1,
}


def score_gemini(model: str) -> float | None:
    m = model.lower()
    if not m.startswith("gemini") or any(x in m for x in GEMINI_EXCLUDE):
        return None
    ver = re.search(r"gemini-(\d+(?:\.\d+)?)", m)
    if not ver:  # aliasy typu "gemini-flash-latest" - pomijamy, bo dubluja wersje z numerem
        return None
    score = float(ver.group(1)) * 10
    if "flash-lite" in m:
        score += 2
    elif "flash" in m:
        score += 5
    elif "pro" in m:
        score += 7
    if "preview" in m or "exp" in m:
        score -= 4                  # mniej stabilne
    if re.search(r"-\d{3}$", m):
        score -= 0.5                # przypiete snapshoty ponizej "glownej" nazwy
    return score


def score_groq(model: str, context_window: int | None = None) -> float | None:
    m = model.lower()
    if any(x in m for x in GROQ_EXCLUDE):
        return None
    params = re.search(r"(\d+(?:\.\d+)?)b\b", m)
    size = float(params.group(1)) if params else 30.0
    bonus = next((b for fam, b in GROQ_FAMILY_BONUS.items() if fam in m), 0)
    score = 20 + bonus + 4 * math.log2(max(size, 1))
    if context_window:
        score += min(context_window / 131072, 1) * 2
    if "preview" in m:
        score -= 2
    return score


def score_ollama(model: str) -> float | None:
    """Lokalne modele: wieksze = lepsze (ale i tak zawsze na koncu listy jako zapas)."""
    m = model.lower()
    if any(x in m for x in ("embed", "vision-only", "whisper")):
        return None
    params = re.search(r"(\d+(?:\.\d+)?)b\b", m)
    size = float(params.group(1)) if params else 7.0
    bonus = next((b for fam, b in {"qwen3": 4, "gemma4": 4, "gemma3": 3, "llama3": 2, "mistral": 1}.items()
                  if fam in m), 0)
    if "bielik" in m:      # Bielik: swietny polski, ale sredni w JSON-owych zadaniach - do scalania (MERGE_MODELS)
        bonus -= 3
    return 20 + bonus + 4 * math.log2(max(size, 1))


SCORERS = {
    "gemini": lambda info: score_gemini(info.model),
    "groq": lambda info: score_groq(info.model, info.context_window),
    "ollama": lambda info: score_ollama(info.model),
}
