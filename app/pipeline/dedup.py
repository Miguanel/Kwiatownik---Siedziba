"""Wykrywanie tego samego przepisu na roznych stronach.

Porownujemy znormalizowany tytul i zestaw skladnikow (podobienstwo Jaccarda).
Duplikat nie jest kasowany - dostaje `duplicate_of`, dzieki czemu w eksporcie przepis ma
liste WSZYSTKICH stron, na ktorych wystepuje (pole "zrodla").
Uwaga: dziala w obrebie jednego jezyka; duplikaty miedzy jezykami wylapie etap tlumaczenia (LLM).
"""
import re
import unicodedata

STOP = {
    "a", "an", "the", "and", "or", "of", "with", "for", "to", "in", "on", "my", "your", "how", "make", "made",
    "recipe", "recipes", "easy", "best", "simple", "homemade", "quick", "diy", "ultimate", "perfect", "healthy",
    "cup", "cups", "tbsp", "tsp", "tablespoon", "tablespoons", "teaspoon", "teaspoons", "g", "kg", "ml", "l", "oz",
    "lb", "pinch", "handful", "fresh", "dried", "chopped", "large", "small", "optional", "taste", "to", "about",
    "przepis", "na", "z", "i", "w", "do", "ze", "lyzka", "lyzki", "szklanka", "szklanki", "g", "ml",
    "und", "mit", "der", "die", "das", "rezept", "de", "la", "le", "et", "el", "y", "con", "receta",
}


def _tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    words = re.findall(r"[a-z]+", t)
    return {w.rstrip("s") if len(w) > 4 else w for w in words if w not in STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def fingerprint(title: str, ingredients: list[str] | None) -> dict:
    return {"t": sorted(_tokens(title or "")), "i": sorted(_tokens(" ".join(ingredients or [])))}


def similarity(fp1: dict, fp2: dict) -> tuple[float, float]:
    return jaccard(set(fp1["t"]), set(fp2["t"])), jaccard(set(fp1["i"]), set(fp2["i"]))


def is_duplicate(fp1: dict, fp2: dict) -> bool:
    t, i = similarity(fp1, fp2)
    has_ingr = bool(fp1["i"] and fp2["i"])
    if has_ingr:
        return (t >= 0.5 and i >= 0.5) or (i >= 0.9 and t >= 0.34)
    return t >= 0.8 and len(fp1["t"]) >= 2


def find_duplicate(fp: dict, candidates: list[tuple[int, dict]]) -> int | None:
    """candidates: [(item_id, fingerprint)] - zwraca id najbardziej podobnego przepisu albo None."""
    best, best_score = None, 0.0
    for cid, cfp in candidates:
        if is_duplicate(fp, cfp):
            t, i = similarity(fp, cfp)
            if t + i > best_score:
                best, best_score = cid, t + i
    return best
