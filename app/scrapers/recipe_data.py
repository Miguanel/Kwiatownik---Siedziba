"""Wspolny, znormalizowany format przepisu wyciagnietego ze strony (przed tlumaczeniem na format Kwiatownika).

{"title", "description", "ingredients": [...], "steps": [...], "image", "yield", "total_time", "notes",
 "category", "keywords"}
"""
import html as html_lib
import re


def _clean(s) -> str:
    s = html_lib.unescape(str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _texts(v) -> list[str]:
    """recipeInstructions moze byc: tekstem, lista tekstow, HowToStep, HowToSection..."""
    out: list[str] = []
    if isinstance(v, str):
        parts = re.split(r"\n+|(?<=\.)\s+(?=\d+[.)]\s)", html_lib.unescape(v))
        out += [_clean(p) for p in parts if _clean(p)]
    elif isinstance(v, list):
        for x in v:
            out += _texts(x)
    elif isinstance(v, dict):
        if "itemListElement" in v:
            out += _texts(v["itemListElement"])
        elif v.get("text") or v.get("name"):
            out.append(_clean(v.get("text") or v.get("name")))
    return [t for t in out if t]


def _first(v):
    if isinstance(v, list):
        return _first(v[0]) if v else None
    if isinstance(v, dict):
        return v.get("url") or v.get("@id")
    return v


def normalize_jsonld_recipe(node: dict) -> dict:
    kw = node.get("keywords")
    return {
        "title": _clean(node.get("name")),
        "description": _clean(node.get("description")) or None,
        "ingredients": [_clean(i) for i in (node.get("recipeIngredient") or node.get("ingredients") or []) if _clean(i)],
        "steps": _texts(node.get("recipeInstructions")),
        "image": _first(node.get("image")),
        "yield": _clean(_first(node.get("recipeYield"))) or None,
        "total_time": node.get("totalTime") or None,
        "notes": None,
        "category": _clean(_first(node.get("recipeCategory"))) or None,
        "keywords": [k.strip() for k in kw.split(",")] if isinstance(kw, str) else (kw or []),
    }


def is_complete(recipe: dict | None) -> bool:
    return bool(recipe and recipe.get("title") and recipe.get("ingredients") and recipe.get("steps"))
