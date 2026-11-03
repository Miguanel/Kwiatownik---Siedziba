"""LLM jako "zakreslacz", nie tlumacz: wskazuje w tekscie strony fragmenty przepisu DOSLOWNIE, w jezyku oryginalu.

Kod sprawdza kazda linie: musi byc w tekscie strony (te same slowa w tej samej kolejnosci, liczby ze zrodla).
Linie przeredagowane albo wymyslone sa odrzucane; gdy odrzuconych jest za duzo, przepis nie idzie dalej
(status "blad", powod "kontrola: ..."). Tlumaczy potem wylacznie tlumacz maszynowy (DeepL / NLLB).
"""
import re
import unicodedata
from dataclasses import dataclass

from app.llm.base import InvalidOutputError, json_object
from app.pipeline.checks import numbers
from app.pipeline.translate import LEAD_FORMAT, MAX_EXTRA_RECIPES, MAX_SOURCE_CHARS, NoRecipeError, clean_leads

SYSTEM_PROMPT = """Wyszukujesz przepis w tekscie strony internetowej. NIE TLUMACZ i NIE PRZEREDAGOWUJ niczego:
kazde pole to fragment SKOPIOWANY DOSLOWNIE z tekstu, w jezyku oryginalu (te same slowa, liczby i jednostki).
Wolno tylko: pominac numeracje/punktory, podzielic tekst na osobne skladniki i kroki, pominac zdania niezwiazane
z wykonaniem. Nie dopisuj skladnikow, ilosci, czasu ani wskazowek, ktorych nie ma w tekscie. Brak pola = null.
Zwroc TYLKO JSON:
{"przepisy": [{"title": "tytul przepisu z tekstu", "description": "zdanie z tekstu o przepisie lub null",
  "ingredients": ["skladnik z iloscia, doslownie"], "steps": ["krok wykonania, doslownie"],
  "yield": "porcje z tekstu lub null", "notes": "uwagi/ostrzezenia z tekstu (doslownie) lub null"}],
 "brak_przepisu": false, "wzmianki": []}
Kilka roznych przepisow w tekscie = kilka obiektow w "przepisy" (najwyzej 7, kazdy z wlasnymi skladnikami i krokami).
Brak konkretnego przepisu (lista linkow, ogolny artykul) = {"przepisy": [], "brak_przepisu": true, "wzmianki": [...]}.
""" + LEAD_FORMAT


class ExtractionCheckError(InvalidOutputError):
    """LLM przeredagowal albo dopisal tresc przepisu - kontrola kodem go zatrzymala."""


@dataclass
class Extracted:
    recipes: list[dict]          # w formacie danych strukturalnych: title, description, ingredients, steps, yield, notes
    rejected: list[str]          # odrzucone linie (do logu)


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text or "").lower()
    for k, v in {"½": " 1/2 ", "¼": " 1/4 ", "¾": " 3/4 ", "⅓": " 1/3 ", "⅔": " 2/3 "}.items():
        t = t.replace(k, v)
    return re.findall(r"\w+", t)


class SourceText:
    def __init__(self, text: str):
        self.tokens = _tokens(text)
        self.vocab = set(self.tokens)
        self.trigrams = set(zip(self.tokens, self.tokens[1:], self.tokens[2:]))
        self.bigrams = set(zip(self.tokens, self.tokens[1:]))
        self.numbers = numbers(text)

    def verbatim(self, line: str) -> float:
        """Jaka czesc slow linii stoi w tekscie zrodla w tej samej kolejnosci (trojki slow; krotkie linie - pary)."""
        toks = _tokens(line)
        if not toks:
            return 0.0
        if len(toks) == 1:
            return 1.0 if toks[0] in self.vocab else 0.0
        covered = [False] * len(toks)
        n, grams = (3, self.trigrams) if len(toks) >= 3 else (2, self.bigrams)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) in grams:
                for j in range(i, i + n):
                    covered[j] = True
        return sum(covered) / len(toks)

    def numbers_ok(self, line: str) -> bool:
        return all(any(abs(x - y) < 0.011 for y in self.numbers) for x in numbers(line))

    def ok(self, line: str, threshold: float) -> bool:
        return self.verbatim(line) >= threshold and self.numbers_ok(line)


def _lines(v) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [re.sub(r"^\s*(?:[-*•·]|\d+[.)]|step\s*\d+[:.)]?)\s*", "", str(x), flags=re.I).strip()
            for x in (v or []) if str(x or "").strip()]


def verify(raw: dict, src: SourceText, fallback_title: str | None, threshold: float,
           max_rejected: float) -> tuple[dict, list[str]]:
    """Jeden przepis z odpowiedzi LLM -> dane strukturalne z doslownymi liniami. Za duzo odrzuconych -> blad."""
    rejected: list[str] = []
    ingredients, steps = [], []
    for kind, out in (("ingredients", ingredients), ("steps", steps)):
        for line in _lines(raw.get(kind)):
            (out if src.ok(line, threshold) else rejected).append(line)
    total = len(_lines(raw.get("ingredients"))) + len(_lines(raw.get("steps")))
    title = str(raw.get("title") or "").strip()
    if not title or src.verbatim(title) < 0.6:
        title = (fallback_title or "").strip()
    if not title or not ingredients or not steps:
        if total and len(rejected) / total > max_rejected:
            raise ExtractionCheckError(_why(rejected, total))
        raise NoRecipeError("brak tytulu, skladnikow albo krokow w tekscie strony")
    if total and len(rejected) / total > max_rejected:
        raise ExtractionCheckError(_why(rejected, total))
    rec = {"title": title, "ingredients": ingredients, "steps": steps}
    for key in ("description", "notes", "yield"):
        val = str(raw.get(key) or "").strip()
        if val and src.ok(val, threshold):
            rec[key] = val
        elif val:
            rejected.append(val)
    return rec, rejected


def _why(rejected: list[str], total: int) -> str:
    return (f"kontrola: LLM przeredagowal lub dopisal {len(rejected)} z {total} linii przepisu "
            f"(np. '{rejected[0][:60]}')")


async def extract_recipes(llm, raw_text: str, meta: dict, threshold: float = 0.8,
                          max_rejected: float = 0.25) -> tuple[Extracted, str]:
    """(przepisy w jezyku oryginalu, 'provider/model'). NoRecipeError - brak przepisu; ExtractionCheckError -
    LLM nie trzymal sie tekstu."""
    text = (raw_text or "")[:MAX_SOURCE_CHARS]
    prompt = (f"JEZYK TEKSTU: {meta.get('language') or 'nieznany'}\nTYTUL STRONY: {meta.get('title')}\n\n"
              f"TEKST STRONY:\n{text}")
    res = await llm.complete(prompt, system=SYSTEM_PROMPT, json_mode=True)
    data = json_object(res.text)
    model = f"{res.provider}/{res.model}"
    items = data.get("przepisy")
    if isinstance(data.get("_lista"), list):
        items = data["_lista"]
    items = [x for x in (items or []) if isinstance(x, dict)][:MAX_EXTRA_RECIPES + 1]
    if data.get("brak_przepisu") is True or not items:
        raise NoRecipeError("w tekscie nie ma konkretnego przepisu (lista lub artykul ogolny)",
                            leads=clean_leads(data.get("wzmianki")))
    src = SourceText(text)
    recipes, rejected = [], []
    first_error: Exception | None = None
    for i, raw in enumerate(items):
        try:
            rec, rej = verify(raw, src, meta.get("title") if i == 0 else None, threshold, max_rejected)
        except (ExtractionCheckError, NoRecipeError) as exc:
            if i == 0:
                first_error = exc
            continue
        recipes.append(rec)
        rejected += rej
    if first_error is not None and not recipes:
        if isinstance(first_error, NoRecipeError):
            first_error.leads = clean_leads(data.get("wzmianki"))
        raise first_error
    if first_error is not None and isinstance(first_error, ExtractionCheckError):
        raise first_error                       # glowny przepis przeredagowany - caly artykul do sprawdzenia
    return Extracted(recipes, rejected), model
