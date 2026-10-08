"""Rozpoznawanie, czy strona to przepis, ciekawostka czy nic ciekawego.

Warstwy (od najtanszej):
  1. JSON-LD schema.org/Recipe     -> przepis, pewnosc 1.0, bez LLM
  2. heurystyki (naglowki "Ingredients", ilosci "2 tbsp", slowa tematyczne)
  3. LLM (router Gemini/Groq) - tylko dla stron niejednoznacznych
Kwiatownik zbiera przepisy NIEKULINARNE (ziololecznictwo, nalewki, barwienie wlosow i tkanin, kosmetyki):
czysto kulinarne przepisy (ciasta, dania, zupy) sa odrzucane jako "other" (SKIP_CULINARY=true).
"""
import json
import logging
import re
from dataclasses import dataclass

from app.scrapers.page import PageData

log = logging.getLogger(__name__)

DEFAULT_KEYWORDS = [
    "recipe", "remedy", "remedies", "herb", "herbal", "plant", "flower", "tea", "tincture", "salve", "balm",
    "syrup", "infusion", "decoction", "poultice", "oil", "vinegar", "cordial", "wine", "forag", "wild",
    "medicinal", "folk", "elder", "nettle", "dandelion", "rosehip", "chamomile", "yarrow", "plantain",
    "rezept", "kraut", "kräuter", "tee", "receta", "hierba", "recette", "plante", "tisane",
    # inne jezyki (dopasowanie po poczatku slowa)
    "zioł", "ziół", "ziel", "nalew", "трав", "рослин", "настоянк", "чай", "рецепт", "bylin", "recept",
    "gyógynövény", "tea", "plant", "erbe", "ricett", "rețet", "plante", "žolel", "билк", "bilj",
    # barwienie, kosmetyka, nalewki
    "dye", "dyeing", "mordant", "henna", "färb", "pflanzenfarb", "teinture", "tinte", "tintur", "barw", "farb",
    "фарб", "барвн", "liqueur", "likör", "liquore", "licor", "likér", "soap", "seife", "savon", "jabón",
    "cream", "lotion", "shampoo", "hydrolat", "oxymel", "elixir", "bitters", "ointment", "salbe", "pommade",
    "maść", "мазь", "компрес", "відвар", "настій", "лікув",
]

# Cel przepisu: lecznictwo / rzemioslo (to zbieramy) vs kuchnia (pomijamy). Dopasowanie po poczatku slowa.
PURPOSE_WORDS = (
    "remed", "medicin", "healing", "=heal", "=cure", "cough", "=cold", "=flu", "=sore", "wound", "burn", "=skin",
    "eczema", "acne", "digest", "sleep", "anxiety", "immun", "inflam", "=pain", "headache", "tincture", "salve",
    "balm", "ointment", "poultice", "compress", "liniment", "decoction", "infusion", "macerat", "oxymel",
    "elixir", "bitters", "liqueur", "cordial", "=dye", "dyeing", "mordant", "henna", "=soap", "cosmetic",
    "lotion", "=cream", "shampoo", "rinse", "hydrolat", "=tonic", "detox", "gargle", "inhal", "=bath",
    # de
    "heil", "hausmittel", "tinktur", "salbe", "umschlag", "aufguss", "likör", "färb", "pflanzenfarb", "seife",
    "husten", "erkält", "wunde", "=haut", "kosmetik",
    # fr / es / it / pt
    "remède", "remede", "soign", "teintur", "teindre", "pommade", "cataplasm", "liqueur", "savon", "toux",
    "remedio", "curar", "tintura", "pomada", "ungüento", "licor", "teñir", "jabón", "=tos", "rimedi", "unguent",
    "impacco", "liquore", "tinger", "sapone", "tosse",
    # pl / cs / sk
    "lecz", "lecznicz", "zioł", "ziół", "nalew", "maść", "=masc", "okład", "oklad", "odwar", "napar", "syrop",
    "kaszel", "przezięb", "barwi", "farbow", "mydł", "kosmety", "płukank", "léčiv", "léčb", "kašel", "=mast",
    "obklad", "likér", "barven", "mýdl",
    # uk / ru / bg
    "лік", "лечеб", "лечен", "народн", "настоянк", "настойк", "настій", "настой", "відвар", "отвар", "мазь",
    "компрес", "кашл", "застуд", "простуд", "фарбув", "краси", "=мило", "=мыло", "билк",
)
CULINARY_WORDS = (
    "dinner", "=lunch", "breakfast", "dessert", "=cake", "cookie", "biscuit", "muffin", "=pie", "=tart", "=bake",
    "baking", "=oven", "fried", "=fry", "=roast", "=grill", "soup", "=stew", "salad", "=pasta", "pizza", "=meat",
    "chicken", "=pork", "=beef", "=lamb", "=fish", "salmon", "shrimp", "sandwich", "burger", "casserole", "risotto",
    "=dough", "pancake", "=sauce", "appetizer", "side dish", "main course",
    "kuchen", "torte", "braten", "suppe", "auflauf", "hähnchen", "fleisch", "nudel", "gebäck", "plätzchen",
    "gâteau", "gateau", "poulet", "soupe", "viande", "tarte", "pâtes", "pastel", "=pollo", "=sopa", "=carne",
    "=torta", "zuppa", "=pasta", "=dolce", "biscott",
    "ciast", "=zupa", "=zupy", "=obiad", "kolac", "śniad", "=deser", "mięs", "kurczak", "pierog", "pieróg",
    "sałat", "makaron", "pieczon", "smażon", "polędwic",
    "торт", "пиріг", "пирог", "печив", "печень", "=суп", "борщ", "м'яс", "мяс", "курк", "курин", "салат",
    "вареник", "запік", "запек", "смаж", "обід", "десерт",
    "koláč", "polévk", "=maso", "kuře", "=dort",
)
CULINARY_CATEGORIES = ("dessert", "=main", "=side", "appetizer", "breakfast", "=lunch", "dinner", "soup", "salad",
                       "=snack", "baking", "=bread", "=cake", "vorspeise", "hauptgericht", "nachtisch", "=plat",
                       "entrée", "postre", "primi", "secondi", "=obiad", "=deser", "=zupa", "десерт", "=суп")
INGREDIENT_HEADINGS = ("ingredient", "you will need", "you'll need", "what you need", "zutaten", "ingrédients",
                       "ingredientes", "ingredienti", "składniki", "materials", "інгредієнти", "складники",
                       "ингредиенты", "suroviny", "ingredience", "ingrediencie", "hozzávalók", "ingrediente",
                       "ingredientai", "съставки", "продукти", "sastojci", "sastojke")
STEP_HEADINGS = ("instruction", "direction", "method", "preparation", "how to make", "steps", "zubereitung",
                 "préparation", "preparación", "procedimento", "przygotowanie", "sposób", "приготування",
                 "приготовление", "postup", "príprava", "elkészítés", "preparare", "gaminimas", "приготвяне",
                 "priprema")
QTY_RE = re.compile(
    r"\b(\d+([.,/]\d+)?|½|¼|¾|one|two|three|a handful|a pinch)\s*"
    r"(g|kg|mg|ml|l|oz|lb|lbs|cups?|tbsps?|tsps?|tablespoons?|teaspoons?|pinch|handful|drops?|parts?|el|tl|"
    r"esslöffel|teelöffel|г|кг|мл|л|ст\.?\s?л|ч\.?\s?л|склянк\w*|ложк\w*|lžíc\w*|łyż\w*|szklan\w*)\b",
    re.I)


@dataclass
class Classification:
    kind: str            # recipe | fact | other
    confidence: float
    method: str          # jsonld | heuristic | llm
    title: str
    reason: str = ""


def jsonld_recipe(page: PageData) -> dict | None:
    for node in page.jsonld:
        t = node.get("@type")
        types = t if isinstance(t, list) else [t]
        if any(isinstance(x, str) and x.lower() == "recipe" for x in types):
            return node
    return None


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'’]+", text.lower())


def _hits(words: list[str], stems: tuple[str, ...]) -> int:
    """Liczy slowa pasujace do rdzeni; rdzen z "=" dopasowuje tylko cale slowo (lub liczbe mnoga -s/-es)."""
    exact = {x[1:] for x in stems if x.startswith("=")}
    prefix = tuple(x for x in stems if not x.startswith("="))
    return sum(1 for w in words if w in exact or w.rstrip("s").removesuffix("e") in exact
               or w.rstrip("s") in exact or w.startswith(prefix))


def purpose_scores(page: PageData, node: dict | None = None) -> tuple[int, int]:
    """(trafienia lecznicze/rzemieslnicze, trafienia kulinarne) w tytule, naglowkach i poczatku tekstu."""
    extra = ""
    if node:
        extra = " ".join(str(node.get(k) or "") for k in ("name", "recipeCategory", "recipeCuisine", "keywords"))
    head = f"{page.title} {' '.join(page.headings[:20])} {extra}"
    words = _words(head) + _words(page.text[:3000])
    purpose = _hits(words, PURPOSE_WORDS)
    culinary = _hits(words, CULINARY_WORDS)
    cats = node.get("recipeCategory") if node else None
    cats = cats if isinstance(cats, list) else [cats] if cats else []
    if any(_hits(_words(str(c)), CULINARY_CATEGORIES) for c in cats):
        culinary += 3
    return purpose, culinary


def is_culinary(page: PageData, node: dict | None = None) -> bool:
    """Wyraznie kulinarny przepis (danie, ciasto, zupa) bez celu leczniczego / rzemieslniczego."""
    purpose, culinary = purpose_scores(page, node)
    return culinary >= 3 and culinary >= 3 * max(purpose, 1) or (purpose == 0 and culinary >= 2)


def is_product_page(page: PageData) -> bool:
    """Karta produktu w sklepie (JSON-LD Product/Offer) - to nie przepis, nawet jesli opis wyglada podobnie."""
    for node in page.jsonld:
        t = node.get("@type")
        if any(x in ("Product", "Offer", "AggregateOffer") for x in (t if isinstance(t, list) else [t])):
            return True
    return False


def quantity_hits(text: str) -> int:
    return len(QTY_RE.findall(text or ""))


def heuristic_scores(page: PageData, keywords: list[str]) -> tuple[float, float]:
    """(recipe_score, topic_score) w zakresie 0..1."""
    heads = " | ".join(page.headings).lower()
    has_ingr = any(h in heads for h in INGREDIENT_HEADINGS)
    has_steps = any(h in heads for h in STEP_HEADINGS)
    qty_items = sum(1 for li in page.list_items if QTY_RE.search(li))
    qty_text = len(QTY_RE.findall(page.text[:8000]))
    recipe = 0.3 * has_ingr + 0.25 * has_steps + min(0.3, qty_items * 0.06) + min(0.15, qty_text * 0.02)

    words = re.findall(r"\w+", f"{page.title} {page.text[:6000]}".lower())
    kw_hits = sum(1 for w in words if any(w.startswith(k) for k in keywords))
    topic = min(1.0, kw_hits / 15)
    return round(min(recipe, 1.0), 3), round(topic, 3)


SYSTEM_PROMPT = """You classify web pages for "Kwiatownik", a Polish website about herbalism: herbal remedies, \
tinctures and liqueurs, natural hair and fabric dyeing, natural cosmetics and household preparations from plants. \
It does NOT collect ordinary cooking recipes. Answer ONLY with JSON:
{"type": "recipe" | "fact" | "other", "confidence": 0.0-1.0, "title": "short English title", "reason": "max 15 words"}
- recipe: the page contains a concrete NON-CULINARY preparation with ingredients and steps: herbal remedy \
(tea/infusion, decoction, tincture, syrup, salve, ointment, oil, poultice, bath, inhalation), herbal liqueur or \
bitters, natural hair dye or rinse, fabric/yarn dyeing with plants, soap, cream, lotion or other natural cosmetic.
Plants (herbs, flowers, roots, bark, fungi) must be the KEY active ingredient of the recipe.
- other (reason starting with "culinary"): ordinary cooking/baking recipes (meals, cakes, soups, salads) even if \
they contain herbs or flowers.
- fact: an article with interesting, specific information about plants, herbs, flowers, trees or fungi \
(botany, history, folklore, traditional uses, identification).
- other: listings/category pages, shops, about/contact pages, unrelated topics, pages without real content."""


CULINARY_REASON = "przepis kulinarny - pominiety (Kwiatownik zbiera przepisy niekulinarne)"


async def classify(page: PageData, keywords: list[str], llm=None, skip_culinary: bool | None = None) -> Classification:
    if skip_culinary is None:
        from app.config import settings
        skip_culinary = settings.skip_culinary
    title = page.title or page.url
    node = jsonld_recipe(page)
    if is_product_page(page) and not node:
        return Classification("other", 0.9, "heuristic", title, "karta produktu w sklepie")
    if node:
        if skip_culinary and is_culinary(page, node):
            return Classification("other", 0.9, "jsonld", node.get("name") or title, CULINARY_REASON)
        return Classification("recipe", 1.0, "jsonld", node.get("name") or title, "schema.org/Recipe")

    recipe, topic = heuristic_scores(page, keywords)
    if recipe >= 0.45 and skip_culinary and is_culinary(page):
        return Classification("other", 0.8, "heuristic", title, CULINARY_REASON)
    if recipe >= 0.7:
        return Classification("recipe", round(recipe, 2), "heuristic", title, "skladniki + kroki + ilosci")
    if len(page.text) < 300 or (topic == 0 and recipe < 0.3):
        return Classification("other", 0.8, "heuristic", title, "brak tresci lub tematu")

    if llm is not None:
        prompt = (f"URL: {page.url}\nTITLE: {title}\nHEADINGS: {' | '.join(page.headings[:15])}\n"
                  f"TEXT (beginning):\n{page.text[:2500]}")
        try:
            res = await llm.complete(prompt, system=SYSTEM_PROMPT, json_mode=True)
            data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
            kind = data.get("type") if data.get("type") in ("recipe", "fact", "other") else "other"
            if kind == "other" and str(data.get("reason", "")).lower().startswith("culinary"):
                data["reason"] = CULINARY_REASON
            if kind == "recipe" and recipe == 0 and quantity_hits(page.text) == 0:
                # LLM widzi "przepis", ale na stronie nie ma zadnych ilosci ani sekcji skladnikow:
                # to lista/kategoria przepisow albo artykul ogolny - nie konkretny przepis
                if len(page.links) >= 60 or len(page.list_items) >= 25:
                    return Classification("other", 0.7, f"llm:{res.model}", data.get("title") or title,
                                          "lista/kategoria przepisow (brak skladnikow i ilosci)")
                kind, data["reason"] = "fact", "artykul bez konkretnych ilosci - zapisany jako ciekawostka"
            conf = float(data.get("confidence") or 0.5)
            return Classification(kind, round(min(max(conf, 0), 1), 2), f"llm:{res.model}",
                                  data.get("title") or title, str(data.get("reason", ""))[:200])
        except Exception as exc:  # LLM niedostepny -> spadamy do heurystyk
            log.warning("Klasyfikacja LLM nieudana dla %s: %s", page.url, exc)

    if recipe >= 0.45:
        return Classification("recipe", recipe, "heuristic", title, "prawdopodobny przepis")
    if topic >= 0.5 and len(page.text) > 1200:
        return Classification("fact", round(topic * 0.6, 2), "heuristic", title, "artykul tematyczny")
    return Classification("other", 0.5, "heuristic", title, "niejednoznaczne")
