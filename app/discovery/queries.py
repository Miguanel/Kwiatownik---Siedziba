"""Zapytania do wyszukiwarek w jezyku danego kraju, bez powtarzania uzytych.

Etap 6 (bez LLM, domyslnie): szablony - nazwa rosliny z katalogu w jezyku kraju (Wikidata: etykiety, aliasy,
tytuly artykulow) + rodzaj przepisu niekulinarnego w tym jezyku (nalewka, masc, syrop, domowy sposob, barwienie),
na zmiane z ogolnymi zapytaniami kraju. LLM tylko gdy DISCOVERY_LLM_QUERIES=true.
"""
import json
import logging
import re

from app.discovery.countries import Country

log = logging.getLogger(__name__)

DEFAULT_TOPIC = ("NON-culinary plant recipes: herbal medicine and folk remedies (infusions, decoctions, tinctures, "
                 "syrups, salves, ointments, poultices, herbal baths), homemade herbal liqueurs and bitters "
                 "(nalewki), natural hair dyeing and herbal hair rinses, dyeing fabric and wool with plants, "
                 "natural cosmetics and soaps from herbs. NOT ordinary cooking or baking")

SYSTEM_PROMPT = """You create web search queries that find websites and blogs with NON-CULINARY plant RECIPES: \
herbal remedies, tinctures and liqueurs, natural dyes for hair and fabric, herbal cosmetics \
(not cooking, not shops, not news). \
Write queries the way a native speaker would type them into Google in the given language. \
Answer ONLY with JSON: {"queries": ["...", "..."]}"""


# rodzaje przepisow niekulinarnych w jezyku wyszukiwania (2 litery kodu jezyka)
PREPS = {
    "pl": ["nalewka przepis", "maść domowa", "syrop przepis", "domowy sposób", "barwienie wełny"],
    "de": ["Tinktur selber machen", "Salbe selber machen", "Sirup Rezept", "Hausmittel", "Wolle färben"],
    "en": ["tincture recipe", "salve recipe", "syrup recipe", "herbal remedy", "natural dye wool"],
    "fr": ["teinture mère recette", "baume maison", "sirop recette", "remède de grand-mère", "teinture végétale laine"],
    "es": ["tintura casera", "pomada casera", "jarabe casero", "remedio casero", "tinte natural lana"],
    "it": ["tintura madre fai da te", "unguento fatto in casa", "sciroppo ricetta", "rimedio della nonna",
           "tintura naturale lana"],
    "uk": ["настоянка рецепт", "мазь своїми руками", "сироп рецепт", "народний рецепт", "фарбування вовни"],
    "cs": ["tinktura recept", "domácí mast", "sirup recept", "babské rady", "barvení vlny"],
    "sk": ["tinktúra recept", "domáca masť", "sirup recept", "babské rady", "farbenie vlny"],
    "hu": ["tinktúra recept", "kenőcs házilag", "szörp recept", "népi gyógymód", "gyapjú festése"],
    "ro": ["tinctură rețetă", "unguent natural", "sirop rețetă", "remediu naturist", "vopsit lână natural"],
    "bg": ["тинктура рецепта", "домашен мехлем", "сироп рецепта", "народна медицина", "боядисване на вълна"],
    "hr": ["tinktura recept", "domaća mast", "sirup recept", "narodni lijek", "bojanje vune"],
    "sr": ["tinktura recept", "domaći melem", "sirup recept", "narodni lek", "bojenje vune"],
    "lt": ["tinktūra receptas", "tepalas namuose", "sirupas receptas", "liaudies medicina", "vilnos dažymas"],
}


def plant_names_for(lang: str, rows: dict[str, dict] | None = None) -> list[str]:
    """Nazwy roslin katalogu w jezyku (etykieta, tytul artykulu), bez nazw lacinskich; po polsku - nazwy katalogu."""
    if rows is None:
        try:
            from app.pipeline.checks import plant_rows
            rows = plant_rows()
        except Exception:  # noqa: BLE001 - bez bazy: same zapytania ogolne
            rows = {}
    out = []
    for r in rows.values():
        if lang == "pl":
            name = re.sub(r"\s*\(.*?\)", "", r.get("pl") or "")
        else:
            name = (r.get("names") or {}).get(lang) or (r.get("titles") or {}).get(lang) or ""
            name = re.sub(r"\s*\(.*?\)", "", name)
        name = name.strip()
        if name and name.lower() != (r.get("lat") or "").lower() and len(name) <= 40:
            out.append(name)
    return sorted(dict.fromkeys(out))


def template_queries(country: Country, n: int, used_low: set[str], rows: dict[str, dict] | None = None) -> list[str]:
    lang = (country.language or "")[:2].lower()
    preps = PREPS.get(lang) or []
    names = plant_names_for(lang, rows)
    out: list[str] = []
    # kolejnosc: rosliny po kolei, kazda z kolejnym rodzajem przepisu; co trzecie - zapytanie ogolne kraju
    combos = [f"{name} {preps[(i + r) % len(preps)]}" for r in range(len(preps)) for i, name in enumerate(names)] \
        if preps else []
    general = list(country.fallback_queries) + preps
    gi = 0
    for q in combos:
        if len(out) >= n:
            break
        if q.lower() in used_low or q in out:
            continue
        out.append(q)
        if len(out) % 3 == 2:
            while gi < len(general):
                g = general[gi]
                gi += 1
                if g.lower() not in used_low and g not in out:
                    out.append(g)
                    break
    for g in general[gi:]:
        if len(out) >= n:
            break
        if g.lower() not in used_low and g not in out:
            out.append(g)
    return out[:n]


async def make_queries(country: Country, n: int, used: set[str], llm=None, topic: str | None = None) -> list[str]:
    from app.config import settings
    used_low = {u.lower().strip() for u in used}
    out: list[str] = []
    if not (settings.discovery_llm_queries and llm is not None):
        llm = None
        out = template_queries(country, n + 3, used_low)
    if llm is not None:
        prompt = (f"Language: {country.language_name} (country: {country.name}, code {country.code}).\n"
                  f"Topic: {topic or DEFAULT_TOPIC}\n"
                  f"Give {n + 3} different, specific search queries (2-6 words each), mixing general and very "
                  f"specific ones (single plant + preparation such as tincture/salve/dye, ailment + herbal remedy, "
                  f"regional folk medicine traditions).\n"
                  + ("Do NOT repeat these already used queries:\n" + "\n".join(sorted(used)[:80]) if used else ""))
        try:
            res = await llm.complete(prompt, system=SYSTEM_PROMPT, json_mode=True)
            data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
            out = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        except Exception as exc:
            log.warning("LLM nie wygenerowal zapytan: %s", exc)
    if not out:
        out = list(country.fallback_queries)
    fresh, seen = [], set()
    for q in out:
        key = q.lower().strip()
        if key not in used_low and key not in seen:
            seen.add(key)
            fresh.append(q)
    return fresh[:n]
