"""Zapytania do wyszukiwarek w jezyku danego kraju - generowane przez LLM, bez powtarzania uzytych."""
import json
import logging

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


async def make_queries(country: Country, n: int, used: set[str], llm=None, topic: str | None = None) -> list[str]:
    used_low = {u.lower().strip() for u in used}
    out: list[str] = []
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
