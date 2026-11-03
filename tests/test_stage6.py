"""Etap 6: tokeny LLM wg celu i dzienny budzet chmury; nowe rosliny ze skladnikow przez Wikipedie + Wikidata."""
import asyncio
from datetime import date

import httpx
from sqlmodel import Session, SQLModel, create_engine, select

from app.knowledge.registry import discover_new_plants, name_phrase
from app.knowledge.wiki import WikiClient
from app.llm.usage import LLMUsage, parse_budget, purpose_of, tokens
from app.models import Plant, PlantAlias


def test_purpose_budget_and_table(tmp_path):
    from app.knowledge.extract import SYSTEM_PROMPT as KNOW
    from app.scrapers.classify import SYSTEM_PROMPT as CLS
    assert purpose_of(KNOW) == "wiedza" and purpose_of(CLS) == "klasyfikacja" and purpose_of("cos") == "inne"
    assert parse_budget("klasyfikacja=150000, zapytania=30_000, zle") == {"klasyfikacja": 150000, "zapytania": 30000}
    u = LLMUsage(tmp_path / "t.json", {"klasyfikacja": 100})
    d = date(2026, 11, 3)
    u.add("klasyfikacja", 60, local=False, today=d)
    assert not u.over_budget("klasyfikacja", d)
    u.add("klasyfikacja", 50, local=False, today=d)
    u.add("klasyfikacja", 500, local=True, today=d)                     # lokalnie nie liczy sie do budzetu
    assert u.over_budget("klasyfikacja", d) and not u.over_budget("wiedza", d)
    row = LLMUsage(tmp_path / "t.json", {"klasyfikacja": 100}).table(d)[0]
    assert row["cel"] == "klasyfikacja" and row["dzis_chmura"] == 110 and row["dzis_lokalnie"] == 500 and row["budzet"] == 100
    assert tokens("abcd" * 10) == 10


def test_router_sends_over_budget_purpose_to_local_model(tmp_path):
    from app.llm.base import LLMResult
    from app.llm.router import ModelEntry, ModelRouter
    usage = LLMUsage(None, {"klasyfikacja": 10})
    usage.add("klasyfikacja", 50, local=False)
    r = ModelRouter([], usage=usage)
    calls = []

    async def fake_complete(prompt, system, json_mode, avoid):
        calls.append(set(avoid))
        return LLMResult("{}", "ollama", "qwen", 0.1)
    r._complete = fake_complete
    r.order = [ModelEntry("groq", "llama", 90), ModelEntry("ollama", "qwen", 10, fallback=True)]
    from app.scrapers.classify import SYSTEM_PROMPT as CLS
    asyncio.run(r.complete("x", system=CLS))
    assert calls[-1] == {"groq:llama"}                                   # chmura pominieta
    asyncio.run(r.complete("x", system="inny cel"))
    assert calls[-1] == set()
    assert usage.table()[0]["dzis_lokalnie"] >= 1


WIKI = {
    "kwiatów czarnego bzu": ["Bez czarny"], "czarnego bzu": ["Bez czarny"], "olej lniany": ["Olej lniany"],
    "lniany": ["Olej lniany"], "miód": ["Miód"], "Miód": ["Miód"], "miodu": ["Miód"],
}
PAGES = {"Bez czarny": "Q22701", "Olej lniany": "Q1144417", "Miód": "Q10987"}
ENTS = {"Q22701": {"claims": {"P225": [{"mainsnak": {"datavalue": {"value": "Sambucus nigra"}}}]},
                   "labels": {"pl": {"value": "bez czarny"}}},
        "Q1144417": {"claims": {"P1582": [{"mainsnak": {"datavalue": {"value": {"id": "Q911332"}}}}]}},
        "Q911332": {"claims": {"P225": [{"mainsnak": {"datavalue": {"value": "Linum usitatissimum"}}}]},
                    "labels": {"pl": {"value": "len zwyczajny"}}},
        "Q10987": {"claims": {}, "labels": {"pl": {"value": "miód"}}}}


def handler(req: httpx.Request):
    p = dict(req.url.params)
    if p.get("list") == "search":
        return httpx.Response(200, json={"query": {"search": [{"title": t} for t in WIKI.get(p["srsearch"], [])]}})
    if p.get("prop") == "pageprops":
        pages = {str(i): {"title": t, "pageprops": {"wikibase_item": PAGES[t]}}
                 for i, t in enumerate(p["titles"].split("|")) if t in PAGES}
        return httpx.Response(200, json={"query": {"pages": pages}})
    if p.get("action") == "wbgetentities":
        return httpx.Response(200, json={"entities": {p["ids"]: ENTS.get(p["ids"], {})}})
    return httpx.Response(404)


def test_new_plants_from_ingredients_by_wikipedia_and_wikidata(tmp_path, monkeypatch):
    import json

    from app.models import Item, ItemKind, ItemStatus
    assert name_phrase("2 łyżki suszonych kwiatów czarnego bzu") == "czarnego bzu"
    eng = create_engine(f"sqlite:///{(tmp_path / 'n.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    rec = {"tytul": "Syrop", "skladniki": [{"nazwa": "Kwiaty czarnego bzu"}, {"nazwa": "Olej lniany"},
                                           {"nazwa": "Miód"}]}
    with Session(eng) as s:
        s.add(Item(url="u", kind=ItemKind.recipe, status=ItemStatus.translated, data_json=json.dumps(rec)))
        s.commit()

    class NoLLM:
        async def complete(self, *a, **k):
            raise AssertionError("bez LLM")

    async def run():
        wiki = WikiClient("T", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), delay_s=0)
        with Session(eng) as s:
            return await discover_new_plants(s, NoLLM(), wiki=wiki)
    created = asyncio.run(run())
    assert sorted(created) == ["bez_czarny", "len_zwyczajny"]
    with Session(eng) as s:
        assert s.get(Plant, "len_zwyczajny").nazwa_lat == "Linum usitatissimum"
        aliases = {a.name: a.plant_id for a in s.exec(select(PlantAlias)).all()}
    assert aliases["miod"] is None and aliases["kwiaty czarnego bzu"] == "bez_czarny"
    assert asyncio.run(run()) == []                                      # bez ponownego pytania


def test_profile_selectors_from_rules_without_llm():
    from app.scrapers.profile import ScraperProfile, rule_field_sets, validate
    body = "<p>" + "Napar z ziół na kaszel, tradycyjny przepis babci. " * 10 + "</p>"
    page = lambda n: (f"<html><body><h1>Syrop {n}</h1><div class='entry-content'>{body}"  # noqa: E731
                      f"<div class='wprm-recipe'><h2 class='wprm-recipe-name'>Syrop {n}</h2>"
                      f"<ul><li class='wprm-recipe-ingredient'>2 łyżki kwiatów</li>"
                      f"<li class='wprm-recipe-ingredient'>woda</li></ul>"
                      f"<div class='wprm-recipe-instruction-text'>Zalać.</div>"
                      f"<div class='wprm-recipe-instruction-text'>Odcedzić.</div></div></div></body></html>")
    samples = [(f"https://x.pl/przepis/syrop-{n}", page(n), []) for n in range(3)]
    sets = dict(rule_field_sets(samples))
    assert "wprm" in sets and sets["wprm"]["content"]["css"] == "article .entry-content" or \
        sets["wprm"]["content"]["css"] == ".entry-content"
    prof = ScraperProfile(recipe_url_regex=r"^/przepis/", fields=sets["wprm"])
    rep = validate(prof, samples, [u for u, _, _ in samples], ["https://x.pl/kontakt"])
    assert rep["passed"] and prof.extract(samples[0][1], [])["ingredients"] == ["2 łyżki kwiatów", "woda"]
