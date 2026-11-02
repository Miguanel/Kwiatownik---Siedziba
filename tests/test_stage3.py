"""Etap 3 diagnostyki 2026-11-02: kolejka wedlug modeli, kolejnosc roslin, szybka ocena zrodla, weryfikacja."""
import asyncio
import json

import httpx
from sqlmodel import Session, SQLModel, create_engine

from app.scrapers.classify import classify, shop_product_signals
from app.scrapers.page import parse_page
from app.worker.runner import JobRunner


# ---------------------------------------------------------------- kolejka wedlug dostepnych modeli
def test_only_local_model_lets_jobs_without_llm_overtake():
    order: list[str] = []

    def work(name, t=0.03):
        async def run():
            order.append(f"start {name}")
            await asyncio.sleep(t)
        return run

    async def main(cap):
        r = JobRunner(3)
        r.llm_capacity = lambda: cap
        r.start(1, work("badanie1"), kind="plant_research")
        r.start(2, work("badanie2"), kind="plant_research")
        r.start(3, work("tlumaczenie"), kind="translate")
        r.start(4, work("zdjecia"), kind="plant_photos")
        await asyncio.sleep(0.01)
        waiting = r.waiting_for_llm()
        await asyncio.gather(*r.tasks.values())
        return waiting

    waiting = asyncio.run(main(1))            # chmura na limitach, Ollama: 1 miejsce
    assert order[:2] == ["start badanie1", "start zdjecia"] and waiting == [2, 3]
    order.clear()
    assert asyncio.run(main(None)) == []      # chmura dziala - zwykla kolejka FIFO
    assert order[:3] == ["start badanie1", "start badanie2", "start tlumaczenie"]


def test_router_capacity_follows_cloud_cooldowns():
    from app.llm.base import LLMProvider, ModelInfo
    from app.llm.router import ModelRouter

    class P(LLMProvider):
        def __init__(self, name, fallback=False, concurrency=1):
            super().__init__("", concurrency=concurrency)
            self.name, self.fallback = name, fallback

        async def list_models(self):
            return [ModelInfo(self.name, "m")]

        async def _generate(self, model, prompt, system, json_mode):
            return "ok"

    r = ModelRouter([P("groq"), P("ollama", fallback=True, concurrency=1)])
    asyncio.run(r.refresh())
    assert r.llm_capacity() is None                         # chmura dostepna
    import time
    for e in r.order:
        if not e.fallback:
            e.cooldown_until = time.time() + 3 * 3600       # limit dzienny
    assert r.llm_capacity() == 1
    for e in r.order:
        if not e.fallback:
            e.cooldown_until = time.time() + 30             # chwilowy limit na minute - chmura "zaraz wroci"
    assert r.llm_capacity() is None


# ---------------------------------------------------------------- sklepy
def _page(html, url):
    return parse_page(html, url)


def test_shop_product_without_jsonld_is_other_but_recipe_on_shop_page_stays():
    prod = ("<html><head><title>Syrop Owocowa Spiżarnia malina z lipą 550g</title></head><body><h1>Syrop</h1><p>"
            + "Syrop z maliną i lipą, idealny na jesień. " * 15 + "</p><p>Cena: 24,99 zł</p>"
            "<button>Dodaj do koszyka</button></body></html>")
    page = _page(prod, "https://aptekacud.pl/nalewki-wyciagi/2936-syrop-malina-z-lipa")
    assert shop_product_signals(page)
    c = asyncio.run(classify(page, ["syrop", "lip"], None))
    assert c.kind == "other" and "produktu" in c.reason
    ean = _page("<html><body><h1>Melisal</h1><p>" + "Melisa na sen. " * 40 + "</p></body></html>",
                "https://aptekacud.pl/uklad-nerwowy/3523-melisal-forte-syrop-125g-herbapol-5909990067916")
    assert shop_product_signals(ean)
    recipe = ("<html><head><title>Syrop z kwiatów lipy</title></head><body><h1>Syrop z lipy</h1>"
              "<h2>Składniki</h2><ul><li>200 g kwiatów lipy</li><li>1 l wody</li><li>500 g cukru</li></ul>"
              "<h2>Przygotowanie</h2><ol><li>Zalać kwiaty wodą.</li><li>Gotować 10 minut.</li></ol>"
              "<p>Syrop z lipy na kaszel i przeziębienie. " * 5 + "</p><p>Polecany produkt: 19,99 zł</p>"
              "<a>Dodaj do koszyka</a></body></html>")
    c = asyncio.run(classify(_page(recipe, "https://sklep.example/blog/syrop-z-lipy"), ["syrop", "lip"], None))
    assert c.kind == "recipe"


def test_probe_stops_early_on_shop():
    from app.scrapers.crawler import SHOP_SAMPLE, CrawlConfig, SmartCrawler
    from app.scrapers.fetcher import Fetcher
    from tests.test_scraper import MemoryStore

    def handler(req: httpx.Request):
        path = req.url.path
        if path == "/robots.txt" or path.endswith(".xml"):
            return httpx.Response(404)
        if path == "/":
            cats = ["herbaty", "syropy", "masci", "olejki", "kremy", "mydla", "nalewki", "kapsulki", "susz", "miody",
                    "balsamy", "toniki", "kadzidla", "przyprawy", "soki", "zele", "plyny", "proszki", "maseczki",
                    "szampony", "pasty", "krople", "spraye", "kompresy", "zestawy"]
            links = "".join(f'<a href="/{c}/produkt-5900956100{i:03d}">p{i}</a>' for i, c in enumerate(cats))
            return httpx.Response(200, text=f"<html><body><h1>Sklep z ziołami</h1>{links}</body></html>",
                                  headers={"content-type": "text/html"})
        body = "<html><body><h1>Herbata ziołowa</h1><p>" + "Mieszanka ziół na trawienie. " * 30 + "</p></body></html>"
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    async def run():
        f = Fetcher("TestBot", delay_s=0, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            return await SmartCrawler("https://shop.example/", f, MemoryStore(),
                                      CrawlConfig(max_pages=80, stop_when={"recipe": 5, "fact": 5})).run()
        finally:
            await f.aclose()

    stats = asyncio.run(run())
    assert stats.get("shop", 0) > SHOP_SAMPLE // 2 and stats["fetched"] < 40


# ---------------------------------------------------------------- kolejnosc roslin i weryfikacja przed badaniem
def test_valuable_plants_first_and_odd_files_skipped(tmp_path, monkeypatch):
    import app.worker.knowledge as kn
    from app.config import settings
    from app.models import Plant, SiteStat
    plants, przepisy = tmp_path / "plants", tmp_path / "przepisy"
    plants.mkdir()
    przepisy.mkdir()
    for pid in ("arnika", "babka", "czystek"):
        (plants / f"{pid}.json").write_text(json.dumps({"id": pid, "nazwa_pl": pid}), encoding="utf-8")
    (plants / "swierzabek.json").write_text(json.dumps({"gatunek": "Świerząbek", "Cechy": {}}), encoding="utf-8")
    (przepisy / "przepisy_medyczne.json").write_text(json.dumps([
        {"tytul": "Napar", "skladniki": [{"nazwa": "liście", "link_id": "czystek"}]},
        {"tytul": "Syrop", "skladniki": [{"nazwa": "liście", "link_id": "czystek"}]}]), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    eng = create_engine(f"sqlite:///{(tmp_path / 'v.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    from datetime import date
    with Session(eng) as s:
        for pid in ("arnika", "babka", "czystek", "swierzabek"):
            s.add(Plant(id=pid, nazwa_pl=pid))
        s.add(SiteStat(day=date.today().isoformat(), kind="plant_view", key="babka", n=9))
        s.commit()
    assert kn.plants_to_research(10) == ["babka", "czystek", "arnika"]       # 9 odslon > 2 przepisy (4) > nic


def test_backlog_of_unverified_facts_is_verified_before_new_research(tmp_path, monkeypatch):
    import app.worker.jobs as jobs_mod
    import app.worker.knowledge as kn
    from app.config import settings
    from app.models import Job, PlantFact
    from tests.test_stage2 import _research, _setup
    eng = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "knowledge_verify_first", 3)
    with Session(eng) as s:
        for i in range(3):
            s.add(PlantFact(plant_id="krwawnik_pospolity", section="ciekawostki", text=f"Informacja numer {i} o krwawniku.",
                            source_url="https://a", fingerprint=f"f{i}", status="new"))
        s.commit()
    from tests.test_knowledge_db import LLM
    LLM.calls = []
    _, log = _research(eng)
    assert "najpierw weryfikacja" in log and not any("redaktorem" in c for c in LLM.calls)
    with Session(eng) as s:
        assert {f.status for f in s.query(PlantFact).all()} == {"verified"}
    assert jobs_mod and kn
