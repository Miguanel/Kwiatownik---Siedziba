"""Wzmianki o przepisach: zapis z tlumaczenia i z ciekawostek, szukanie pelnego przepisu, panel /wzmianki."""
import asyncio
import json

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.discovery.search import SearchResult
from app.models import Item, ItemKind, ItemStatus, Job, RecipeLead
from app.pipeline import translate as tr
from app.scrapers.fetcher import FetchResult

LEAD = {"tytul": "Herbata z dokudami na tradzik", "tytul_oryginalny": "ドクダミ茶", "roslina": "pstrolistka sercowata",
        "roslina_lacinska": "Houttuynia cordata", "typ": "medyczne_wewnetrzne_napar",
        "co_wiadomo": "W Japonii pije sie ja na tradzik.", "znane_skladniki": ["liscie pstrolistki"], "cytat": "ドクダミ茶は"}


@pytest.fixture
def eng(tmp_path, monkeypatch):
    import importlib
    e = create_engine(f"sqlite:///{(tmp_path / 'l.db').as_posix()}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(e)
    for name in ("app.pipeline.leads", "app.worker.leads", "app.worker.jobs", "app.web.leads_routes"):
        monkeypatch.setattr(importlib.import_module(name), "engine", e)
    return e


class _Res:
    def __init__(self, text):
        self.text, self.provider, self.model = text, "fake", "m"


class _LLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def complete(self, *a, **k):
        self.calls += 1
        return _Res(json.dumps(self.answer))


def test_clean_leads_and_partial_recipe():
    leads = tr.clean_leads([LEAD, {"tytul": "Zupa", "typ": "kulinarne"}, {"tytul": ""}, "x"])
    assert len(leads) == 1 and leads[0]["typ"] == "medyczne_wewnetrzne_napar"
    lead = tr.lead_from_partial({"tytul": "Masc z nagietka", "skladniki": [{"nazwa": "nagietek"}], "opis": "na rany"})
    assert lead["tytul"] == "Masc z nagietka" and lead["znane_skladniki"] == ["nagietek"]
    assert tr.lead_from_partial({"tytul": "Pierogi", "typ": "kulinarne"}) is None


def test_translate_raises_with_leads():
    llm = _LLM({"brak_przepisu": True, "wzmianki": [LEAD]})
    with pytest.raises(tr.NoRecipeError) as ei:
        asyncio.run(tr.translate_recipe(llm, None, "tekst", {"item_id": 1, "title": "t"}, {}, set()))
    assert ei.value.leads[0]["tytul_oryginalny"] == "ドクダミ茶"
    llm = _LLM({"tytul": "Nalewka z dziurawca", "skladniki": [{"nazwa": "dziurawiec"}], "sposob_przygotowania": []})
    with pytest.raises(tr.IncompleteRecipeError) as ei:
        asyncio.run(tr.translate_recipe(llm, None, "tekst", {"item_id": 2, "title": "t"}, {}, set()))
    assert ei.value.leads[0]["tytul"] == "Nalewka z dziurawca"


def test_save_leads_dedupes(eng):
    from app.pipeline.leads import save_leads
    a = save_leads([LEAD], "https://jp.example/a", "jp.example", "ja", 7, "m")
    b = save_leads([dict(LEAD, cytat="inny")], "https://jp.example/b", "jp.example", "ja", 8, "m")
    assert len(a) == 1 and b == []
    with Session(eng) as s:
        row = s.get(RecipeLead, a[0])
        assert row.status == "szukam" and json.loads(row.ingredients_json) == ["liscie pstrolistki"]


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages

    async def get(self, url, html_only=True):
        if url in self.pages:
            return FetchResult(url, 200, self.pages[url], "text/html")
        return FetchResult(url, 404, error="404")

    async def aclose(self):
        pass


class FakeBackend:
    name = "fake"

    def __init__(self, urls):
        self.urls, self.queries = urls, []

    async def search(self, query, language, country, page=1):
        self.queries.append((query, language))
        return [SearchResult(u) for u in self.urls]


RECIPE_HTML = """<html><head><title>Dokudami tea</title><script type="application/ld+json">
{"@type": "Recipe", "name": "Dokudami tea (Houttuynia cordata)", "recipeIngredient": ["10 g dried Houttuynia cordata leaves", "500 ml water"],
 "recipeInstructions": ["Bring water to a boil.", "Add the dried leaves and simmer 10 minutes as a herbal remedy for skin."]}
</script></head><body><article><h1>Dokudami tea</h1><p>Houttuynia cordata herbal remedy for acne and skin. """ + "Herbal tea. " * 60 + """</p></article></body></html>"""


def test_search_lead_finds_recipe_and_creates_item(eng):
    from app.pipeline import leads as lm
    lid = lm.save_leads([LEAD], "https://jp.example/a", "jp.example", "ja", None, "m")[0]
    other = "https://x.example/glycyrrhiza"
    good = "https://herbs.example/dokudami-tea"
    pages = {other: "<html><body><p>" + "Glycyrrhiza uralensis root. " * 80 + "</p></body></html>", good: RECIPE_HTML}
    backend = FakeBackend([other, "https://www.youtube.com/watch?v=1", good])
    logs = []
    res = asyncio.run(lm.search_lead(lid, None, FakeFetcher(pages), [backend], logs.append))
    assert res["found"]
    with Session(eng) as s:
        lead = s.get(RecipeLead, lid)
        it = s.get(Item, res["found"])
        assert lead.status == "znaleziony" and lead.found_url == good and lead.attempts == 1
        assert it.kind == ItemKind.recipe and it.status == ItemStatus.raw and it.method == "wzmianka:jsonld"
        assert json.loads(it.structured_json)["ingredients"]
        log = json.loads(lead.search_log)
        assert any(h.get("wynik") == "nie o tej roslinie" for h in log)
    assert backend.queries[0] == ('"ドクダミ茶" 作り方', "ja")


def test_search_lead_gives_up_after_attempts(eng):
    from app.pipeline import leads as lm
    lid = lm.save_leads([LEAD], "https://jp.example/a", "jp.example", "ja", None, "m")[0]
    for _ in range(lm.MAX_ATTEMPTS):
        asyncio.run(lm.search_lead(lid, None, FakeFetcher({}), [FakeBackend([])], lambda m: None))
    with Session(eng) as s:
        assert s.get(RecipeLead, lid).status == "nie_znaleziono"
    assert lm.leads_to_search(10) == []


def test_extract_job_reads_fact_articles(eng):
    from app.worker import leads as wl
    from app.worker.jobs import make_runner
    with Session(eng) as s:
        s.add(Item(url="https://jp.example/fact", kind=ItemKind.fact, status=ItemStatus.raw, title="Dokudami",
                   raw_text="ドクダミ茶 " * 50, language="ja"))
        job = Job(kind="leads_extract")
        s.add(job)
        s.commit()
        jid = job.id
    llm = _LLM({"wzmianki": [LEAD]})

    async def run():
        runner = make_runner(1)
        runner.start(jid, lambda: wl.run_leads_extract(jid, runner, llm, 10))
        await asyncio.gather(*runner.tasks.values())

    asyncio.run(run())
    with Session(eng) as s:
        assert s.exec(select(RecipeLead)).one().item_id is not None
        assert s.exec(select(Item)).one().leads_at is not None
        assert json.loads(s.get(Job, jid).stats_json)["wzmianki"] == 1


def test_leads_panel(eng):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.pipeline.leads import save_leads
    lid = save_leads([LEAD], "https://jp.example/a", "jp.example", "ja", None, "m")[0]
    with TestClient(app) as c:
        r = c.get("/wzmianki")
        assert r.status_code == 200 and "Herbata z dokudami" in r.text and "Szukaj teraz" in r.text
        r = c.post(f"/wzmianki/{lid}/status", data={"status": "odrzucony"}, follow_redirects=False)
        assert r.status_code == 303
        with Session(eng) as s:
            assert s.get(RecipeLead, lid).status == "odrzucony"
        assert "Herbata z dokudami" in c.get("/wzmianki?status=odrzucony").text
        assert c.get("/tasks").status_code == 200
