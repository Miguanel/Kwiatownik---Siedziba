"""Zagraniczne encyklopedie: wybor zrodel, adresy, szukanie hasla (szablon / wyszukiwarka serwisu / site:), panel."""
import asyncio
import json

from sqlmodel import Session, SQLModel, create_engine, select

from app.config import settings
from app.discovery.search import SearchResult
from app.knowledge import encyclopedias as encmod
from app.models import Plant, PlantQuery
from app.scrapers.fetcher import FetchResult

NAMES = {"la": "Houttuynia cordata", "ja": "ドクダミ", "zh": "蕺菜", "pl": "pstrolistka sercowata"}


def _html(body: str) -> str:
    return f"<html><head><title>t</title></head><body><article><p>{body}</p></article></body></html>"


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages, self.calls = pages, []

    async def get(self, url, html_only=True):
        self.calls.append(url)
        if url in self.pages:
            return FetchResult(url, 200, self.pages[url], "text/html")
        return FetchResult(url, 404, "", "text/html", error="404")

    async def aclose(self):
        pass


class FakeBackend:
    name = "fake"

    def __init__(self, urls):
        self.urls, self.queries = urls, []

    async def search(self, query, language, country, page=1):
        self.queries.append(query)
        return [SearchResult(u) for u in self.urls]


def test_urls_and_names():
    kb = encmod.BY_KEY["kotobank"]
    assert encmod.urls_for(kb, NAMES) == ["https://kotobank.jp/word/%E3%83%89%E3%82%AF%E3%83%80%E3%83%9F"]
    assert encmod.urls_for(encmod.BY_KEY["pfaf"], NAMES) == [
        "https://pfaf.org/user/Plant.aspx?LatinName=Houttuynia+cordata"]
    assert encmod.plant_name(encmod.BY_KEY["encykorea"], NAMES) is None        # brak nazwy koreanskiej


def test_choose_prefers_unused_sources_and_matching_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    picks = encmod.choose(NAMES, ["wschod"], set(), 3)
    assert len(picks) == 3 and all(p.name_lang in ("ja", "zh", "la") for p in picks)
    assert picks[0].country in ("jp", "cn")                                       # medycyna Wschodu -> Azja
    used = {p.domain for p in picks}
    again = encmod.choose(NAMES, ["wschod"], used, 3)
    assert not used & {p.domain for p in again}                                   # rotacja: kolejne zrodla
    encmod.set_enabled("kotobank", False)
    assert "kotobank" not in {p.key for p in encmod.choose(NAMES, ["wschod"], set(), 20)}
    assert json.loads((tmp_path / "encyclopedias.json").read_text())["kotobank"] is False


def test_fetch_template_then_site_search_then_rejects_other_plant():
    text = "ドクダミ（蕺草、学名: Houttuynia cordata）は、ドクダミ科ドクダミ属の多年草。" * 20
    kb = encmod.BY_KEY["kotobank"]
    url = encmod.urls_for(kb, NAMES)[0]
    f = asyncio.run(encmod.fetch(kb, NAMES, FakeFetcher({url: _html(text)})))
    assert f.url == url and f.how == "adres hasla" and "Houttuynia" in f.text

    # szablon 404 -> wyszukiwarka serwisu (link do hasla)
    search = "https://kotobank.jp/search?q=%E3%83%89%E3%82%AF%E3%83%80%E3%83%9F"
    pages = {search: '<a href="/word/ドクダミ-104693">x</a>', "https://kotobank.jp/word/ドクダミ-104693": _html(text)}
    f = asyncio.run(encmod.fetch(kb, NAMES, FakeFetcher(pages)))
    assert f.how == "wyszukiwarka serwisu" and f.url.endswith("104693")

    # zrodlo bez szablonu -> SearXNG site:, strona o innej roslinie odrzucona
    mp = encmod.BY_KEY["mpdb"]
    other = "https://mpdb.nibiohn.go.jp/x/1"
    good = "https://mpdb.nibiohn.go.jp/x/2"
    fetcher = FakeFetcher({other: _html("Glycyrrhiza uralensis " * 60), good: _html("Houttuynia cordata 十薬 " * 60)})
    backend = FakeBackend([other, "https://inna.example/z", good])
    f = asyncio.run(encmod.fetch(mp, NAMES, fetcher, [backend]))
    assert f.url == good and f.how == "wyszukiwarka (site:)"
    assert backend.queries == ["Houttuynia cordata site:mpdb.nibiohn.go.jp"]
    assert "https://inna.example/z" not in fetcher.calls                         # inne domeny pomijane

    nothing = asyncio.run(encmod.fetch(mp, NAMES, FakeFetcher({}), [FakeBackend([])]))
    assert nothing.url is None and "nie znalazla" in nothing.reason


def test_research_step_logs_queries_and_skips_recent(tmp_path, monkeypatch):
    import app.worker.knowledge as kn
    eng = create_engine(f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "knowledge_enc_max", 2)
    with Session(eng) as s:
        s.add(Plant(id="pstrolistka", nazwa_pl="pstrolistka sercowata", nazwa_lat="Houttuynia cordata",
                    names_json=json.dumps({"ja": "ドクダミ", "zh": "蕺菜"})))
        s.commit()
        plant = s.get(Plant, "pstrolistka")
        s.expunge(plant)
    seen = []

    async def fake_fetch(enc, names, fetcher, backends=None, skip_urls=None, log=None):
        seen.append(enc.key)
        f = encmod.Found(enc)
        if len(seen) == 1:
            f.url, f.text, f.how = f"https://{enc.domain}/hasla", "Houttuynia cordata " * 50, "adres hasla"
        else:
            f.reason = "nie znaleziono hasla"
        return f

    monkeypatch.setattr(encmod, "fetch", fake_fetch)
    monkeypatch.setattr("app.worker.discover.build_backends", lambda: [])
    logs = []
    out = asyncio.run(kn._encyclopedia_sources(plant, logs.append, None))
    assert len(seen) == 2 and len(out) == 1 and out[0][0].endswith("/hasla") and out[0][5]
    with Session(eng) as s:
        rows = s.exec(select(PlantQuery)).all()
        assert {r.slot for r in rows} == {f"enc:{k}" for k in seen} and sum(1 for r in rows if r.used_url) == 1
    out2 = asyncio.run(kn._encyclopedia_sources(plant, logs.append, None))          # rotacja: inne zrodla
    assert len(seen) == 4 and not set(seen[:2]) & set(seen[2:])


def test_panel_page_toggle_and_test(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def fake_fetch(enc, names, fetcher, backends=None, skip_urls=None, log=None):
        f = encmod.Found(enc)
        if enc.key == "pfaf":
            f.url, f.text, f.how = "https://pfaf.org/user/Plant.aspx?LatinName=X", "Opis " * 80, "adres hasla"
        else:
            f.reason = "zablokowane przez robots.txt"
        return f

    monkeypatch.setattr(encmod, "fetch", fake_fetch)
    with TestClient(app) as c:
        from app.db import engine
        with Session(engine) as s:
            if not s.get(Plant, "enc-test"):
                s.add(Plant(id="enc-test", nazwa_pl="krwawnik", nazwa_lat="Achillea millefolium",
                            names_json=json.dumps({"ja": "セイヨウノコギリソウ"})))
                s.commit()
        r = c.get("/encyklopedie")
        assert r.status_code == 200 and "Kotobank" in r.text and "Native American Ethnobotany" in r.text
        r = c.post("/encyklopedie/kotobank", data={"on": "0"}, follow_redirects=False)
        assert r.status_code == 303 and encmod.enabled_map()["kotobank"] is False
        r = c.post("/encyklopedie/test", data={"plant_id": "enc-test", "key": ""})
        assert r.status_code == 200 and "znaleziono" in r.text and "robots.txt" in r.text
        assert "brak nazwy gatunku w jezyku &#x27;zh&#x27;" in r.text or "brak nazwy gatunku w jezyku 'zh'" in r.text
        assert c.get("/plants").status_code == 200
