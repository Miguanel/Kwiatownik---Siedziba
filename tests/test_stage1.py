"""Etap 1 diagnostyki 2026-11-02: bledy, ktore powtarzaly sie przy kazdym ponowieniu zadania."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.llm.base import InvalidOutputError, is_garbage, json_object


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- odpowiedzi JSON (bledy 1 i 2)
def test_json_object_accepts_list_and_rejects_garbage():
    assert json_object('```json\n{"typ": "x"}\n```') == {"typ": "x"}
    assert json_object('[{"typ": "a"}, {"typ": "b"}]') == {"typ": "a", "_lista": [{"typ": "b"}]}
    for bad in ('[1, 2]', '"tekst"', 'nie json', '{"error": "Nieprawidlowy format"}',
                '{"question": "Pytanie 2 z 8", "answer": "W jamie ustnej"}', '{"Q": "x", "A": "y"}'):
        with pytest.raises(InvalidOutputError):
            json_object(bad)
    assert is_garbage({"error": "x", "message": "y"}) and not is_garbage({"error": "x", "fakty": []})


def test_llm_classify_survives_list_answer():
    """Blad 1: lista od LLM (gpt-oss-120b) przerywala cale zadanie translate AttributeError-em, a pamiec LLM
    oddawala te sama liste przy kazdym ponowieniu."""
    from app.pipeline.fast_translate import llm_classify

    class LLM:
        def __init__(self, text):
            self.text = text

        async def complete(self, prompt, system=None, json_mode=False):
            return SimpleNamespace(text=self.text, provider="f", model="m")

    rec = {"tytul": "Nalewka", "skladniki_txt": ["jarzebina"], "kroki": ["zalac"]}
    lst = '[{"typ": "medyczne_wewnetrzne_nalewka", "ziola_kluczowe": true}]'
    assert _run(llm_classify(LLM(lst), rec)) == ("medyczne_wewnetrzne_nalewka", True)
    assert _run(llm_classify(LLM('[1]'), rec)) == (None, None)
    assert _run(llm_classify(LLM('{"error": "x"}'), rec)) == (None, None)


def test_translate_list_answer_becomes_extra_recipes():
    from app.pipeline.translate import translate_recipe

    class LLM:
        async def complete(self, prompt, system=None, json_mode=False):
            return SimpleNamespace(text=json.dumps([
                {"tytul": "Syrop z babki", "typ": "medyczne_wewnetrzne_syrop", "skladniki": [{"nazwa": "Liście babki"}],
                 "sposob_przygotowania": ["Zalać.", "Odcedzić."]},
                {"tytul": "Napar z babki", "typ": "medyczne_wewnetrzne_napar", "skladniki": [{"nazwa": "Liście babki"}],
                 "sposob_przygotowania": ["Zaparzyć."]}], ensure_ascii=False), provider="f", model="m")

    rec, _ = _run(translate_recipe(LLM(), None, "tekst", {"item_id": 1, "title": "x", "language": "en"}, {}, set()))
    assert rec["tytul"] == "Syrop z babki" and [e["tytul"] for e in rec["_dodatkowe"]] == ["Napar z babki"]


def test_router_skips_garbage_and_cache_never_keeps_it(tmp_path):
    from app.llm.base import LLMProvider, ModelInfo
    from app.llm.cache import LLMCache
    from app.llm.router import ModelRouter

    class P(LLMProvider):
        name = "ollama"

        def __init__(self):
            super().__init__("", concurrency=1)
            self.calls = []

        async def list_models(self):
            return [ModelInfo("ollama", "zly"), ModelInfo("ollama", "dobry")]

        async def _generate(self, model, prompt, system, json_mode):
            self.calls.append(model)
            return '{"error": "Oczekiwano obiektu"}' if model == "zly" else '{"fakty": []}'

    cache = LLMCache(tmp_path / "c.db")
    p = P()
    r = ModelRouter([p], cache=cache, priority=["ollama:zly", "ollama:dobry"])
    _run(r.refresh())
    res = _run(r.complete("x", json_mode=True))
    assert res.text == '{"fakty": []}' and p.calls == ["zly", "dobry"]
    # stary smiec w pamieci (sprzed poprawki) - usuwany przy starcie i nigdy nie oddawany
    cache.put("klucz-ok", '{"fakty": [1]}', "o", "m")
    import sqlite3
    with sqlite3.connect(tmp_path / "c.db") as c:
        c.execute("INSERT INTO odpowiedzi VALUES ('smiec', '{\"error\": \"x\"}', 'ollama', 'qwen3:8b', ?)",
                  (__import__("time").time(),))
    assert cache.get("smiec") is None
    LLMCache(tmp_path / "c.db")
    with sqlite3.connect(tmp_path / "c.db") as c:
        keys = {k for (k,) in c.execute("SELECT klucz FROM odpowiedzi")}
    assert "smiec" not in keys and "klucz-ok" in keys


# ---------------------------------------------------------------- czas modelu bez kolejki (blad 3)
def test_model_time_excludes_waiting_in_provider_queue():
    from app.llm.base import LLMProvider, ModelInfo
    from app.llm.router import ModelRouter

    class Slow(LLMProvider):
        name = "ollama"

        def __init__(self):
            super().__init__("", concurrency=1)

        async def list_models(self):
            return [ModelInfo("ollama", "m")]

        async def _generate(self, model, prompt, system, json_mode):
            await asyncio.sleep(0.2)
            return "ok"

    async def main():
        r = ModelRouter([Slow()])
        await r.refresh()
        return await asyncio.gather(*(r.complete(f"p{i}") for i in range(3)))

    res = _run(main())
    # trzy zapytania na jednym miejscu: ostatnie czekalo ~0,4 s w kolejce, ale jego czas to tylko ~0,2 s
    assert max(x.seconds for x in res) < 0.35


def test_waiting_for_model_beats_so_watchdog_sees_life(monkeypatch):
    from app.llm import router as router_mod
    beats = []
    monkeypatch.setattr(router_mod, "_beat", lambda job_id=None: beats.append(job_id) or 7)

    async def main():
        task = asyncio.create_task(router_mod._heartbeat(7, every=0.01))
        await asyncio.sleep(0.05)
        task.cancel()

    _run(main())
    assert beats.count(7) >= 3


# ---------------------------------------------------------------- log bledu z miejscem (blad 8)
def test_error_place_points_to_siedziba_code():
    from app.worker.runner import error_place
    try:
        json_object("[1]")
    except InvalidOutputError as exc:
        place = error_place(exc)
    assert place.startswith(" (app/llm/base.py:") and "json_object" in place


# ---------------------------------------------------------------- ten sam blad w kolko (blad 9)
def test_error_signature_ignores_times_and_numbers():
    from app.agents.planner import error_signature
    a = error_signature("17:07:35 Start\n17:07:35 BLAD: AttributeError: 'list' (app/x.py:212 w f)")
    b = error_signature("09:00:01 BLAD: AttributeError: 'list' (app/x.py:212 w f)\n09:10:00 Ponowiono jako #5")
    assert a == b and error_signature("bez bledu") is None


def test_planner_pauses_proposal_failing_with_same_error(tmp_path, monkeypatch):
    from app.agents import planner, store
    from app.models import AgentTask, Job
    eng = create_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(store, "now", lambda: now)
    with Session(eng) as s:
        for i, err in enumerate(["AttributeError: x", "AttributeError: x", "AttributeError: x", "KeyError: y"]):
            j = Job(kind="translate", status="failed", log=f"10:0{i}:00 BLAD: {err}")
            s.add(j)
            s.commit()
            key = "grupa:translate" if i < 3 else "grupa:scan"
            s.add(AgentTask(agent="planista", kind="tlumacz", title="t", status="failed", job_id=j.id,
                            params_json=json.dumps({"klucz": key}), ordered_at=now - timedelta(hours=i)))
        s.commit()
        rep = planner._repeated_errors(s)
    assert rep == {"grupa:translate": "AttributeError: x"}


# ---------------------------------------------------------------- bezpiecznik 403/429
def test_crawler_stops_when_site_refuses_access():
    from app.scrapers.crawler import BLOCK_STREAK, CrawlConfig, SmartCrawler
    from app.scrapers.fetcher import Fetcher
    from tests.test_scraper import MemoryStore
    hits = []

    def handler(req: httpx.Request):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        if req.url.path == "/":
            links = "".join(f'<a href="/brand/{i}">b{i}</a>' for i in range(60))
            return httpx.Response(200, text=f"<html><body><h1>Sklep</h1>{links}</body></html>",
                                  headers={"content-type": "text/html"})
        if req.url.path.startswith("/brand/"):
            hits.append(req.url.path)
        return httpx.Response(403)

    async def run():
        f = Fetcher("TestBot", delay_s=0, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            return await SmartCrawler("https://shop.example/", f, MemoryStore(), CrawlConfig(max_pages=60)).run()
        finally:
            await f.aclose()

    stats = _run(run())
    assert stats["blocked"] == BLOCK_STREAK and len(hits) == BLOCK_STREAK


def test_blocked_source_pauses_then_turns_off(monkeypatch):
    from app.worker import scan
    logs = []
    monkeypatch.setattr(scan, "job_log", lambda jid, m: logs.append(m))
    src = SimpleNamespace(block_count=0, paused_until=None, active=True, auto_off_reason=None, auto_off_at=None)
    t0 = datetime.now(timezone.utc)
    for hours in (1, 6, 24):
        scan._blocked(1, src)
        assert src.active and timedelta(hours=hours) - timedelta(minutes=1) < src.paused_until - t0 \
            < timedelta(hours=hours, minutes=1)
        assert scan.scan_paused(src)
    scan._blocked(1, src)
    assert not src.active and src.paused_until is None and "blokuje" in src.auto_off_reason
    assert not scan.scan_paused(SimpleNamespace(paused_until=t0 - timedelta(minutes=1)))


def test_unreadable_robots_closes_site_instead_of_crashing(monkeypatch):
    from app.scrapers.fetcher import Fetcher
    f = Fetcher("TestBot", delay_s=0, client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))

    async def boom(url):
        raise ValueError("unknown url type: '/robots.txt'")

    monkeypatch.setattr(f, "_raw_get", boom)
    assert _run(f.allowed("https://www.ukrinform.ua/x")) is False
    _run(f.aclose())


# ---------------------------------------------------------------- dwa wpisy jednego gatunku (kasztanowiec)
def test_duplicate_species_from_siedziba_merges_into_handmade_plant(tmp_path):
    from app.knowledge import registry
    from app.knowledge.sections import fingerprint
    from app.models import Item, ItemKind, Plant, PlantAlias, PlantFact, PlantQuery
    plants, backups = tmp_path / "plants", tmp_path / "backups"
    plants.mkdir()
    (plants / "kasztanowiec_zwyczajny.json").write_text(json.dumps(
        {"id": "kasztanowiec_zwyczajny", "nazwa_pl": "Kasztanowiec zwyczajny", "nazwa_lat": "Aesculus hippocastanum"}),
        encoding="utf-8")
    (plants / "kasztanowiec.json").write_text(json.dumps(
        {"id": "kasztanowiec", "nazwa_pl": "Kasztanowiec pospolity", "nazwa_lat": "Aesculus hippocastanum L.",
         "_siedziba_nowa": True}), encoding="utf-8")
    (plants / "dab.json").write_text(json.dumps({"id": "dab", "nazwa_pl": "Dąb", "nazwa_lat": "Quercus robur"}),
                                     encoding="utf-8")
    eng = create_engine(f"sqlite:///{(tmp_path / 'k.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        for pid in ("kasztanowiec_zwyczajny", "kasztanowiec", "dab"):
            s.add(Plant(id=pid, nazwa_pl=pid))
        s.commit()
        same = "Nasiona zawieraja escyne i sa trujace dla ludzi."
        s.add(PlantFact(plant_id="kasztanowiec_zwyczajny", section="bezpieczenstwo", text=same, source_url="https://a",
                        fingerprint=fingerprint("kasztanowiec_zwyczajny", same), status="applied"))
        for txt, st in ((same, "applied"), ("Z kwiatow robiono nalewke na zylaki w Galicji.", "applied"),
                        ("Kora sluzyla do barwienia welny na brazowo.", "new")):
            s.add(PlantFact(plant_id="kasztanowiec", section="ciekawostki", text=txt, source_url="https://b",
                            fingerprint=fingerprint("kasztanowiec", txt), status=st))
        s.add(PlantQuery(plant_id="kasztanowiec", slot="x", language="pl", query="q", results=0))
        s.add(Item(url="https://a/x", kind=ItemKind.recipe, title="Nalewka", data_json=json.dumps(
            {"tytul": "Nalewka", "roslina_id": "kasztanowiec", "skladniki": [{"nazwa": "kwiaty", "link_id": "kasztanowiec"}]})))
        s.commit()
        msgs = registry.merge_duplicate_species(s, plants, backups)
        facts = s.exec(select(PlantFact).where(PlantFact.plant_id == "kasztanowiec_zwyczajny")).all()
        by_text = {f.text: f.status for f in facts}
        assert s.get(Plant, "kasztanowiec") is None and len(msgs) == 1
        assert by_text["Z kwiatow robiono nalewke na zylaki w Galicji."] == "verified"     # wraca do zapisu (brama)
        assert by_text["Kora sluzyla do barwienia welny na brazowo."] == "new"
        assert sorted(by_text.values()).count("rejected") == 1                           # powtorka
        assert s.exec(select(PlantQuery)).one().plant_id == "kasztanowiec_zwyczajny"
        rec = json.loads(s.exec(select(Item)).one().data_json)
        assert rec["roslina_id"] == "kasztanowiec_zwyczajny" and rec["skladniki"][0]["link_id"] == "kasztanowiec_zwyczajny"
        assert s.get(PlantAlias, "kasztanowiec").plant_id == "kasztanowiec_zwyczajny"
    assert not (plants / "kasztanowiec.json").exists() and (plants / "kasztanowiec_zwyczajny.json").exists()
    assert len(list((backups / "kasztanowiec").glob("*.json"))) == 1


def test_handmade_duplicates_are_never_merged(tmp_path):
    from app.knowledge import registry
    from app.models import Plant
    plants = tmp_path / "plants"
    plants.mkdir()
    for pid in ("a", "b"):
        (plants / f"{pid}.json").write_text(json.dumps({"id": pid, "nazwa_pl": pid, "nazwa_lat": "Aesculus x"}),
                                            encoding="utf-8")
    eng = create_engine(f"sqlite:///{(tmp_path / 'k.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Plant(id="a", nazwa_pl="a"))
        s.add(Plant(id="b", nazwa_pl="b"))
        s.commit()
        assert registry.merge_duplicate_species(s, plants, tmp_path / "bk") == []
    assert (plants / "a.json").exists() and (plants / "b.json").exists()
