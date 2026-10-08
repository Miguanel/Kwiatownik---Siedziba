"""Caly przebieg wiedzy o roslinie z baza SQLite: Wikipedia -> informacje -> weryfikacja -> zapis do pliku."""
import asyncio
import json

import httpx
from sqlmodel import Session, SQLModel, create_engine, select

import app.worker.jobs as jobs_mod
import app.worker.knowledge as kn
from app.knowledge.registry import discover_new_plants, sync_kwiatownik
from app.knowledge.wiki import WikiClient
from app.models import Item, ItemKind, ItemStatus, Job, Plant, PlantAlias, PlantFact
from tests.test_knowledge import wiki_handler


class LLM:
    """Wyciaganie: 2 informacje (jedna z prawdziwym cytatem, druga zmyslona); weryfikacja: potwierdza."""
    calls: list = []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None):
        LLM.calls.append((system or "")[:30])

        class R:
            provider, model = "fake", "m"
        r = R()
        if "weryfikatorem" in (system or ""):
            ids = [int(x) for x in __import__("re").findall(r"ID (\d+)", prompt)]
            r.text = json.dumps({"oceny": [{"id": i, "ok": True} for i in ids]})
        elif "skladnikow" in (system or "").lower() or "SKLADNIKI" in prompt:
            r.text = json.dumps({"wyniki": [{"nazwa": "liście szczawiu", "roslina": True, "nazwa_pl": "Szczaw zwyczajny",
                                             "nazwa_lat": "Rumex acetosa"},
                                            {"nazwa": "cukier", "roslina": False}]})
        else:
            r.text = json.dumps({"fakty": [
                {"sekcja": "barwienie", "czesc": "kwiaty", "tekst": "Kwiatami krwawnika barwiono dawniej wełnę na żółto.",
                 "cytat": "Kwiaty służyły do barwienia wełny na żółto."},
                {"sekcja": "kultura", "tekst": "Krwawnik wieszano nad drzwiami, by chronił dom przed złem.",
                 "cytat": "zmyslony cytat ktorego nie ma w zrodle wcale"}]})
        return r


def test_research_verifies_and_writes_plant_file(tmp_path, monkeypatch):
    from app.config import settings
    plants = tmp_path / "plants"
    plants.mkdir()
    original = {"id": "krwawnik_pospolity", "nazwa_pl": "Krwawnik pospolity", "nazwa_lat": "Achillea millefolium",
                "opis": "Ręczny opis.", "ciekawostki": ["Ręczna ciekawostka."]}
    (plants / "krwawnik_pospolity.json").write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "backups_dir", tmp_path / "backups")
    monkeypatch.setattr(settings, "knowledge_web_pages", 0)
    monkeypatch.setattr(settings, "knowledge_auto_apply", True)
    eng = create_engine(f"sqlite:///{(tmp_path / 'k.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        assert sync_kwiatownik(s, plants) == 1
        job = Job(kind="plant_research")
        s.add(job)
        s.commit()
        jid = job.id

    async def run():
        wiki = WikiClient("T", client=httpx.AsyncClient(transport=httpx.MockTransport(wiki_handler)), delay_s=0)
        try:
            return await kn.run_plant_research(jid, jobs_mod.make_runner(1), LLM(), ["krwawnik_pospolity"], wiki=wiki)
        finally:
            await wiki.aclose()
    stats = asyncio.run(run())
    assert stats["applied"] == 1 and stats["facts"] >= 1
    data = json.loads((plants / "krwawnik_pospolity.json").read_text(encoding="utf-8"))
    assert data["opis"] == "Ręczny opis." and data["ciekawostki"] == ["Ręczna ciekawostka."]   # reczne pola bez zmian
    facts = data["wiedza"]["fakty"]
    assert [f["sekcja"] for f in facts] == ["barwienie"]                   # zmyslony cytat odrzucony przez kod
    assert facts[0]["zrodlo"]["url"].startswith("https://pl.wikipedia.org/")
    w = data["wiedza"]                                                    # agregator: uklad dla strony
    assert w["wersja"] == 2 and w["sekcje"][0]["klucz"] == "barwienie" and w["zrodla"][0]["nr"] == 1
    assert w["sekcje"][0]["punkty"][0]["zrodla"] == [1] and w["czesci"] == {"kwiaty": [facts[0]["id"]]}
    assert list((tmp_path / "backups" / "plants" / "krwawnik_pospolity").glob("*.json"))    # kopia zapasowa
    with Session(eng) as s:
        assert s.exec(select(PlantFact)).one().status == "applied"
        assert s.get(Plant, "krwawnik_pospolity").wikidata == "Q25408"


def test_new_plants_discovered_from_recipe_ingredients(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{(tmp_path / 'n.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Plant(id="krwawnik_pospolity", nazwa_pl="Krwawnik pospolity", nazwa_lat="Achillea millefolium"))
        s.add(Item(url="https://a.pl/1", kind=ItemKind.recipe, status=ItemStatus.translated, data_json=json.dumps({
            "tytul": "Napar", "skladniki": [{"nazwa": "liście szczawiu", "link_id": None},
                                            {"nazwa": "cukier", "link_id": None},
                                            {"nazwa": "krwawnik", "link_id": "krwawnik_pospolity"}]})))
        s.commit()
        created = asyncio.run(discover_new_plants(s, LLM()))
        assert created == ["szczaw_zwyczajny"]
        p = s.get(Plant, "szczaw_zwyczajny")
        assert p.origin == "nowa" and p.nazwa_lat == "Rumex acetosa" and p.mentions == 1
        assert {a.name: a.plant_id for a in s.exec(select(PlantAlias)).all()} == {
            "liscie szczawiu": "szczaw_zwyczajny", "cukier": None}
        calls = len(LLM.calls)
        assert asyncio.run(discover_new_plants(s, LLM())) == [] and len(LLM.calls) == calls   # bez ponownego pytania


def test_organize_job_rewrites_only_knowledge_block(tmp_path, monkeypatch):
    from app.config import settings
    from tests.test_organize import FACTS, GOOD
    plants = tmp_path / "plants"
    plants.mkdir()
    original = {"id": "babka", "nazwa_pl": "Babka lancetowata", "opis": "Ręczny opis.", "wiedza": {"fakty": FACTS}}
    raw = json.dumps(original, ensure_ascii=False, indent=2).replace("\n", "\r\n")
    (plants / "babka.json").write_bytes(raw.encode("utf-8"))
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "backups_dir", tmp_path / "backups")
    eng = create_engine(f"sqlite:///{(tmp_path / 'o.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        s.add(Plant(id="babka", nazwa_pl="Babka lancetowata", nazwa_lat="Plantago lanceolata"))
        job = Job(kind="plant_organize")
        s.add(job)
        s.commit()
        jid = job.id

    class Editor:
        async def complete(self, prompt, **kw):
            class R:
                provider, model, text = "fake", "ed", json.dumps({"sekcje": GOOD}, ensure_ascii=False)
            return R()
    assert kn.organized_plant_ids() == ["babka"]
    stats = asyncio.run(kn.run_plant_organize(jid, jobs_mod.make_runner(1), Editor()))
    assert stats["organized"] == 1
    body = (plants / "babka.json").read_bytes()
    data = json.loads(body.decode("utf-8"))
    assert b"\r\n" in body and data["opis"] == "Ręczny opis." and len(data["wiedza"]["fakty"]) == 6
    assert data["wiedza"]["uklad"] == "llm:fake/ed" and len(data["wiedza"]["sekcje"]) == 4
    assert list((tmp_path / "backups" / "plants" / "babka").glob("*.json"))


def test_photos_job_fills_empty_gallery_and_keeps_manual(tmp_path, monkeypatch):
    from app.config import settings
    from tests import test_photos
    plants = tmp_path / "plants"
    plants.mkdir()
    (plants / "barszcz_zwyczajny.json").write_bytes(json.dumps(
        {"id": "barszcz_zwyczajny", "nazwa_pl": "Barszcz zwyczajny", "nazwa_lat": "Heracleum sphondylium",
         "opis": "Ręczny.", "url": None}, ensure_ascii=False).replace("}", "}\r\n").encode("utf-8"))
    manual = {"kwiaty": test_photos.UP.format("Heracleum_sphondylium_habitus.jpg").replace("/a/ab/", "/6/61/")}
    (plants / "bez.json").write_text(json.dumps({"id": "bez", "nazwa_pl": "Bez", "nazwa_lat": "Heracleum sphondylium",
                                                 "url": manual}), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "backups_dir", tmp_path / "backups")
    eng = create_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        job = Job(kind="plant_photos")
        s.add(job)
        s.commit()
        jid = job.id

    def handler(request):
        p = dict(request.url.params)
        if request.url.host == "www.wikidata.org" and p.get("action") == "wbsearchentities":
            return httpx.Response(200, json={"search": [{"id": "Q5"}]})
        if request.url.host == "www.wikidata.org":
            return httpx.Response(200, json={"entities": {"Q5": {
                "claims": {"P225": [{"mainsnak": {"datavalue": {"value": "Heracleum sphondylium"}}}],
                           "P18": [{"mainsnak": {"datavalue": {"value": "Heracleum sphondylium habitus.jpg"}}}],
                           "P373": [{"mainsnak": {"datavalue": {"value": "Heracleum sphondylium"}}}]},
                "sitelinks": {"plwiki": {"title": "Barszcz zwyczajny"}}}}})
        return test_photos.handler(request)

    async def run():
        wiki = WikiClient("T", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), delay_s=0)
        try:
            return await kn.run_plant_photos(jid, jobs_mod.make_runner(1), None, wiki=wiki)
        finally:
            await wiki.aclose()
    stats = asyncio.run(run())
    assert stats["applied"] == 2
    body = (plants / "barszcz_zwyczajny.json").read_bytes()
    data = json.loads(body.decode("utf-8"))
    assert b"\r\n" in body and data["opis"] == "Ręczny." and list(data["url"])[:2] == ["Pokrój", "Kwiaty"]
    assert data["zdjecia_wiki"]["w_url"] and data["zdjecia_wiki"]["zdjecia"][0]["autor"] == "Jan Kowalski"
    bez = json.loads((plants / "bez.json").read_text(encoding="utf-8"))
    assert bez["url"] == manual and bez["zdjecia_wiki"]["w_url"] is False
    assert bez["zdjecia_wiki"]["zdjecia"][0]["reczne"] is True


def test_merge_job_writes_scalone_and_unmerge_restores(tmp_path, monkeypatch):
    """plant_merge: wiedza z sieci wmontowana w rozdzialy (blok "scalone"), reczne pola bez zmian, CRLF zostaje;
    unmerge usuwa scalenie. Model scalania niedostepny -> scalenie regulami."""
    from app.config import settings
    from tests.test_merge import PLANT
    plants = tmp_path / "plants"
    plants.mkdir()
    raw = json.dumps(PLANT, ensure_ascii=False, indent=2).replace("\n", "\r\n")
    (plants / "dziurawiec_zwyczajny.json").write_bytes(raw.encode("utf-8"))
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "backups_dir", tmp_path / "backups")
    eng = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        s.add(Plant(id="dziurawiec_zwyczajny", nazwa_pl="Dziurawiec zwyczajny", nazwa_lat="Hypericum perforatum"))
        job = Job(kind="plant_merge")
        s.add(job)
        s.commit()
        jid = job.id

    class Down:
        async def complete(self, prompt, **kw):
            raise RuntimeError("brak modeli")
    stats = asyncio.run(kn.run_plant_merge(jid, jobs_mod.make_runner(1), Down()))
    assert stats["merged"] == 1
    body = (plants / "dziurawiec_zwyczajny.json").read_bytes()
    data = json.loads(body.decode("utf-8"))
    assert b"\r\n" in body and data["zastosowanie"] == PLANT["zastosowanie"] and data["ciekawostki"] == PLANT["ciekawostki"]
    pola = data["scalone"]["pola"]
    assert pola["zastosowanie.rzemieslnicze"]["tresc"][0]["zrodla"] == [2]
    assert pola["ciekawostki"]["tresc"][0]["tekst"] == "Nazywany zielem świętojańskim."
    with Session(eng) as s:
        row = s.get(Plant, "dziurawiec_zwyczajny")
        assert row.last_merge_at is not None and row.coverage and "profil_energetyczny.termika" in row.gaps_json
    assert kn.unmerge_plant("dziurawiec_zwyczajny")
    assert "scalone" not in json.loads((plants / "dziurawiec_zwyczajny.json").read_text(encoding="utf-8"))


def test_gap_queries_are_logged_and_not_repeated(tmp_path, monkeypatch):
    from app.knowledge import gaps as gapmod
    from app.models import PlantQuery
    eng = create_engine(f"sqlite:///{(tmp_path / 'q.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    gq = gapmod.GapQuery(gapmod.Gap("profil_energetyczny.termika", "Termika", "wschod"), "zh", '"贯叶连翘" 性味归经 功能主治')
    qid = kn._log_query("dziurawiec", gq, 8, "https://example.cn/a")
    kn._query_facts(qid, 3)
    done, urls = kn._recent_queries("dziurawiec")
    assert gq.query in done and "https://example.cn/a" in urls
    with Session(eng) as s:
        assert s.get(PlantQuery, qid).facts == 3
