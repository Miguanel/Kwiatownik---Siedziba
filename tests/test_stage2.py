"""Etap 2 diagnostyki 2026-11-02: mniej powtorzonej pracy LLM przy zbieraniu wiedzy."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
from sqlmodel import Session, SQLModel, create_engine, select

import app.worker.jobs as jobs_mod
import app.worker.knowledge as kn
from app.config import settings
from app.knowledge import novelty
from app.knowledge.registry import sync_kwiatownik
from app.knowledge.wiki import WikiClient
from app.models import Job, Plant, PlantQuery, SourceRead
from tests.test_knowledge import wiki_handler
from tests.test_knowledge_db import LLM

LONG = " ".join(["Zdanie o czyms zupelnie innym, co nie dotyczy rosliny ani zielnika."] * 6)


# ---------------------------------------------------------------- filtr nowosci (bez bazy)
def test_paragraphs_join_short_lines_and_cut_long_ones():
    text = "Naglowek\nKrotka linia.\n\n" + ("Dlugie zdanie o krwawniku. " * 120)
    paras = novelty.paragraphs(text)
    assert paras[0].startswith("Naglowek\nKrotka linia.") and all(len(p) <= 1600 for p in paras)
    assert novelty.para_hash("Krwawnik  ROSNIE.") == novelty.para_hash("krwawnik rosnie")


def test_second_read_of_same_text_needs_no_llm():
    text = "Krwawnik rosnie na lakach i miedzach. " * 8 + "\n\n" + "Kwiaty barwia welne na zolto. " * 8
    first = novelty.select_new(text, set())
    assert first.why_skip is None and first.text and len(first.done) == first.total == 2
    again = novelty.select_new(text, set(first.done))
    assert again.why_skip == "bez zmian od ostatniego odczytu" and again.text == ""
    grown = novelty.select_new(text + "\n\n" + "Nowy akapit: z lisci robiono oklady na rany. " * 6, set(first.done))
    assert grown.why_skip is None and grown.text.startswith("Nowy akapit") and grown.seen == 2


def test_general_page_sends_only_paragraphs_about_the_plant():
    text = f"{LONG}\n\nKrwawnik pospolity stosowano na rany. {LONG}\n\nCiag dalszy o okladach.\n\n{LONG}"
    nv = novelty.select_new(text, set(), ["krwawnik"])
    assert "Krwawnik" in nv.text and nv.off_topic == 1 and len(nv.done) == nv.total
    nothing = novelty.select_new(LONG + "\n\n" + LONG, set(), ["krwawnik"])
    assert nothing.why_skip == "nowe akapity nie dotycza tej rosliny"


def test_small_change_waits_for_more_text():
    base = "Krwawnik rosnie na lakach i miedzach. " * 8
    first = novelty.select_new(base, set())
    small = novelty.select_new(base + "\n\nKrotki dopisek o krwawniku, ktory ma ponad szescdziesiat znakow.",
                               set(first.done), min_chars=200)
    assert small.why_skip == "za malo nowego tekstu" and len(small.done) == 1     # dopisek nie jest zapamietany


# ---------------------------------------------------------------- przebieg z baza
def _setup(tmp_path, monkeypatch):
    from app.config import settings
    plants = tmp_path / "plants"
    plants.mkdir()
    (plants / "krwawnik_pospolity.json").write_text(json.dumps(
        {"id": "krwawnik_pospolity", "nazwa_pl": "Krwawnik pospolity", "nazwa_lat": "Achillea millefolium",
         "opis": "Ręczny opis."}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "backups_dir", tmp_path / "backups")
    monkeypatch.setattr(settings, "knowledge_web_pages", 0)
    monkeypatch.setattr(settings, "knowledge_encyclopedias", False)      # bez sieci w testach (szybko)
    monkeypatch.setattr(settings, "knowledge_auto_apply", False)
    eng = create_engine(f"sqlite:///{(tmp_path / 'k.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        sync_kwiatownik(s, plants)
    return eng


def _research(eng, **kw):
    with Session(eng) as s:
        job = Job(kind="plant_research")
        s.add(job)
        s.commit()
        jid = job.id

    async def run():
        wiki = WikiClient("T", client=httpx.AsyncClient(transport=httpx.MockTransport(wiki_handler)), delay_s=0)
        try:
            return await kn.run_plant_research(jid, jobs_mod.make_runner(1), LLM(), ["krwawnik_pospolity"], wiki=wiki,
                                               **kw)
        finally:
            await wiki.aclose()
    stats = asyncio.run(run())
    with Session(eng) as s:
        log = s.get(Job, jid).log or ""
    return stats, log


def test_research_twice_reads_the_article_only_once(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "knowledge_wiki_refresh_days", 0)     # wszystkie Wikipedie przeczytane -> odswiezaj
    LLM.calls = []
    first, _ = _research(eng)
    extract_calls = sum(1 for c in LLM.calls if "redaktorem" in c)
    assert first["facts"] >= 1 and extract_calls >= 1
    with Session(eng) as s:
        assert s.exec(select(SourceRead)).first().reads == 1
    LLM.calls = []
    second, log = _research(eng)
    assert not any("redaktorem" in c for c in LLM.calls)              # artykul bez zmian - bez ekstrakcji LLM
    assert "pominiete bez LLM (bez zmian od ostatniego odczytu)" in log and second["skipped_sources"] >= 1


def test_low_yield_plant_pauses_for_agents_but_not_for_manual_research(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    assert kn.note_yield("krwawnik_pospolity", 0, now) is None          # 1. slabe badanie - jeszcze bez przerwy
    until = kn.note_yield("krwawnik_pospolity", 1, now)                  # 2. z rzedu -> 7 dni
    assert until is not None and abs((until - now) - timedelta(days=7)) < timedelta(seconds=5)
    assert kn.note_yield("krwawnik_pospolity", 0, now) - now > timedelta(days=13)     # 3. -> 14 dni
    _, log = _research(eng, respect_pause=True)
    assert "ostatnie badania bez nowych informacji" in log
    assert kn.plants_to_research(5) == []
    _, log = _research(eng)                                              # reczne zlecenie - bada mimo przerwy
    assert "ostatnie badania bez nowych informacji" not in log
    assert kn.note_yield("krwawnik_pospolity", 3, now) is None           # wynik kasuje przerwe
    with Session(eng) as s:
        assert s.get(Plant, "krwawnik_pospolity").low_yield_runs == 0


def test_gap_without_answers_closes_and_useless_encyclopedia_is_skipped(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    with Session(eng) as s:
        for i in range(5):
            s.add(PlantQuery(plant_id="krwawnik_pospolity", slot="zastosowanie.kosmetyczne", language="pl",
                             query=f"q{i}", results=0))
        s.add(PlantQuery(plant_id="krwawnik_pospolity", slot="ostrzezenia", language="pl", query="a", results=3))
        s.add(PlantQuery(plant_id="krwawnik_pospolity", slot="ostrzezenia", language="pl", query="b", results=3,
                         facts=2))
        for i in range(10):
            s.add(PlantQuery(plant_id="krwawnik_pospolity", slot="enc:mpdb", language="ja", query=f"e{i}", results=0))
            s.add(PlantQuery(plant_id="krwawnik_pospolity", slot="enc:pfaf", language="en", query=f"p{i}", results=1,
                             facts=1 if i == 3 else 0))
        s.add(PlantQuery(plant_id="krwawnik_pospolity", slot="zastosowanie.rzemieslnicze", language="pl", query="old",
                         results=0, created_at=datetime.now(timezone.utc) - timedelta(days=90)))
        s.commit()
    assert kn.closed_gaps("krwawnik_pospolity") == {"zastosowanie.kosmetyczne"}
    assert kn.useless_encyclopedias() == {"mpdb"}


def test_second_research_moves_on_to_unread_wikipedias(tmp_path, monkeypatch):
    """Rotacja Wikipedii: drugie badanie nie czyta znow tych samych wersji jezykowych (sa tylko pl i en - obie
    przeczytane, a odswiezanie dopiero po KNOWLEDGE_WIKI_REFRESH_DAYS)."""
    eng = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "knowledge_wiki_langs", 1)
    monkeypatch.setattr(settings, "knowledge_wiki_refresh_days", 30)
    _, log1 = _research(eng)
    assert "nowe: pl" in log1
    _, log2 = _research(eng)
    assert "nowe: en" in log2
    _, log3 = _research(eng)
    assert "nowe: -" in log3 and "przeczytane 2 z 2" in log3   # en: 404 w tescie - tez odnotowana, nie wraca
    assert set(kn.wiki_read_at("krwawnik_pospolity")) == {"pl", "en"}
