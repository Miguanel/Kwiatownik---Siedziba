"""Magia roslin: tezaurus, generator zapytan z tropami, kontrola odpowiedzi LLM, rozpoznawanie roslin, rozdzial
"Magia i zabobony" w rozmieszczeniu, zadanie plant_magic na atrapach wyszukiwarki/stron/LLM i panel /magia."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select

from app.knowledge import catalog, magic as mg, placement, plantfile
from app.knowledge.magic_lexicon import (CONCEPTS, DEEP_SOURCES, LANGS, RODZAJE, SWEEPS, magic_score)
from app.models import MagicLead, Plant, PlantFact, PlantQuery

SRC = ("In der Johannisnacht hängte man Johanniskraut über die Stalltür, um Hexen und böse Geister abzuwehren. "
       "Mädchen warfen Kränze aus Johanniskraut, um ihren künftigen Bräutigam zu erkennen. Man nahm 20 g Kraut. "
       "Heute gilt es in der Wicca als Kraut der Sonne.")
NAMES = {"pl": "Dziurawiec zwyczajny", "de": "Echtes Johanniskraut", "ru": "Зверобой продырявленный",
         "la": "Hypericum perforatum"}


def test_lexicon_is_consistent():
    assert {c.rodzaj for c in CONCEPTS} <= set(RODZAJE)
    assert {r for _, _, r in SWEEPS} <= set(RODZAJE)
    for title, icon, desc, words in RODZAJE.values():
        assert icon in catalog.ICONS and title and desc and words
    for c in CONCEPTS:
        assert set(c.langs) <= set(LANGS) and all(lang in c.terms for lang in c.langs), c.key
    assert len({d.domain for d in DEEP_SOURCES}) == len(DEEP_SOURCES)
    assert magic_score(SRC.lower()) >= 2 and magic_score("Johanniskraut wächst auf Wiesen.") == 0


def test_parse_facts_checks_quotes_doses_modern_and_kinds():
    raw = json.dumps({"fakty": [
        {"rodzaj": "ochrona", "tekst": "W Niemczech w noc świętojańską wieszano dziurawiec nad drzwiami obory.",
         "cytat": "hängte man Johanniskraut über die Stalltür", "kraj": "Niemcy", "epoka": "XIX w.", "toksyczne": False},
        {"rodzaj": "magia miłosna", "tekst": "Dziewczęta rzucały wianki z dziurawca, by poznać przyszłego męża.",
         "cytat": "Mädchen warfen Kränze aus Johanniskraut", "kraj": "null"},
        {"rodzaj": "lecznicza", "tekst": "Do zabiegu brano 20 g ziela dziurawca zebranego o świcie.",
         "cytat": "Man nahm 20 g Kraut."},
        {"rodzaj": "ochrona", "tekst": "W tradycji wicca dziurawiec jest uważany za ziele słońca.",
         "cytat": "Heute gilt es in der Wicca als Kraut der Sonne.", "epoka": "współczesność"},
        {"rodzaj": "czarna", "tekst": "Dziurawiec służył do rzucania klątw na sąsiadów w Bawarii.",
         "cytat": "zitat das es nicht gibt im text"},
        {"rodzaj": "cos innego", "tekst": "Wierzono, że dziurawiec zebrany w Johannisnacht odpędza złe duchy.",
         "cytat": "um Hexen und böse Geister abzuwehren", "epoka": "średniowiecze"}],
        "tropy": [{"termin": "Johannisnacht", "jezyk": "de", "typ": "praktyka", "opis": "obrzed"},
                  {"termin": "Johannisnacht", "jezyk": "de", "typ": "praktyka"},
                  {"termin": "Іванове зілля", "jezyk": "uk", "typ": "nazwa"},
                  {"termin": "x", "jezyk": "de"}, {"termin": "Hexenkraut", "jezyk": "deutsch"},
                  {"termin": "вінок на Купала", "jezyk": "uk", "typ": "cos"}]})
    facts, leads, rejected = mg.parse_facts(raw, SRC)
    assert [f.meta["rodzaj"] for f in facts] == ["ochrona", "milosna", "zabobony"]
    assert facts[0].meta == {"rodzaj": "ochrona", "kraj": "Niemcy", "epoka": "XIX w."} and facts[0].section == "magia"
    assert "kraj" not in facts[1].meta
    assert any("dawki" in r for r in rejected) and any("wspolczesna" in r for r in rejected)
    assert any("cytatu" in r for r in rejected)
    assert [(ld["termin"], ld["typ"]) for ld in leads] == [("Johannisnacht", "praktyka"), ("Іванове зілля", "nazwa"),
                                                           ("вінок на Купала", "praktyka")]
    facts, _, _ = mg.parse_facts(raw, SRC, include_modern=True)
    assert len(facts) == 4


def test_sweep_needs_plant_and_index_finds_it():
    raw = json.dumps({"fakty": [
        {"roslina": "Johanniskraut", "lacina": "Hypericum perforatum L.", "rodzaj": "ochrona",
         "tekst": "Dziurawiec wieszano nad drzwiami obory, aby odpędzić czarownice.",
         "cytat": "hängte man Johanniskraut über die Stalltür"},
        {"rodzaj": "ochrona", "tekst": "Wieszano rośliny nad drzwiami obory przeciw czarownicom.",
         "cytat": "hängte man Johanniskraut über die Stalltür"}]})
    facts, _, rejected = mg.parse_facts(raw, SRC, sweep=True)
    assert len(facts) == 1 and facts[0].latin.startswith("Hypericum") and any("bez nazwy" in r for r in rejected)
    idx = mg.PlantIndex([("dziurawiec", "Dziurawiec zwyczajny", "Hypericum perforatum", {"de": "Echtes Johanniskraut"}),
                         ("bylica", "Bylica pospolita", "Artemisia vulgaris", {})])
    assert idx.find(facts[0].plant, facts[0].latin) == "dziurawiec"
    assert idx.find("Echtes Johanniskraut", None) == "dziurawiec"
    assert idx.find("bylicy pospolitej", None) is None and idx.find("Bylica pospolita", None) == "bylica"
    assert idx.find("Nieznane ziele", "Abies alba") is None


def test_queries_rotate_concepts_follow_leads_and_reach_archives():
    leads = [SimpleNamespace(id=7, term="Hexenkraut", language="de", kind="nazwa", rodzaj="czarownice", depth=1),
             SimpleNamespace(id=8, term="купальские травы", language="ru", kind="praktyka", rodzaj="obrzedy", depth=2)]
    done = {"magia:czarownice:de"}
    qs = mg.plant_queries(NAMES, list(LANGS), done, leads, {"uk": ["Іванове зілля"]}, limit=9)
    assert len(qs) == 9 and len({q.query for q in qs}) == 9
    assert qs[0].kind == "trop" and qs[0].query.startswith('"Hexenkraut"') and qs[0].lead_id == 7
    assert any(q.kind == "trop" and q.query == '"купальские травы" "Зверобой продырявленный"' for q in qs)
    czar = next(q for q in qs if q.key.startswith("magia:czarownice:"))
    assert czar.key == "magia:czarownice:pl" and czar.query.startswith('"Dziurawiec zwyczajny"')
    assert any(q.key == "magia:ochrona:uk" and "Іванове зілля" in q.query for q in qs)   # nazwa ludowa z tropu
    arch = [q for q in qs if q.kind == "archiwum"]
    assert arch and all(q.query.startswith("site:") for q in arch)
    assert any("site:thelatinlibrary.com" in q.query and '"Hypericum"' in q.query
               for q in mg.archive_queries(mg.name_variants(NAMES), NAMES["la"], set(), set()))
    # kolejne przebiegi: kazde pojecie w kolejnym jezyku, az wszystko bedzie zadane niedawno -> nic
    done, seen = set(), []
    for _ in range(80):
        qs = mg.plant_queries(NAMES, list(LANGS), done, [], limit=50)
        if not qs:
            break
        assert len({q.key.split(":")[1] for q in qs if q.kind == "koncept"}) == sum(q.kind == "koncept" for q in qs)
        done |= {q.key for q in qs}
        seen += qs
    assert mg.plant_queries(NAMES, list(LANGS), done, [], limit=50) == []
    assert {q.lang for q in seen if q.kind == "koncept"} == {"pl", "de", "ru", "en", "la"}


def test_focus_text_and_relevance():
    text = ("Lorem ipsum dolor sit amet. " * 600) + " Dziurawiec chronił przed czarami i urokami. " + ("x " * 3000)
    out = mg.focus_text(text, ["dziurawiec"], window=400, limit=2000)
    assert "Dziurawiec chronił" in out and len(out) <= 2100
    assert mg.page_relevant("Dziurawiec chronił przed czarami i urokami.", ["dziurawiec"])[0]
    assert not mg.page_relevant("Dziurawiec rośnie na łąkach.", ["dziurawiec"])[0]
    assert not mg.page_relevant("Bylica chroniła przed czarami i urokami.", ["dziurawiec"])[0]


def _plant_with_magic():
    fakty = [{"id": "s1", "sekcja": "magia", "czesc": None, "tekst": "Dziewczęta rzucały wianki z dziurawca na wodę.",
              "jezyk": "de", "zrodlo": {"nazwa": "zeno.org", "url": "https://zeno.org/a"},
              "magia": {"rodzaj": "milosna", "kraj": "Niemcy", "epoka": "XIX w."}},
             {"id": "s2", "sekcja": "magia", "czesc": None,
              "tekst": "Dziurawiec wieszany nad drzwiami chronił dom przed piorunem i czarownicami.",
              "jezyk": "de", "zrodlo": {"nazwa": "zeno.org", "url": "https://zeno.org/a"},
              "magia": {"rodzaj": "ochrona", "toksyczne": False}}]
    return {"id": "dziurawiec", "nazwa_pl": "Dziurawiec", "opis": "Bylina.",
            "wiedza": {"wersja": 2, "zrodla": [{"nr": 1, "nazwa": "zeno.org", "url": "https://zeno.org/a"}],
                       "fakty": fakty, "sekcje": [{"klucz": "magia", "tytul": "Magia i zabobony", "punkty": [
                           {"tekst": f["tekst"], "czesc": None, "zrodla": [1], "fakty": [f["id"]]} for f in fakty]}]}}


@pytest.fixture
def no_catalog():
    catalog.path().unlink(missing_ok=True)
    yield
    catalog.path().unlink(missing_ok=True)


def test_placement_puts_magic_into_own_chapter_by_kind(no_catalog):
    data = _plant_with_magic()
    ids = [c["id"] for c in catalog.chapters()]
    assert ids.index("magia") == ids.index("tajemna") + 1
    block = asyncio.run(placement.build_placement(None, data, "Dziurawiec", today="2026-10-21"))
    by = {w["miejsce"]: w for w in block["wstawki"]}
    assert set(by) == {"magia/milosna", "magia/ochrona"}
    w = by["magia/milosna"]
    assert w["rozdzial"] == "Magia i zabobony" and w["rozdzial_ikona"] == "ra-cauldron" and w["rozdzial_po"] == "tajemna"
    assert w["tytul"] == "Magia miłosna" and w["nowy"] is True
    cand = placement.with_placement(data, block)
    assert placement.validate_placement(cand) == []


def test_fact_entry_carries_magic_meta_and_file_validates():
    f = SimpleNamespace(id=5, section="magia", part=None, text="Dziurawiec chronił dom przed czarownicami w Bawarii.",
                        language="de", source_name="zeno.org", source_url="https://zeno.org/a", quote="hängte man",
                        meta_json=json.dumps({"rodzaj": "ochrona", "kraj": "Niemcy", "epoka": "<b>XIX</b>",
                                              "toksyczne": True, "inne": "x"}))
    e = plantfile.fact_entry(f)
    assert e["magia"] == {"rodzaj": "ochrona", "kraj": "Niemcy", "toksyczne": True}
    cand = plantfile.build_candidate({"id": "dziurawiec", "nazwa_pl": "Dziurawiec"}, {}, [f], today="2026-10-21")
    assert not [e for e in plantfile.validate_candidate({"id": "dziurawiec", "nazwa_pl": "Dziurawiec"}, cand, "dziurawiec")
                if "s5" in e]


def test_merge_never_weaves_magic_into_chapters():
    from app.knowledge import merge

    class Spy:
        prompts = []

        async def complete(self, prompt, **k):
            Spy.prompts.append(prompt)
            return SimpleNamespace(text='{"przydzial": []}', provider="x", model="y")
    data = _plant_with_magic()
    pts = [{"pid": 1, "sekcja": "magia", "czesc": None, "tekst": "Wianki z dziurawca puszczano na wodę."},
           {"pid": 2, "sekcja": "historia", "czesc": None, "tekst": "Nazwa pochodzi od dziurawych liści."}]
    routes = asyncio.run(merge.route(Spy(), data, "Dziurawiec", pts))
    assert routes[1] is None and not any("Wianki" in p for p in Spy.prompts)


def test_agents_know_magic():
    from app.agents.planner import GROUPS
    from app.agents.tasks import EXPERT_KINDS, KINDS, PLANT_KINDS
    assert KINDS["magia"].job == "plant_magic" and "magia" in PLANT_KINDS and "magia" in EXPERT_KINDS
    assert GROUPS["plants_magic"][0] == "magia"
    from app.agents.watchdog import LLM_JOBS
    assert "plant_magic" in LLM_JOBS


# ------------------------------------------------------------------ zadanie na atrapach
HTML = ("<html lang='de'><body><article><h1>Johanniskraut im Volksglauben</h1><p>" + SRC + " " +
        "Das Echte Johanniskraut (Hypericum perforatum) galt als Hexenkraut und Zauberpflanze. " * 6 +
        "</p></article></body></html>")


class FakeFetcher:
    def __init__(self):
        self.urls = []

    async def get(self, url, html_only=True):
        from app.scrapers.fetcher import FetchResult
        self.urls.append(url)
        return FetchResult(url, 200, text=HTML, content_type="text/html")

    async def aclose(self):
        pass


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, prompt, system=None, **k):
        self.calls += 1
        n = self.calls
        text = ["Wierzono, że dziurawiec zawieszony u wejścia obory odpędza czarownice w Niemczech.",
                "Bawarscy chłopi w noc świętojańską zatykali gałązki ziela za belką stajni.",
                "Według niemieckich zapisów ludowych ziele chroniło bydło przed złymi mocami.",
                "Świętojańskie wiązki tego ziela wisiały nad progiem gospodarstw w Turyngii.",
                "Dawniej uważano, że roślina zerwana przed świtem ma największą siłę ochronną.",
                "Gospodynie wplatały pędy w wieńce chroniące spiżarnie przed urokami sąsiadek."][(n - 1) % 6]
        ans = {"fakty": [{"rodzaj": "ochrona", "tekst": text,
                          "cytat": "hängte man Johanniskraut über die Stalltür", "kraj": "Niemcy", "epoka": "XIX w."}],
               "tropy": [{"termin": f"Hexenkraut{n}", "jezyk": "de", "typ": "nazwa", "opis": "nazwa ludowa"},
                         {"termin": f"Kupala{n}", "jezyk": "uk", "typ": "analogia"}]}
        return SimpleNamespace(text=json.dumps(ans, ensure_ascii=False), provider="fake", model="m")


@pytest.fixture
def eng(tmp_path, monkeypatch):
    e = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(e)
    for name in ("app.worker.magic", "app.worker.knowledge", "app.worker.jobs"):
        monkeypatch.setattr(importlib.import_module(name), "engine", e)
    with Session(e) as s:
        s.add(Plant(id="dziurawiec", nazwa_pl="Dziurawiec zwyczajny", nazwa_lat="Hypericum perforatum",
                    names_json=json.dumps({"de": "Echtes Johanniskraut"}, ensure_ascii=False)))
        s.commit()
    from app.discovery import search
    from app.discovery.search import SearchResult

    async def fake_search(backends, query, language, country, pages=1, log_fn=None):
        return [SearchResult(f"https://example.{language}/{abs(hash(query)) % 10000}", "t")]
    monkeypatch.setattr(search, "search_all", fake_search)
    return e


def test_research_stores_facts_with_meta_leads_and_follows_them(eng, monkeypatch):
    from app.config import settings
    from app.worker import magic as wm
    monkeypatch.setattr(settings, "magic_pages", 2)
    monkeypatch.setattr(settings, "magic_queries", 4)
    llm, fetcher, log = FakeLLM(), FakeFetcher(), []
    st = asyncio.run(wm.research_plant_magic("dziurawiec", llm, fetcher, ["b"], log.append))
    assert st["strony"] == 2 and st["fakty"] == 2 and st["tropy"] == 4
    with Session(eng) as s:
        facts = s.exec(select(PlantFact)).all()
        leads = s.exec(select(MagicLead)).all()
        qs = s.exec(select(PlantQuery)).all()
    assert all(f.section == "magia" and json.loads(f.meta_json)["kraj"] == "Niemcy" for f in facts)
    assert {ld.depth for ld in leads} == {1} and {ld.plant_id for ld in leads} == {"dziurawiec"}
    assert all(q.slot.startswith("magia:") for q in qs) and sum(q.facts for q in qs) == 2
    # drugi przebieg: najpierw tropy; trop "nazwa" staje sie tez wariantem nazwy rosliny
    assert "hexenkraut1" in [n.lower() for v in wm.folk_names("dziurawiec").values() for n in v]
    st2 = asyncio.run(wm.research_plant_magic("dziurawiec", llm, fetcher, ["b"], log.append))
    with Session(eng) as s:
        leads = s.exec(select(MagicLead)).all()
    done = [ld for ld in leads if ld.status == "sprawdzony"]
    assert st2["strony"] == 2 and done and all(ld.facts >= 1 for ld in done)
    assert any(ld.depth == 2 and ld.parent_id in {d.id for d in done} for ld in leads)
    assert wm.plants_for_magic(5) == ["dziurawiec"]


def test_sweep_maps_facts_to_plants(eng, monkeypatch):
    from app.worker import magic as wm

    class SweepLLM(FakeLLM):
        async def complete(self, prompt, system=None, **k):
            assert system == mg.SWEEP_SYSTEM
            ans = {"fakty": [{"roslina": "Johanniskraut", "lacina": "Hypericum perforatum", "rodzaj": "ochrona",
                              "tekst": "W Niemczech wieszano dziurawiec nad drzwiami obory przeciw czarownicom.",
                              "cytat": "hängte man Johanniskraut über die Stalltür"},
                             {"roslina": "Eisenkraut", "lacina": "Verbena officinalis", "rodzaj": "ochrona",
                              "tekst": "Werbena chroniła przed czarownicami według wierzeń niemieckich.",
                              "cytat": "um Hexen und böse Geister abzuwehren"}],
                   "tropy": [{"termin": "Kräuterbuschen", "jezyk": "de", "typ": "praktyka"}]}
            return SimpleNamespace(text=json.dumps(ans, ensure_ascii=False), provider="fake", model="m")
    st, touched = asyncio.run(wm.sweep(SweepLLM(), FakeFetcher(), ["b"], lambda m: None, limit=1))
    assert touched == {"dziurawiec"} and st["spoza"] >= 1 and st["fakty"] == 1
    with Session(eng) as s:
        general = s.exec(select(MagicLead).where(MagicLead.plant_id.is_(None))).all()
    assert [g.term for g in general] == ["Kräuterbuschen"]
    assert wm.general_lead_queries(set(), 2)[0].query.startswith('"Kräuterbuschen"')


def test_magic_panel_renders():
    from app.main import app
    with TestClient(app) as c:
        html = c.get("/magia").text
    assert "Magia roślin" in html and "Tropy (azymuty)" in html and 'href="/magia"' in html
