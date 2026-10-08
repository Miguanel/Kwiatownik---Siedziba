"""Dzial marketingu: posty Lesnego Dziadygi (pora roku, tematy, kontrola kodem, kolejnosc, panel /posts)."""
import asyncio
import json
from datetime import date

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select

import app.marketing.service as service
import app.web.marketing_routes as routes_mod
from app.config import settings
from app.marketing import picker, topics
from app.marketing.persona import Persona
from app.marketing.season import season_for
from app.marketing.writer import Material, compose, parse, word_count
from app.models import FbPost, Item, ItemKind, ItemStatus, PlantFact

BEZ = {"id": "bez_czarny", "nazwa_pl": "Bez czarny", "nazwa_lat": "Sambucus nigra",
       "ciekawostki": ["Z pędów bzu robiono fujarki."],
       "ostrzezenia": "Surowe owoce są toksyczne!",
       "przepisy_medyczne": [{"tytul": "Syrop z owoców bzu"}],
       "kalendarz_ogrodnika": {"zadania": [{"czynnosc": "Zbiór dojrzałych owoców", "miesiace": [9, 10],
                                            "opis": "Ścinamy całe grona."}]}}
LIPA = {"id": "lipa", "nazwa_pl": "Lipa drobnolistna", "nazwa_lat": "Tilia cordata",
        "kalendarz_ogrodnika": {"zadania": [{"czynnosc": "Zbiór kwiatostanów", "miesiace": [6, 7]}]}}
TARNINA = {"id": "tarnina", "nazwa_pl": "Śliwa tarnina", "nazwa_lat": "Prunus spinosa",
           "kalendarz_ogrodnika": {"zadania": [{"czynnosc": "Zbiór owoców po przymrozku", "miesiace": [10, 11]}]}}


def _words(n: int, extra: str = "") -> str:
    return (extra + " " + " ".join(["godom"] * n)).strip()


def test_season_october():
    s = season_for(date(2026, 10, 6))
    assert (s.month_name, s.season, s.phase) == ("październik", "jesień", "początek")
    assert any("babie lato" in f for f in s.folk)
    assert season_for(date(2026, 8, 12)).folk and "Zielnej" in season_for(date(2026, 8, 12)).folk[0]
    assert "kwiecie" not in season_for(date(2026, 12, 2)).label


def test_picker_by_month():
    plants = {"bez_czarny": BEZ, "lipa": LIPA, "tarnina": TARNINA}
    s = season_for(date(2026, 10, 6))
    ids = [p.id for p in picker.candidates(plants, s, topics.SITE_URL, seed=1)]
    assert set(ids) == {"bez_czarny", "tarnina"}          # lipa nie ma nic do roboty w pazdzierniku
    chosen = picker.choose(plants, s, topics.SITE_URL, recent={"bez_czarny", "tarnina"} - {"tarnina"}, seed=1)
    assert chosen[0].id == "tarnina"                       # ostatnio uzyta roslina spada na koniec
    forced = picker.choose(plants, s, topics.SITE_URL, plant_id="lipa", seed=1)
    assert forced[0].id == "lipa" and forced[0].url.endswith("/plant/lipa/")


def test_parse_and_compose():
    persona = Persona()
    m = Material(kind="pora_roku", label="bez", task="t", data="d", main_name="Bez czarny",
                 main_latin="Sambucus nigra", link_url="https://kwiatownik.onrender.com/plant/bez_czarny/",
                 sources=["https://example.de/holunder"])
    text = json.dumps({"tresc": _words(140, "Ady bez czarny dojrzał, panie dzieju. www.zmyslony.pl")
                       + "\n#bez #jesien\nLeśny Dziadyga", "hashtagi": ["#bez", "jesien", "#zielarstwo"]})
    d = parse(text, persona, m)
    assert not d.problems, d.problems
    assert "www." not in d.body and "Leśny Dziadyga" not in d.body and "#bez" not in d.body
    out = compose(d, persona, m)
    assert out.count("#Kwiatownik") == 1 and "#jesien" in out and out.count("#zielarstwo") == 1
    assert persona.signature in out and m.link_url in out and "Źródło: https://example.de/holunder" in out
    short = parse(json.dumps({"tresc": "Krótko o lipie."}), persona, m)
    assert any("za krótki" in p for p in short.problems) and any("nazwa" in p for p in short.problems)
    assert word_count("Ady, panie dzieju – ino tyz!") == 5


class FakeLLM:
    def __init__(self):
        self.prompts = []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None):
        self.prompts.append((system, prompt))
        name = "Siedziba, bez czarny, śliwa tarnina i lipa"

        class R:
            provider, model = "fake", "m"
        r = R()
        r.text = json.dumps({"tresc": _words(150, f"Juzci, {name or 'ziele'} to rzecz."), "hashtagi": ["#jesien"]})
        return r


def _setup(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(service, "engine", eng)
    monkeypatch.setattr(routes_mod, "engine", eng)
    pdir = tmp_path / "plants"
    pdir.mkdir()
    for d in (BEZ, LIPA, TARNINA):
        (pdir / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", pdir)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with Session(eng) as s:
        s.add(PlantFact(plant_id="bez_czarny", section="kultura", language="de", status="applied",
                        text="W Niemczech bez nazywano drzewem pani Holle.", source_url="https://example.de/holunder",
                        source_name="example.de", fingerprint="f1"))
        s.add(PlantFact(plant_id="bez_czarny", section="opis", language="de", status="applied",
                        text="Krzew do 10 m.", source_url="https://example.de/holunder", fingerprint="f2"))
        s.add(PlantFact(plant_id="bez_czarny", section="kultura", language="pl", status="applied",
                        text="Polska informacja.", source_url="https://pl.wikipedia.org/x", fingerprint="f3"))
        s.add(Item(url="https://komoratravnyka.com.ua/chai/", kind=ItemKind.recipe, status=ItemStatus.exported,
                   language="uk", title="Karpacka herbatka",
                   data_json=json.dumps({"tytul": "Karpacki napar na sen", "typ": "medyczne_napar",
                                         "pochodzenie": ["Tradycja ukraińska"], "skladniki": [{"nazwa": "chmiel"}]},
                                        ensure_ascii=False)))
        s.add(Item(url="archiwum://kwiatownik1/x/1", kind=ItemKind.recipe, status=ItemStatus.exported,
                   data_json=json.dumps({"tytul": "Z archiwum"})))
        s.commit()
    return eng


def test_auto_order_and_topics(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    llm = FakeLLM()
    day = date(2026, 10, 6)
    run = lambda kind="auto", **kw: asyncio.run(service.generate_post(llm, kind, day=day, **kw))  # noqa: E731
    ids = [run() for _ in range(6)]
    with Session(eng) as s:
        posts = [s.get(FbPost, i) for i in ids]
    kinds = [p.kind for p in posts]
    assert kinds[:2] == ["siedziba", "siedziba"]                  # najpierw 2 posty o Siedzibie
    assert set(kinds[2:5]) == {"ciekawostka", "przepis", "pora_roku"}
    assert "PIERWSZY post" in llm.prompts[0][1] and "PIERWSZY" not in llm.prompts[1][1]
    assert "Leśny Dziadyga" in llm.prompts[0][0] and "godk" in llm.prompts[0][0].lower()
    fact = next(p for p in posts if p.kind == "ciekawostka")
    assert fact.ref == "fact:1" and "Źródło: https://example.de/holunder" in fact.text
    assert "niemieckiej" in next(pr for _, pr in llm.prompts if "pani Holle" in pr)
    rec = next(p for p in posts if p.kind == "przepis")
    assert rec.ref.startswith("item:") and "komoratravnyka.com.ua" in rec.text and "archiwum" not in rec.text
    assert all(p.text and "Wasz Leśny Dziadyga" in p.text for p in posts)
    # szosty post: ciekawostki i przepisy sie skonczyly -> pora roku
    assert kinds[5] == "pora_roku" and not posts[5].problems


def test_posts_page(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from app.main import app
    asyncio.run(service.generate_post(FakeLLM(), "siedziba", day=date(2026, 10, 6)))
    with TestClient(app) as client:
        r = client.get("/posts")
        assert r.status_code == 200 and "Lesny Dziadyga" in r.text or "Leśny Dziadyga" in r.text
        assert "Siedziba Kwiatownika" in r.text and "Kopiuj" in r.text
        assert client.get("/posts/list").status_code == 200
        # przycisk "Napisz": zadanie musi ruszyc (trasa async - petla zdarzen), bez LLM konczy sie bledem
        r = client.post("/posts/generate", data={"kind": "auto", "plant_id": "bez_czarny", "hint": "x"},
                        follow_redirects=False)
        assert r.status_code == 303
        import time
        from app.models import Job
        for _ in range(50):
            from app.db import engine as main_engine
            with Session(main_engine) as s2:
                job = s2.exec(select(Job).where(Job.kind == "fb_post").order_by(Job.id.desc())).first()
            if job and job.status.value not in ("queued", "running"):
                break
            time.sleep(0.1)
        assert job.status.value == "failed" and "Brak modeli LLM" in job.log
        assert json.loads(job.params_json)["kind"] == "pora_roku"
        r = client.post("/posts/1/save", data={"text": "Nowy tekst", "status": "approved"}, follow_redirects=False)
        assert r.status_code == 303
        r = client.post("/posts/persona", data={"name": "Leśny Dziadyga", "min_words": "100", "max_words": "50"},
                        follow_redirects=False)
        assert r.status_code == 303 and Persona.load(service.persona_path()).max_words == 100
    with Session(service.engine) as s:
        p = s.get(FbPost, 1)
        assert (p.text, p.status, p.edited) == ("Nowy tekst", "approved", True)
