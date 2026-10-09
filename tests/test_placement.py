"""Rozmieszczanie wiedzy z sieci w rozdzialach strony (placement.py) i Uklad strony (catalog.py)."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.knowledge import catalog, placement


def plant(points, czesci=None, scalone=None):
    fakty, sekcje = [], {}
    for i, (sek, czesc, tekst) in enumerate(points, 1):
        fakty.append({"id": f"s{i}", "sekcja": sek, "czesc": czesc, "tekst": tekst, "jezyk": "pl",
                      "zrodlo": {"nazwa": "Wikipedia", "url": "https://pl.wikipedia.org/wiki/Bez"},
                      "cytat": "Sambucus nigra fragment zrodla"})
        sekcje.setdefault(sek, []).append({"tekst": tekst, "czesc": czesc, "zrodla": [1], "fakty": [f"s{i}"]})
    data = {"id": "bez", "nazwa_pl": "Bez czarny", "opis": "Krzew.",
            "wiedza": {"wersja": 2, "zrodla": [{"nr": 1, "nazwa": "Wikipedia", "url": "https://pl.wikipedia.org/wiki/Bez"}],
                       "fakty": fakty, "sekcje": [{"klucz": k, "tytul": k, "punkty": v} for k, v in sekcje.items()]}}
    if czesci:
        data["czesci_rosliny"] = czesci
    if scalone:
        data["scalone"] = scalone
    return data


POINTS = [
    ("opis", None, "Roślina często bywa mylona z barszczem syberyjskim o żółtozielonych kwiatach."),
    ("bezpieczenstwo", None, "Na polach uprawnych roślina jest uciążliwym chwastem ograniczającym plony zbóż."),
    ("ciekawostki", "kwiaty", "Nektar wydzielany jest u nasady pręcików tylko w słoneczne dni, co przyciąga owady."),
    ("historia", None, "W starożytnym Egipcie tę roślinę wykorzystywano do leczenia oparzeń."),
    ("nazwy_ludowe", None, "Wśród ludowych nazw bzu wyróżnia się bzowinę, buzinę i hyczkę."),
    ("wystepowanie", None, "Naturalny obszar występowania obejmuje Europę i zachodnią Azję aż po Kaukaz."),
    ("czesci_rosliny", "owoce", "Owoce zbiera się we wrześniu, gdy są całkiem czarne, i suszy w cieniu."),
    ("ciekawostki", None, "Charakterystyczny zapach pędów wykorzystywano dawniej do odstraszania much w oborach."),
]


@pytest.fixture(autouse=True)
def clean_catalog():
    catalog.path().unlink(missing_ok=True)
    yield
    catalog.path().unlink(missing_ok=True)


def test_rules_place_every_point_in_its_chapter():
    data = plant(POINTS, czesci={"owoce": {"nazwa_surowca": "Owoc bzu"}})
    blk = asyncio.run(placement.build_placement(None, data, "Bez czarny"))
    where = {p["fakty"][0]: w["miejsce"] for w in blk["wstawki"] for p in w["punkty"]}
    assert where == {"s1": "rozpoznawanie/pomylki", "s2": "bezpieczenstwo/chwast", "s3": "cykl/biologia",
                     "s4": "tajemna/historia", "s5": "tajemna/nazwy", "s6": "natura/wystepowanie",
                     "s7": "surowce/czesc:owoce", "s8": "zastosowanie/gospodarstwo"}
    owoce = next(w for w in blk["wstawki"] if w["miejsce"] == "surowce/czesc:owoce")
    assert owoce["tytul"] == "Owoc bzu" and owoce["nowy"] is False          # istniejaca czesc pliku
    assert next(w for w in blk["wstawki"] if w["miejsce"] == "tajemna/historia")["nowy"] is True
    order = [w["miejsce"] for w in blk["wstawki"]]
    assert order.index("natura/wystepowanie") < order.index("rozpoznawanie/pomylki") < order.index("tajemna/nazwy")
    assert placement.validate_placement(dict(data, rozmieszczenie=blk)) == []
    assert blk["jak"] == "reguly" and blk["wejscie"]


def test_new_part_gets_own_subchapter_and_merged_points_are_skipped():
    sc = {"wersja": 1, "pola": {"ciekawostki": {"typ": "lista", "oryginal": None,
                                                "tresc": [{"tekst": "W starożytnym Egipcie leczono nią oparzenia.",
                                                           "zrodla": [1], "fakty": ["s1"]}]}},
          "uzyte_fakty": ["s1"]}
    data = plant(POINTS[3:4] + [("czesci_rosliny", "korzeń", "Korzeń zbiera się jesienią w pierwszym roku wzrostu.")],
                 scalone=sc)
    blk = asyncio.run(placement.build_placement(None, data, "Bez"))
    assert [w["miejsce"] for w in blk["wstawki"]] == ["surowce/czesc:korzeń"]  # s1 juz wmontowany w tekst
    assert blk["wstawki"][0]["nowy"] is True and blk["wstawki"][0]["tytul"] == "Korzeń"
    bad = dict(data, rozmieszczenie={"wstawki": [{"miejsce": "tajemna/historia", "tytul": "H", "rozdzial": "W",
                                                   "punkty": [{"tekst": "abc def", "zrodla": [1], "fakty": ["s1"]}]}]})
    assert any("juz scalona" in e for e in placement.validate_placement(bad))


def test_unchanged_points_reuse_block_and_cohesion_keeps_section_together():
    data = plant([("historia", None, f"Dawniej w średniowieczu roślinę stosowano na wiele sposobów nr {i}.") for i in range(3)]
                 + [("historia", None, "Na Węgrzech produkuje się brandy z owoców tej rośliny.")])
    blk = asyncio.run(placement.build_placement(None, data, "Bez"))
    assert {w["miejsce"] for w in blk["wstawki"]} == {"tajemna/historia"}
    data["rozmieszczenie"] = blk
    assert asyncio.run(placement.build_placement(None, data, "Bez")) is blk


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    async def complete(self, prompt, system=None, **kw):
        self.prompt = prompt
        return SimpleNamespace(text=json.dumps(self.reply), provider="test", model="m", seconds=1)


def test_llm_moves_points_and_creates_agent_subchapter(monkeypatch):
    data = plant(POINTS)
    pts = placement.leftover_points(data)
    pid = {p["fakty"][0]: p["pid"] for p in pts}                  # numery punktow w kolejnosci sekcji
    places = placement.places_for(data, pts)
    m_kuchnia = next(f"M{i}" for i, p in enumerate(places, 1) if p.id == "zastosowanie/kuchnia")
    llm = FakeLLM({"miejsca": [{"id": pid["s4"], "miejsce": m_kuchnia},
                               {"id": pid["s8"], "miejsce": "NOWY", "rozdzial": "zastosowanie", "tytul": "Odstraszanie owadów"},
                               {"id": 99, "miejsce": "M1"}, {"id": pid["s2"], "miejsce": "M999"}]})

    async def fake_complete(llm_, prompt, system, prefer, timeout=None):
        return await llm_.complete(prompt, system)
    monkeypatch.setattr("app.knowledge.merge._complete", fake_complete)
    blk = asyncio.run(placement.build_placement(llm, data, "Bez"))
    where = {p["fakty"][0]: w for w in blk["wstawki"] for p in w["punkty"]}
    assert where["s4"]["miejsce"] == "zastosowanie/kuchnia"
    assert where["s8"]["tytul"] == "Odstraszanie owadów" and where["s8"].get("od_agenta")
    assert where["s2"]["miejsce"] == "bezpieczenstwo/chwast"                     # zly numer miejsca -> regula
    assert blk["jak"] == "llm:test/m" and "ROZDZIALY" in llm.prompt
    row = next(r for r in catalog.load()["podrozdzialy"] if r["tytul"] == "Odstraszanie owadów")
    assert row["zrodlo"] == "agent" and row["zatwierdzony"] is False


def test_catalog_custom_chapter_and_subchapter_used_by_algorithm():
    ch = catalog.add_chapter("Obrzędy i rytuały", "ra-candle", po="tajemna")
    sub = catalog.add_subchapter(ch["id"], "Noc Kupały", "ra-crystal-ball", {"kultura": 4}, ["kupal", "sobotk"])
    with pytest.raises(catalog.CatalogError):
        catalog.add_subchapter(ch["id"], "noc kupaly")
    assert [c["id"] for c in catalog.chapters()][-1] == ch["id"]
    data = plant([("kultura", None, "W noc Kupały wplatano kwiaty bzu w wianki na sobótkę.")])
    blk = asyncio.run(placement.build_placement(None, data, "Bez"))
    w = blk["wstawki"][0]
    assert w["miejsce"] == sub["id"] and w["rozdzial"] == "Obrzędy i rytuały"
    assert w["rozdzial_ikona"] == "ra-candle" and w["rozdzial_po"] == "tajemna"
    assert placement.validate_placement(dict(data, rozmieszczenie=blk)) == []
    catalog.toggle_builtin("tajemna/kultura", False)
    assert "tajemna/kultura" not in {p.id for p in placement.places_for(data, [])}
    assert catalog.remove(ch["id"]) and not catalog.load()["podrozdzialy"]


def test_layout_page_and_forms():
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/uklad/rozdzial", data={"tytul": "Weterynaria", "ikona": "ra-wolf-head", "po": "zastosowanie"},
                   follow_redirects=False)
        assert r.status_code == 303
        r = c.post("/uklad/podrozdzial", data={"rozdzial": "weterynaria", "tytul": "Bydło", "sek_zastosowanie_lecznicze": "2",
                                                "slowa": "krow, bydl", "opis": "leczenie bydla"}, follow_redirects=False)
        assert r.status_code == 303 and "Dodano" in r.headers["location"]
        page = c.get("/uklad").text
        assert "Weterynaria" in page and "Bydło" in page and "Wystepowanie i zasieg" not in page
        assert "Występowanie i zasięg" in page and "Nowy rozdzial" in page
