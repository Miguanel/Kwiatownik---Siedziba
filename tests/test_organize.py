"""Agregator wiedzy: czesci rosliny, numerowane zrodla, redakcja LLM tylko po kontroli kodem."""
import asyncio
import json
from types import SimpleNamespace

from app.knowledge import organize, plantfile
from app.knowledge.sections import norm_part

WIKI = "https://pl.wikipedia.org/wiki/Babka_lancetowata"
PAGE = "https://ziola.example.org/babka"


def _f(i, sekcja, tekst, url=WIKI, czesc=None):
    return {"id": f"s{i}", "sekcja": sekcja, "czesc": czesc, "tekst": tekst, "jezyk": "pl",
            "zrodlo": {"nazwa": "Wikipedia (pl)" if "wikipedia" in url else "ziola.example.org", "url": url}}


FACTS = [
    _f(1, "kultura", "Dzieci bawiły się łodygami babki, strącając nimi kwiatostany innych roślin.", PAGE),
    _f(2, "kultura", "W zabawie dziecięcej łodygami babki strącano kwiatostany innych roślin.", WIKI),
    _f(3, "zastosowanie_lecznicze", "Młode liście babki jedzono jako warzywo w sałatkach.", PAGE, "liscie"),
    _f(4, "zastosowanie_lecznicze", "Liście babki przykładano na rany i ukąszenia owadów.", WIKI, "lisci"),
    _f(5, "opis", "Roślina osiąga do 50 cm wysokości i ma lancetowate liście.", WIKI, "cała roślina"),
    _f(6, "zastosowanie_lecznicze", "Napar z liści babki stosowano przy kaszlu i zapaleniu gardła.", PAGE, "Liście"),
]


def test_norm_part_maps_messy_names():
    assert norm_part("lodygi i liscie") == "ziele" and norm_part("calej rosliny") is None
    assert norm_part("Liście") == "liście" and norm_part("mlode lisci") == "liście"
    assert norm_part("pąki kwiatowe") == "pąki" and norm_part("korzenie") == "korzeń" and norm_part("kora") == "kora"
    assert norm_part("toksyczność") is None and norm_part("null") is None and norm_part(None) is None


def test_deterministic_layout_keeps_every_fact():
    lay = organize.deterministic(FACTS)
    assert [z["url"] for z in lay["zrodla"]] == [WIKI, PAGE] and [z["nr"] for z in lay["zrodla"]] == [1, 2]
    assert [s["klucz"] for s in lay["sekcje"]] == ["opis", "zastosowanie_lecznicze", "kultura"]
    ids = [fid for s in lay["sekcje"] for p in s["punkty"] for fid in p["fakty"]]
    assert sorted(ids) == sorted(f["id"] for f in FACTS)
    assert set(lay["czesci"]["liście"]) == {"s3", "s4", "s6"}


def _llm_answer(sekcje):
    return json.dumps({"sekcje": sekcje}, ensure_ascii=False)


GOOD = [
    {"klucz": "opis", "podsumowanie": None,
     "punkty": [{"id": [5], "tekst": "Roślina osiąga do 50 cm wysokości i ma lancetowate liście.", "czesc": None}]},
    {"klucz": "zastosowanie_lecznicze",
     "podsumowanie": "Liście babki przykładano na rany, a napar z liści stosowano przy kaszlu.",
     "punkty": [{"id": [4], "tekst": "Liście babki przykładano na rany i ukąszenia owadów.", "czesc": "liście"},
                {"id": [6], "tekst": "Napar z liści babki stosowano przy kaszlu i zapaleniu gardła.",
                 "czesc": "liście"}]},
    {"klucz": "ciekawostki", "podsumowanie": None,
     "punkty": [{"id": [3], "tekst": "Młode liście babki jedzono jako warzywo w sałatkach.", "czesc": "liście"}]},
    {"klucz": "kultura", "podsumowanie": None,
     "punkty": [{"id": [1, 2], "tekst": "Dzieci bawiły się łodygami babki, strącając nimi kwiatostany innych roślin.",
                 "czesc": None}]},
]


def test_llm_layout_accepted_merges_and_sums_sources():
    lay, errors = organize.apply_llm_layout(_llm_answer(GOOD), FACTS)
    assert errors == [] and lay is not None
    keys = [s["klucz"] for s in lay["sekcje"]]
    assert keys == ["opis", "zastosowanie_lecznicze", "kultura", "ciekawostki"]          # kolejnosc Kwiatownika
    merged = lay["sekcje"][2]["punkty"][0]
    assert merged["fakty"] == ["s1", "s2"] and merged["zrodla"] == [1, 2]
    assert lay["sekcje"][1]["podsumowanie"] is None                     # tylko 2 punkty -> bez podsumowania
    assert organize.validate_layout({**lay, "fakty": FACTS}) == []


def test_llm_layout_rejected_when_facts_lost_invented_or_doubled():
    lost = json.loads(json.dumps(GOOD))
    lost[3]["punkty"][0]["id"] = [1]                                    # s2 zgubione
    assert organize.apply_llm_layout(_llm_answer(lost), FACTS)[0] is None
    invented = json.loads(json.dumps(GOOD))
    invented[0]["punkty"][0]["tekst"] = "Roślina osiąga do 120 cm wysokości i ma lancetowate liście."
    lay, errors = organize.apply_llm_layout(_llm_answer(invented), FACTS)
    assert lay is None and any("odbiega" in e for e in errors)
    drift = json.loads(json.dumps(GOOD))
    drift[2]["punkty"][0]["tekst"] = "Babka jest symbolem szczęścia i chroni domostwa przed piorunami."
    assert organize.apply_llm_layout(_llm_answer(drift), FACTS)[0] is None
    doubled = json.loads(json.dumps(GOOD))
    doubled[2]["punkty"][0]["id"] = [3, 4]
    assert any("wiele razy" in e for e in organize.apply_llm_layout(_llm_answer(doubled), FACTS)[1])
    html = json.loads(json.dumps(GOOD))
    html[0]["punkty"][0]["tekst"] = "<b>Roślina</b> osiąga do 50 cm wysokości i ma lancetowate liście."
    assert organize.apply_llm_layout(_llm_answer(html), FACTS)[0] is None
    assert organize.apply_llm_layout("to nie json", FACTS)[0] is None


class FakeLLM:
    def __init__(self, text=None, fail=False):
        self.text, self.fail = text, fail

    async def complete(self, prompt, **kw):
        if self.fail:
            raise RuntimeError("limit")
        assert "ID 6 |" in prompt
        return SimpleNamespace(text=self.text, provider="fake", model="m1")


def test_organize_falls_back_to_deterministic():
    lay, how = asyncio.run(organize.organize(FakeLLM(_llm_answer(GOOD)), "Babka", "Plantago lanceolata", FACTS))
    assert how == "llm:fake/m1" and len(lay["sekcje"]) == 4
    bad = json.loads(json.dumps(GOOD))
    bad.pop()
    lay, how = asyncio.run(organize.organize(FakeLLM(_llm_answer(bad)), "Babka", None, FACTS))
    assert how.startswith("deterministyczny") and sum(len(s["punkty"]) for s in lay["sekcje"]) == 6
    lay, how = asyncio.run(organize.organize(FakeLLM(fail=True), "Babka", None, FACTS))
    assert how.startswith("deterministyczny")


def test_candidate_with_layout_validated():
    original = {"id": "babka", "nazwa_pl": "Babka", "opis": "Ręczny.", "wiedza": {"fakty": FACTS}}
    cand = plantfile.with_layout(json.loads(json.dumps(original)), organize.deterministic(FACTS), "deterministyczny",
                                 today="2026-09-29")
    w = cand["wiedza"]
    assert w["wersja"] == 2 and w["fakty"][2]["czesc"] == "liście" and w["fakty"][4]["czesc"] is None
    assert cand["opis"] == "Ręczny." and plantfile.validate_candidate(original, cand, "babka") == []
    broken = json.loads(json.dumps(cand))
    broken["wiedza"]["sekcje"][0]["punkty"][0]["zrodla"] = [9]
    broken["wiedza"]["zrodla"][0]["url"] = "javascript:alert(1)"
    errors = plantfile.validate_candidate(original, broken, "babka")
    assert any("zrodla" in e for e in errors) and any("adres" in e for e in errors)
    lost = json.loads(json.dumps(cand))
    lost["wiedza"]["sekcje"][0]["punkty"] = []
    assert any("poza ukladem" in e for e in plantfile.validate_candidate(original, lost, "babka"))
