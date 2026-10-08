"""Wiedza o roslinach: Wikipedia (sztuczne API), wyciaganie informacji z cytatami, kopia pliku i jej kontrola."""
import asyncio
import json
from types import SimpleNamespace

import httpx

from app.knowledge import plantfile
from app.knowledge.extract import numbers_supported, parse_facts, quote_in_text
from app.knowledge.sections import fingerprint, similar
from app.knowledge.verify import parse_verdicts
from app.knowledge.wiki import WikiClient, select_sections, split_sections

WIKI_TEXT = """Krwawnik pospolity – gatunek rośliny z rodziny astrowatych. W Polsce pospolity na łąkach.

== Morfologia ==
Łodyga do 80 cm wysokości, liście pierzastosieczne.

== Zastosowanie ==
W medycynie ludowej krwawnikiem tamowano krwawienia z ran. Kwiaty służyły do barwienia wełny na żółto.

=== W kulturze ===
Nazwa Achillea pochodzi od Achillesa, który według legendy leczył nim rany żołnierzy.

== Przypisy ==
1. Jakaś książka
"""


def wiki_handler(request: httpx.Request) -> httpx.Response:
    p = dict(request.url.params)
    if request.url.host == "www.wikidata.org" and p.get("action") == "wbsearchentities":
        return httpx.Response(200, json={"search": [{"id": "Q1"}, {"id": "Q25408"}]})
    if request.url.host == "www.wikidata.org" and p.get("action") == "wbgetentities":
        return httpx.Response(200, json={"entities": {
            "Q1": {"claims": {}, "sitelinks": {}},                       # nie takson (np. miasto) -> pomijany
            "Q25408": {"claims": {"P225": [{"mainsnak": {"datavalue": {"value": "Achillea millefolium"}}}]},
                       "labels": {"pl": {"value": "krwawnik pospolity"}},
                       "sitelinks": {"plwiki": {"title": "Krwawnik pospolity"},
                                     "enwiki": {"title": "Achillea millefolium"},
                                     "commonswiki": {"title": "x"}}}}})
    if request.url.host == "pl.wikipedia.org":
        return httpx.Response(200, json={"query": {"pages": {"1": {
            "title": "Krwawnik pospolity", "extract": WIKI_TEXT,
            "fullurl": "https://pl.wikipedia.org/wiki/Krwawnik_pospolity"}}}})
    return httpx.Response(404)


def test_wikipedia_sections_and_client():
    secs = split_sections(WIKI_TEXT)
    assert [t for t, _ in secs] == ["", "Morfologia", "Zastosowanie", "Przypisy"]
    assert "W kulturze:" in secs[2][1]                       # podsekcja zostaje w sekcji rodzica
    chosen, used = select_sections(WIKI_TEXT)
    assert "Przypisy" not in chosen and used[:2] == ["wstep", "Zastosowanie"]   # zastosowanie przed morfologia

    async def run():
        wc = WikiClient("TestBot", client=httpx.AsyncClient(transport=httpx.MockTransport(wiki_handler)), delay_s=0)
        try:
            wp = await wc.find_plant("Achillea millefolium")
            art = await wc.article("pl", wp.titles["pl"])
            return wp, art
        finally:
            await wc.aclose()
    wp, art = asyncio.run(run())
    assert wp.qid == "Q25408" and wp.titles == {"pl": "Krwawnik pospolity", "en": "Achillea millefolium"}
    assert art.url.endswith("Krwawnik_pospolity") and "barwienia wełny" in art.text


def test_facts_need_real_quotes_and_supported_numbers():
    raw = json.dumps({"fakty": [
        {"sekcja": "barwienie", "czesc": "kwiaty", "tekst": "Kwiatami krwawnika barwiono dawniej wełnę na żółto.",
         "cytat": "Kwiaty służyły do barwienia wełny na żółto."},
        {"sekcja": "historia", "czesc": None, "tekst": "Nazwa pochodzi od Achillesa, który leczył nim rany.",
         "cytat": "Nazwa Achillea pochodzi od Achillesa, ktory wedlug legendy leczyl nim rany"},   # bez ogonkow - ok
        {"sekcja": "opis", "tekst": "Roślina osiąga do 150 cm wysokości w dobrych warunkach.",
         "cytat": "Łodyga do 80 cm wysokości"},                                   # liczba spoza cytatu
        {"sekcja": "kultura", "tekst": "Krwawnik wieszano nad drzwiami, by chronił dom przed złem.",
         "cytat": "Krwawnik wieszano nad drzwiami domow jako ochrone"},           # zmyslony cytat
        {"sekcja": "zla_sekcja", "tekst": "W medycynie ludowej krwawnikiem tamowano krwawienia.",
         "cytat": "W medycynie ludowej krwawnikiem tamowano krwawienia z ran."},  # nieznana sekcja -> ciekawostki
    ]})
    facts, rejected = parse_facts(raw, WIKI_TEXT)
    assert [f.section for f in facts] == ["barwienie", "historia", "ciekawostki"]
    assert facts[0].part == "kwiaty" and len(rejected) == 2
    assert quote_in_text("tamowano krwawienia z ran", WIKI_TEXT) and not quote_in_text("krótkie", WIKI_TEXT)
    assert numbers_supported("W 1820 roku opisano 3 odmiany", "opisano w 1820 r. 3 odmiany")
    assert not numbers_supported("dawka 5 g", "dawka 2 g")


def test_fingerprint_and_similarity():
    a = "Kwiatami krwawnika barwiono dawniej wełnę na żółto."
    b = "Dawniej kwiatami krwawnika barwiono na żółto wełnę."
    assert fingerprint("k", a) == fingerprint("k", b) and fingerprint("k", a) != fingerprint("m", a)
    assert similar(a, b) == 1.0 and similar(a, "Liście są pierzastosieczne i miękkie.") < 0.2


def _fact(i, text, url="https://pl.wikipedia.org/wiki/Krwawnik", section="ciekawostki"):
    return SimpleNamespace(id=i, section=section, part=None, text=text, language="pl", source_url=url,
                           source_name="Wikipedia (pl)")


def test_candidate_keeps_manual_fields_and_validation_catches_problems():
    original = {"id": "krwawnik", "nazwa_pl": "Krwawnik", "opis": "Ręczny opis.", "ciekawostki": ["A"],
                "wiedza": {"fakty": [{"id": "s1", "sekcja": "opis", "tekst": "Stara informacja o krwawniku z sieci.",
                                      "zrodlo": {"nazwa": "x", "url": "https://a.pl/1"}}]}}
    facts = [_fact(1, "Stara informacja o krwawniku z sieci."), _fact(2, "Kwiatami krwawnika barwiono wełnę na żółto.",
                                                                    section="barwienie")]
    cand = plantfile.build_candidate(original, {}, facts, today="2026-09-30")
    assert cand["opis"] == "Ręczny opis." and cand["ciekawostki"] == ["A"]
    assert [f["id"] for f in cand["wiedza"]["fakty"]] == ["s1", "s2"]          # s1 nie zdublowany
    assert [z["url"] for z in cand["wiedza"]["zrodla"]] == ["https://a.pl/1", "https://pl.wikipedia.org/wiki/Krwawnik"]
    assert plantfile.validate_candidate(original, cand, "krwawnik") == []

    broken = json.loads(json.dumps(cand))
    broken["opis"] = "zmienione przez automat"
    broken["wiedza"]["fakty"].append({"id": "s9", "sekcja": "opis", "tekst": "<script>alert(1)</script> tekst tekst",
                                      "zrodlo": {"url": "javascript:alert(1)"}})
    errors = plantfile.validate_candidate(original, broken, "krwawnik")
    assert any("opis" in e for e in errors) and any("HTML" in e for e in errors) and any("adresu" in e for e in errors)
    lost = json.loads(json.dumps(cand))
    lost["wiedza"]["fakty"] = lost["wiedza"]["fakty"][1:]
    assert any("zniknely" in e for e in plantfile.validate_candidate(original, lost, "krwawnik"))


def test_new_plant_file_and_safe_write(tmp_path):
    plants, backups = tmp_path / "plants", tmp_path / "backups"
    plants.mkdir()
    base = plantfile.skeleton("szczaw", "Szczaw", "Rumex acetosa")
    cand = plantfile.build_candidate(None, base, [_fact(5, "Szczaw rośnie na łąkach i przydrożach.", section="opis")])
    assert cand["opis"].startswith("Szczaw rośnie") and plantfile.validate_candidate(None, cand, "szczaw") == []
    assert plantfile.write_plant(plants, "szczaw", cand, False, backups) is None          # nowy plik - brak kopii
    (plants / "krwawnik.json").write_bytes(json.dumps({"id": "krwawnik", "nazwa_pl": "K"}).encode().replace(b"}", b"}\r\n"))
    orig, crlf = plantfile.read_plant(plants, "krwawnik")
    assert crlf
    cand2 = plantfile.build_candidate(orig, {}, [_fact(6, "Krwawnik tamował krwawienia z ran w medycynie ludowej.")])
    backup = plantfile.write_plant(plants, "krwawnik", cand2, crlf, backups)
    assert backup.exists() and b"\r\n" in (plants / "krwawnik.json").read_bytes()
    assert json.loads((plants / "krwawnik.json").read_text(encoding="utf-8"))["wiedza"]["fakty"][0]["id"] == "s6"


def test_verifier_answers_parsed_only_for_known_ids():
    raw = json.dumps({"oceny": [{"id": 1, "ok": True}, {"id": "2", "ok": False, "powod": "cytat mowi co innego"},
                                {"id": 99, "ok": True}, {"id": "x"}]})
    assert parse_verdicts(raw, [1, 2, 3]) == {1: (True, ""), 2: (False, "cytat mowi co innego")}
