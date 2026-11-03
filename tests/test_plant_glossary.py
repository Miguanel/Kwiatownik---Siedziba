"""Nazwy roslin z Wikidata (bez LLM): uzupelnianie, poziomy dopasowania w kontroli i slownik DeepL."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
from sqlmodel import Session, SQLModel, create_engine

from app.knowledge.plant_names import plants_missing_names, refresh_plant_names
from app.knowledge.wiki import aliases_by_lang
from app.models import Plant
from app.mt.deepl import DeepLTranslator
from app.pipeline.checks import ForeignNames, check_translation, glossary_entries

ROWS = {
    "rumianek_pospolity": {"lat": "Matricaria chamomilla", "pl": "Rumianek pospolity",
                           "names": {"en": "Matricaria chamomilla", "de": "Echte Kamille"},
                           "aliases": {"en": ["German chamomile", "chamomile"]},
                           "titles": {"de": "Echte Kamille", "uk": "Ромашка лікарська (рослина)"}},
    "lipa_drobnolistna": {"lat": "Tilia cordata", "pl": "Lipa drobnolistna", "names": {"en": "small-leaved lime"},
                          "aliases": {"en": ["littleleaf linden", "lime"]}},
}
PLANTS = {"rumianek_pospolity": "Rumianek pospolity (Matricaria chamomilla)", "lipa_drobnolistna": "Lipa drobnolistna"}


def test_levels_from_labels_aliases_and_titles():
    n = ForeignNames(ROWS)
    assert n.found("2 tbsp dried chamomile flowers", "en") == {"rumianek_pospolity"}
    assert n.found_levels("juice of 1 lime", "en") == {}                         # 'lime' = limonka
    assert n.found_levels("a handful of linden flowers", "en") == {"lipa_drobnolistna": "uwaga"}
    assert n.found("100 г квітів ромашки", "uk") == {"rumianek_pospolity"}       # tytul artykulu, bez nawiasu
    rep = check_translation([("skladnik 1", "a handful of linden flowers", "garść kwiatów")], "en", n, PLANTS)
    assert rep.ok() and len(rep.warnings) == 1                                   # niepewne -> tylko uwaga


def test_glossary_entries():
    de = glossary_entries(ROWS, "de")
    assert de == {"Echte Kamille": "rumianek pospolity"}
    en = glossary_entries(ROWS, "en")
    assert en["chamomile"] == "rumianek pospolity" and en["German chamomile"] == "Rumianek pospolity"
    assert en["littleleaf linden"] == "lipa drobnolistna"
    assert "lime" not in en and "Matricaria chamomilla" not in en and "matricaria chamomilla" not in en
    assert glossary_entries(ROWS, "uk")["ромашка лікарська"] == "rumianek pospolity"
    assert aliases_by_lang({"en": [{"value": "chamomile"}], "zh": [{"value": "x"}]}) == {"en": ["chamomile"]}


def test_deepl_creates_glossary_once_and_uses_it():
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content) if req.content else None
        calls.append((req.method, req.url.path, body))
        if req.url.path == "/v2/glossary-language-pairs":
            return httpx.Response(200, json={"supported_languages": [{"source_lang": "de", "target_lang": "pl"}]})
        if req.url.path == "/v2/glossaries" and req.method == "GET":
            return httpx.Response(200, json={"glossaries": [{"glossary_id": "old", "name": "kwiatownik-de-pl-0000",
                                                             "ready": True}]})
        if req.url.path == "/v2/glossaries" and req.method == "POST":
            return httpx.Response(201, json={"glossary_id": "g1", "ready": True})
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"translations": [{"text": "PL:" + t} for t in body["text"]]})

    d = DeepLTranslator("k:fx", client=httpx.Client(transport=httpx.MockTransport(handler)),
                        glossary=lambda src: glossary_entries(ROWS, src))
    assert d.translate(["Kamillentee"], "de") == ["PL:Kamillentee"]
    made = [c for c in calls if c[0] == "POST" and c[1] == "/v2/glossaries"]
    assert len(made) == 1 and made[0][2]["entries"] == "Echte Kamille\trumianek pospolity"
    assert ("DELETE", "/v2/glossaries/old", None) in calls                    # stara wersja slownika usunieta
    assert [c for c in calls if c[1] == "/v2/translate"][0][2]["glossary_id"] == "g1"
    n = len(calls)
    d.translate(["Kamillenblüten"], "de")
    assert len(calls) == n + 1 and calls[-1][2]["glossary_id"] == "g1"         # bez ponownego tworzenia
    d.translate(["chamomile"], "en")                                           # para en-pl nieobslugiwana
    assert "glossary_id" not in calls[-1][2] and d.status()["glossaries"] == {"de-pl": 1}


def test_refresh_plant_names(tmp_path):
    eng = create_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    old = datetime.now(timezone.utc) - timedelta(days=5)
    with Session(eng) as s:
        s.add(Plant(id="pokrzywa_zwyczajna", nazwa_pl="Pokrzywa zwyczajna", nazwa_lat="Urtica dioica"))
        s.add(Plant(id="mniszek_lekarski", nazwa_pl="Mniszek lekarski", nazwa_lat="Taraxacum officinale"))
        s.add(Plant(id="krwawnik", nazwa_pl="Krwawnik", nazwa_lat="Achillea millefolium", names_json="{}",
                    aliases_json="{}", names_checked_at=old))                   # sprawdzony niedawno
        s.commit()
    assert plants_missing_names(eng, 10) == ["mniszek_lekarski", "pokrzywa_zwyczajna"]

    class Wiki:
        async def find_plant(self, latin, name_pl):
            if latin == "Urtica dioica":
                return SimpleNamespace(qid="Q25351", labels={"de": "Große Brennnessel"},
                                       aliases={"en": ["stinging nettle"]}, titles={"de": "Große Brennnessel"})
            return None

    logs = []
    stats = asyncio.run(refresh_plant_names(eng, Wiki(), logs.append))
    assert stats == {"checked": 2, "found": 1, "errors": 0}
    with Session(eng) as s:
        p = s.get(Plant, "pokrzywa_zwyczajna")
        assert json.loads(p.aliases_json) == {"en": ["stinging nettle"]} and p.wikidata == "Q25351"
        assert s.get(Plant, "mniszek_lekarski").names_checked_at is not None
    assert plants_missing_names(eng, 10) == []                                  # nie pyta drugi raz
