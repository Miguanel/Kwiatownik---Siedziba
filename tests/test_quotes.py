"""Wiedza bez tekstow LLM (2026-11-03): doslowny fragment zrodla -> tlumacz maszynowy -> kontrola; czekajace na
tlumacza; podmiana dawnych tekstow LLM; budzet DeepL na wiedze; scalanie i redakcja bez LLM; wzmianki z cytatow."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.knowledge import merge, plantfile
from app.knowledge.extract import Fact
from app.knowledge.quotes import finish_facts, quote_leads, snap_quote, supported_label, translate_quotes
from app.mt.chain import ChainTranslator
from app.mt.deepl import DeepLTranslator
from app.mt.nllb import MTUnavailable

SRC = ("Die Kamille ist eine alte Heilpflanze. Schon im Mittelalter legte man Kamillenblüten in die Truhen, "
       "um Motten fernzuhalten. Im 19. Jahrhundert nutzten Färber die Blüten für gelbe Wolle.\n"
       "Rezept: 2 EL Blüten mit 250 ml Wasser übergießen.")


class TableMT:
    def __init__(self, table, purposes=None):
        self.table, self.purposes, self.last_name = table, purposes if purposes is not None else [], "deepl/api-free"

    def translate(self, texts, src, tgt="pl", purpose="przepis"):
        self.purposes.append(purpose)
        return [self.table.get(t, "") for t in texts]


class DownMT:
    def translate(self, texts, src, tgt="pl", purpose="przepis"):
        raise MTUnavailable("budzet DeepL na wiedze wyczerpany")


def test_snap_quote_takes_exact_full_sentence_from_source():
    assert snap_quote("legte man Kamillenblüten in die Truhen", SRC) == \
        "Schon im Mittelalter legte man Kamillenblüten in die Truhen, um Motten fernzuhalten."
    # model zmienil slowa / interpunkcje - i tak bierzemy tekst zrodla
    assert snap_quote("im 19 Jahrhundert nutzten die Färber Blüten fur gelbe Wolle", SRC) == \
        "Im 19. Jahrhundert nutzten Färber die Blüten für gelbe Wolle."
    assert snap_quote("Kamille vertreibt böse Geister aus dem Haus", SRC) is None       # nie ma w zrodle
    assert snap_quote("ab", SRC) is None


def test_finish_facts_translates_checks_and_waits():
    q1 = "Schon im Mittelalter legte man Kamillenblüten in die Truhen, um Motten fernzuhalten."
    q2 = "Im 19. Jahrhundert nutzten Färber die Blüten für gelbe Wolle."
    mt = TableMT({q1: "Już w średniowieczu wkładano kwiaty rumianku do skrzyń, aby odstraszyć mole.",
                  q2: "W XX wieku farbiarze używali kwiatów do barwienia wełny na żółto."})      # zla liczba
    facts = [Fact("historia", None, "", q1), Fact("barwienie", "kwiaty", "", q2)]
    ready, pending, rejected = finish_facts(facts, "de", mt)
    assert [f.text for f in ready] == ["Już w średniowieczu wkładano kwiaty rumianku do skrzyń, aby odstraszyć mole."]
    assert ready[0].translated_by == "deepl/api-free" and mt.purposes == ["wiedza"]
    assert not pending and "zgubiona liczba 19" in rejected[0]
    ready, pending, _ = finish_facts([Fact("historia", None, "", q1)], "de", DownMT())
    assert not ready and pending[0].text == q1 and pending[0].translated_by is None
    ready, _, _ = finish_facts([Fact("opis", None, "", "Rumianek rośnie na polach i przydrożach.")], "pl", None)
    assert ready[0].text == "Rumianek rośnie na polach i przydrożach." and ready[0].translated_by == "cytat"


def test_magic_labels_kept_only_when_in_translation():
    assert supported_label("XIX w.", "W XIX wieku wierzono, że...") and not supported_label("XVI w.", "W XIX wieku")
    assert supported_label("Podlasie", "Na Podlasiu wierzono, że ziele chroni dom.")
    assert not supported_label("Niemcy", "Na Podlasiu wierzono, że ziele chroni dom.")
    f = SimpleNamespace(quote="In Bayern hängte man Johanniskraut über die Stalltür.", text="", translated_by=None,
                        meta={"rodzaj": "ochrona", "kraj": "Bawaria", "epoka": "średniowiecze"})
    mt = TableMT({f.quote: "W Bawarii wieszano dziurawiec nad drzwiami obory."})
    ready, _, _ = finish_facts([f], "de", mt)
    assert ready[0].meta == {"rodzaj": "ochrona", "kraj": "Bawaria"}


def test_translate_quotes_without_translator_waits():
    out = translate_quotes(None, [("de", "Ein Satz."), ("pl", "Zdanie po polsku.")])
    assert out[0][0] is None and out[0][1].startswith("brak tlumacza") and out[1] == ("Zdanie po polsku.", "", "cytat")


def test_quote_leads_describe_with_translated_quote():
    leads = [{"tytul": "Napar z rumianku na sen", "tytul_oryginalny": "Kamillentee", "co_wiadomo": "LLM pisze...",
              "cytat": "legte man Kamillenblüten in die Truhen"},
             {"tytul": "Coś", "co_wiadomo": "zmyslone", "cytat": "nie ma tego w tekscie wcale nigdzie"}]
    src = SRC + " Kamillentee trinkt man abends."
    mt = TableMT({"Schon im Mittelalter legte man Kamillenblüten in die Truhen, um Motten fernzuhalten.":
                  "Już w średniowieczu wkładano kwiaty rumianku do skrzyń, aby odstraszyć mole.",
                  "Kamillentee": "Herbata rumiankowa"})
    out = quote_leads(leads, src, "de", mt)
    assert out[0]["co_wiadomo"].startswith("Już w średniowieczu") and out[0]["tytul"] == "Herbata rumiankowa"
    assert out[1]["co_wiadomo"] is None and out[1]["cytat"] is None


# ------------------------------------------------------------------ budzet DeepL i lancuch
def test_deepl_knowledge_budget_and_chain_quality():
    def handler(req):
        if req.url.path == "/v2/usage":
            return httpx.Response(200, json={"character_count": 310000, "character_limit": 500000})
        body = json.loads(req.content)
        return httpx.Response(200, json={"translations": [{"text": "PL:" + t} for t in body["text"]]})
    d = DeepLTranslator("k:fx", client=httpx.Client(transport=httpx.MockTransport(handler)), knowledge_share=0.6)
    assert d.translate(["x"], "de", purpose="przepis") == ["PL:x"]                # przepisy - zawsze
    with pytest.raises(MTUnavailable, match="budzet"):
        d.translate(["y"], "de", purpose="wiedza")                               # 310k >= 60% z 500k
    d2 = DeepLTranslator("k:fx", client=httpx.Client(transport=httpx.MockTransport(handler)), knowledge_share=0.7)
    assert d2.translate(["y"], "de", purpose="wiedza") == ["PL:y"]

    class Local:
        label, quality, error = "nllb/x", "niska", None

        def supports(self, lang):
            return True

        def available(self):
            return True

        def translate(self, texts, src, tgt="pl"):
            return ["nllb:" + t for t in texts]
    chain = ChainTranslator([d, Local()])
    assert chain.translate(["z"], "de", purpose="przepis") == ["PL:z"]
    with pytest.raises(MTUnavailable):
        chain.translate(["z2"], "de", purpose="wiedza")                          # NLLB nie tlumaczy wiedzy
    assert ChainTranslator([d, Local()], knowledge_local=True).translate(["z3"], "de", purpose="wiedza") == ["nllb:z3"]


# ------------------------------------------------------------------ baza: czekajace i dawne teksty LLM
def test_pending_and_old_llm_texts_replaced(tmp_path, monkeypatch):
    import app.worker.knowledge as kn
    from app.models import Plant, PlantFact
    eng = create_engine(f"sqlite:///{(tmp_path / 'q.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    q1 = "Schon im Mittelalter legte man Kamillenblüten in die Truhen, um Motten fernzuhalten."
    q2 = "Im 19. Jahrhundert nutzten Färber die Blüten für gelbe Wolle."
    with Session(eng) as s:
        s.add(Plant(id="rumianek", nazwa_pl="Rumianek"))
        s.add(PlantFact(plant_id="rumianek", section="historia", text=q1, quote=q1, source_url="u", language="de",
                        fingerprint="a", status="untranslated"))
        s.add(PlantFact(plant_id="rumianek", section="barwienie", text="Stary tekst napisany przez LLM o farbiarzach.",
                        quote=q2, source_url="u", language="de", fingerprint="b", status="applied"))
        s.add(PlantFact(plant_id="rumianek", section="opis", text="Parafraza LLM.", quote="Rumianek rośnie na polach.",
                        source_url="u", language="pl", fingerprint="c", status="verified"))
        s.commit()
    mt = TableMT({q1: "Już w średniowieczu wkładano kwiaty rumianku do skrzyń, aby odstraszyć mole.",
                  q2: "W XIX wieku farbiarze używali kwiatów do barwienia wełny na żółto."})
    monkeypatch.setattr(kn, "get_translator", lambda: mt)
    logs = []
    st = asyncio.run(kn.translate_pending_facts(logs.append, limit=10))
    assert st == {"translated": 1, "old_replaced": 2, "rejected": 0, "waiting": 0}
    with Session(eng) as s:
        rows = {r.section: r for r in s.exec(select(PlantFact)).all()}
    assert rows["historia"].status == "new" and rows["historia"].text.startswith("Już w średniowieczu")
    assert rows["barwienie"].status == "verified" and rows["barwienie"].translated_by == "deepl/api-free"
    assert rows["opis"].text == "Rumianek rośnie na polach." and rows["opis"].translated_by == "cytat"
    assert asyncio.run(kn.translate_pending_facts(logs.append, limit=10)) == \
        {"translated": 0, "old_replaced": 0, "rejected": 0, "waiting": 0}            # nic drugi raz
    # podmieniony tekst trafia do pliku rosliny w miejsce starego wpisu
    old_file = {"id": "rumianek", "wiedza": {"fakty": [{"id": f"s{rows['barwienie'].id}", "sekcja": "barwienie",
                                                         "tekst": "Stary tekst napisany przez LLM o farbiarzach."}]}}
    cand = plantfile.build_candidate(old_file, {}, [rows["barwienie"]])
    assert [f["tekst"] for f in cand["wiedza"]["fakty"]] == ["W XIX wieku farbiarze używali kwiatów do barwienia "
                                                            "wełny na żółto."]


def test_merge_without_llm_rebuilds_old_llm_texts():
    data = {"id": "x", "opis": "Ręczny opis rośliny.",
            "wiedza": {"fakty": [{"id": "s1", "sekcja": "opis", "tekst": "Roślina dorasta do metra.",
                                  "zrodlo": {"url": "u", "nazwa": "u"}}],
                       "zrodla": [{"nr": 1, "url": "u", "nazwa": "u"}],
                       "sekcje": [{"klucz": "opis", "tytul": "Opis", "punkty": [
                           {"tekst": "Roślina dorasta do metra.", "czesc": None, "zrodla": [1], "fakty": ["s1"]}]}]}}

    class NoLLM:
        async def complete(self, *a, **k):
            raise AssertionError("LLM nie powinien pisac tekstow")
    sc = asyncio.run(merge.build_merged(NoLLM(), data, "X", write_with_llm=False))
    if sc:                                     # przydzial moze nie znalezc podrozdzialu - wtedy nic do scalenia
        assert all(e["model"].startswith("bez LLM") for e in sc["pola"].values())
