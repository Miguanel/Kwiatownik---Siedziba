"""Tlumaczenie bez LLM (2026-11-03): kontrole kodem, DeepL + lancuch tlumaczy, LLM tylko wskazuje doslowne
fragmenty, czyszczenie tekstu stron."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.mt.chain import ChainTranslator
from app.mt.deepl import DeepLTranslator
from app.mt.nllb import MTQualityError, MTUnavailable
from app.pipeline import fast_translate as ft
from app.pipeline.checks import ForeignNames, check_segment, check_translation, numbers
from app.pipeline.extract import ExtractionCheckError, SourceText, extract_recipes
from app.pipeline.translate import NoRecipeError
from app.scrapers.clean import clean_text, site_boilerplate

PLANTS = {"rumianek_pospolity": "Rumianek pospolity (Matricaria chamomilla)",
          "bez_czarny": "Bez czarny (Sambucus nigra)"}
NAMES = ForeignNames({
    "rumianek_pospolity": {"lat": "Matricaria chamomilla", "pl": "Rumianek pospolity",
                           "names": {"de": "Echte Kamille", "en": "chamomile", "uk": "ромашка лікарська"}},
    "bez_czarny": {"lat": "Sambucus nigra", "pl": "Bez czarny", "names": {"de": "Schwarzer Holunder"}}})


def errs(where, src, out, lang="en"):
    return [p.what for p in check_segment(where, src, out, lang) if p.level == "blad"]


# ------------------------------------------------------------ kontrole
def test_numbers_and_fractions():
    assert {1.0, 0.5, 1.5} <= numbers("1 1/2 cups") and 1000.0 in numbers("1.000 ml") and 0.5 in numbers("½ tsp")
    assert errs("s", "1 1/2 cups water", "1,5 szklanki wody") == []
    assert errs("s", "2 eggs", "dwa jajka") == []                                   # liczba slownie
    assert errs("s", "Steep for 10 minutes", "Parzyć 15 minut") == ["dopisana liczba 15", "zgubiona liczba 10"]
    assert errs("s", "Steep for 10 minutes", "Parzyć kilka minut") == ["zgubiona liczba 10"]
    assert errs("s", "Im 19. Jahrhundert färbte man Wolle", "W XIX wieku barwiono wełnę", "de") == []   # rzymskie
    assert numbers("zbierać i w lesie") == set() and numbers("XVIe siècle") == {16.0}


def test_foreign_script_untranslated_and_loops():
    assert "obcy alfabet" in errs("k", "Ромашку залити окропом", "Ромашку zalać wrzątkiem", "uk")[0]
    assert errs("k", "Pour the water over herbs", "Pour the water over herbs") == ["fragment nieprzetlumaczony"]
    assert "zapetlenie" in errs("k", "Mix well", "mieszać mieszać mieszać mieszać")[0]
    assert "uciete" in errs("k", "Pour boiling water over the flowers and steep for a while.", "Zalać.")[0]
    assert errs("k", "Ромашку залити окропом", "Rumianek zalać wrzątkiem", "uk") == []
    assert errs("k", "Rumianek", "Rumianek", "pl") == []


def test_plant_lost_in_translation():
    bad = check_translation([("skladnik 1", "2 EL Kamillenblüten", "2 łyżki płytek")], "de", NAMES, PLANTS)
    assert not bad.ok() and "rumianek" in bad.summary().lower()
    good = check_translation([("skladnik 1", "2 EL Kamillenblüten", "2 łyżki kwiatów rumianku")], "de", NAMES, PLANTS)
    assert good.ok()
    # "schwarzer Tee" to nie bez czarny: liczy sie rzeczownik nazwy (Holunder), nie przymiotnik
    assert NAMES.found("1 Beutel schwarzer Tee", "de") == set()
    assert NAMES.found("Holunderblütensirup", "de") == {"bez_czarny"}
    assert NAMES.found("100 г квіток ромашки", "uk") == {"rumianek_pospolity"}
    assert NAMES.found("Sambucus nigra flowers", "en") == {"bez_czarny"}


RECIPE = {"title": "Chamomile tea", "ingredients": ["2 tbsp dried chamomile flowers", "1 cup water"],
          "steps": ["Pour boiling water over the flowers.", "Steep for 10 minutes and strain."]}


class DictMT:
    model_dir = Path("/models/nllb-200-distilled-600M")

    def __init__(self, table):
        self.table = table

    def translate(self, texts, src, tgt="pl"):
        return [self.table.get(t, t) for t in texts]


GOOD = {"Chamomile tea": "Herbata rumiankowa", "2 tbsp dried chamomile flowers": "2 łyżki suszonych kwiatów rumianku",
        "240 ml (1 cup) water": "240 ml (1 szklanka) wody", "Pour boiling water over the flowers.": "Zalać kwiaty wrzątkiem.",
        "Steep for 10 minutes and strain.": "Parzyć 10 minut i odcedzić."}


def test_fast_translate_stops_bad_translation():
    meta = {"item_id": 1, "title": "Chamomile tea", "language": "en"}
    rec, _ = asyncio.run(ft.fast_translate_recipe(DictMT(GOOD), None, RECIPE, meta, PLANTS, set(), NAMES))
    assert rec["siedziba"]["kontrola"] == "ok" and rec["skladniki"][0]["link_id"] == "rumianek_pospolity"
    bad = dict(GOOD, **{"Steep for 10 minutes and strain.": "Parzyć 20 minut i odcedzić."})
    with pytest.raises(MTQualityError, match="kontrola: krok 2: dopisana liczba 20"):
        asyncio.run(ft.fast_translate_recipe(DictMT(bad), None, RECIPE, meta, PLANTS, set(), NAMES))
    lost = dict(GOOD, **{"2 tbsp dried chamomile flowers": "2 łyżki suszonych płytek"})
    with pytest.raises(MTQualityError, match="Rumianek pospolity"):
        asyncio.run(ft.fast_translate_recipe(DictMT(lost), None, RECIPE, meta, PLANTS, set(), NAMES))


# ------------------------------------------------------------ DeepL i lancuch
def deepl_client(log, status=200):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        log.append(body)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"translations": [{"text": "PL:" + t} for t in body["text"]]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_deepl_batches_caches_and_pauses_on_quota():
    log = []
    d = DeepLTranslator("abc:fx", client=deepl_client(log))
    assert d.url == "https://api-free.deepl.com" and d.supports("uk") and not d.supports("hr")
    assert d.translate(["a", "b", "a", ""], "de") == ["PL:a", "PL:b", "PL:a", ""]
    assert log[0]["source_lang"] == "DE" and log[0]["target_lang"] == "PL" and log[0]["text"] == ["a", "b"]
    assert d.translate(["a"], "de") == ["PL:a"] and len(log) == 1                    # pamiec podreczna
    q = DeepLTranslator("abc:fx", client=deepl_client([], status=456))
    with pytest.raises(MTUnavailable, match="limit"):
        q.translate(["x"], "en")
    assert not q.available()


def test_chain_falls_back_to_next_engine():
    class Engine:
        def __init__(self, label, langs, fail=False):
            self.label, self.langs, self.fail, self.error = label, langs, fail, None

        def supports(self, lang):
            return lang in self.langs

        def available(self):
            return True

        def model_present(self):
            return True

        def status(self):
            return {}

        def translate(self, texts, src, tgt="pl"):
            if self.fail:
                raise MTUnavailable("limit")
            return [f"{self.label}:{t}" for t in texts]

    chain = ChainTranslator([Engine("deepl", {"de"}, fail=True), Engine("nllb", {"de", "hr"})])
    assert chain.translate(["x"], "de") == ["nllb:x"] and chain.last_name == "nllb"
    chain = ChainTranslator([Engine("deepl", {"de"}), Engine("nllb", {"de", "hr"})])
    assert chain.translate(["x"], "hr") == ["nllb:x"] and chain.translate(["x"], "de") == ["deepl:x"]
    with pytest.raises(MTUnavailable):
        chain.translate(["x"], "ja")


# ------------------------------------------------------------ LLM wskazuje doslowne fragmenty
PAGE = """Kamillentee gegen Husten
Ein einfacher Tee aus der Hausapotheke.
Zutaten:
2 EL getrocknete Kamillenblüten
250 ml Wasser
Zubereitung:
1. Die Kamillenblüten mit kochendem Wasser übergießen.
2. 10 Minuten ziehen lassen und abseihen."""


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def complete(self, prompt, **kw):
        self.calls += 1
        return SimpleNamespace(text=json.dumps(self.answer, ensure_ascii=False), provider="fake", model="m")


META = {"item_id": 3, "title": "Kamillentee gegen Husten", "language": "de"}


def test_extract_verbatim_lines_pass():
    answer = {"przepisy": [{"title": "Kamillentee gegen Husten", "ingredients": ["2 EL getrocknete Kamillenblüten",
                                                                               "250 ml Wasser"],
                            "steps": ["Die Kamillenblüten mit kochendem Wasser übergießen.",
                                      "10 Minuten ziehen lassen und abseihen."],
                            "notes": "Dreimal täglich eine Tasse trinken."}]}
    ex, model = asyncio.run(extract_recipes(FakeLLM(answer), PAGE, META))
    rec = ex.recipes[0]
    assert model == "fake/m" and rec["ingredients"][1] == "250 ml Wasser" and len(rec["steps"]) == 2
    assert "notes" not in rec and ex.rejected == ["Dreimal täglich eine Tasse trinken."]   # nie ma w tekscie


def test_extract_rejects_rewritten_or_invented():
    rewritten = {"przepisy": [{"title": "Kamillentee", "ingredients": ["2 EL Kamille", "1 Liter heißes Wasser",
                                                                       "1 TL Honig"],
                               "steps": ["Alles mischen und 15 Minuten warten."]}]}
    with pytest.raises(ExtractionCheckError, match="kontrola: LLM przeredagowal"):
        asyncio.run(extract_recipes(FakeLLM(rewritten), PAGE, META))
    src = SourceText(PAGE)
    assert src.verbatim("2 EL getrocknete Kamillenblüten") == 1.0 and not src.numbers_ok("3 EL Kamillenblüten")
    with pytest.raises(NoRecipeError):
        asyncio.run(extract_recipes(FakeLLM({"przepisy": [], "brak_przepisu": True}), PAGE, META))


def test_extracted_recipe_is_translated_by_machine():
    answer = {"przepisy": [{"title": "Kamillentee gegen Husten", "ingredients": ["2 EL getrocknete Kamillenblüten",
                                                                               "250 ml Wasser"],
                            "steps": ["Die Kamillenblüten mit kochendem Wasser übergießen.",
                                      "10 Minuten ziehen lassen und abseihen."]}]}
    table = {"Kamillentee gegen Husten": "Herbata rumiankowa na kaszel",
             "2 EL getrocknete Kamillenblüten": "2 łyżki suszonych kwiatów rumianku", "250 ml Wasser": "250 ml wody",
             "Die Kamillenblüten mit kochendem Wasser übergießen.": "Kwiaty rumianku zalać wrzącą wodą.",
             "10 Minuten ziehen lassen und abseihen.": "Parzyć 10 minut i odcedzić."}
    ex, _ = asyncio.run(extract_recipes(FakeLLM(answer), PAGE, META))
    rec, _ = asyncio.run(ft.fast_translate_recipe(DictMT(table), None, ex.recipes[0], META, PLANTS, set(), NAMES))
    assert rec["tytul"] == "Herbata rumiankowa na kaszel" and rec["skladniki"][0]["link_id"] == "rumianek_pospolity"
    assert rec["sposob_przygotowania"][1] == "Krok 2: Parzyć 10 minut i odcedzić."


# ------------------------------------------------------------ czyszczenie tekstu stron
def test_clean_text_removes_page_noise_only():
    text = """Spis treści
Właściwości rumianku
Napar z rumianku
Właściwości rumianku
Rumianek od wieków stosowany jest na problemy z trawieniem i stany zapalne skóry.
Czytaj też: 10 ziół na kaszel
Napar z rumianku
2 łyżki kwiatów rumianku zalać szklanką wrzątku i parzyć 10 minut pod przykryciem.
Ciasteczka owsiane z rumiankiem
Zapisz się do newslettera
Udostępnij
Reklama
Dodaj do koszyka
2 łyżki kwiatów rumianku zalać szklanką wrzątku i parzyć 10 minut pod przykryciem.
***
Następne pytanie"""
    out = clean_text(text)
    assert out.splitlines() == ["Właściwości rumianku",
                                "Rumianek od wieków stosowany jest na problemy z trawieniem i stany zapalne skóry.",
                                "Napar z rumianku",
                                "2 łyżki kwiatów rumianku zalać szklanką wrzątku i parzyć 10 minut pod przykryciem.",
                                "Ciasteczka owsiane z rumiankiem"]
    assert clean_text(out) == out                                                    # idempotentne
    long_line = "Zapisz się do newslettera, a dostaniesz przepis na syrop z mniszka, który robimy co roku w maju."
    assert clean_text(long_line) == long_line                                        # dlugich zdan nie ruszamy


def test_site_boilerplate_lines():
    pages = [f"Przepis {i}\nTreść przepisu numer {i} z ziołami i miodem, długa linia.\nO nas | Kontakt\n"
             f"Sklep zielarski Ziółko" for i in range(6)]
    bp = site_boilerplate(pages)
    assert clean_text(pages[0], bp).splitlines() == ["Przepis 0", "Treść przepisu numer 0 z ziołami i miodem, długa linia."]
    assert site_boilerplate(pages[:2]) == set()


# ------------------------------------------------------------ calosc: zadanie "translate" bez tlumaczenia przez LLM
def test_run_translate_extracts_verbatim_and_machine_translates(tmp_path, monkeypatch):
    from sqlmodel import Session, SQLModel, create_engine

    import app.worker.export as exp
    import app.worker.jobs as jobs_mod
    from app.config import settings
    from app.models import Item, ItemKind, ItemStatus, Job, Source

    przepisy, plants = tmp_path / "przepisy", tmp_path / "plants"
    przepisy.mkdir()
    plants.mkdir()
    (plants / "rumianek_pospolity.json").write_text(json.dumps({"id": "rumianek_pospolity",
                                                                "nazwa_pl": "Rumianek pospolity"}), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "translate_with_llm", False)
    eng = create_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(exp, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        src = Source(name="DE", base_url="https://de.example/")
        s.add(src)
        s.commit()
        good = Item(source_id=src.id, url="https://de.example/tee", kind=ItemKind.recipe, language="de",
                    title="Kamillentee gegen Husten", raw_text=PAGE + "\nTeilen\nReklama")
        bad = Item(source_id=src.id, url="https://de.example/tee2", kind=ItemKind.recipe, language="de",
                   title="Kamillentee", raw_text=PAGE)
        job = Job(kind="translate")
        for x in (good, bad, job):
            s.add(x)
        s.commit()
        ids, jid = (good.id, bad.id), job.id

    table = {"Kamillentee gegen Husten": "Herbata rumiankowa na kaszel",
             "2 EL getrocknete Kamillenblüten": "2 łyżki suszonych kwiatów rumianku", "250 ml Wasser": "250 ml wody",
             "Die Kamillenblüten mit kochendem Wasser übergießen.": "Kwiaty rumianku zalać wrzącą wodą.",
             "10 Minuten ziehen lassen und abseihen.": "Parzyć 10 minut i odcedzić."}

    class MT(DictMT):
        last_name = "deepl/api-free"

        def supports(self, lang):
            return True

        def model_present(self):
            return True

        def available(self):
            return True

    monkeypatch.setattr(exp, "get_translator", lambda: MT(table))
    verbatim = {"przepisy": [{"title": "Kamillentee gegen Husten",
                              "ingredients": ["2 EL getrocknete Kamillenblüten", "250 ml Wasser"],
                              "steps": ["Die Kamillenblüten mit kochendem Wasser übergießen.",
                                        "10 Minuten ziehen lassen und abseihen."]}]}
    invented = {"przepisy": [{"title": "Kamillentee", "ingredients": ["3 EL Kamille", "1 TL Honig"],
                              "steps": ["Alles mischen und 15 Minuten warten."]}]}
    answers = [verbatim, invented]

    class LLM:
        prompts = []

        async def complete(self, prompt, system=None, json_mode=False):
            self.prompts.append((prompt, system))
            return SimpleNamespace(text=json.dumps(answers.pop(0), ensure_ascii=False), provider="fake", model="m")

    llm = LLM()
    stats = asyncio.run(exp.run_translate(jid, jobs_mod.make_runner(1), llm, item_ids=list(ids)))
    assert stats["translated"] == 1 and stats["error"] == 1
    assert "NIE TLUMACZ" in llm.prompts[0][1] and "Teilen" not in llm.prompts[0][0]     # tekst oczyszczony
    with Session(eng) as s:
        g, b = s.get(Item, ids[0]), s.get(Item, ids[1])
        rec = json.loads(g.data_json)
        assert rec["tytul"] == "Herbata rumiankowa na kaszel" and rec["siedziba"]["model"] == "deepl/api-free"
        assert "doslowne" in rec["siedziba"]["wyciagnal"]
        assert b.status == ItemStatus.error and b.reason.startswith("kontrola: LLM przeredagowal")
