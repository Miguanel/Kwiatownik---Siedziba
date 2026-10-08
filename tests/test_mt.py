"""Tlumaczenie bez LLM: NLLB (sztuczny model), reguly typu, skladniki, jednostki, powrot do LLM."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from app.mt.nllb import MTQualityError, MTUnavailable, NLLBTranslator, looks_broken, nllb_code, split_segments
from app.pipeline import fast_translate as ft

PLANTS = {"rumianek_pospolity": "Rumianek pospolity (Matricaria chamomilla)",
          "mieta_pieprzowa": "Mięta pieprzowa (Mentha piperita)", "bez_czarny": "Bez czarny (Sambucus nigra)"}
DICT = {
    "Chamomile tea for sleep": "Herbata rumiankowa na sen",
    "A calming evening tea.": "Uspokajająca wieczorna herbata.",
    "2 tbsp dried chamomile flowers": "2 łyżki suszonych kwiatów rumianku",
    "240 ml (1 cup) water": "240 ml (1 szklanka) wody",
    "1 tsp honey": "1 łyżeczka miodu",
    "Pour boiling water over the flowers.": "Zalać kwiaty wrzątkiem.",
    "Steep for 10 minutes and strain.": "Parzyć 10 minut i odcedzić.",
    "1 cup": "1 filiżanka",
}
RECIPE = {"title": "Chamomile tea for sleep", "description": "A calming evening tea.",
          "ingredients": ["2 tbsp dried chamomile flowers", "1 cup water", "1 tsp honey"],
          "steps": ["Pour boiling water over the flowers.", "Steep for 10 minutes and strain."],
          "yield": "1 cup", "total_time": "PT15M"}


class FakeMT:
    model_dir = Path("/models/nllb-200-distilled-600M")

    def __init__(self, broken=False):
        self.broken, self.calls = broken, 0

    def translate(self, texts, src, tgt="pl"):
        self.calls += 1
        return [("i i i i i" if self.broken else DICT.get(t, t)) for t in texts]


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def complete(self, prompt, **kw):
        self.calls += 1
        return SimpleNamespace(text=json.dumps(self.answer), provider="fake", model="m")


META = {"item_id": 7, "title": "Chamomile tea for sleep", "language": "en", "country": "gb"}


def test_helpers():
    assert nllb_code("uk") == "ukr_Cyrl" and nllb_code("pl-PL") == "pol_Latn" and nllb_code("xx") is None
    long = "To jest zdanie numer jeden o ziołach. " * 20
    segs = split_segments(long)
    assert len(segs) > 1 and all(len(s) <= 400 for s in segs) and " ".join(segs) == long.strip()
    assert looks_broken("salt", "") and looks_broken("tea", "i i i i i") and not looks_broken("salt", "sól")
    assert ft.metric_units("1 cup water, 350°F") == "240 ml (1 cup) water, 175°C (350°F)"
    assert ft.split_quantity("2 łyżki suszonych kwiatów") == ("2 łyżki", "suszonych kwiatów")
    assert ft.split_quantity("Sól") == (None, "Sól") and ft.split_quantity("3 gruszki") == ("3", "gruszki")
    assert ft.iso_duration("PT1H30M") == "1 godz. 30 min"
    assert ft.PlantIndex(PLANTS).match("liście mięty") == "mieta_pieprzowa"
    assert ft.PlantIndex(PLANTS).match("bez cukru") is None
    assert ft.classify("Syrop z mniszka na kaszel") == "medyczne_wewnetrzne_syrop"
    assert ft.classify("Farbowanie włosów henną") == "barwierskie_wlosy"
    assert ft.classify("Ciasto z rabarbarem") == "kulinarne" and ft.classify("Coś") is None
    assert ft.classify("Настойка календулы", "Настойка календулы") == "medyczne_wewnetrzne_nalewka"


def test_nllb_translator_with_fake_model(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"x")
    (tmp_path / "sentencepiece.bpe.model").write_bytes(b"x")
    tr = NLLBTranslator(tmp_path)
    seen = []

    class SP:
        def encode(self, s, out_type=str):
            return s.split()

        def decode(self, toks):
            return " ".join(toks)

    class CT2:
        def translate_batch(self, batch, target_prefix, **kw):
            seen.append((batch, target_prefix))
            return [SimpleNamespace(hypotheses=[[p[0]] + [w.upper() for w in b[1:-1]] + ["."]])
                    for b, p in zip(batch, target_prefix)]

    tr._translator, tr._sp = CT2(), SP()
    assert tr.translate(["salt", "hot water"], "en") == ["SALT", "HOT WATER"]      # kropka dopisana przez model usunieta
    assert seen[0][0][0] == ["eng_Latn", "hot", "water", "</s>"] and seen[0][1][0] == ["pol_Latn"]
    assert tr.translate(["salt"], "en") == ["SALT"] and len(seen) == 1              # pamiec podreczna
    assert tr.translate(["sól"], "pl") == ["sól"]
    try:
        tr.translate(["x"], "xx")
        raise AssertionError("nieznany jezyk")
    except MTUnavailable:
        pass
    assert NLLBTranslator(tmp_path / "brak").available() is False


def test_fast_translate_structured_recipe_without_llm():
    mt = FakeMT()
    rec, model = asyncio.run(ft.fast_translate_recipe(mt, None, RECIPE, META, PLANTS, set()))
    assert model == "nllb/nllb-200-distilled-600M" and mt.calls == 1
    assert rec["tytul"] == "Herbata rumiankowa na sen" and rec["typ"] == "medyczne_wewnetrzne_napar"
    assert rec["metoda"] == "napar" and rec["slug"] == "rumianek_pospolity"
    assert rec["skladniki"][0] == {"nazwa": "Suszonych kwiatów rumianku", "ilosc": "2 łyżki",
                                   "link_id": "rumianek_pospolity"}
    assert rec["skladniki"][1]["ilosc"] == "240 ml (1 szklanka)"                  # jednostki przed tlumaczeniem
    assert rec["sposob_przygotowania"] == ["Krok 1: Zalać kwiaty wrzątkiem.", "Krok 2: Parzyć 10 minut i odcedzić."]
    assert rec["czas_przygotowania"] == "15 min" and rec["siedziba"]["typ_ustalono"] == "reguly"
    assert rec["pochodzenie"] == ["Tradycja brytyjska"]


def test_unclear_type_asks_llm_only_for_classification():
    recipe = dict(RECIPE, title="Evening ritual", description=None,
                  steps=["Pour boiling water over the flowers."])
    llm = FakeLLM({"typ": "medyczne_wewnetrzne_napar", "ziola_kluczowe": True})
    rec, _ = asyncio.run(ft.fast_translate_recipe(FakeMT(), llm, recipe, META, PLANTS, set()))
    assert llm.calls == 1 and rec["typ"] == "medyczne_wewnetrzne_napar"
    assert rec["siedziba"]["typ_ustalono"] == "llm-klasyfikacja"


def test_broken_mt_or_raw_article_goes_to_llm():
    try:
        asyncio.run(ft.fast_translate_recipe(FakeMT(broken=True), None, RECIPE, META, PLANTS, set()))
        raise AssertionError("zepsute tlumaczenie powinno trafic do LLM")
    except MTQualityError:
        pass
    assert not ft.can_fast_translate({"title": "x", "content": "...", "article": True}, "en")
    assert not ft.can_fast_translate(RECIPE, None) and ft.can_fast_translate(RECIPE, "de")
