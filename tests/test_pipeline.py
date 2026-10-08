import asyncio
import json

import pytest

from app.llm.base import InvalidOutputError
from app.pipeline.catalog import load_kwiatownik_recipes, load_plants
from app.pipeline.matching import Candidate, find_match, fingerprint, score, verdict
from app.pipeline.translate import normalize_record, slugify, translate_recipe

PLANTS = {"mniszek_lekarski": "Mniszek lekarski (Taraxacum officinale)", "bez_czarny": "Bez czarny (Sambucus nigra)"}

LLM_ANSWER = {
    "tytul": "Syrop z kwiatów mniszka", "opis": "Gęsty syrop.", "typ": "kulinarno_medyczne_syrop", "metoda": "syrop",
    "roslina": "Mniszek lekarski", "roslina_id": "mniszek_lekarski",
    "skladniki": [{"nazwa": "Kwiaty mniszka", "ilosc": "300 szt.", "czesc_rosliny": "kwiat", "link_id": "mniszek_lekarski"},
                  {"nazwa": "Cukier", "ilosc": "1 kg", "link_id": "cukier_wymyslony"}, {"nazwa": ""}],
    "sposob_przygotowania": ["1. Zalać kwiaty wodą.", "Krok 2: Gotować 30 minut.", "Przecedzić i dodać cukier."],
    "porcje": None, "stosowanie_i_dawkowanie": None, "tagi": ["syrop", 5]}


class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    async def complete(self, prompt, system=None, json_mode=False):
        self.prompts.append((system, prompt))
        text = self.answers.pop(0)

        class R:
            provider, model = "fake", "m"
        r = R()
        r.text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        return r


def test_slugify_polish():
    assert slugify("Syrop z kwiatów mniszka – łatwy!") == "syrop_z_kwiatow_mniszka_latwy"


def test_translate_normalizes_and_validates():
    llm = FakeLLM([LLM_ANSWER])
    meta = {"item_id": 7, "title": "Dandelion syrup", "language": "uk", "country": None}
    rec, model = asyncio.run(translate_recipe(llm, {"title": "Dandelion syrup", "ingredients": ["x"], "steps": ["y"]},
                                              None, meta, PLANTS, {"syrop_z_kwiatow_mniszka"}))
    assert model == "fake/m" and rec["id"] == "syrop_z_kwiatow_mniszka_s7"          # kolizja id -> sufiks
    assert rec["slug"] == "mniszek_lekarski" and rec["pochodzenie"] == ["Tradycja ukraińska"]
    assert [s.get("link_id") for s in rec["skladniki"]] == ["mniszek_lekarski", None]   # nieznane id -> brak pola, puste pominiete
    assert "link_id" not in rec["skladniki"][1]
    assert rec["sposob_przygotowania"] == ["Krok 1: Zalać kwiaty wodą.", "Krok 2: Gotować 30 minut.",
                                           "Krok 3: Przecedzić i dodać cukier."]
    assert rec["tagi"] == ["syrop"] and "porcje" not in rec and rec["siedziba"]["jezyk_oryginalu"] == "uk"
    system, prompt = llm.prompts[0]
    assert "NIE dodawaj" in system and "mniszek_lekarski: Mniszek lekarski" in prompt


def test_translate_rejects_incomplete():
    with pytest.raises(InvalidOutputError):
        normalize_record({"tytul": "X", "skladniki": [], "sposob_przygotowania": ["a"]}, {}, PLANTS, set())


def test_matching_polish_inflection_and_llm_judge():
    a = fingerprint("Syrop z kwiatów mniszka lekarskiego", ["kwiaty mniszka", "cukier", "cytryna", "woda"])
    b = fingerprint("Syrop z mniszka (miód mniszkowy)", ["300 kwiatów mniszka", "1 kg cukru", "2 cytryny", "1 litr wody"])
    assert verdict(*score(a, b)) == "maybe"
    cands = [Candidate("kwiatownik", "przepisy_kulinarne.json#nalewka", "Nalewka z korzenia mniszka", ["korzeń mniszka", "wódka"]),
             Candidate("kwiatownik", "przepisy_kulinarne.json#miod", "Syrop z mniszka (miód mniszkowy)",
                       ["300 kwiatów mniszka", "1 kg cukru", "2 cytryny", "1 litr wody"])]
    # bez LLM: watpliwy przypadek nie jest oznaczany
    assert asyncio.run(find_match("Syrop z kwiatów mniszka lekarskiego",
                                  ["kwiaty mniszka", "cukier", "cytryna", "woda"], cands)) is None
    llm = FakeLLM([{"same": True, "reason": "ten sam syrop"}])
    m = asyncio.run(find_match("Syrop z kwiatów mniszka lekarskiego", ["kwiaty mniszka", "cukier", "cytryna", "woda"],
                               cands, llm))
    assert m and m.candidate.ref.endswith("#miod") and m.how == "llm" and len(llm.prompts) == 1


def test_matching_clear_duplicate_without_llm():
    cands = [Candidate("siedziba", "12", "Nalewka z kwiatów czarnego bzu", ["kwiaty czarnego bzu", "wódka", "cukier"])]
    m = asyncio.run(find_match("Nalewka z kwiatów bzu czarnego", ["kwiaty bzu czarnego", "wódka 40%", "cukier"], cands))
    assert m and m.how == "podobienstwo"


def test_catalog_loading(tmp_path):
    (tmp_path / "przepisy").mkdir()
    (tmp_path / "plants" / "drzewa").mkdir(parents=True)
    (tmp_path / "przepisy" / "a.json").write_text(json.dumps([{"id": "x", "tytul": "Napar", "skladniki": [{"nazwa": "Mięta"}]}]), encoding="utf-8")
    (tmp_path / "przepisy" / "b.json").write_text(json.dumps({"przepisy": [{"tytul": "Syrop", "skladniki": ["Cukier"]}]}), encoding="utf-8")
    (tmp_path / "przepisy" / "wzorzec_przepisu.json").write_text(json.dumps({"tytul": "W"}), encoding="utf-8")
    (tmp_path / "przepisy" / "siedziba_przepisy.json").write_text(json.dumps([{"tytul": "Nasz"}]), encoding="utf-8")
    (tmp_path / "plants" / "drzewa" / "dab.json").write_text(json.dumps({"id": "dab", "nazwa_pl": "Dąb", "nazwa_lat": "Quercus"}), encoding="utf-8")
    recs = load_kwiatownik_recipes(tmp_path / "przepisy", exclude={"siedziba_przepisy.json"})
    assert [(r.title, r.ingredients) for r in recs] == [("Napar", ["Mięta"]), ("Syrop", ["Cukier"])]
    assert load_plants(tmp_path / "plants") == {"dab": "Dąb (Quercus)"}


def test_enrichment_keeps_source_fields_and_adds_disclaimer():
    from app.pipeline.enrich import NO_DOSAGE, enrich_recipe, parse_systems
    rec = {"tytul": "Nalewka z dziurawca", "typ": "medyczne_wewnetrzne_nalewka",
           "skladniki": [{"nazwa": "Ziele dziurawca", "ilosc": "50 g"}, {"nazwa": "Wódka", "ilosc": "250 ml"}],
           "sposob_przygotowania": ["Krok 1: Zalać ziele wódką."]}
    answer = {"mechanizm_tworzenia": ["Ekstrakcja alkoholowa", 7],
              "klasyfikacja_dzialania": {"skalowanie_toksykologiczne": "Średnia moc.",
                                         "matryca_wu_xing": {"zywiol_leczony": "Drewno (Wątroba)", "x": "y"},
                                         "humory_galena": {"jakosc": "ciepła i sucha", "stopien": "II"}},
              "skladniki": [{"nazwa": "ziele dziurawca", "filar": "1_bazowy", "smak_ajurweda": "gorzki",
                             "ilosc": "999 kg"}],
              "stosowanie_i_dawkowanie": {"dawkowanie_standardowe": "20 kropli", "skalowanie_pacjenta": "Nie dla dzieci."},
              "bezpieczenstwo_i_interakcje": {"ostrzezenia": "Fotouczulenie.", "interakcje_z_lekami": "Antykoncepcja."},
              "wskazowki_tradycyjne": ["Zbierać w dzień św. Jana."]}
    llm = FakeLLM([answer])
    out, model = asyncio.run(enrich_recipe(llm, rec, parse_systems("wu_xing,humory,piec_filarow,ajurweda"), "raw text"))
    assert model == "fake/m"
    assert out["skladniki"][0] == {"nazwa": "Ziele dziurawca", "ilosc": "50 g"}  # bez filaru i smaku ajurwedy
    assert out["sposob_przygotowania"] == rec["sposob_przygotowania"]
    assert out["stosowanie_i_dawkowanie"]["dawkowanie_standardowe"] == NO_DOSAGE          # nie z LLM
    assert out["stosowanie_i_dawkowanie"]["skalowanie_pacjenta"] == "Nie dla dzieci."
    assert out["klasyfikacja_dzialania"]["matryca_wu_xing"] == {"zywiol_leczony": "Drewno (Wątroba)"}
    assert out["klasyfikacja_dzialania"]["humory_galena"]["stopien"] == "II"
    assert out["mechanizm_tworzenia"] == ["Ekstrakcja alkoholowa"]
    assert out["opracowanie"]["systemy"] == ["wu_xing", "humory"]          # ajurweda i 5 filarow nie istnieja
    system, prompt = llm.prompts[0]
    assert "Nie wymyslaj dawkowania" in system and "Piec Przemian" not in prompt and "Pięć Przemian" in prompt


def test_translate_type_mapping():
    base = {"tytul": "X", "skladniki": [{"nazwa": "a"}], "sposob_przygotowania": ["b"]}
    assert normalize_record(dict(base, typ="barwierskie_wlosy"), {}, PLANTS, set())["typ"] == "barwierskie_wlosy"
    assert normalize_record(dict(base, typ="medyczne_zewnetrzne_czopki"), {}, PLANTS, set())["typ"] == "medyczne_zewnetrzne_czopki"
    assert normalize_record(dict(base, typ="kulinarny"), {}, PLANTS, set())["typ"] == "kulinarne"
    assert normalize_record(base, {}, PLANTS, set())["typ"] == "inne"


def test_skip_reason_culinary_and_non_herbal():
    from app.pipeline.translate import skip_reason
    base = {"tytul": "X", "skladniki": [{"nazwa": "glinka"}], "sposob_przygotowania": ["b"]}
    clay = normalize_record(dict(base, typ="kosmetyczne_skora", ziola_kluczowe=False), {}, PLANTS, set())
    herb = normalize_record(dict(base, typ="medyczne_wewnetrzne_napar", ziola_kluczowe=True), {}, PLANTS, set())
    assert "kluczowym" in skip_reason(clay, True, True) and skip_reason(clay, True, False) is None
    assert skip_reason(herb, True, True) is None and "ziola_kluczowe" not in herb
    assert "kulinarny" in skip_reason({"typ": "kulinarne"}, True, True)


def test_convert_kwiatownik1_archive_recipe():
    from app.pipeline.archive import convert_k1_recipe, guess_type
    plants = {"skrzyp_polny": "Skrzyp polny (Equisetum arvense)", "pokrzywa_zwyczajna": "Pokrzywa zwyczajna"}
    r = {"tytul": "napar z ziele skrzypu", "typ": "medyczny", "metoda": "",
         "skladniki": ["1 łyżka suszonego ziela skrzypu polnego", "250 ml wrzątku"],
         "sposob_przygotowania": "Ziele zalać wrzątkiem, parzyć pod przykryciem 10–15 minut. Przecedzić przez sitko i pić ciepłe, najlepiej rano i wieczorem przez trzy tygodnie.",
         "dawkowanie": "1 filiżanka 2 razy dziennie", "uwagi": "", "cechy": ["gorzki smak"],
         "wlasciwosci": ["moczopędny"], "zrodla": ["https://www.herbazone.pl/x", "ESCOP"], "roslina": "Skrzyp polny",
         "slug": "skrzyp_polny"}
    rec = convert_k1_recipe(r, plants, {"napar_z_ziele_skrzypu"}, "przepisy_medyczne_global_0")
    assert rec["tytul"] == "Napar z ziele skrzypu" and rec["typ"] == "medyczne_wewnetrzne_napar"
    assert rec["id"] != "napar_z_ziele_skrzypu" and rec["slug"] == "skrzyp_polny"
    assert rec["skladniki"][0]["link_id"] == "skrzyp_polny" and "link_id" not in rec["skladniki"][1]
    assert rec["sposob_przygotowania"][0].startswith("Krok 1: Ziele") and len(rec["sposob_przygotowania"]) == 2
    assert rec["stosowanie_i_dawkowanie"] == {"dawkowanie_standardowe": "1 filiżanka 2 razy dziennie"}
    assert rec["opis"] == "Działanie: moczopędny" and rec["uwagi"] == "gorzki smak"
    assert rec["zrodla"] == ["https://www.herbazone.pl/x", "ESCOP"]
    assert guess_type("płukanka", "Płukanka do włosów z pokrzywy") == "kosmetyczne_wlosy"
    assert guess_type("maść_na_smalcu", "Maść") == "medyczne_zewnetrzne_masc"
    assert convert_k1_recipe({"tytul": "x", "skladniki": [], "sposob_przygotowania": "a"}, plants, set(), "r") is None


def test_with_sources_keeps_own_bibliography():
    from app.exporters.kwiatownik import with_sources
    out = with_sources({"tytul": "X", "zrodla": ["ESCOP"]},
                       [{"url": "archiwum://kwiatownik1/a/1", "nazwa": "arch"}, {"url": "https://a.pl/1", "nazwa": "a"}])
    assert [z if isinstance(z, str) else z["url"] for z in out["zrodla"]] == ["https://a.pl/1", "ESCOP"]


def test_translate_splits_article_with_many_recipes_and_detects_no_recipe():
    from app.pipeline.translate import NoRecipeError
    second = {"tytul": "Napar z mięty", "typ": "medyczne_wewnetrzne_napar", "skladniki": [{"nazwa": "Mięta"}],
              "sposob_przygotowania": ["Zalać."]}
    answer = dict(LLM_ANSWER, dodatkowe_przepisy=[second, {"tytul": "Pusty"}, "zly"])
    meta = {"item_id": 9, "title": "5 naparow", "language": "uk", "country": None}
    rec, _ = asyncio.run(translate_recipe(FakeLLM([answer]), None, "tekst", meta, PLANTS, set()))
    extras = rec["_dodatkowe"]
    assert [e["tytul"] for e in extras] == ["Napar z mięty"] and extras[0]["siedziba"]["item_id"] == 9
    assert extras[0]["id"] != rec["id"]
    with pytest.raises(NoRecipeError):
        asyncio.run(translate_recipe(FakeLLM([{"brak_przepisu": True}]), None, "lista", meta, PLANTS, set()))
    with pytest.raises(NoRecipeError):   # niekompletny przepis tez = brak konkretnego przepisu
        asyncio.run(translate_recipe(FakeLLM([{"tytul": "X", "skladniki": []}]), None, "t", meta, PLANTS, set()))


def test_strip_invented_cleans_old_records():
    from app.pipeline.enrich import strip_invented
    rec = {"tytul": "Balsam", "mechanizm_tworzenia": ["Maceracja", "Wzorzec 5 filarów"],
           "klasyfikacja_dzialania": {"ajurweda": {"dosze": "Vata"}, "matryca_wu_xing": {"zywiol_leczony": "Ogień"}},
           "skladniki": [{"nazwa": "Nagietek", "filar": "3_minister", "tropizm_organowy": "Skóra",
                          "smak_ajurweda": "Słodki", "link_id": "nagietek_lekarski"},
                         {"nazwa": "Wosk", "link_id": None}, "sól"],
           "opracowanie": {"systemy": ["wu_xing", "ajurweda", "piec_filarow"]}}
    out = strip_invented(rec)
    assert out["skladniki"] == [{"nazwa": "Nagietek", "link_id": "nagietek_lekarski"}, {"nazwa": "Wosk"}, "sól"]
    assert out["klasyfikacja_dzialania"] == {"matryca_wu_xing": {"zywiol_leczony": "Ogień"}}
    assert out["mechanizm_tworzenia"] == ["Maceracja"] and out["opracowanie"]["systemy"] == ["wu_xing"]
    assert rec["skladniki"][0]["filar"] == "3_minister"          # oryginal nietkniety


def test_recipe_store_reads_newest_file_of_each_series(tmp_path):
    from app.exporters import store
    d = tmp_path / "przepisy"
    d.mkdir()
    for name, title in [("siedziba_przepisy.json", "Stary eksport"), ("siedziba_przepisy_2026-10-01.json", "Z 1 pazdziernika"),
                        ("siedziba_przepisy_2026-10-05.json", "Najnowszy"), ("reczne.json", "Reczny"),
                        ("wzorzec_przepisu.json", "Wzorzec")]:
        (d / name).write_text(json.dumps([{"tytul": title, "skladniki": ["x"]}]), encoding="utf-8")
    assert [f.name for f in store.current_files(d)] == ["reczne.json", "siedziba_przepisy_2026-10-05.json"]
    assert store.parse_name("siedziba_przepisy_2026-10-05_1430.json") == ("siedziba_przepisy", "2026-10-05_1430")
    assert store.series_of("siedziba_przepisy.json") == "siedziba_przepisy"
    assert [r.title for r in load_kwiatownik_recipes(d)] == ["Reczny", "Najnowszy"]
    assert [r.title for r in load_kwiatownik_recipes(d, exclude={"siedziba_przepisy.json"})] == ["Reczny"]
    used = {f["name"]: f["used"] for f in store.all_files(d)}
    assert used == {"reczne.json": True, "siedziba_przepisy.json": False, "siedziba_przepisy_2026-10-01.json": False,
                    "siedziba_przepisy_2026-10-05.json": True}
