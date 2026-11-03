"""Etap 4 (2026-11-02): opracowanie przepisu bez LLM - zadnych halucynacji, tylko zrodlo i karty roslin."""
from app.pipeline.enrich import NO_DOSAGE
from app.pipeline.opracowanie import ALCOHOL_NOTE, MORDANT_NOTE, NO_WARNINGS, mechanism_for, opracuj

ST_JOHN = {"id": "dziurawiec_zwyczajny", "nazwa_pl": "Dziurawiec zwyczajny",
           "ostrzezenia": "Unikać opalania podczas stosowania.",
           "interakcje": "Induktor CYP3A4: osłabia antykoncepcję hormonalną i leki przeciwzakrzepowe.",
           "czesci_rosliny": {"ziele": {"czas_zbioru": "Czerwiec - Sierpień (w pełni kwitnienia).",
                                         "ostrzezenia": "Świeże ziele silniej uczula na słońce."}},
           "profil_energetyczny": {"smak": "Gorzki, cierpki", "termika": "Ciepła", "wilgotnosc": "Osuszająca"}}
PLANTS = {"dziurawiec_zwyczajny": ST_JOHN}

OLD_LLM = {  # przepis z dawnym opracowaniem przez LLM (tak wygladal eksport 2026-10-30)
    "tytul": "Nalewka z dziurawca", "typ": "medyczne_wewnetrzne_nalewka", "roslina_id": "dziurawiec_zwyczajny",
    "skladniki": [{"nazwa": "Ziele dziurawca", "ilosc": "50 g", "link_id": "dziurawiec_zwyczajny",
                   "czesc_rosliny": "ziele"}, {"nazwa": "Wódka 40%", "ilosc": "250 ml"}],
    "sposob_przygotowania": ["Zalać ziele wódką.", "Odstawić na 14 dni."],
    "mechanizm_tworzenia": ["Ekstrakcja alkoholowa", "Wzorzec 5 filarów"],
    "klasyfikacja_dzialania": {"skalowanie_toksykologiczne": "Średnia moc.",
                               "doktryna_sygnatur": {"sygnatura": "Wiatr", "interpretacja": "..."},
                               "alchemia_spageryczna": {"zasada": "Separacja"}, "triada_kampo": {"cel_glowny": "Ki"}},
    "wymogi_szamanskie_i_czasowe": {"chronoterapia": "rano", "astrologia_zbioru": "wiosną",
                                    "dieta_i_zakazy": "Unikać tłustych potraw."},
    "wskazowki_tradycyjne": ["Zbierać w noc świętojańską."],
    "wlasciwosci_fizyczne": {"konsystencja_i_slady": "czerwona"},
    "bezpieczenstwo_i_interakcje": {"ostrzezenia": "Wymyślone ostrzeżenie modelu."},
    "stosowanie_i_dawkowanie": {"okolicznosci_stosowania": "Zgodnie z chronoterapią, w porze płuc (godz. 3–5).",
                                "skalowanie_pacjenta": "Ostrożność: dzieci, ciąża.",
                                "dawkowanie_standardowe": "20 kropli 3 razy dziennie"},
    "opracowanie": {"systemy": ["kampo"], "model": "groq/x"},
    "uwagi": "Z tekstu źródła: nie stosować w ciąży.",
}


def test_old_llm_fields_are_gone_and_safety_comes_from_plant_cards():
    out = opracuj(OLD_LLM, PLANTS.get, source_text="Take 20 drops 3 times a day. 50 g herb, 250 ml vodka")
    for k in ("klasyfikacja_dzialania", "wymogi_szamanskie_i_czasowe", "wskazowki_tradycyjne", "wlasciwosci_fizyczne"):
        assert k not in out
    assert out["mechanizm_tworzenia"] == ["Nalewka (maceracja w alkoholu)"]
    warn = out["bezpieczenstwo_i_interakcje"]
    assert "Wymyślone" not in warn["ostrzezenia"] and "Dziurawiec zwyczajny: Unikać opalania" in warn["ostrzezenia"]
    assert "Świeże ziele silniej uczula" in warn["ostrzezenia"] and ALCOHOL_NOTE in warn["ostrzezenia"]
    assert "CYP3A4" in warn["interakcje_z_lekami"]
    sd = out["stosowanie_i_dawkowanie"]
    assert sd == {"dawkowanie_standardowe": "20 kropli 3 razy dziennie"}      # chronoterapia i "skalowanie" z LLM - out
    assert out["zbior"]["kiedy"].startswith("Czerwiec") and out["zbior"]["czesc"] == "ziele"
    assert out["uwagi"] == OLD_LLM["uwagi"] and out["sposob_przygotowania"] == OLD_LLM["sposob_przygotowania"]
    assert out["opracowanie"]["jak"].startswith("bez LLM") and out["opracowanie"]["rosliny"] == ["dziurawiec_zwyczajny"]
    tip = out["ciekawostka_tradycyjna"]["wu_xing"]["tekst"]
    assert "Ognia" in tip and "zegara narządów pora serca to godz. 11–13" in tip


def test_dosage_with_numbers_not_in_source_is_dropped():
    out = opracuj(OLD_LLM, PLANTS.get, source_text="Pour vodka over the herb and wait two weeks.")
    assert out["stosowanie_i_dawkowanie"]["dawkowanie_standardowe"] == NO_DOSAGE
    no_src = opracuj(OLD_LLM, PLANTS.get, source_text=None)                  # brak tekstu zrodla - nie da sie sprawdzic
    assert no_src["stosowanie_i_dawkowanie"]["dawkowanie_standardowe"] == "20 kropli 3 razy dziennie"


def test_dye_recipe_gets_mordant_rule_and_unknown_plant_generic_warning():
    dye = {"tytul": "Barwienie wełny wrotyczem", "typ": "barwierskie_tkaniny",
           "skladniki": [{"nazwa": "Kwiaty wrotyczu", "link_id": "wrotycz"}, {"nazwa": "Ałun", "ilosc": "10 g"}],
           "sposob_przygotowania": ["Zaprawić wełnę ałunem.", "Gotować w wywarze."]}
    out = opracuj(dye, {}.get)
    assert out["mechanizm_tworzenia"] == ["Barwienie roślinne"] and MORDANT_NOTE in out["bezpieczenstwo_i_interakcje"]["ostrzezenia"]
    assert "ciekawostka_tradycyjna" not in out and "zbior" not in out
    plain = opracuj({"tytul": "Krem", "typ": "kosmetyczne_skora", "skladniki": [{"nazwa": "Masło shea"}],
                     "sposob_przygotowania": ["Wymieszać."]}, {}.get)
    assert plain["bezpieczenstwo_i_interakcje"] == {"ostrzezenia": NO_WARNINGS} and "mechanizm_tworzenia" not in plain


def test_opracuj_is_idempotent_and_mechanism_table():
    once = opracuj(OLD_LLM, PLANTS.get, "20 3 50 250", today="2026-11-02")
    assert opracuj(once, PLANTS.get, "20 3 50 250", today="2026-11-02") == once
    assert mechanism_for("medyczne_wewnetrzne_miod_pitny") == ["Miód pitny (fermentacja)"]
    assert mechanism_for("kulinarno_medyczne_plyn") is None and mechanism_for(None) is None
