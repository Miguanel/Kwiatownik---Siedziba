"""Dymek-ciekawostka Wu Xing / Galen liczony algorytmem z profilu energetycznego rosliny (bez LLM)."""
from app.pipeline.traditions import annotate, galen_quality, part_of, pick, segments, taste, tradition

NETTLE = {"id": "pokrzywa", "nazwa_pl": "Pokrzywa zwyczajna",
          "profil_energetyczny": {"smak": "Słony, lekko gorzki", "termika": "Chłodna", "wilgotnosc": "Osuszająca",
                                  "zywiol": "Ziemia i Powietrze"}}
ELDER = {"id": "bez", "nazwa_pl": "Bez czarny",
         "profil_energetyczny": {"smak": "Słodki, lekko gorzki (kwiaty), kwaśno-słodki (owoce)",
                                 "termika": "Chłodząca (kwiaty), neutralna ku ogrzewającej (owoce)",
                                 "wilgotnosc": "Osuszająca (napotna)"}}


def rec(typ="medyczne_wewnetrzne_napar", pid="pokrzywa", nazwa="Liście pokrzywy", **kw):
    return {"tytul": "Napar", "typ": typ, "roslina_id": pid,
            "skladniki": [{"nazwa": nazwa, "link_id": pid, **kw}, {"nazwa": "Woda"}]}


def test_segments_split_by_plant_part_and_keep_descriptions():
    assert segments("Słodki (sok), gorzki, lekko cierpki (liście, pąki)") == [
        ({"sok"}, "Słodki"), ({"liście", "pąki"}, "gorzki, lekko cierpki")]
    assert segments("Osuszająca (ściągająca)")[0][0] is None          # nawias bez czesci = opis
    assert part_of("Ziele dziurawca") == "ziele" and part_of("korzeń łopianu") == "korzeń"
    assert part_of("Kora dębu") == "kora" and part_of("Woda") is None


def test_pick_chooses_the_recipe_part_or_gives_up():
    t = ELDER["profil_energetyczny"]["smak"]
    assert pick(t, "kwiaty") == "Słodki, lekko gorzki"
    assert pick(t, "owoce") == "kwaśno-słodki"
    assert pick(t, "liście") is None                                    # brak tej czesci - nic nie zgadujemy
    assert pick("Chłodna", "korzeń") == "Chłodna"                       # wpis ogolny pasuje do kazdej czesci


def test_taste_prefers_strong_flavour():
    assert taste("lekko słodki, gorzki")[0] == "gorzki"
    assert taste("Kwaśno-słodki")[1] == "Drewno"
    assert taste("Cierpki, trawiasty") is None                          # cierpki nie ma odpowiednika w 5 smakach


def test_galen_quality_skips_neutral_and_contradictory_axes():
    assert galen_quality("Chłodna", "Osuszająca") == ["zimne", "suche"]
    assert galen_quality("Rozgrzewająca", "Nawilżająca (śluzowa)") == ["ciepłe", "wilgotne"]
    assert galen_quality("Neutralna ku chłodnej", "Neutralna") == []
    assert galen_quality("Gorąca", "Nawilżająca, ale jednocześnie osuszająca zastoje") == ["ciepłe"]


def test_tradition_texts_for_medicinal_recipe():
    c = tradition(rec(), NETTLE)
    assert c["wu_xing"]["przemiana"] == "Woda" and "z nerkami" in c["wu_xing"]["tekst"]
    assert c["galen"]["jakosc"] == "zimne i suche"
    assert "dolegliwościach ciepłych i wilgotnych" in c["galen"]["tekst"]
    assert c["czesc"] == "liście" and "nie porada medyczna" in c["uwaga"]


def test_only_medicinal_recipes_get_the_tooltip():
    for typ in ("barwierskie_tkaniny", "kosmetyczne_skora", "kulinarno_medyczne_nalewka", "kulinarne", ""):
        assert tradition(rec(typ=typ), NETTLE) is None
    assert tradition(rec(), {"id": "x"}) is None                        # roslina bez profilu


def test_part_specific_profile_uses_the_recipe_part():
    c = tradition(rec(pid="bez", nazwa="Kwiaty bzu czarnego"), ELDER)
    assert c["wu_xing"]["smak"] == "słodki" and c["galen"]["jakosc"] == "zimne i suche"
    c = tradition(rec(pid="bez", nazwa="Owoce bzu"), ELDER)
    assert c["wu_xing"]["smak"] == "kwaśny" and c["galen"]["jakosc"] == "suche"   # termika owocow neutralna


def test_annotate_replaces_old_field_and_never_touches_recipe():
    r = {**rec(), "ciekawostka_tradycyjna": {"stare": True}}
    out = annotate(r, {"pokrzywa": NETTLE}.get)
    assert out["ciekawostka_tradycyjna"]["roslina_id"] == "pokrzywa"
    assert out["skladniki"] == r["skladniki"] and r["ciekawostka_tradycyjna"] == {"stare": True}
    out = annotate({**rec(typ="barwierskie_tkaniny"), "ciekawostka_tradycyjna": {"x": 1}}, {"pokrzywa": NETTLE}.get)
    assert "ciekawostka_tradycyjna" not in out
    assert "ciekawostka_tradycyjna" not in annotate(rec(pid="nieznana"), {}.get)
