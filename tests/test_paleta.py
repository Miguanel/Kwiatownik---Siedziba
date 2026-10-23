"""Paleta Barw (app/knowledge/paleta.py): dopasowanie ziol do roslin, wybor informacji o wlosach, zapis pliku."""
import json

from app.knowledge import paleta as pal

HERBS = [
    {"id": "henna", "nazwa": "Henna", "lat": "Lawsonia inermis", "karta": None, "dziala": "Od jasnej miedzi po rdzawa czerwien."},
    {"id": "cassia", "nazwa": "Cassia", "lat": "Senna italica (Cassia obovata)", "karta": None},
    {"id": "rzewien", "nazwa": "Rzewien", "lat": "Rheum spp.", "karta": None},
    {"id": "malwa", "nazwa": "Malwa", "lat": "Alcea rosea 'Nigra'", "karta": None},
    {"id": "katam", "nazwa": "Katam", "lat": "Buxus dioica (wg sprzedawców)", "karta": None},
    {"id": "marzanna", "nazwa": "Marzanna", "lat": "Rubia tinctorum", "karta": "marzanna_barwierska"},
]


def test_binomials():
    assert pal.binomials("Senna italica (Cassia obovata)") == {"senna italica", "cassia obovata"}
    assert pal.binomials("Rheum spp.") == {"rheum"}
    assert pal.binomials("Alcea rosea 'Nigra'") == {"alcea rosea"}
    assert pal.binomials("Buxus dioica (wg sprzedawców)") == {"buxus dioica"}


def test_match_plants():
    plants = [("henna_x", "Lawsonia inermis L."), ("cassia_x", "Cassia obovata Collad."),
              ("rabarbar", "Rheum rhabarbarum"), ("malwa_x", "Alcea rosea"), ("marzanna_barwierska", None),
              ("lipa", "Tilia cordata")]
    m = pal.match_plants(HERBS, plants)
    assert m["henna"] == ["henna_x"]
    assert m["cassia"] == ["cassia_x"]
    assert m["rzewien"] == ["rabarbar"]          # rodzaj
    assert m["malwa"] == ["malwa_x"]
    assert m["katam"] == []
    assert m["marzanna"] == ["marzanna_barwierska"]   # przez pole "karta"


def _f(i, text, section="barwienie", status="verified", url="https://example.org/a"):
    return {"id": i, "section": section, "text": text, "source_url": url, "source_name": None, "language": "en",
            "status": status}


def test_fits_only_hair_and_verified():
    assert pal.fits(_f(1, "Pasta z liści henny barwi siwe włosy na intensywny rudy kolor."))
    assert not pal.fits(_f(2, "Liście henny barwią wełnę i jedwab na pomarańczowo."))            # tkaniny
    assert not pal.fits(_f(3, "Pasta z liści henny barwi siwe włosy na rudo.", status="new"))     # niezweryfikowane
    assert not pal.fits(_f(4, "Pasta z liści henny barwi siwe włosy na rudo.", url="ftp://x"))  # zrodlo
    assert not pal.fits(_f(5, "Krótko o włosach."))                                              # za krotkie


def test_build_dedup_and_known_text():
    facts = {"henna_x": [
        _f(1, "Pasta z liści henny barwi siwe włosy na intensywny rudy kolor.", url="https://a.org/1"),
        _f(2, "Pasta z liści henny barwi siwe włosy na intensywnie rudy kolor!", url="https://b.org/2"),  # powtorka
        _f(3, "Od jasnej miedzi po rdzawa czerwien na włosach.", url="https://c.org/3"),               # jest w palecie
        _f(4, "Henna nałożona na skórę głowy może wywołać podrażnienie u osób wrażliwych.", section="bezpieczenstwo",
           url="https://d.org/4"),
    ]}
    data = pal.build(HERBS, {"henna": ["henna_x"]}, facts, "2026-10-23T12:00:00+00:00")
    u = data["uzupelnienia"]["henna"]
    assert len(u["uwagi"]) == 1 and len(u["ostrzezenia"]) == 1
    assert [z["url"] for z in u["zrodla"]] == ["https://a.org/1", "https://d.org/4"]
    assert "cassia" in data["bez_rosliny_w_siedzibie"]


def test_write_if_changed_ignores_date(tmp_path):
    (tmp_path / "static" / "json").mkdir(parents=True)
    d1 = pal.build(HERBS, {}, {}, "2026-10-23T12:00:00+00:00")
    assert pal.write_if_changed(tmp_path, d1) is True
    d2 = dict(d1, zaktualizowano="2026-10-24T12:00:00+00:00")
    assert pal.write_if_changed(tmp_path, d2) is False
    saved = json.loads((tmp_path / pal.WYJSCIE).read_text(encoding="utf-8"))
    assert saved["zaktualizowano"].startswith("2026-10-23")
