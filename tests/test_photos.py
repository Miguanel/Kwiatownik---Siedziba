"""Zdjecia roslin z Wikipedii / Commons: wybor, licencje, inny gatunek odrzucony, zapis w pliku rosliny."""
import asyncio
import json

import httpx

from app.knowledge import photos, plantfile
from app.knowledge.wiki import WikiClient

UP = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/{0}/640px-{0}"


def _ii(name, desc="", cats="", lic="CC BY-SA 4.0", width=2000, mime="image/jpeg", artist='<a href="x">Jan Kowalski</a>'):
    f = name.replace(" ", "_")
    return {"title": f"File:{name}", "imageinfo": [{
        "mime": mime, "width": width, "thumburl": UP.format(f),
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{f}",
        "extmetadata": {"Artist": {"value": artist}, "LicenseShortName": {"value": lic},
                        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
                        "ImageDescription": {"value": desc}, "Categories": {"value": cats}}}]}


FILES = {
    "Heracleum sphondylium habitus.jpg": _ii("Heracleum sphondylium habitus.jpg"),
    "Barszcz kwiaty.jpg": _ii("Barszcz kwiaty.jpg", desc="Heracleum sphondylium - flowers"),
    "Heracleum mantegazzianum flowers.jpg": _ii("Heracleum mantegazzianum flowers.jpg"),     # inny gatunek
    "Heracleum sphondylium leaf.jpg": _ii("Heracleum sphondylium leaf.jpg", lic="All rights reserved"),
    "Heracleum sphondylium fruits.jpg": _ii("Heracleum sphondylium fruits.jpg", width=200),  # za male
    "Hs seeds.jpg": _ii("Hs seeds.jpg", cats="Heracleum sphondylium|Seeds"),
    "Heracleum sphondylium blatt.jpg": _ii("Heracleum sphondylium blatt.jpg", lic="CC0"),
    "Heracleum distribution map.png": _ii("Heracleum distribution map.png", mime="image/png"),
    "Kohler Heracleum sphondylium.jpg": _ii("Kohler Heracleum sphondylium.jpg", lic="Public domain"),
}


def handler(request: httpx.Request) -> httpx.Response:
    p = dict(request.url.params)
    if request.url.host == "pl.wikipedia.org" and p.get("prop") == "images":
        names = ["Plik:Heracleum sphondylium habitus.jpg", "Plik:Barszcz kwiaty.jpg",
                 "Plik:Heracleum mantegazzianum flowers.jpg", "Plik:Heracleum sphondylium leaf.jpg",
                 "Plik:Heracleum sphondylium fruits.jpg", "Plik:Heracleum distribution map.png",
                 "Plik:Commons-logo.svg", "Plik:Kohler Heracleum sphondylium.jpg"]
        return httpx.Response(200, json={"query": {"pages": {"1": {"images": [{"title": n} for n in names]}}}})
    if request.url.host == "commons.wikimedia.org" and p.get("list") == "categorymembers":
        return httpx.Response(200, json={"query": {"categorymembers": [
            {"title": "File:Hs seeds.jpg"}, {"title": "File:Heracleum sphondylium blatt.jpg"}]}})
    if request.url.host == "commons.wikimedia.org" and p.get("prop") == "imageinfo":
        titles = p["titles"].split("|")
        pages = {str(i): FILES[t[5:]] for i, t in enumerate(titles) if t[5:] in FILES}
        return httpx.Response(200, json={"query": {"pages": pages}})
    return httpx.Response(404)


def _run(coro_fn):
    async def run():
        wc = WikiClient("T", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), delay_s=0)
        try:
            return await coro_fn(wc)
        finally:
            await wc.aclose()
    return asyncio.run(run())


def test_find_photos_picks_one_per_label_with_free_license():
    pics = _run(lambda wc: photos.find_photos(wc, "Heracleum sphondylium", "Heracleum sphondylium habitus.jpg",
                                              "Heracleum sphondylium", {"pl": "Barszcz zwyczajny"}, ["pl"]))
    by = {p.podpis: p for p in pics}
    assert list(by) == ["Pokrój", "Kwiaty", "Liście", "Nasiona", "Rycina"]   # max 5, kolejnosc Kwiatownika
    assert by["Kwiaty"].plik.endswith("Barszcz_kwiaty.jpg")                   # nie H. mantegazzianum
    assert by["Liście"].licencja == "CC0"                                     # "All rights reserved" odrzucone
    assert by["Pokrój"].autor == "Jan Kowalski" and by["Pokrój"].url.startswith("https://upload.wikimedia.org/")
    assert all("map" not in p.url for p in pics)


def test_labels_and_filters():
    assert photos.label_for("Plantago lanceolata leaves") == "Liście"
    assert photos.label_for("Köhler's Medizinal-Pflanzen") == "Rycina"
    assert photos.label_for("Orange flower of Calendula") == "Kwiaty"
    assert photos.excluded("Heracleum map.png") and not photos.excluded("Orange flower.jpg")
    assert photos.about_taxon("Heracleum sphondylium", "H. sphondylium in Poland")
    assert not photos.about_taxon("Heracleum sphondylium", "Heracleum mantegazzianum")
    assert photos.file_title("Plik:Abc_d.jpg") == "File:Abc d.jpg" and photos.file_title("File:x.svg") is None


def _pic(label, name="Heracleum_sphondylium.jpg"):
    return {"podpis": label, "url": UP.format(name), "plik": f"https://commons.wikimedia.org/wiki/File:{name}",
            "autor": "Jan", "licencja": "CC BY-SA 4.0", "licencja_url": "https://creativecommons.org/licenses/by-sa/4.0",
            "zrodlo": "Wikimedia Commons"}


def test_gallery_written_only_when_plant_has_no_manual_photos():
    original = {"id": "barszcz", "nazwa_pl": "Barszcz", "url": None, "opis": "Ręczny."}
    cand = plantfile.with_photos(json.loads(json.dumps(original)), original, [_pic("Pokrój"), _pic("Kwiaty", "B.jpg")])
    assert cand["url"] == {"Pokrój": UP.format("Heracleum_sphondylium.jpg"), "Kwiaty": UP.format("B.jpg")}
    assert cand["zdjecia_wiki"]["w_url"] and plantfile.validate_candidate(original, cand, "barszcz") == []
    # nastepna aktualizacja: galeria nalezy juz do Siedziby - moze ja zmienic
    cand2 = plantfile.with_photos(json.loads(json.dumps(cand)), cand, [_pic("Liście", "C.jpg")])
    assert list(cand2["url"]) == ["Liście"] and plantfile.validate_candidate(cand, cand2, "barszcz") == []

    manual = {"id": "bez", "nazwa_pl": "Bez", "url": {"kwiaty": UP.format("S.jpg")}}
    credits = [dict(_pic("kwiaty", "S.jpg"), reczne=True)]
    cand3 = plantfile.with_photos(json.loads(json.dumps(manual)), manual, [_pic("Pokrój")], credits)
    assert cand3["url"] == manual["url"] and cand3["zdjecia_wiki"]["zdjecia"] == credits
    assert plantfile.validate_candidate(manual, cand3, "bez") == []
    hacked = json.loads(json.dumps(cand3))
    hacked["url"] = {"x": UP.format("Z.jpg")}
    assert any("reczn" in e for e in plantfile.validate_candidate(manual, hacked, "bez"))


def test_photo_validation_rejects_foreign_hosts_and_html():
    original = {"id": "b", "nazwa_pl": "B"}
    bad = plantfile.with_photos(dict(original), original, [dict(_pic("Pokrój"), url="https://evil.example/x.jpg",
                                                                   autor="<script>x</script>")])
    errors = plantfile.validate_candidate(original, bad, "b")
    assert any("upload.wikimedia" in e for e in errors) and any("HTML" in e for e in errors)


def test_credits_for_manual_wikimedia_urls():
    urls = {"Pokrój": UP.format("Heracleum_sphondylium_habitus.jpg").replace("/a/ab/", "/6/61/"),
            "Kwiatostan": "https://www.atlas-roslin.pl/x.jpg"}
    out = _run(lambda wc: photos.credits_for_urls(wc, urls))
    assert len(out) == 1 and out[0]["podpis"] == "Pokrój" and out[0]["autor"] == "Jan Kowalski" and out[0]["reczne"]


def test_thumbnails_from_new_thumb_host_are_accepted():
    """Wikimedia podaje miniatury z thumb.wikimedia.org - wczesniej kontrola przepuszczala tylko upload.* i
    roslina bez recznej galerii nie dostawala zadnego zdjecia."""
    from app.knowledge import photos
    url = "https://thumb.wikimedia.org/wikipedia/commons/thumb/5/58/20160906Plantago_lanceolata.jpg/960px-20160906Plantago_lanceolata.jpg"
    assert photos.UPLOAD_RE.match(url)
    assert photos.file_from_upload_url(url) == "File:20160906Plantago lanceolata.jpg"
    info = {"imageinfo": [{"mime": "image/jpeg", "width": 3000, "thumburl": url,
                           "descriptionurl": "https://commons.wikimedia.org/wiki/File:20160906Plantago_lanceolata.jpg",
                           "extmetadata": {"LicenseShortName": {"value": "CC0"}, "Artist": {"value": "Autor"}}}]}
    assert photos.photo_from_info("File:20160906Plantago lanceolata.jpg", info, "Pokrój").url == url


def test_other_species_in_category_is_skipped():
    from app.knowledge.photos import other_taxon
    t = "Plantago lanceolata"
    assert other_taxon("File:Echium vulgare (15664670844).jpg", "Plantago lanceolata|Echium vulgare in Slovenia", t)
    assert other_taxon("File:Philopedon plagiatum sur Plantago lanceolata 04.2025 (1).jpg",
                       "Plantago lanceolata|Philopedon plagiatum", t)
    assert other_taxon("File:Formica cunicularia gyne Trockenrasen 2026.jpg", "Plantago lanceolata|Formica cunicularia", t)
    assert not other_taxon("File:Babka lancetowata 3420.jpg", "Plantago lanceolata|Plantago lanceolata in Poland", t)
    assert not other_taxon("File:20160906Plantago lanceolata.jpg", "Plantago lanceolata|CC-Zero", t)
    assert not other_taxon("File:Plantago lanceolata flowers.jpg", "Plantago lanceolata", t)


def test_real_commons_response_2026_babka():
    """Odpowiedz Commons z 2026-10 (kategoria Plantago lanceolata): miniatury z thumb.wikimedia.org z ?utm_...;
    wczesniej kontrola odrzucala wszystkie i babka nie dostawala zadnego zdjecia."""
    import asyncio
    from app.knowledge import photos
    files = [("File:Babka lancetowata 3420.jpg", "CC BY-SA 3.0", "Babka_lancetowata",
              "Plantago lanceolata|Plantago lanceolata in Poland"),
             ("File:Echium vulgare (15664670844).jpg", "CC BY 2.0", "Bugloss sp.",
              "Plantago lanceolata|Echium vulgare in Slovenia"),
             ("File:20160906Plantago lanceolata.jpg", "CC0", "Spitzwegerich (Plantago lanceolata) in Hockenheim",
              "Plantago lanceolata|CC-Zero|Self-published work")]

    def page(t, lic, desc, cats):
        name = t[5:].replace(" ", "_")
        return {"title": t, "imageinfo": [{
            "mime": "image/jpeg", "width": 3000,
            "thumburl": f"https://thumb.wikimedia.org/wikipedia/commons/thumb/1/11/{name}/960px-{name}"
                        "?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail",
            "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{name}",
            "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": "<a href='x'>Autor</a>"},
                            "ImageDescription": {"value": desc}, "Categories": {"value": cats}}}]}

    class Wiki:
        async def _get(self, url, params):
            if params.get("list") == "categorymembers":
                return {"query": {"categorymembers": [{"title": f[0]} for f in files]}}
            return {"query": {"pages": {str(i): page(*f) for i, f in enumerate(files)}}}
    got = asyncio.run(photos.find_photos(Wiki(), "Plantago lanceolata", None, "Plantago lanceolata", {}, ["pl"]))
    assert got and all(p.url.startswith("https://thumb.wikimedia.org/") and "?" not in p.url for p in got)
    assert not any("Echium" in p.plik for p in got)
    assert got[0].autor == "Autor"
