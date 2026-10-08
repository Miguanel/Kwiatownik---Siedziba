"""Scalanie wiedzy z sieci z rozdzialami pliku rosliny + pokrycie schematu + zapytania o luki (bez bazy)."""
import asyncio
import json

from app.knowledge import gaps, merge, plantfile, schema
from app.knowledge.extract import is_cjk, quote_in_text

PLANT = {
    "id": "dziurawiec_zwyczajny", "nazwa_pl": "Dziurawiec zwyczajny", "nazwa_lat": "Hypericum perforatum",
    "opis": "Wieloletnia bylina o złocistożółtych kwiatach, pospolita na łąkach.",
    "profil_energetyczny": {"smak": "Gorzki", "termika": "", "wilgotnosc": "Osuszająca", "zywiol": "Ogień", "opis": ""},
    "identyfikacja": {"cechy_kluczowe": ["Liście z przeświecającymi kropkami.", "Łodyga z dwiema listewkami."]},
    "czesci_rosliny": {"ziele": {"nazwa_surowca": "Herba Hyperici", "opis_botaniczny": "Górne części pędów.",
                                 "skladniki_aktywne": ["hiperycyna"], "wlasciwości": "Przeciwdepresyjne.",
                                 "czas_zbioru": "Czerwiec - Lipiec."}},
    "zastosowanie": {"medyczne": "Leczenie łagodnych stanów depresyjnych i lękowych.", "kulinarne": ""},
    "ciekawostki": ["Nazywany zielem świętojańskim."],
    "wiedza": {
        "wersja": 2,
        "zrodla": [{"nr": 1, "nazwa": "Wikipedia (pl)", "url": "https://pl.wikipedia.org/wiki/Dziurawiec"},
                   {"nr": 2, "nazwa": "Wikipedia (ja)", "url": "https://ja.wikipedia.org/wiki/x"}],
        "sekcje": [
            {"klucz": "zastosowanie_lecznicze", "punkty": [
                {"tekst": "Ziele dziurawca stosowano w medycynie ludowej na rany i oparzenia.", "czesc": "ziele",
                 "zrodla": [1], "fakty": ["s1"]},
                {"tekst": "Dziurawiec stosuje się przy łagodnych stanach depresyjnych.", "czesc": None,
                 "zrodla": [2], "fakty": ["s2"]}]},
            {"klucz": "barwienie", "punkty": [
                {"tekst": "Kwiatami dziurawca barwiono wełnę na żółto i czerwono.", "czesc": "kwiaty",
                 "zrodla": [2], "fakty": ["s3"]}]},
            {"klucz": "kultura", "punkty": [
                {"tekst": "W noc świętojańską wieszano dziurawiec w oknach dla ochrony przed złem.", "czesc": None,
                 "zrodla": [1], "fakty": ["s4"]}]},
        ],
        "fakty": [{"id": f"s{i}", "sekcja": "ciekawostki", "tekst": "x" * 25, "zrodlo": {"url": "https://pl.wikipedia.org/"}}
                  for i in range(1, 5)],
    },
}


def test_coverage_and_gaps():
    cov = schema.coverage(PLANT, {"barwienie": 1})
    st = {r["path"]: r["status"] for r in cov["sloty"]}
    assert st["zastosowanie.medyczne"] == "reczne"
    assert st["profil_energetyczny.termika"] == "brak"
    assert st["zastosowanie.rzemieslnicze"] == "czeka"          # jest wiedza o barwieniu, nie scalona
    assert "zastosowanie.rzemieslnicze" in cov["braki"] and "zastosowanie.rzemieslnicze" not in cov["puste"]
    found = gaps.find_gaps(PLANT, cov["sloty"])
    paths = [g.path for g in found]
    assert "profil_energetyczny.termika" in paths and "zastosowanie.rzemieslnicze" not in paths
    assert "czesci_rosliny.ziele.ostrzezenia" in paths            # pusta czesc pliku tez jest luka
    assert found[0].topic == "lecznicze" or found[0].topic == "toksycznosc"


def test_queries_in_chinese_and_japanese():
    names = gaps.plant_names("Hypericum perforatum", "Dziurawiec zwyczajny", {"zh": "贯叶连翘", "ja": "セイヨウオトギリソウ"},
                             {"en": "Hypericum perforatum"})
    found = [gaps.Gap("profil_energetyczny.termika", "Termika", "wschod"),
             gaps.Gap("zastosowanie.rzemieslnicze", "Rzemiosło", "barwienie")]
    qs = gaps.build_queries(found, names, ["en", "de", "zh", "ja"], done=set(), max_gaps=2)
    texts = [q.query for q in qs]
    assert '"贯叶连翘" 性味归经 功能主治' in texts
    assert any(q.lang == "ja" and "草木染め" in q.query and "セイヨウオトギリソウ" in q.query for q in qs)
    # zapytanie zadane niedawno nie jest powtarzane
    again = gaps.build_queries(found, names, ["en", "de", "zh", "ja"], done=set(texts), max_gaps=2)
    assert not set(q.query for q in again) & set(texts)
    assert gaps.page_about_plant("贯叶连翘是一种药用植物", names, "zh")
    assert not gaps.page_about_plant("完全无关的页面", names, "zh")


def test_cjk_quotes():
    text = "贯叶连翘，性寒，味苦、涩。归肝经。具有疏肝解郁，清热利湿的功效。"
    assert is_cjk(text) and not is_cjk("Dziurawiec zwyczajny")
    assert quote_in_text("性寒，味苦、涩。归肝经", text)
    assert not quote_in_text("这句话在原文里根本不存在", text)


def test_default_routing():
    pts = {p["fakty"][0]: p for p in merge.collect_points(PLANT)}
    assert merge.default_route(PLANT, pts["s1"]) == "czesci_rosliny.ziele.wlasciwości"
    assert merge.default_route(PLANT, pts["s2"]) == "zastosowanie.medyczne"
    assert merge.default_route(PLANT, pts["s3"]) == "zastosowanie.rzemieslnicze"   # brak czesci "kwiaty" w pliku
    assert merge.default_route(PLANT, pts["s4"]) == "ciekawostki"


class FakeLLM:
    """Przydzial: regula; scalanie medycyny: poprawna odpowiedz; reszta: zmyslenie (ma odpasc)."""
    def __init__(self):
        self.prefer_seen = []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None, prefer=None):
        self.prefer_seen.append(prefer)

        class R:
            provider, model = "ollama", "SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M"
        r = R()
        if "Przydziel" in (system or ""):
            r.text = json.dumps({"przydzial": []})
        elif "Medycyna" in prompt:
            r.text = json.dumps({"zdania": [
                {"tekst": "Leczenie łagodnych stanów depresyjnych i lękowych.", "z": [1]}], "nie_pasuje": []})
        elif "Rzemiosło" in prompt:
            r.text = json.dumps({"zdania": [{"tekst": "Kwiatami barwiono jedwab na fioletowo w 1820 roku.", "z": [1]}],
                                 "nie_pasuje": []})
        else:
            r.text = "to nie jest JSON"
        return r


def test_build_merged_with_llm_and_fallback():
    llm = FakeLLM()
    prefer = ["ollama:SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M", "ollama:SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0"]
    sc = asyncio.run(merge.build_merged(llm, PLANT, "Dziurawiec zwyczajny", "Hypericum perforatum", prefer=prefer,
                                        today="2026-10-01"))
    assert prefer in llm.prefer_seen                              # najpierw Bielik
    pola = sc["pola"]
    med = pola["zastosowanie.medyczne"]
    assert med["model"].startswith("ollama/SpeakLeash/bielik") and med["tresc"][0]["zrodla"] == [2]
    assert med["oryginal"] == PLANT["zastosowanie"]["medyczne"]
    # zmyslony rok/kolor -> kontrola odrzuca -> scalenie bez LLM (tekst punktu ze zrodlem)
    rz = pola["zastosowanie.rzemieslnicze"]
    assert rz["model"].startswith("bez LLM") and rz["oryginal"] is None
    assert rz["tresc"] == [{"tekst": "Kwiatami dziurawca barwiono wełnę na żółto i czerwono.", "zrodla": [2],
                            "fakty": ["s3"]}]
    ciek = pola["ciekawostki"]                                    # lista: element reczny zostaje + nowy
    assert ciek["typ"] == "lista" and ciek["tresc"][0]["tekst"] == "Nazywany zielem świętojańskim."
    assert ciek["tresc"][1]["zrodla"] == [1]
    assert set(sc["uzyte_fakty"]) == {"s1", "s2", "s3", "s4"}
    # kontrola kopii pliku: reczne pola nietkniete, scalenie poprawne
    cand = merge.with_merged(json.loads(json.dumps(PLANT)), sc)
    assert merge.validate_merged(cand) == []
    assert plantfile.validate_candidate(PLANT, cand, PLANT["id"]) == []
    # ponowne scalenie bez zmian na wejsciu -> wynik z pamieci (bez LLM)
    llm2 = FakeLLM()
    sc2 = asyncio.run(merge.build_merged(llm2, cand, "Dziurawiec zwyczajny", prefer=prefer, today="2026-10-02"))
    assert sc2["pola"]["zastosowanie.medyczne"] == med
    assert len(llm2.prefer_seen) == 1                             # tylko przydzial, scalanie z pamieci


def test_merge_repairs_instead_of_rejecting():
    slot = schema.SLOT_BY_PATH["zastosowanie.medyczne"]
    orig = "Leczenie łagodnych stanów depresyjnych i lękowych. Działa żółciopędnie i rozkurczowo."
    pts = [{"pid": 1, "tekst": "Napar z ziela działa żółciopędnie.", "zrodla": [1], "fakty": ["s1"]},
           {"pid": 2, "tekst": "Stosowany na oparzenia skóry.", "zrodla": [2], "fakty": ["s2"]}]
    ok = json.dumps({"zdania": [
        {"tekst": "Leczenie łagodnych stanów depresyjnych i lękowych.", "z": []},
        {"tekst": "Działa żółciopędnie (także napar z ziela) i rozkurczowo.", "z": [1]}], "nie_pasuje": [2]})
    tresc, rejected, notes = merge.apply_merge(ok, slot, orig, pts)
    assert notes == [] and rejected == [2] and tresc[1]["zrodla"] == [1]
    # LLM pomylil numer (podal 2 zamiast 1) -> zrodlo wg tresci zdania
    wrong = json.dumps({"zdania": [{"tekst": orig, "z": []},
                                   {"tekst": "Napar z ziela działa żółciopędnie.", "z": [2]}], "nie_pasuje": [2]})
    tresc, _, _ = merge.apply_merge(wrong, slot, orig, pts)
    assert tresc[-1]["zrodla"] == [1]
    # zgubiony tekst Kwiatownika i pominiety punkt -> dopisane, a nie odrzucenie calej odpowiedzi
    lost = json.dumps({"zdania": [{"tekst": "Napar z ziela działa żółciopędnie.", "z": [1]}]})
    tresc, _, notes = merge.apply_merge(lost, slot, orig, pts)
    texts = [t["tekst"] for t in tresc]
    assert "Leczenie łagodnych stanów depresyjnych i lękowych." in texts
    assert "Stosowany na oparzenia skóry." in texts and any("zgubiony" in n for n in notes)
    # zdanie zmyslone (nowe fakty) albo z HTML -> usuniete; tresc z oryginalu wraca
    bad = json.dumps({"zdania": [{"tekst": orig, "z": []},
                                 {"tekst": "Napar z ziela działa żółciopędnie.", "z": [1]},
                                 {"tekst": "Leczy też cukrzycę i nowotwory trzustki u dzieci.", "z": [2]},
                                 {"tekst": "Na oparzenia skóry<script>.", "z": [2]}], "nie_pasuje": []})
    tresc, _, notes = merge.apply_merge(bad, slot, orig, pts)
    texts = " ".join(t["tekst"] for t in tresc)
    assert "cukrzyc" not in texts and "<script" not in texts and "Stosowany na oparzenia skóry." in texts
    # wiecej niz polowa zdan zmyslona -> cala odpowiedz odrzucona (scalenie bez LLM)
    junk = json.dumps({"zdania": [{"tekst": "Roślina rośnie na Marsie od 1990 roku.", "z": [1]},
                                  {"tekst": "Leczy wszystkie choroby świata natychmiast.", "z": [2]}]})
    assert merge.apply_merge(junk, slot, orig, pts)[0] is None
    # parafraza z "lacznikowymi" slowami przechodzi (log z prawdziwego scalenia: odrzucane wczesniej)
    assert merge.faithful("Garbniki występują w stężeniu ok. 6% w liściach.",
                          ["Garbniki (ok. 6%)", "Liście zawierają około 6,5% garbników."])


def test_deterministic_merges_similar_points():
    slot = schema.SLOT_BY_PATH["ciekawostki"]
    out = merge.deterministic(slot, ["Ziele zbierano w noc świętojańską i wieszano w oknach."], [
        {"pid": 1, "tekst": "Ziele zbierano w noc świętojańską i wieszano w oknach domów.", "zrodla": [3], "fakty": ["s9"]},
        {"pid": 2, "tekst": "Nazwa pochodzi od kropek widocznych na liściach pod światło.", "zrodla": [1], "fakty": ["s8"]}])
    assert len(out) == 2 and out[0]["zrodla"] == [3] and out[1]["zrodla"] == [1]


def test_validate_merged_detects_changed_original_and_bad_sources():
    cand = json.loads(json.dumps(PLANT))
    cand["scalone"] = {"pola": {"zastosowanie.medyczne": {
        "typ": "tekst", "oryginal": "inny tekst", "tresc": [{"tekst": "Coś.", "zrodla": [9], "fakty": ["s99"]}]}},
        "uzyte_fakty": ["s99"]}
    errs = merge.validate_merged(cand)
    assert any("oryginal" in e for e in errs) and any("zrodlo spoza" in e for e in errs)
    assert any("nieistniejaca" in e for e in errs)


def test_new_plant_part_from_web_for_generator():
    """Korzen: tylko w wiedzy z sieci (plik opisuje ziele) -> nowa czesc w bloku "scalone";
    czesc z samym ostrzezeniem (kora) nie powstaje - punkt idzie do ostrzezen calej rosliny."""
    data = json.loads(json.dumps(PLANT))
    data["wiedza"]["sekcje"].append({"klucz": "zastosowanie_lecznicze", "punkty": [
        {"tekst": "Odwar z korzenia stosowano przy biegunkach.", "czesc": "korzeń", "zrodla": [1], "fakty": ["s5"]}]})
    data["wiedza"]["sekcje"].append({"klucz": "sklad", "punkty": [
        {"tekst": "Korzeń zawiera garbniki i flawonoidy.", "czesc": "korzeń", "zrodla": [2], "fakty": ["s6"]}]})
    data["wiedza"]["sekcje"].append({"klucz": "bezpieczenstwo", "punkty": [
        {"tekst": "Kora może podrażniać skórę wrażliwych osób.", "czesc": "kora", "zrodla": [1], "fakty": ["s7"]}]})
    data["wiedza"]["fakty"] += [{"id": f"s{i}", "sekcja": "ciekawostki", "tekst": "x" * 25,
                                 "zrodlo": {"url": "https://pl.wikipedia.org/"}} for i in (5, 6, 7)]
    assert {"korzeń", "kora"} <= set(schema.web_parts(data)) and "ziele" not in schema.web_parts(data)
    sc = asyncio.run(merge.build_merged(None, data, "Dziurawiec zwyczajny"))
    pola = sc["pola"]
    assert pola["czesci_rosliny.korzeń.wlasciwości"]["tresc"][0]["zrodla"] == [1]
    assert pola["czesci_rosliny.korzeń.skladniki_aktywne"]["typ"] == "lista"
    assert pola["czesci_rosliny.korzeń.wlasciwości"]["oryginal"] is None
    assert not any(p.startswith("czesci_rosliny.kora.") for p in pola)
    assert any("Kora" in t["tekst"] for t in pola["ostrzezenia"]["tresc"])
    cand = merge.with_merged(data, sc)
    assert merge.validate_merged(cand) == []
    assert "czesci_rosliny" in cand and "korzeń" not in cand["czesci_rosliny"]      # reczne pole nietkniete


def test_redo_slots_merged_in_cloud_once_bielik_is_available():
    prefer = ["ollama:SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M"]
    sc = asyncio.run(merge.build_merged(None, PLANT, "Dziurawiec zwyczajny", prefer=prefer))
    for e in sc["pola"].values():
        e["model"] = "gemini/gemini-3.5-flash"                     # scalone w chmurze, gdy Bielik sie pobieral
    cand = merge.with_merged(json.loads(json.dumps(PLANT)), sc)
    keep = FakeLLM()
    asyncio.run(merge.build_merged(keep, cand, "Dziurawiec zwyczajny", prefer=prefer))
    assert len(keep.prefer_seen) == 1                              # bez Bielika: wynik z pamieci (tylko przydzial)
    redo = FakeLLM()
    sc2 = asyncio.run(merge.build_merged(redo, cand, "Dziurawiec zwyczajny", prefer=prefer, redo_other_models=True))
    assert len(redo.prefer_seen) > 1
    assert sc2["pola"]["zastosowanie.medyczne"]["model"].startswith("ollama/SpeakLeash/bielik")


def test_cloud_down_once_means_no_more_waiting_in_this_plant():
    """Gemini/Groq z wyczerpanymi limitami: po pierwszej porazce kolejne podrozdzialy nie pytaja chmury
    (tylko Bielik z limitem czasu); bez Bielika - scalenie regulami."""
    class Down:
        calls = []

        async def complete(self, prompt, system=None, json_mode=False, prefer=None, prefer_timeout=None,
                           cloud_only=False, prefer_only=False):
            Down.calls.append((bool(prefer), prefer_only))
            raise RuntimeError("wszystkie modele maja limit")
    sc = asyncio.run(merge.build_merged(Down(), PLANT, "Dziurawiec zwyczajny", prefer=["ollama:bielik"]))
    assert sc and all(e["model"].startswith("bez LLM") for e in sc["pola"].values())
    assert Down.calls[0] == (False, False)                       # przydzial: chmura raz zawiodla
    assert all(only for pref, only in Down.calls[1:] if pref)      # potem tylko Bielik (bez chmury)
    assert all(pref for pref, _ in Down.calls[1:])                # i zadnych zapytan do samej chmury


def test_part_with_composition_only_is_created_even_if_llm_routed_it_away():
    """Klacze: tylko sklad (bez dzialania) -> nowa czesc; LLM wyslal punkt do 'Medycyna' - wraca do czesci."""
    data = json.loads(json.dumps(PLANT))
    data["wiedza"]["sekcje"].append({"klucz": "sklad", "punkty": [
        {"tekst": "Kłącze zawiera garbniki i olejek eteryczny.", "czesc": "kłącze", "zrodla": [1], "fakty": ["s5"]}]})
    data["wiedza"]["fakty"].append({"id": "s5", "sekcja": "sklad", "tekst": "x" * 25,
                                    "zrodlo": {"url": "https://pl.wikipedia.org/"}})
    pts = merge.collect_points(data)
    pid = next(p["pid"] for p in pts if p["fakty"] == ["s5"])
    routes = {p["pid"]: merge.default_route(data, p) for p in pts}
    routes[pid] = "zastosowanie.medyczne"                       # zly przydzial LLM
    logs = []
    out = merge.require_core_parts(data, routes, pts, logs.append)
    assert out[pid] == "czesci_rosliny.kłącze.skladniki_aktywne"
    assert any("kłącze" in m and "generatora" in m for m in logs)
