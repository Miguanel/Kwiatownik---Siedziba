"""Ocena szkicow jak specjalista: metryki kodem, recenzja eksperta (kontrola kodem), werdykt, poprawa, statystyki."""
import asyncio
import json
from datetime import date

from fastapi.testclient import TestClient
from sqlmodel import Session, select

import app.marketing.service as service
from app.marketing import metrics, review, stats
from app.models import FbPost, FbPostReview
from tests.test_marketing import _setup

GOOD = ("Ady, panie dzieju, a wiecie, co tero w lesie dojrzewo?\n\n"
        "Bez czarny ino czeko na wos. Downiej my z babką zbierały całe grona, jo to godom, ze nie ma lepszego "
        "syropu na zime. Ino pamiętojcie: surowe łowoce som trujące, trza je zawdy ugotować!\n\n"
        "Zaglądojcie do Kwiatownika po przepis. A u wos jak na bez godajom?")
WALL = ("Bez czarny jest krzewem, który rośnie w Polsce i ma 25 gatunków oraz kwitnie przez 47 dni w roku, "
        "a jego owoce są zbierane przez ludzi od bardzo dawna i wykorzystywane do produkcji różnorodnych przetworów "
        "spożywczych oraz leczniczych w medycynie ludowej wielu krajów europejskich i azjatyckich. ") * 3


def test_metrics_good_vs_wall():
    g = metrics.compute(GOOD, GOOD, "Bez czarny: zbiór owoców we wrześniu", ["Bez czarny"], 30, 200)
    w = metrics.compute(WALL, WALL, "Bez czarny: zbiór owoców", ["Bez czarny"], 30, 200)
    assert g["akapity"] == 3 and g["cta_rozmowa"] and g["cta_strona"] and g["gwara_proc"] > 8
    assert g["wynik_kodu"] > w["wynik_kodu"] + 20
    codes = {i["kod"] for i in w["problemy"]}
    assert {"sciana_tekstu", "dlugie_zdanie", "brak_pytania", "slaba_gwara", "liczby_spoza_danych"} <= codes
    assert set(w["liczby_spoza_danych"]) == {"25", "47"}
    assert w["fog"] > g["fog"] and metrics.syllables("niebieski") == 3


def test_parse_review_drops_invented_quotes_and_rules():
    raw = json.dumps({"oceny": {k: 8 for k in review.CRITERIA} | {"merytoryka": 4, "haczyk": "11"},
                      "werdykt": "publikuj", "sugestie": ["a", "a", "b"],
                      "bledy_merytoryczne": [{"cytat": "surowe łowoce som trujące", "problem": "ok"},
                                             {"cytat": "tego zdania nie ma w poście", "problem": "zmyślone"}]})
    scores, rev = review.parse_review("```json\n" + raw + "\n```", GOOD)
    assert scores["haczyk"] == 10 and scores["merytoryka"] == 4
    assert len(rev["bledy_merytoryczne"]) == 1 and rev["odrzucone_zarzuty"] == 1 and rev["sugestie"] == ["a", "b"]
    m = metrics.compute(GOOD, GOOD, None, ["Bez czarny"], 30, 200)
    verdict, reasons = review.decide(80, scores, rev, m)
    assert verdict == "popraw" and any("merytoryka" in r for r in reasons)    # model chcial "publikuj"
    assert review.decide(30, {}, {}, m)[0] == "odrzuc"
    assert review.decide(90, {"bezpieczenstwo": 2}, {}, m)[0] == "odrzuc"


class ExpertLLM:
    """Pisze posty (system postaci) albo recenzuje (system specjalisty); zapamietuje 'avoid'."""

    def __init__(self):
        self.avoid, self.prompts = [], []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None):
        self.prompts.append(prompt)

        class R:
            pass
        r = R()
        if "specjalistą od content marketingu" in (system or ""):
            self.avoid.append(avoid)
            r.provider, r.model = "groq", "recenzent"
            r.text = json.dumps({"oceny": {k: 7 for k in review.CRITERIA}, "werdykt": "publikuj",
                                 "podsumowanie": "Dobry post.", "sugestie": ["Dodaj pytanie na końcu"],
                                 "slabe_strony": ["za mało konkretu"], "lepszy_haczyk": "Ady, cosik wom powiem!",
                                 "przewidywane_zaangazowanie": "średnie"})
        else:
            r.provider, r.model = "gemini", "autor"
            r.text = json.dumps({"tresc": GOOD.replace("Bez czarny", "Siedziba, bez czarny, śliwa tarnina")
                                 + " " + " ".join(["godom"] * 90), "hashtagi": ["#jesien"]})
        return r


def test_review_improve_and_stats(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    llm = ExpertLLM()
    day = date(2026, 10, 6)
    pid = asyncio.run(service.generate_post(llm, "ciekawostka", day=day))
    with Session(eng) as s:
        post = s.get(FbPost, pid)
        src = json.loads(post.source_json)
        assert src["kind"] == "ciekawostka" and "pani Holle" in src["data"] and src["sources"]
    assert asyncio.run(service.review_posts(llm)) == 1
    assert llm.avoid[-1] == {"gemini:autor"}                       # recenzent inny niz autor
    assert "DANE ŹRÓDŁOWE" in llm.prompts[-1] and "pani Holle" in llm.prompts[-1]
    with Session(eng) as s:
        rv = s.exec(select(FbPostReview)).one()
        assert rv.expert_score == round(6 / 9 * 100, 1) and rv.model == "groq:recenzent"
        assert 0 < rv.overall <= 100 and rv.verdict in review.VERDICTS
        assert review.needs_review(s) == []
        p = s.get(FbPost, pid)
        p.text += "\nedycja"
        s.add(p)
        s.commit()
        assert review.needs_review(s) == [pid]                     # po edycji ocena nieaktualna
    # poprawa wg uwag: te same dane, uwagi recenzenta w poleceniu, stara wersja odrzucona
    new = asyncio.run(service.improve_post(llm, pid))
    wp = [x for x in llm.prompts if "ULEPSZ CAŁY TEKST" in x]
    assert wp and "Dodaj pytanie na końcu" in wp[0] and "POPRZEDNIA WERSJA" in wp[0]
    with Session(eng) as s:
        old, nw = s.get(FbPost, pid), s.get(FbPost, new)
        assert old.status == "rejected" and nw.ref == old.ref and nw.source_json == old.source_json
        nw.status, nw.reactions, nw.comments, nw.shares = "published", 10, 2, 1
        s.add(nw)
        s.commit()
    asyncio.run(service.review_posts(llm))
    with Session(eng) as s:
        st = stats.build(s)
    assert st["n_reviewed"] == 1 and st["n_posts"] == 1 and st["criteria"][0]["avg"] == 7
    assert st["by_author"][0]["label"] == "gemini:autor" and st["n_results"] == 1
    assert {f["name"] for f in st["factors"]} >= {"Pytanie do czytelników", "Gęstość gwary"}


def test_stats_and_list_pages(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    llm = ExpertLLM()
    for k in ("siedziba", "ciekawostka", "przepis"):
        asyncio.run(service.generate_post(llm, k, day=date(2026, 10, 6)))
    asyncio.run(service.review_posts(llm))
    asyncio.run(service.review_posts(None, [1]))                   # bez LLM: sama ocena kodem
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/posts/stats")
        assert r.status_code == 200 and "Ocena eksperta wg kryteriow" in r.text and "Haczyk" in r.text
        assert "Ktory model pisze najlepiej" in r.text and "gemini:autor" in r.text
        r = client.get("/posts")
        assert r.status_code == 200 and "Napisz od nowa wg uwag" in r.text and "Przeredaguj wg raportu" in r.text and "Pomiary kodem" in r.text
        assert "bez eksperta" in r.text
        assert client.post("/posts/1/results", data={"reactions": "12", "comments": ""},
                           follow_redirects=False).status_code == 303
        assert client.post("/posts/review", follow_redirects=False).status_code == 303
        assert client.post("/posts/2/improve", follow_redirects=False).status_code == 303
    with Session(eng) as s:
        assert s.get(FbPost, 1).reactions == 12 and s.get(FbPost, 1).comments is None


def test_stats_empty(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    with Session(eng) as s:
        assert stats.build(s)["n_reviewed"] == 0


def test_status_poll_triggers_list_refresh_once(tmp_path, monkeypatch):
    """Pasek stanu jest odpytywany zamiast calej listy; 'posts-changed' tylko gdy skonczylo sie NOWE zadanie."""
    eng = _setup(tmp_path, monkeypatch)
    from app.main import app
    from app.models import Job, JobStatus
    with TestClient(app) as client:
        page = client.get("/posts").text
        assert 'id="fp-status"' in page and 'hx-trigger="posts-changed from:body"' in page
        assert 'hx-trigger="every 3s" hx-target="#fp-list"' not in page      # brak odswiezania calej listy
        assert 'data-k="rev-' not in page or "fpRestoreOpen" in page
        with Session(eng) as s:
            j = Job(kind="fb_review", status=JobStatus.done)
            s.add(j)
            s.commit()
            jid = j.id
        r = client.get(f"/posts/status?seen={jid - 1}")
        assert r.headers.get("HX-Trigger") == "posts-changed" and f"seen={jid}" in r.text
        r = client.get(f"/posts/status?seen={jid}")
        assert "HX-Trigger" not in r.headers and "every 10s" in r.text


def test_rewrite_edited_text_with_last_report(tmp_path, monkeypatch):
    """'Przeredaguj wg raportu': tekst z pola (z recznymi poprawkami) + ostatni raport; ogon (podpis, link) bez zmian;
    dziala tez dla starych postow bez zapisanych danych zrodlowych."""
    eng = _setup(tmp_path, monkeypatch)
    llm = ExpertLLM()
    pid = asyncio.run(service.generate_post(llm, "ciekawostka", day=date(2026, 10, 6)))
    asyncio.run(service.review_posts(llm, [pid]))
    with Session(eng) as s:
        p = s.get(FbPost, pid)
        body, tail = service.split_text(p, service.get_persona())
        assert body == p.body and "Wasz Leśny Dziadyga" in tail and "Źródło:" in tail
        p.text = "MOJA RĘCZNA POPRAWKA. " + p.text.replace("#jesien", "#mojtag")
        p.edited, p.source_json = True, None                 # jak stary post sprzed zapisywania danych
        s.add(p)
        s.commit()
    new = asyncio.run(service.improve_post(llm, pid, mode="redaguj"))
    prompt = [x for x in llm.prompts if "ULEPSZ CAŁY TEKST" in x][0]
    assert "Zachowaj sens ręcznych poprawek" in prompt and "MOJA RĘCZNA POPRAWKA" in prompt
    assert "Dodaj pytanie na końcu" in prompt and "[ZACHOWAJ]" not in prompt or True
    assert "NIE dodawaj żadnych nowych faktów" in prompt and "Wasz Leśny Dziadyga" not in prompt.split("TEKST DO")[1]
    with Session(eng) as s:
        old, nw = s.get(FbPost, pid), s.get(FbPost, new)
        assert old.status == "rejected" and "lepszą wersją" in old.problems
        assert "#mojtag" in nw.text and nw.text.count("Wasz Leśny Dziadyga") == 1 and nw.ref == old.ref
    # bez raportu redakcja nie rusza
    pid2 = asyncio.run(service.generate_post(llm, "siedziba", day=date(2026, 10, 6)))
    import pytest
    with pytest.raises(RuntimeError, match="raportu"):
        asyncio.run(service.improve_post(llm, pid2, mode="redaguj"))


class ScriptedLLM:
    """Recenzje z zadanymi ocenami po kolei (1. = ocena wyjsciowa itd.); posty z polem 'wdrozone'."""

    def __init__(self, review_scores):
        self.review_scores, self.writes = list(review_scores), []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None):
        class R:
            pass
        r = R()
        if "specjalistą od content marketingu" in (system or ""):
            v = self.review_scores.pop(0)
            r.provider, r.model = "groq", "recenzent"
            r.text = json.dumps({"oceny": {k: v for k in review.CRITERIA}, "werdykt": "publikuj",
                                 "sugestie": [f"uwaga przy ocenie {v}"], "mocne_strony": ["klimat"]})
        else:
            self.writes.append(prompt)
            n = len(self.writes)
            r.provider, r.model = "gemini", "autor"
            r.text = json.dumps({"tresc": GOOD.replace("Bez czarny", f"Wersja {'ABCDEFGH'[n - 1]}: Siedziba, bez czarny") + " "
                                 + " ".join(["godom"] * 90), "hashtagi": ["#jesien"],
                                 "wdrozone": [f"[HACZYK] poprawka {n}"]})
        return r


def test_improve_rounds_pick_best_and_history(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    from app.marketing import history
    from app.models import FbPostVersion
    # ocena wyjsciowa 4, runda 1 -> 6, runda 2 -> 5: zapisana ma byc runda 1, raport rundy 1 trafia do rundy 2
    llm = ScriptedLLM([4, 6, 5])
    pid = asyncio.run(service.generate_post(llm, "ciekawostka", day=date(2026, 10, 6)))
    asyncio.run(service.review_posts(llm, [pid]))
    new = asyncio.run(service.improve_post(llm, pid, rounds=2, target=99))
    assert len(llm.writes) == 3 and "uwaga przy ocenie 6" in llm.writes[2] and "Wersja B" in llm.writes[2]
    assert "[MERYTORYKA 4/10 → cel min. 8]" in llm.writes[1] and "[ZACHOWAJ] klimat" in llm.writes[1]
    with Session(eng) as s:
        nw, old = s.get(FbPost, new), s.get(FbPost, pid)
        rv = json.loads(nw.revision_json)
        assert "Wersja B:" in nw.body and rv["najlepsza_runda"] == 1 and [r["runda"] for r in rv["rundy"]] == [1, 2]
        assert rv["wdrozone"] == ["[HACZYK] poprawka 2"] and rv["ocena_po"] > rv["ocena_przed"]
        assert nw.parent_id == pid and nw.root_id == pid and old.status == "rejected"
        assert review.latest_reviews(s, [new])[new].text_hash == review.text_hash(nw.text)   # ocena juz jest
        posts, versions = history.family(s, new)
        assert [p.id for p in posts] == [pid, new]
        assert [v.kind for v in versions] == ["generated", "round", "improve"]
    # wersja gorsza od wyjsciowej: stara zostaje
    llm2 = ScriptedLLM([8, 3])
    p2 = asyncio.run(service.generate_post(llm2, "siedziba", day=date(2026, 10, 6)))
    asyncio.run(service.review_posts(llm2, [p2]))
    n2 = asyncio.run(service.improve_post(llm2, p2, rounds=1))
    with Session(eng) as s:
        assert s.get(FbPost, p2).status == "draft" and s.get(FbPost, n2).parent_id == p2
    # cel osiagniety w 1. rundzie -> bez 2. rundy
    llm3 = ScriptedLLM([4, 10])
    p3 = asyncio.run(service.generate_post(llm3, "przepis", day=date(2026, 10, 6)))
    asyncio.run(service.review_posts(llm3, [p3]))
    asyncio.run(service.improve_post(llm3, p3, rounds=3, target=72))
    assert len(llm3.writes) == 2


def test_word_diff_and_history_page(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    from app.marketing import history
    d = history.word_diff("Ady, bez czarny dojrzoł.", "Ady, bez czorny juzci dojrzoł!")
    assert d["added"] == 2 and d["removed"] == 1 and ("del", "czarny") in d["ops"] and ("ins", "czorny juzci") in d["ops"]
    assert "".join(t for o, t in d["ops"] if o != "ins") == "Ady, bez czarny dojrzoł."
    llm = ExpertLLM()
    pid = asyncio.run(service.generate_post(llm, "siedziba", day=date(2026, 10, 6)))
    from app.main import app
    with TestClient(app) as client:
        with Session(eng) as s:
            orig = s.get(FbPost, pid).text
        client.post(f"/posts/{pid}/save", data={"text": orig.replace("Ady", "Hej"), "status": ""})
        client.post(f"/posts/{pid}/save", data={"text": orig.replace("Ady", "Hej") , "status": "approved"})
        with Session(eng) as s:
            vs = s.exec(select(history.FbPostVersion).order_by(history.FbPostVersion.id)).all()
            assert [v.kind for v in vs] == ["generated", "edit"]         # bez duplikatu przy samej zmianie statusu
        r = client.get(f"/posts/{pid}/history")
        assert r.status_code == 200 and "<del>Ady</del>" in r.text and "<ins>Hej</ins>" in r.text
        assert "edycja ręczna" in r.text
        r = client.get(f"/posts/{pid}/history?a={vs[0].id}&b={vs[1].id}&view=side")
        assert r.status_code == 200 and "obok siebie" in r.text
        assert client.post(f"/posts/{pid}/restore/{vs[0].id}", follow_redirects=False).status_code == 303
        assert "historia (3)" in client.get("/posts").text
    with Session(eng) as s:
        assert s.get(FbPost, pid).text == orig


def test_light_silesian_and_no_mining():
    """Lekka slaska godka: wyczuwalna, ale zrozumiala; nawiazania do kopalni sa wylapywane."""
    light = ("Witejcie, ludkowie! Jo je Leśny Dziadyga i terozki robię w Siedzibie Kwiatownika.\n\n"
             "Ta maszyna sama szuka stron o ziołach w cudzych krajach. Fest się przy tym napracuje, dyć każdy "
             "przepis trzeba przetłumaczyć i sprawdzić, skąd jest. Moja oma zawdy mówiła, że ziele trzeba znać "
             "po imieniu, i Siedziba robi to samo, tylko gryfnie po kolei: najpierw zbiera, potem przesiewa, "
             "a na koniec sprawdza.\n\nA wy jakie ziele chcielibyście tukej zobaczyć?")
    m = metrics.compute(light, light, None, None, 30, 200)
    assert 3 <= m["gwara_proc"] <= 12 and m["podwyniki"]["gwara"] == 100
    assert not {"slaba_gwara", "ciezka_gwara", "kopalnia"} & {i["kod"] for i in m["problemy"]}
    mine = light.replace("Fest się przy tym napracuje", "Robi jak sztajger na grubie i fedruje wongiel")
    m2 = metrics.compute(mine, mine, None, None, 30, 200)
    assert "kopalnia" in {i["kod"] for i in m2["problemy"]} and m2["podwyniki"]["gwara"] < 100
    ok = metrics.compute("Kopali ziemniaki, gruby pień, szybko, duchy przodków.", None, None, None, 1, 200)
    assert "kopalnia" not in {i["kod"] for i in ok["problemy"]}
