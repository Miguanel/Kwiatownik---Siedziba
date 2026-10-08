"""Pula ciekawostek: zbieranie (wiedza + artykuly z kontrola cytatu), ocena, podsuwanie modelom piszacym, panel."""
import asyncio
import json
import re
from datetime import date

from fastapi.testclient import TestClient
from sqlmodel import Session, select

import app.marketing.service as service
from app.marketing import ideas, topics
from app.marketing.season import season_for
from app.models import FbIdea, FbPost, Item, ItemKind, ItemStatus
from tests.test_marketing import _setup

ARTICLE = ("Бузина чорна. У Карпатах гуцули вірили, що під кущем бузини живуть духи предків, тому її ніколи "
           "не рубали без дозволу. З квітів варили сироп.")


class IdeasLLM:
    def __init__(self):
        self.calls = []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None):
        self.calls.append(system or "")

        class R:
            provider, model = "groq", "zbieracz"
        r = R()
        if system == ideas.EXTRACT_SYSTEM:
            r.text = json.dumps({"ciekawostki": [
                {"tekst": "Huculi w Karpatach wierzyli, że pod krzakiem bzu mieszkają duchy przodków.",
                 "cytat": "гуцули вірили, що під кущем бузини живуть духи предків", "roslina": "Bez czarny"},
                {"tekst": "Bez czarny leczy wszystkie choroby w tydzień – zmyślone.",
                 "cytat": "такого речення в статті немає зовсім", "roslina": "Bez czarny"}]})
        elif system == ideas.RATE_SYSTEM:
            ids = [int(x) for x in re.findall(r"ID (\d+)", prompt)]
            r.text = json.dumps({"oceny": [
                {"id": i, "ciekawosc": 9 if n == 0 else 6, "zaskoczenie": 8, "zrozumialosc": 8,
                 "ryzyko": "leczy" in prompt.split(f"ID {i}:")[1].split("\n")[0], "zajawka": f"Zajawka {i}",
                 "miesiace": [10] if n == 0 else []} for n, i in enumerate(ids)]})
        else:
            r.provider, r.model = "gemini", "autor"
            r.text = json.dumps({"tresc": "Ady, ludziska! Siedziba, bez czarny i śliwa tarnina. "
                                 + " ".join(["godom"] * 130) + " A wy?", "hashtagi": ["#jesien"]})
        return r


def _with_article(eng):
    with Session(eng) as s:
        s.add(Item(url="https://karpaty.example.ua/buzyna", kind=ItemKind.fact, status=ItemStatus.raw,
                   title="Бузина", raw_text=ARTICLE, language="uk"))
        s.commit()


def test_collect_rate_and_pool(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    _with_article(eng)
    llm = IdeasLLM()
    out = asyncio.run(service.collect_ideas(llm))
    assert out["z_wiedzy"] == 2                       # fakty "kultura" (de, pl); "opis" pominiety
    assert out["ideas"] == 1 and out["bad_quote"] == 1   # zmyslony cytat odrzucony
    with Session(eng) as s:
        rows = s.exec(select(FbIdea)).all()
        art = next(x for x in rows if x.source_type == "item")
        assert art.plant_id == "bez_czarny" and art.language == "uk" and "duchy przodków" in art.text
        assert all(x.status == "rated" and x.llm_score for x in rows)
        assert ideas.is_foreign("uk") and ideas.is_foreign(None, ARTICLE) and not ideas.is_foreign("pl-PL")
        de = next(x for x in rows if x.key == "fact:1")
        pl = next(x for x in rows if x.key == "fact:3")
        assert de.code_score > pl.code_score              # z zagranicy wyzej
        best = ideas.pool(s, month=10)
        assert best[0][1] >= best[-1][1] and all(not x.risk for x, _ in best)
    # drugi przebieg: nic nowego, artykul nie jest czytany ponownie
    n_calls = len(llm.calls)
    out2 = asyncio.run(service.collect_ideas(llm))
    assert out2["z_wiedzy"] == 0 and out2.get("articles", 0) == 0 and len(llm.calls) == n_calls


def test_posts_use_pool_without_repeats_and_intro(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    _with_article(eng)
    llm = IdeasLLM()
    intro = asyncio.run(service.generate_post(llm, "siedziba", day=date(2026, 10, 6)))
    asyncio.run(service.collect_ideas(llm))
    day = date(2026, 10, 6)
    a = asyncio.run(service.generate_post(llm, "ciekawostka", day=day))
    b = asyncio.run(service.generate_post(llm, "ciekawostka", day=day))
    with Session(eng) as s:
        pa, pb = s.get(FbPost, a), s.get(FbPost, b)
        assert pa.ref.startswith("idea:") and pb.ref.startswith("idea:") and pa.ref != pb.ref
        src = json.loads(pa.source_json)
        assert "Siedziba wybrała ją z puli" in src["data"] and "Zajawka" in src["data"]
        first = s.get(FbIdea, int(pa.ref[5:]))
        assert first.score == max(x.score for x in s.exec(select(FbIdea)).all() if not x.risk)
        intro_src = json.loads(s.get(FbPost, intro).source_json)
        assert "pracujesz w Siedzibie Kwiatownika" in intro_src["task"] and "Zakończ pytaniem" in intro_src["task"]
        # konkretna ciekawostka z panelu
        art = s.exec(select(FbIdea).where(FbIdea.source_type == "item", FbIdea.status == "rated")).first()
    c = asyncio.run(service.generate_post(llm, idea_id=art.id, day=day))
    with Session(eng) as s:
        pc = s.get(FbPost, c)
        assert pc.kind == "ciekawostka" and pc.ref == f"idea:{art.id}" and "karpaty.example.ua" in pc.text
        assert art.id in ideas.used_idea_ids(s)
        m = topics.siedziba(s, season_for(day))
    assert "kandydatów na ciekawostki" in m.data and "ukraińskich" in m.data


def test_ideas_page(tmp_path, monkeypatch):
    eng = _setup(tmp_path, monkeypatch)
    _with_article(eng)
    asyncio.run(service.collect_ideas(IdeasLLM()))
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/posts/ideas")
        assert r.status_code == 200 and "duchy przodków" in r.text and "Napisz post" in r.text and "na czasie" in r.text
        for show in ("all", "used", "risk", "rejected"):
            assert client.get(f"/posts/ideas?show={show}").status_code == 200
        with Session(eng) as s:
            iid = s.exec(select(FbIdea).where(FbIdea.status == "rated")).first().id
        assert client.post(f"/posts/ideas/{iid}/status", data={"status": "rejected"},
                           follow_redirects=False).status_code == 303
        assert client.post("/posts/ideas/collect", follow_redirects=False).status_code == 303
        assert client.post(f"/posts/ideas/{iid}/write", data={"hint": ""}, follow_redirects=False).status_code == 303
        assert "Ciekawostki - pula Siedziby" in client.get("/posts").text
    with Session(eng) as s:
        assert s.get(FbIdea, iid).status == "rejected"
        assert iid not in [x.id for x, _ in ideas.pool(s)]
