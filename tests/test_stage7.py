"""Etap 7: siec jako zapas bez odrzucania stron, ponowna ocena odrzuconych, czyste etykiety, klasy aktywne."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Item, ItemKind, ItemStatus, NNModel, NNPrediction, NNSample
from app.nn.runtime import Decision
from app.scrapers import classify as cl
from app.scrapers.page import PageData

TEXT = ("Rumianek to roślina lecznicza znana od wieków. W ziołolecznictwie ludowym stosowano jego kwiaty, "
        "a o jego właściwościach pisali dawni zielarze. ") * 8


def page(url="https://x.pl/rumianek"):
    return PageData(url=url, canonical=None, title="Rumianek w tradycji", lang="pl", text=TEXT, headings=[],
                    list_items=[], jsonld=[], links=[])


class DownLLM:
    async def complete(self, *a, **k):
        raise RuntimeError("limit")


def _nn(label, conf=0.9):
    return Decision("page_kind", 50, label, conf, [], "cien", "cien")


@pytest.mark.parametrize("label,kind,method", [("other", "other", cl.DEFERRED), ("fact", "fact", "nn:page_kind@v50")])
def test_fallback_other_is_deferred_not_lost(monkeypatch, label, kind, method):
    monkeypatch.setattr(cl, "_nn_gate", lambda p, t: _nn(label))
    monkeypatch.setattr(cl, "heuristic_scores", lambda p, k: (0.2, 0.3))
    import app.nn.runtime as rt
    monkeypatch.setattr(rt.brain, "teach", lambda *a, **k: None)
    c = asyncio.run(cl.classify(page(), ["rumianek"], DownLLM()))
    assert c.kind == kind and c.method == method
    c2 = asyncio.run(cl.classify(page(), ["rumianek"], DownLLM(), nn_fallback=False))
    assert c2.method == cl.DEFERRED                                    # bez zapasu sieci: zawsze do ponownej oceny


@pytest.fixture
def eng(tmp_path, monkeypatch):
    e = create_engine(f"sqlite:///{(tmp_path / 's7.db').as_posix()}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(e)
    from app.config import settings
    monkeypatch.setattr(settings, "nn_dir", tmp_path / "nn")
    return e


def test_recheck_restores_wrongly_rejected_pages(eng, monkeypatch):
    import app.worker.jobs as jobs_mod
    import app.worker.recheck as rc
    from app.models import Job
    monkeypatch.setattr(rc, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        a = Item(url="https://x.pl/rumianek", kind=ItemKind.other, status=ItemStatus.skipped,
                 method="nn:page_kind@v50", reason="siec neuronowa (zapas, LLM niedostepny): nic (0.66)")
        b = Item(url="https://x.pl/kontakt", kind=ItemKind.other, status=ItemStatus.skipped, method="odlozone:llm")
        c = Item(url="https://x.pl/sklep", kind=ItemKind.other, status=ItemStatus.skipped, method="heuristic")
        job = Job(kind="page_recheck")
        for x in (a, b, c, job):
            s.add(x)
        s.commit()
        ids, jid = (a.id, b.id, c.id), job.id
    assert rc.pending_count() == 2

    class Fetcher:
        async def get(self, url, html_only=True):
            from app.scrapers.fetcher import FetchResult
            body = TEXT if "rumianek" in url else "Kontakt z redakcją. " * 30
            return FetchResult(url, 200, text=f"<html lang='pl'><body><article><h1>T</h1><p>{body}</p></article></body></html>",
                               content_type="text/html")

    async def fake_classify(p, kw, llm, skip_culinary=None, nn_fallback=True):
        assert nn_fallback is False
        kind = "fact" if "rumianek" in p.url else "other"
        return cl.Classification(kind, 0.9, "llm:m", "Rumianek", "artykul" if kind == "fact" else "nic")
    monkeypatch.setattr(cl, "classify", fake_classify)
    runner = SimpleNamespace(should_stop=lambda j: False)
    st = asyncio.run(rc.run_page_recheck(jid, runner, object(), 10, fetcher=Fetcher()))
    assert st["ciekawostki"] == 1 and st["nic"] == 1
    with Session(eng) as s:
        a, b, c = (s.get(Item, i) for i in ids)
    assert a.kind == ItemKind.fact and a.status == ItemStatus.raw and "Rumianek to roślina" in a.raw_text
    assert b.kind == ItemKind.other and b.method == "llm:m" and c.method == "heuristic"
    assert rc.pending_count() == 0


def test_clean_labels_prefer_better_teacher(eng):
    from app.nn.tasks import build_page_kind, is_local_teacher, teacher_rank
    assert is_local_teacher("llm:qwen3:8b") and not is_local_teacher("llm:openai/gpt-oss-120b")
    assert teacher_rank("jsonld") > teacher_rank("llm:openai/gpt-oss-120b") > teacher_rank("heuristic") > \
        teacher_rank("llm:qwen3:8b")
    with Session(eng) as s:
        s.add(NNSample(task="page_kind", ref="u1", text="t1", label="fact", origin="llm:openai/gpt-oss-120b", weight=1.0))
        s.add(NNSample(task="page_kind", ref="u1", text="t1", label="other", origin="llm:qwen3:8b", weight=1.0))
        s.add(NNSample(task="page_kind", ref="u2", text="t2", label="recipe", origin="llm:qwen3:8b", weight=1.0))
        s.commit()
        ds = build_page_kind(s)
    got = dict(zip(ds.refs, zip(ds.labels, ds.weights)))
    assert got["u1"] == ("fact", 1.0) and got["u2"] == ("recipe", 0.5)       # nowsza etykieta malego modelu przegrywa


def test_active_classes_from_pipeline_agreement(eng, monkeypatch):
    from app.nn import classes, store
    from app.nn.classes import update_active_classes, wilson_lower
    assert round(wilson_lower(50, 50), 2) == 0.93 and wilson_lower(0, 0) == 0.0
    with Session(eng) as s:
        s.add(NNModel(task="page_kind", version=3, encoder="ngram", status="biezaca", n_train=500, threshold=0.9))
        for i in range(60):
            s.add(NNPrediction(task="page_kind", version=3, ref=f"r{i}", label="recipe", confidence=0.95,
                               mode="cien", decision="cien", agree=True))
            s.add(NNPrediction(task="page_kind", version=3, ref=f"o{i}", label="other", confidence=0.95,
                               mode="cien", decision="cien", agree=True))
        for i in range(30):
            s.add(NNPrediction(task="page_kind", version=3, ref=f"f{i}", label="fact", confidence=0.95,
                               mode="cien", decision="cien", agree=i % 3 != 0))
        s.add(NNPrediction(task="page_kind", version=3, ref="low", label="fact", confidence=0.5, mode="cien",
                           decision="cien", agree=False))
        s.commit()
    store.set_task_state("page_kind", mode="cien")
    res = update_active_classes(eng)
    assert res["page_kind"] == ["recipe"]                                  # other nigdy, fact za slaby
    assert store.task_state("page_kind")["active_classes"] == ["recipe"]
    q = classes.class_quality("page_kind", eng)
    assert q["fact"]["n"] == 30 and q["recipe"]["zgodnosc"] == 1.0
