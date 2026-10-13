"""Sieci neuronowe: glowica MLP, ocena (grupy, prog), trening na bazie, bramka w potoku, kontrola faktow, panel /nn."""
import asyncio
import json
import random

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.config import settings
from app.models import Item, ItemKind, ItemStatus, NNModel, NNPrediction, NNSample, PlantFact
from app.nn import store
from app.nn.encoders import E5Encoder, NgramEncoder, Sparse
from app.nn.evaluate import cap_groups, choose_threshold, group_kfold, summarize, wilson
from app.nn.mlp import MLP

MODULES = ["app.nn.train", "app.nn.runtime", "app.nn.status", "app.nn.hooks", "app.web.nn_routes", "app.worker.nn",
           "app.worker.jobs", "app.worker.knowledge"]


@pytest.fixture
def nn_env(tmp_path, monkeypatch):
    import importlib
    eng = create_engine(f"sqlite:///{(tmp_path / 'nn.db').as_posix()}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(eng)
    for name in MODULES:
        monkeypatch.setattr(importlib.import_module(name), "engine", eng)
    monkeypatch.setattr(settings, "nn_dir", tmp_path / "nn")
    monkeypatch.setattr(settings, "nn_encoder", "ngram")
    monkeypatch.setattr(settings, "nn_audit_rate", 0.0)
    from app.nn.runtime import brain
    brain.invalidate()
    brain.recent.clear()
    import app.nn.status as st
    st._cache.clear()
    yield eng
    brain.invalidate()


# ------------------------------------------------------------------ dane syntetyczne
VOCAB = {
    "opis": "liscie jajowate lodyga wzniesiona wysokosc kwiatostan barwa",
    "wystepowanie": "rosnie lasy laki siedlisko europa azja pospolity wilgotne",
    "historia": "dawniej sredniowiecze nazwa pochodzi grecki rzymianie wiek",
    "bezpieczenstwo": "trujaca toksyczna przeciwwskazania ciaza interakcje dawki zatrucie",
}
PARTS = {"liście": "liscie lisci blaszka", "kwiaty": "kwiaty platki kwiatostan", "brak": "roslina gatunek ogolnie"}


def _fact_rows(n_plants=12, per=20, seed=1):
    rnd = random.Random(seed)
    rows = []
    for p in range(n_plants):
        for i in range(per):
            sec = list(VOCAB)[i % 4]
            part = list(PARTS)[i % 3]
            words = VOCAB[sec].split() * 2 + PARTS[part].split() * 2 + [f"roslina{p}", "jest", "oraz"]
            rnd.shuffle(words)
            rows.append(PlantFact(plant_id=f"r{p}", section=sec, part=None if part == "brak" else part,
                                  text=" ".join(words[:14]), quote=" ".join(words[3:10]), source_url="https://x/",
                                  fingerprint=f"{p}-{i}", status="applied", extracted_by="gemini/x"))
    return rows


# ------------------------------------------------------------------ MLP i ocena
def test_mlp_dense_and_sparse_learn_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((3, 384))
    y = rng.integers(0, 3, 300)
    X = (centers[y] + rng.standard_normal((300, 384)) * 2).astype(np.float32)
    m = MLP(384, 3, False)
    rep = m.fit(X[:240], y[:240], None, X[240:], y[240:])
    assert (m.predict_proba(X[240:]).argmax(1) == y[240:]).mean() > 0.9
    assert rep.best_epoch >= 1 and m.temperature > 0
    texts = [("napar herbata zaparzyc" if c == 0 else "masc wosk smarowac" if c == 1 else "nalewka spirytus")
             + f" slowo{i % 7}" for i, c in enumerate(y)]
    enc = NgramEncoder()
    Xs = enc.encode(texts)
    ms = MLP(enc.dim, 3, True)
    ms.fit(Xs.take(range(240)), y[:240], None, Xs.take(range(240, 300)), y[240:])
    assert (ms.predict_proba(Xs.take(range(240, 300))).argmax(1) == y[240:]).mean() > 0.95
    ms.save(tmp_path / "w.npz")
    back = MLP.load(tmp_path / "w.npz")
    assert np.allclose(back.predict_proba(Xs.take([0, 1])), ms.predict_proba(Xs.take([0, 1])), atol=1e-6)


def test_group_kfold_keeps_groups_apart_and_caps_big_groups():
    groups = ["duza"] * 60 + [f"g{i % 10}" for i in range(40)]
    capped = cap_groups(groups, max_share=0.1)
    assert len({g for g in capped if g.startswith("duza")}) == 6
    y = np.array([i % 2 for i in range(100)])
    folds = group_kfold(capped, y, k=5)
    for a in folds:
        for b in folds:
            if a is not b:
                assert not ({capped[i] for i in a} & {capped[i] for i in b})
    assert sum(len(f) for f in folds) == 100


def test_threshold_selection_and_summary():
    conf = np.array([0.99, 0.98, 0.97, 0.9, 0.8, 0.6, 0.55, 0.5] * 5)
    correct = np.array([1, 1, 1, 1, 1, 0, 1, 0] * 5, dtype=bool)
    t = choose_threshold(conf, correct, target=0.95, min_n=10)
    assert t["threshold"] == 0.8 and t["accuracy"] == 1.0 and t["coverage"] == 0.625
    assert choose_threshold(conf, correct, target=1.01)["threshold"] == 1.01   # cel nieosiagalny = tylko cien
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99
    proba = np.array([[0.9, 0.1], [0.2, 0.8]] * 20)
    s = summarize(np.array([0, 1] * 20), proba, ["a", "b"], 0.95, never=set())
    assert s["accuracy"] == 1.0 and set(s["allowed"]) == {"a", "b"} and s["effective"]["coverage"] == 1.0


def test_ngram_encoder_handles_cjk_and_is_deterministic():
    enc = NgramEncoder()
    a, b = enc.encode_one("金银花，清热解毒"), enc.encode_one("金银花，清热解毒")
    assert len(a[0]) > 5 and np.array_equal(a[0], b[0])
    assert abs(float(np.linalg.norm(a[1])) - 1.0) < 1e-5
    assert isinstance(enc.encode(["a", "b"]), Sparse)


def test_e5_encoder_pooling_with_fake_onnx_session():
    enc = E5Encoder()

    class FakeSP:
        def encode(self, text, out_type=int):
            return [0, 5, 6][: 1 + len(text) % 3]

    class FakeSession:
        def run(self, _, feed):
            ids = feed["input_ids"]
            return [np.ones((*ids.shape, 384), np.float32) * ids[..., None]]

    enc._session, enc._sp, enc._inputs = FakeSession(), FakeSP(), {"input_ids", "attention_mask"}
    assert enc._ids("ab") == [0, 3, 6, 7, 2]           # unk (0) -> 3, reszta +1 (uklad fairseq XLM-R)
    v = enc.encode(["krotki", "dluzszy tekst"])
    assert v.shape == (2, 384) and np.allclose(np.linalg.norm(v, axis=1), 1.0)


# ------------------------------------------------------------------ trening i kontrola faktow
def test_train_fact_networks_then_review_and_resolve(nn_env):
    from app.nn.hooks import resolve, review_facts
    from app.nn.runtime import brain
    from app.nn.train import train_task
    with Session(nn_env) as s:
        for r in _fact_rows():
            s.add(r)
        s.commit()
    row, note = train_task("fact_section", "ngram")
    assert row.version == 1 and row.status == "biezaca" and "pierwsza" in note
    assert store.task_state("fact_section") == {"mode": "cien", "version": 1}
    assert store.model_dir("fact_section", 1).joinpath("weights.npz").exists()
    m = json.loads(row.metrics_json)
    assert m["accuracy"] > 0.9 and m["groups"] >= 5 and m["confusion"]
    part_row, _ = train_task("fact_part", "ngram")
    assert part_row.status == "biezaca"

    # w cieniu: tylko zapis porownania
    with Session(nn_env) as s:
        f = PlantFact(plant_id="nowa", section="historia", part=None, text="trujaca toksyczna zatrucie dawki ciaza "
                      "kwiaty platki kwiatostan", quote="toksyczna zatrucie", source_url="https://y/",
                      fingerprint="n1", extracted_by="groq/y")
        s.add(f)
        s.commit()
        fid = f.id
    review_facts([fid], "groq/y")
    with Session(nn_env) as s:
        preds = s.exec(select(NNPrediction).where(NNPrediction.ref == f"fact:{fid}")).all()
        assert {p.decision for p in preds} == {"cien"} and {p.task for p in preds} == {"fact_section", "fact_part"}

    # aktywnie: brakujaca czesc rosliny uzupelniona, sekcja = konflikt do rozstrzygniecia
    store.set_task_state("fact_section", mode="aktywna")
    store.set_task_state("fact_part", mode="aktywna")
    brain.invalidate()
    with Session(nn_env) as s:
        for p in s.exec(select(NNModel)).all():
            p.threshold = 0.0
        s.commit()
    for t in ("fact_section", "fact_part"):        # prog 0 = siec zawsze pewna (test sciezek)
        meta_p = store.model_dir(t, 1) / "meta.json"
        meta = json.loads(meta_p.read_text())
        meta["threshold"], meta["allowed"] = 0.0, meta["labels"]
        meta_p.write_text(json.dumps(meta))
    brain.invalidate()
    stats = review_facts([fid], "groq/y")
    assert stats["sekcja"].get("konflikt") == 1
    with Session(nn_env) as s:
        assert s.get(PlantFact, fid).part == "kwiaty"
        conf = s.exec(select(NNPrediction).where(NNPrediction.decision == "konflikt")).first()
        assert conf.label == "bezpieczenstwo" and conf.teacher_label == "historia"
    msg = resolve(conf.id, "nn")
    assert "bezpieczenstwo" in msg
    with Session(nn_env) as s:
        assert s.get(PlantFact, fid).section == "bezpieczenstwo"
        smp = s.exec(select(NNSample).where(NNSample.origin == "czlowiek")).one()
        assert smp.task == "fact_section" and smp.weight == 3.0


def test_challenger_worse_than_champion_is_rejected(nn_env):
    from app.nn import train as tr
    with Session(nn_env) as s:
        for r in _fact_rows():
            s.add(r)
        s.commit()
    first, _ = tr.train_task("fact_section", "ngram")
    with Session(nn_env) as s:                       # mistrz z bardzo wysoka ocena -> pretendent przegrywa
        champ = s.get(NNModel, first.id)
        m = json.loads(champ.metrics_json)
        m["macro_f1"], m["effective"]["coverage"] = 1.0, 1.0
        champ.metrics_json = json.dumps(m)
        s.add(champ)
        s.commit()
    second, note = tr.train_task("fact_section", "ngram")
    assert second.version == 1 and "gorsza" in note
    with Session(nn_env) as s:
        assert s.exec(select(NNModel).where(NNModel.version == 2)).one().status == "odrzucona"


def test_not_enough_data(nn_env):
    from app.nn.train import NotEnoughData, train_task
    with pytest.raises(NotEnoughData):
        train_task("recipe_type", "ngram")


def test_recipe_dataset_skips_own_unconfirmed_nn_decisions(nn_env):
    from app.nn.tasks import build_recipe_type
    data = {"tytul": "Napar", "typ": "medyczne_wewnetrzne_napar", "skladniki": [{"nazwa": "rumianek"}],
            "sposob_przygotowania": ["zaparzyc"]}
    with Session(nn_env) as s:
        s.add(Item(url="u1", kind=ItemKind.recipe, status=ItemStatus.translated, data_json=json.dumps(
            {**data, "siedziba": {"typ_ustalono": "nn:recipe_type@v1"}})))
        s.add(Item(url="u2", kind=ItemKind.recipe, status=ItemStatus.approved, data_json=json.dumps(
            {**data, "siedziba": {"typ_ustalono": "nn:recipe_type@v1"}})))
        s.add(Item(url="u3", kind=ItemKind.recipe, status=ItemStatus.translated, data_json=json.dumps(
            {**data, "siedziba": {"typ_ustalono": "reguly"}})))
        s.commit()
        ds = build_recipe_type(s)
    assert sorted(ds.origins) == ["reguly", "zatwierdzony"]


# ------------------------------------------------------------------ bramka w potoku
class _Res:
    def __init__(self, text, model="m1"):
        self.text, self.model, self.provider = text, model, "fake"


class _LLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def complete(self, *a, **k):
        self.calls += 1
        return _Res(json.dumps(self.answer))


def _page(text="Some ambiguous herbal text about plants and remedies. " * 12):
    from app.scrapers.page import PageData
    return PageData(url="https://site.example/a/b", canonical=None, title="Herbal thing", lang="en", text=text,
                    headings=[], list_items=[], jsonld=[], links=[])


def _decision(**kw):
    from app.nn.runtime import Decision
    base = dict(task="page_kind", version=3, label="fact", confidence=0.97, top=[("fact", 0.97)], mode="aktywna",
                decision="przyjeta", use=True, consult=False, pred_id=None)
    base.update(kw)
    return Decision(**base)


def test_classify_uses_network_when_confident(nn_env, monkeypatch):
    from app.nn import runtime
    from app.scrapers import classify as cl
    monkeypatch.setattr(runtime.brain, "gate", lambda *a, **k: _decision())
    llm = _LLM({"type": "recipe", "confidence": 0.9})
    out = asyncio.run(cl.classify(_page(), ["herb", "plant", "remed"], llm, skip_culinary=False))
    assert out.method == "nn:page_kind@v3" and out.kind == "fact" and llm.calls == 0


def test_classify_shadow_asks_llm_and_records_teacher(nn_env, monkeypatch):
    from app.nn import runtime
    from app.scrapers import classify as cl
    with Session(nn_env) as s:
        p = NNPrediction(task="page_kind", version=3, ref="https://site.example/a/b", label="fact", confidence=0.6,
                         mode="cien", decision="cien")
        s.add(p)
        s.commit()
        pid = p.id
    monkeypatch.setattr(runtime.brain, "gate", lambda *a, **k: _decision(decision="cien", use=False, consult=True,
                                                                          mode="cien", pred_id=pid))
    llm = _LLM({"type": "fact", "confidence": 0.8, "reason": "article"})
    out = asyncio.run(cl.classify(_page(), ["herb", "plant", "remed"], llm, skip_culinary=False))
    assert out.method == "llm:m1" and llm.calls == 1
    with Session(nn_env) as s:
        p = s.get(NNPrediction, pid)
        assert p.teacher_label == "fact" and p.agree is True
        smp = s.exec(select(NNSample).where(NNSample.task == "page_kind")).one()
        assert smp.label == "fact" and smp.origin == "llm:m1" and "Herbal thing" in smp.text


def test_classify_falls_back_to_network_when_llm_fails(nn_env, monkeypatch):
    from app.nn import runtime
    from app.scrapers import classify as cl

    class Broken:
        async def complete(self, *a, **k):
            raise RuntimeError("503")

    monkeypatch.setattr(runtime.brain, "gate", lambda *a, **k: _decision(decision="do_llm", use=False, consult=True,
                                                                          confidence=0.7))
    out = asyncio.run(cl.classify(_page(), ["herb", "plant", "remed"], Broken(), skip_culinary=False))
    assert out.method == "nn:page_kind@v3" and "zapas" in out.reason


def test_gate_modes_with_trained_page_network(nn_env):
    from app.nn.runtime import brain
    from app.nn.tasks import page_input
    from app.nn.train import train_task
    kinds = {"recipe": "ingredients tbsp dried flowers steep minutes salve beeswax",
             "fact": "history legend folklore medieval monks believed named after",
             "other": "shop cart checkout price shipping category login"}
    rnd = random.Random(3)
    with Session(nn_env) as s:
        for i in range(240):
            k = list(kinds)[i % 3]
            words = kinds[k].split() * 3
            rnd.shuffle(words)
            url = f"https://site{i % 12}.example/p{i}"
            s.add(NNSample(task="page_kind", ref=url, text=page_input(url, k, " ".join(words)), label=k,
                           origin="llm:x", group=f"site{i % 12}"))
        s.commit()
    row, _ = train_task("page_kind", "ngram")
    text = page_input("https://new.example/z", "x", "shop cart checkout price shipping login category")
    assert brain.gate("page_kind", text, "https://new.example/z").decision == "cien"     # nowa siec startuje w cieniu
    store.set_task_state("page_kind", mode="aktywna")
    d = brain.gate("page_kind", text, "https://new.example/z")
    if row.threshold <= 1.0 and d.label in brain.loaded("page_kind").allowed and d.confidence >= row.threshold:
        assert d.decision == "przyjeta" and d.use and not d.consult
    else:
        assert d.decision == "do_llm" and d.consult
    store.set_task_state("page_kind", mode="wylaczona")
    assert brain.gate("page_kind", text, "x") is None
    with Session(nn_env) as s:
        assert len(s.exec(select(NNPrediction)).all()) == 2


def test_recipe_type_network_before_llm(nn_env, monkeypatch):
    from app.nn import runtime
    from app.pipeline import fast_translate as ft
    monkeypatch.setattr(runtime.brain, "gate", lambda *a, **k: _decision(task="recipe_type",
                                                                          label="medyczne_wewnetrzne_napar"))
    llm = _LLM({"typ": "kulinarne"})
    typ, zk, how = asyncio.run(ft.classify_type(llm, "Napar", ["rumianek"], ["zaparzyc"], "item:1"))
    assert typ == "medyczne_wewnetrzne_napar" and how == "nn:recipe_type@v3" and llm.calls == 0


# ------------------------------------------------------------------ zadanie w kolejce i panel
def _job_status(eng, jid):
    from app.models import Job
    with Session(eng) as s:
        return s.get(Job, jid).status.value
def test_nn_train_job_and_panel(nn_env, monkeypatch):
    import app.worker.nn as wnn
    from app.models import Job
    from app.worker.jobs import make_runner
    with Session(nn_env) as s:
        for r in _fact_rows():
            s.add(r)
        job = Job(kind="nn_train")
        s.add(job)
        s.commit()
        jid = job.id

    async def run():
        runner = make_runner(1)
        runner.start(jid, lambda: wnn.run_nn_train(jid, runner, None, ["fact_section", "recipe_type"], "ngram"))
        await asyncio.gather(*runner.tasks.values())

    asyncio.run(run())
    with Session(nn_env) as s:
        job = s.get(Job, jid)
        stats = json.loads(job.stats_json)
        assert stats["zadania"]["fact_section"]["wersja"] == 1, job.log
        assert "pominiete" in stats["zadania"]["recipe_type"]
        assert "Sekcja informacji" in job.log and "czesc 1/" in job.log

    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        r = c.get("/nn")
        assert r.status_code == 200 and "Sekcja informacji" in r.text and "v1 ngram" in r.text
        r = c.get("/nn/zadanie/fact_section")
        assert r.status_code == 200 and "Macierz pomylek" in r.text and "<svg" in r.text
        r = c.post("/nn/test", data={"text": "rosnie lasy laki siedlisko", "task": "fact_section"})
        assert r.status_code == 200 and "ag-row" in r.text
        r = c.post("/nn/fact_section/mode", data={"mode": "aktywna", "back": "/nn"}, follow_redirects=False)
        assert r.status_code == 303 and store.task_state("fact_section")["mode"] == "aktywna"
        assert c.get("/nn/decyzje?task=fact_section").status_code == 200
        r = c.get("/tasks")
        assert r.status_code == 200
        # przycisk "Trenuj" musi faktycznie uruchomic zadanie (asyncio.create_task wymaga petli zdarzen)
        r = c.post("/nn/train", data={"task": "fact_part", "encoder": "ngram"}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/jobs/")
        new_id = int(r.headers["location"].rsplit("/", 1)[1])
        assert new_id in app.state.jobs.tasks or _job_status(nn_env, new_id) != "queued"


def test_e5_download_resumes_after_broken_transfer_and_skips_404(tmp_path, monkeypatch):
    import httpx

    from app.nn import encoders
    monkeypatch.setattr(settings, "nn_dir", tmp_path / "nn")
    monkeypatch.setattr(settings, "nn_retry_wait_s", 0.0)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    body = bytes(range(256)) * 400
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers.get("range")))
        if request.url.path.endswith("model_quantized.onnx") and "Xenova" in request.url.path:
            rng = request.headers.get("range")
            if not rng:                                  # pierwsza proba: polowa pliku i zerwane polaczenie
                return httpx.Response(200, headers={"content-length": str(len(body))}, content=body[:40000])
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=body[start:])
        if request.url.path.endswith("tokenizer.json"):
            return httpx.Response(200, content=b"{}")
        return httpx.Response(404)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(encoders, "_has_tokenizers", lambda: True)
    got = asyncio.run(encoders.download_e5())
    d = encoders.encoder_dir()
    assert (d / "model.onnx").read_bytes() == body
    assert (d / "tokenizer.json").exists() and "sentencepiece.bpe.model" not in got
    assert any(r and r.startswith("bytes=") for _, r in calls)          # wznowienie od miejsca przerwania
    assert not list((tmp_path / "tmp").glob("*.part"))                   # pliki tymczasowe posprzatane


def test_auto_training_schedule(nn_env, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from app.models import Job, JobStatus
    from app.nn import schedule
    from app.nn.train import train_task
    import app.nn.schedule as sch
    monkeypatch.setattr(sch, "engine", nn_env)
    import app.nn.status as st
    monkeypatch.setattr(settings, "timezone", "UTC")
    with Session(nn_env) as s:
        for r in _fact_rows():
            s.add(r)
        s.commit()
    schedule.save_auto(on=False)
    assert schedule.due()[0] == [] and "wylaczony" in schedule.due()[1]
    schedule.save_auto(on=True, hours=24, time="", only_new=False)
    tasks, why = schedule.due()
    assert set(tasks) == {"fact_section", "fact_part"}                 # pierwszy trening: sieci z danymi
    train_task("fact_section", "ngram")
    now = datetime.now(timezone.utc)
    assert schedule.due(now)[0] == [] and "nastepny po 24 h" in schedule.due(now)[1]
    later = now + timedelta(hours=25)
    schedule.save_auto(only_new=True)
    st._cache.clear()
    assert schedule.due(later)[0] == ["fact_part"]                      # tylko siec z nowymi danymi (bez wersji)
    schedule.save_auto(time="03:30")                                    # okno godzinowe przy interwale 24 h
    at_noon = (now + timedelta(days=2)).replace(hour=12, minute=0)
    at_night = (now + timedelta(days=2)).replace(hour=4, minute=0)
    assert "03:30" in schedule.due(at_noon)[1] and schedule.due(at_night)[0] == ["fact_part"]
    with Session(nn_env) as s:
        s.add(Job(kind="nn_train", status=JobStatus.queued))
        s.commit()
    assert "juz trwa" in schedule.due(at_night)[1]


def test_auto_training_panel_settings(nn_env):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.nn import schedule
    with TestClient(app) as c:
        r = c.post("/nn/auto", data={"on": "1", "hours": "48", "time": "02:15", "only_new": "1"}, follow_redirects=False)
        assert r.status_code == 303
        a = schedule.auto_settings()
        assert a["on"] and a["hours"] == 48 and a["time"] == "02:15" and a["only_new"]
        c.post("/nn/auto", data={"hours": "5", "time": "25:99"})
        a = schedule.auto_settings()
        assert not a["on"] and a["hours"] == 24 and a["time"] == ""
        assert "Automatyczny trening" in c.get("/nn").text


def test_auto_training_starts_job_from_event_loop(nn_env, monkeypatch):
    import types

    import app.nn.schedule as sch
    from app.worker.jobs import make_runner
    monkeypatch.setattr(sch, "engine", nn_env)
    monkeypatch.setattr(sch, "due", lambda now=None: (["fact_section"], "test"))
    import app.worker.actions as actions_mod
    monkeypatch.setattr(actions_mod, "engine", nn_env)
    started = {}

    async def run():
        state = types.SimpleNamespace(jobs=make_runner(0), llm=None)     # limit 0: zadanie tylko czeka w kolejce
        started["jid"] = sch.run_if_due(state)
        started["tasks"] = list(state.jobs.tasks)
        for t in state.jobs.tasks.values():
            t.cancel()

    asyncio.run(run())
    assert started["jid"] and started["jid"] in started["tasks"]
    hist = sch.auto_settings()["history"]
    assert hist[0]["job"] == started["jid"] and hist[0]["tasks"] == ["fact_section"]
