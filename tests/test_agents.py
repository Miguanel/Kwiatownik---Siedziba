"""Agenci (/agents): audytor wiedzy, zleceniodawca, planista ciaglosci - metryki, zalecenia, kontrola eksperta,
zlecanie bez powtorek, jedno zadanie gdy kolejka stoi, przerwa po bledach, historia i strony panelu."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete, select

from app.agents import audit, dispatch, planner, store, tasks
from app.config import settings
from app.db import engine, init_db
from app.models import AgentRun, AgentTask, Job, JobStatus, Plant

ZH = {"nr": 3, "nazwa": "baike", "url": "https://baike.baidu.com/item/x", "jezyk": "zh-Hans", "typ": "strona"}


def _pt(text, src, czesc=None):
    return {"tekst": text, "czesc": czesc, "zrodla": src, "fakty": []}


RICH = {  # roslina z wiedza z sieci: zrodla pl, uk, zh; opowiesci z zagranicy; barwienie
    "id": "krwawnik", "nazwa_pl": "Krwawnik pospolity", "nazwa_lat": "Achillea millefolium",
    "opis": "Opis.", "ciekawostki": ["Reczna ciekawostka 1.", "Reczna 2."],
    "ostrzezenia": "Uczula.", "url": {"Pokroj": "https://upload.wikimedia.org/a.jpg"},
    "profil_energetyczny": {"opis": "Chlodny."},
    "wiedza": {"wersja": 2, "zrodla": [
        {"nr": 1, "nazwa": "Wikipedia (pl)", "url": "https://pl.wikipedia.org/wiki/K", "jezyk": "pl", "typ": "wikipedia"},
        {"nr": 2, "nazwa": "Wikipedia (uk)", "url": "https://uk.wikipedia.org/wiki/K", "jezyk": "uk", "typ": "wikipedia"}, ZH],
        "sekcje": [
            {"klucz": "kultura", "punkty": [_pt("Wieszano nad drzwiami.", [2]), _pt("Legenda o Achillesie.", [1])]},
            {"klucz": "medycyna_wschodu", "punkty": [_pt("W TCM natura chlodna.", [3])]},
            {"klucz": "barwienie", "punkty": [_pt("Barwi welne na zolto.", [1], "kwiaty")]},
            {"klucz": "opis", "punkty": [_pt("Lisce pierzaste.", [1], "liście")]}]},
}
POOR = {"id": "tarnina", "nazwa_pl": "Śliwa tarnina", "nazwa_lat": "Prunus spinosa", "opis": "Krzew."}
RECIPES = [
    {"id": "r1", "tytul": "Nalewka z tarniny", "typ": "medyczne_wewnetrzne_nalewka", "skladniki": ["owoce śliwy tarniny 1 kg"]},
    {"id": "r2", "tytul": "Napar z krwawnika", "typ": "medyczne_wewnetrzne_napar", "skladniki": ["ziele krwawnika pospolitego"],
     "zrodla": [{"url": "https://x.de"}]},
    {"id": "r3", "tytul": "Ciasto", "typ": "kulinarne", "skladniki": ["mąka", "kwiaty krwawnika"]},
    {"id": "r1", "tytul": "Nalewka z tarniny (duplikat)", "typ": "medyczne"},
]


class FakeLLM:
    def __init__(self, payload: dict | None = None):
        self.payload = payload
        self.prompts = []

    async def complete(self, prompt, system=None, json_mode=False, avoid=None, **kw):
        self.prompts.append(prompt)
        return SimpleNamespace(text=json.dumps(self.payload or {}), provider="fake", model="ekspert")

    async def aclose(self):
        pass


class FakeRunner:
    """Kolejka bez uruchamiania pracy - zadania zostaja 'queued' (jak w prawdziwej kolejce, ktora czeka)."""
    max_concurrent = 3

    def __init__(self):
        self.started = []

    def start(self, job_id, work, bypass_queue=False):
        self.started.append(job_id)


@pytest.fixture()
def kw(tmp_path, monkeypatch):
    plants, przepisy = tmp_path / "plants", tmp_path / "przepisy"
    plants.mkdir()
    przepisy.mkdir()
    for d in (RICH, POOR):
        (plants / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    (przepisy / "przepisy.json").write_text(json.dumps(RECIPES, ensure_ascii=False), encoding="utf-8")
    (przepisy / "wzorzec_przepisu.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "agents_auto_continue", True)
    init_db()
    with Session(engine) as s:
        for model in (AgentTask, AgentRun, Job, Plant):
            s.exec(delete(model))
        s.add(Plant(id="krwawnik", nazwa_pl="Krwawnik pospolity", nazwa_lat="Achillea millefolium"))
        s.add(Plant(id="tarnina", nazwa_pl="Śliwa tarnina", nazwa_lat="Prunus spinosa"))
        s.commit()
    return tmp_path


def _state(llm=True):
    return SimpleNamespace(jobs=FakeRunner(), llm=FakeLLM() if llm else None)


# ------------------------------------------------------------------ metryki i zalecenia (bez bazy)
def test_plant_metrics_hard_knowledge_and_stories():
    m = audit.plant_metrics("krwawnik", RICH, recipes=3, noncul=2)
    assert m["punkty"] == 5 and m["zagraniczne"] == 2 and m["azja"] == 1 and m["strony"] == 1
    assert m["jezyki"] == ["uk", "zh"]                          # zh-Hans -> zh
    assert m["trudne"] == 4                                     # uk, zh/strona, barwienie(sekcja), kultura(sekcja)
    assert m["opowiesci"] == {"reczne": 2, "z_sieci": 2, "zagraniczne": 1}
    assert "brak_azji" not in m["braki"] and "malo_opowiesci" not in m["braki"]
    assert "do_scalenia" in m["braki"] and "bez_zdjec" not in m["braki"]
    poor = audit.plant_metrics("tarnina", POOR)
    assert {"bez_wiedzy_z_sieci", "malo_opowiesci", "brak_azji", "bez_zdjec", "brak_bezpieczenstwa"} <= set(poor["braki"])
    assert poor["oceny"]["trudna_wiedza"] == 0 and poor["wynik"] < m["wynik"]


def test_recipes_index_categories_and_duplicates(kw):
    recipes = audit.read_recipes(settings.kwiatownik_przepisy_dir)
    assert [r["id"] for r in recipes] == ["r1", "r2", "r3"]      # wzorzec i duplikat id pominiete
    idx = audit.PlantIndex(audit.read_plants(settings.kwiatownik_plants_dir))
    assert idx.match(recipes[0]) == {"tarnina"} and idx.match(recipes[1]) == {"krwawnik"}
    assert [audit.recipe_category(r) for r in recipes] == ["nalewki i trunki", "ziololecznictwo", "kulinarne"]
    assert audit.from_abroad(recipes[1]) and not audit.from_abroad(recipes[0])


def test_compute_and_rule_recommendations(kw):
    m = audit.compute(web_counts=lambda pid: {}, with_db=False)
    assert m["kpi"]["rosliny"] == 2 and m["rosliny"][0]["id"] == "tarnina"   # od najslabszej
    by_id = {r["id"]: r for r in m["rosliny"]}
    assert by_id["krwawnik"]["przepisy"] == 2 and by_id["krwawnik"]["przepisy_niekulinarne"] == 1
    recs = audit.recommend(m, per_task=4)
    kinds = {r["typ"]: r for r in recs}
    assert kinds["nowa_wiedza"]["rosliny"] == ["tarnina"] and kinds["nowa_wiedza"]["priorytet"] == 1
    assert kinds["scal"]["rosliny"] == ["krwawnik"]
    assert kinds["wschod"]["sloty"][0].startswith("profil_energetyczny")
    assert "nowe_zrodla" in kinds and kinds["nowe_zrodla"]["kraj"] != "pl"
    assert [r["nr"] for r in recs] == list(range(1, len(recs) + 1))


def test_expert_answer_is_checked_by_code(kw):
    m = audit.compute(web_counts=lambda pid: {}, with_db=False)
    recs = audit.recommend(m)
    out = audit.check_expert({
        "oceny": {"kompletnosc": 14, "trudna_wiedza": 0, "opowiesci": "5", "przepisy": 6},
        "priorytety": [{"nr": 2, "priorytet": 9, "uzasadnienie": "wazne"}, {"nr": 99, "priorytet": 1}],
        "nowe_zalecenia": [{"typ": "opowiesci", "rosliny": ["tarnina", "mandragora"], "priorytet": 1, "uzasadnienie": "legendy"},
                           {"typ": "usun_wszystko", "rosliny": ["tarnina"]}, {"typ": "wschod", "rosliny": ["nieznana"]}],
        "pomysly": ["Ukrainskie zielniki ludowe"]}, m, recs)
    assert out["oceny"] == {"kompletnosc": 10, "trudna_wiedza": 1, "opowiesci": 5, "przepisy": 6}
    assert out["priorytety"] == [{"nr": 2, "priorytet": 5, "uzasadnienie": "wazne"}]
    assert out["nowe_zalecenia"] == [{"typ": "opowiesci", "rosliny": ["tarnina"], "priorytet": 1, "uzasadnienie": "legendy"}]
    assert any("mandragora" in x for x in out["odrzucone"]) and any("usun_wszystko" in x for x in out["odrzucone"])
    assert any("99" in x for x in out["odrzucone"]) and any("zrodla" in x for x in out["odrzucone"])  # brak oceny


# ------------------------------------------------------------------ audyt z baza
EXPERT = {"oceny": {"kompletnosc": 6, "trudna_wiedza": 4, "opowiesci": 5, "przepisy": 7, "zrodla": 8},
          "podsumowanie": "Baza ma luki w wiedzy z Azji.", "mocne": ["krwawnik"], "slabe": ["tarnina"],
          "priorytety": [{"nr": 1, "priorytet": 2, "uzasadnienie": "najpierw opowiesci"}],
          "nowe_zalecenia": [{"typ": "opowiesci", "rosliny": ["tarnina"], "priorytet": 1, "uzasadnienie": "legendy o tarninie"}],
          "pomysly": ["Japonskie 草木染め"]}


def test_audit_run_saves_report_tasks_and_compares(kw):
    llm = FakeLLM(EXPERT)
    rid = asyncio.run(audit.run_audit(llm, trigger="test"))
    with Session(engine) as s:
        run = s.get(AgentRun, rid)
        props = s.exec(select(AgentTask).where(AgentTask.run_id == rid)).all()
    rep = json.loads(run.report_json)
    assert run.status == "done" and run.expert_score == 60.0 and run.model == "fake/ekspert"
    assert run.score == round(0.6 * run.code_score + 0.4 * 60.0, 1)
    assert "tarnina" in llm.prompts[0] and "SLOWNIK TYPOW" in llm.prompts[0]
    assert rep["ekspert"]["pomysly"] == ["Japonskie 草木染め"]
    expert_rec = [r for r in rep["zalecenia"] if r["zrodlo"] == "ekspert"]
    assert expert_rec and expert_rec[0]["rosliny"] == ["tarnina"]
    assert all(t.status == "proposed" for t in props) and len(props) == len(rep["zalecenia"])
    assert any(t.reason.endswith("Ekspert: najpierw opowiesci") for t in props)
    # drugi audyt: stare zalecenia zastapione, porownanie z poprzednim
    data = json.loads((settings.kwiatownik_plants_dir / "tarnina.json").read_text(encoding="utf-8"))
    data["ciekawostki"] = ["a", "b", "c", "d", "e"]
    (settings.kwiatownik_plants_dir / "tarnina.json").write_text(json.dumps(data), encoding="utf-8")
    rid2 = asyncio.run(audit.run_audit(None, trigger="test"))
    with Session(engine) as s:
        old = s.exec(select(AgentTask).where(AgentTask.run_id == rid)).all()
        run2 = s.get(AgentRun, rid2)
    assert all(t.status == "expired" for t in old)
    rep2 = json.loads(run2.report_json)
    assert rep2["poprzedni"] == rid and rep2["ekspert"] is None and run2.expert_score is None
    assert rep2["zmiany"]["lepsze"][0]["id"] == "tarnina"
    hist = audit.plant_history("tarnina")
    assert [h["run"] for h in hist] == [rid, rid2] and hist[1]["wynik"] > hist[0]["wynik"]


def test_audit_survives_broken_expert(kw):
    class Broken:
        async def complete(self, *a, **k):
            return SimpleNamespace(text="to nie jest JSON", provider="x", model="y")
    rid = asyncio.run(audit.run_audit(Broken()))
    with Session(engine) as s:
        run = s.get(AgentRun, rid)
    assert run.status == "done" and run.expert_score is None and run.score == run.code_score
    assert "proba 2 nieudana" in run.log


# ------------------------------------------------------------------ zleceniodawca
def test_dispatch_orders_by_priority_with_limit_and_no_repeats(kw, monkeypatch):
    monkeypatch.setattr(settings, "agents_dispatch_max", 2)
    monkeypatch.setattr(settings, "agents_max_llm_jobs", 10)
    asyncio.run(audit.run_audit(None))
    state = _state()
    rid = asyncio.run(dispatch.run_dispatch(state))
    with Session(engine) as s:
        ordered = s.exec(select(AgentTask).where(AgentTask.dispatch_run_id == rid)).all()
        jobs = s.exec(select(Job)).all()
        run = s.get(AgentRun, rid)
    assert len(ordered) == 2 and all(t.status == "ordered" and t.job_id for t in ordered)
    assert min(t.priority for t in ordered) == 1 and len(state.jobs.started) == 2
    assert {j.kind for j in jobs} <= {"plant_research", "plant_merge", "plant_photos", "discover:ua", "discover:de"}
    rep = json.loads(run.report_json)
    assert len(rep["czekaja"]) >= 1 and rep["czekaja"][0]["powod"] == "limit zlecen w przebiegu"
    # nowy audyt -> te same zalecenia; rosliny zlecone przed chwila sa pomijane
    asyncio.run(audit.run_audit(None))
    first = ordered[0]
    rid2 = asyncio.run(dispatch.run_dispatch(state))
    rep2 = store.report(store.last_run("zleceniodawca"))
    with Session(engine) as s:
        again = s.exec(select(AgentTask).where(AgentTask.dispatch_run_id == rid2, AgentTask.kind == first.kind)).all()
    assert not again and any(first.kind == x["typ"] for x in rep2["pominiete"])


def test_dispatch_without_llm_keeps_llm_tasks_waiting(kw):
    asyncio.run(audit.run_audit(None))
    rid = asyncio.run(dispatch.run_dispatch(_state(llm=False)))
    rep = store.report(store.last_run("zleceniodawca"))
    assert rid and all(x["typ"] in ("scal", "zdjecia", "nowe_zrodla") for x in rep["zlecone"])
    assert any(x["powod"] == "brak modeli LLM" for x in rep["czekaja"])
    with Session(engine) as s:
        research = s.exec(select(AgentTask).where(AgentTask.kind == "nowa_wiedza")).all()
    assert all(t.status == "proposed" for t in research)


# ------------------------------------------------------------------ planista ciaglosci
def test_planner_orders_one_task_when_idle_then_waits(kw):
    state = _state()
    rid = planner.run_planner(state, "auto")
    with Session(engine) as s:
        ordered = s.exec(select(AgentTask).where(col_not_null())).all()
        jobs = s.exec(select(Job)).all()
    rep = store.report(store.last_run("planista"))
    assert len(ordered) == 1 and len(jobs) == 1 and rep["zlecono"]["job"] == jobs[0].id
    assert rep["propozycje"][0]["key"] == "audyt"           # brak audytu -> najpierw audyt
    assert jobs[0].kind == "agent_audit"
    # audyt czeka w kolejce, ale nie jest zadaniem "produkcyjnym" -> kolejka dalej stoi; audyt juz w toku,
    # wiec planista nie proponuje drugiego audytu
    rid2 = planner.run_planner(state, "auto")
    rep2 = store.report(store.last_run("planista"))
    assert rid2 != rid and rep2["zlecono"] and rep2["zlecono"]["typ"] != "audyt"
    # teraz pracuje zadanie produkcyjne -> nic nie zlecamy; to samo sprawdzenie dwa razy = jeden wpis z licznikiem
    rid3 = planner.run_planner(state, "auto")
    rid4 = planner.run_planner(state, "auto")
    with Session(engine) as s:
        run = s.get(AgentRun, rid3)
    assert rid3 == rid4 and run.repeats == 1 and "pracuje" in store.report(run)["decyzja"]


def col_not_null():
    from sqlmodel import col
    return col(AgentTask.job_id).is_not(None)


def test_planner_respects_auto_switch_and_manual_pick(kw):
    store.set_runtime(auto=False)
    state = _state()
    planner.run_planner(state, "auto")
    rep = store.report(store.last_run("planista"))
    assert rep["zlecono"] is None and "wylaczony" in rep["decyzja"] and rep["propozycje"]
    with Session(engine) as s:
        assert not s.exec(select(Job)).all()
    asyncio.run(audit.run_audit(None))
    props = planner.candidates(state)
    pick = next(p for p in props if p.source == "audyt")
    planner.run_planner(state, "reczny", execute=True, force=True, pick=pick.key)
    with Session(engine) as s:
        t = s.get(AgentTask, pick.task_id)
    assert t.status == "ordered" and t.dispatch_run_id == store.last_run("planista").id
    store.set_runtime(auto=True)


def test_planner_pauses_after_three_failures(kw):
    state = _state()
    now = store.now()
    with Session(engine) as s:
        for i in range(3):
            s.add(AgentTask(agent="planista", kind="luki", status="failed", ordered_at=now, finished_at=now))
        s.commit()
    planner.run_planner(state, "auto")
    rep = store.report(store.last_run("planista"))
    assert rep["zlecono"] is None and "przerwa" in rep["decyzja"]


def test_sync_tasks_copies_job_result(kw):
    with Session(engine) as s:
        job = Job(kind="plant_research", status=JobStatus.done, stats_json=json.dumps({"plants": 2, "facts": 14}))
        s.add(job)
        s.commit()
        t = AgentTask(agent="audytor", kind="opowiesci", status="ordered", job_id=job.id)
        s.add(t)
        s.commit()
        tid = t.id
    assert tasks.sync_tasks() == 1
    with Session(engine) as s:
        t = s.get(AgentTask, tid)
    assert t.status == "done" and json.loads(t.result_json)["facts"] == 14


def test_execute_rejects_unknown_and_missing_llm(kw):
    state = _state(llm=False)
    with pytest.raises(ValueError):
        tasks.execute(state, "kasuj_baze", {})
    with pytest.raises(ValueError, match="LLM"):
        tasks.execute(state, "opowiesci", {"rosliny": ["tarnina"]})
    with Session(engine) as s:
        t = AgentTask(agent="audytor", kind="opowiesci", params_json=json.dumps({"rosliny": ["tarnina"]}))
        s.add(t)
        s.commit()
        tid = t.id
    ok, msg = tasks.order(state, tid)
    with Session(engine) as s:
        t = s.get(AgentTask, tid)
    assert not ok and t.status == "skipped" and "LLM" in t.note


# ------------------------------------------------------------------ panel
def test_agents_pages(kw):
    from app.main import app
    with TestClient(app) as client:
        app.state.llm = FakeLLM(EXPERT)
        assert client.get("/agents").status_code == 200
        asyncio.run(audit.run_audit(FakeLLM(EXPERT)))
        asyncio.run(dispatch.run_dispatch(_state()))
        planner.run_planner(_state(), "auto")
        r = client.get("/agents")
        assert r.status_code == 200 and "Ostatni raport audytu" in r.text and "Propozycje planisty" in r.text
        for run in ("audytor", "zleceniodawca", "planista"):
            rid = store.last_run(run).id
            page = client.get(f"/agents/runs/{rid}")
            assert page.status_code == 200, run
        assert "Gdzie szukac" in client.get(f"/agents/runs/{store.last_run('audytor').id}").text
        assert client.get(f"/agents/runs/{store.last_run('audytor').id}?braki=bez_zdjec").status_code == 200
        assert client.get("/agents/runs?agent=planista").status_code == 200
        assert client.get("/agents/tasks?status=proposed").status_code == 200
        assert client.get("/agents/plants/tarnina").status_code == 200
        assert client.get("/agents/status").status_code == 200
        r = client.post("/agents/auto", data={"value": "off"}, follow_redirects=False)
        assert r.status_code == 303 and store.runtime()["auto"] is False
        r = client.post("/agents/run/planista", data={"mode": "check"}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/agents/runs/")
        with Session(engine) as s:
            t = s.exec(select(AgentTask).where(AgentTask.status == "proposed")).first()
        r = client.post(f"/agents/tasks/{t.id}/skip", data={"back": "/agents"}, follow_redirects=False)
        assert r.status_code == 303
        with Session(engine) as s:
            assert s.get(AgentTask, t.id).status == "skipped"
        store.set_runtime(auto=True)


# ------------------------------------------------------------------ straznik kolejki (zawieszone / martwe zadania, obciazenie LLM)
class LiveRunner(FakeRunner):
    """Kolejka jak prawdziwa: tasks (w pamieci), active (dzialajace), stop_flags; request_stop zapamietany."""

    def __init__(self):
        super().__init__()
        self.tasks, self.active, self.stop_flags, self.stops = {}, set(), set(), []

    def start(self, job_id, work, bypass_queue=False):
        super().start(job_id, work, bypass_queue)
        self.tasks[job_id] = object()

    def request_stop(self, job_id, grace=15.0):
        self.stops.append((job_id, grace))
        self.stop_flags.add(job_id)
        return True


def _job(kind="plant_research", status=JobStatus.running, minutes_ago=0):
    from datetime import timedelta
    with Session(engine) as s:
        at = store.now() - timedelta(minutes=minutes_ago)
        j = Job(kind=kind, status=status, created_at=at, started_at=at if status == JobStatus.running else None)
        s.add(j)
        s.commit()
        return j.id


def test_watchdog_stops_stalled_and_marks_dead_jobs(kw, monkeypatch):
    import time
    from app.agents import watchdog
    from app.worker import activity
    monkeypatch.setattr(settings, "agents_stall_minutes", 30)
    state = SimpleNamespace(jobs=LiveRunner(), llm=None)
    stalled, alive, waiting, dead, young = (_job(minutes_ago=50), _job(minutes_ago=50), _job(minutes_ago=50),
                                            _job(minutes_ago=10), _job(minutes_ago=0))
    for jid in (stalled, alive, waiting):
        state.jobs.tasks[jid] = object()
    state.jobs.active |= {stalled, alive}                       # waiting: czeka w kolejce na miejsce
    state.jobs.tasks[young] = object()
    state.jobs.active.add(young)
    activity.beat(alive)                                        # wlasnie cos zrobilo
    activity._BEAT[stalled] = time.time() - 45 * 60             # ostatni znak zycia 45 min temu
    events = watchdog.check(state)
    by_job = {e["job"]: e for e in events}
    assert set(by_job) == {stalled, dead}
    assert by_job[stalled]["typ"] == "zawieszone" and by_job[stalled]["akcja"] == "zatrzymano"
    assert state.jobs.stops == [(stalled, 30)]
    with Session(engine) as s:
        assert s.get(Job, dead).status == JobStatus.failed and "martwe" in s.get(Job, dead).log
        assert "brak postepu od 45 min" in s.get(Job, stalled).log
    # drugie sprawdzenie: zatrzymywanie w toku - bez drugiego request_stop
    again = watchdog.check(state)
    assert [e["akcja"] for e in again] == ["zatrzymywanie w toku"] and len(state.jobs.stops) == 1


def test_watchdog_report_mode_and_planner_history(kw, monkeypatch):
    import time
    from app.worker import activity
    monkeypatch.setattr(settings, "agents_stall_action", "report")
    state = SimpleNamespace(jobs=LiveRunner(), llm=FakeLLM())
    jid = _job(minutes_ago=90)
    state.jobs.tasks[jid] = object()
    state.jobs.active.add(jid)
    activity._BEAT[jid] = time.time() - 80 * 60
    rid = planner.run_planner(state, "auto")
    rep = store.report(store.last_run("planista"))
    assert rep["straznik"][0]["job"] == jid and rep["straznik"][0]["akcja"] == "tylko zgloszono"
    assert not state.jobs.stops and "zawieszone: #" in store.last_run("planista").summary
    assert "pracuje" in rep["decyzja"]                          # zadanie dalej liczy sie jako praca
    planner.run_planner(state, "auto")                          # w tym okresie ciszy juz zgloszone
    assert not store.report(store.last_run("planista")).get("straznik")
    assert rid


def test_llm_limit_holds_llm_tasks_and_planner_prefers_tasks_without_llm(kw, monkeypatch):
    from app.agents import watchdog
    monkeypatch.setattr(settings, "agents_max_llm_jobs", 1)
    asyncio.run(audit.run_audit(None))
    _job("plant_research")                                      # juz dziala jedno zadanie z LLM
    state = _state()
    load = watchdog.llm_load(state)
    assert load["overloaded"] and "limit 1" in load["powod"]
    asyncio.run(dispatch.run_dispatch(state))
    rep = store.report(store.last_run("zleceniodawca"))
    assert rep["zlecone"] and all(not watchdog.is_llm_job(tasks.KINDS[x["typ"]].job) for x in rep["zlecone"])
    assert any("limit zadan z LLM" in x["powod"] for x in rep["czekaja"])
    props = planner.candidates(state, load=load)
    llm_props = [p for p in props if watchdog.is_llm_job(tasks.KINDS[p.kind].job)]
    assert llm_props and all("[odlozone:" in p.why for p in llm_props)
    best_llm = max(p.score for p in llm_props)
    assert any(p.score > best_llm for p in props if not watchdog.is_llm_job(tasks.KINDS[p.kind].job))


def test_router_marks_job_alive_while_asking_model():
    from app.llm.router import _beat
    from app.worker import activity, snapshots
    token = snapshots.current_job.set(4242)
    try:
        assert _beat() == 4242 and activity.last_beat(4242)
    finally:
        snapshots.current_job.reset(token)


def test_organize_and_verify_survive_odd_model_answers():
    from app.knowledge import organize, verify
    facts = [{"id": f"s{i}", "sekcja": "opis", "tekst": f"Liscie sa pierzaste i drobne numer {i}.", "jezyk": "pl",
              "zrodlo": {"nazwa": "W", "url": "https://pl.wikipedia.org/wiki/X"}} for i in range(1, 4)]
    for raw in ('{"sekcje": {"opis": [{"id": 1, "tekst": "x"}]}}', '{"sekcje": ["opis", "kultura"]}',
                '{"sekcje": [{"klucz": "opis", "punkty": ["tekst"]}]}', '[1, 2]', '{"sekcje": "opis"}'):
        layout, errors = organize.apply_llm_layout(raw, facts)
        assert layout is None and errors, raw
    assert verify.parse_verdicts('{"oceny": ["ok", {"id": 3, "ok": true}]}', [3]) == {3: (True, "")}
    assert verify.parse_verdicts('[1]', [3]) == {}


def test_planner_does_not_repeat_task_without_effect(kw):
    """Zadanie z listy, ktore niczego nie zmienilo (np. eksport bez nowych przepisow), nie jest zlecane w kolko."""
    from app.models import Item, ItemKind, ItemStatus
    with Session(engine) as s:
        s.exec(delete(Item))
        main = Item(url="https://x.de/a", kind=ItemKind.recipe, status=ItemStatus.exported)
        s.add(main)
        s.commit()
        # zatwierdzony duplikat: eksport dolacza go jako 2. zrodlo - nie "czeka na eksport"
        s.add(Item(url="https://x.de/b", kind=ItemKind.recipe, status=ItemStatus.approved, duplicate_of=main.id))
        s.add(Item(url="https://x.de/c", kind=ItemKind.recipe, status=ItemStatus.approved, data_json="{}"))
        s.commit()
    state = _state()
    props = {p.key: p for p in planner.candidates(state)}
    assert props["grupa:export"].params["stan"].startswith("1|1 zatwierdzonych")   # tylko przepis bez duplikatu
    planner.run_planner(state, "reczny", execute=True, force=True, pick="grupa:export")
    with Session(engine) as s:
        t = s.exec(select(AgentTask).where(AgentTask.kind == "eksport")).one()
        job = s.get(Job, t.job_id)
        job.status = JobStatus.done                     # eksport sie wykonal, ale przepis dalej "czeka"
        s.add(job)
        s.commit()
    tasks.sync_tasks()
    assert "grupa:export" not in {p.key for p in planner.candidates(state)}


def test_loop_checks_again_soon_when_queue_is_idle(kw, monkeypatch):
    from app.agents import loop
    calls = []
    monkeypatch.setattr(settings, "agents_enabled", True)
    monkeypatch.setattr(planner, "run_planner", lambda state, trigger: calls.append(trigger))

    async def run():
        task = asyncio.create_task(loop.agents_loop(_state(), first_delay=0, poll_s=0.01))
        await asyncio.sleep(0.1)
        task.cancel()
    asyncio.run(run())
    assert len(calls) >= 3                              # kolejka stoi -> sprawdzenie co poll_s, nie co 5 min
    calls.clear()
    _job("plant_research", JobStatus.running)           # Siedziba pracuje -> tylko pelne sprawdzenie co 5 min
    asyncio.run(run())
    assert len(calls) == 1


def test_planner_does_not_resume_jobs_stopped_by_hand(kw):
    stopped = _job("plant_research", JobStatus.cancelled)
    failed = _job("plant_research", JobStatus.failed)
    props = [p for p in planner.candidates(_state()) if p.kind == "ponow"]
    assert [p.params["ids"] for p in props] == [[str(failed)]] and stopped
