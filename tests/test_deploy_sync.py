"""Wdrozeniowiec (commit + push danych Kwiatownika2 z kronika) i polaczenie z backendem Kwiatownika."""
import asyncio
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest
from sqlmodel import Session, delete, select

from app.agents import backend_sync, deploy, store
from app.config import settings
from app.db import engine, init_db
from app.models import AgentRun, Job, JobStatus, Plant, SiteStat

LIPA = {"id": "lipa", "nazwa_pl": "Lipa drobnolistna", "nazwa_lat": "Tilia cordata", "opis": "Drzewo."}


def sh(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def wiedza(n, lang="zh"):
    return {"wersja": 2, "zrodla": [{"nr": 1, "url": "https://baike.baidu.com/x", "jezyk": lang, "typ": "strona"}],
            "sekcje": [{"klucz": "kultura", "punkty": [{"tekst": f"Informacja {i}", "zrodla": [1]} for i in range(n)]}]}


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    work = tmp_path / "Kwiatownik2"
    subprocess.run(["git", "clone", str(remote), str(work)], check=True, capture_output=True)
    for k, v in (("user.name", "Michal"), ("user.email", "m@example.com")):
        sh(work, "config", k, v)
    (work / "data" / "plants").mkdir(parents=True)
    (work / "data" / "przepisy").mkdir(parents=True)
    (work / "data" / "plants" / "lipa.json").write_text(json.dumps(LIPA, ensure_ascii=False), encoding="utf-8")
    (work / "data" / "przepisy" / "przepisy_medyczne.json").write_text(
        json.dumps([{"id": "r1", "tytul": "Napar z lipy"}], ensure_ascii=False), encoding="utf-8")
    (work / "app.py").write_text("print('kwiatownik')\n", encoding="utf-8")
    sh(work, "add", ".")
    sh(work, "commit", "-m", "start")
    sh(work, "branch", "-M", "main")
    sh(work, "push", "-u", "origin", "main")
    monkeypatch.setattr(settings, "deploy_repo_dir", work)
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", work / "data" / "plants")
    monkeypatch.setattr(settings, "github_token", "ghp_TAJNYTOKEN")
    monkeypatch.setattr(settings, "deploy_min_points", 5)
    monkeypatch.setattr(settings, "deploy_min_recipes", 3)
    monkeypatch.setattr(settings, "deploy_min_hours", 6)
    monkeypatch.setattr(settings, "deploy_enabled", True)
    monkeypatch.setattr(deploy, "push_url", lambda repo: str(remote))   # zamiast GitHuba lokalne repozytorium
    monkeypatch.setattr(settings, "backend_url", "")
    init_db()
    with Session(engine) as s:
        for m in (AgentRun, Job, SiteStat):
            s.exec(delete(m))
        s.commit()
    deploy._last_check["t"] = None
    return SimpleNamespace(work=work, remote=remote)


def test_summary_counts_new_knowledge_recipes_and_validates(repo):
    w = repo.work
    data = dict(LIPA, wiedza=wiedza(7), zdjecia_wiki={"zdjecia": [{"url": "u1"}, {"url": "u2"}]})
    (w / "data" / "plants" / "lipa.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (w / "data" / "plants" / "bez.json").write_text(json.dumps({"id": "bez", "nazwa_pl": "Bez czarny",
                                                                 "wiedza": wiedza(2, "uk")}), encoding="utf-8")
    (w / "data" / "przepisy" / "siedziba_przepisy_2026-10-08.json").write_text(json.dumps(
        [{"id": "r1", "tytul": "Napar z lipy"}, {"id": "r2", "tytul": "Nalewka z bzu",
                                                 "zrodla": [{"url": "https://www.amol.pl/x"}]}]), encoding="utf-8")
    (w / "app.py").write_text("print('zmiana uzytkownika')\n", encoding="utf-8")       # nie nalezy do danych
    files = deploy.changed_files(w)
    assert {p for _, p in files} == {"data/plants/lipa.json", "data/plants/bez.json",
                                     "data/przepisy/siedziba_przepisy_2026-10-08.json"}
    s = deploy.summarize(w, files)
    assert s["informacje"] == 9 and s["nowe_rosliny"] == 1 and s["zdjecia"] == 2 and s["przepisy"] == 1
    assert s["przepisy_lista"] == [{"tytul": "Nalewka z bzu", "zrodlo": "amol.pl"}] and not s["bledy"]
    assert s["rosliny"][0] == {"id": "lipa", "nazwa": "Lipa drobnolistna", "nowe": 7, "jezyki": ["zh"], "nowa": False}
    (w / "data" / "plants" / "zepsuta.json").write_text("{zly json", encoding="utf-8")
    s2 = deploy.summarize(w, deploy.changed_files(w))
    go, why = deploy.decide(s2, None)
    assert not go and "uszkodzone" in why


def test_decide_thresholds():
    base = {"informacje": 2, "przepisy": 1, "zdjecia": 0, "nowe_rosliny": 0, "pliki": 2, "bledy": [], "rosliny": []}
    now = store.now()
    from datetime import timedelta
    assert deploy.decide(base, now - timedelta(hours=10), now)[0] is False            # za malo, niedawno
    assert deploy.decide(base, now - timedelta(hours=60), now)[0] is True             # po 48 h - cokolwiek
    assert deploy.decide({**base, "informacje": 40}, now - timedelta(hours=2), now)[0] is False   # min 6 h
    assert deploy.decide({**base, "informacje": 40}, now - timedelta(hours=7), now)[0] is True
    assert deploy.decide({**base, "pliki": 0, "informacje": 0, "przepisy": 0}, None, now)[1] == "brak nowych danych"


def test_run_deploy_commits_only_data_writes_changelog_and_pushes(repo):
    w = repo.work
    (w / "data" / "plants" / "lipa.json").write_text(json.dumps(dict(LIPA, wiedza=wiedza(6)), ensure_ascii=False),
                                                     encoding="utf-8")
    (w / "app.py").write_text("print('zmiana uzytkownika')\n", encoding="utf-8")
    rid = asyncio.run(deploy.run_deploy())
    with Session(engine) as s:
        run = s.get(AgentRun, rid)
    rep = store.report(run)
    assert rep["opublikowano"] is True and run.status == "done" and "Opublikowano" in run.summary
    remote_files = sh(repo.remote, "show", "--name-only", "--format=%an|%s", "main")
    assert "Siedziba Kwiatownika|Siedziba: Nowa wiedza o 1 roślinie" in remote_files
    assert "data/plants/lipa.json" in remote_files and "data/changelog.json" in remote_files
    assert "app.py" not in remote_files                                    # zmiana uzytkownika nietknieta
    assert sh(w, "status", "--porcelain", "--", "app.py").startswith("M")
    log = json.loads((w / "data" / "changelog.json").read_text(encoding="utf-8"))
    entry = log["wpisy"][0]
    assert entry["liczby"]["informacje"] == 6 and entry["rosliny"][0]["id"] == "lipa"
    assert "chińskie" in entry["opis"] and "ghp_TAJNYTOKEN" not in (run.log or "")
    # zaraz potem: nic nowego / za wczesnie
    rid2 = asyncio.run(deploy.run_deploy())
    assert store.report(store.last_run("wdrozeniowiec")).get("opublikowano") is False and rid2 != rid


def test_run_deploy_refuses_other_branch_and_dry_run(repo):
    w = repo.work
    (w / "data" / "plants" / "lipa.json").write_text(json.dumps(dict(LIPA, wiedza=wiedza(9))), encoding="utf-8")
    asyncio.run(deploy.run_deploy(dry_run=True))
    run = store.last_run("wdrozeniowiec")
    assert run.summary.startswith("Gotowe do publikacji") and not store.report(run)["opublikowano"]
    assert sh(repo.remote, "rev-list", "--count", "main") == "1"           # nic nie wypchnieto
    sh(w, "checkout", "-q", "-b", "eksperyment")
    asyncio.run(deploy.run_deploy(force=True))
    assert "galezi 'eksperyment'" in store.last_run("wdrozeniowiec").summary


def test_maybe_start_only_when_due(repo):
    started = []
    from app.worker import actions
    orig = actions.start_agent_deploy
    actions.start_agent_deploy = lambda state, trigger="auto": started.append(trigger) or 1
    try:
        assert asyncio.run(deploy.maybe_start(None)) is None and not started   # brak zmian
        (repo.work / "data" / "plants" / "lipa.json").write_text(json.dumps(dict(LIPA, wiedza=wiedza(8))), encoding="utf-8")
        assert asyncio.run(deploy.maybe_start(None)) is None                   # sprawdzono przed chwila (co 30 min)
        deploy._last_check["t"] = None
        assert asyncio.run(deploy.maybe_start(None)) == 1 and started == ["auto"]
    finally:
        actions.start_agent_deploy = orig


# ------------------------------------------------------------------ backend Kwiatownika
class FakeBackend:
    def __init__(self, need_copy=False):
        self.calls, self.need_copy, self.restored = [], need_copy, None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tok"
        path = request.url.path
        self.calls.append(path)
        if path == "/api/siedziba/heartbeat":
            self.heartbeat = json.loads(request.content)
            return httpx.Response(200, json={"ok": True, "boot": "b1", "potrzebna_kopia": self.need_copy})
        if path == "/api/siedziba/restore":
            self.restored = json.loads(request.content)["wiersze"]
            return httpx.Response(200, json={"ok": True})
        if path == "/api/siedziba/counters":
            return httpx.Response(200, json={"wiersze": [
                {"day": "2026-10-08", "kind": "plant_view", "key": "lipa", "n": 12},
                {"day": "2026-10-08", "kind": "plantid_click", "key": "", "n": 3}]})
        return httpx.Response(404)


def _client(fake):
    return httpx.AsyncClient(transport=httpx.MockTransport(fake), base_url="https://backend.test",
                             headers={"Authorization": "Bearer tok"})


def test_backend_sync_heartbeat_events_counters_and_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "backend_url", "https://backend.test")
    monkeypatch.setattr(settings, "backend_token", "tok")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    init_db()
    from datetime import datetime, timezone
    with Session(engine) as s:
        for m in (Job, SiteStat, AgentRun):
            s.exec(delete(m))
        if not s.get(Plant, "lipa"):
            s.add(Plant(id="lipa", nazwa_pl="Lipa drobnolistna"))
        done = Job(kind="plant_research", status=JobStatus.done, finished_at=datetime.now(timezone.utc),
                   params_json=json.dumps({"plant_ids": ["lipa"]}), stats_json=json.dumps({"verified": 14}))
        quiet = Job(kind="plant_research", status=JobStatus.done, finished_at=datetime.now(timezone.utc),
                    params_json="{}", stats_json=json.dumps({"verified": 0}))
        running = Job(kind="plant_photos", status=JobStatus.running, started_at=datetime.now(timezone.utc))
        s.add_all([done, quiet, running])
        s.add(SiteStat(day="2020-01-01", kind="plant_view", key="lipa", n=100))
        s.commit()
    fake = FakeBackend(need_copy=True)
    out = asyncio.run(backend_sync.sync(_client(fake)))
    assert out["ok"] and out["wpisy"] == 1 and out["liczniki"] == 2
    hb = fake.heartbeat
    assert hb["status"] == "pracuje" and hb["zadania"][0]["opis"].startswith("pobiera zdjęcia")
    assert hb["wpisy"][0]["text"].startswith("Zebrano 14 sprawdzonych informacji o: Lipa drobnolistna")
    assert hb["wpisy"][0]["url"] == "/plant/lipa/"
    assert {"day": "archiwum", "kind": "plant_view", "key": "lipa", "n": 100} in fake.restored
    assert backend_sync.site_views(9999).get("lipa", 0) >= 12
    # drugi raz: te same wpisy nie sa wysylane ponownie
    fake2 = FakeBackend()
    asyncio.run(backend_sync.sync(_client(fake2)))
    assert fake2.heartbeat["wpisy"] == [] and "/api/siedziba/restore" not in fake2.calls


def test_backend_sync_reports_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "backend_url", "https://backend.test")
    monkeypatch.setattr(settings, "backend_token", "tok")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    down = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)), base_url="https://backend.test")
    out = asyncio.run(backend_sync.sync(down))
    assert not out["ok"] and "503" in out["blad"] and backend_sync.load_state()["last_error"]
    monkeypatch.setattr(settings, "backend_url", "")
    assert asyncio.run(backend_sync.sync())["ok"] is False


def test_site_and_agents_pages(repo, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    with Session(engine) as s:
        s.add(SiteStat(day=store.now().strftime("%Y-%m-%d"), kind="plant_view", key="lipa", n=5))
        s.add(SiteStat(day=store.now().strftime("%Y-%m-%d"), kind="plantid_unknown", key="Ginkgo biloba", n=2))
        s.commit()
    with TestClient(app) as c:
        r = c.get("/site")
        assert r.status_code == 200 and "Ginkgo biloba" in r.text and "Brak polaczenia" in r.text
        r = c.get("/agents")
        assert r.status_code == 200 and "Wdrozeniowiec" in r.text and "Opublikuj teraz" in r.text
        rid = asyncio.run(deploy.run_deploy(dry_run=True))
        assert c.get(f"/agents/runs/{rid}").status_code == 200
