"""Kampanie "od A do Z" planisty (app/agents/focus.py)."""
import json
from datetime import timedelta

import pytest

from sqlmodel import Session, delete, select

from app.agents import focus, planner, store
from app.config import settings
from app.db import engine
from app.models import AgentTask, Item, ItemKind, ItemStatus, Job, Plant, PlantFact, Source
from tests.test_agents import _state, kw  # noqa: F401 - fixture


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    _clean()                                             # wspolna baza testow - nie zostawiaj faktow innym plikom


def _clean():
    with Session(engine) as s:
        for model in (PlantFact, Item, Source):
            s.exec(delete(model))
        s.commit()


def _verified(pid: str, n: int = 2):
    with Session(engine) as s:
        p = s.get(Plant, pid)
        p.last_research_at = store.now()
        s.add(p)
        for i in range(n):
            s.add(PlantFact(plant_id=pid, section="ciekawostki", text=f"fakt {i}", source_url=f"https://x.de/{pid}{i}",
                            fingerprint=f"{pid}{i}", status="verified"))
        s.commit()


def _props(state=None):
    info = {}
    props = {p.key: p for p in planner.candidates(state or _state(), info=info)}
    return props, info["kampania"]


def test_plant_campaign_finishes_started_plants_before_new_research(kw, monkeypatch):
    monkeypatch.setattr(settings, "agents_focus_plants", 1)
    _clean()
    _verified("tarnina")                                 # tarnina: wiedza zebrana, czeka na zapis
    props, k = _props()
    assert k["plants"] == ["tarnina"] and k["stages"]["tarnina"] == "zapis"
    apply = props["grupa:plants_apply"]
    assert apply.params["rosliny"][0] == "tarnina" and "kampania roslin" in apply.why
    new = props["grupa:plants_new"]                      # krwawnik - nowe badania czekaja na domkniecie tarniny
    assert "krwawnik" in new.params["rosliny"] and "poza kampania" in new.why
    assert apply.score > new.score
    assert store.runtime()["kampania_roslin"]["rosliny"] == ["tarnina"]

    with Session(engine) as s:                           # zapis sie wykonal - tarnina zeszla z listy
        for f in s.exec(select(PlantFact)).all():
            f.status = "applied"
            s.add(f)
        s.commit()
    monkeypatch.setattr(focus, "_membership", lambda groups, skip: {
        pid: keys for pid, keys in _real_membership(groups, skip).items() if pid != "tarnina"})
    props, k = _props()
    assert k["history"][0]["co"] == ["tarnina"] and k["history"][0]["wynik"] == "domknieta"
    assert k["plants"] == ["krwawnik"] and k["stages"]["krwawnik"] == "badania"
    assert "kampania roslin" in props["grupa:plants_new"].why


_real_membership = focus._membership


def test_task_without_effect_is_not_repeated_for_the_same_plant(kw):
    _clean()
    _verified("tarnina")
    with Session(engine) as s:                           # zapis tarniny skonczyl sie godzine temu, a ona dalej czeka
        s.add(AgentTask(agent="planista", kind="zapis", title="Zapis", status="done",
                        params_json=json.dumps({"rosliny": ["tarnina"]}),
                        ordered_at=store.now() - timedelta(hours=1), finished_at=store.now() - timedelta(hours=1)))
        s.commit()
    props, k = _props()
    assert "grupa:plants_apply" not in props
    assert k["stages"].get("tarnina") != "zapis"


def _source(name, **kw):
    with Session(engine) as s:
        src = Source(name=name, base_url=f"https://{name}.example", **kw)
        s.add(src)
        s.commit()
        return src.id


def test_source_campaign_and_translation_limit(kw, monkeypatch):
    _clean()
    fresh = _source("nowa")                               # nigdy nie skanowana
    ready = _source("gotowa", profile_status="ready", last_scan_at=store.now())   # scraper gotowy -> pobranie
    props, k = _props()
    assert k["source"] == ready and k["source_stage"] == "pobranie calej strony"
    assert "kampania zrodla" in props["grupa:harvest"].why
    scan = props["grupa:scan_new"]
    assert scan.params["ids"] == [str(fresh)] and "poza kampania zrodla" in scan.why
    assert props["grupa:harvest"].score > scan.score

    monkeypatch.setattr(settings, "agents_translate_wip", 2)
    with Session(engine) as s:                            # 2 surowe przepisy = limit pracy w toku
        for i in range(2):
            s.add(Item(url=f"https://x.de/r{i}", kind=ItemKind.recipe, status=ItemStatus.raw, title=f"r{i}"))
        s.commit()
    props, k = _props()
    assert k["wip_hold"] and k["backlog"] == 2
    assert "grupa:harvest" not in props and "grupa:scan_new" not in props     # nie dokladamy surowych przepisow
    assert "nowe skany wstrzymane" in props["grupa:translate"].why
    assert "czeka" in k["source_stage"]


def test_source_campaign_gives_way_after_time_limit(kw, monkeypatch):
    _clean()
    stuck = _source("zawieszona")
    other = _source("druga")
    store.set_runtime(kampania_zrodla={"zrodlo": stuck, "nazwa": "zawieszona",
                                       "od": (store.now() - timedelta(hours=settings.agents_focus_hours + 1)).isoformat()})
    _, k = _props()
    assert k["source"] == other
    assert k["history"][0]["co"] == ["zawieszona"] and k["history"][0]["wynik"].startswith("limit")
    assert str(stuck) in store.runtime()["kampanie_pominiete_zrodla"]


def test_retry_of_campaign_source_goes_first(kw):
    from app.models import JobStatus
    _clean()
    a = _source("a-zrodlo", profile_status="ready", last_scan_at=store.now())
    b = _source("b-zrodlo")
    with Session(engine) as s:
        jb = Job(kind="scan", status=JobStatus.failed, source_id=b)
        ja = Job(kind="harvest", status=JobStatus.failed, source_id=a)
        s.add(jb)
        s.add(ja)
        s.commit()
        ja_id = ja.id
    props, k = _props()
    assert k["source"] == a
    assert props["grupa:retry"].params["ids"] == [str(ja_id)] and "kampania zrodla" in props["grupa:retry"].why


def test_focus_can_be_switched_off(kw, monkeypatch):
    monkeypatch.setattr(settings, "agents_focus", False)
    _clean()
    _verified("tarnina")
    props, k = _props()
    assert not k["on"] and "kampania" not in props["grupa:plants_apply"].why
    assert "kampania_roslin" not in store.runtime()


def test_agents_page_shows_campaigns(kw):
    from fastapi.testclient import TestClient

    from app.main import app
    _clean()
    _verified("tarnina")
    planner.run_planner(_state(), "reczny", execute=False)
    with TestClient(app) as client:
        html = client.get("/agents").text
    assert 'Kampanie "od A do Z"' in html and "/agents/plants/tarnina" in html


def test_overlapping_audit_recommendations_are_not_both_proposed(kw):
    _clean()
    with Session(engine) as s:
        for nr, plants in ((1, ["tarnina", "krwawnik"]), (2, ["tarnina"])):
            s.add(AgentTask(agent="audytor", kind="barwienie", title=f"Barwienie {nr}", status="proposed", priority=3,
                            params_json=json.dumps({"rosliny": plants})))
        s.commit()
    props, _ = _props()
    assert [p.title for p in props.values() if p.kind == "barwienie"] == ["Barwienie 1"]
