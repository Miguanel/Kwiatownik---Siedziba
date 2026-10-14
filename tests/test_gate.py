"""Brama zapisu plikow roslin (ekspertyza 2026-10-14: R1 rejestr pol, R2 kanoniczne id, R3 bezpiecznik)."""
import copy
import json

import pytest
from sqlmodel import Session, SQLModel, create_engine

import app.knowledge.gate as gate
from app.agents.tasks import gate_blocked
from app.knowledge import plantfile
from app.models import Plant, PlantAlias

BASE = {"id": "lipa", "nazwa_pl": "Lipa drobnolistna", "nazwa_lat": "Tilia cordata", "opis": "Ręczny opis.",
        "tagi": ["drzewo"]}

# przykladowa zawartosc kazdego bloku Siedziby - pierwszy i DRUGI zapis musza przejsc kontrole
SAMPLE = {
    "wiedza": ({"zaktualizowano": "2026-10-01", "fakty": [], "zrodla": []},
               {"zaktualizowano": "2026-10-02", "fakty": [], "zrodla": []}),
    "zdjecia_wiki": ({"zaktualizowano": "2026-10-01", "w_url": False, "zdjecia": []},
                     {"zaktualizowano": "2026-10-02", "w_url": False, "zdjecia": []}),
    "scalone": ({"zaktualizowano": "2026-10-01", "pola": {}}, {"zaktualizowano": "2026-10-02", "pola": {}}),
    "rozmieszczenie": ({"wersja": 1, "zaktualizowano": "2026-10-01", "wstawki": [], "duplikaty": []},
                       {"wersja": 1, "zaktualizowano": "2026-10-02", "wstawki": [], "duplikaty": []}),
}


@pytest.fixture()
def db(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{(tmp_path / 'g.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(gate, "engine", eng)
    plants = tmp_path / "plants"
    plants.mkdir()
    from app.config import settings
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    with Session(eng) as s:
        s.add(Plant(id="lipa", nazwa_pl="Lipa drobnolistna"))
        s.commit()
    return eng, plants


def test_every_siedziba_block_is_registered():
    assert set(SAMPLE) == set(gate.SIEDZIBA_BLOCKS)       # nowy blok Siedziby = wpis w rejestrze i w tym tescie


@pytest.mark.parametrize("block", sorted(SAMPLE))
def test_second_write_of_every_siedziba_block_passes(block):
    first, second = SAMPLE[block]
    original = copy.deepcopy(BASE)
    cand1 = {**original, block: first}
    assert not [e for e in plantfile.validate_candidate(original, cand1, "lipa") if "reczne pole" in e]
    cand2 = {**cand1, block: second}                      # dawniej: "zmieniono reczne pole 'rozmieszczenie'"
    assert not [e for e in plantfile.validate_candidate(cand1, cand2, "lipa") if "reczne pole" in e]


def test_manual_field_change_is_still_refused():
    cand = {**BASE, "opis": "Zmieniony przez LLM."}
    assert "zmieniono reczne pole 'opis'" in plantfile.validate_candidate(BASE, cand, "lipa")


def test_id_is_aligned_to_file_name_and_kept_as_alias(db):
    eng, plants = db
    original = {**BASE, "id": "lipa_drobnolistna"}
    cand = copy.deepcopy(original)
    errors, note = gate.align_identity(cand, "lipa")
    assert errors == [] and cand["id"] == "lipa" and "lipa_drobnolistna -> lipa" in note
    assert plantfile.validate_candidate(original, cand, "lipa") == []
    with Session(eng) as s:
        assert s.get(PlantAlias, "lipa drobnolistna").plant_id == "lipa"


def test_file_holding_another_plant_is_stopped(db):
    eng, plants = db
    (plants / "poziomka.json").write_text("{}", encoding="utf-8")
    cand = {**BASE, "id": "poziomka"}                     # tarnina.json z danymi poziomki
    errors, note = gate.align_identity(cand, "tarnina")
    assert errors and "innej rosliny (poziomka.json)" in errors[0] and cand["id"] == "poziomka" and note is None


def test_circuit_breaker_blocks_after_two_same_rejections_until_file_changes(db):
    eng, plants = db
    path = plants / "lipa.json"
    path.write_text(json.dumps(BASE), encoding="utf-8")
    err = ["zmieniono reczne pole 'opis'"]
    assert gate.record_rejection("lipa", err) == 1 and gate.blocked("lipa") is None
    assert gate.record_rejection("lipa", err) == 2 and gate.blocked("lipa") == err[0]
    assert gate.blocked_map() == {"lipa": err[0]}
    s = gate.summary()
    assert s["zablokowane"] == 1 and s["lista"][0]["odmow"] == 2
    path.write_text(json.dumps({**BASE, "opis": "Poprawione recznie."}), encoding="utf-8")
    assert gate.blocked("lipa") is None                   # plik sie zmienil - roslina wraca do pracy
    gate.record_success("lipa")
    with Session(eng) as s2:
        assert s2.get(Plant, "lipa").gate_count == 0


def test_fact_errors_count_as_one_reason(db):
    assert gate._reason_key(["s12: zla dlugosc tekstu"]) == gate._reason_key(["s99: zla dlugosc tekstu"])


def test_job_stats_with_only_refusals_mark_task_blocked():
    assert gate_blocked(json.dumps({"applied": 0, "invalid": 3}))
    assert gate_blocked(json.dumps({"merged": 0, "zablokowane": 2}))
    assert not gate_blocked(json.dumps({"applied": 1, "invalid": 3}))
    assert not gate_blocked(json.dumps({"applied": 0, "nothing": 2}))
    assert not gate_blocked(None)


def test_retry_resumes_from_checkpoint(tmp_path, monkeypatch):
    """R4: ponowienie przerwanego zadania roslin pomija rosliny skonczone przed restartem."""
    import app.worker.actions as actions
    import app.worker.jobs as jobs_mod
    import app.worker.knowledge as kn
    from app.models import Job, JobStatus
    eng = create_engine(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    for m in (actions, jobs_mod, kn):
        monkeypatch.setattr(m, "engine", eng)
    with Session(eng) as s:
        job = Job(kind="plant_apply", status=JobStatus.failed, params_json=json.dumps({"plant_ids": ["a", "b", "c"]}))
        s.add(job)
        s.commit()
        jid = job.id
    kn._checkpoint(jid, "a")
    kn._checkpoint(jid, "a")                              # drugi raz nic nie zmienia
    seen = {}
    monkeypatch.setattr(actions, "start_plant_apply", lambda state, ids, retry_of=None: seen.update(ids=ids) or 99)
    assert actions.retry_job(None, jid) == 99 and seen["ids"] == ["b", "c"]
    kn._checkpoint(jid, "b")
    kn._checkpoint(jid, "c")
    assert actions.retry_job(None, jid) is None           # wszystko skonczone - nie ma czego ponawiac
    with Session(eng) as s:
        assert json.loads(s.get(Job, jid).state_json)["done"] == ["a", "b", "c"]


def test_interrupted_jobs_get_finish_time(tmp_path, monkeypatch):
    import app.worker.jobs as jobs_mod
    from app.models import Job, JobStatus
    eng = create_engine(f"sqlite:///{(tmp_path / 'i.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        s.add(Job(kind="scan", status=JobStatus.running))
        s.commit()
    ids = jobs_mod.mark_interrupted_jobs()
    with Session(eng) as s:
        job = s.get(Job, ids[0])
        assert job.status == JobStatus.failed and job.finished_at is not None


def test_species_change_moves_knowledge_to_the_right_plant(tmp_path):
    """tarnina.json byl kopia poziomki: po przywroceniu tarniny wiedza o poziomce przechodzi do poziomki."""
    from app.knowledge.registry import sync_kwiatownik
    from app.models import PlantFact
    eng = create_engine(f"sqlite:///{(tmp_path / 's.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    plants = tmp_path / "plants"
    plants.mkdir()
    (plants / "poziomka.json").write_text(json.dumps({"id": "poziomka", "nazwa_pl": "Poziomka pospolita",
                                                      "nazwa_lat": "Fragaria vesca"}), encoding="utf-8")
    (plants / "tarnina.json").write_text(json.dumps({"id": "tarnina", "nazwa_pl": "Śliwa tarnina",
                                                     "nazwa_lat": "Prunus spinosa"}), encoding="utf-8")
    with Session(eng) as s:
        s.add(Plant(id="poziomka", nazwa_pl="Poziomka pospolita", nazwa_lat="Fragaria vesca"))
        s.add(Plant(id="tarnina", nazwa_pl="Poziomka pospolita", nazwa_lat="Fragaria vesca", wikidata="Q146684"))
        for i, (fp, st) in enumerate((("a", "verified"), ("b", "applied"), ("dup", "new"), ("c", "rejected"))):
            s.add(PlantFact(plant_id="tarnina", section="opis", text=f"Poziomka {i}", source_url="https://x",
                            fingerprint=fp, status=st))
        s.add(PlantFact(plant_id="poziomka", section="opis", text="Poziomka", source_url="https://x", fingerprint="dup"))
        s.commit()
        sync_kwiatownik(s, plants)
        t = s.get(Plant, "tarnina")
        assert t.nazwa_lat == "Prunus spinosa" and t.wikidata is None and "przeniesiono 2" in t.note
        moved = {f.fingerprint: f.status for f in s.exec(__import__("sqlmodel").select(PlantFact)
                                                          .where(PlantFact.plant_id == "poziomka")).all()}
        assert moved == {"a": "verified", "b": "verified", "dup": "new"}
        left = {f.fingerprint: f.status for f in s.exec(__import__("sqlmodel").select(PlantFact)
                                                         .where(PlantFact.plant_id == "tarnina")).all()}
        assert left == {"dup": "rejected", "c": "rejected"}


def test_fact_pages_are_linked_to_plants_by_names_and_not_reused(tmp_path, monkeypatch):
    """R11: strony z ciekawostkami (skaner) trafiaja do zbierania wiedzy o roslinie, ktorej dotycza."""
    import app.worker.knowledge as kn
    from app.models import Item, ItemKind, PlantFact
    eng = create_engine(f"sqlite:///{(tmp_path / 'f.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(kn, "engine", eng)
    with Session(eng) as s:
        p = Plant(id="rumianek_pospolity", nazwa_pl="Rumianek pospolity", nazwa_lat="Matricaria chamomilla",
                  names_json=json.dumps({"uk": "Ромашка лікарська"}))
        s.add(p)
        s.add(Item(url="https://a/1", kind=ItemKind.fact, raw_text="O rumianku i jego naparach. Rumianek koi."))
        s.add(Item(url="https://a/2", kind=ItemKind.fact, raw_text="Ромашка лікарська - Matricaria chamomilla, "
                                                                    "ромашка лікарська в народній медицині"))
        s.add(Item(url="https://a/3", kind=ItemKind.fact, raw_text="Pokrzywa i jej zastosowania."))
        s.add(PlantFact(plant_id="rumianek_pospolity", section="opis", text="x" * 30, source_url="https://a/1",
                        fingerprint="f"))
        s.commit()
        s.refresh(p)
        s.expunge(p)
    pages = kn.fact_pages_for(p, limit=3)
    assert [it.url for it in pages] == ["https://a/2"]       # a/1 juz uzyta, a/3 o innej roslinie
