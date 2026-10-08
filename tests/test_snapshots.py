"""Migawki "przed/po" zapisu plikow JSON i ich podglad w zakladce "Na zywo"."""
import json

from fastapi.testclient import TestClient

from app.config import settings
from app.knowledge import plantfile
from app.worker import snapshots


def test_write_plant_records_before_and_after(tmp_path):
    plants, backups = tmp_path / "plants", tmp_path / "bk"
    plants.mkdir()
    (plants / "mieta.json").write_text(json.dumps({"id": "mieta", "opis": "stary"}), encoding="utf-8")
    token = snapshots.current_job.set(77)
    try:
        plantfile.write_plant(plants, "mieta", {"id": "mieta", "opis": "nowy"}, False, backups, "test")
    finally:
        snapshots.current_job.reset(token)
    meta = snapshots.recent(1)[0]
    assert meta["file"] == "mieta.json" and meta["job_id"] == 77 and meta["operation"] == "test: mieta"
    assert meta["applied"] and not meta["new_file"] and meta["added"] >= 1 and meta["removed"] >= 1
    s = snapshots.get(meta["id"])
    assert json.loads(s["before"])["opis"] == "stary" and json.loads(s["after"])["opis"] == "nowy"
    lines, cut = snapshots.diff_lines(s["before"], s["after"])
    assert not cut and any(d["t"] == "add" and "nowy" in d["s"] for d in lines)


def test_new_file_rejected_and_unchanged(tmp_path):
    sid = snapshots.record(tmp_path / "a.json", None, "{}\n", "nowy")
    assert snapshots.get(sid)["before"] is None and snapshots.get(sid)["new_file"]
    assert snapshots.record(tmp_path / "a.json", "{}\r\n", "{}\n", "bez zmian") is None
    sid = snapshots.record_dict(tmp_path / "b.json", {"a": 1}, {"a": 2}, "kontrola", applied=False, note="zly")
    assert not snapshots.get(sid)["applied"] and snapshots.get(sid)["note"] == "zly"
    assert snapshots.get("../../etc") is None and snapshots.raw(sid, "x") is None


def test_prune_keeps_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "snapshots_dir", tmp_path / "snaps")
    monkeypatch.setattr(settings, "snapshots_keep", 3)
    ids = [snapshots.record(tmp_path / "x.json", None, f'{{"n": {i}}}\n', "op") for i in range(5)]
    assert [m["id"] for m in snapshots.recent(10)] == ids[::-1][:3]


def test_monitor_shows_snapshot_and_preview(tmp_path):
    sid = snapshots.record(tmp_path / "podglad.json", '{"a": 1}\n', '{"a": 2}\n', "podglad testowy")
    from app.main import app
    with TestClient(app) as client:
        live = client.get("/monitor/live").text
        assert "podglad.json" in live and f"/monitor/json/{sid}" in live
        page = client.get(f"/monitor/json/{sid}")
        assert page.status_code == 200 and "podglad testowy" in page.text and "Przed" in page.text
        assert client.get(f"/monitor/json/{sid}/before").json() == {"a": 1}
        assert client.get(f"/monitor/json/{sid}/after").json() == {"a": 2}
        assert client.get("/monitor/json/zle-id").status_code == 404


def test_item_translation_shows_original_and_history():
    from types import SimpleNamespace
    item = SimpleNamespace(id=987654, title="Herbatka", data_json=None, raw_text="Mint tea recipe",
                           structured_json='{"name": "Mint tea"}')
    sid = snapshots.record_item(item, '{"tytul": "Herbatka z mieta"}', "tlumaczenie (test)")
    s = snapshots.get(sid)
    assert "Mint tea" in s["before"] and "Herbatka z mieta" in s["after"] and s["link"] == "/items/987654"
    assert s["before_label"].startswith("oryginal") and s["file"].startswith("przepis #987654")
    item.data_json = '{"tytul": "Herbatka z mieta"}'
    sid2 = snapshots.record_item(item, '{"tytul": "Herbatka z mieta", "systemy": {}}', "opracowanie (test)")
    assert [h["id"] for h in snapshots.for_link("/items/987654")] == [sid2, sid]
    assert snapshots.get(sid2)["before_label"] is None
