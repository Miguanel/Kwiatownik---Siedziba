"""Etap 5 (Docker): codzienna kopia bazy SQLite przez VACUUM INTO, limit kopii, compose z limitem logow."""
import sqlite3
from datetime import date
from pathlib import Path

import yaml

from app import backup
from app.config import settings


def test_backup_vacuum_into_once_a_day_and_keep_last(tmp_path, monkeypatch):
    db = tmp_path / "siedziba.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("create table t (x)")
    con.execute("insert into t values (1)")
    con.commit()                                                     # polaczenie otwarte - baza "dziala"
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db.as_posix()}")
    out = tmp_path / "kopie"
    made = backup.backup_databases(date(2026, 11, 1), keep=2, folder=out)
    assert [p.name for p in made] == ["siedziba_2026-11-01.db"]
    assert sqlite3.connect(made[0]).execute("select x from t").fetchall() == [(1,)]
    assert backup.backup_databases(date(2026, 11, 1), keep=2, folder=out) == []      # dzis juz jest
    backup.backup_databases(date(2026, 11, 2), keep=2, folder=out)
    backup.backup_databases(date(2026, 11, 3), keep=2, folder=out)
    assert sorted(p.name for p in out.glob("*.db")) == ["siedziba_2026-11-02.db", "siedziba_2026-11-03.db"]
    con.close()


def test_compose_limits_logs_and_checks_health():
    data = yaml.safe_load((Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text(encoding="utf-8"))
    app = data["services"]["siedziba"]
    assert app["logging"]["options"]["max-size"] == "10m" and "/health" in " ".join(app["healthcheck"]["test"])
    assert "no-new-privileges:true" in app["security_opt"] and "NET_RAW" in app["cap_drop"]
    assert all(s.get("logging") for s in data["services"].values())
