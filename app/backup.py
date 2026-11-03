"""Kopie zapasowe baz SQLite (etap 5): VACUUM INTO -> data/backups/db/<baza>_<RRRR-MM-DD>.db, raz na dobe.

VACUUM INTO robi spojna kopie dzialajacej bazy (tez w trybie WAL) bez zatrzymywania Siedziby. Zostaje
DB_BACKUP_KEEP najnowszych kopii kazdej bazy. Kopie trafiaja obok danych (wolumen ./data), wiec przezywaja
przebudowe kontenera; pamiec LLM (llm_cache.db) nie jest kopiowana - mozna ja odtworzyc.
"""
import asyncio
import logging
import sqlite3
from datetime import date
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)
CHECK_EVERY_S = 6 * 3600


def _databases() -> list[Path]:
    url = settings.database_url
    main = Path(url.removeprefix("sqlite:///")) if url.startswith("sqlite:///") else None
    return [p for p in [main] if p and p.exists()]


def backup_databases(today: date | None = None, keep: int | None = None, folder: Path | None = None) -> list[Path]:
    """Kopia kazdej bazy, jesli dzisiejszej jeszcze nie ma. Zwraca nowe pliki."""
    keep = settings.db_backup_keep if keep is None else keep
    folder = Path(folder or Path(settings.backups_dir) / "db")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (today or date.today()).isoformat()
    made = []
    for db in _databases():
        out = folder / f"{db.stem}_{stamp}.db"
        if out.exists():
            continue
        tmp = out.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(str(db), timeout=60)
        try:
            con.execute("VACUUM INTO ?", (str(tmp),))
        finally:
            con.close()
        tmp.replace(out)
        made.append(out)
        old = sorted(folder.glob(f"{db.stem}_????-??-??.db"))
        for extra in old[:-keep] if keep > 0 else []:
            extra.unlink(missing_ok=True)
        log.warning("Kopia bazy: %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return made


async def backup_loop() -> None:
    """Co 6 godzin sprawdza, czy jest dzisiejsza kopia; bledy tylko w logu."""
    await asyncio.sleep(120)                       # po starcie: najpierw niech wstana zadania
    while True:
        if settings.db_backup_keep > 0:
            try:
                await asyncio.to_thread(backup_databases)
            except Exception as exc:  # noqa: BLE001 - kopia nigdy nie zatrzymuje Siedziby
                log.warning("Kopia bazy nie powiodla sie: %s", exc)
        await asyncio.sleep(CHECK_EVERY_S)
