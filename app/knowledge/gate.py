"""Brama zapisu plikow roslin Kwiatownika (R1-R3 z ekspertyzy systemu, 2026-10-14).

1. Rejestr wlasnosci pol (SIEDZIBA_BLOCKS): jedyne miejsce, ktore mowi, ktore bloki pliku rosliny pisze Siedziba.
   Kontrola kopii (plantfile.validate_candidate) nie traktuje ich jak recznych pol uzytkownika. Nowy blok Siedziby
   = dopisanie go TUTAJ (wczesniej brak "rozmieszczenie" na liscie zamrozil kazda rozmieszczona rosline).
2. Kanoniczna tozsamosc: nazwa pliku jest identyfikatorem rosliny. Rozne "id" w pliku jest wyrownywane do nazwy
   pliku (stare id trafia do aliasow), chyba ze wskazuje na INNY istniejacy plik - wtedy plik zawiera dane innej
   rosliny (jak tarnina.json z danymi poziomki) i zapis jest zatrzymany do recznej poprawki.
3. Bezpiecznik: odmowa kontroli jest zapisywana przy roslinie (powod, licznik, odcisk pliku). Po GATE_TRIPS
   odmowach z tego samego powodu roslina jest pomijana przez zadania i planiste, dopoki plik sie nie zmieni
   (albo nie minie GATE_BLOCK_HOURS) - koniec zlecania w kolko tej samej pracy, ktora i tak nie trafi na strone.
"""
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import Plant, PlantAlias, PlantFact

log = logging.getLogger(__name__)
SIEDZIBA_BLOCKS: tuple[str, ...] = ("wiedza", "zdjecia_wiki", "scalone", "rozmieszczenie")
GATE_TRIPS = 2
GATE_BLOCK_HOURS = 24


def is_siedziba_block(key: str) -> bool:
    return key in SIEDZIBA_BLOCKS


def _plants_dir(plants_dir: Path | None = None) -> Path:
    return Path(plants_dir or settings.kwiatownik_plants_dir)


def file_sig(pid: str, plants_dir: Path | None = None) -> str:
    path = _plants_dir(plants_dir) / f"{pid}.json"
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _alias_name(old_id: str) -> str:
    return str(old_id).replace("_", " ").strip().lower()


def align_identity(cand: dict, pid: str, plants_dir: Path | None = None) -> tuple[list[str], str | None]:
    """Wyrownuje "id" kopii do nazwy pliku. Zwraca (bledy, notatka do dziennika)."""
    cid = cand.get("id")
    if not cid or cid == pid:
        return [], None
    if (_plants_dir(plants_dir) / f"{cid}.json").exists():
        return [f"plik {pid}.json zawiera dane innej rosliny ({cid}.json) - popraw go recznie w Kwiatowniku"], None
    cand["id"] = pid
    try:
        with Session(engine) as s:
            s.merge(PlantAlias(name=_alias_name(cid), plant_id=pid))
            s.commit()
    except Exception:  # noqa: BLE001 - alias to tylko udogodnienie, nie blokuje zapisu
        pass
    return [], f"id w pliku wyrownane do nazwy pliku: {cid} -> {pid} (stare id zapisane jako alias)"


def _reason_key(errors: list[str]) -> str:
    """Powod bez numerow i szczegolow - zeby ta sama przyczyna liczyla sie jako ta sama."""
    first = (errors or ["?"])[0]
    if first.startswith("s") and first[1:2].isdigit() and ":" in first:      # "s123: zla dlugosc tekstu"
        first = first.split(":", 1)[1].strip()
    return first[:200]


def _safe(fn, default):
    """Bezpiecznik nie moze zatrzymac zapisu wiedzy (np. baza w trakcie migracji) - wtedy dziala jak wylaczony."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.warning("Brama zapisu: %s", exc)
        return default


def record_rejection(pid: str, errors: list[str], plants_dir: Path | None = None) -> int:
    """Zapisuje odmowe bramy przy roslinie. Zwraca licznik odmow z rzedu z tego samego powodu."""
    return _safe(lambda: _record_rejection(pid, errors, plants_dir), 0)


def _record_rejection(pid: str, errors: list[str], plants_dir: Path | None = None) -> int:
    reason = _reason_key(errors)
    with Session(engine) as s:
        row = s.get(Plant, pid)
        if row is None:
            return 0
        same = row.gate_reason == reason and row.gate_sig == file_sig(pid, plants_dir)
        row.gate_count = (row.gate_count or 0) + 1 if same else 1
        row.gate_reason, row.gate_sig, row.gate_at = reason, file_sig(pid, plants_dir), datetime.now(timezone.utc)
        s.add(row)
        s.commit()
        return row.gate_count


def record_success(pid: str) -> None:
    _safe(lambda: _record_success(pid), None)


def _record_success(pid: str) -> None:
    with Session(engine) as s:
        row = s.get(Plant, pid)
        if row is not None and (row.gate_reason or row.gate_count):
            row.gate_reason, row.gate_count, row.gate_sig, row.gate_at = None, 0, None, None
            s.add(row)
            s.commit()


def _is_blocked(row: Plant, plants_dir: Path | None = None, now: datetime | None = None) -> bool:
    if not row.gate_reason or (row.gate_count or 0) < GATE_TRIPS:
        return False
    at = row.gate_at if row.gate_at is None or row.gate_at.tzinfo else row.gate_at.replace(tzinfo=timezone.utc)
    if at and (now or datetime.now(timezone.utc)) - at > timedelta(hours=GATE_BLOCK_HOURS):
        return False
    return row.gate_sig == file_sig(row.id, plants_dir)


def blocked(pid: str, plants_dir: Path | None = None) -> str | None:
    """Powod blokady albo None (roslina moze byc opracowywana)."""
    def run():
        with Session(engine) as s:
            row = s.get(Plant, pid)
        return row.gate_reason if row is not None and _is_blocked(row, plants_dir) else None
    return _safe(run, None)


def blocked_map(plants_dir: Path | None = None) -> dict[str, str]:
    def run():
        with Session(engine) as s:
            rows = s.exec(select(Plant).where(col(Plant.gate_reason).is_not(None))).all()
        return {r.id: r.gate_reason for r in rows if _is_blocked(r, plants_dir)}
    return _safe(run, {})


def summary(plants_dir: Path | None = None) -> dict:
    """Wiedza zatrzymana w bramie: rosliny z ostatnia odmowa i zweryfikowane informacje, ktore na nie czekaja."""
    return _safe(lambda: _summary(plants_dir), {"rosliny": 0, "zablokowane": 0, "informacje": 0, "lista": []})


def _summary(plants_dir: Path | None = None) -> dict:
    with Session(engine) as s:
        rows = s.exec(select(Plant).where(col(Plant.gate_reason).is_not(None))).all()
        waiting = dict(s.exec(select(PlantFact.plant_id, func.count()).where(PlantFact.status == "verified")
                              .group_by(PlantFact.plant_id)).all())
    items = [{"id": r.id, "nazwa": r.nazwa_pl, "powod": r.gate_reason, "odmow": r.gate_count or 0,
              "zablokowana": _is_blocked(r, plants_dir), "informacje": int(waiting.get(r.id, 0)),
              "kiedy": r.gate_at} for r in rows]
    items.sort(key=lambda x: (-x["informacje"], x["nazwa"] or ""))
    return {"rosliny": len(items), "zablokowane": sum(1 for x in items if x["zablokowana"]),
            "informacje": sum(x["informacje"] for x in items), "lista": items}
