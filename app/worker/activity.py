"""Biezaca aktywnosc zadan w tle (w pamieci, bez bazy) - co dokladnie robi teraz kazde zadanie.

Zadania wolaja set(job_id, "pobieram", url); widok "Na zywo" czyta snapshot().
"""
import time
from collections import deque

_ACT: dict[int, dict] = {}
_EVENTS: deque = deque(maxlen=200)   # ostatnie kroki wszystkich zadan (os czasu na zywo)
_BEAT: dict[int, float] = {}         # ostatni znak zycia zadania (krok, wpis w logu, zapytanie do LLM)


def beat(job_id: int | None) -> None:
    """Zadanie zyje (planista ciaglosci wykrywa zadania bez znaku zycia - zawieszone)."""
    if job_id is not None:
        _BEAT[job_id] = time.time()


def last_beat(job_id: int) -> float | None:
    return _BEAT.get(job_id)


def set(job_id: int | None, step: str, detail: str = "") -> None:  # noqa: A001 - krotka nazwa celowo
    if job_id is None:
        return
    now = time.time()
    _BEAT[job_id] = now
    prev = _ACT.get(job_id)
    _ACT[job_id] = {"step": step, "detail": detail[:300], "since": now,
                    "count": (prev["count"] + 1) if prev else 1}
    _EVENTS.append({"ts": now, "job_id": job_id, "step": step, "detail": detail[:200]})


def clear(job_id: int) -> None:
    _ACT.pop(job_id, None)
    _BEAT.pop(job_id, None)


def get(job_id: int) -> dict | None:
    a = _ACT.get(job_id)
    return {**a, "age": round(time.time() - a["since"], 1)} if a else None


def snapshot() -> dict[int, dict]:
    return {j: get(j) for j in list(_ACT)}


def recent_events(n: int = 40) -> list[dict]:
    return list(_EVENTS)[-n:][::-1]
