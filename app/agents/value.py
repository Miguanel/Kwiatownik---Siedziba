"""Wartosc informacji (R9 z ekspertyzy 2026-10-14): ile pracy, ktora DOTARLA do plikow i bazy, daje minuta zadania
danego rodzaju. Planista dokleja do oceny propozycji premie z wielorekiego bandyty (UCB1): rodzaje, ktore dawaly
najwiecej na minute, ida wyzej, a rzadko probowane dostaja premie za ciekawosc - zeby nie utknac na jednym.
"""
import json
import math
from datetime import timedelta

from sqlmodel import Session, col, select

from app.agents import store
from app.db import engine
from app.models import AgentTask, Job

# co liczy sie jako efekt (waga = jak blisko strony): zapis do pliku rosliny > przepis > sprawdzona informacja
VALUE_KEYS = {"applied": 3.0, "merged": 2.0, "placed": 2.0, "organized": 1.0, "verified": 1.0, "recipe": 2.0,
              "exported": 2.0, "translated": 1.0, "enriched": 0.5, "fact": 0.3}
WINDOW_DAYS = 14
EXPLORE = 0.5
MAX_BONUS = 10.0


def job_value(stats_json: str | None) -> float:
    try:
        st = json.loads(stats_json) if stats_json else {}
    except ValueError:
        return 0.0
    if not isinstance(st, dict):
        return 0.0
    total = 0.0
    for k, w in VALUE_KEYS.items():
        v = st.get(k)
        if isinstance(v, (int, float)):
            total += w * v
    return total


def kind_yields(days: int = WINDOW_DAYS) -> dict[str, dict]:
    """{rodzaj: {"n", "na_minute"}} z wykonanych (i zablokowanych) zlecen agentow z ostatnich `days` dni."""
    since = store.now() - timedelta(days=days)
    out: dict[str, dict] = {}
    with Session(engine) as s:
        rows = s.exec(select(AgentTask, Job).where(AgentTask.job_id == Job.id,
                                                   col(AgentTask.status).in_(["done", "blocked"]),
                                                   col(AgentTask.created_at) >= since)).all()
    for t, j in rows:
        if not (j.started_at and j.finished_at):
            continue
        minutes = max(1.0, (store.utc(j.finished_at) - store.utc(j.started_at)).total_seconds() / 60)
        d = out.setdefault(t.kind, {"n": 0, "wartosc": 0.0, "minuty": 0.0})
        d["n"] += 1
        d["wartosc"] += job_value(j.stats_json)
        d["minuty"] += minutes
    for d in out.values():
        d["na_minute"] = round(d["wartosc"] / d["minuty"], 4) if d["minuty"] else 0.0
    return out


def bonus(kind: str, yields: dict[str, dict]) -> float:
    """Premia UCB1 do oceny propozycji: od -MAX_BONUS/2 do +MAX_BONUS."""
    if not yields:
        return 0.0
    best = max((d["na_minute"] for d in yields.values()), default=0.0) or 1.0
    total = sum(d["n"] for d in yields.values())
    d = yields.get(kind)
    n = d["n"] if d else 0
    mean = (d["na_minute"] / best) if d else 0.5          # nieznany rodzaj: srodek skali
    ucb = mean + EXPLORE * math.sqrt(math.log(total + 1) / (n + 1))
    return round(max(-MAX_BONUS / 2, min(MAX_BONUS, MAX_BONUS * (ucb - 0.5))), 2)
