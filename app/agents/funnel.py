"""Miernik lejka wiedzy (ekspertyza 2026-10-14): ile wiedzy jest na kazdym etapie drogi od zebrania do strony.

zebrane (new) -> sprawdzone (verified) -> [brama zapisu] -> w plikach roslin (applied) -> commit -> strona.
Planista (R9) i panel /commit czytaja te liczby: system steruje sie wiedza, ktora dotarla na strone,
a nie sama praca wykonana w tle.
"""
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, col, func, select

from app.db import engine
from app.models import PlantFact


def counts() -> dict:
    with Session(engine) as s:
        rows = dict(s.exec(select(PlantFact.status, func.count()).group_by(PlantFact.status)).all())
        week = datetime.now(timezone.utc) - timedelta(days=7)
        fresh = s.exec(select(func.count()).select_from(PlantFact).where(col(PlantFact.created_at) >= week)).one()
    return {"zebrane": int(rows.get("new", 0)), "sprawdzone": int(rows.get("verified", 0)),
            "w_plikach": int(rows.get("applied", 0)), "odrzucone": int(rows.get("rejected", 0)),
            "zebrane_7d": int(fresh)}


def snapshot(pending: dict | None = None) -> dict:
    """Lejek + brama + to, co czeka na commit (z deploy.pending)."""
    from app.knowledge import gate
    c = counts()
    g = gate.summary()
    z = (pending or {}).get("zmiany") or {}
    c.update(w_bramie=g["informacje"], rosliny_w_bramie=g["zablokowane"], czeka_na_commit=int(z.get("informacje") or 0))
    return c
