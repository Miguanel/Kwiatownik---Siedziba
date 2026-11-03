"""Nazwy roslin katalogu w obcych jezykach z Wikidata (bez LLM) - dla tlumacza maszynowego i kontroli tlumaczenia.

Badanie rosliny (zadanie "plant_research") i tak zapisuje etykiety; ta funkcja uzupelnia nazwy roslin jeszcze
nie badanych - jedno zapytanie wyszukiwania + jedno o encje na rosline (po nazwie lacinskiej), bez czytania
artykulow. Zapisuje: Plant.names_json (etykiety), aliases_json (inne nazwy), wiki_titles_json (tytuly artykulow,
jesli ich nie bylo), wikidata, names_checked_at. Rosliny sprawdzone wczesniej niz NAMES_REFRESH_DAYS temu pomija.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, col, or_, select

from app.models import Plant

NAMES_REFRESH_DAYS = 90
MAX_NET_ERRORS = 3


def plants_missing_names(engine, limit: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=NAMES_REFRESH_DAYS)
    with Session(engine) as s:
        rows = s.exec(select(Plant.id).where(
            col(Plant.nazwa_lat).is_not(None),
            or_(col(Plant.names_checked_at).is_(None), col(Plant.names_checked_at) < cutoff),
            or_(col(Plant.names_json).is_(None), col(Plant.aliases_json).is_(None)))
            .order_by(col(Plant.origin) != "kwiatownik", col(Plant.id)).limit(limit)).all()
    return list(rows)


async def refresh_plant_names(engine, wiki, log, limit: int = 40) -> dict:
    """Uzupelnia nazwy do `limit` roslin. Bledy sieci: po MAX_NET_ERRORS z rzedu - koniec (sprobuje nastepnym razem)."""
    stats = {"checked": 0, "found": 0, "errors": 0}
    net_errors = 0
    for pid in plants_missing_names(engine, limit):
        with Session(engine) as s:
            p = s.get(Plant, pid)
            latin, name_pl = p.nazwa_lat, p.nazwa_pl
        try:
            wp = await wiki.find_plant(latin, name_pl)
        except Exception as exc:  # noqa: BLE001 - siec / Wikidata niedostepna
            stats["errors"] += 1
            net_errors += 1
            if net_errors >= MAX_NET_ERRORS:
                log(f"Nazwy roslin: Wikidata niedostepna ({str(exc)[:100]}) - sprobuje przy nastepnym tlumaczeniu")
                break
            continue
        net_errors = 0
        stats["checked"] += 1
        with Session(engine) as s:
            row = s.get(Plant, pid)
            row.names_checked_at = datetime.now(timezone.utc)
            if wp:
                stats["found"] += 1
                row.wikidata = row.wikidata or wp.qid
                if wp.labels:
                    row.names_json = json.dumps(wp.labels, ensure_ascii=False)
                row.aliases_json = json.dumps(wp.aliases or {}, ensure_ascii=False)
                if wp.titles and not row.wiki_titles_json:
                    row.wiki_titles_json = json.dumps(wp.titles, ensure_ascii=False)
            s.add(row)
            s.commit()
        await asyncio.sleep(0)
    if stats["checked"]:
        log(f"Nazwy roslin z Wikidata: sprawdzono {stats['checked']}, znaleziono {stats['found']}")
    return stats
