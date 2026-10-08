"""Magazyn przepisow Kwiatownika (folder Kwiatownik2/data/przepisy) - wspolna konwencja z Kwiatownikiem2.

Pliki maja nazwy <seria>_<RRRR-MM-DD>.json (np. siedziba_przepisy_2026-10-05.json, opcjonalnie z godzina:
siedziba_przepisy_2026-10-05_1430.json). Kazdy eksport to pelny stan serii, wiec z kazdej serii czytany jest
tylko NAJNOWSZY plik; starsze zostaja jako historia. Plik bez daty (np. przepisy_medyczne.json) to seria
sama w sobie (stary format) - przegrywa z kazdym plikiem tej serii z data.
Ta sama logika jest w Kwiatownik2/app/utils/recipe_store.py - zmieniajac jedna, zmien druga.
"""
import re
from datetime import datetime
from pathlib import Path

SKIP = {"wzorzec_przepisu.json"}
DATED = re.compile(r"^(?P<series>.+?)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>\d{4,6}))?_?\.json$")


def parse_name(name: str) -> tuple[str, str]:
    """'siedziba_przepisy_2026-10-05.json' -> ('siedziba_przepisy', '2026-10-05'); bez daty -> (stem, '')."""
    m = DATED.match(name)
    if m:
        return m.group("series"), m.group("date") + (f"_{m.group('time')}" if m.group("time") else "")
    return name[:-5] if name.endswith(".json") else name, ""


def series_of(filename: str) -> str:
    """Seria z nazwy pliku z ustawien (EXPORT_FILENAME=siedziba_przepisy.json -> siedziba_przepisy)."""
    return parse_name(Path(filename).name)[0]


def dated_name(series: str, when: datetime | None = None) -> str:
    return f"{series}_{(when or datetime.now()).strftime('%Y-%m-%d')}.json"


def current_files(folder: Path) -> list[Path]:
    """Pliki, ktore Kwiatownik wczytuje: najnowszy z kazdej serii (alfabetycznie po serii)."""
    best: dict[str, tuple[str, Path]] = {}
    d = Path(folder)
    if not d.exists():
        return []
    for f in d.glob("*.json"):
        if f.name in SKIP:
            continue
        series, stamp = parse_name(f.name)
        if series not in best or stamp > best[series][0]:
            best[series] = (stamp, f)
    return [best[s][1] for s in sorted(best)]


def all_files(folder: Path) -> list[dict]:
    """Wszystkie pliki magazynu z informacja, czy sa czytane (do panelu)."""
    current = {f.name for f in current_files(folder)}
    out = []
    for f in sorted(Path(folder).glob("*.json")) if Path(folder).exists() else []:
        if f.name in SKIP:
            continue
        series, stamp = parse_name(f.name)
        out.append({"name": f.name, "series": series, "date": stamp, "used": f.name in current,
                    "size": f.stat().st_size})
    return out
