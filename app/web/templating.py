from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.templating import Jinja2Templates

from app.config import settings
from app.worker import snapshots

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
_TZ = ZoneInfo(settings.timezone)


def _dt(value: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if not value:
        return "-"
    if value.tzinfo is None:  # SQLite zwraca daty bez strefy - zapisujemy je w UTC
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(_TZ).strftime(fmt)


templates.env.filters["dt"] = _dt
templates.env.globals["snapshot_history"] = snapshots.for_link   # historia zmian pozycji (przepis, roslina)
templates.env.globals["app_name"] = settings.app_name


# ------------------------------------------------------------------ menu (base.html)
# Zakladki w grupach (bez nazw - w pasku oddziela je cienka kreska), w kolejnosci pracy:
# 1. praca Siedziby, 2. zbieranie, 3. zebrana wiedza, 4. modele, 5. strona i promocja, 6. dla programisty.
NAV: list[list[tuple[str, str]]] = [
    [("/", "Panel"), ("/monitor", "Na zywo"), ("/tasks", "Do zrobienia"), ("/jobs", "Zadania"),
     ("/agents", "Agenci")],
    [("/sources", "Zrodla"), ("/scrapers", "Scrapery"), ("/discover", "Odkrywanie"),
     ("/encyklopedie", "Encyklopedie"), ("/wzmianki", "Wzmianki"), ("/magia", "Magia")],
    [("/plants", "Rosliny"), ("/items", "Zebrane"), ("/paleta", "Paleta"), ("/export", "Eksport")],
    [("/llm", "LLM"), ("/nn", "Sieci")],
    [("/site", "Strona"), ("/uklad", "Uklad strony"), ("/posts", "Fanpage")],
    [("/docs", "API")],
]
# strony spoza paska, ktore tez maja nazwe w naglowku
_NAV_EXTRA = [("/commit", "Commit")]


def nav_current(path: str | None) -> tuple[str, str] | None:
    """Zakladka, do ktorej nalezy adres (najdluzszy pasujacy prefiks; "/" tylko dokladnie)."""
    path = (path or "/").rstrip("/") or "/"
    best = None
    for href, label in [item for group in NAV for item in group] + _NAV_EXTRA:
        if path == href or (href != "/" and path.startswith(href + "/")):
            if best is None or len(href) > len(best[0]):
                best = (href, label)
    return best


templates.env.globals["nav_groups"] = NAV
templates.env.globals["nav_current"] = nav_current
