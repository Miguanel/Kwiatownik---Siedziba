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
