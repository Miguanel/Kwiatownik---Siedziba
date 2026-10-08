"""Codzienne automatyczne skanowanie zrodel z wlaczona opcja 'auto' (o AUTO_SCAN_TIME, czas lokalny)."""
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import settings

log = logging.getLogger(__name__)


def seconds_until(hhmm: str, tz: str, now: datetime | None = None) -> float:
    h, m = (int(x) for x in hhmm.split(":"))
    now = now or datetime.now(ZoneInfo(tz))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def auto_scan_loop(state) -> None:
    if not settings.auto_scan_time.strip():
        return
    from app.worker.actions import scan_all
    while True:
        await asyncio.sleep(seconds_until(settings.auto_scan_time, settings.timezone))
        try:
            ids = scan_all(state, only_auto=True)
            log.info("Automatyczny skan: uruchomiono %d zadan", len(ids))
        except Exception:
            log.exception("Automatyczny skan nieudany")
        await asyncio.sleep(61)  # nie odpal dwa razy w tej samej minucie
