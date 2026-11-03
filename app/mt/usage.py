"""Licznik znakow wyslanych do tlumaczy maszynowych: dzien i miesiac, osobno przepisy i wiedza (data/mt_usage.json).

DeepL ma wlasny licznik (/v2/usage), Azure Translator nie - jego budzet liczy sie stad. Plik trzyma 3 ostatnie
miesiace; blad zapisu nigdy nie zatrzymuje tlumaczenia.
"""
import json
import logging
import threading
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


class UsageBook:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}

    @staticmethod
    def _keys(today: date | None = None) -> tuple[str, str]:
        d = today or date.today()
        return d.strftime("%Y-%m"), d.isoformat()

    def add(self, engine: str, purpose: str, chars: int, today: date | None = None) -> None:
        if chars <= 0:
            return
        month, day = self._keys(today)
        with self._lock:
            m = self._data.setdefault(month, {}).setdefault(engine, {"razem": 0, "wiedza": 0, "dni": {}})
            m["razem"] += chars
            if purpose != "przepis":
                m["wiedza"] += chars
            m["dni"][day] = m["dni"].get(day, 0) + chars
            for old in sorted(self._data)[:-3]:
                self._data.pop(old, None)
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("Licznik tlumaczy: nie zapisano (%s)", exc)

    def month(self, engine: str, purpose: str | None = None, today: date | None = None) -> int:
        month, _ = self._keys(today)
        m = (self._data.get(month) or {}).get(engine) or {}
        return int(m.get("wiedza" if purpose == "wiedza" else "razem", 0))

    def day(self, engine: str, today: date | None = None) -> int:
        month, day = self._keys(today)
        return int((((self._data.get(month) or {}).get(engine) or {}).get("dni") or {}).get(day, 0))

    def summary(self, today: date | None = None) -> dict[str, dict]:
        month, _ = self._keys(today)
        return {eng: {"dzis": self.day(eng, today), "miesiac": int(v.get("razem", 0)),
                      "wiedza": int(v.get("wiedza", 0))}
                for eng, v in (self._data.get(month) or {}).items()}
