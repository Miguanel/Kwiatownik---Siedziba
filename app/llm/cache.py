"""Pamiec podreczna odpowiedzi LLM (R4 z ekspertyzy 2026-10-14).

Ponowienie zadania przerwanego restartem wysyla do modeli te same zapytania (ta sama strona, ta sama roslina).
Udana odpowiedz jest zapisywana pod skrotem (system, prompt, tryb JSON, modele preferowane/unikane) w osobnej bazie
data/llm_cache.db, wiec powtorzone zapytanie kosztuje 0 s i nie zuzywa limitow. Domyslnie tylko zapytania JSON
(wyciaganie, weryfikacja, klasyfikacja) - teksty "tworcze" (posty na fanpage) maja byc za kazdym razem nowe.
Wpisy starsze niz LLM_CACHE_DAYS sa usuwane przy starcie.
"""
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class LLMCache:
    def __init__(self, path: Path, days: float = 14.0):
        self.path = Path(path)
        self.ttl = days * 86400
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS odpowiedzi (klucz TEXT PRIMARY KEY, tekst TEXT NOT NULL, "
                      "provider TEXT, model TEXT, czas REAL NOT NULL)")
            c.execute("DELETE FROM odpowiedzi WHERE czas < ?", (time.time() - self.ttl,))

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    @staticmethod
    def key(prompt: str, system: str | None, json_mode: bool, prefer=None, avoid=None, cloud_only=False) -> str:
        raw = json.dumps([system or "", prompt, bool(json_mode), list(prefer or []), sorted(avoid or []),
                          bool(cloud_only)], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[str, str, str] | None:
        try:
            with self._lock, self._conn() as c:
                row = c.execute("SELECT tekst, provider, model, czas FROM odpowiedzi WHERE klucz = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            log.warning("Pamiec LLM: %s", exc)
            return None
        if row is None or time.time() - row[3] > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        return row[0], row[1], row[2]

    def put(self, key: str, text: str, provider: str, model: str) -> None:
        try:
            with self._lock, self._conn() as c:
                c.execute("INSERT OR REPLACE INTO odpowiedzi VALUES (?, ?, ?, ?, ?)",
                          (key, text, provider, model, time.time()))
        except sqlite3.Error as exc:
            log.warning("Pamiec LLM: %s", exc)

    def stats(self) -> dict:
        try:
            with self._conn() as c:
                n = c.execute("SELECT COUNT(*) FROM odpowiedzi").fetchone()[0]
        except sqlite3.Error:
            n = 0
        return {"wpisy": n, "trafienia": self.hits, "chybienia": self.misses}
