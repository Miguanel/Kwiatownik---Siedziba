"""Licznik tokenow LLM wedlug celu (etap 6) i dzienny budzet chmury.

Cel zapytania rozpoznawany jest po poczatku promptu systemowego (kazde zadanie ma swoj). Tokeny liczone
w przyblizeniu: znaki / 4 (prompt + odpowiedz). Zapis: data/llm_usage.json, 14 ostatnich dni.
Budzet (LLM_DAILY_BUDGET, np. "klasyfikacja=150000,zapytania=30000"): gdy cel zuzyl dzis tyle tokenow w chmurze,
kolejne zapytania tego celu ida tylko do modelu lokalnego (Ollama) - limity Groq/Gemini zostaja dla waznych zadan.
"""
import json
import logging
import threading
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

# (fragment promptu systemowego, cel) - pierwszy pasujacy wygrywa
PURPOSES = [
    ("WSKAZUJESZ fragmenty", "wiedza"),
    ("badaczem etnobotaniki", "magia"),
    ("weryfikatorem faktow", "weryfikacja"),
    ("Piszesz na nowo JEDEN podrozdzial", "scalanie"),
    ("Strona rosliny ma rozdzialy", "rozmieszczenie"),
    ("Dostajesz ponumerowane informacje", "redakcja_wiedzy"),
    ("Wyszukujesz przepis", "przepisy"),
    ("redaktorem polskiego serwisu", "przepisy"),
    ("Klasyfikujesz przepis", "typ_przepisu"),
    ("Porownaj dwa przepisy", "duplikaty"),
    ("Czytasz artykul o roslinach", "wzmianki"),
    ("You classify web pages", "klasyfikacja"),
    ("web scraper profile", "profil_scrapera"),
    ("web search queries", "zapytania"),
    ("nazwy skladnikow", "nowe_rosliny"),
    ("CEL KWIATOWNIKA", "audyt"),
]
LABELS = {"wiedza": "wiedza o roslinach (cytaty)", "magia": "magia roslin (cytaty)", "weryfikacja": "weryfikacja wiedzy",
          "scalanie": "scalanie z rozdzialami", "rozmieszczenie": "przydzial do rozdzialow",
          "redakcja_wiedzy": "redakcja wiedzy", "przepisy": "przepisy (fragmenty)", "typ_przepisu": "typ przepisu",
          "duplikaty": "porownanie przepisow", "wzmianki": "wzmianki o przepisach", "klasyfikacja": "rodzaj strony",
          "profil_scrapera": "profil scrapera", "zapytania": "zapytania wyszukiwarki", "nowe_rosliny": "nowe rosliny",
          "audyt": "audyt", "marketing": "marketing", "inne": "inne"}
KEEP_DAYS = 14


def purpose_of(system: str | None, prompt: str = "") -> str:
    s = system or ""
    for marker, purpose in PURPOSES:
        if marker in s:
            return purpose
    if "post" in s.lower() and ("facebook" in s.lower() or "instagram" in s.lower() or "marketing" in s.lower()):
        return "marketing"
    return "inne"


def tokens(*texts: str | None) -> int:
    return max(1, sum(len(t or "") for t in texts) // 4)


def parse_budget(raw: str | None) -> dict[str, int]:
    out = {}
    for part in (raw or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = int(v.strip().replace("_", ""))
            except ValueError:
                continue
    return out


class LLMUsage:
    def __init__(self, path: Path | None, budget: dict[str, int] | None = None):
        self.path = Path(path) if path else None
        self.budget = budget or {}
        self._lock = threading.Lock()
        self._data: dict = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}

    def add(self, purpose: str, n_tokens: int, local: bool, today: date | None = None) -> None:
        day = (today or date.today()).isoformat()
        with self._lock:
            d = self._data.setdefault(day, {}).setdefault(purpose, {"chmura": 0, "lokalnie": 0, "zapytania": 0})
            d["lokalnie" if local else "chmura"] += n_tokens
            d["zapytania"] += 1
            for old in sorted(self._data)[:-KEEP_DAYS]:
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
            log.warning("Licznik tokenow: nie zapisano (%s)", exc)

    def cloud_today(self, purpose: str, today: date | None = None) -> int:
        day = (today or date.today()).isoformat()
        return int(((self._data.get(day) or {}).get(purpose) or {}).get("chmura", 0))

    def over_budget(self, purpose: str, today: date | None = None) -> bool:
        limit = self.budget.get(purpose)
        return bool(limit) and self.cloud_today(purpose, today) >= limit

    def table(self, today: date | None = None) -> list[dict]:
        """Wiersze dla panelu: dzis i 7 dni, wg celu (najwiecej tokenow na gorze)."""
        t = today or date.today()
        days = sorted(self._data)[-7:]
        rows: dict[str, dict] = {}
        for day in days:
            for purpose, v in (self._data.get(day) or {}).items():
                r = rows.setdefault(purpose, {"cel": purpose, "nazwa": LABELS.get(purpose, purpose), "dzis_chmura": 0,
                                              "dzis_lokalnie": 0, "dzis_zapytania": 0, "tydzien": 0,
                                              "budzet": self.budget.get(purpose)})
                r["tydzien"] += v.get("chmura", 0) + v.get("lokalnie", 0)
                if day == t.isoformat():
                    r["dzis_chmura"] += v.get("chmura", 0)
                    r["dzis_lokalnie"] += v.get("lokalnie", 0)
                    r["dzis_zapytania"] += v.get("zapytania", 0)
        return sorted(rows.values(), key=lambda r: -(r["dzis_chmura"] + r["dzis_lokalnie"] + r["tydzien"] / 100))
