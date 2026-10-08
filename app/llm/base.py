import asyncio
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx


# ---------- bledy (router decyduje na ich podstawie, jak ukarac model) ----------
class LLMError(Exception):
    cooldown_s: float = 30.0


class RateLimitError(LLMError):
    """429 - model chwilowo przeciazony / wyczerpany limit."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.cooldown_s = retry_after or 60.0


class QuotaExceededError(RateLimitError):
    """Wyczerpany limit dzienny / kwota konta - model nie odpowie przez dluzszy czas."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg, retry_after)
        self.cooldown_s = retry_after or 6 * 3600.0


class RequestTooLargeError(LLMError):
    """To zapytanie jest za duze dla limitu tego modelu (np. Groq 413 TPM) - pomin model tylko teraz."""
    cooldown_s = 0.0


class ServerError(LLMError):
    """5xx, timeout, zerwane polaczenie."""
    cooldown_s = 30.0


class ModelError(LLMError):
    """400/404/422 - model nie istnieje, nie obsluguje opcji (np. trybu JSON) itp."""
    cooldown_s = 30 * 60.0


class AuthError(LLMError):
    """401/403 - zly klucz; blokuje wszystkie modele providera."""
    cooldown_s = 60 * 60.0


class InvalidOutputError(LLMError):
    """Pusta odpowiedz albo niepoprawny JSON - kara bez cooldownu."""
    cooldown_s = 0.0


# ---------- rozpoznawanie limitow ----------
_RETRY_DELAY = re.compile(r'retryDelay"?\s*:\s*"(\d+(?:\.\d+)?)s"')
_TRY_AGAIN = re.compile(r"(?:try again|retry) in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)(ms|s))?", re.I)
_DAILY = ("per day", "perday", "(tpd)", "(rpd)", "daily")
_MINUTE = ("per minute", "perminute", "(tpm)", "(rpm)", "(otpm)", "(itpm)", "per second")


def parse_retry_seconds(text: str) -> float | None:
    """'retryDelay': '34s' (Gemini) albo 'Please try again in 1m23.4s' (Groq) -> sekundy."""
    m = _RETRY_DELAY.search(text or "")
    if m:
        return float(m.group(1))
    m = _TRY_AGAIN.search(text or "")
    if m and any(m.groups()):
        h, mnt, val, unit = m.groups()
        sec = int(h or 0) * 3600 + int(mnt or 0) * 60
        if val:
            sec += float(val) / (1000 if unit == "ms" else 1)
        return sec or None
    return None


def classify_limit(detail: str, body: str, retry: float | None) -> RateLimitError:
    """Limit na minute (krotka przerwa) czy wyczerpana kwota dzienna/konta (dluga przerwa)?"""
    low = (body or "").lower()
    if any(k in low for k in _DAILY) and not any(k in low for k in _MINUTE):
        return QuotaExceededError(detail, retry)
    if not any(k in low for k in _MINUTE) and ("exceeded your current quota" in low or "billing" in low) \
            and (retry is None or retry > 300):
        return QuotaExceededError(detail, retry)
    return RateLimitError(detail, retry or 60.0)


def short_error(body: str) -> str:
    """Z odpowiedzi JSON wyciaga sam komunikat bledu (zamiast calego JSON-a w logach)."""
    try:
        import json as _json
        data = _json.loads(body)
        err = data.get("error", data)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return (msg or body)[:220]
    except (ValueError, AttributeError):
        return (body or "")[:220]


# ---------- struktury ----------
@dataclass
class ModelInfo:
    provider: str
    model: str
    context_window: int | None = None


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    seconds: float
    attempts: list[str] = field(default_factory=list)  # nieudane proby przed sukcesem


class RateLimiter:
    """Max `rpm` startow na minute. delay() tylko sprawdza, reserve() rezerwuje slot."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0

    def delay(self) -> float:
        return max(0.0, self._next - time.monotonic()) if self.interval else 0.0

    def reserve(self) -> float:
        d = self.delay()
        self._next = max(time.monotonic(), self._next) + self.interval
        return d


# ---------- provider ----------
class LLMProvider(ABC):
    name: str = "base"
    fallback: bool = False   # True = model zapasowy (lokalny), uzywany dopiero gdy inne zawioda

    def __init__(self, api_key: str, rpm: int = 0, concurrency: int = 3,
                 client: httpx.AsyncClient | None = None, timeout: float = 120):
        self.api_key = api_key
        self.rpm = rpm
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self._client = client
        self._timeout = timeout

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(self, model: str, prompt: str, system: str | None = None,
                       json_mode: bool = False) -> str:
        async with self.semaphore:
            try:
                text = await self._generate(model, prompt, system, json_mode)
            except httpx.TransportError as exc:  # timeout, brak sieci
                raise ServerError(f"{self.name}: {type(exc).__name__}") from exc
        if not text or not text.strip():
            raise InvalidOutputError(f"{self.name}/{model}: pusta odpowiedz")
        return text

    def _check(self, r: httpx.Response) -> dict:
        code = r.status_code
        if code < 400:
            return r.json()
        detail = f"{self.name}: HTTP {code}: {short_error(r.text)}"
        if code == 413:
            raise RequestTooLargeError(detail)
        if code == 429:
            ra = r.headers.get("retry-after")
            retry = float(ra) if ra and ra.replace(".", "").isdigit() else parse_retry_seconds(r.text)
            raise classify_limit(detail, r.text, retry)
        if code in (401, 403):
            raise AuthError(detail)
        if code >= 500:
            raise ServerError(detail)
        raise ModelError(detail)

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    async def _generate(self, model: str, prompt: str, system: str | None, json_mode: bool) -> str: ...
