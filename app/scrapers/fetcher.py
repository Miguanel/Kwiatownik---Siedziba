"""Uprzejmy, asynchroniczny pobieracz stron: robots.txt, opoznienia per host, adaptacja do 429."""
import asyncio
import gzip
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger(__name__)

TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_", "ref", "share", "replytocom", "amp")
SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip", ".mp3", ".mp4",
                   ".avi", ".mov", ".css", ".js", ".ico", ".woff", ".woff2", ".ttf", ".xml", ".rss", ".json")


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Absolutny URL bez #fragmentu i parametrow sledzacych; None dla nie-http(s)."""
    try:
        p = urlparse(urljoin(base, url) if base else url)
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(TRACKING_PARAMS)]
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path or "/", "", urlencode(query), ""))


def site_key(url: str) -> str:
    """Host bez 'www.' - do sprawdzania, czy link jest w obrebie tej samej strony."""
    return urlparse(url).netloc.lower().removeprefix("www.")


_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([\w-]+)""", re.I)


def sniff_encoding(body: bytes, content_type: str = "") -> str:
    """Kodowanie strony: naglowek HTTP, potem <meta charset> (strony chinskie i japonskie czesto podaja
    GBK / Shift_JIS tylko w HTML), na koncu UTF-8."""
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    if m:
        return m.group(1).lower()
    m = _META_CHARSET.search(body[:4096])
    enc = m.group(1).decode("ascii", "ignore").lower() if m else "utf-8"
    return {"gb2312": "gb18030", "gbk": "gb18030", "x-sjis": "shift_jis", "sjis": "shift_jis"}.get(enc, enc)


@dataclass
class FetchResult:
    url: str                 # adres po przekierowaniach
    status: int
    text: str = ""
    content_type: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300


class Fetcher:
    def __init__(self, user_agent: str, delay_s: float = 2.0, client: httpx.AsyncClient | None = None,
                 timeout: float = 30, max_bytes: int = 4_000_000, respect_robots: bool = True,
                 proxy: str | None = None, accept_language: str = "en,de;q=0.8,*;q=0.5"):
        self.user_agent = user_agent
        self.base_delay = delay_s
        self.max_bytes = max_bytes
        self.respect_robots = respect_robots
        self.proxy = proxy
        # proxy = serwer VPN kraju (np. kontener gluetun), przez ktory idzie caly ruch tego pobieracza
        self._client = client or httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, proxy=proxy,
            headers={"User-Agent": user_agent, "Accept-Language": accept_language})
        self._robots: dict[str, RobotFileParser | None] = {}
        self._host_delay: dict[str, float] = {}
        self._host_next: dict[str, float] = {}
        self._host_lock: dict[str, asyncio.Lock] = {}
        self.requests = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    # -------------------------------------------------------------- robots.txt
    async def robots(self, url: str) -> RobotFileParser | None:
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            rp = RobotFileParser()
            res = await self._raw_get(f"{host}/robots.txt")
            if res.status in (401, 403):
                rp.disallow_all = True
            elif res.ok:
                rp.parse(res.text.splitlines())
            else:
                rp.allow_all = True
            self._robots[host] = rp
            crawl_delay = rp.crawl_delay(self.user_agent) if res.ok else None
            if crawl_delay:
                self._host_delay[p.netloc] = max(self.base_delay, float(crawl_delay))
        return self._robots[host]

    async def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = await self.robots(url)
        return rp is None or rp.can_fetch(self.user_agent, url)

    async def sitemaps_from_robots(self, url: str) -> list[str]:
        rp = await self.robots(url)
        return list(rp.site_maps() or []) if rp else []

    # -------------------------------------------------------------- pobieranie
    async def _wait_turn(self, host: str) -> None:
        lock = self._host_lock.setdefault(host, asyncio.Lock())
        async with lock:
            delay = self._host_delay.get(host, self.base_delay)
            wait = self._host_next.get(host, 0) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_next[host] = time.monotonic() + delay

    async def _raw_get(self, url: str) -> FetchResult:
        host = urlparse(url).netloc
        for attempt in range(3):
            await self._wait_turn(host)
            self.requests += 1
            try:
                r = await self._client.get(url)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    return FetchResult(url, 0, error=f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(3 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code == 503:
                # strona prosi o wolniejsze tempo -> zwalniamy na stale dla tego hosta
                self._host_delay[host] = min(self._host_delay.get(host, self.base_delay) * 2, 60)
                ra = r.headers.get("retry-after", "")
                await asyncio.sleep(min(float(ra), 60) if ra.isdigit() else 10)
                continue
            if r.status_code >= 500 and attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
                continue
            body = r.content[: self.max_bytes]
            if body[:2] == b"\x1f\x8b":  # sitemap.xml.gz
                try:
                    body = gzip.decompress(body)
                except OSError:
                    pass
            encoding = sniff_encoding(body, r.headers.get("content-type", ""))
            try:
                text = body.decode(encoding, errors="replace")
            except LookupError:
                text = body.decode("utf-8", errors="replace")
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            err = None if r.status_code < 400 else f"HTTP {r.status_code}"
            return FetchResult(str(r.url), r.status_code, text, ctype, err)
        return FetchResult(url, 429, error="HTTP 429 - strona ogranicza ruch")

    async def get(self, url: str, html_only: bool = True) -> FetchResult:
        if not await self.allowed(url):
            return FetchResult(url, 0, error="zablokowane przez robots.txt")
        res = await self._raw_get(url)
        if res.ok and html_only and res.content_type and "html" not in res.content_type:
            res.error = f"pominieto typ {res.content_type}"
        return res
