"""Wyszukiwarki: SearXNG (lokalny kontener, laczy Google/Bing/DuckDuckGo/Brave...) i opcjonalnie Brave Search API."""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    engines: list[str] = field(default_factory=list)


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, language: str, country: str, page: int = 1) -> list[SearchResult]: ...


class SearxngSearch:
    name = "searxng"

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=30)

    async def search(self, query: str, language: str, country: str, page: int = 1) -> list[SearchResult]:
        r = await self.client.get(f"{self.base_url}/search", params={
            "q": query, "format": "json", "language": language, "pageno": page, "categories": "general",
            "safesearch": 0})
        if r.status_code == 403:
            raise RuntimeError("SearXNG odrzuca format JSON - wlacz 'json' w searxng/settings.yml (search.formats)")
        r.raise_for_status()
        return [SearchResult(x.get("url", ""), x.get("title", ""), x.get("content", ""),
                             x.get("engines") or [x.get("engine", "")])
                for x in r.json().get("results", []) if x.get("url")]

    async def aclose(self) -> None:
        await self.client.aclose()


class BraveSearch:
    name = "brave"
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=30)

    async def search(self, query: str, language: str, country: str, page: int = 1) -> list[SearchResult]:
        params = {"q": query, "count": 20, "offset": page - 1, "search_lang": language.split("-")[0],
                  "country": country.upper()}
        r = await self.client.get(self.URL, params=params, headers={
            "X-Subscription-Token": self.api_key, "Accept": "application/json"})
        if r.status_code == 422:  # nieobslugiwany kraj/jezyk -> sprobuj bez nich
            params.pop("country")
            params.pop("search_lang")
            r = await self.client.get(self.URL, params=params, headers={"X-Subscription-Token": self.api_key})
        r.raise_for_status()
        return [SearchResult(x.get("url", ""), x.get("title", ""), x.get("description", ""), ["brave"])
                for x in (r.json().get("web") or {}).get("results", []) if x.get("url")]

    async def aclose(self) -> None:
        await self.client.aclose()


async def search_all(backends: list, query: str, language: str, country: str, pages: int = 1,
                     log_fn=None) -> list[SearchResult]:
    """Pyta wszystkie wyszukiwarki rownolegle i laczy wyniki (bez powtorzen adresow)."""
    async def one(b, page):
        try:
            return await b.search(query, language, country, page)
        except Exception as exc:
            if log_fn:
                log_fn(f"[{b.name}] blad wyszukiwania '{query}': {exc}")
            return []

    batches = await asyncio.gather(*(one(b, p) for b in backends for p in range(1, pages + 1)))
    merged: dict[str, SearchResult] = {}
    for batch in batches:
        for r in batch:
            if r.url in merged:
                merged[r.url].engines = sorted(set(merged[r.url].engines) | set(r.engines))
            else:
                merged[r.url] = r
    return list(merged.values())
