"""SiteFinder - automatyczne szukanie NOWYCH stron z przepisami w wybranym kraju.

1. LLM uklada zapytania w jezyku kraju (nie powtarza juz uzytych - pamiec w bazie).
2. Wyszukiwarki (SearXNG: Google/Bing/DuckDuckGo/..., opcjonalnie Brave API) zwracaja wyniki.
3. Wyniki sa sprowadzane do domen; odrzucamy serwisy spolecznosciowe/sklepy/encyklopedie,
   domeny juz znane (zrodla, wczesniej znalezione, odrzucone) - kazda strona tylko raz.
4. Nowa domena jest weryfikowana: pobieramy znaleziona podstrone (przez VPN kraju, jesli ustawiony)
   i sprawdzamy, czy to przepis (JSON-LD / heurystyki / LLM).
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from app.discovery.countries import Country
from app.discovery.domains import is_blocked, registrable_domain
from app.discovery.queries import make_queries
from app.discovery.search import SearchResult, search_all
from app.scrapers.classify import DEFAULT_KEYWORDS, classify, jsonld_recipe
from app.scrapers.fetcher import Fetcher
from app.scrapers.page import parse_page

log = logging.getLogger(__name__)


@dataclass
class FoundSite:
    domain: str
    country: str
    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    engines: list[str] = field(default_factory=list)
    status: str = "new"          # verified | maybe | rejected | unreachable
    score: float = 0.0
    language: str | None = None
    reason: str = ""


class DiscoveryStore(Protocol):
    def known_domains(self) -> set[str]: ...
    def used_queries(self, country: str) -> set[str]: ...
    def save_query(self, country: str, text: str, results: int, new_sites: int) -> None: ...
    def save_site(self, site: FoundSite) -> None: ...
    def bump_site(self, domain: str) -> None: ...
    def log(self, message: str) -> None: ...
    def progress(self, done: int, total: int) -> None: ...
    def should_stop(self) -> bool: ...
    def activity(self, step: str, detail: str = "") -> None: ...   # opcjonalne


@dataclass
class DiscoveryConfig:
    queries: int = 5
    pages_per_query: int = 1
    max_new_sites: int = 30
    verify: bool = True
    topic: str | None = None
    extra_queries: list[str] = field(default_factory=list)   # wlasne zapytania uzytkownika
    verify_concurrency: int = 4
    pause_between_queries_s: float = 2.0


class SiteFinder:
    def __init__(self, country: Country, backends: list, store: DiscoveryStore, fetcher: Fetcher,
                 config: DiscoveryConfig, llm=None, proxy_used: bool = False):
        self.country = country
        self.backends = backends
        self.store = store
        self.fetcher = fetcher
        self.cfg = config
        self.llm = llm
        self.proxy_used = proxy_used
        self.stats = {"queries": 0, "results": 0, "new_domains": 0, "known_skipped": 0, "blocked": 0,
                      "verified": 0, "maybe": 0, "rejected": 0, "unreachable": 0}

    def _act(self, step: str, detail: str = "") -> None:
        fn = getattr(self.store, "activity", None)
        if fn:
            fn(step, detail)

    async def run(self) -> dict:
        c = self.country
        self._act("LLM uklada zapytania", f"{c.name} ({c.language})")
        used = self.store.used_queries(c.code)
        used_low = {u.lower().strip() for u in used}
        own = [q for q in dict.fromkeys(self.cfg.extra_queries) if q.lower().strip() not in used_low]
        queries = own + await make_queries(
            c, self.cfg.queries, used, self.llm, self.cfg.topic)
        queries = queries[: max(self.cfg.queries, len(own))]
        if not queries:
            self.store.log("Brak nowych zapytan (wszystkie uzyte) - podaj wlasne albo wlacz LLM")
            return self.stats
        self.store.log(f"Kraj: {c.name} ({c.language}); zapytania: " + " | ".join(queries)
                       + ("; ruch przez VPN" if self.proxy_used else ""))

        known = self.store.known_domains()
        candidates: list[FoundSite] = []
        for i, q in enumerate(queries):
            if self.store.should_stop():
                break
            self._act(f"szukam {i + 1}/{len(queries)}", q)
            results = await search_all(self.backends, q, c.language, c.code, self.cfg.pages_per_query,
                                       log_fn=self.store.log)
            self.stats["queries"] += 1
            self.stats["results"] += len(results)
            new_here = self._collect(results, q, known, candidates)
            self.store.save_query(c.code, q, len(results), new_here)
            self.store.log(f"'{q}': {len(results)} wynikow, {new_here} nowych stron")
            self.store.progress(i + 1, len(queries))
            if len(candidates) >= self.cfg.max_new_sites:
                self.store.log(f"Osiagnieto limit {self.cfg.max_new_sites} nowych stron")
                break
            await asyncio.sleep(self.cfg.pause_between_queries_s)

        if self.cfg.verify and candidates:
            self.store.log(f"Weryfikacja {len(candidates)} stron...")
            sem = asyncio.Semaphore(self.cfg.verify_concurrency)

            async def check(site: FoundSite):
                async with sem:
                    # inne zadanie (np. szukanie w innym kraju) moglo ja juz znalezc lub dodac w miedzyczasie
                    if site.domain in self.store.known_domains():
                        self.stats["known_skipped"] += 1
                        self.stats["new_domains"] -= 1
                        return
                    if not self.store.should_stop():
                        self._act("sprawdzam strone", f"{site.domain} - {site.url}")
                        await self._verify(site)
                self.store.save_site(site)

            await asyncio.gather(*(check(s) for s in candidates))
        else:
            for s in candidates:
                self.store.save_site(s)
        self.store.log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in self.stats.items()))
        return self.stats

    def _collect(self, results: list[SearchResult], query: str, known: set[str], out: list[FoundSite]) -> int:
        new = 0
        for r in results:
            domain = registrable_domain(r.url)
            if is_blocked(domain):
                self.stats["blocked"] += 1
                continue
            if domain in known:
                self.stats["known_skipped"] += 1
                self.store.bump_site(domain)
                continue
            if len(out) >= self.cfg.max_new_sites:
                break
            known.add(domain)
            out.append(FoundSite(domain, self.country.code, r.url, r.title[:300], r.snippet[:500], query, r.engines))
            self.stats["new_domains"] += 1
            new += 1
        return new

    async def _verify(self, site: FoundSite) -> None:
        res = await self.fetcher.get(site.url)
        if not res.ok:
            site.status, site.reason = "unreachable", (res.error or f"HTTP {res.status}")
            if not self.proxy_used:
                site.reason += " (moze wymagac VPN tego kraju)"
            self.stats["unreachable"] += 1
            return
        final = registrable_domain(res.url)
        if final and final != site.domain and final in self.store.known_domains():
            site.status, site.reason = "rejected", f"przekierowuje do znanej strony {final}"
            self.stats["rejected"] += 1
            return
        page = await asyncio.to_thread(parse_page, res.text, res.url)
        site.language = page.lang
        if is_shop_page(res.text, page) and not jsonld_recipe(page):
            site.status, site.reason = "rejected", "sklep / strona produktu"
            self.stats["rejected"] += 1
            return
        c = await classify(page, DEFAULT_KEYWORDS, self.llm)
        if c.kind == "recipe":  # classify odrzuca juz przepisy czysto kulinarne
            site.status, site.score = "verified", max(c.confidence, 0.6)
        elif c.kind == "fact":
            site.status, site.score = "maybe", round(c.confidence * 0.6, 2)
        else:
            site.status, site.score = "rejected", 0.0
        site.reason = f"{c.kind} ({c.method}): {c.reason}"[:300]
        if not site.title:
            site.title = page.title
        self.stats[site.status] += 1


CART_PHRASES = ("add to cart", "add to basket", "dodaj do koszyka", "do koszyka", "в кошик", "додати в кошик",
                "купити", "in den warenkorb", "ajouter au panier", "añadir al carrito", "aggiungi al carrello",
                "do košíku", "kosárba", "adaugă în coș")


def is_shop_page(html: str, page) -> bool:
    """Strona produktu w sklepie: JSON-LD Product/Offer albo przycisk 'do koszyka' + cena."""
    for node in page.jsonld:
        t = node.get("@type")
        if any(x in ("Product", "Offer", "AggregateOffer") for x in (t if isinstance(t, list) else [t])):
            return True
    low = html.lower()
    return any(p in low for p in CART_PHRASES) and any(x in low for x in ("zł", "грн", "€", "$", "price", "cena"))


def site_root(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"
