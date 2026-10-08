"""SmartCrawler - dynamiczny crawler, ktory uczy sie struktury kazdej strony.

Algorytm:
  1. Start: strona glowna + adresy z sitemap i RSS.
  2. Kolejka priorytetowa: priorytet = skutecznosc wzorca URL + slowa kluczowe + glebokosc + wartosc huba.
  3. Dla kazdej strony: pobierz -> przeanalizuj -> dodaj nowe linki -> sklasyfikuj (JSON-LD/heurystyki/LLM).
  4. Wynik klasyfikacji aktualizuje statystyki wzorca (PatternBook); co kilka stron kolejka jest
     przeliczana od nowa, wiec crawler coraz trafniej wybiera kolejne adresy, a martwe
     czesci serwisu (np. sklep, tagi bez tresci) sa pomijane.
  5. Tryb profilu: jesli strona ma zbudowany (przez LLM) profil scrapera, adresy pasujace do
     regexu przepisow sa wyciagane selektorami CSS/JSON-LD bez LLM, a pozostale strony sluza tylko
     do znajdowania linkow (klasyfikacja heurystykami, bez LLM).
  6. Wyniki trafiaja do `store` (baza danych) - crawler nie zna SQL-a.
"""
import asyncio
import heapq
import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from app.scrapers.classify import (CULINARY_REASON, DEFAULT_KEYWORDS, Classification, classify, is_culinary,
                                   heuristic_scores, jsonld_recipe, quantity_hits)
from app.scrapers.discovery import feed_urls, sitemap_urls
from app.scrapers.fetcher import Fetcher, normalize_url, site_key
from app.scrapers.page import PageData, parse_page
from app.scrapers.patterns import PatternBook, PatternStat, should_skip, url_pattern, url_priority
from app.scrapers.profile import ScraperProfile
from app.scrapers.recipe_data import normalize_jsonld_recipe

log = logging.getLogger(__name__)


def skip_culinary() -> bool:
    from app.config import settings
    return settings.skip_culinary



@dataclass
class CrawlConfig:
    max_pages: int = 50            # ile stron maksymalnie pobrac w jednym skanowaniu
    max_depth: int = 3             # jak gleboko od strony startowej
    keywords: list[str] = field(default_factory=list)
    use_sitemap: bool = True
    llm_concurrency: int = 3
    reprioritize_every: int = 8
    max_query_variants: int = 2    # ile wariantow ?param=... tej samej sciezki odwiedzic (filtry, sortowanie)
    profile: ScraperProfile | None = None
    seeds: list[str] | None = None  # tryb "lista adresow": tylko te strony, bez chodzenia po linkach
    resume_queue: list[dict] | None = None  # wznowienie: kolejka zapisana przez przerwany skan
    refresh_seeds: bool = False     # "skanuj zaznaczone": pobierz ponownie znane strony z listy i zaktualizuj
    stop_when: dict | None = None   # etap rozpoznania: np. {"recipe": 5, "fact": 5} - stop po zebraniu probek
    harvest: bool = False           # pelne pobieranie strony z profilem: sitemapy zawsze, tylko adresy z profilu


@dataclass
class PageResult:
    url: str
    title: str
    kind: str
    confidence: float
    method: str
    reason: str
    text: str
    structured: dict | None
    language: str | None
    pattern: str
    refresh: bool = False          # strona juz byla w bazie - zaktualizuj zamiast pomijac


class CrawlStore(Protocol):
    def known_urls(self) -> set[str]: ...
    def load_patterns(self) -> dict[str, PatternStat]: ...
    def save_patterns(self, stats: dict[str, PatternStat]) -> None: ...
    def save_result(self, result: PageResult) -> None: ...
    def log(self, message: str) -> None: ...
    def progress(self, done: int, total: int) -> None: ...
    def should_stop(self) -> bool: ...
    def flag_js(self) -> None: ...
    def profile_feedback(self, used: int, failed: int) -> None: ...
    def save_frontier(self, queue: list[dict]) -> None: ...   # opcjonalne - do wznawiania
    def activity(self, step: str, detail: str = "") -> None: ...  # opcjonalne - podglad "na zywo"


@dataclass(order=True)
class _QItem:
    neg_priority: float
    seq: int
    url: str = field(compare=False)
    anchor: str = field(compare=False, default="")
    depth: int = field(compare=False, default=0)
    parent: str | None = field(compare=False, default=None)
    from_sitemap: bool = field(compare=False, default=False)


class SmartCrawler:
    def __init__(self, start_url: str, fetcher: Fetcher, store: CrawlStore, config: CrawlConfig, llm=None):
        self.start_url = normalize_url(start_url) or start_url
        self.site = site_key(self.start_url)
        self.fetcher = fetcher
        self.store = store
        self.cfg = config
        self.llm = llm
        self.keywords = [k.lower() for k in (config.keywords or [])] + DEFAULT_KEYWORDS
        self.book = PatternBook(store.load_patterns())
        self.queue: list[_QItem] = []
        self.seen: set[str] = set()
        self.done_urls: set[str] = set(store.known_urls())
        self._seq = itertools.count()
        self._llm_sem = asyncio.Semaphore(max(1, config.llm_concurrency))
        self._tasks: list[asyncio.Task] = []
        self.profile = config.profile
        self.stats = {"fetched": 0, "recipe": 0, "fact": 0, "other": 0, "errors": 0,
                      "skipped_dead": 0, "skipped_known": 0, "robots_blocked": 0, "llm_calls": 0}
        if self.profile:
            self.stats |= {"profile_used": 0, "profile_failed": 0}
        self._js_pages = 0
        self._query_variants: Counter = Counter()
        self._refresh: set[str] = set()      # adresy do ponownego pobrania
        self._refreshing: set[str] = set()   # adresy aktualnie odswiezane (zapis = aktualizacja)

    def _bonus(self, url: str) -> float:
        if not self.profile:
            return 0.0
        if self.profile.is_recipe_url(url):
            return 0.5
        if self.profile.is_fact_url(url):
            return 0.35
        return 0.25 if self.profile.is_listing_url(url) else 0.0

    # ------------------------------------------------------------ kolejka
    def _push(self, url: str, anchor: str = "", depth: int = 0, parent: str | None = None,
              from_sitemap: bool = False) -> None:
        if url in self.seen or site_key(url) != self.site or should_skip(url) or depth > self.cfg.max_depth:
            return
        if "?" in url:  # np. ?f=jelly, ?sort=... - ta sama strona z innym filtrem
            base = url.split("?", 1)[0]
            if self._query_variants[base] >= self.cfg.max_query_variants:
                return
            self._query_variants[base] += 1
        self.seen.add(url)
        prio = url_priority(url, anchor, depth, self.book, self.keywords, from_sitemap) + self._bonus(url)
        heapq.heappush(self.queue, _QItem(-prio, next(self._seq), url, anchor, depth, parent, from_sitemap))

    def frontier(self, limit: int = 5000) -> list[dict]:
        """Nieodwiedzone adresy z kolejki (najwazniejsze pierwsze) - do wznowienia po restarcie."""
        return [{"url": q.url, "anchor": q.anchor, "depth": q.depth, "parent": q.parent, "sm": q.from_sitemap}
                for q in sorted(self.queue)[:limit]]

    def _act(self, step: str, detail: str = "") -> None:
        fn = getattr(self.store, "activity", None)
        if fn:
            fn(step, detail)

    def _save_frontier(self) -> None:
        save = getattr(self.store, "save_frontier", None)
        if save:
            save(self.frontier())

    def _reprioritize(self) -> None:
        items = [(q.url, q.anchor, q.depth, q.parent, q.from_sitemap) for q in self.queue]
        self.queue = []
        for url, anchor, depth, parent, sm in items:
            prio = url_priority(url, anchor, depth, self.book, self.keywords, sm) + self._bonus(url)
            heapq.heappush(self.queue, _QItem(-prio, next(self._seq), url, anchor, depth, parent, sm))

    def _targets_reached(self) -> bool:
        goal = self.cfg.stop_when
        return bool(goal) and all(self.stats.get(k, 0) >= v for k, v in goal.items())

    # ------------------------------------------------------------ glowna petla
    async def run(self) -> dict:
        try:
            return await self._run()
        except asyncio.CancelledError:  # zadanie przerwane sila - zatrzymaj tez klasyfikacje w tle
            for t in self._tasks:
                t.cancel()
            raise

    async def _run(self) -> dict:
        if self.cfg.resume_queue:
            pending = [q for q in self.cfg.resume_queue if q.get("url") not in self.done_urls]
            self.store.log(f"Wznawiam przerwany skan: {len(pending)} adresow w kolejce")
            for q in pending:
                self._push(q["url"], q.get("anchor", ""), q.get("depth", 1), q.get("parent"), q.get("sm", False))
        elif self.cfg.seeds is not None:
            self.store.log(f"Tryb listy: {len(self.cfg.seeds)} adresow" + (" (z profilem scrapera)" if self.profile else ""))
            self._refresh = {normalize_url(u) for u in self.cfg.seeds} if self.cfg.refresh_seeds else set()
            for u in self.cfg.seeds:
                nu = normalize_url(u)
                if nu and nu in self.done_urls and nu not in self._refresh:
                    self.stats["skipped_known"] += 1
                elif nu:
                    self._push(nu, depth=0)
        else:
            self.store.log(f"Start: {self.start_url} (max {self.cfg.max_pages} stron, glebokosc {self.cfg.max_depth})"
                           + (" - tryb profilu scrapera" if self.profile else ""))
            self._push(self.start_url, depth=0)
        if self.cfg.use_sitemap and self.cfg.seeds is None and (self.cfg.harvest or not self.cfg.resume_queue):
            self._act("czytam sitemapy", self.start_url)
            limit = max(3000, self.cfg.max_pages * 3) if self.cfg.harvest else 3000
            urls = await sitemap_urls(self.fetcher, self.start_url, self.keywords, limit=limit)
            fresh = [u for u in urls if u not in self.done_urls]
            if self.cfg.harvest and self.profile:
                # pelne pobieranie: z sitemap bierzemy tylko strony przepisow i artykulow wg profilu
                fresh = [u for u in fresh if self.profile.is_recipe_url(u) or self.profile.is_fact_url(u)
                         or self.profile.is_listing_url(u)]
            self.store.log(f"Sitemapy: {len(urls)} adresow ({len(fresh)} nowych"
                           + (", pasujacych do profilu" if self.cfg.harvest and self.profile else "") + ")")
            for u in fresh:
                self._push(u, depth=1, from_sitemap=True)

        since_reprio = 0
        while self.queue and self.stats["fetched"] < self.cfg.max_pages:
            if self.store.should_stop():
                self.store.log("Zatrzymano na zadanie uzytkownika")
                break
            if self._targets_reached():
                self.store.log("Rozpoznanie zakonczone: zebrano probki " + ", ".join(
                    f"{k}={self.stats.get(k, 0)}/{v}" for k, v in self.cfg.stop_when.items()))
                break
            item = heapq.heappop(self.queue)
            pattern = url_pattern(item.url)
            if self.book.is_dead(pattern):
                self.stats["skipped_dead"] += 1
                continue
            await self._process(item, pattern)
            since_reprio += 1
            if since_reprio >= self.cfg.reprioritize_every:
                self._reprioritize()
                self._save_frontier()
                since_reprio = 0
            self.store.progress(self.stats["fetched"], self.cfg.max_pages)

        if self._tasks:
            if self.store.should_stop():  # zatrzymanie: nie czekaj na klasyfikacje (LLM moze czekac minutami)
                for t in self._tasks:
                    t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._save_frontier()
        self.store.save_patterns(self.book.stats)
        if self.profile:
            self.store.profile_feedback(self.stats["profile_used"], self.stats["profile_failed"])
        best = ", ".join(f"{p} ({s.hits}/{s.hits + s.misses})" for p, s in self.book.best(3))
        self.store.log(f"Koniec. Wzorce z trafieniami: {best or 'brak'}")
        self.store.log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in self.stats.items()))
        return self.stats

    async def _process(self, item: _QItem, pattern: str) -> None:
        if not await self.fetcher.allowed(item.url):
            self.stats["robots_blocked"] += 1
            return
        self._act(f"pobieram strone {self.stats['fetched'] + 1}/{self.cfg.max_pages}", item.url)
        res = await self.fetcher.get(item.url)
        self.stats["fetched"] += 1
        if not res.ok:
            self.stats["errors"] += 1
            self.store.log(f"[blad] {item.url} - {res.error or res.status}")
            return
        final_url = normalize_url(res.url) or item.url
        page = await asyncio.to_thread(parse_page, res.text, final_url)  # CPU w osobnym watku
        url = page.canonical if page.canonical and site_key(page.canonical) == self.site else final_url

        for link, anchor in page.links:  # nowe linki do kolejki (tez ze stron, ktore znamy)
            self._push(link, anchor, item.depth + 1, parent=pattern)
        if item.depth == 0 and page.feeds:
            for u in await feed_urls(self.fetcher, page.feeds):
                self._push(u, depth=1, from_sitemap=True)

        if page.looks_js_rendered:
            self._js_pages += 1
            if self._js_pages == 3:
                self.store.flag_js()
                self.store.log("Uwaga: strona wyglada na renderowana w JavaScript - tresc moze byc niepelna")
        refresh = url in self._refresh or final_url in self._refresh or item.url in self._refresh
        if url in self.done_urls and not refresh:
            self.stats["skipped_known"] += 1
            return
        self.done_urls.add(url)
        self._refresh.discard(url)
        self._refresh.discard(item.url)
        if refresh:
            self._refreshing.add(url)
            self.stats["refreshed"] = self.stats.get("refreshed", 0) + 1

        if self.profile and self.profile.is_recipe_url(url):
            data = await asyncio.to_thread(self.profile.extract, res.text, page.jsonld)
            if data and skip_culinary() and is_culinary(page, jsonld_recipe(page)):
                self.stats["culinary"] = self.stats.get("culinary", 0) + 1
                self.book.record(pattern, False, item.parent)
                self.store.save_result(PageResult(url=url, title=(data["title"] or page.title or url)[:300],
                                                  kind="other", confidence=0.8, method="profile",
                                                  reason=CULINARY_REASON, text="", structured=None,
                                                  language=page.lang, pattern=pattern,
                                                  refresh=url in self._refreshing))
                return
            if data and data.get("article"):
                # tryb artykulu: tresc z selektora profilu (bez menu/stopki); musi zawierac konkretne ilosci
                if quantity_hits(data["content"]) > 0:
                    self.stats["profile_used"] += 1
                    self._store_hit(page, url, pattern, item.parent, "recipe", 0.85, "profile:artykul",
                                    data["title"] or page.title, None, reason="profil scrapera (artykul)",
                                    text=data["content"])
                    return
                data = None  # artykul bez ilosci - zwykla klasyfikacja (moze byc ciekawostka)
            elif data:
                self.stats["profile_used"] += 1
                self._store_hit(page, url, pattern, item.parent, "recipe", 0.95, "profile",
                                data["title"] or page.title, data, reason="profil scrapera")
                return
            self.stats["profile_failed"] += 1
            self.store.log(f"[profil nie zadzialal] {url} - klasyfikuje zwyklym sposobem")

        if self.profile and self.profile.is_fact_url(url):
            fact = await asyncio.to_thread(self.profile.extract_fact, res.text)
            if fact and heuristic_scores(page, self.keywords)[1] > 0:   # artykul na temat roslin
                self.stats["profile_used"] += 1
                self._store_hit(page, url, pattern, item.parent, "fact", 0.8, "profile:fakt", fact["title"], None,
                                reason="profil scrapera (artykul)", text=fact["content"])
                return

        task = asyncio.create_task(self._classify_and_store(page, url, pattern, item.parent))
        self._tasks.append(task)
        await asyncio.sleep(0)  # daj klasyfikacji wystartowac (heurystyki koncza sie od razu)

    def _store_hit(self, page: PageData, url: str, pattern: str, parent: str | None, kind: str, conf: float,
                   method: str, title: str, structured: dict | None, reason: str = "",
                   text: str | None = None) -> None:
        self.book.record(pattern, kind in ("recipe", "fact"), parent)
        self.stats[kind] += 1
        self.store.save_result(PageResult(url=url, title=(title or url)[:300], kind=kind, confidence=conf,
                                          method=method, reason=reason, text=text or page.text, structured=structured,
                                          language=page.lang, pattern=pattern, refresh=url in self._refreshing))
        self.store.log(f"[{kind} {conf:.2f} {method}] {(title or '')[:80]} - {url}")

    async def _classify_and_store(self, page: PageData, url: str, pattern: str, parent: str | None) -> None:
        # z gotowym profilem LLM nie jest potrzebny dla stron spoza regexu przepisow (to tylko huby)
        llm = None if (self.profile and self.profile.validation.get("passed")) else self.llm
        async with self._llm_sem:
            self._act("rozpoznaje strone" + (" (LLM gdy niejednoznaczna)" if llm else ""), url)
            try:
                c: Classification = await classify(page, self.keywords, llm)
            except Exception as exc:  # pragma: no cover
                self.store.log(f"[blad klasyfikacji] {url}: {exc}")
                return
        if c.method.startswith("llm"):
            self.stats["llm_calls"] += 1
        node = jsonld_recipe(page)
        structured = normalize_jsonld_recipe(node) if node else None
        if c.kind in ("recipe", "fact"):
            self._store_hit(page, url, pattern, parent, c.kind, c.confidence, c.method, c.title, structured, c.reason)
        else:
            self.book.record(pattern, False, parent)
            self.stats["other"] += 1
            if c.reason == CULINARY_REASON:
                self.stats["culinary"] = self.stats.get("culinary", 0) + 1
            self.store.save_result(PageResult(url=url, title=c.title[:300], kind="other", confidence=c.confidence,
                                              method=c.method, reason=c.reason, text="", structured=None,
                                              language=page.lang, pattern=pattern, refresh=url in self._refreshing))
