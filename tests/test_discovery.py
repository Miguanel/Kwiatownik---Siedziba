import asyncio
import json

import httpx

from app.config import Settings
from app.discovery.countries import COUNTRIES
from app.discovery.domains import is_blocked, registrable_domain
from app.discovery.finder import DiscoveryConfig, SiteFinder
from app.discovery.queries import make_queries
from app.discovery.search import SearchResult, SearxngSearch, search_all
from app.scrapers.fetcher import Fetcher
from tests.test_scraper import recipe_html

UA = COUNTRIES["ua"]


def test_registrable_domain_and_blocklist():
    assert registrable_domain("https://www.retsepty.com.ua/chai") == "retsepty.com.ua"
    assert registrable_domain("https://m.blog.example.co.uk/x") == "example.co.uk"
    assert registrable_domain("https://en.herbs.example/x") == "herbs.example"
    assert is_blocked("youtube.com") and is_blocked("pinterest.de") and is_blocked("uk.wikipedia.org")
    assert not is_blocked("herbs.example")


def test_proxy_per_country_from_settings():
    s = Settings(country_proxies="ua=http://vpn-ua:8888, de = http://vpn-de:8888", _env_file=None)
    assert s.proxy_for("UA") == "http://vpn-ua:8888" and s.proxy_for("de") == "http://vpn-de:8888"
    assert s.proxy_for("fr") is None and s.proxy_for(None) is None


def test_searxng_parsing_and_merge():
    def handler(req: httpx.Request):
        assert req.url.params["format"] == "json" and req.url.params["language"] == "uk-UA"
        return httpx.Response(200, json={"results": [
            {"url": "https://a.example/1", "title": "A", "content": "x", "engines": ["google"]},
            {"url": "https://b.example/2", "title": "B", "content": "y", "engine": "bing"}]})

    class Other:
        name = "other"

        async def search(self, q, lang, country, page=1):
            return [SearchResult("https://a.example/1", "A", "", ["brave"])]

    sx = SearxngSearch("http://searxng:8080", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    res = asyncio.run(search_all([sx, Other()], "рецепти з трав", "uk-UA", "ua"))
    by_url = {r.url: r for r in res}
    assert len(res) == 2 and by_url["https://a.example/1"].engines == ["brave", "google"]


def test_queries_skip_used_and_fallback():
    class LLM:
        async def complete(self, prompt, system=None, json_mode=False):
            assert "Ukrainian" in prompt and "чай з трав рецепт" in prompt   # uzyte przekazane do LLM

            class R:
                text = json.dumps({"queries": ["чай з трав рецепт", "настоянка з калини", "сироп з бузини"]})
            return R()

    q = asyncio.run(make_queries(UA, 5, {"чай з трав рецепт"}, LLM()))
    assert q == ["настоянка з калини", "сироп з бузини"]
    q2 = asyncio.run(make_queries(UA, 2, {"рецепти з трав"}, None))
    assert q2 == ["народні рецепти лікарські рослини", "настоянка з трав рецепт"]


class Store:
    def __init__(self, known=()):
        self.known = set(known)
        self.sites, self.queries, self.logs, self.bumped = {}, [], [], []

    def known_domains(self): return set(self.known)
    def used_queries(self, c): return {q for _, q, _, _ in self.queries}
    def save_query(self, c, t, r, n): self.queries.append((c, t, r, n))
    def save_site(self, s): self.sites[s.domain] = s
    def bump_site(self, d): self.bumped.append(d)
    def log(self, m): self.logs.append(m)
    def progress(self, d, t): pass
    def should_stop(self): return False


def test_finder_dedups_and_verifies():
    class Engine:
        name = "fake"

        async def search(self, q, lang, country, page=1):
            return [SearchResult("https://www.recipes-ua.example/chai-z-trav", "Чай з трав", ""),
                    SearchResult("https://youtube.com/watch?v=1", "video", ""),
                    SearchResult("https://known.example/x", "known", ""),
                    SearchResult("https://down.example/x", "down", ""),
                    SearchResult("https://m.recipes-ua.example/other", "again", "")]  # ta sama domena

    def site(req: httpx.Request):
        if req.url.host == "down.example":
            return httpx.Response(403)
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=recipe_html("Чай з трав"), headers={"content-type": "text/html"})

    store = Store(known={"known.example"})
    fetcher = Fetcher("t", 0, client=httpx.AsyncClient(transport=httpx.MockTransport(site)))
    cfg = DiscoveryConfig(queries=2, extra_queries=["рецепти з трав"], pause_between_queries_s=0)
    stats = asyncio.run(SiteFinder(UA, [Engine()], store, fetcher, cfg).run())

    assert set(store.sites) == {"recipes-ua.example", "down.example"}      # kazda domena raz
    assert store.sites["recipes-ua.example"].status == "verified"
    assert store.sites["down.example"].status == "unreachable" and "VPN" in store.sites["down.example"].reason
    assert stats["blocked"] >= 1 and "known.example" in store.bumped
    assert stats["queries"] == 2 and len(store.queries) == 2


def test_domain_suffixes_regional():
    assert registrable_domain("https://drink.co.ua/x") == "drink.co.ua"
    assert registrable_domain("https://www.herbarium.katowice.pl/a") == "herbarium.katowice.pl"
    assert registrable_domain("https://www.ziolowyzakatek.sklep.pl/") == "ziolowyzakatek.sklep.pl"


def test_finder_rejects_shops_and_skips_sites_found_meanwhile():
    shop = ('<html><head><title>Syrop</title><script type="application/ld+json">{"@type": "Product", "name": "Syrop"}'
            '</script></head><body><h1>Syrop z bzu 250 ml</h1><button>Dodaj do koszyka</button> 19,99 zł</body></html>')

    class Engine:
        name = "fake"

        async def search(self, q, lang, country, page=1):
            return [SearchResult("https://shop.example/syrop", "Syrop 250 ml", ""),
                    SearchResult("https://raced.example/x", "x", "")]

    def site(req: httpx.Request):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=shop, headers={"content-type": "text/html"})

    class RaceStore(Store):
        def known_domains(self):
            # inne zadanie zapisalo raced.example juz PO wyszukaniu, a PRZED weryfikacja
            return set(self.known) | ({"raced.example"} if self.queries else set())

    store = RaceStore()
    fetcher = Fetcher("t", 0, client=httpx.AsyncClient(transport=httpx.MockTransport(site)))
    asyncio.run(SiteFinder(COUNTRIES["pl"], [Engine()], store, fetcher,
                           DiscoveryConfig(queries=1, extra_queries=["syrop z bzu"], pause_between_queries_s=0)).run())
    assert set(store.sites) == {"shop.example"} and store.sites["shop.example"].status == "rejected"
    assert "sklep" in store.sites["shop.example"].reason
