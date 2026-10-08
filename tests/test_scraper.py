import asyncio
import json

import httpx

from app.scrapers.classify import classify
from app.scrapers.crawler import CrawlConfig, PageResult, SmartCrawler
from app.scrapers.fetcher import Fetcher, normalize_url
from app.scrapers.page import parse_page
from app.scrapers.patterns import PatternBook, url_pattern

# ------------------------------------------------------------------ sztuczna strona
JSONLD = json.dumps({"@context": "https://schema.org", "@graph": [
    {"@type": "WebPage"}, {"@type": "Recipe", "name": "Nettle Soup", "recipeIngredient": ["200 g nettle tops"]}]})

LOREM = " ".join(["Stinging nettle is a perennial herbal plant used in folk medicine and wild foraging."] * 30)


def recipe_html(name, jsonld=False):
    ld = f'<script type="application/ld+json">{JSONLD}</script>' if jsonld else ""
    return f"""<html lang="en"><head><title>{name}</title>{ld}</head><body><article>
    <h1>{name}</h1><p>A traditional herbal remedy with nettle and elder flower. {LOREM}</p>
    <h2>Ingredients</h2><ul><li>2 cups fresh nettle leaves</li><li>1 tbsp honey</li><li>500 ml water</li></ul>
    <h2>Instructions</h2><ol><li>Boil the water.</li><li>Steep the herb for 10 minutes.</li></ol>
    <a href="/recipes/">All recipes</a></article></body></html>"""


SHOP = "<html><head><title>Shop</title></head><body><h1>Buy seeds</h1><p>Cheap seeds, discount.</p></body></html>"

PAGES = {
    "/robots.txt": "User-agent: *\nDisallow: /private/\nSitemap: https://herbs.example/sitemap_index.xml\n",
    "/sitemap_index.xml": """<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://herbs.example/post-sitemap.xml</loc></sitemap></sitemapindex>""",
    "/post-sitemap.xml": """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://herbs.example/recipes/nettle-soup-recipe/?utm_source=x</loc></url>
        <url><loc>https://herbs.example/shop/seed-pack-one/</loc></url></urlset>""",
    "/": """<html><head><title>Herbs</title></head><body><h1>Herbal kitchen</h1>
        <a href="/recipes/">Recipes</a> <a href="/shop/">Shop</a> <a href="/private/secret-page">x</a>
        <a href="/about">About</a> <a href="https://other.example/x">ext</a></body></html>""",
    "/recipes/": """<html><head><title>Recipes</title></head><body><h1>Recipes</h1>
        <a href="/recipes/elderflower-cordial-recipe/">Elderflower cordial</a>
        <a href="/recipes/dandelion-tea-recipe/">Dandelion tea</a>
        <a href="/recipes/nettle-soup-recipe/">Nettle soup</a></body></html>""",
    "/recipes/nettle-soup-recipe/": recipe_html("Nettle Soup", jsonld=True),
    "/recipes/elderflower-cordial-recipe/": recipe_html("Elderflower Cordial"),
    "/recipes/dandelion-tea-recipe/": recipe_html("Dandelion Tea"),
    "/shop/": "<html><body><h1>Shop</h1>" + "".join(
        f'<a href="/shop/seed-pack-{i}-x/">Seeds {i}</a>' for i in range(12)) + "</body></html>",
    "/about": "<html><head><title>About</title></head><body><h1>About us</h1><p>We are a small team.</p></body></html>",
}


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if request.url.host != "herbs.example":
        return httpx.Response(404)
    if path.startswith("/shop/seed"):
        return httpx.Response(200, text=SHOP, headers={"content-type": "text/html"})
    if path.startswith("/private"):
        raise AssertionError("robots.txt zabrania /private/")
    if path in PAGES:
        ctype = "application/xml" if path.endswith(".xml") else "text/plain" if path.endswith(".txt") else "text/html"
        return httpx.Response(200, text=PAGES[path], headers={"content-type": ctype})
    return httpx.Response(404)


class MemoryStore:
    def __init__(self):
        self.results: dict[str, PageResult] = {}
        self.logs: list[str] = []
        self.patterns = {}
        self.js = False
        self.profile = None

    def known_urls(self): return set(self.results)
    def load_patterns(self): return self.patterns
    def save_patterns(self, stats): self.patterns = stats
    def save_result(self, r): self.results[r.url] = r
    def log(self, m): self.logs.append(m)
    def progress(self, d, t): pass
    def should_stop(self): return False
    def flag_js(self): self.js = True
    def profile_feedback(self, used, failed): self.profile = (used, failed)


def make_fetcher():
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return Fetcher("TestBot", delay_s=0, client=client)


def crawl(store, **cfg):
    async def run():
        f = make_fetcher()
        try:
            return await SmartCrawler("https://herbs.example/", f, store, CrawlConfig(**cfg)).run()
        finally:
            await f.aclose()
    return asyncio.run(run())


# ------------------------------------------------------------------ testy
def test_url_pattern_and_normalize():
    assert url_pattern("https://a.com/recipes/nettle-soup") == "/recipes/{slug}"
    assert url_pattern("https://a.com/2024/05/elder-flower-cordial/") == "/{n}/{n}/{slug}"
    assert normalize_url("/x/?utm_source=a&id=3#top", "https://A.com/") == "https://a.com/x/?id=3"


def test_parse_page_and_classify_layers():
    page = parse_page(recipe_html("Nettle Soup", jsonld=True), "https://herbs.example/r")
    c = asyncio.run(classify(page, ["nettle"]))
    assert (c.kind, c.method, c.title) == ("recipe", "jsonld", "Nettle Soup")

    page2 = parse_page(recipe_html("Dandelion Tea"), "https://herbs.example/d")
    c2 = asyncio.run(classify(page2, ["dandelion", "herb"]))
    assert c2.kind == "recipe" and c2.method == "heuristic"

    c3 = asyncio.run(classify(parse_page(SHOP, "https://herbs.example/shop"), ["herb"]))
    assert c3.kind == "other"


def test_crawler_finds_recipes_respects_robots_and_stays_on_site():
    store = MemoryStore()
    stats = crawl(store, max_pages=30)
    recipes = {u for u, r in store.results.items() if r.kind == "recipe"}
    assert recipes == {
        "https://herbs.example/recipes/nettle-soup-recipe/",
        "https://herbs.example/recipes/elderflower-cordial-recipe/",
        "https://herbs.example/recipes/dandelion-tea-recipe/",
    }
    assert store.results["https://herbs.example/recipes/nettle-soup-recipe/"].structured["title"] == "Nettle Soup"
    assert stats["recipe"] == 3 and all("other.example" not in u for u in store.results)


def test_crawler_learns_dead_patterns_and_skips_them():
    store = MemoryStore()
    stats = crawl(store, max_pages=40, use_sitemap=False)
    shop = store.patterns["/shop/{slug}"]
    assert shop.hits == 0 and 8 <= shop.misses < 12     # po 8 pudlach wzorzec uznany za martwy
    assert stats["skipped_dead"] >= 1
    assert store.patterns["/recipes/{slug}"].hits == 3


def test_second_scan_skips_known_pages():
    store = MemoryStore()
    crawl(store, max_pages=30)
    stats2 = crawl(store, max_pages=30)
    assert stats2["recipe"] == 0 and stats2["skipped_known"] >= 3


def test_pattern_book_priorities():
    book = PatternBook()
    for _ in range(3):
        book.record("/recipes/{slug}", True, parent="/recipes")
    for _ in range(8):
        book.record("/shop/{slug}", False, parent="/shop")
    assert book.is_dead("/shop/{slug}") and not book.is_dead("/recipes/{slug}")
    assert book.stats["/recipes"].discovered == 3


def test_ambiguous_page_goes_to_llm():
    class FakeLLM:
        calls = 0

        async def complete(self, prompt, system=None, json_mode=False):
            FakeLLM.calls += 1
            assert json_mode and "Kwiatownik" in system

            class R:
                model = "fake-model"
                text = '{"type": "fact", "confidence": 0.9, "title": "Nettle history", "reason": "plant facts"}'
            return R()

    html = f"<html><head><title>Nettle</title></head><body><article><h1>Nettle</h1><p>{LOREM}</p></article></body></html>"
    c = asyncio.run(classify(parse_page(html, "https://herbs.example/nettle"), ["nettle"], FakeLLM()))
    assert (c.kind, c.method, c.title) == ("fact", "llm:fake-model", "Nettle history") and FakeLLM.calls == 1


def test_frontier_saved_and_scan_resumes_where_it_stopped():
    class FrontierStore(MemoryStore):
        def __init__(self):
            super().__init__()
            self.frontier = None

        def save_frontier(self, q):
            self.frontier = q

    store = FrontierStore()
    crawl(store, max_pages=3, use_sitemap=False)       # przerwane limitem po 3 stronach
    assert store.frontier and all("url" in q for q in store.frontier)
    first = set(store.results)

    store2 = FrontierStore()
    store2.results = dict(store.results)               # to, co juz zebrane, zostaje w bazie
    stats = crawl(store2, max_pages=30, resume_queue=store.frontier)
    assert stats["recipe"] + len([r for u, r in store.results.items() if r.kind == "recipe"]) == 3
    assert not (set(store2.results) - first) & first   # nic nie pobrano drugi raz
    assert any("Wznawiam" in m for m in store2.logs)


def test_scan_selected_refreshes_known_pages_and_follows_links():
    store = MemoryStore()
    old = PageResult(url="https://herbs.example/recipes/", title="stary", kind="other", confidence=0.5,
                     method="heuristic", reason="", text="", structured=None, language="en", pattern="/recipes")
    store.results[old.url] = old
    stats = crawl(store, max_pages=10, max_depth=1, use_sitemap=False, seeds=["https://herbs.example/recipes/"],
                  refresh_seeds=True)
    assert store.results["https://herbs.example/recipes/"].refresh is True      # znana strona pobrana ponownie
    assert stats["refreshed"] == 1 and stats["recipe"] == 3                       # + przepisy z jej linkow


def test_crawler_reports_live_activity():
    class ActStore(MemoryStore):
        def __init__(self):
            super().__init__()
            self.steps = []

        def activity(self, step, detail=""):
            self.steps.append((step, detail))

    store = ActStore()
    crawl(store, max_pages=5)
    steps = [s for s, _ in store.steps]
    assert "czytam sitemapy" in steps and any(s.startswith("pobieram strone") for s in steps)
    assert any(d.startswith("https://herbs.example/") for _, d in store.steps)


def test_culinary_recipes_are_skipped_but_remedies_and_dyes_kept():
    import json as _json
    cake = ("<html><head><title>Cherry cake</title><script type='application/ld+json'>"
            + _json.dumps({"@type": "Recipe", "name": "Cherry cake", "recipeCategory": "Dessert",
                           "recipeIngredient": ["200 g flour"], "recipeInstructions": "Bake in the oven."})
            + "</script></head><body><h1>Cherry cake</h1><p>" + "Bake the cake in the oven. " * 30 + "</p></body></html>")
    salve = ("<html><head><title>Calendula salve</title><script type='application/ld+json'>"
             + _json.dumps({"@type": "Recipe", "name": "Calendula healing salve", "recipeIngredient": ["50 g flowers"],
                            "recipeInstructions": "Infuse oil"})
             + "</script></head><body><h1>Calendula healing salve for skin</h1><p>" + "Healing salve for dry skin. " * 30
             + "</p></body></html>")
    c1 = asyncio.run(classify(parse_page(cake, "https://x.example/cake"), ["herb"], skip_culinary=True))
    c2 = asyncio.run(classify(parse_page(salve, "https://x.example/salve"), ["herb"], skip_culinary=True))
    c3 = asyncio.run(classify(parse_page(cake, "https://x.example/cake"), ["herb"], skip_culinary=False))
    assert c1.kind == "other" and "kulinarny" in c1.reason
    assert c2.kind == "recipe" and c3.kind == "recipe"


def _llm_says(kind):
    class FakeLLM:
        async def complete(self, prompt, system=None, json_mode=False):
            class R:
                model = "fake"
                text = json.dumps({"type": kind, "confidence": 0.9, "title": "T", "reason": "r"})
            return R()
    return FakeLLM()


def test_llm_recipe_without_quantities_is_not_a_recipe():
    """LLM mowi 'recipe', ale na stronie nie ma zadnych ilosci: kategoria z linkami -> other, artykul -> fact."""
    import json as _json  # noqa: F401
    text = "Herbal tea is a wonderful way to relax. " * 30
    article = f"<html><head><title>Herbal tea</title></head><body><article><p>{text}</p></article></body></html>"
    links = "".join(f'<a href="/recipe-{i}/">Herbal recipe {i}</a> ' for i in range(80))
    listing = f"<html><head><title>Herbal recipes</title></head><body><p>{text}</p>{links}</body></html>"
    c1 = asyncio.run(classify(parse_page(article, "https://h.example/tea"), ["herb"], _llm_says("recipe")))
    c2 = asyncio.run(classify(parse_page(listing, "https://h.example/recipes"), ["herb"], _llm_says("recipe")))
    assert c1.kind == "fact" and c2.kind == "other" and "lista" in c2.reason


def test_shop_product_page_is_other():
    prod = ('<html><head><title>Herbal syrup 420 ml</title><script type="application/ld+json">'
            '{"@type": "Product", "name": "Herbal syrup"}</script></head><body><p>'
            + "Syrup with linden and raspberry, 2 tbsp daily. " * 20 + "</p></body></html>")
    c = asyncio.run(classify(parse_page(prod, "https://shop.example/p/1"), ["herb"], _llm_says("recipe")))
    assert c.kind == "other" and "produktu" in c.reason


def test_probe_stops_after_collecting_samples():
    """Etap rozpoznania: skan konczy sie, gdy zbierze zadana liczbe probek (tu 2 przepisy)."""
    full = crawl(MemoryStore(), max_pages=30)
    probe = crawl(MemoryStore(), max_pages=30, stop_when={"recipe": 2})
    assert probe["recipe"] >= 2 and probe["fetched"] < full["fetched"]
