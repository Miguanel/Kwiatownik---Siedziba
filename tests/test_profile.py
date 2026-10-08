import asyncio
import json

from app.pipeline.dedup import find_duplicate, fingerprint, is_duplicate
from app.scrapers.crawler import CrawlConfig, SmartCrawler
from app.scrapers.patterns import PatternStat
from app.scrapers.profile import ScraperProfile, build_profile, html_skeleton, regex_from_patterns, validate
from app.scrapers.recipe_data import normalize_jsonld_recipe
from tests.test_scraper import LOREM, MemoryStore, make_fetcher


def plugin_recipe(name: str) -> str:
    """Przepis bez JSON-LD, z klasami jak we wtyczce WordPress."""
    return f"""<html><head><title>{name} | Herbs</title></head><body><article>
      <h1 class="entry-title">{name}</h1><p>{LOREM[:300]}</p>
      <div class="wprm-recipe"><ul>
        <li class="wprm-recipe-ingredient">2 cups nettle</li><li class="wprm-recipe-ingredient">1 tbsp honey</li>
        <li class="wprm-recipe-ingredient">500 ml water</li></ul>
      <ol><li class="wprm-recipe-instruction">Boil water.</li><li class="wprm-recipe-instruction">Steep 10 min.</li></ol>
      </div></article></body></html>"""


GOOD_FIELDS = {"title": {"css": "h1.entry-title"}, "ingredients": {"css": "li.wprm-recipe-ingredient"},
               "steps": {"css": "li.wprm-recipe-instruction"}}
RECIPES = ["https://herbs.example/recipes/nettle-tea-recipe/", "https://herbs.example/recipes/honey-syrup-recipe/"]
OTHERS = ["https://herbs.example/shop/seed-pack-1-x/", "https://herbs.example/about"]


def test_normalize_jsonld_howto_sections():
    node = {"@type": "Recipe", "name": "Tea &amp; Honey", "recipeIngredient": ["1 cup <b>water</b>"],
            "recipeInstructions": [{"@type": "HowToSection", "itemListElement": [
                {"@type": "HowToStep", "text": "Boil."}, {"@type": "HowToStep", "text": "Steep."}]}],
            "image": [{"url": "https://x/img.jpg"}], "recipeYield": ["2", "2 cups"]}
    r = normalize_jsonld_recipe(node)
    assert r["title"] == "Tea & Honey" and r["ingredients"] == ["1 cup water"]
    assert r["steps"] == ["Boil.", "Steep."] and r["image"] == "https://x/img.jpg" and r["yield"] == "2"


def test_profile_extracts_with_css_and_validates():
    p = ScraperProfile(recipe_url_regex=r"^/recipes/[^/]+/?$", fields=GOOD_FIELDS)
    data = p.extract(plugin_recipe("Nettle Tea"))
    assert data["title"] == "Nettle Tea" and len(data["ingredients"]) == 3 and len(data["steps"]) == 2
    samples = [(u, plugin_recipe("X"), []) for u in RECIPES]
    rep = validate(p, samples, RECIPES, OTHERS)
    assert rep["passed"] and rep["coverage"] == 1.0 and rep["regex"]["precision"] == 1.0


def test_regex_from_learned_patterns():
    rx = regex_from_patterns({"/recipes/{slug}": PatternStat(5, 1, 0), "/shop/{slug}": PatternStat(0, 9, 0)})
    p = ScraperProfile(recipe_url_regex=rx)
    assert p.is_recipe_url(RECIPES[0]) and not p.is_recipe_url(OTHERS[0])


def test_skeleton_is_compact():
    sk = html_skeleton(plugin_recipe("Nettle Tea"))
    assert "li.wprm-recipe-ingredient" in sk and "<" not in sk and len(sk) < 2000


def test_build_profile_retries_with_feedback():
    class FakeLLM:
        def __init__(self):
            self.prompts = []

        async def complete(self, prompt, system=None, json_mode=False):
            self.prompts.append(prompt)
            fields = GOOD_FIELDS if len(self.prompts) > 1 else {"title": {"css": "h1"}, "ingredients": {"css": ".nope"},
                                                                 "steps": {"css": ".nope"}}

            class R:
                provider, model = "fake", "m1"
                text = json.dumps({"recipe_url_regex": r"^/recipes/", "listing_url_regex": None, "fields": fields})
            return R()

    llm = FakeLLM()
    samples = [(u, plugin_recipe("X"), []) for u in RECIPES]
    prof, rep = asyncio.run(build_profile(llm, samples, RECIPES, OTHERS, {}))
    assert rep["passed"] and len(llm.prompts) == 2
    assert "FAILED VALIDATION" in llm.prompts[1] and prof.built_with == "fake/m1"
    assert prof.validation["passed"] and ScraperProfile.from_json(prof.to_json()).fields == prof.fields


def test_crawler_profile_mode_uses_selectors_without_llm():
    import tests.test_scraper as ts
    old = dict(ts.PAGES)
    ts.PAGES["/recipes/elderflower-cordial-recipe/"] = plugin_recipe("Elderflower Cordial")
    ts.PAGES["/recipes/dandelion-tea-recipe/"] = plugin_recipe("Dandelion Tea")
    try:
        class NoLLM:
            async def complete(self, *a, **k):
                raise AssertionError("LLM nie powinien byc wolany w trybie profilu")

        prof = ScraperProfile(recipe_url_regex=r"^/recipes/[^/]+/?$", fields=GOOD_FIELDS,
                              validation={"passed": True})
        store = MemoryStore()

        async def run():
            f = make_fetcher()
            try:
                return await SmartCrawler("https://herbs.example/", f, store,
                                          CrawlConfig(max_pages=25, profile=prof), NoLLM()).run()
            finally:
                await f.aclose()
        stats = asyncio.run(run())
    finally:
        ts.PAGES.clear()
        ts.PAGES.update(old)
    # nettle-soup ma inny uklad (tylko niepelny JSON-LD) -> profil nie zadziala, strona idzie zwykla sciezka
    assert stats["profile_used"] == 2 and stats["profile_failed"] == 1 and stats["llm_calls"] == 0
    assert stats["recipe"] == 3
    r = store.results["https://herbs.example/recipes/dandelion-tea-recipe/"]
    assert r.method == "profile" and r.structured["ingredients"][0] == "2 cups nettle"
    assert store.profile == (2, 1)


def test_list_mode_fetches_only_given_urls():
    store = MemoryStore()

    async def run():
        f = make_fetcher()
        try:
            cfg = CrawlConfig(max_pages=5, max_depth=0, use_sitemap=False,
                              seeds=[RECIPES[0].replace("nettle-tea", "nettle-soup"), "https://other.example/x"])
            return await SmartCrawler("https://herbs.example/", f, store, cfg).run()
        finally:
            await f.aclose()
    stats = asyncio.run(run())
    assert stats["fetched"] == 1 and list(store.results) == ["https://herbs.example/recipes/nettle-soup-recipe/"]


def test_query_variants_limited():
    store = MemoryStore()
    crawler = SmartCrawler("https://herbs.example/", make_fetcher(), store, CrawlConfig())
    for f in ("a", "b", "c", "d"):
        crawler._push(f"https://herbs.example/produkty.html?f={f}", depth=1)
    assert len(crawler.queue) == 2


def test_dedup_same_recipe_different_sites():
    a = fingerprint("Easy Elderflower Cordial Recipe", ["20 elderflower heads", "1 kg sugar", "2 lemons", "1.5 l water"])
    b = fingerprint("Homemade elderflower cordial", ["25 elderflower heads", "1kg sugar", "3 lemons", "1.5 litres water"])
    c = fingerprint("Elderflower champagne", ["8 elderflower heads", "700 g sugar", "2 lemons", "4.5 l water", "vinegar"])
    assert is_duplicate(a, b) and not is_duplicate(a, c)
    assert find_duplicate(b, [(1, c), (2, a)]) == 2


def test_profile_article_mode_returns_title_and_content():
    """Strona zielarska z przepisem opisanym proza: profil zwraca tytul + tresc artykulu (tryb artykulu)."""
    body = "<p>Rumianek: 1 lyzke kwiatow zalac 200 ml wrzatku, parzyc 10 minut.</p>" * 8
    html = f"<html><body><nav>menu menu</nav><h1 class='t'>Napary na trawienie</h1><div class='post'>{body}</div>" \
           f"<footer>stopka</footer></body></html>"
    prof = ScraperProfile("/blog/", fields={"title": {"css": "h1.t"}, "content": {"css": "div.post"},
                                            "ingredients": None, "steps": None})
    data = prof.extract(html)
    assert data and data["article"] and data["title"] == "Napary na trawienie"
    assert "200 ml" in data["content"] and "menu" not in data["content"] and "stopka" not in data["content"]
    assert data["content"].count("\n") >= 7                       # akapity zachowane
    short = ScraperProfile("/blog/", fields={"title": {"css": "h1.t"}, "content": {"css": "nav"}})
    assert short.extract(html) is None                            # za malo tresci -> brak danych


def test_profile_facts_regex_and_content():
    from app.scrapers.profile import validate_facts
    art = ("<html><body><nav>menu</nav><h1 class='t'>Historia pokrzywy</h1><div class='post'>"
           + "<p>Pokrzywa byla uzywana w sredniowieczu do wyrobu tkanin i jako lek.</p>" * 10 + "</div></body></html>")
    prof = ScraperProfile("^/przepisy/", fact_url_regex="^/wiedza/",
                          fields={"title": {"css": "h1.t"}, "content": {"css": "div.post"}})
    fact = prof.extract_fact(art)
    assert fact["title"] == "Historia pokrzywy" and "tkanin" in fact["content"] and "menu" not in fact["content"]
    assert prof.is_fact_url("https://a.pl/wiedza/pokrzywa") and not prof.is_fact_url("https://a.pl/przepisy/x")
    rep = validate_facts(prof, [("https://a.pl/wiedza/pokrzywa", art, [])] * 3,
                         ["https://a.pl/wiedza/pokrzywa", "https://a.pl/wiedza/mieta"],
                         ["https://a.pl/przepisy/x", "https://a.pl/kontakt"])
    assert rep["ok"] and rep["pages_ok"] == 3
    bad = ScraperProfile("^/przepisy/", fact_url_regex="^/", fields=prof.fields)   # lapie wszystko
    assert not validate_facts(bad, [("u", art, [])], ["https://a.pl/wiedza/a"], ["https://a.pl/przepisy/x",
                                                                                   "https://a.pl/kontakt"])["ok"]


def test_harvest_reads_sitemap_even_when_resuming_and_keeps_only_profile_urls():
    """Pelne pobieranie: sitemapy czytane takze przy wznowieniu; z sitemap tylko adresy pasujace do profilu."""
    prof = ScraperProfile(recipe_url_regex=r"^/recipes/[^/]+/?$", fields=GOOD_FIELDS, validation={"passed": True})
    store = MemoryStore()

    async def run():
        f = make_fetcher()
        try:
            cfg = CrawlConfig(max_pages=25, profile=prof, harvest=True,
                              resume_queue=[{"url": "https://herbs.example/about", "anchor": "", "depth": 1}])
            return await SmartCrawler("https://herbs.example/", f, store, cfg).run()
        finally:
            await f.aclose()
    asyncio.run(run())
    assert any("Sitemapy" in m and "pasujacych do profilu" in m for m in store.logs)
    assert "https://herbs.example/recipes/nettle-soup-recipe/" in store.results        # z sitemap mimo wznowienia
    assert "https://herbs.example/shop/seed-pack-one/" not in store.results            # nie pasuje do profilu
