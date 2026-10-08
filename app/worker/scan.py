"""Zadania: 'scan' (crawler), 'fetch_list' (lista adresow) i 'build_profile' (LLM buduje scraper)."""
import asyncio
import json
from datetime import datetime, timezone

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import Item, ItemKind, ItemStatus, Job, SitePattern, Source
from app.pipeline.dedup import find_duplicate, fingerprint
from app.scrapers.crawler import CrawlConfig, PageResult, SmartCrawler
from app.scrapers.fetcher import Fetcher
from app.scrapers.page import parse_page
from app.scrapers.patterns import PatternStat
from app.scrapers.profile import ScraperProfile, build_profile
from app.worker import activity
from app.worker.jobs import JobRunner, job_log


class DbStore:
    """Adapter: crawler <-> baza danych (implementuje CrawlStore)."""

    def __init__(self, job_id: int, source_id: int, runner: JobRunner):
        self.job_id = job_id
        self.source_id = source_id
        self.runner = runner
        self._last_progress = -1

    def known_urls(self) -> set[str]:
        with Session(engine) as s:
            return set(s.exec(select(Item.url).where(Item.source_id == self.source_id)).all())

    def load_patterns(self) -> dict[str, PatternStat]:
        with Session(engine) as s:
            rows = s.exec(select(SitePattern).where(SitePattern.source_id == self.source_id)).all()
            return {r.pattern: PatternStat(r.hits, r.misses, r.discovered) for r in rows}

    def save_patterns(self, stats: dict[str, PatternStat]) -> None:
        with Session(engine) as s:
            rows = {r.pattern: r for r in s.exec(
                select(SitePattern).where(SitePattern.source_id == self.source_id)).all()}
            for pattern, st in stats.items():
                row = rows.get(pattern) or SitePattern(source_id=self.source_id, pattern=pattern)
                row.hits, row.misses, row.discovered = st.hits, st.misses, st.discovered
                s.add(row)
            s.commit()

    def save_result(self, r: PageResult) -> None:
        with Session(engine) as s:
            old = s.exec(select(Item).where(Item.url == r.url)).first()
            if old and r.refresh:
                self._update(s, old, r)
                return
            if old:
                return
            hit = r.kind in ("recipe", "fact")
            item = Item(
                source_id=self.source_id, url=r.url, kind=ItemKind(r.kind),
                status=ItemStatus.raw if hit else ItemStatus.skipped,
                title=r.title, raw_text=r.text or None,
                structured_json=json.dumps(r.structured, ensure_ascii=False) if r.structured else None,
                confidence=r.confidence, method=r.method, reason=r.reason, language=r.language, pattern=r.pattern)
            dup = None
            if r.kind == "recipe":
                fp = fingerprint(r.title, (r.structured or {}).get("ingredients"))
                item.fingerprint = json.dumps(fp)
                rows = s.exec(select(Item.id, Item.fingerprint).where(
                    Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None),
                    col(Item.fingerprint).is_not(None))).all()
                dup = find_duplicate(fp, [(i, json.loads(f)) for i, f in rows])
                if dup:
                    item.duplicate_of = dup
            s.add(item)
            s.commit()
        if dup:  # log po zamknieciu transakcji (patrz export.translate_one)
            self.log(f"[duplikat] {r.title[:60]} = przepis #{dup} (dodano jako kolejne zrodlo)")

    @staticmethod
    def _update(s: Session, old: Item, r: PageResult) -> None:
        """Odswiezenie znanej strony: nowa tresc i klasyfikacja; przetlumaczonych/zatwierdzonych nie cofamy."""
        hit = r.kind in ("recipe", "fact")
        if hit:
            old.raw_text = r.text or old.raw_text
            if r.structured:
                old.structured_json = json.dumps(r.structured, ensure_ascii=False)
            old.title = r.title or old.title
        old.confidence, old.method, old.reason, old.pattern = r.confidence, r.method, r.reason, r.pattern
        if old.status in (ItemStatus.raw, ItemStatus.skipped, ItemStatus.error):
            old.kind = ItemKind(r.kind)
            old.status = ItemStatus.raw if hit else ItemStatus.skipped
        old.updated_at = datetime.now(timezone.utc)
        s.add(old)
        s.commit()

    def log(self, message: str) -> None:
        job_log(self.job_id, message)

    def progress(self, done: int, total: int) -> None:
        if done == self._last_progress:
            return
        self._last_progress = done
        with Session(engine) as s:
            job = s.get(Job, self.job_id)
            job.progress, job.total = done, total
            s.add(job)
            s.commit()

    def should_stop(self) -> bool:
        return self.runner.should_stop(self.job_id)

    def activity(self, step: str, detail: str = "") -> None:
        activity.set(self.job_id, step, detail)

    def save_frontier(self, queue: list[dict]) -> None:
        with Session(engine) as s:
            job = s.get(Job, self.job_id)
            job.state_json = json.dumps({"frontier": queue}, ensure_ascii=False)
            s.add(job)
            s.commit()

    def flag_js(self) -> None:
        with Session(engine) as s:
            src = s.get(Source, self.source_id)
            src.needs_js = True
            s.add(src)
            s.commit()

    def profile_feedback(self, used: int, failed: int) -> None:
        with Session(engine) as s:
            src = s.get(Source, self.source_id)
            prof = ScraperProfile.from_json(src.scraper_profile)
            if not prof:
                return
            prof.stats["used"] = prof.stats.get("used", 0) + used
            prof.stats["failed"] = prof.stats.get("failed", 0) + failed
            total = used + failed
            degraded = total >= 5 and failed / total > 0.3
            if degraded:
                src.profile_status = "degraded"
            src.scraper_profile = prof.to_json()
            s.add(src)
            s.commit()
        if degraded:
            self.log(f"Profil scrapera zawiodl na {failed}/{total} stronach - strona mogla sie zmienic, "
                     "przebuduj profil")


def _load_source(source_id: int) -> tuple[Source, ScraperProfile | None]:
    with Session(engine) as s:
        src = s.get(Source, source_id)
        s.expunge(src)
    prof = ScraperProfile.from_json(src.scraper_profile) if src.profile_status in ("ready", "degraded") else None
    return src, prof


def _proxy(source_id: int, store: "DbStore") -> str | None:
    with Session(engine) as s:
        country = s.get(Source, source_id).country
    proxy = settings.proxy_for(country)
    if proxy:
        store.log(f"Ruch przez VPN kraju '{country}' ({proxy})")
    return proxy


def _keywords(src: Source) -> list[str]:
    return [k.strip() for k in (src.keywords or "").split(",") if k.strip()]


def _finish(job_id: int, source_id: int, stats: dict) -> None:
    with Session(engine) as s:
        src = s.get(Source, source_id)
        src.last_scan_at = datetime.now(timezone.utc)
        src.pages_scanned = (src.pages_scanned or 0) + stats.get("fetched", 0)
        s.add(src)
        job = s.get(Job, job_id)
        job.stats_json = json.dumps(stats)
        s.add(job)
        s.commit()


async def _crawl(job_id: int, source_id: int, runner: JobRunner, llm, cfg: CrawlConfig, start_url: str) -> dict:
    store = DbStore(job_id, source_id, runner)
    if llm is None and not cfg.profile:
        store.log("Uwaga: brak LLM - klasyfikacja tylko heurystykami")
    fetcher = Fetcher(settings.user_agent, settings.request_delay_s, proxy=_proxy(source_id, store))
    try:
        stats = await SmartCrawler(start_url, fetcher, store, cfg, llm).run()
    finally:
        await fetcher.aclose()
    _finish(job_id, source_id, stats)
    return stats


def load_frontier(job_id: int | None) -> list[dict] | None:
    if not job_id:
        return None
    with Session(engine) as s:
        job = s.get(Job, job_id)
        state = json.loads(job.state_json) if job and job.state_json else {}
    return state.get("frontier") or None


def _useful_counts(source_id: int) -> dict:
    """Ile przydatnych przepisow i ciekawostek z tej strony jest juz w bazie."""
    with Session(engine) as s:
        rec = s.exec(select(func.count()).select_from(Item).where(
            Item.source_id == source_id, Item.kind == ItemKind.recipe, Item.status != ItemStatus.skipped)).one()
        fact = s.exec(select(func.count()).select_from(Item).where(
            Item.source_id == source_id, Item.kind == ItemKind.fact)).one()
    return {"recipe": rec, "fact": fact}


async def run_scan(job_id: int, source_id: int, runner: JobRunner, llm=None, max_pages: int | None = None,
                   resume_from: int | None = None) -> None:
    src, prof = _load_source(source_id)
    limit = max_pages or src.max_pages or 50
    frontier = load_frontier(resume_from)
    store = DbStore(job_id, source_id, runner)
    phased = settings.phased_scan and llm is not None and (prof is None or src.profile_status == "degraded")
    if not phased:
        cfg = CrawlConfig(max_pages=limit, max_depth=src.max_depth or 3, keywords=_keywords(src), profile=prof,
                          resume_queue=frontier)
        await _crawl(job_id, source_id, runner, llm, cfg, src.base_url)
        return

    # --- Etap 1/3: rozpoznanie - skanuj do zebrania probek przepisow i ciekawostek (z LLM)
    stats: dict = {}
    have = _useful_counts(source_id)
    goal = {"recipe": settings.probe_recipes, "fact": settings.probe_facts}
    need = {k: max(0, v - have[k]) for k, v in goal.items()}
    if any(need.values()) and src.profile_status != "degraded":
        store.log(f"Etap 1/3 - rozpoznanie strony: szukam probek (przepisy {have['recipe']}/{goal['recipe']}, "
                  f"ciekawostki {have['fact']}/{goal['fact']}), max {settings.probe_max_pages} stron")
        cfg = CrawlConfig(max_pages=settings.probe_max_pages, max_depth=src.max_depth or 3, keywords=_keywords(src),
                          resume_queue=frontier, stop_when={k: v for k, v in need.items() if v})
        stats["rozpoznanie"] = await _crawl(job_id, source_id, runner, llm, cfg, src.base_url)
        frontier = load_frontier(job_id)
        if runner.should_stop(job_id):
            return
    else:
        store.log(f"Etap 1/3 - pominiety: w bazie sa juz probki (przepisy {have['recipe']}, ciekawostki {have['fact']})")

    # --- Etap 2/3: budowa profilu scrapera z zebranych probek
    ok = False
    if _useful_counts(source_id)["recipe"] >= 2:
        store.log("Etap 2/3 - budowa profilu scrapera z zebranych probek")
        try:
            ok = await build_source_profile(job_id, source_id, llm, store)
        except RuntimeError as exc:
            store.log(f"Profil nie powstal: {exc}")
    else:
        store.log("Etap 2/3 - pominiety: za malo przepisow na tej stronie do zbudowania profilu")
    stats["profil"] = {"passed": ok}
    if runner.should_stop(job_id):
        return

    # --- Etap 3/3: pobieranie - z profilem (bez LLM dla stron przepisow/artykulow) albo zwyklym sposobem
    src, prof = _load_source(source_id)
    store.log(f"Etap 3/3 - pobieranie {'z profilem scrapera' if prof else 'bez profilu (LLM)'}: max {limit} stron")
    cfg = CrawlConfig(max_pages=limit, max_depth=src.max_depth or 3, keywords=_keywords(src), profile=prof,
                      resume_queue=frontier)
    stats["pobieranie"] = await _crawl(job_id, source_id, runner, llm, cfg, src.base_url)
    _save_phase_stats(job_id, stats)


async def run_harvest(job_id: int, source_id: int, runner: JobRunner, llm=None, max_pages: int | None = None,
                      resume_from: int | None = None) -> None:
    """Pelne pobieranie strony gotowym profilem scrapera: sitemapy + linki, przepisy i artykuly selektorami."""
    src, prof = _load_source(source_id)
    if prof is None:
        raise RuntimeError("Ta strona nie ma jeszcze profilu scrapera - najpierw zbuduj profil (albo zwykly skan)")
    limit = max_pages or settings.harvest_max_pages
    store = DbStore(job_id, source_id, runner)
    facts = "tak" if prof.fact_url_regex else "nie (zwykla klasyfikacja)"
    store.log(f"Pelne pobieranie z profilem ({prof.built_with}): max {limit} stron; przepisy wg "
              f"{prof.recipe_url_regex!r}; ciekawostki z profilu: {facts}")
    cfg = CrawlConfig(max_pages=limit, max_depth=max(src.max_depth or 3, 6), keywords=_keywords(src), profile=prof,
                      resume_queue=load_frontier(resume_from), harvest=True)
    await _crawl(job_id, source_id, runner, llm, cfg, src.base_url)


def _save_phase_stats(job_id: int, phases: dict) -> None:
    total: dict = {}
    for st in (v for k, v in phases.items() if k != "profil"):
        for k, v in st.items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    total["profile_ok"] = phases.get("profil", {}).get("passed", False)
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.stats_json = json.dumps(total)
        s.add(job)
        s.commit()


async def run_fetch_list(job_id: int, source_id: int, urls: list[str], runner: JobRunner, llm=None) -> None:
    src, prof = _load_source(source_id)
    cfg = CrawlConfig(max_pages=len(urls), max_depth=0, use_sitemap=False, keywords=_keywords(src),
                      profile=prof, seeds=urls)
    await _crawl(job_id, source_id, runner, llm, cfg, src.base_url)


async def run_scan_pages(job_id: int, source_id: int, urls: list[str], runner: JobRunner, llm=None,
                         depth: int = 1, max_new: int = 100) -> None:
    """"Skanuj zaznaczone": pobiera ponownie wybrane strony (aktualizuje je) i idzie po ich linkach
    `depth` poziomow w glab, szukajac nowych przepisow."""
    src, prof = _load_source(source_id)
    cfg = CrawlConfig(max_pages=len(urls) + max_new, max_depth=depth, use_sitemap=False, keywords=_keywords(src),
                      profile=prof, seeds=urls, refresh_seeds=True)
    await _crawl(job_id, source_id, runner, llm, cfg, src.base_url)


async def run_build_profile(job_id: int, source_id: int, runner: JobRunner, llm=None) -> None:
    await build_source_profile(job_id, source_id, llm, DbStore(job_id, source_id, runner), save_stats=True)


def _sample_urls(source_id: int) -> tuple[list[str], list[str], list[str]]:
    """(przepisy, ciekawostki, inne) tej strony - do budowy profilu. Przepisy kulinarne/pominiete to 'inne'."""
    with Session(engine) as s:
        recipes = s.exec(select(Item.url).where(Item.source_id == source_id, Item.kind == ItemKind.recipe,
                                                Item.status != ItemStatus.skipped)
                         .order_by(col(Item.confidence).desc()).limit(40)).all()
        facts = s.exec(select(Item.url).where(Item.source_id == source_id, Item.kind == ItemKind.fact)
                       .order_by(col(Item.confidence).desc()).limit(20)).all()
        others = s.exec(select(Item.url).where(Item.source_id == source_id, (Item.kind == ItemKind.other) | (
            (Item.kind == ItemKind.recipe) & (Item.status == ItemStatus.skipped))).limit(40)).all()
    return [u for u in recipes if "#" not in u], [u for u in facts if "#" not in u], list(others)


async def build_source_profile(job_id: int, source_id: int, llm, store: "DbStore", save_stats: bool = False) -> bool:
    """Buduje profil scrapera z przepisow (i ciekawostek) znalezionych wczesniej na tej stronie. True = gotowy."""
    recipes, facts, others = _sample_urls(source_id)
    if len(recipes) < 2:
        raise RuntimeError("Za malo znanych przepisow z tej strony (min. 2) - najpierw uruchom zwykly skan")
    if llm is None:
        store.log("Uwaga: brak LLM - profil tylko z wyuczonych wzorcow i JSON-LD")

    fetcher = Fetcher(settings.user_agent, settings.request_delay_s, proxy=_proxy(source_id, store))
    samples, fact_samples = [], []
    try:
        for kind, urls, bucket in (("przepisu", recipes[: settings.profile_samples], samples),
                                   ("ciekawostki", facts[: min(3, settings.profile_samples)], fact_samples)):
            for url in urls:
                store.activity(f"pobieram probke {kind}", url)
                res = await fetcher.get(url)
                if res.ok:
                    page = await asyncio.to_thread(parse_page, res.text, url)
                    bucket.append((url, res.text, page.jsonld))
                    store.log(f"Probka {kind}: {url}")
    finally:
        await fetcher.aclose()
    if len(samples) < 2:
        raise RuntimeError("Nie udalo sie pobrac probek stron z przepisami")

    store.activity("LLM buduje i sprawdza profil scrapera", f"{len(samples)} przepisow, {len(fact_samples)} ciekawostek")
    prof, rep = await build_profile(llm, samples, list(recipes), list(others), store.load_patterns(),
                                    log_fn=store.log, fact_samples=fact_samples, fact_urls=list(facts))
    stats = {"passed": rep["passed"], "coverage": rep.get("coverage"),
             "regex_precision": (rep.get("regex") or {}).get("precision"),
             "regex_recall": (rep.get("regex") or {}).get("recall"),
             "facts_ok": (rep.get("facts") or {}).get("ok", False)}
    with Session(engine) as s:
        src = s.get(Source, source_id)
        src.scraper_profile = prof.to_json()
        src.profile_status = "ready" if rep["passed"] else "failed"
        s.add(src)
        if save_stats:
            job = s.get(Job, job_id)
            job.stats_json = json.dumps(stats)
            s.add(job)
        s.commit()
    if rep["passed"]:
        store.log(f"Profil gotowy ({prof.built_with}). Kolejne skany tej strony uzyja selektorow zamiast LLM.")
    else:
        store.log("Profil nie przeszedl walidacji: " + "; ".join(rep.get("errors", [])[:5]))
    return rep["passed"]


def backfill_duplicates() -> int:
    """Jednorazowo dla starszych danych: normalizuje JSON-LD, liczy odciski i laczy duplikaty."""
    from app.scrapers.recipe_data import normalize_jsonld_recipe

    with Session(engine) as s:
        todo = s.exec(select(Item).where(Item.kind == ItemKind.recipe, col(Item.fingerprint).is_(None))
                      .order_by(Item.id)).all()
        if not todo:
            return 0
        known = [(i.id, json.loads(i.fingerprint)) for i in s.exec(select(Item).where(
            Item.kind == ItemKind.recipe, col(Item.fingerprint).is_not(None), col(Item.duplicate_of).is_(None))).all()]
        merged = 0
        for it in todo:
            st = json.loads(it.structured_json) if it.structured_json else {}
            if "@type" in st:  # stary zapis: surowy JSON-LD -> format znormalizowany
                st = normalize_jsonld_recipe(st)
                it.structured_json = json.dumps(st, ensure_ascii=False)
            fp = fingerprint(it.title or st.get("title") or "", st.get("ingredients"))
            it.fingerprint = json.dumps(fp)
            dup = find_duplicate(fp, known)
            if dup:
                it.duplicate_of = dup
                merged += 1
            else:
                known.append((it.id, fp))
            s.add(it)
        s.commit()
        return merged
