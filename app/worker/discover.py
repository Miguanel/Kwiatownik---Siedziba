"""Zadanie 'discover': szukanie nowych stron z przepisami w wybranym kraju."""
import json
from datetime import datetime, timezone

from sqlmodel import Session, func, select

from app.config import settings
from app.db import engine
from app.discovery.countries import get_country
from app.discovery.domains import registrable_domain
from app.discovery.finder import DiscoveryConfig, FoundSite, SiteFinder, site_root
from app.discovery.search import BraveSearch, SearxngSearch
from app.models import DiscoveredSite, Job, SearchQuery, Source
from app.scrapers.fetcher import Fetcher
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


class DbDiscoveryStore:
    def __init__(self, job_id: int, runner: JobRunner):
        self.job_id = job_id
        self.runner = runner

    def known_domains(self) -> set[str]:
        with Session(engine) as s:
            found = set(s.exec(select(DiscoveredSite.domain)).all())
            sources = {registrable_domain(u) for u in s.exec(select(Source.base_url)).all()}
        return found | {d for d in sources if d}

    def used_queries(self, country: str) -> set[str]:
        with Session(engine) as s:
            return set(s.exec(select(SearchQuery.text).where(SearchQuery.country == country)).all())

    def save_query(self, country: str, text: str, results: int, new_sites: int) -> None:
        with Session(engine) as s:
            s.add(SearchQuery(country=country, text=text, results=results, new_sites=new_sites))
            s.commit()

    def save_site(self, site: FoundSite) -> None:
        with Session(engine) as s:
            if s.exec(select(DiscoveredSite).where(DiscoveredSite.domain == site.domain)).first():
                return
            s.add(DiscoveredSite(domain=site.domain, country=site.country, url=site.url, title=site.title,
                                 snippet=site.snippet, query=site.query, engines=",".join(site.engines),
                                 status=site.status, score=site.score, language=site.language,
                                 reason=site.reason))
            s.commit()
        if site.status in ("verified", "maybe"):
            self.log(f"[{site.status}] {site.domain} - {site.title[:60]}")

    def bump_site(self, domain: str) -> None:
        with Session(engine) as s:
            row = s.exec(select(DiscoveredSite).where(DiscoveredSite.domain == domain)).first()
            if row:
                row.hits = (row.hits or 1) + 1
                s.add(row)
                s.commit()

    def log(self, message: str) -> None:
        job_log(self.job_id, message)

    def progress(self, done: int, total: int) -> None:
        with Session(engine) as s:
            job = s.get(Job, self.job_id)
            job.progress, job.total = done, total
            s.add(job)
            s.commit()

    def should_stop(self) -> bool:
        return self.runner.should_stop(self.job_id)

    def activity(self, step: str, detail: str = "") -> None:
        activity.set(self.job_id, step, detail)


def build_backends() -> list:
    backends = []
    if settings.searxng_url.strip():
        backends.append(SearxngSearch(settings.searxng_url))
    if settings.brave_api_key:
        backends.append(BraveSearch(settings.brave_api_key))
    return backends


async def run_discover(job_id: int, runner: JobRunner, llm, country_code: str, cfg: DiscoveryConfig,
                       auto_add: bool = False) -> None:
    country = get_country(country_code)
    if not country:
        raise RuntimeError(f"Nieznany kraj: {country_code}")
    backends = build_backends()
    if not backends:
        raise RuntimeError("Brak wyszukiwarki - ustaw SEARXNG_URL albo BRAVE_API_KEY w .env")
    store = DbDiscoveryStore(job_id, runner)
    proxy = settings.proxy_for(country.code)
    fetcher = Fetcher(settings.user_agent, settings.request_delay_s, proxy=proxy,
                      accept_language=f"{country.language},{country.language.split('-')[0]};q=0.9,en;q=0.5")
    try:
        stats = await SiteFinder(country, backends, store, fetcher, cfg, llm, proxy_used=bool(proxy)).run()
    finally:
        await fetcher.aclose()
        for b in backends:
            await b.aclose()
    if auto_add:
        added = add_verified_as_sources(country.code)
        store.log(f"Dodano {added} zweryfikowanych stron jako zrodla")
        stats["added"] = added
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.stats_json = json.dumps(stats)
        s.add(job)
        s.commit()


def add_site_as_source(site_id: int) -> int | None:
    """Tworzy Source z odkrytej strony (albo zwraca istniejace) i usuwa ja z listy odkrytych.
    Domena nadal jest "znana" (przez liste zrodel), wiec wyszukiwarka nie znajdzie jej ponownie."""
    with Session(engine) as s:
        site = s.get(DiscoveredSite, site_id)
        if not site:
            return None
        root = site_root(site.url)
        existing = next((src for src in s.exec(select(Source)).all()
                         if registrable_domain(src.base_url) == site.domain), None)
        src = existing or Source(name=site.domain, base_url=root,
                                 language=(site.language or "").split("-")[0] or "en", country=site.country)
        s.add(src)
        s.delete(site)
        s.commit()
        return src.id


def cleanup_discoveries() -> int:
    """Porzadki przy starcie: usuwa z odkrytych strony juz dodane jako zrodla i poprawia domeny
    (wczesniejsza wersja myla np. drink.co.ua z co.ua). Zwraca liczbe zmienionych wierszy."""
    changed = 0
    with Session(engine) as s:
        sources = {registrable_domain(u) for u in s.exec(select(Source.base_url)).all()}
        rows = s.exec(select(DiscoveredSite).order_by(DiscoveredSite.id)).all()
        seen: set[str] = set()
        for row in rows:
            dom = registrable_domain(row.url) or row.domain
            if row.status == "added" or dom in sources or dom in seen:
                s.delete(row)
                changed += 1
                continue
            if dom != row.domain:
                row.domain = dom
                s.add(row)
                changed += 1
            seen.add(dom)
        s.commit()
    return changed


def add_verified_as_sources(country: str | None = None) -> int:
    with Session(engine) as s:
        stmt = select(DiscoveredSite.id).where(DiscoveredSite.status == "verified")
        if country:
            stmt = stmt.where(DiscoveredSite.country == country)
        ids = list(s.exec(stmt).all())
    return sum(1 for i in ids if add_site_as_source(i))


def discovery_counts() -> dict[str, dict[str, int]]:
    with Session(engine) as s:
        rows = s.exec(select(DiscoveredSite.country, DiscoveredSite.status, func.count())
                      .group_by(DiscoveredSite.country, DiscoveredSite.status)).all()
    out: dict[str, dict[str, int]] = {}
    for c, st, n in rows:
        out.setdefault(c, {})[st] = n
    return out


async def vpn_check(country: str) -> dict:
    """Sprawdza, z jakiego kraju widzi nas internet przez proxy danego kraju."""
    import httpx
    proxy = settings.proxy_for(country)
    started = datetime.now(timezone.utc)
    try:
        async with httpx.AsyncClient(proxy=proxy, timeout=20) as c:
            data = (await c.get("https://ipinfo.io/json")).json()
        return {"ok": True, "proxy": proxy, "ip": data.get("ip"), "country": (data.get("country") or "").lower(),
                "city": data.get("city"), "ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000)}
    except Exception as exc:
        return {"ok": False, "proxy": proxy, "error": f"{type(exc).__name__}: {exc}"}
