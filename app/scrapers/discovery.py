"""Szukanie adresow podstron: sitemapy (z robots.txt i standardowych lokalizacji) oraz kanaly RSS."""
import logging
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from app.scrapers.fetcher import Fetcher, normalize_url, site_key

log = logging.getLogger(__name__)

STANDARD_SITEMAPS = ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml", "/sitemap-index.xml")
# pod-sitemapy, ktore zwykle nie zawieraja tresci
BORING_SITEMAPS = ("image", "video", "tag", "author", "category", "attachment", "product", "page-sitemap")


def _parse_xml(text: str) -> BeautifulSoup:
    return BeautifulSoup(text, "xml")


async def sitemap_urls(fetcher: Fetcher, base_url: str, keywords: list[str], limit: int = 3000,
                       max_sitemaps: int = 25) -> list[str]:
    """Zwraca adresy stron z sitemap, najpierw te, ktore wygladaja na trafne (slowa kluczowe)."""
    root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    queue = list(dict.fromkeys(await fetcher.sitemaps_from_robots(base_url) + [root + p for p in STANDARD_SITEMAPS]))
    seen_maps: set[str] = set()
    pages: list[str] = []
    fetched = 0
    while queue and fetched < max_sitemaps and len(pages) < limit:
        sm = queue.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        res = await fetcher.get(sm, html_only=False)
        fetched += 1
        if not res.ok or "<" not in res.text[:200]:
            continue
        doc = _parse_xml(res.text)
        children = [loc.get_text(strip=True) for s in doc.find_all("sitemap") for loc in s.find_all("loc")]
        if children:
            # najpierw pod-sitemapy "z trescia" i ze slowami kluczowymi w nazwie
            def rank(u: str) -> int:
                low = u.lower()
                return (2 if any(k in low for k in keywords) else 0) + (1 if "post" in low or "recipe" in low else 0) \
                    - (3 if any(b in low for b in BORING_SITEMAPS) else 0)
            queue = sorted(children, key=rank, reverse=True) + queue
            continue
        for u in doc.find_all("url"):
            loc = u.find("loc")
            nu = normalize_url(loc.get_text(strip=True)) if loc else None
            if nu and site_key(nu) == site_key(base_url):
                pages.append(nu)
    pages = list(dict.fromkeys(pages))
    log.info("Sitemapy: %d map, %d adresow", len(seen_maps), len(pages))
    return pages[:limit]


async def feed_urls(fetcher: Fetcher, feeds: list[str], limit: int = 200) -> list[str]:
    """Adresy wpisow z kanalow RSS/Atom."""
    out: list[str] = []
    for f in feeds[:3]:
        res = await fetcher.get(f, html_only=False)
        if not res.ok:
            continue
        doc = _parse_xml(res.text)
        for item in doc.find_all(["item", "entry"]):
            link = item.find("link")
            href = (link.get("href") or link.get_text(strip=True)) if link else None
            nu = normalize_url(href) if href else None
            if nu:
                out.append(nu)
    return list(dict.fromkeys(out))[:limit]
