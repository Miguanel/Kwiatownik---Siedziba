"""Analiza pobranej strony HTML: tekst, JSON-LD, linki, naglowki, wykrywanie stron JS."""
import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from app.scrapers.fetcher import normalize_url

try:
    import trafilatura
except ImportError:  # pragma: no cover - awaryjnie ekstrakcja przez BeautifulSoup
    trafilatura = None


@dataclass
class PageData:
    url: str
    canonical: str | None
    title: str
    lang: str | None
    text: str
    headings: list[str]
    list_items: list[str]
    jsonld: list[dict]
    links: list[tuple[str, str]]          # (url, tekst linku)
    feeds: list[str] = field(default_factory=list)
    html_len: int = 0
    script_count: int = 0

    @property
    def looks_js_rendered(self) -> bool:
        """Duzo HTML/skryptow, a prawie zero tresci -> strona budowana w JavaScript."""
        return self.html_len > 30_000 and len(self.text) < 250 and self.script_count > 10


def _flatten_jsonld(node) -> list[dict]:
    out = []
    if isinstance(node, list):
        for n in node:
            out += _flatten_jsonld(n)
    elif isinstance(node, dict):
        out.append(node)
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in node:
                out += _flatten_jsonld(node[key])
    return out


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_page(html: str, url: str) -> PageData:
    soup = BeautifulSoup(html, "lxml")

    jsonld = []
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            jsonld += _flatten_jsonld(json.loads(tag.string or tag.get_text() or "", strict=False))
        except ValueError:
            continue

    canonical = None
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        canonical = normalize_url(link["href"], url)

    feeds = [normalize_url(l["href"], url) for l in soup.find_all("link", rel="alternate")
             if l.get("href") and "xml" in (l.get("type") or "")]

    links = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(a["href"], url)
        if u:
            links.append((u, _clean(a.get_text())[:120]))

    title = _clean(soup.title.get_text()) if soup.title else ""
    h1 = soup.find("h1")
    if h1 and _clean(h1.get_text()):
        title = _clean(h1.get_text())

    headings = [_clean(h.get_text()) for h in soup.find_all(["h1", "h2", "h3", "h4"])][:60]
    list_items = [_clean(li.get_text()) for li in soup.find_all("li")][:400]
    script_count = len(soup.find_all("script"))
    lang = (soup.html.get("lang") if soup.html else None) or None

    text = ""
    if trafilatura is not None:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True,
                                   favor_recall=True) or ""
    if not text:
        for bad in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            bad.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = "\n".join(_clean(t) for t in main.get_text("\n").splitlines() if _clean(t))

    return PageData(url=url, canonical=canonical, title=title, lang=lang, text=text.strip(),
                    headings=headings, list_items=list_items, jsonld=jsonld, links=links,
                    feeds=[f for f in feeds if f], html_len=len(html), script_count=script_count)
