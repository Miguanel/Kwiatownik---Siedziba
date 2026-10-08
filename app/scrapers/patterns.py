"""Uczenie sie struktury strony.

Kazdy URL zamieniamy na wzorzec, np.
  /recipes/nettle-soup          -> /recipes/{slug}
  /2024/05/elderflower-cordial/ -> /{n}/{n}/{slug}
Dla kazdego wzorca liczymy: trafienia (przepis/ciekawostka), pudla (inna strona)
oraz "odkrycia" (ile trafien znaleziono przez linki z takich stron - strony-huby).
Na tej podstawie crawler ustala priorytety i pomija nieproduktywne czesci serwisu.
"""
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.scrapers.fetcher import SKIP_EXTENSIONS

NEVER_VISIT = ("login", "logout", "signin", "signup", "register", "cart", "checkout", "account", "wp-admin",
               "wp-login", "/feed", "/comments", "print=", "share=", "/search", "?s=", "privacy", "cookie",
               "terms", "impressum", "datenschutz", "mailto:", "javascript:", "/wp-json")
HUB_HINTS = ("/tag/", "/category/", "/categories/", "/page/", "/archive", "/author/", "/topics/", "/index")


def url_pattern(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "/"
    out = []
    parts = path.split("/")
    for i, seg in enumerate(parts):
        low = seg.lower()
        if re.fullmatch(r"\d+", low):
            out.append("{n}")
        elif re.fullmatch(r"[\w-]*\d{3,}[\w-]*", low) or len(low) > 40:
            out.append("{id}")
        elif ("-" in low or "_" in low) and (low.count("-") + low.count("_") >= 2 or i == len(parts) - 1):
            out.append("{slug}")
        else:
            out.append(low)
    return "/" + "/".join(out)


def should_skip(url: str) -> bool:
    low = url.lower()
    path = urlparse(low).path
    return path.endswith(SKIP_EXTENSIONS) or any(x in low for x in NEVER_VISIT)


@dataclass
class PatternStat:
    hits: int = 0
    misses: int = 0
    discovered: int = 0

    @property
    def productivity(self) -> float:
        """0..1 - wygladzona skutecznosc (nowy wzorzec startuje z 0.5)."""
        return (self.hits + 1) / (self.hits + self.misses + 2)

    @property
    def hub_value(self) -> float:
        return min(1.0, self.discovered / (self.hits + self.misses + 1))

    @property
    def dead(self) -> bool:
        return self.misses >= 8 and self.hits == 0 and self.discovered == 0


class PatternBook:
    def __init__(self, stats: dict[str, PatternStat] | None = None):
        self.stats: dict[str, PatternStat] = stats or {}

    def get(self, pattern: str) -> PatternStat:
        return self.stats.setdefault(pattern, PatternStat())

    def record(self, pattern: str, hit: bool, parent: str | None) -> None:
        s = self.get(pattern)
        if hit:
            s.hits += 1
            if parent is not None:
                self.get(parent).discovered += 1
        else:
            s.misses += 1

    def is_dead(self, pattern: str) -> bool:
        return pattern in self.stats and self.stats[pattern].dead

    def best(self, n: int = 5) -> list[tuple[str, PatternStat]]:
        return sorted(((p, s) for p, s in self.stats.items() if s.hits),
                      key=lambda x: (x[1].productivity, x[1].hits), reverse=True)[:n]


def keyword_score(url: str, anchor: str, keywords: list[str]) -> float:
    hay = f"{urlparse(url).path} {anchor}".lower()
    hits = sum(1 for k in keywords if k in hay)
    return min(1.0, hits / 2)


def url_priority(url: str, anchor: str, depth: int, book: PatternBook, keywords: list[str],
                 from_sitemap: bool = False) -> float:
    stat = book.stats.get(url_pattern(url))
    prod = stat.productivity if stat else 0.5
    hub = stat.hub_value if stat else 0.0
    score = 0.45 * prod + 0.35 * keyword_score(url, anchor, keywords) + 0.2 / (1 + depth) + 0.25 * hub
    if from_sitemap:
        score += 0.05
    if any(h in url.lower() for h in HUB_HINTS):
        score -= 0.15 if not hub else 0.0
    return round(score, 4)
