"""Skupione pelzanie (R10 z ekspertyzy 2026-10-14): ocena adresu ZANIM strona zostanie pobrana.

72% pobranych stron trafialo do "inne". Naiwny klasyfikator Bayesa uczy sie na stronach juz zebranych z TEJ
SAMEJ witryny: slowa sciezki adresu i tekstu linku -> czy strona okazala sie przydatna (przepis albo ciekawostka).
Crawler dodaje te ocene do priorytetu kolejki, wiec limit stron na skan idzie najpierw na adresy podobne do
przydatnych. Nic nie jest pomijane na sztywno - adresy "na pewno inne" ida tylko na koniec kolejki.
"""
import math
import re
from collections import Counter
from urllib.parse import unquote, urlparse

_SPLIT = re.compile(r"[^\w]+|_", re.U)
MIN_PAGES = 30          # tyle stron witryny z ocena, zeby model mial sens
MIN_CLASS = 5           # ...w tym tyle przydatnych i tyle nieprzydatnych


def tokens(url: str, anchor: str = "") -> list[str]:
    p = urlparse(url)
    words = _SPLIT.split(unquote(p.path).lower()) + _SPLIT.split((anchor or "").lower())
    out = [w for w in words if len(w) >= 3 and not w.isdigit()]
    if p.query:
        out.append("?query")
    out.append(f"#depth{min(4, p.path.strip('/').count('/') + 1)}")
    return out


class UrlModel:
    def __init__(self, pos: Counter, neg: Counter, n_pos: int, n_neg: int):
        self.pos, self.neg, self.n_pos, self.n_neg = pos, neg, n_pos, n_neg
        self.vocab = len(set(pos) | set(neg)) or 1
        self._tot_pos = sum(pos.values())
        self._tot_neg = sum(neg.values())

    @classmethod
    def train(cls, rows: list[tuple[str, bool]]) -> "UrlModel | None":
        """rows: (adres, czy przydatna). None, gdy za malo danych z obu stron."""
        pos, neg = Counter(), Counter()
        n_pos = n_neg = 0
        for url, useful in rows:
            t = set(tokens(url))
            if useful:
                pos.update(t)
                n_pos += 1
            else:
                neg.update(t)
                n_neg += 1
        if n_pos + n_neg < MIN_PAGES or n_pos < MIN_CLASS or n_neg < MIN_CLASS:
            return None
        return cls(pos, neg, n_pos, n_neg)

    def probability(self, url: str, anchor: str = "") -> float:
        """P(strona przydatna | slowa adresu) - wygladzony naiwny Bayes."""
        lp = math.log(self.n_pos / (self.n_pos + self.n_neg))
        ln = math.log(self.n_neg / (self.n_pos + self.n_neg))
        for t in set(tokens(url, anchor)):
            lp += math.log((self.pos[t] + 1) / (self._tot_pos + self.vocab))
            ln += math.log((self.neg[t] + 1) / (self._tot_neg + self.vocab))
        d = max(-30.0, min(30.0, ln - lp))
        return 1.0 / (1.0 + math.exp(d))

    def bonus(self, url: str, anchor: str = "") -> float:
        """Dodatek do priorytetu kolejki crawlera: od -0.3 (na pewno inne) do +0.3 (na pewno przydatne)."""
        return round(0.6 * (self.probability(url, anchor) - 0.5), 4)


# ------------------------------------------------------------ wydajnosc zrodla
AUTO_OFF_PAGES = 200     # po tylu ocenionych stronach bez zadnego przepisu i prawie bez ciekawostek...
AUTO_OFF_FACT_SHARE = 0.05


def poor_yield(pages: int, recipes: int, facts: int) -> str | None:
    """Powod wylaczenia zrodla albo None."""
    if pages >= AUTO_OFF_PAGES and recipes == 0 and facts / max(1, pages) < AUTO_OFF_FACT_SHARE:
        return (f"{pages} ocenionych stron, 0 przepisow, {facts} ciekawostek "
                f"({100 * facts / max(1, pages):.0f}%) - za malo, zeby oplacalo sie skanowac dalej")
    return None
