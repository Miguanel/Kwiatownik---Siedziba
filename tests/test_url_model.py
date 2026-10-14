"""Skupione pelzanie (R10): ocena adresu przed pobraniem + wylaczanie zrodel bez wydajnosci."""
from app.scrapers.url_model import UrlModel, poor_yield, tokens


def _rows():
    rows = [(f"https://ex.com/przepisy/nalewka-z-{i}", True) for i in range(12)]
    rows += [(f"https://ex.com/ziola/{i}-wlasciwosci-lecznicze", True) for i in range(8)]
    rows += [(f"https://ex.com/sklep/produkt-{i}?koszyk=1", False) for i in range(20)]
    rows += [(f"https://ex.com/tag/sport-{i}", False) for i in range(10)]
    return rows


def test_model_needs_enough_pages_of_both_kinds():
    assert UrlModel.train(_rows()[:10]) is None
    assert UrlModel.train([(f"https://ex.com/a{i}", True) for i in range(40)]) is None


def test_model_prefers_addresses_like_useful_pages():
    m = UrlModel.train(_rows())
    assert m.probability("https://ex.com/przepisy/nalewka-z-pigwy") > 0.9
    assert m.probability("https://ex.com/sklep/produkt-99?koszyk=1") < 0.1
    assert m.bonus("https://ex.com/przepisy/nalewka-z-pigwy") > 0.2 > -0.2 > m.bonus("https://ex.com/tag/sport-x")
    assert "?query" in tokens("https://ex.com/a?b=1") and "#depth2" in tokens("https://ex.com/a/b")


def test_poor_yield_turns_off_only_useless_sources():
    assert poor_yield(1198, 0, 20) is not None            # jak karpaty-ua bez przepisow
    assert poor_yield(1198, 3, 20) is None                 # sa przepisy - zostaje
    assert poor_yield(150, 0, 0) is None                   # za malo stron, zeby ocenic
    assert poor_yield(400, 0, 60) is None                  # 15% ciekawostek - oplaca sie
