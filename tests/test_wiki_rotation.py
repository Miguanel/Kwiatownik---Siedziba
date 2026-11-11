"""Rotacja Wikipedii (2026-11-11): kolejne badania rosliny czytaja kolejne, nieczytane wersje jezykowe."""
from datetime import datetime, timedelta

from app.knowledge.wiki import plan_wiki_langs, wiki_lang_of

TITLES = {k: f"T-{k}" for k in ("pl", "en", "de", "fr", "ru", "uk", "ja", "zh", "ceb", "nl", "ka", "xx", "simple")}
PRIO = ["pl", "en", "de", "uk", "ru", "fr"]
SUPPORTED = {"pl", "en", "de", "fr", "ru", "uk", "ja", "zh", "nl", "ka", "ceb"}


def sup(lang):
    return lang in SUPPORTED


def test_first_research_reads_priority_languages():
    plan = plan_wiki_langs(TITLES, {}, 3, PRIO, ["zh", "ja"], sup)
    assert plan.new == ["pl", "en", "de"]
    assert plan.langs == ["pl", "en", "de", "zh", "ja"]       # chinska / japonska poza limitem, jak dotad
    assert plan.available == 10                                 # bez simple/ceb (boty) i xx (brak tlumacza)


def test_next_research_takes_unread_countries_then_wider_order():
    now = datetime(2026, 11, 11)
    read = {x: now for x in ("pl", "en", "de", "zh", "ja")}
    plan = plan_wiki_langs(TITLES, read, 3, PRIO, ["zh", "ja"], sup)
    assert plan.new == ["uk", "ru", "fr"] and not plan.refresh
    read |= {x: now for x in ("uk", "ru", "fr")}
    plan = plan_wiki_langs(TITLES, read, 3, PRIO, ["zh", "ja"], sup)
    assert plan.new == ["nl", "ka"]                             # spoza KNOWLEDGE_LANGUAGES: WIKI_LANG_ORDER
    assert plan.read == 8 and plan.left == []


def test_all_read_refreshes_oldest_only_after_refresh_days():
    now = datetime(2026, 11, 11)
    read = {x: now - timedelta(days=i) for i, x in enumerate(["pl", "en", "de", "fr", "ru", "uk", "nl", "ka"])}
    read |= {"ja": now, "zh": now}
    fresh = plan_wiki_langs(TITLES, read, 2, PRIO, [], sup, refresh_before=now - timedelta(days=30))
    assert fresh.new == [] and fresh.refresh == []
    old = plan_wiki_langs(TITLES, read, 2, PRIO, [], sup, refresh_before=now - timedelta(days=5))
    assert old.refresh == ["ka", "nl"]                         # najdawniej czytane najpierw


def test_wiki_lang_of_url():
    assert wiki_lang_of("https://de.wikipedia.org/wiki/Schafgarbe") == "de"
    assert wiki_lang_of("https://zh.m.wikipedia.org/wiki/x") == "zh"
    assert wiki_lang_of("https://example.com/wikipedia.org/") is None
