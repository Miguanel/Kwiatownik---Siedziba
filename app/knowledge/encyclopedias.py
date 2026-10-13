"""Zagraniczne encyklopedie i bazy wiedzy o roslinach (poza Wikipedia).

Kazde zrodlo wie, w jakim jezyku szukac (nazwa gatunku z Wikidata: ja / zh / ko albo lacinska) i jak trafic na
strone rosliny:
  1. szablony adresow (bezposrednio, np. https://kotobank.jp/word/ドクダミ),
  2. strona wynikow wyszukiwania w samym serwisie + wzorzec linku do hasla,
  3. lokalna wyszukiwarka (SearXNG) z ograniczeniem "site:domena".
Strona musi naprawde dotyczyc tej rosliny (nazwa lacinska rodzaju albo nazwa w jezyku zrodla w tekscie).
robots.txt jest respektowany (Fetcher) - serwis, ktory nie zyczy sobie robotow, jest pomijany.

Zrodla wlacza / wylacza sie w panelu /plants/zrodla; stan: DATA_DIR/encyclopedias.json.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

from app.config import settings


@dataclass(frozen=True)
class Encyclopedia:
    key: str
    name: str
    country: str                       # kod kraju (flaga w panelu)
    lang: str                          # jezyk tekstu (do promptu LLM)
    name_lang: str                     # nazwa gatunku w tym jezyku: ja | zh | ko | la | en
    domain: str
    kind: str                          # encyklopedia | baza naukowa | zielnik | teksty dawne
    topics: tuple[str, ...]            # tematy luk, w ktorych zrodlo jest najmocniejsze (gaps.TOPICS)
    urls: tuple[str, ...] = ()         # szablony: {name} (nazwa w name_lang), {latin}, {latin_plus}
    search_url: str | None = None      # strona wynikow w serwisie
    link_re: str | None = None         # wzorzec linku do hasla na stronie wynikow
    query_prefix: str = ""             # np. "本草綱目" przed nazwa przy wyszukiwaniu site:
    min_len: int = 400
    about: str = ""


ENCYCLOPEDIAS: tuple[Encyclopedia, ...] = (
    # ---------------- Japonia
    Encyclopedia("kotobank", "Kotobank (Nipponica, Heibonsha, Britannica JP, Mypedia)", "jp", "ja", "ja", "kotobank.jp",
                 "encyklopedia", ("wschod", "lecznicze", "folklor", "barwienie", "kuchnia"),
                 urls=("https://kotobank.jp/word/{name}",),
                 search_url="https://kotobank.jp/search?q={name}", link_re=r'href="(/word/[^"#?]+)"', min_len=250,
                 about="hasla z kilku japonskich encyklopedii naraz: Nihon Daihyakka Zensho (Nipponica), Sekai "
                       "Daihyakka Jiten (Heibonsha), Britannica Kokusai, Mypedia - historia, uzycie, nazwy"),
    Encyclopedia("mpdb", "NIBIOHN - baza roslin leczniczych (Japonia)", "jp", "ja", "la", "mpdb.nibiohn.go.jp",
                 "baza naukowa", ("lecznicze", "sklad", "wschod", "toksycznosc"), min_len=250,
                 about="Narodowy Instytut Innowacji Biomedycznych: surowce, kampo, sklad, uprawa"),
    Encyclopedia("kumamoto", "Uniwersytet Kumamoto - ogrod roslin leczniczych", "jp", "ja", "ja",
                 "pharm.kumamoto-u.ac.jp", "zielnik", ("lecznicze", "wschod", "sklad"), min_len=250,
                 about="baza ogrodu roslin leczniczych wydzialu farmacji: surowiec, dzialanie, kampo"),
    Encyclopedia("weblio", "Weblio (slowniki i encyklopedie JP)", "jp", "ja", "ja", "weblio.jp", "encyklopedia",
                 ("folklor", "kuchnia", "uprawa"), urls=("https://www.weblio.jp/content/{name}",), min_len=300,
                 about="zbiorcze hasla z japonskich slownikow i encyklopedii"),
    # ---------------- Chiny
    Encyclopedia("baidu", "Baidu Baike (百度百科)", "cn", "zh", "zh", "baike.baidu.com", "encyklopedia",
                 ("wschod", "lecznicze", "folklor", "kuchnia", "uprawa"),
                 urls=("https://baike.baidu.com/item/{name}",), min_len=300,
                 about="najwieksza chinska encyklopedia: TCM (性味归经, 功效), historia, uprawa"),
    Encyclopedia("ahospital", "A+ Yixue Baike (A+医学百科)", "cn", "zh", "zh", "a-hospital.com", "encyklopedia",
                 ("wschod", "lecznicze", "sklad", "toksycznosc"), urls=("https://www.a-hospital.com/w/{name}",),
                 min_len=300, about="chinska encyklopedia medyczna: hasla surowcow TCM, przepisy, przeciwwskazania"),
    Encyclopedia("iplant", "Flora Chin / iPlant (中国植物志)", "cn", "zh", "la", "iplant.cn", "baza naukowa",
                 ("rozpoznawanie", "ekologia", "lecznicze"), urls=("https://www.iplant.cn/info/{latin}",),
                 min_len=250, about="Flora Reipublicae Popularis Sinicae: opis, wystepowanie, zastosowania"),
    Encyclopedia("zysj", "Zhongyi Shijia (中医世家) - Zhongyao Dacidian", "cn", "zh", "zh", "zysj.com.cn",
                 "teksty dawne", ("wschod", "lecznicze"), min_len=300,
                 about="klasyczne kompendia TCM, m.in. 中药大辞典 i 中华本草"),
    Encyclopedia("bencao", "Wikisource: Bencao gangmu (本草綱目)", "cn", "zh", "zh", "zh.wikisource.org",
                 "teksty dawne", ("wschod", "folklor", "lecznicze"), query_prefix="本草綱目", min_len=300,
                 about="Li Shizhen, 1578 - klasyczny chinski zielnik; dawne nazwy, uzycie, wierzenia"),
    # ---------------- Korea
    Encyclopedia("encykorea", "Encyklopedia Kultury Koreanskiej (한국민족문화대백과사전)", "kr", "ko", "ko",
                 "encykorea.aks.ac.kr", "encyklopedia", ("folklor", "lecznicze", "kuchnia"), min_len=250,
                 about="Akademia Studiow Koreanskich: zwyczaje, medycyna ludowa, uzycie roslin w kulturze"),
    Encyclopedia("naver", "Naver Terms (네이버 지식백과)", "kr", "ko", "ko", "terms.naver.com", "encyklopedia",
                 ("lecznicze", "wschod", "kuchnia"), min_len=250,
                 about="zbiorcze hasla z koreanskich encyklopedii i slownikow medycyny koreanskiej (한의학)"),
    Encyclopedia("nature_kr", "Krajowy system wiedzy o gatunkach (국가생물종지식정보)", "kr", "ko", "ko",
                 "nature.go.kr", "baza naukowa", ("rozpoznawanie", "lecznicze", "ekologia"), min_len=250,
                 about="koreanska baza gatunkow: opis, wystepowanie, uzycie"),
    # ---------------- bazy naukowe i zielniki (USA / UK)
    Encyclopedia("pfaf", "Plants For A Future (PFAF)", "gb", "en", "la", "pfaf.org", "baza naukowa",
                 ("lecznicze", "kuchnia", "barwienie", "toksycznosc", "uprawa"),
                 urls=("https://pfaf.org/user/Plant.aspx?LatinName={latin_plus}",), min_len=500,
                 about="ok. 7000 roslin uzytkowych: zastosowania lecznicze, jadalne, barwierskie, ostrzezenia"),
    Encyclopedia("naeb", "Native American Ethnobotany (BRIT, D. Moerman)", "us", "en", "la", "naeb.brit.org",
                 "baza naukowa", ("lecznicze", "folklor", "barwienie", "kuchnia"),
                 urls=("http://naeb.brit.org/uses/search/?string={latin_plus}",), min_len=300,
                 about="44 000 zastosowan roslin przez ludy rdzenne Ameryki Polnocnej, z publikacjami zrodlowymi"),
    Encyclopedia("duke", "Dr. Duke's Phytochemical and Ethnobotanical DB (USDA)", "us", "en", "la",
                 "phytochem.nal.usda.gov", "baza naukowa", ("sklad", "lecznicze", "toksycznosc"), min_len=300,
                 about="USDA: substancje czynne w czesciach rosliny i ich dzialanie, etnobotanika"),
    Encyclopedia("bhl", "Biodiversity Heritage Library (m.in. biblioteki botaniczne Harvardu)", "us", "en", "la",
                 "biodiversitylibrary.org", "teksty dawne", ("folklor", "lecznicze", "rozpoznawanie"),
                 min_len=500, about="zdigitalizowane dawne zielniki i czasopisma botaniczne (Harvard, Kew, Smithsonian)"),
    Encyclopedia("henriette", "Henriette's Herbal (King's Dispensatory, Felter, Grieve)", "us", "en", "la",
                 "henriettes-herb.com", "teksty dawne", ("lecznicze", "folklor", "toksycznosc"), min_len=500,
                 about="pelne teksty dawnych farmakopei i zielnikow amerykanskich i brytyjskich"),
    Encyclopedia("grieve", "A Modern Herbal - M. Grieve (botanical.com)", "gb", "en", "la", "botanical.com",
                 "teksty dawne", ("lecznicze", "folklor", "kuchnia"), min_len=500,
                 about="klasyczny brytyjski zielnik (1931): historia, wierzenia, zastosowanie"),
)
BY_KEY = {e.key: e for e in ENCYCLOPEDIAS}
FLAGS = {"jp": "🇯🇵", "cn": "🇨🇳", "kr": "🇰🇷", "us": "🇺🇸", "gb": "🇬🇧"}


# ------------------------------------------------------------------ stan (wlaczone / wylaczone)
def _state_path() -> Path:
    return Path(settings.data_dir) / "encyclopedias.json"


def enabled_map() -> dict[str, bool]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {e.key: bool(data.get(e.key, True)) for e in ENCYCLOPEDIAS}


def set_enabled(key: str, on: bool) -> None:
    data = enabled_map()
    if key in data:
        data[key] = on
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ wybor zrodel dla rosliny
def plant_name(enc: Encyclopedia, names: dict[str, str]) -> str | None:
    return names.get(enc.name_lang) if enc.name_lang != "la" else names.get("la")


def choose(names: dict[str, str], topics: list[str], used_domains: set[str], limit: int) -> list[Encyclopedia]:
    """Zrodla na ten przebieg: wlaczone, z nazwa gatunku w potrzebnym jezyku; najpierw te, ktorych jeszcze nie
    uzyto dla tej rosliny (rotacja - kolejne przebiegi siegaja po kolejne bazy), potem pasujace do luk."""
    on = enabled_map()
    cands = [e for e in ENCYCLOPEDIAS if on.get(e.key) and plant_name(e, names)]

    def score(e: Encyclopedia) -> tuple:
        fresh = not any(e.domain in d for d in used_domains)
        hits = sum(1 for t in topics if t in e.topics)
        first = min((topics.index(t) for t in e.topics if t in topics), default=99)
        return (not fresh, -hits, first, ENCYCLOPEDIAS.index(e))

    picked: list[Encyclopedia] = []
    for e in sorted(cands, key=score):
        if len(picked) >= limit:
            break
        if any(p.lang == e.lang and p.kind == e.kind for p in picked) and len(cands) > limit:
            continue                                   # roznorodnosc: nie dwa takie same zrodla z jednego kraju
        picked.append(e)
    for e in sorted(cands, key=score):                 # uzupelnienie, gdy roznorodnosc za bardzo ograniczyla
        if len(picked) >= limit:
            break
        if e not in picked:
            picked.append(e)
    return picked


def urls_for(enc: Encyclopedia, names: dict[str, str]) -> list[str]:
    name = plant_name(enc, names) or ""
    latin = names.get("la") or ""
    return [t.format(name=quote(name), latin=quote(latin), latin_plus=quote(latin).replace("%20", "+"))
            for t in enc.urls]


def about_plant(text: str, enc: Encyclopedia, names: dict[str, str]) -> bool:
    """Strona naprawde o tej roslinie: rodzaj lacinski albo nazwa w jezyku zrodla (dla CJK bez granic slow)."""
    low = (text or "").lower()
    latin = names.get("la") or ""
    if latin and latin.split()[0].lower() in low:
        return True
    for lang in {enc.lang, enc.name_lang}:
        local = (names.get(lang) or "").lower()
        if local and local in low:
            return True
    return False


# ------------------------------------------------------------------ pobieranie
@dataclass
class Found:
    enc: Encyclopedia
    url: str | None = None
    text: str = ""
    how: str = ""                      # szablon | wyszukiwarka serwisu | site:
    tried: list[str] = field(default_factory=list)
    reason: str = ""


async def _page(fetcher, url: str, enc: Encyclopedia, names: dict[str, str], found: Found) -> bool:
    from app.scrapers.page import parse_page
    found.tried.append(url)
    res = await fetcher.get(url)
    if not res.ok:
        found.reason = f"{url}: {res.error or res.status}"
        return False
    page = await asyncio.to_thread(parse_page, res.text, res.url)
    if len(page.text) < enc.min_len:
        found.reason = f"{url}: za malo tresci ({len(page.text)} znakow)"
        return False
    if not about_plant(page.text, enc, names):
        found.reason = f"{url}: strona nie o tej roslinie"
        return False
    found.url, found.text = res.url, page.text
    return True


async def fetch(enc: Encyclopedia, names: dict[str, str], fetcher, backends=None, skip_urls: set[str] | None = None,
                log=None) -> Found:
    """Szuka strony rosliny w jednym zrodle. Nigdy nie rzuca wyjatku - powod porazki w Found.reason."""
    found = Found(enc)
    skip = skip_urls or set()
    try:
        for url in urls_for(enc, names):                               # 1. szablony adresow
            if url not in skip and await _page(fetcher, url, enc, names, found):
                found.how = "adres hasla"
                return found
        name = plant_name(enc, names) or names.get("la") or ""
        if enc.search_url and enc.link_re:                             # 2. wyszukiwarka serwisu
            surl = enc.search_url.format(name=quote(name))
            res = await fetcher.get(surl)
            found.tried.append(surl)
            if res.ok:
                for href in list(dict.fromkeys(re.findall(enc.link_re, res.text)))[:3]:
                    url = urljoin(surl, href)
                    if url not in skip and await _page(fetcher, url, enc, names, found):
                        found.how = "wyszukiwarka serwisu"
                        return found
        if backends:                                                   # 3. SearXNG z site:
            from app.discovery.search import search_all
            q = f"{enc.query_prefix} {name} site:{enc.domain}".strip()
            results = await search_all(backends, q, enc.lang, "", log_fn=log)
            for r in results[:5]:
                if enc.domain not in urlparse(r.url).netloc + urlparse(r.url).path or r.url in skip:
                    continue
                if await _page(fetcher, r.url, enc, names, found):
                    found.how = "wyszukiwarka (site:)"
                    return found
            if not results and not found.reason:
                found.reason = f"wyszukiwarka nie znalazla nic dla '{q}'"
    except Exception as exc:  # noqa: BLE001 - jedno zrodlo nie zatrzymuje zbierania wiedzy
        found.reason = f"{type(exc).__name__}: {exc}"
    found.url, found.text = None, ""
    if not found.reason:
        found.reason = "nie znaleziono hasla"
    return found
