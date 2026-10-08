"""Rozpoznawanie 'tej samej strony' - po domenie rejestrowanej (bez www., m., en. itd.).

Uzywa pelnej listy Public Suffix (tldextract, wbudowana kopia - bez internetu), dzieki czemu
drink.co.ua, herbarium.katowice.pl czy blog.blogspot.com sa poprawnie rozpoznawane jako osobne strony.
"""
from urllib.parse import urlparse

try:
    import tldextract
    _EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
except ImportError:  # pragma: no cover - awaryjnie wlasna lista koncowek
    _EXTRACT = None

# awaryjna lista wieloczlonowych koncowek (gdy brak tldextract)
MULTI_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "com.ua", "org.ua", "net.ua", "in.ua", "kiev.ua", "kyiv.ua",
    "lviv.ua", "od.ua", "com.pl", "net.pl", "org.pl", "info.pl", "waw.pl", "com.au", "net.au", "org.au", "co.nz",
    "com.br", "co.za", "co.jp", "com.tr", "com.mx", "com.ar", "co.in", "co.il", "com.cn", "com.es", "com.gr",
    "co.at", "or.at", "com.hr", "co.rs", "com.ro", "co.hu", "co.ua", "pp.ua", "dp.ua", "zp.ua", "kharkov.ua",
    "biz.pl", "edu.pl", "gov.pl", "art.pl", "tm.pl", "nom.pl", "media.pl", "sklep.pl", "shop.pl", "auto.pl",
    "katowice.pl", "krakow.pl", "wroclaw.pl", "poznan.pl", "gda.pl", "gdansk.pl", "lodz.pl", "szczecin.pl",
    "lublin.pl", "slask.pl", "warszawa.pl", "blogspot.com", "wordpress.com",
}
# serwisy, ktore nie sa "stronami z przepisami" (spolecznosci, sklepy, encyklopedie, wyszukiwarki)
BLOCKED = {
    "youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "reddit.com",
    "twitter.com", "x.com", "linkedin.com", "wikipedia.org", "wikihow.com", "amazon.com", "amazon.de", "ebay.com",
    "etsy.com", "allegro.pl", "aliexpress.com", "quora.com", "google.com", "bing.com", "yandex.ru", "yandex.ua",
    "duckduckgo.com", "vk.com", "ok.ru", "telegram.org", "t.me", "medium.com", "rozetka.com.ua", "prom.ua",
    "olx.ua", "olx.pl", "apple.com", "spotify.com", "tripadvisor.com", "booking.com",
}


def registrable_domain(url_or_host: str) -> str | None:
    host = urlparse(url_or_host).netloc if "//" in url_or_host else url_or_host
    host = host.lower().split("@")[-1].split(":")[0].strip(".")
    if not host or "." not in host:
        return None
    if _EXTRACT is not None:
        ext = _EXTRACT(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        # nieznana koncowka (np. testowe .example, siec lokalna) -> prosta regula ponizej
    parts = host.split(".")
    n = 3 if ".".join(parts[-2:]) in MULTI_SUFFIXES else 2
    return ".".join(parts[-n:])


def is_blocked(domain: str | None) -> bool:
    if not domain:
        return True
    base = domain.split(".")[0]
    return domain in BLOCKED or any(domain == b or domain.endswith("." + b) for b in BLOCKED) \
        or base in {"pinterest", "amazon", "ebay", "google", "facebook", "wikipedia"}
