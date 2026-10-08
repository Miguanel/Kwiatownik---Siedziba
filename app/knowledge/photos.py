"""Zdjecia roslin z Wikipedii / Wikimedia Commons (oficjalne API, bez pobierania plikow - tylko adresy).

Skad:
1. Wikidata P18 (glowne zdjecie gatunku) -> "Pokrój",
2. zdjecia z artykulow Wikipedii o gatunku (kilka jezykow),
3. kategoria gatunku na Commons (Wikidata P373).
Kontrola: tylko JPG/PNG/WEBP min. 400 px, wolne licencje (CC0, CC BY, CC BY-SA, domena publiczna), bez map
zasiegu, herbarium, ikon i znaczkow; zdjecie z artykulu musi dotyczyc TEGO gatunku (nazwa lacinska w nazwie
pliku, opisie albo kategoriach Commons) - artykul o barszczu pokazuje tez barszcz Sosnowskiego.
Podpis (Pokrój, Kwiaty, Liście, Owoce...) z nazwy pliku / opisu / kategorii w wielu jezykach.
Kazde zdjecie ma autora, licencje i link do strony pliku (wymog licencji CC BY / BY-SA).
"""
import html as html_lib
import re
from dataclasses import asdict, dataclass
from urllib.parse import unquote

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 640
MAX_PHOTOS = 5
MIN_WIDTH = 400
# miniatury: Wikimedia podaje je teraz z thumb.wikimedia.org (wczesniej upload.wikimedia.org) - oba hosty
UPLOAD_RE = re.compile(r"^https://(?:upload|thumb)\.wikimedia\.org/[A-Za-z0-9/_%.,()~+\-]+$")
FILEPAGE_RE = re.compile(r"^https://commons\.wikimedia\.org/wiki/File:[^\s<>\"']+$")

# (podpis, slowa kluczowe w nazwie pliku / opisie / kategoriach - rozne jezyki, bez ogonkow)
LABELS: list[tuple[str, tuple[str, ...]]] = [
    ("Rycina", ("kohler", "köhler", "illustration", "drawing", "flora von deutschland", "thome", "sturm",
                "botanical plate", "tafel", "rycina", "ilustrac", "lindman", "bilder ur nordens")),
    ("Kwiaty", ("flower", "blossom", "bloom", "inflorescen", "blute", "blüte", "bluete", "fleur", "flor", "fiore",
                "kwiat", "цвет", "квіт", "kvet", "květ", "virag", "virág", "žied", "zied", "umbel", "baldach")),
    ("Liście", ("leaf", "leaves", "foliage", "blatt", "blätter", "blaetter", "feuille", "hoja", "foglia", "lisc",
                "liść", "liscie", "liście", "лист", "levél", "lapas", "rosette", "rozet", "needle", "igł")),
    ("Owoce", ("fruit", "berr", "frucht", "fruchte", "früchte", "beere", "fruto", "frutto", "owoc", "jagod", "плод",
               "ягод", "plod", "termés", "cone", "zapfen", "szyszk", "acorn", "eichel", "żołęd", "zoled")),
    ("Nasiona", ("seed", "samen", "graine", "semill", "seme ", "nasion", "насін", "семен", "semen")),
    ("Korzeń", ("root", "wurzel", "racine", "raiz", "raíz", "radice", "korzen", "korzeń", "корен", "корін", "rhizom",
                "kłącz", "klacz", "tuber", "bulw")),
    ("Kora", ("bark", "rinde", "borke", "écorce", "ecorce", "corteza", "corteccia", "kora ", "кора", "trunk", "stamm")),
    ("Pąki", ("bud", "knospe", "bourgeon", "pąk", "pak ", "брунь", "почк")),
    ("Pokrój", ("habit", "habitus", "whole plant", "pokroj", "pokrój", "plant ", " tree", "baum", "shrub", "strauch",
                "bush", "arbre", "árbol", "albero", "rostlina", "растен", "рослин", "field", "meadow", "wiese")),
]
ORDER = ["Pokrój", "Kwiaty", "Liście", "Owoce", "Nasiona", "Korzeń", "Kora", "Pąki", "Rycina"]
# poczatki slow (regex, bez ogonkow i z nimi): mapy zasiegu, herbarium, ikony, znaczki, mikroskopia, produkty
EXCLUDE = re.compile(r"(?<![a-zа-яё])(map|maps|distribution|range|verbreitung|areal|ареал|карта|mapa|rozmieszcz|logo|"
                     r"icon|herbar|specimen|stamp|briefmarke|znaczek|coin|sem|micrograph|pollen|diagram|chart|"
                     r"chemical|structure|formula|wappen|coat of arms|flag|label|packag|product|tablet|capsule|"
                     r"kapsu|drug|apothe|market|dish|food|recipe)(?![a-zа-яё])", re.I)
FREE_LICENSE = re.compile(r"\bcc0\b|\bcc[- ]by\b|cc[- ]by[- ]sa|public domain|\bpd\b|pd-|no restrictions", re.I)


@dataclass
class Photo:
    podpis: str
    url: str               # miniatura (THUMB_WIDTH px) na upload.wikimedia.org
    plik: str              # strona pliku na Commons (autor, licencja, oryginal)
    autor: str
    licencja: str
    licencja_url: str | None
    zrodlo: str = "Wikimedia Commons"

    def as_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"[_\-]+", " ", unquote(s or "")).lower()


def _plain(value: str | None, limit: int = 160) -> str:
    """HTML z extmetadata (np. <a href=...>Autor</a>) -> czysty tekst."""
    t = re.sub(r"<[^>]+>", " ", html_lib.unescape(value or ""))
    return re.sub(r"\s+", " ", t).strip()[:limit]


def file_title(name: str) -> str | None:
    """'Plik:Abc.jpg' / 'File:Abc.jpg' / 'Файл:Abc.jpg' -> 'File:Abc.jpg' (tylko zdjecia rastrowe)."""
    base = name.split(":", 1)[1] if ":" in name else name
    base = base.strip().replace("_", " ")
    if not re.search(r"\.(jpe?g|png|webp|tiff?)$", base, re.I):
        return None
    return f"File:{base[:1].upper()}{base[1:]}"


def file_from_upload_url(url: str) -> str | None:
    """Adres miniatury / pliku z upload.wikimedia.org -> 'File:Nazwa.jpg'."""
    m = re.match(r"^https?://(?:upload|thumb)\.wikimedia\.org/wikipedia/commons/(?:thumb/)?[0-9a-f]/[0-9a-f]{2}/([^/]+)",
                 url or "")
    return file_title(unquote(m.group(1))) if m else None


def label_for(text: str, default: str = "Pokrój") -> str:
    t = f" {_norm(text)} "
    for label, words in LABELS:
        if any(w in t for w in words):
            return label
    return default


def excluded(text: str) -> bool:
    return bool(EXCLUDE.search(_norm(text)))


def about_taxon(taxon: str, *texts: str) -> bool:
    """Czy plik dotyczy tego gatunku: pelna nazwa lacinska (albo skrot 'H. sphondylium')."""
    if not taxon or len(taxon.split()) < 2:
        return False
    genus, species = taxon.lower().split()[:2]
    t = " ".join(_norm(x) for x in texts)
    return f"{genus} {species}" in t or f"{genus[0]}. {species}" in t or f"{genus[0]} {species}" in t


_BINOMIAL = re.compile(r"^(?:file:)?\s*([a-z][a-z]+) ([a-z]{3,})\b")


def other_taxon(title: str, cats: str, taxon: str) -> bool:
    """Plik glownie o INNYM gatunku, choc jest w kategorii naszego (np. "Echium vulgare (…).jpg" albo
    "Philopedon plagiatum sur Plantago…" - zuk na babce): nazwa pliku zaczyna sie od innej nazwy lacinskiej,
    a plik ma kategorie z ta nazwa."""
    m = _BINOMIAL.match(_norm(title))
    if not m or not taxon:
        return False
    name = f"{m.group(1)} {m.group(2)}"
    if name == " ".join(taxon.lower().split()[:2]):
        return False
    return name in _norm(cats)


def photo_from_info(title: str, info: dict, label: str) -> Photo | None:
    """imageinfo z Commons -> Photo albo None (zla licencja, za male, zly format)."""
    ii = (info.get("imageinfo") or [{}])[0]
    meta = {k: (v or {}).get("value") for k, v in (ii.get("extmetadata") or {}).items()}
    if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp", "image/tiff"):
        return None
    if (ii.get("width") or 0) < MIN_WIDTH:
        return None
    lic = _plain(meta.get("LicenseShortName"), 60)
    if not lic or not FREE_LICENSE.search(lic):
        return None
    # Commons dopisuje do miniatur parametry ?utm_source=...&utm_campaign=... - adres bez nich
    thumb = (ii.get("thumburl") or ii.get("url") or "").split("?", 1)[0]
    page = (ii.get("descriptionurl") or "").split("?", 1)[0]
    if not UPLOAD_RE.match(thumb) or not FILEPAGE_RE.match(page):
        return None
    lic_url = meta.get("LicenseUrl") or None
    if lic_url and not re.match(r"^https?://[^\s<>\"']+$", lic_url):
        lic_url = None
    author = _plain(meta.get("Artist"), 120) or _plain(meta.get("Credit"), 120) or "nieznany autor"
    return Photo(label, thumb, page, author, lic, lic_url)


async def _imageinfo(wiki, titles: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 40):
        data = await wiki._get(COMMONS_API, {
            "action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata", "iiurlwidth": THUMB_WIDTH,
            "iiextmetadatafilter": "Artist|Credit|LicenseShortName|LicenseUrl|ImageDescription|ObjectName|Categories"})
        q = data.get("query") or {}
        norm = {n["to"]: n["from"] for n in q.get("normalized") or []}
        for p in (q.get("pages") or {}).values():
            if "missing" in p or "imageinfo" not in p:
                continue
            out[norm.get(p["title"], p["title"])] = p
            out[p["title"]] = p
    return out


async def article_files(wiki, lang: str, title: str) -> list[str]:
    data = await wiki._get(f"https://{lang}.wikipedia.org/w/api.php", {
        "action": "query", "prop": "images", "titles": title, "imlimit": 60, "redirects": 1})
    names = []
    for p in (data.get("query", {}).get("pages") or {}).values():
        names += [i["title"] for i in p.get("images") or []]
    return [t for t in (file_title(n) for n in names) if t]


async def category_files(wiki, category: str, limit: int = 40) -> list[str]:
    data = await wiki._get(COMMONS_API, {"action": "query", "list": "categorymembers", "cmtype": "file",
                                         "cmtitle": f"Category:{category}", "cmlimit": limit})
    return [t for t in (file_title(m["title"]) for m in data.get("query", {}).get("categorymembers") or []) if t]


async def find_photos(wiki, taxon: str | None, main_image: str | None, commons_category: str | None,
                      titles: dict[str, str], langs: list[str], log=None, max_photos: int = MAX_PHOTOS) -> list[Photo]:
    """Najlepsze zdjecia gatunku: po jednym na podpis (Pokrój, Kwiaty, Liście, Owoce...), max `max_photos`."""
    say = log or (lambda m: None)
    trusted: set[str] = set()                 # pliki na pewno o tym gatunku (P18, kategoria gatunku)
    cands: list[str] = []
    main = file_title(main_image) if main_image else None
    if main:
        cands.append(main)
        trusted.add(main)
    for lang in [x for x in langs if x in titles][:4]:
        try:
            cands += await article_files(wiki, lang, titles[lang])
        except Exception as exc:                      # jeden jezyk niedostepny - reszta dalej
            say(f"  zdjecia ({lang}): {str(exc)[:80]}")
    if commons_category:
        try:
            cat = await category_files(wiki, commons_category)
            cands += cat
            trusted.update(cat)
        except Exception as exc:
            say(f"  zdjecia (Commons): {str(exc)[:80]}")
    seen, uniq = set(), []
    for c in cands:
        if c not in seen and not excluded(c):
            seen.add(c)
            uniq.append(c)
    if not uniq:
        return []
    infos = await _imageinfo(wiki, uniq[:120])
    chosen: dict[str, Photo] = {}
    for t in uniq:
        info = infos.get(t)
        if not info:
            continue
        ii = (info.get("imageinfo") or [{}])[0]
        meta = {k: (v or {}).get("value") or "" for k, v in (ii.get("extmetadata") or {}).items()}
        desc = " ".join([_plain(meta.get("ImageDescription"), 400), _plain(meta.get("ObjectName"), 200)])
        cats = str(meta.get("Categories") or "")
        if excluded(f"{desc} {cats}"):
            continue
        if other_taxon(t, cats, taxon or ""):
            continue                                  # inny gatunek (roslina, owad) sfotografowany z naszym
        if t not in trusted and not about_taxon(taxon or "", t, desc, cats):
            continue                                  # zdjecie innego gatunku z tego samego artykulu
        label = "Pokrój" if t == main else label_for(f"{t} {desc} {cats}")
        if label in chosen:
            continue
        photo = photo_from_info(t, info, label)
        if photo:
            chosen[label] = photo
        if len(chosen) >= max_photos + 3:
            break
    ordered = [chosen[k] for k in ORDER if k in chosen]
    return ordered[:max_photos]


async def credits_for_urls(wiki, urls: dict[str, str]) -> list[dict]:
    """Autor i licencja dla zdjec wpisanych recznie (adresy z upload.wikimedia.org) - bez zmiany adresow."""
    files = {k: file_from_upload_url(v) for k, v in urls.items() if isinstance(v, str)}
    files = {k: f for k, f in files.items() if f}
    if not files:
        return []
    infos = await _imageinfo(wiki, sorted(set(files.values())))
    out = []
    for caption, f in files.items():
        info = infos.get(f)
        ii = (info or {}).get("imageinfo") or [{}]
        meta = {k: (v or {}).get("value") for k, v in (ii[0].get("extmetadata") or {}).items()}
        page = ii[0].get("descriptionurl") or ""
        if not info or not FILEPAGE_RE.match(page):
            continue
        lic_url = meta.get("LicenseUrl") if re.match(r"^https?://[^\s<>\"']+$", meta.get("LicenseUrl") or "") else None
        out.append({"podpis": caption[:80], "url": urls[caption], "plik": page,
                    "autor": _plain(meta.get("Artist"), 120) or "nieznany autor",
                    "licencja": _plain(meta.get("LicenseShortName"), 60) or "?", "licencja_url": lic_url,
                    "zrodlo": "Wikimedia Commons", "reczne": True})
    return out
