"""Czyszczenie tekstu strony PO wycieciu znacznikow HTML (trafilatura / BeautifulSoup w page.py).

LLM i tlumacz dostaja sam tekst dokumentu - bez <div>, <select> itd. Tu znikaja jeszcze resztki "chaosu" strony,
ktore trafilatura czasem zostawia: spis tresci, "Czytaj tez", zachety do subskrypcji i komentarzy, udostepnianie,
reklamy, quizy, przyciski sklepu, linie powtorzone w tekscie i linie powtarzajace sie na wielu stronach tej samej
witryny (stopki, paski boczne). Usuwane sa tylko KROTKIE linie - dluzsze zdania z trescia zostaja zawsze.
"""
import hashlib
import re
from collections import Counter

SHORT = 90                 # dluzsze linie nigdy nie sa usuwane wzorcem
TOC_RUN = 3                # tyle krotkich linii z rzedu powtorzonych dalej jako naglowki = spis tresci
SITE_MIN_PAGES = 4         # boilerplate witryny: linia na co najmniej tylu stronach...
SITE_SHARE = 0.4           # ...i na takiej czesci porownywanych stron

TOC_HEAD = re.compile(
    r"^(spis tre[sś]ci|table of contents|contents|inhalt(sverzeichnis)?|зміст|содержание|sommaire|obsah|"
    r"[ií]ndice|turinys|tartalom(jegyz[eé]k)?|cuprins)\s*:?\s*(\[?(ukryj|poka[zż]|hide|show)\]?)?$", re.I)
NOISE = re.compile("|".join([
    # spis tresci
    r"^(spis tre[sś]ci|table of contents|contents|inhalt(sverzeichnis)?|зміст|содержание|sommaire|obsah|"
    r"[ií]ndice|turinys|tartalom(jegyz[eé]k)?|cuprins)\s*:?\s*(\[?(ukryj|poka[zż]|hide|show)\]?)?$",
    # "czytaj tez", powiazane wpisy
    r"^(czytaj|zobacz|przeczytaj)\s+(te[zż]|tak[zż]e|wi[eę]cej)|^(read|see)\s+(also|more)\b|^related\b|"
    r"^you (may|might) also like|^lesen sie (auch|mehr)|^siehe auch|^das k[oö]nnte|^voir aussi|^[àa] lire aussi|"
    r"^leggi anche|^lee tambi[eé]n|^p[rř]e[cč]t[eě]te si|^читайте (також|также|ще|еще)|^дивіться також|"
    r"^смотрите также|^схожі|^похожие|^podobne (wpisy|artyku)",
    # subskrypcja, komentarze, udostepnianie, drukowanie
    r"subscri|newsletter|abonnier|abonnez|iscriviti|suscr[ií]b|zapisz si[eę]|zasubskrybuj|підпис|подпис|"
    r"^(leave|post|add) a (comment|reply)|^(\d+\s+)?(comments?|kommentare?|komentarz[ey]?|коментар\w*|комментар\w*)$|"
    r"^(share|teilen|partager|condividi|compartir|udost[eę]pnij|поділит\w*|поделит\w*)\b|"
    r"^(pin it|tweet|print( recipe)?|drukuj( przepis)?|rezept drucken|imprimer|jump to recipe|"
    r"przejd[zź] do przepisu|zum rezept|rate this recipe|oce[nń] przepis|bewerten|"
    r"follow us|obserwuj nas|folge uns|suivez-nous)\b",
    # reklamy i ciasteczka
    r"^(advertisement|anzeige|werbung|publicit[eé]|pubblicit[aà]|publicidad|reklama|реклама)$|"
    r"(we|this site) uses? cookies|cookies? (policy|settings|consent|einstellungen)|u[zż]ywamy (plik[oó]w )?cookies?|"
    r"polityk\w* (prywatno[sś]ci|cookies)|privacy policy|datenschutz|файли cookie|akceptuj[eę]|accept all",
    # quizy
    r"\bquiz\b|^(next|previous) question|^(nast[eę]pne|poprzednie) pytanie|^n[aä]chste frage|"
    r"^наступне питання|^следующий вопрос",
    # sklep
    r"add to (cart|basket)|in den warenkorb|do koszyka|ajouter au panier|aggiungi al carrello|"
    r"a[nñ]adir al carrito|в кошик|в корзину|free shipping|kostenlose[rn]? versand|darmowa dostawa|"
    r"безкоштовна доставка|бесплатная доставка|money.back|zwrot pieni[eę]dzy",
    # linie bez tresci
    r"^[\W_]*$|^©|^\(c\)\s|all rights reserved|wszelkie prawa zastrze[zż]one|alle rechte vorbehalten",
]), re.I)


# linie z iloscia (250 ml, 2 łyżki, 1 cup...) nigdy nie sa boilerplate'em - to tresc przepisu
QTY = re.compile(r"\d+\s*(ml|l|g|kg|dag|łyż|lyz|szklan|cup|tbsp|tsp|oz|el|tl|г|мл|ст)\w*\b|"
                 r"\b(łyżk|szklank|szczypt|garś|ložk|cucchia|cuill|tasse|tassen)", re.I)


def _site_line(s: str) -> bool:
    """Czy linie wolno usunac jako boilerplate witryny: 3+ slowa, nie naglowek sekcji ("Skladniki:"), bez ilosci."""
    return len(s.split()) >= 3 and not s.rstrip().endswith(":") and not QTY.search(s)


def _key(line: str) -> str:
    return re.sub(r"\W+", " ", line.lower()).strip()


def line_hashes(text: str) -> set[str]:
    """Odciski krotkich linii strony (do wykrywania boilerplate'u witryny)."""
    return {hashlib.blake2b(_key(l).encode(), digest_size=8).hexdigest()
            for l in (text or "").splitlines() if 0 < len(l.strip()) <= SHORT and _key(l)}


def site_boilerplate(texts: list[str]) -> set[str]:
    """Odciski linii powtarzajacych sie na wielu stronach tej samej witryny (stopka, pasek boczny, menu)."""
    texts = [t for t in texts if t]
    if len(texts) < SITE_MIN_PAGES:
        return set()
    counts = Counter(h for t in texts for h in line_hashes(t))
    need = max(SITE_MIN_PAGES, int(SITE_SHARE * len(texts) + 0.999))
    return {h for h, c in counts.items() if c >= need}


def clean_text(text: str, boilerplate: set[str] | None = None) -> str:
    """Tekst strony bez resztek nawigacji, reklam, spisu tresci i powtorzen. Idempotentne."""
    lines = [l.rstrip() for l in (text or "").splitlines()]
    keys = [_key(l) for l in lines]
    later: Counter = Counter(k for k in keys if k)
    drop = [False] * len(lines)
    seen_long: set[str] = set()
    for i, (line, k) in enumerate(zip(lines, keys)):
        s = line.strip()
        if not s:
            continue
        short = len(s) <= SHORT
        if short and NOISE.search(s):
            drop[i] = True
        elif short and boilerplate and _site_line(s) and hashlib.blake2b(k.encode(), digest_size=8).hexdigest() in boilerplate:
            drop[i] = True
        elif len(s) > 40:
            if k in seen_long:                              # to samo zdanie drugi raz
                drop[i] = True
            seen_long.add(k)
    # spis tresci: >= TOC_RUN krotkich linii z rzedu (2 tuz pod "Spis tresci"), z ktorych kazda wraca dalej
    # w tekscie (jako naglowek)
    i = 0
    while i < len(lines):
        after_head = i > 0 and bool(TOC_HEAD.match(lines[i - 1].strip()))
        j = i
        while j < len(lines) and keys[j] and len(lines[j].strip()) <= 70 and later[keys[j]] >= 2 \
                and not QTY.search(lines[j]) and keys[j] in keys[j + 1:]:
            j += 1
        if j - i >= (2 if after_head else TOC_RUN):
            for x in range(i, j):
                drop[x] = True
            i = j
        else:
            i += 1
    out, blank = [], False
    for line, d in zip(lines, drop):
        if d:
            continue
        if not line.strip():
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip()
