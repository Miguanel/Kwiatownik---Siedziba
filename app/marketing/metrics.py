"""Metryki posta liczone kodem (bez LLM) - powtarzalne, porownywalne miedzy postami.

Czytelnosc: indeks FOG-PL (jak w Jasnopisie: 0.4 * (slowa/zdania + 100 * trudne/slowa), trudne = 4+ sylaby),
mniej wiecej "ile lat nauki potrzeba". Sylaby = grupy samoglosek (i+samogloska = jedna sylaba, jak w polszczyznie)."""
import re
from collections import Counter

WORD = re.compile(r"[A-Za-zÀ-žĄĆĘŁŃÓŚŹŻąćęłńóśźż]+(?:['-][A-Za-zÀ-žąćęłńóśźż]+)*")
VOWELS = re.compile(r"[aąeęioóuyAĄEĘIOÓUY]+")
SENT = re.compile(r"[^.!?…]+[.!?…]*")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
NUM = re.compile(r"\d+(?:[.,]\d+)?")
HOOK_CHARS = 125   # tyle mniej wiecej widac na telefonie, zanim FB utnie tekst ("...Wiecej")

GWARA_WORDS = {
    "ino", "juzci", "ady", "downiej", "kiela", "wte", "tyz", "cheba", "kaj", "jo", "bede", "nase", "nasy", "wos",
    "godać", "godam", "godom", "godoł", "godała", "godajom", "dziołcha", "dziołchy", "chłopok", "galante", "galanty",
    "se", "łońskiego", "łoński", "łońskim", "prowdzie", "prowda", "piykny", "piekny", "cłek", "casem", "scęście",
    "coby", "zawdy", "teroz", "tera", "dziyń", "mo", "momy", "som", "bedom", "wiyncej", "jesce", "pódźcie",
    "pockojcie", "zaglądojcie", "pamiętojcie", "chałupa", "chałupie", "zielsko", "boci", "kany",
    # slaska godka (lekka)
    "terozki", "teroz", "tukej", "fest", "gryfnie", "gryfny", "gryfno", "gryfnych", "dyć", "bajtel", "bajtle",
    "bajtli", "starzik", "starzika", "oma", "omy", "omie", "omom", "opa", "kożdy", "kożdo", "kożdej", "kożdym",
    "wiela", "ciepnąć", "ciepnij", "ciepie", "wyciepnąć", "wyciepuje", "szkryfnąć", "szkryfać", "porzondek",
    "nojpierw", "nojpiyrw", "nojlepszy", "nojlepsze", "ôd", "ôbejrzeć", "łonki", "łonkach", "fajrant", "żech",
    "zbiyrał", "zbiyrałech", "pierwyj", "przaja", "przaje", "maszina", "dejcie",
    "pozór", "godka", "wejzdrzeć", "wejrzijcie", "niy",
}
GWARA_PHRASES = ("panie dzieju", "po prowdzie", "co by nie godać", "jo to godom", "dejcie pozór",
                 "co jo wom powiem")
MINING = re.compile(r"\b((?:na|z|ze|do|po) grub\w*|sztajger\w*|wōngl\w*|wongl\w*|wongiel|wōngiel|fedr\w*|kopalni\w*|kopalń|"
                    r"górnik\w*|gōrnik\w*|hajer\w*|szychta\w*|szychcie)\b", re.I)
GWARA_PATTERNS = (re.compile(r"^ło[a-ząćęłńóśźż]{2,}$"), re.compile(r"[a-ząćęłńóśźż]{2,}ołem$"),
                  re.compile(r"[a-ząćęłńóśźż]{2,}ojcie$"))
NOT_GWARA = re.compile(r"^(łodyg|łoś|łono|łoż|łowi|łopat|łotr|łokie|łoskot)")
CTA_SITE = re.compile(r"zajrz|zaglą|kwiatownik|przepis[a-z]* w|sprawdź|sprawdz", re.I)
CTA_TALK = re.compile(r"napisz|piszcie|dajcie znać|dajcie znac|komentarz|podzielcie|powiedzcie|a wy|a u was|"
                      r"pochwalcie|udostępn", re.I)
STOP = {"się", "nie", "jak", "tak", "ale", "bo", "to", "co", "na", "w", "i", "a", "z", "że", "ze", "do", "po", "od",
        "jest", "był", "było", "tylko", "jeszcze", "albo", "czy", "już", "też", "tyz", "ino", "jo", "mi", "go", "ich",
        "ten", "ta", "te", "tej", "tego", "tym", "dla", "przy", "pod", "nad", "bez", "jako", "kiedy", "gdy", "który",
        "która", "które", "jego", "jej", "nam", "was", "wos", "wam", "nas", "mnie", "sobie", "se", "tam", "tu", "tera"}


def syllables(word: str) -> int:
    return max(1, len(VOWELS.findall(word)))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.findall(text.replace("\n", " ")) if WORD.search(s)]


def _gwara_hits(words: list[str], low_text: str) -> int:
    hits = sum(1 for w in words if w in GWARA_WORDS
               or (not NOT_GWARA.match(w) and any(p.match(w) for p in GWARA_PATTERNS)))
    return hits + sum(low_text.count(p) for p in GWARA_PHRASES)


def fog_label(fog: float) -> str:
    if fog <= 7:
        return "bardzo łatwy (szkoła podstawowa)"
    if fog <= 9:
        return "łatwy (gimnazjum / kl. 7-8)"
    if fog <= 12:
        return "średni (liceum)"
    if fog <= 15:
        return "trudny (studia)"
    return "bardzo trudny"


def compute(body: str, text: str | None = None, source_data: str | None = None,
            names: list[str] | None = None, min_words: int = 120, max_words: int = 200,
            max_emoji: int = 3) -> dict:
    """Metryki tresci + lista problemow (kod, opis) + wynik kodu 0-100 i podwyniki."""
    body = body or ""
    full = text or body
    words = [w.lower() for w in WORD.findall(body)]
    n = len(words) or 1
    sents = sentences(body)
    ns = len(sents) or 1
    sent_len = [len(WORD.findall(s)) for s in sents] or [0]
    hard = sum(1 for w in words if syllables(w) >= 4)
    fog = round(0.4 * (n / ns + 100 * hard / n), 1)
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    low = body.lower()
    gwara = _gwara_hits(words, low)
    content = [w for w in words if w not in STOP and len(w) > 3]
    counts = Counter(content)
    repeats = [(w, c) for w, c in counts.most_common(8) if c >= 3]
    first = sents[0] if sents else ""
    tail = " ".join(sents[-2:]) if sents else ""
    questions = body.count("?")
    nums_post = set(NUM.findall(body))
    nums_src = set(NUM.findall(source_data or ""))
    unknown_nums = sorted(x for x in nums_post if source_data is not None and x not in nums_src)
    name_hits = {nm: low.count(nm.lower().split()[0][:5]) for nm in (names or []) if nm}

    m = {
        "slowa": len(words), "znaki_tresci": len(body), "znaki_calosc": len(full), "zdania": len(sents),
        "akapity": len(paras), "srednia_dl_zdania": round(sum(sent_len) / ns, 1), "najdluzsze_zdanie": max(sent_len),
        "najdluzszy_akapit_znaki": max((len(p) for p in paras), default=0),
        "fog": fog, "fog_opis": fog_label(fog), "trudne_slowa_proc": round(100 * hard / n, 1),
        "sylaby_na_slowo": round(sum(syllables(w) for w in words) / n, 2),
        "gwara_proc": round(100 * gwara / n, 1), "ttr": round(len(set(words)) / n, 2),
        "powtorzenia": repeats, "pytania": questions, "wykrzykniki": body.count("!"),
        "emoji": len(EMOJI.findall(body)), "hashtagi": len(re.findall(r"#\w+", full)),
        "link": "http" in full, "haczyk": first[:200], "haczyk_slowa": len(WORD.findall(first)),
        "zajawka": body[:HOOK_CHARS], "cta_strona": bool(CTA_SITE.search(tail)),
        "cta_rozmowa": bool(CTA_TALK.search(body)) or "?" in tail,
        "liczby_spoza_danych": unknown_nums, "nazwy": name_hits,
    }
    issues: list[tuple[str, str]] = []
    sub: dict[str, float] = {}

    # dlugosc wzgledem ustawien postaci
    lo, hi = min_words, max_words
    if m["slowa"] < lo:
        sub["dlugosc"] = max(0.0, 100 - (lo - m["slowa"]) * 2)
        issues.append(("za_krotki", f"{m['slowa']} słów (cel {lo}-{hi})"))
    elif m["slowa"] > hi:
        sub["dlugosc"] = max(0.0, 100 - (m["slowa"] - hi) * 2)
        issues.append(("za_dlugi", f"{m['slowa']} słów (cel {lo}-{hi})"))
    else:
        sub["dlugosc"] = 100.0
    # czytelnosc: gawęda dla wszystkich -> FOG 6-10 idealnie
    sub["czytelnosc"] = 100.0 if fog <= 10 else max(0.0, 100 - (fog - 10) * 15)
    if fog > 11:
        issues.append(("trudny_tekst", f"FOG-PL {fog} - {fog_label(fog)}"))
    if m["najdluzsze_zdanie"] > 30:
        sub["czytelnosc"] -= 15
        issues.append(("dlugie_zdanie", f"zdanie ma {m['najdluzsze_zdanie']} słów"))
    # struktura na telefonie: akapity, brak "sciany tekstu"
    sub["struktura"] = 100.0
    if m["akapity"] < 2 and m["slowa"] > 80:
        sub["struktura"] -= 40
        issues.append(("sciana_tekstu", "jeden akapit - na telefonie to ściana tekstu"))
    if m["najdluzszy_akapit_znaki"] > 600:
        sub["struktura"] -= 25
        issues.append(("dlugi_akapit", f"akapit ma {m['najdluzszy_akapit_znaki']} znaków"))
    # haczyk: pierwsze zdanie krotkie i z emocja / pytaniem / konkretem
    sub["haczyk"] = 100.0
    if m["haczyk_slowa"] > 18:
        sub["haczyk"] -= 40
        issues.append(("dlugi_haczyk", f"pierwsze zdanie ma {m['haczyk_slowa']} słów - nie zmieści się w zajawce"))
    if not re.search(r"[!?]|\d", first) and m["haczyk_slowa"] > 10:
        sub["haczyk"] -= 20
    # gwara: wyrazna, ale do czytania
    g = m["gwara_proc"]
    # lekka godka: wyczuwalna, ale zrozumiala (ok. 3-12% slow)
    sub["gwara"] = 100.0 if 3 <= g <= 12 else max(0.0, 100 - (3 - g) * 25) if g < 3 else max(0.0, 100 - (g - 12) * 8)
    if g < 2:
        issues.append(("slaba_gwara", f"tylko {g}% słów gwarowych - postać brzmi zbyt poprawnie"))
    elif g > 18:
        issues.append(("ciezka_gwara", f"{g}% słów gwarowych - może być trudne do czytania"))
    mining = sorted({w.lower() for w in MINING.findall(body)})
    if mining:
        sub["gwara"] = max(0.0, sub["gwara"] - 30)
        issues.append(("kopalnia", "nawiązania do kopalni/gruby: " + ", ".join(mining[:5])))
    # zaangazowanie
    sub["zaangazowanie"] = 40.0 + (35 if m["cta_rozmowa"] else 0) + (25 if m["cta_strona"] else 0)
    if not m["cta_rozmowa"]:
        issues.append(("brak_pytania", "brak pytania / zaproszenia do komentarzy"))
    if not m["cta_strona"]:
        issues.append(("brak_zachety", "na końcu brak zachęty do zajrzenia do Kwiatownika"))
    # jezyk: roznorodnosc, powtorzenia, emoji
    sub["jezyk"] = min(100.0, m["ttr"] * 130) - 8 * len(repeats)
    if repeats:
        issues.append(("powtorzenia", ", ".join(f"{w}×{c}" for w, c in repeats[:4])))
    if m["emoji"] > max_emoji:
        sub["jezyk"] -= 15
        issues.append(("za_duzo_emoji", f"{m['emoji']} emoji (max {max_emoji})"))
    # zgodnosc z danymi (liczby, nazwy)
    sub["zgodnosc"] = 100.0
    if unknown_nums:
        sub["zgodnosc"] -= min(60, 20 * len(unknown_nums))
        issues.append(("liczby_spoza_danych", "liczby, których nie ma w danych: " + ", ".join(unknown_nums[:6])))
    if names and not any(name_hits.values()):
        sub["zgodnosc"] -= 30
        issues.append(("brak_nazwy", "nie pada nazwa rośliny"))

    sub = {k: round(max(0.0, min(100.0, v)), 1) for k, v in sub.items()}
    weights = {"dlugosc": 1, "czytelnosc": 2, "struktura": 1.5, "haczyk": 1.5, "gwara": 1.5, "zaangazowanie": 1.5,
               "jezyk": 1, "zgodnosc": 2}
    m["podwyniki"] = sub
    m["wynik_kodu"] = round(sum(sub[k] * w for k, w in weights.items()) / sum(weights.values()), 1)
    m["problemy"] = [{"kod": k, "opis": d} for k, d in issues]
    return m


ISSUE_LABELS = {
    "za_krotki": "za krótki", "za_dlugi": "za długi", "trudny_tekst": "trudny tekst", "dlugie_zdanie": "długie zdanie",
    "sciana_tekstu": "ściana tekstu", "dlugi_akapit": "długi akapit", "dlugi_haczyk": "długie pierwsze zdanie",
    "slaba_gwara": "słaba gwara", "ciezka_gwara": "zbyt ciężka gwara", "brak_pytania": "brak pytania do czytelników",
    "brak_zachety": "brak zachęty do Kwiatownika", "powtorzenia": "powtórzenia słów", "za_duzo_emoji": "za dużo emoji",
    "liczby_spoza_danych": "liczby spoza danych", "brak_nazwy": "brak nazwy rośliny",
    "kopalnia": "nawiązania do kopalni",
}
SUBSCORE_LABELS = {"dlugosc": "Długość", "czytelnosc": "Czytelność", "struktura": "Struktura (telefon)",
                   "haczyk": "Pierwsze zdanie", "gwara": "Gwara", "zaangazowanie": "Wezwanie do działania",
                   "jezyk": "Język i powtórzenia", "zgodnosc": "Zgodność z danymi"}
