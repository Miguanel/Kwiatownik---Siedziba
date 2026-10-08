"""Sekcje wiedzy o roslinie (blok "wiedza" w pliku rosliny Kwiatownika)."""
import hashlib
import re
import unicodedata

SECTIONS: dict[str, str] = {
    "opis": "Opis i wyglad (morfologia, cechy rozpoznawcze)",
    "wystepowanie": "Wystepowanie i siedlisko",
    "czesci_rosliny": "Czesci rosliny i surowce zielarskie (co sie zbiera, kiedy, jak suszyc)",
    "sklad": "Sklad chemiczny i substancje czynne",
    "zastosowanie_lecznicze": "Zastosowanie w ziololecznictwie i medycynie ludowej",
    "medycyna_wschodu": "Tradycyjna medycyna chinska, kampo, ajurweda: natura (chlodna/ciepla), smak, meridiany, wskazania",
    "barwienie": "Barwienie (tkaniny, welna, wlosy, pisanki) i inne zastosowania rzemieslnicze",
    "kosmetyka": "Zastosowanie w kosmetyce",
    "historia": "Historia, etymologia nazwy, dawne zastosowania",
    "kultura": "Wierzenia, obrzedy, legendy i tradycje ludowe roznych krajow",
    "nazwy_ludowe": "Nazwy ludowe i regionalne",
    "bezpieczenstwo": "Toksycznosc, przeciwwskazania, interakcje, mozliwe pomylki",
    "uprawa": "Uprawa i rozmnazanie",
    "ciekawostki": "Inne ciekawostki",
}
SECTION_TITLES_PL: dict[str, str] = {
    "opis": "Opis i wygląd", "wystepowanie": "Występowanie", "czesci_rosliny": "Części rośliny i surowce",
    "sklad": "Skład i substancje czynne", "zastosowanie_lecznicze": "Ziołolecznictwo i medycyna ludowa",
    "medycyna_wschodu": "Medycyna chińska, kampo, ajurweda",
    "barwienie": "Barwienie i rzemiosło", "kosmetyka": "Kosmetyka", "historia": "Historia i nazwa",
    "kultura": "Wierzenia i tradycje", "nazwy_ludowe": "Nazwy ludowe", "bezpieczenstwo": "Bezpieczeństwo",
    "uprawa": "Uprawa", "ciekawostki": "Ciekawostki",
}


def norm_text(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").lower().replace("ł", "l"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", t).strip()


def fingerprint(plant_id: str, text: str) -> str:
    """Odcisk informacji: te same slowa (bez kolejnosci i odmiany koncowek) = ta sama informacja."""
    words = sorted({w[:6] for w in norm_text(text).split() if len(w) > 3})
    return hashlib.sha1(f"{plant_id}|{' '.join(words)}".encode()).hexdigest()[:16]


def similar(a: str, b: str) -> float:
    """Podobienstwo dwoch zdan (Jaccard rdzeni slow) - do wykrywania powtorzen z innych zrodel."""
    wa = {w[:6] for w in norm_text(a).split() if len(w) > 3}
    wb = {w[:6] for w in norm_text(b).split() if len(w) > 3}
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


# czesci rosliny: znormalizowany rdzen -> nazwa w Kwiatowniku (None = to nie czesc rosliny)
PART_MAP: list[tuple[tuple[str, ...], str | None]] = [
    (("lisc", "lisci", "liść", "blaszk"), "liście"),
    (("kwiatost", "kwiat", "platk", "baldach", "koszycz", "kłos", "klos"), "kwiaty"),
    (("pak", "pąk"), "pąki"),
    (("ziel", "nadziem", "lodyg", "łodyg", "ped", "pęd"), "ziele"),
    (("klacz", "kłącz", "rhizom"), "kłącze"),
    (("korz",), "korzeń"),
    (("nasion", "nasien"), "nasiona"),
    (("owoc", "jagod", "szyszkojag"), "owoce"),
    (("kor",), "kora"),
    (("sok", "mlecz"), "sok"),
    (("zywic", "żywic"), "żywica"),
    (("pylek", "pyłek"), "pyłek"),
    (("cala", "cała", "calej", "całej", "roslin", "rośl"), None),
]
PARTS = ("liście", "kwiaty", "pąki", "ziele", "kłącze", "korzeń", "nasiona", "owoce", "kora", "sok", "żywica",
         "pyłek")


def norm_part(part: str | None) -> str | None:
    """Dowolny zapis czesci ("calej rosliny", "lodygi i liscie", "toksycznosc") -> slownik Kwiatownika albo None."""
    if not part or str(part).strip().lower() in ("null", "none", "-"):
        return None
    p = norm_text(str(part))
    if not p:
        return None
    for stems, name in PART_MAP:                    # 1) poczatek zapisu ("paki kwiatowe" -> paki)
        if any(p.startswith(norm_text(s)) for s in stems):
            return name
    for stems, name in PART_MAP:                    # 2) dalsze slowa ("mlode lisci" -> liscie)
        if any(f" {norm_text(s)}" in f" {p}" for s in stems):
            return name
    return None
