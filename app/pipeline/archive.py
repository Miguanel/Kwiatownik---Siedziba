"""Import przepisow z archiwum starego Kwiatownika (1): static/data/przepisy/*_global.json.

Przepisy sa juz po polsku - konwersja do formatu Kwiatownika2 jest deterministyczna (bez LLM):
tytul, typ (z metody/tytulu), skladniki (+ link_id rosliny z katalogu), kroki, dawkowanie, dzialanie, zrodla.
Opracowanie w tradycyjnych systemach mozna potem dodac zadaniem "Opracuj".
"""
import re

from app.pipeline.translate import slugify

# (fragment metody lub tytulu, typ Kwiatownika) - pierwsze dopasowanie wygrywa
TYPE_RULES: list[tuple[str, str]] = [
    ("włos", "kosmetyczne_wlosy"), ("barwi", "barwierskie_tkaniny"), ("farb", "barwierskie_tkaniny"),
    ("inhalacja_dymna", "medyczne_zewnetrzne_dym"), ("okadza", "medyczne_zewnetrzne_dym"), ("dym", "medyczne_zewnetrzne_dym"),
    ("inhalac", "medyczne_zewnetrzne_wziewne"), ("kataplazm", "medyczne_zewnetrzne_kataplazm"),
    ("okład", "medyczne_zewnetrzne_oklad"), ("oklad", "medyczne_zewnetrzne_oklad"), ("kompres", "medyczne_zewnetrzne_oklad"),
    ("kąpiel", "medyczne_zewnetrzne_kapiel"), ("maść", "medyczne_zewnetrzne_masc"), ("masc", "medyczne_zewnetrzne_masc"),
    ("balsam", "medyczne_zewnetrzne_balsam"), ("ocet", "medyczne_zewnetrzne_ocet"), ("octy", "medyczne_zewnetrzne_ocet"),
    ("olej", "medyczne_zewnetrzne_olej"), ("płukank", "medyczne_zewnetrzne_plyn"), ("przemyw", "medyczne_zewnetrzne_plyn"),
    ("krople", "medyczne_zewnetrzne_plyn"),
    ("nalewk", "medyczne_wewnetrzne_nalewka"), ("tinctur", "medyczne_wewnetrzne_nalewka"), ("tinktur", "medyczne_wewnetrzne_nalewka"),
    ("miód_pitny", "medyczne_wewnetrzne_miod_pitny"), ("miód pitny", "medyczne_wewnetrzne_miod_pitny"),
    ("wino", "medyczne_wewnetrzne_wino"), ("piwo", "medyczne_wewnetrzne_ferment"), ("ferment", "medyczne_wewnetrzne_ferment"),
    ("eliksir", "medyczne_wewnetrzne_eliksir"), ("esencj", "medyczne_wewnetrzne_esencja"),
    ("syrop", "medyczne_wewnetrzne_syrop"), ("oksymel", "medyczne_wewnetrzne_syrop"), ("miodek", "medyczne_wewnetrzne_syrop"),
    ("macerat", "medyczne_wewnetrzne_macerat"), ("proszek", "medyczne_wewnetrzne_proszek"),
    ("odwar", "medyczne_wewnetrzne_odwar"), ("napar", "medyczne_wewnetrzne_napar"), ("herbat", "medyczne_wewnetrzne_napar"),
    ("mieszank", "medyczne_wewnetrzne_napar"), ("mikstur", "medyczne_wewnetrzne_plyn"), ("sok", "medyczne_wewnetrzne_plyn"),
]


def guess_type(metoda: str | None, tytul: str) -> str:
    text = f"{metoda or ''} {tytul}".lower()
    for frag, typ in TYPE_RULES:
        if frag in text:
            return typ
    return "medyczne"


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def _steps(v) -> list[str]:
    parts = _as_list(v)
    if len(parts) == 1 and len(parts[0]) > 120:  # jeden dlugi opis -> zdania
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", parts[0]) if p.strip()]
    clean = [re.sub(r"^(krok\s*\d+[:.)]\s*|\d+[.)]\s*)", "", p, flags=re.I) for p in parts]
    return [f"Krok {i}: {p}" for i, p in enumerate(clean, 1)]


def _plant_keys(plants: dict[str, str]) -> list[tuple[str, list[str]]]:
    """[(id, [rdzenie slow nazwy polskiej])] - do rozpoznawania roslin w opisie skladnika."""
    out = []
    for pid, name in plants.items():
        words = re.findall(r"\w+", name.split("(")[0].lower())[:2]
        if words:
            out.append((pid, [w[:max(4, len(w) - 2)] for w in words]))
    return out


def link_plant(text: str, keys: list[tuple[str, list[str]]]) -> str | None:
    low = text.lower()
    best = None
    for pid, stems in keys:
        if all(st in low for st in stems) and (best is None or len(stems) > len(best[1])):
            best = (pid, stems)
    return best[0] if best else None


def convert_k1_recipe(r: dict, plants: dict[str, str], taken_ids: set[str], ref: str) -> dict | None:
    tytul = str(r.get("tytul") or "").strip()
    skl_raw = _as_list(r.get("skladniki"))
    steps = _steps(r.get("sposob_przygotowania"))
    if not tytul or not skl_raw or not steps:
        return None
    tytul = tytul[0].upper() + tytul[1:]
    keys = _plant_keys(plants)
    skladniki = [{"nazwa": s, **({"link_id": lid} if (lid := link_plant(s, keys)) else {})} for s in skl_raw]
    slug = r.get("slug") if r.get("slug") in plants else (link_plant(r.get("roslina") or "", keys) if r.get("roslina") else None)
    base = slugify(tytul)
    rid = base if base not in taken_ids else f"{base}_{slugify(ref)[-12:]}"
    wl = _as_list(r.get("wlasciwosci"))
    cechy = _as_list(r.get("cechy"))
    sd = {k: v for k, v in {"okolicznosci_stosowania": r.get("zastosowanie") or None,
                            "dawkowanie_standardowe": r.get("dawkowanie") or None}.items() if v}
    uwagi = "; ".join(x for x in [str(r.get("uwagi") or "").strip()] + cechy if x)
    rec = {
        "id": rid,
        "tytul": tytul,
        "typ": guess_type(r.get("metoda"), tytul),
        "roslina": r.get("roslina") or None,
        "slug": slug,
        "pochodzenie": ["Archiwum Kwiatownika"],
        "metoda": (r.get("metoda") or "").replace("_", " ") or None,
        "opis": ("Działanie: " + "; ".join(wl)) if wl else None,
        "skladniki": skladniki,
        "sposob_przygotowania": steps,
        "stosowanie_i_dawkowanie": sd or None,
        "uwagi": uwagi or None,
        "tagi": [t for t in (r.get("tagi") or []) if isinstance(t, str)][:10],
        "zrodla": _as_list(r.get("zrodla")) + _as_list(r.get("zrodlo")),
    }
    return {k: v for k, v in rec.items() if v not in (None, [], "")}
