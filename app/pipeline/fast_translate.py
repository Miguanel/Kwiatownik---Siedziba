"""Szybkie tlumaczenie przepisu BEZ LLM: struktura ze strony (profil scrapera / JSON-LD) + NLLB + reguly.

Kiedy dziala: przepis ma dane strukturalne (tytul, skladniki, kroki) i znany jezyk obslugiwany przez NLLB.
Surowe artykuly (bez struktury, czesto kilka przepisow w jednym tekscie) nadal idzie przez LLM.

Kroki:
1. jednostki amerykanskie -> metryczne (w oryginale angielskim, oryginal zostaje w nawiasie),
2. NLLB tlumaczy wszystkie pola jednym wsadem (tytul, opis, skladniki, kroki, uwagi, porcje),
3. skladniki: ilosc oddzielona od nazwy, rosliny dopasowane do katalogu Kwiatownika (nazwa polska / lacinska),
4. typ przepisu z regul slow kluczowych (po polsku i w oryginale); gdy reguly nie rozstrzygaja - krotkie
   pytanie do LLM (tylko klasyfikacja, bez tlumaczenia) albo "inne",
5. ten sam `normalize_record` co sciezka LLM -> ten sam format Kwiatownika.
"""
import asyncio
import json
import re
from datetime import date
from fractions import Fraction

from app.knowledge.sections import norm_text
from app.mt.nllb import MTQualityError, MTUnavailable, looks_broken, nllb_code
from app.pipeline.translate import TYPES, normalize_record

MAX_BROKEN_SHARE = 0.2           # wiecej zepsutych fragmentow -> cale tlumaczenie przez LLM

# ------------------------------------------------------------ jednostki (tylko tekst angielski)
_FRAC = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8"}
_NUM = r"(\d+(?:[.,]\d+)?(?:\s+\d/\d)?|\d/\d)"
_UNITS = [(r"cups?", 240, "ml"), (r"fl\.?\s*oz\.?|fluid\s+ounces?", 30, "ml"), (r"oz\.?|ounces?", 28, "g"),
          (r"lbs?\.?|pounds?", 454, "g"), (r"quarts?|qt\.?", 946, "ml"), (r"pints?|pt\.?", 473, "ml"),
          (r"gallons?|gal\.?", 3785, "ml")]


def _num(s: str) -> float:
    s = s.replace(",", ".").strip()
    return float(sum(Fraction(p) for p in s.split())) if "/" in s else float(s)


def _round(v: float, unit: str) -> str:
    if unit == "ml" and v >= 1000:
        return f"{v / 1000:.2g} l".replace(".", ",")
    step = 5 if v >= 20 else 1
    return f"{int(round(v / step) * step) or 1} {unit}"


def metric_units(text: str) -> str:
    """'2 cups water' -> '480 ml (2 cups) water'; '350°F' -> '175°C (350°F)'."""
    for k, v in _FRAC.items():
        text = text.replace(k, f" {v}")
    for pat, factor, unit in _UNITS:
        text = re.sub(rf"(?<![\w(]){_NUM}\s*({pat})(?![\w)])",
                      lambda m: f"{_round(_num(m.group(1)) * factor, unit)} ({m.group(1).strip()} {m.group(2)})",
                      text, flags=re.I)
    text = re.sub(r"(\d{2,3})\s*°?\s*F\b", lambda m: f"{round((int(m.group(1)) - 32) * 5 / 9 / 5) * 5}°C "
                  f"({m.group(1)}°F)", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ------------------------------------------------------------ skladniki
_QTY = re.compile(
    r"^\s*((?:ok\.\s*|około\s*)?[\d½¼¾⅓⅔][\d\s.,/½¼¾⅓⅔–-]*"
    r"(?:\s*(?:\([^)]*\)))?"
    r"(?:\s*(?:g|kg|dag|mg|ml|l|litr\w*|łyż\w*|lyz\w*|szklan\w*|filiżan\w*|kub\w*|garś\w*|szczyp\w*|krop\w*|"
    r"sztuk\w*|szt\.?|plast\w*|gałąz\w*|gałąź|łodyg\w*|pęcz\w*|główk\w*|ząb\w*|ząbk\w*|uncj\w*|funt\w*|kwart\w*|"
    r"cm|%)\.?(?:\s*\([^)]*\))?)?)\s+(.+)$", re.I)


def split_quantity(text: str) -> tuple[str | None, str]:
    """'2 łyżki suszonych kwiatów rumianku' -> ('2 łyżki', 'suszonych kwiatów rumianku')."""
    m = _QTY.match(text or "")
    if not m or len(m.group(2)) < 2:
        return None, (text or "").strip()
    return m.group(1).strip(), m.group(2).strip(" ,;")


_STOP = {"zwyczajny", "zwyczajna", "pospolity", "pospolita", "lekarski", "lekarska", "wlasciwa", "wlasciwy",
         "czarny", "czarna", "bialy", "biala", "polny", "polna"}


def _root(word: str) -> str:
    """Rdzen polskiego slowa bez koncowki odmiany: rumianek -> rumian, mieta -> miet, lipa -> lip."""
    return word[:len(word) - 2] if len(word) >= 6 else word[:max(3, len(word) - 1)]


class PlantIndex:
    """Dopasowanie nazw skladnikow do katalogu roslin: rdzen polskiej nazwy albo rodzaj/gatunek lacinski."""

    def __init__(self, plants: dict[str, str]):
        self.entries = []
        for pid, label in plants.items():
            m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", label)
            pl, lat = (m.group(1), m.group(2)) if m else (label, "")
            words = [w for w in norm_text(pl).split() if len(w) > 2]
            self.entries.append((pid, words, norm_text(lat).split()))

    def match(self, *texts: str) -> str | None:
        t = " " + " ".join(norm_text(x) for x in texts if x) + " "
        best, best_score, tie = None, 0, False
        for pid, words, lat in self.entries:
            score = 0
            if len(lat) >= 2 and f" {lat[0]} {lat[1]}" in t:
                score = 10
            elif words and len(words[0]) <= 3:          # "bez czarny", "len": samo "bez" to tez przyimek
                if f" {words[0]} " in t and any(f" {w[:4]}" in t for w in words[1:]):
                    score = 4
            elif words and f" {_root(words[0])}" in t:
                score = 3 + sum(1 for w in words[1:] if w not in _STOP and f" {_root(w)}" in t) * 2
            if score > best_score:
                best, best_score, tie = pid, score, False
            elif score and score == best_score:
                tie = True
        return None if tie or best_score == 0 else best


# ------------------------------------------------------------ typ przepisu (reguly)
def _stem(s: str) -> str:
    return (" " if s.startswith(" ") else "") + norm_text(s) + (" " if s.endswith(" ") else "")


def _has(t: str, *stems: str) -> bool:
    """Czy tekst (po norm_text) zawiera ktorys rdzen. Krotkie rdzenie tylko od poczatku slowa."""
    for s in stems:
        n = _stem(s)
        if (len(n.strip()) < 5 and re.search(rf"(?<!\w){re.escape(n)}", t)) or (len(n.strip()) >= 5 and n in t):
            return True
    return False


CULINARY = ("ciast", "zup", "salat", "deser", "placek", "plack", "chleb", "pieczen", "obiad", "makaron", "risott",
            "kotlet", "muffin", "ciastecz", "dzem", "konfitur", "smoothie", "koktajl", "lody", "tort", "pierog",
            "nalesnik", "omlet", "sos ", "marynat", "cake", "soup", "salad", "dessert", "cookie", "bread", "pasta",
            "kuchen", "suppe", "торт", "суп", "салат", "пирог")
MEDICINAL = ("leczn", "zdrow", "kaszel", "kaszl", "gardl", "przezięb", "przezieb", "odpornos", "trawien", "bol",
             "zapalen", "gorączk", "goraczk", "stres", "sen ", "bezsenn", "rany", "skor", "remedy", "cough",
             "healing", "medicin", "heilkr", "heilmittel", "erkält", "лечеб", "лікув", "от кашля", "від кашлю")


def classify(pl: str, original: str = "") -> str | None:
    """Typ przepisu z tekstu po polsku (+ oryginalu). None = reguly nie rozstrzygaja."""
    t = " " + norm_text(pl) + " " + norm_text(original) + " "
    dye = _has(t, "farb", "barwi", "koloryz", "henn", "dye", "färb", "фарб", "окраш", "teinture")
    if dye and _has(t, "wlos", "hair", "haar", "волос", "cheveu"):
        return "barwierskie_wlosy"
    if dye and _has(t, "weln", "tkanin", "przedz", "wloczk", "jedwab", "bawel", "wool", "yarn", "fabric", "wolle",
                    "stoff", "пряж", "ткан", "шерст"):
        return "barwierskie_tkaniny"
    if dye and _has(t, "pisank", "jaj", "egg", "ostereier"):
        return "barwierskie_inne"
    if _has(t, "mydl", "soap", "seife", "мыл", "мил"):
        return "kosmetyczne_mydlo"
    if _has(t, "szampon", "odzywk", "plukank", "shampoo", "conditioner", "haarspül") or \
            (_has(t, "wlos", "hair", "haar") and _has(t, "masecz", "olejek", "mask")):
        return "kosmetyczne_wlosy"
    if _has(t, "krem", "tonik", "masecz", "peeling", "serum", "pomadk", "balsam do ust", "lip balm", "cream",
            "face mask", "gesichts", "крем", "маск"):
        return "kosmetyczne_skora"
    if _has(t, "masc", "salve", "ointment", "salbe", "мазь", "мазі"):
        return "medyczne_zewnetrzne_masc"
    if _has(t, "kataplazm", "poultice", "cataplasm"):
        return "medyczne_zewnetrzne_kataplazm"
    if _has(t, "oklad", "kompres", "compress", "umschlag", "компрес"):
        return "medyczne_zewnetrzne_oklad"
    if _has(t, "kapiel", "bath", "badezusatz", "vollbad", "ванн"):
        return "medyczne_zewnetrzne_kapiel"
    if _has(t, "inhalac", "inhal", "ингаля", "інгаля"):
        return "medyczne_zewnetrzne_wziewne"
    if _has(t, "likier", "liqueur", "likör", "ликер", "лікер"):
        return "kulinarno_medyczne_nalewka"
    if _has(t, "nalewk", "tinktur", "tincture", "настойк", "настоянк", "настоян"):
        return "medyczne_wewnetrzne_nalewka"
    if _has(t, "miod pitn", "mead", "met ", "медовух"):
        return "medyczne_wewnetrzne_miod_pitny"
    if _has(t, "syrop", "syrup", "sirup", "сироп"):
        return "medyczne_wewnetrzne_syrop"
    if _has(t, "odwar", "decoction", "abkoch", "отвар", "відвар"):
        return "medyczne_wewnetrzne_odwar"
    if _has(t, "macerat", "olej ziol", "olejek ziol", "infused oil", "ölauszug", "масляный экстракт"):
        return "medyczne_zewnetrzne_olej"
    culinary = _has(t, *CULINARY)
    if _has(t, "napar", "herbat", "herbal tea", "tea ", "kräutertee", " tee ", "чай", "infusion", "tisane"):
        return "kulinarne" if culinary and not _has(t, *MEDICINAL) else "medyczne_wewnetrzne_napar"
    if culinary and not _has(t, *MEDICINAL):
        return "kulinarne"
    return None


METHODS = {"napar": "napar", "odwar": "odwar", "syrop": "syrop", "nalewka": "nalewka", "masc": "maść",
           "oklad": "okład", "kataplazm": "kataplazm", "kapiel": "kąpiel", "olej": "macerat olejowy",
           "wziewne": "inhalacja", "miod_pitny": "miód pitny", "mydlo": "mydło", "wlosy": "pielęgnacja włosów",
           "skora": "kosmetyk", "tkaniny": "barwienie", "inne": "barwienie"}


def method_for(typ: str | None) -> str | None:
    if not typ:
        return None
    if typ.startswith("barwierskie"):
        return "barwienie"
    return METHODS.get(typ.rsplit("_", 1)[-1]) or METHODS.get("_".join(typ.split("_")[-2:]))


CLASSIFY_PROMPT = """Klasyfikujesz przepis polskiego zielnika. Zwroc TYLKO JSON: {"typ": "jeden z listy TYPY",
"ziola_kluczowe": true}. "kulinarne" = zwykle danie bez celu leczniczego. ziola_kluczowe=false, gdy rosliny
nie sa kluczowym skladnikiem czynnym. TYPY: """ + ", ".join(TYPES)


async def llm_classify(llm, rec: dict) -> tuple[str | None, bool | None]:
    """Krotkie pytanie o typ (gdy reguly nie rozstrzygaja). Blad -> (None, None), nigdy wyjatek."""
    if llm is None:
        return None, None
    prompt = f"TYTUL: {rec['tytul']}\nSKLADNIKI: {', '.join(rec['skladniki_txt'])}\nKROKI: {' '.join(rec['kroki'])[:800]}"
    try:
        res = await llm.complete(prompt, system=CLASSIFY_PROMPT, json_mode=True)
        data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
    except Exception:
        return None, None
    typ = str(data.get("typ") or "").strip().lower()
    zk = data.get("ziola_kluczowe")
    return (typ if typ in TYPES else None), (zk if isinstance(zk, bool) else None)


def iso_duration(v) -> str | None:
    """'PT1H30M' -> '1 godz. 30 min'; inne teksty bez zmian."""
    if not v:
        return None
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:\d+S)?", str(v).strip(), re.I)
    if not m or not any(m.groups()):
        return str(v)
    d, h, mi = (int(x) if x else 0 for x in m.groups())
    h += d * 24
    return " ".join(p for p in (f"{h} godz." if h else "", f"{mi} min" if mi else "") if p) or None


# ------------------------------------------------------------ calosc
def can_fast_translate(recipe: dict | None, language: str | None) -> bool:
    return bool(recipe and not recipe.get("article") and recipe.get("title") and recipe.get("ingredients")
                and recipe.get("steps") and nllb_code(language))


async def fast_translate_recipe(mt, llm, recipe: dict, meta: dict, plants: dict[str, str],
                                taken_ids: set[str]) -> tuple[dict, str]:
    """(rekord Kwiatownika, 'nllb/<model>'). Rzuca MTUnavailable / MTQualityError -> sciezka LLM."""
    lang = (meta.get("language") or "")[:2].lower()
    if not can_fast_translate(recipe, lang):
        raise MTUnavailable("brak danych strukturalnych albo nieobslugiwany jezyk")
    fix = metric_units if lang == "en" else (lambda s: s)
    fields = {"title": [recipe["title"]], "description": [recipe.get("description") or ""],
              "notes": [recipe.get("notes") or ""], "yield": [str(recipe.get("yield") or "")],
              "ingredients": [fix(x) for x in recipe["ingredients"]][:60],
              "steps": [fix(x) for x in recipe["steps"]][:40]}
    flat = [(k, i, s) for k, vals in fields.items() for i, s in enumerate(vals) if s.strip()]
    outs = await asyncio.to_thread(mt.translate, [s for _, _, s in flat], lang, "pl")
    broken = sum(1 for (_, _, s), o in zip(flat, outs) if looks_broken(s, o))
    if flat and broken / len(flat) > MAX_BROKEN_SHARE:
        raise MTQualityError(f"tlumaczenie maszynowe zepsute w {broken}/{len(flat)} fragmentach")
    tr: dict[str, list[str]] = {k: list(v) for k, v in fields.items()}
    for (k, i, s), o in zip(flat, outs):
        tr[k][i] = o if not looks_broken(s, o) else s

    index = PlantIndex(plants)
    skl, names = [], []
    for orig, pl in zip(fields["ingredients"], tr["ingredients"]):
        qty, name = split_quantity(pl)
        link = index.match(name, orig)
        skl.append({"nazwa": name[:1].upper() + name[1:] if name else pl, "ilosc": qty, **({"link_id": link} if link else {})})
        names.append(pl)
    text_pl = " ".join([tr["title"][0], tr["description"][0], *names, *tr["steps"]])
    text_orig = " ".join([recipe["title"], recipe.get("description") or "", *recipe["ingredients"]])
    typ = classify(f"{tr['title'][0]} {tr['description'][0]}", recipe["title"]) or classify(text_pl, text_orig)
    zk = None
    how = "reguly"
    if typ is None:
        typ, zk = await llm_classify(llm, {"tytul": tr["title"][0], "skladniki_txt": names, "kroki": tr["steps"]})
        how = "llm-klasyfikacja" if typ else "brak"
    main = next((s["link_id"] for s in skl if s.get("link_id")), None) or index.match(tr["title"][0], recipe["title"])
    data = {"tytul": tr["title"][0], "opis": tr["description"][0] or None, "typ": typ or "inne",
            "metoda": method_for(typ), "roslina": (plants.get(main) or "").split(" (")[0] or None, "roslina_id": main,
            "skladniki": skl, "sposob_przygotowania": tr["steps"], "porcje": tr["yield"][0] or None,
            "czas_przygotowania": iso_duration(recipe.get("total_time")), "uwagi": tr["notes"][0] or None,
            "tagi": [t for t in [method_for(typ)] if t]}
    if zk is False:
        data["ziola_kluczowe"] = False
    rec = normalize_record(data, meta, plants, taken_ids)
    model = f"nllb/{mt.model_dir.name}"
    rec["siedziba"] = {"item_id": meta.get("item_id"), "jezyk_oryginalu": meta.get("language"),
                       "przetlumaczono": date.today().isoformat(), "model": model, "typ_ustalono": how}
    return rec, model
