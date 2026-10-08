"""Tlumaczenie przepisu na polski i ulozenie go w formacie Kwiatownika (LLM + walidacja).

Zasada: LLM tlumaczy i porzadkuje TYLKO to, co jest w zrodle - nie dopisuje wlasciwosci leczniczych,
dawkowania ani pol interpretacyjnych (wu xing, kampo...), ktorych oryginal nie zawiera.
"""
import json
import re
import unicodedata
from datetime import date

from app.llm.base import InvalidOutputError


class NoRecipeError(InvalidOutputError):
    """Strona nie zawiera konkretnego przepisu - to nie blad modelu."""


class IncompleteRecipeError(NoRecipeError):
    """Brak tytulu, skladnikow albo krokow - zwykle lista przepisow albo artykul ogolny."""

TYPES = ["kulinarne", "kulinarno_medyczne_plyn", "kulinarno_medyczne_syrop", "kulinarno_medyczne_stale",
         "kulinarno_medyczne_nalewka",
         "medyczne_wewnetrzne_napar", "medyczne_wewnetrzne_odwar", "medyczne_wewnetrzne_syrop",
         "medyczne_wewnetrzne_nalewka", "medyczne_wewnetrzne_macerat", "medyczne_wewnetrzne_wino",
         "medyczne_wewnetrzne_ferment", "medyczne_wewnetrzne_proszek", "medyczne_wewnetrzne_plyn",
         "medyczne_wewnetrzne_eliksir", "medyczne_wewnetrzne_esencja", "medyczne_wewnetrzne_miod_pitny",
         "medyczne_zewnetrzne_masc", "medyczne_zewnetrzne_oklad", "medyczne_zewnetrzne_olej",
         "medyczne_zewnetrzne_plyn", "medyczne_zewnetrzne_kapiel", "medyczne_zewnetrzne_balsam",
         "medyczne_zewnetrzne_ocet", "medyczne_zewnetrzne_wziewne", "medyczne_zewnetrzne_kataplazm",
         "medyczne_zewnetrzne_proszek", "medyczne_zewnetrzne_dym",
         "barwierskie_wlosy", "barwierskie_tkaniny", "barwierskie_inne",
         "kosmetyczne_skora", "kosmetyczne_wlosy", "kosmetyczne_mydlo", "gospodarcze"]
MAX_SOURCE_CHARS = 12000   # tekst artykulu dla LLM (dluzsze artykuly z wieloma przepisami)
MAX_EXTRA_RECIPES = 6      # ile dodatkowych przepisow z jednego artykulu
TYPE_RE = re.compile(r"^(medyczne|kulinarno_medyczne|barwierskie|kosmetyczne|gospodarcze)[a-z_]*$")
CULINARY_TYPE = "kulinarne"


def is_culinary_type(typ: str | None) -> bool:
    return typ == CULINARY_TYPE


def skip_reason(rec: dict, skip_culinary: bool, require_plants: bool) -> str | None:
    """Dlaczego przepis nie pasuje do Kwiatownika (None = pasuje)."""
    if skip_culinary and is_culinary_type(rec.get("typ")):
        return "przepis kulinarny - pominiety (SKIP_CULINARY)"
    if require_plants and rec.get("ziola_kluczowe") is False:
        return "rosliny nie sa kluczowym skladnikiem - pominiety (REQUIRE_KEY_PLANTS)"
    return None


TRADITION = {"ua": "Tradycja ukraińska", "de": "Tradycja niemiecka", "at": "Tradycja austriacka",
             "fr": "Tradycja francuska", "es": "Tradycja hiszpańska", "it": "Tradycja włoska", "cz": "Tradycja czeska",
             "sk": "Tradycja słowacka", "lt": "Tradycja litewska", "hu": "Tradycja węgierska",
             "ro": "Tradycja rumuńska", "bg": "Tradycja bułgarska", "hr": "Tradycja chorwacka",
             "rs": "Tradycja serbska", "gb": "Tradycja brytyjska", "us": "Tradycja amerykańska",
             "pl": "Tradycja polska"}
LANG_COUNTRY = {"uk": "ua", "de": "de", "fr": "fr", "es": "es", "it": "it", "cs": "cz", "sk": "sk", "lt": "lt",
                "hu": "hu", "ro": "ro", "bg": "bg", "hr": "hr", "sr": "rs", "pl": "pl"}

SYSTEM_PROMPT = """Jestes redaktorem polskiego serwisu "Kwiatownik" (ziololecznictwo: leki ziolowe, nalewki,
barwienie wlosow i tkanin roslinami, naturalna kosmetyka). Tlumaczysz przepis z obcego jezyka na naturalny, poprawny jezyk polski i ukladasz go w JSON.
ZASADY:
- Tlumacz wiernie. NIE dodawaj informacji, ktorych nie ma w oryginale (zadnych nowych wlasciwosci leczniczych,
  dawkowania, ostrzezen ani skladnikow). Jesli czegos nie ma - daj null albo pusta liste.
- Nazwy roslin podawaj po polsku (nazwa zwyczajowa). Jednostki amerykanskie (cup, oz, F) zamien na metryczne,
  zostawiajac oryginal w nawiasie, np. "240 ml (1 cup)".
- Kroki przygotowania: krotkie, w trybie bezokolicznikowym ("Zalać wrzątkiem...").
- "typ": wybierz z listy TYPY. Zwykle danie, ciasto, zupa, salatka czy deser bez celu leczniczego = "kulinarne".
  Barwienie wlosow = "barwierskie_wlosy", tkanin/wlóczki = "barwierskie_tkaniny", kosmetyki = "kosmetyczne_*",
  nalewka/likier ziolowy pity dla smaku = "kulinarno_medyczne_nalewka", nalewka lecznicza = "medyczne_wewnetrzne_nalewka".
- "ziola_kluczowe": true, gdy rosliny (ziola, kwiaty, korzenie, kora, grzyby) sa KLUCZOWYM skladnikiem czynnym
  przepisu; false, gdy rosliny sa tylko dodatkiem lub ich brak (np. maseczka z samej glinki, mydlo bez ziol).
Zwroc TYLKO JSON:
{"tytul": "...", "opis": "1-2 zdania lub null", "typ": "jeden z listy TYPY", "ziola_kluczowe": true, "metoda": "np. napar, nalewka, maść, barwienie",
 "roslina": "glowna roslina po polsku lub null", "roslina_id": "id z KATALOGU ROSLIN lub null",
 "skladniki": [{"nazwa": "...", "ilosc": "... lub null", "czesc_rosliny": "np. kwiat, liść, korzeń lub null",
                "link_id": "id z KATALOGU ROSLIN lub null"}],
 "sposob_przygotowania": ["...", "..."], "porcje": "... lub null", "czas_przygotowania": "... lub null",
 "stosowanie_i_dawkowanie": {"okolicznosci_stosowania": "...", "dawkowanie_standardowe": "..."} lub null,
 "uwagi": "... lub null", "tagi": ["..."],
 "dodatkowe_przepisy": [ ...kolejne ODREBNE przepisy z tego samego tekstu, w tym samym formacie (bez tego pola)... ]}
Jesli tekst zawiera kilka roznych przepisow (np. artykul "5 naparow na kaszel"), pierwszy opisz w glownych polach,
a pozostale w "dodatkowe_przepisy" (max 6). Jesli w tekscie NIE MA konkretnego przepisu (tylko lista linkow,
ogolny artykul), zwroc {"brak_przepisu": true}."""


def slugify(text: str) -> str:
    t = text.lower().replace("ł", "l")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")[:60] or "przepis"


def build_prompt(recipe: dict | None, raw_text: str | None, meta: dict, plants: dict[str, str]) -> str:
    src = json.dumps(recipe, ensure_ascii=False, indent=1) if recipe else (raw_text or "")[:MAX_SOURCE_CHARS]
    catalog = "\n".join(f"{pid}: {name}" for pid, name in sorted(plants.items()))
    return (f"JEZYK ORYGINALU: {meta.get('language') or 'nieznany'}\nTYTUL ORYGINALNY: {meta.get('title')}\n"
            f"TYPY: {', '.join(TYPES)}\n\nKATALOG ROSLIN (id: nazwa):\n{catalog or '(brak)'}\n\n"
            f"PRZEPIS ({'dane strukturalne' if recipe else 'tekst strony - najpierw wyciagnij z niego przepis'}):\n{src}")


def normalize_record(data: dict, meta: dict, plants: dict[str, str], taken_ids: set[str]) -> dict:
    """Porzadkuje odpowiedz LLM do formatu Kwiatownika i sprawdza wymagane pola."""
    tytul = str(data.get("tytul") or "").strip()
    skl = []
    for s in data.get("skladniki") or []:
        if isinstance(s, str):
            s = {"nazwa": s}
        if not isinstance(s, dict) or not str(s.get("nazwa") or "").strip():
            continue
        link = s.get("link_id") if s.get("link_id") in plants else None
        skl.append({"nazwa": str(s["nazwa"]).strip(), "ilosc": s.get("ilosc") or None,
                    **({"czesc_rosliny": s["czesc_rosliny"]} if s.get("czesc_rosliny") else {}),
                    **({"link_id": link} if link else {})})
    steps = [re.sub(r"^\s*(krok\s*\d+[:.)]\s*|\d+[.)]\s*)", "", str(x), flags=re.I).strip()
             for x in (data.get("sposob_przygotowania") or []) if str(x).strip()]
    if not tytul or not skl or not steps:
        raise IncompleteRecipeError("tlumaczenie niekompletne (brak tytulu, skladnikow lub krokow)")

    base = slugify(tytul)
    rid = base if base not in taken_ids else f"{base}_s{meta.get('item_id')}"
    country = meta.get("country") or LANG_COUNTRY.get((meta.get("language") or "")[:2])
    typ = str(data.get("typ") or "").strip().lower()
    if typ not in TYPES and not TYPE_RE.match(typ):
        typ = "inne" if not typ else ("kulinarne" if typ.startswith("kulinar") else "inne")
    roslina_id = data.get("roslina_id") if data.get("roslina_id") in plants else None
    rec = {
        "id": rid,
        "tytul": tytul,
        "typ": typ,
        "roslina": data.get("roslina") or None,
        "slug": roslina_id,
        "pochodzenie": [TRADITION[country]] if country in TRADITION else None,
        "metoda": data.get("metoda") or None,
        "opis": data.get("opis") or None,
        "skladniki": skl,
        "sposob_przygotowania": [f"Krok {i}: {s}" for i, s in enumerate(steps, 1)],
        "porcje": data.get("porcje") or None,
        "czas_przygotowania": data.get("czas_przygotowania") or None,
        "stosowanie_i_dawkowanie": data.get("stosowanie_i_dawkowanie") or None,
        "uwagi": data.get("uwagi") or None,
        "tagi": [t for t in (data.get("tagi") or []) if isinstance(t, str)][:10],
    }
    rec = {k: v for k, v in rec.items() if v not in (None, [], "")}
    if data.get("ziola_kluczowe") is False or str(data.get("ziola_kluczowe")).lower() == "false":
        rec["ziola_kluczowe"] = False      # informacja dla filtra - usuwana przed eksportem
    return rec


async def translate_recipe(llm, recipe: dict | None, raw_text: str | None, meta: dict, plants: dict[str, str],
                           taken_ids: set[str]) -> tuple[dict, str]:
    """Zwraca (rekord Kwiatownika po polsku, 'provider/model'). Rzuca LLMError przy niepowodzeniu."""
    res = await llm.complete(build_prompt(recipe, raw_text, meta, plants), system=SYSTEM_PROMPT, json_mode=True)
    try:
        data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
    except ValueError as exc:
        raise InvalidOutputError("LLM zwrocil niepoprawny JSON") from exc
    if data.get("brak_przepisu") is True:
        raise NoRecipeError("w tekscie nie ma konkretnego przepisu (lista lub artykul ogolny)")
    rec = normalize_record(data, meta, plants, taken_ids)
    extras = []
    for i, extra in enumerate((data.get("dodatkowe_przepisy") or [])[:MAX_EXTRA_RECIPES], 2):
        if not isinstance(extra, dict):
            continue
        try:
            e = normalize_record(extra, dict(meta, item_id=f"{meta.get('item_id')}_{i}"), plants,
                                 taken_ids | {rec["id"]} | {x["id"] for x in extras})
        except InvalidOutputError:
            continue
        extras.append(e)
    rec["siedziba"] = {"item_id": meta.get("item_id"), "jezyk_oryginalu": meta.get("language"),
                       "przetlumaczono": date.today().isoformat(), "model": f"{res.provider}/{res.model}"}
    for e in extras:
        e["siedziba"] = dict(rec["siedziba"])
    if extras:
        rec["_dodatkowe"] = extras      # odbiera export.translate_one (osobne przepisy z tego samego artykulu)
    return rec, f"{res.provider}/{res.model}"
