"""Opracowanie przepisu BEZ LLM (decyzja 2026-11-02: zadnych halucynacji LLM w przepisach - tylko to, co wyciagnieto
ze zrodla przepisu, i dane z plikow roslin Kwiatownika).

Dawniej LLM dopisywal do kazdego przepisu: doktryne sygnatur ("kminek ma sygnature wiatru"), spagiryke, Kampo,
chronoterapie, "zakazy" dietetyczne, wlasciwosci fizyczne, wskazowki tradycyjne i ostrzezenia z glowy. Te pola sa
teraz usuwane, a w ich miejsce:
- mechanizm_tworzenia: nazwa metody z typu przepisu (napar, odwar, nalewka...) - nic ponad typ,
- bezpieczenstwo_i_interakcje: pola "ostrzezenia" i "interakcje" plikow WSZYSTKICH roslin ze skladnikow (i ostrzezenia
  czesci rosliny uzytej w przepisie) + stale reguly (alkohol, zaprawy farbiarskie),
- zbior: czas zbioru uzytej czesci rosliny z jej pliku,
- ciekawostka_tradycyjna: dymek Wu Xing / Galen (app/pipeline/traditions.py),
- stosowanie_i_dawkowanie: tylko pola ze zrodla; dawkowanie, ktorego liczb nie ma w tekscie zrodla, jest zastepowane
  "Zrodlo nie podaje dawkowania" (tak jak liczby w wiedzy o roslinach musza byc w cytacie).
"""
import re
from datetime import date

from app.pipeline.enrich import NO_DOSAGE, strip_invented
from app.pipeline.traditions import annotate, main_plant, part_of

# pola dopisywane dawniej przez LLM - nigdy nie ida dalej
LLM_KEYS = ("klasyfikacja_dzialania", "wymogi_szamanskie_i_czasowe", "wskazowki_tradycyjne", "wlasciwosci_fizyczne",
            "bezpieczenstwo_i_interakcje", "mechanizm_tworzenia", "opracowanie", "zbior")
# "okolicznosci stosowania" z dawnego opracowania LLM (systemy, ktorych nie ma w zrodlach) - usuwane
SYSTEM_WORDS = re.compile(r"chronoterap|żywio|zywio|wu ?xing|przemian|meridian|matryc|zegar|kampo|ajurwed|pitta|"
                          r"vata|kapha|dosz|humor|sygnatur|spagir|płucom|w okolicy wody|godz\.?\s*\d+\s*[–-]", re.I)
SOURCE_NOTE = ("Ostrzeżenia pochodzą z kart roślin Kwiatownika (ich źródła są na stronach roślin). Przepis ma charakter "
               "edukacyjny i historyczny - nie jest poradą medyczną. Przed stosowaniem, zwłaszcza w ciąży, u dzieci, "
               "przy chorobach przewlekłych i lekach, skonsultuj się z lekarzem lub farmaceutą.")
NO_WARNINGS = "Brak opisanych ostrzeżeń w kartach roślin - przed użyciem sprawdź przeciwwskazania każdej rośliny."
ALCOHOL_NOTE = "Zawiera alkohol: nie dla dzieci, kobiet w ciąży i karmiących, osób prowadzących pojazdy i leczonych z uzależnień."
MORDANT_NOTE = ("Zaprawy farbiarskie (ałun, siarczan żelaza lub miedzi) - pracuj w rękawicach, w przewietrzonym miejscu, "
                "nie używaj naczyń kuchennych.")
ALCOHOL_RE = re.compile(r"wódk|wodk|spirytus|alkohol|koniak|brandy|rum\b|gin\b|wino|wina\b|likier|bimber", re.I)
MORDANT_RE = re.compile(r"ałun|alun|siarczan|chrom|dwuchromian", re.I)

# typ przepisu -> nazwa metody (ostatni czlon typu, np. medyczne_wewnetrzne_napar -> napar)
METHODS = {"napar": "Napar", "odwar": "Odwar", "syrop": "Syrop", "nalewka": "Nalewka (maceracja w alkoholu)",
           "macerat": "Macerat", "wino": "Wino ziołowe", "ferment": "Fermentacja", "proszek": "Proszek",
           "eliksir": "Eliksir", "esencja": "Esencja", "pitny": "Miód pitny (fermentacja)", "masc": "Maść",
           "oklad": "Okład", "olej": "Macerat olejowy", "kapiel": "Kąpiel", "balsam": "Balsam",
           "ocet": "Ocet ziołowy (maceracja w occie)", "wziewne": "Inhalacja", "kataplazm": "Kataplazm",
           "dym": "Okadzanie", "puder": "Puder", "mydlo": "Mydło"}


def mechanism_for(typ: str | None) -> list[str] | None:
    t = str(typ or "").strip().lower()
    if t.startswith("barwierskie"):
        return ["Barwienie roślinne"]
    last = t.rsplit("_", 1)[-1] if t else ""
    return [METHODS[last]] if last in METHODS else None


def _plant_ids(rec: dict) -> list[str]:
    ids = []
    for pid in [rec.get("roslina_id")] + [s.get("link_id") for s in rec.get("skladniki") or [] if isinstance(s, dict)]:
        if isinstance(pid, str) and pid and pid not in ids:
            ids.append(pid)
    return ids


def _part_data(plant: dict, part: str | None) -> dict | None:
    if not part:
        return None
    for key, val in (plant.get("czesci_rosliny") or {}).items():
        if isinstance(val, dict) and part_of(key.replace("_", " ")) == part:
            return val
    return None


def _clean(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip() if isinstance(v, str) else ""


def safety(rec: dict, load_plant) -> tuple[dict, list[str]]:
    """(blok bezpieczenstwa, id roslin, z ktorych pochodzi) - z kart roslin i stalych regul, bez LLM."""
    warns, inter, used = [], [], []
    parts = {s.get("link_id"): part_of(s.get("czesc_rosliny")) or part_of(s.get("nazwa"))
             for s in rec.get("skladniki") or [] if isinstance(s, dict) and s.get("link_id")}
    for pid in _plant_ids(rec):
        plant = load_plant(pid)
        if not isinstance(plant, dict):
            continue
        name = plant.get("nazwa_pl") or pid
        bits = [_clean(plant.get("ostrzezenia"))]
        part = _part_data(plant, parts.get(pid))
        if part:
            bits.append(_clean(part.get("ostrzezenia")))
        bits = [b for b in dict.fromkeys(bits) if b]
        if bits:
            warns.append(f"{name}: " + " ".join(bits))
        if _clean(plant.get("interakcje")):
            inter.append(f"{name}: {_clean(plant.get('interakcje'))}")
        if bits or _clean(plant.get("interakcje")):
            used.append(pid)
    text = " ".join([str(rec.get("typ") or ""), str(rec.get("tytul") or "")] +
                    [str(s.get("nazwa") if isinstance(s, dict) else s) for s in rec.get("skladniki") or []])
    rules = []
    if ALCOHOL_RE.search(text) or "nalewka" in str(rec.get("typ") or "") or "miod_pitny" in str(rec.get("typ") or ""):
        rules.append(ALCOHOL_NOTE)
    if str(rec.get("typ") or "").startswith("barwierskie") and MORDANT_RE.search(text):
        rules.append(MORDANT_NOTE)
    out = {"ostrzezenia": " • ".join(warns + rules) if warns or rules else NO_WARNINGS}
    if inter:
        out["interakcje_z_lekami"] = " • ".join(inter)
    return out, used


def harvest(rec: dict, load_plant) -> dict | None:
    """Czas zbioru czesci rosliny glownej uzytej w przepisie - z karty rosliny."""
    pid, part = main_plant(rec)
    plant = load_plant(pid) if pid else None
    if not isinstance(plant, dict):
        return None
    pd = _part_data(plant, part)
    when = _clean(pd.get("czas_zbioru")) if pd else ""
    if not when:
        return None
    return {"roslina": plant.get("nazwa_pl") or pid, "roslina_id": pid, "czesc": part, "kiedy": when}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", text or "")}


def usage(rec: dict, source_text: str | None) -> dict | None:
    """Stosowanie i dawkowanie tylko ze zrodla. Dawkowanie z liczbami, ktorych nie ma w tekscie zrodla -> NO_DOSAGE."""
    sd = rec.get("stosowanie_i_dawkowanie") if isinstance(rec.get("stosowanie_i_dawkowanie"), dict) else {}
    out = {}
    occ = _clean(sd.get("okolicznosci_stosowania"))
    if occ and not SYSTEM_WORDS.search(occ):
        out["okolicznosci_stosowania"] = occ
    dose = _clean(sd.get("dawkowanie_standardowe"))
    if dose and dose != NO_DOSAGE and source_text is not None and not _numbers(dose) <= _numbers(source_text):
        dose = ""                                     # liczby spoza zrodla - nie pokazujemy wymyslonej dawki
    if dose:
        out["dawkowanie_standardowe"] = dose
    elif str(rec.get("typ") or "").startswith(("medyczne", "kulinarno_medyczne")):
        out["dawkowanie_standardowe"] = NO_DOSAGE
    return out or None


def opracuj(rec: dict, load_plant, source_text: str | None = None, today: str | None = None) -> dict:
    """Przepis po tlumaczeniu -> przepis do Kwiatownika: bez pol dopisanych przez LLM, z danymi z kart roslin.
    load_plant(id) -> dane pliku rosliny albo None. source_text: tekst zrodla (sprawdzanie liczb dawkowania)."""
    out = {k: v for k, v in strip_invented(rec).items() if k not in LLM_KEYS}
    sd = usage(rec, source_text)
    out.pop("stosowanie_i_dawkowanie", None)
    if sd:
        out["stosowanie_i_dawkowanie"] = sd
    mech = mechanism_for(out.get("typ"))
    if mech:
        out["mechanizm_tworzenia"] = mech
    out["bezpieczenstwo_i_interakcje"], used = safety(out, load_plant)
    z = harvest(out, load_plant)
    if z:
        out["zbior"] = z
    out = annotate(out, load_plant)
    out["opracowanie"] = {"jak": "bez LLM: karty roślin Kwiatownika i źródło przepisu", "rosliny": used,
                          "data": today or date.today().isoformat(), "uwaga": SOURCE_NOTE}
    return out
