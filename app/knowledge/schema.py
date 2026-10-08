"""Schemat pliku rosliny Kwiatownika (rozdzialy i podrozdzialy strony) + pokrycie: co jest, czego brakuje.

Kazdy "slot" to jedno pole pliku rosliny, ktore strona pokazuje jako rozdzial/podrozdzial, np.
"zastosowanie.medyczne" (Praktyczne zastosowanie -> Medycyna) albo "czesci_rosliny.ziele.czas_zbioru".
Slot moze byc wypelniony:
  - recznie (pole w pliku, wpisane w Kwiatowniku),
  - z sieci (blok "scalone" - wiedza z Siedziby wmontowana w ten podrozdzial, ze zrodlami),
  - albo pusty -> luka, ktora kolejne skany sieci maja wypelnic (app/knowledge/gaps.py).
"""
from dataclasses import dataclass

from app.knowledge.sections import PARTS, norm_part, norm_text


@dataclass(frozen=True)
class Slot:
    path: str                     # sciezka w pliku, np. "zastosowanie.medyczne"
    chapter: str                  # rozdzial strony
    title: str                    # podrozdzial
    kind: str = "tekst"           # tekst | lista (napisy) | obiekty (lista slownikow - tylko pokrycie)
    sections: tuple = ()          # sekcje wiedzy z sieci, ktore moga go wypelnic
    mergeable: bool = True        # czy Siedziba moze wmontowac tu wiedze z sieci
    hint: str = ""                # dla LLM: co nalezy do tego podrozdzialu


# Kolejnosc = kolejnosc rozdzialow na stronie rosliny (templates/plant_detail.html w Kwiatownik2)
SLOTS: tuple[Slot, ...] = (
    Slot("opis", "Opis", "Opis ogólny", sections=("opis", "wystepowanie"),
         hint="krotki ogolny opis rosliny: czym jest, gdzie rosnie, z czego slynie"),
    Slot("profil_energetyczny.smak", "Natura i siedlisko", "Smak", sections=("medycyna_wschodu",),
         hint="smak surowca w tradycyjnych systemach (gorzki, slodki, ostry, kwasny, slony, cierpki)"),
    Slot("profil_energetyczny.termika", "Natura i siedlisko", "Termika", sections=("medycyna_wschodu",),
         hint="natura termiczna: chlodna, zimna, neutralna, ciepla, goraca"),
    Slot("profil_energetyczny.wilgotnosc", "Natura i siedlisko", "Wilgotność", sections=("medycyna_wschodu",),
         hint="dzialanie osuszajace albo nawilzajace"),
    Slot("profil_energetyczny.zywiol", "Natura i siedlisko", "Żywioł", sections=("medycyna_wschodu",),
         hint="zywiol / przemiana (Wu Xing), planeta, meridiany / kanaly"),
    Slot("profil_energetyczny.opis", "Natura i siedlisko", "Opis energetyczny", sections=("medycyna_wschodu",),
         hint="dzialanie w medycynie chinskiej, kampo, ajurwedzie: wskazania, meridiany, syndromy"),
    Slot("wymagania.stanowisko", "Natura i siedlisko", "Stanowisko", sections=("uprawa", "wystepowanie"),
         hint="naslonecznienie, typ siedliska"),
    Slot("wymagania.gleba", "Natura i siedlisko", "Gleba", sections=("uprawa", "wystepowanie"),
         hint="rodzaj gleby, zyznosc, przepuszczalnosc"),
    Slot("wymagania.ph_gleby", "Natura i siedlisko", "pH gleby", sections=("uprawa",), hint="odczyn gleby"),
    Slot("wymagania.mrozoodpornosc", "Natura i siedlisko", "Mrozoodporność", sections=("uprawa",),
         hint="strefa mrozoodpornosci, odpornosc na mroz"),
    Slot("wymagania.woda", "Natura i siedlisko", "Woda", sections=("uprawa",),
         hint="wilgotnosc gleby, podlewanie, odpornosc na susze"),
    Slot("identyfikacja.cechy_kluczowe", "Rozpoznawanie w terenie", "Cechy kluczowe", kind="lista",
         sections=("opis",), hint="cechy morfologiczne pozwalajace rozpoznac rosline (liscie, kwiaty, lodyga, zapach)"),
    Slot("identyfikacja.mozliwe_pomyłki", "Rozpoznawanie w terenie", "Możliwe pomyłki", kind="obiekty",
         sections=("bezpieczenstwo",), mergeable=False),
    Slot("czesci_rosliny", "Surowce i zbiory", "Części rośliny (surowce)", kind="obiekty",
         sections=("czesci_rosliny", "sklad"), mergeable=False),
    Slot("zastosowanie.medyczne", "Praktyczne zastosowanie", "Medycyna", sections=("zastosowanie_lecznicze",),
         hint="dzialanie lecznicze calej rosliny, ziololecznictwo, medycyna ludowa"),
    Slot("zastosowanie.kulinarne", "Praktyczne zastosowanie", "Kuchnia", sections=(),
         hint="jadalnosc, uzycie w kuchni, przyprawy, napoje"),
    Slot("zastosowanie.rzemieslnicze", "Praktyczne zastosowanie", "Rzemiosło", sections=("barwienie",),
         hint="barwienie tkanin i wlosow, drewno, wlokna, garbowanie, inne rzemiosla"),
    Slot("zastosowanie.kosmetyczne", "Praktyczne zastosowanie", "Kosmetyka", sections=("kosmetyka",),
         hint="kosmetyki, pielegnacja skory i wlosow"),
    Slot("zastosowanie.ogrodowe", "Praktyczne zastosowanie", "Ogród", sections=("uprawa",),
         hint="uprawa w ogrodzie, rozmnazanie, roslina miododajna, ozdobna"),
    Slot("interakcje", "Bezpieczeństwo i interakcje", "Interakcje z lekami", sections=("bezpieczenstwo",),
         hint="interakcje z lekami i substancjami"),
    Slot("ostrzezenia", "Bezpieczeństwo i interakcje", "Przeciwwskazania", sections=("bezpieczenstwo",),
         hint="toksycznosc, przeciwwskazania, skutki uboczne"),
    Slot("kalendarz_ogrodnika.zadania", "Cykl życia i ekosystem", "Kalendarz", kind="obiekty",
         sections=("uprawa", "czesci_rosliny"), mergeable=False),
    Slot("permakultura.funkcje", "Cykl życia i ekosystem", "Permakultura - funkcje", kind="lista",
         sections=("uprawa",), hint="rola w ekosystemie: miododajna, wiaze azot, okrywa glebe, schronienie"),
    Slot("permakultura.gildie", "Cykl życia i ekosystem", "Gildie", kind="obiekty", mergeable=False),
    Slot("ciekawostki", "Wiedza tajemna", "Ciekawostki", kind="lista",
         sections=("ciekawostki", "historia", "kultura", "nazwy_ludowe"),
         hint="historia, etymologia, wierzenia, obrzedy, legendy, nazwy ludowe, inne ciekawostki"),
)
SLOT_BY_PATH = {s.path: s for s in SLOTS}

# podrozdzialy jednej czesci rosliny (czesci_rosliny.<czesc>.<pole>)
PART_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    # pole, tytul, rodzaj, podpowiedz dla LLM
    ("opis_botaniczny", "Opis surowca", "tekst", "jak wyglada ta czesc, co dokladnie sie zbiera"),
    ("wlasciwości", "Działanie", "tekst", "dzialanie i zastosowanie lecznicze TEJ czesci rosliny"),
    ("skladniki_aktywne", "Substancje czynne", "lista", "substancje czynne TEJ czesci (nazwy zwiazkow)"),
    ("czas_zbioru", "Zbiór", "tekst", "kiedy i jak zbierac oraz suszyc TE czesc"),
    ("ostrzezenia", "Ostrzeżenia", "tekst", "toksycznosc i przeciwwskazania TEJ czesci"),
)
PART_FIELD_ALIASES = {"wlasciwości": ("wlasciwości", "wlasciwosci", "właściwości")}


def get_path(data: dict, path: str):
    """Wartosc pola po sciezce z kropkami (None, gdy brak)."""
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip() not in ("-", "?", "brak", "brak danych")
    if isinstance(value, (list, dict)):
        return any(filled(v) for v in (value.values() if isinstance(value, dict) else value))
    return True


def part_key_for(data: dict, part: str | None) -> str | None:
    """Klucz w czesci_rosliny pliku dla czesci ze slownika Siedziby ("liście" -> "liscie", "korzen_i_klacze"...)."""
    if not part:
        return None
    parts = data.get("czesci_rosliny") if isinstance(data.get("czesci_rosliny"), dict) else {}
    want = norm_text(part)
    for key in parts:
        if norm_text(key.replace("_", " ")) == want:
            return key
    for key in parts:                                   # "korzen_i_klacze" -> korzeń, "kwiatostany" -> kwiaty
        if norm_part(key.replace("_", " ")) == part:
            return key
    return None


def part_field_key(part_data: dict, field: str) -> str:
    """Pole czesci w pisowni pliku (np. 'wlasciwosci' bez ogonka w starszych plikach)."""
    for alias in PART_FIELD_ALIASES.get(field, (field,)):
        if alias in part_data:
            return alias
    return field


# podrozdzialy NOWEJ czesci rosliny (znanej tylko z sieci) - bez tych dwoch czesc nie trafia do generatora
CORE_PART_FIELDS = ("wlasciwości", "skladniki_aktywne")


def web_parts(data: dict) -> list[str]:
    """Czesci rosliny, o ktorych jest wiedza z sieci, a ktorych nie ma w czesci_rosliny pliku (np. "korzeń",
    gdy plik opisuje tylko ziele). Klucz = nazwa ze slownika PARTS; Siedziba tworzy dla nich podrozdzialy
    w bloku "scalone", a Kwiatownik pokazuje je w "Surowce i zbiory" i w generatorze."""
    found: list[str] = []
    wiedza = data.get("wiedza") if isinstance(data.get("wiedza"), dict) else {}
    for sec in wiedza.get("sekcje") or []:
        for p in sec.get("punkty") or []:
            part = norm_part(p.get("czesc"))
            if part and part not in found:
                found.append(part)
    pola = ((data.get("scalone") or {}).get("pola") or {}) if isinstance(data.get("scalone"), dict) else {}
    for path in pola:
        bits = path.split(".")
        if len(bits) == 3 and bits[0] == "czesci_rosliny" and bits[1] in PARTS and bits[1] not in found:
            found.append(bits[1])
    return [p for p in found if part_key_for(data, p) is None]


def part_slots(data: dict) -> list[Slot]:
    """Sloty czesci rosliny: istniejace w pliku + nowe czesci znane tylko z sieci (web_parts)."""
    out = []
    parts = data.get("czesci_rosliny") if isinstance(data.get("czesci_rosliny"), dict) else {}
    for part in web_parts(data):
        for field, title, kind, hint in PART_FIELDS:
            out.append(Slot(f"czesci_rosliny.{part}.{field}", "Surowce i zbiory",
                            f"{part.capitalize()} (nowa część z sieci): {title}", kind=kind,
                            sections=("czesci_rosliny", "sklad", "zastosowanie_lecznicze", "bezpieczenstwo"),
                            hint=f"{hint} ({part})"))
    for key, pdata in parts.items():
        if not isinstance(pdata, dict):
            continue
        label = pdata.get("nazwa_surowca") or key.replace("_", " ")
        for field, title, kind, hint in PART_FIELDS:
            fk = part_field_key(pdata, field)
            out.append(Slot(f"czesci_rosliny.{key}.{fk}", "Surowce i zbiory", f"{label}: {title}", kind=kind,
                            sections=("czesci_rosliny", "sklad", "zastosowanie_lecznicze", "bezpieczenstwo"),
                            hint=f"{hint} ({key.replace('_', ' ')})"))
    return out


def merge_slots(data: dict) -> list[Slot]:
    """Wszystkie podrozdzialy pliku, do ktorych mozna wmontowac wiedze z sieci."""
    return [s for s in SLOTS if s.mergeable] + part_slots(data)


def slot_for(data: dict, path: str) -> Slot | None:
    return SLOT_BY_PATH.get(path) or next((s for s in part_slots(data) if s.path == path), None)


# ------------------------------------------------------------------ pokrycie
def coverage(data: dict | None, web_sections: dict[str, int] | None = None) -> dict:
    """Ktore podrozdzialy schematu sa wypelnione recznie, ktore wiedza z sieci, a ktorych brakuje.

    web_sections: {sekcja wiedzy: liczba informacji w bazie Siedziby (zweryfikowanych/zapisanych)} - informacje
    jeszcze niewmontowane w podrozdzial tez sie licza ("czeka na scalenie").
    Zwraca {"sloty": [{path, chapter, title, status: reczne|z_sieci|czeka|brak, zrodla}], "procent", "braki": [...]}
    """
    data = data or {}
    pola = ((data.get("scalone") or {}).get("pola") or {}) if isinstance(data.get("scalone"), dict) else {}
    web_sections = web_sections or {}
    rows = []
    for slot in SLOTS:
        value = get_path(data, slot.path)
        merged = pola.get(slot.path) or {}
        n_web = sum(web_sections.get(sec, 0) for sec in slot.sections)
        if merged.get("tresc") and any(t.get("zrodla") for t in merged["tresc"]):
            status = "z_sieci" if not filled(value) else "reczne+sieci"
        elif filled(value):
            status = "reczne"
        elif n_web:
            status = "czeka"
        else:
            status = "brak"
        rows.append({"path": slot.path, "chapter": slot.chapter, "title": slot.title, "status": status,
                     "zrodla": len({n for t in merged.get("tresc") or [] for n in t.get("zrodla") or []}),
                     "web": n_web, "mergeable": slot.mergeable})
    have = sum(1 for r in rows if r["status"] != "brak" and r["status"] != "czeka")
    return {"sloty": rows, "procent": round(100 * have / len(rows)) if rows else 0,
            "braki": [r["path"] for r in rows if r["status"] in ("brak", "czeka")],
            "puste": [r["path"] for r in rows if r["status"] == "brak"]}


def chapters(cov: dict) -> list[tuple[str, list[dict]]]:
    """Wiersze pokrycia pogrupowane w rozdzialy (kolejnosc strony)."""
    out: dict[str, list[dict]] = {}
    for r in cov["sloty"]:
        out.setdefault(r["chapter"], []).append(r)
    return list(out.items())
