"""Ciekawostka tradycyjna przy przepisie leczniczym: Pięć Przemian (Wu Xing) i humory Galena - BEZ LLM.

Decyzja z 2026-11-02: Wu Xing i profil energetyczny rośliny nie są głównymi czynnikami opracowania przepisu.
Przepis zostaje taki jak w źródle; Kwiatownik pokazuje przy nim tylko mały dymek-ciekawostkę.
Wszystko jest liczone z pliku rośliny głównej w Kwiatowniku (profil_energetyczny: smak, termika, wilgotnosc):

- Wu Xing: pierwszy wyraźny smak -> przemiana i narząd (klasyczne pięć smaków: kwaśny-Drewno-wątroba,
  gorzki-Ogień-serce, słodki-Ziemia-śledziona, ostry-Metal-płuca, słony-Woda-nerki). Pole "zywiol" w plikach
  opisuje cztery żywioły zachodnie (Powietrze, Ogień...), więc NIE jest używane.
- Galen: termika -> ciepłe / zimne, wilgotność -> suche / wilgotne. Stopnia (I-IV) nie zgadujemy.
- Wartości zależne od części rośliny ("Chłodna (liście), rozgrzewająca (owoce)") - wybierana część z przepisu;
  sprzeczne albo neutralne wpisy -> brak tej części ciekawostki (nic nie jest zgadywane).
- Tylko przepisy lecznicze (typ "medyczne_..."); barwienie, kosmetyki, nalewki-likiery i kuchnia - bez dymka.
"""
import re
import unicodedata

SOURCE = "profil energetyczny rośliny w Kwiatowniku (smak, termika, wilgotność)"
NOTE = "Ciekawostka z dawnych systemów ziołolecznictwa - nie porada medyczna."

# czesci rosliny: rdzen -> nazwa (kolejnosc ma znaczenie: "korz" przed "kor")
PARTS = (("korz", "korzeń"), ("kłą", "kłącze"), ("kla", "kłącze"), ("kor", "kora"), ("liś", "liście"),
         ("lis", "liście"), ("liść", "liście"), ("kwia", "kwiaty"), ("płat", "płatki"), ("owoc", "owoce"),
         ("nasi", "nasiona"), ("pąk", "pąki"), ("pak", "pąki"), ("ziel", "ziele"), ("sok", "sok"),
         ("miąż", "miąższ"), ("orzech", "orzechy"), ("bulw", "bulwy"), ("igł", "igły"), ("pęd", "pędy"),
         ("żywic", "żywica"), ("szysz", "szyszki"), ("jagod", "owoce"))

# smak -> (smak, przemiana, przemiana w dopelniaczu, narzad w narzedniku)
TASTES = (("kwa", ("kwaśny", "Drewno", "Drewna", "wątrobą")),
          ("gorz", ("gorzki", "Ogień", "Ognia", "sercem")),
          ("słod", ("słodki", "Ziemia", "Ziemi", "śledzioną")),
          ("slod", ("słodki", "Ziemia", "Ziemi", "śledzioną")),
          ("ostr", ("ostry", "Metal", "Metalu", "płucami")),
          ("piek", ("ostry", "Metal", "Metalu", "płucami")),
          ("słon", ("słony", "Woda", "Wody", "nerkami")),
          ("slon", ("słony", "Woda", "Wody", "nerkami")))
WEAK = ("lekko", "delikatnie", "nieco", "lekki", "słabo")
# chinski zegar narzadow (pora najwiekszej aktywnosci) dla narzadow Pieciu Przemian - narzednik -> (dopelniacz, godziny)
ORGAN_CLOCK = {"wątrobą": ("wątroby", "1–3"), "sercem": ("serca", "11–13"), "śledzioną": ("śledziony", "9–11"),
               "płucami": ("płuc", "3–5"), "nerkami": ("nerek", "17–19")}

HOT = ("ciepł", "gorąc", "gorac", "rozgrzew", "ogrzew", "grzej")
COLD = ("chłod", "chlod", "zimn")
DRY = ("osusz",)
WET = ("nawilż", "nawilz")
OPPOSITE = {"ciepłe": "zimnych", "zimne": "ciepłych", "suche": "wilgotnych", "wilgotne": "suchych"}


def part_of(text: str | None) -> str | None:
    """Nazwa czesci rosliny w tekscie (np. 'Ziele dziurawca' -> 'ziele', 'kwiaty' -> 'kwiaty')."""
    for w in re.findall(r"\w+", (text or "").lower()):
        for stem, name in PARTS:
            if w.startswith(stem):
                return name
    return None


def segments(text: str) -> list[tuple[set[str] | None, str]]:
    """'Słodki (sok), gorzki (liście, pąki)' -> [({'sok'}, 'Słodki'), ({'liście','pąki'}, 'gorzki')].
    Nawias bez nazw czesci (np. '(ściągająca)') to tylko opis - zostaje w tekscie segmentu."""
    out: list[tuple[set[str] | None, str]] = []
    rest = text or ""
    pos = 0
    buf = ""
    for m in re.finditer(r"\(([^)]*)\)", rest):
        inner = m.group(1)
        parts = {p for p in (part_of(x) for x in re.split(r"[,/;]| i ", inner)) if p}
        buf += rest[pos:m.start()]
        if parts:
            out.append((parts, buf.strip(" ,;/")))
            buf = ""
        else:
            buf += " " + inner + " "
        pos = m.end()
    buf += rest[pos:]
    if buf.strip(" ,;/.-"):
        out.append((None, buf.strip(" ,;/")))
    return out


def pick(text: str | None, part: str | None) -> str | None:
    """Fragment wpisu dla danej czesci rosliny; None = nie da sie jednoznacznie wybrac."""
    segs = segments(text or "")
    if not segs:
        return None
    if part:
        for parts, seg in segs:
            if parts and part in parts:
                return seg
    general = [seg for parts, seg in segs if parts is None]
    if len(segs) == 1:
        return segs[0][1]
    if general and all(p is None for p, _ in segs):
        return " ".join(general)
    return general[0] if general and not part else None


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.lower())


def _has(seg: str, stems: tuple[str, ...]) -> bool:
    return any(w.startswith(stems) for w in re.findall(r"\w+", _norm(seg)))


def taste(seg: str | None) -> tuple[str, str, str, str] | None:
    """Pierwszy wyrazny smak (bez 'lekko ...'); gdy sa tylko slabe - pierwszy z nich."""
    if not seg:
        return None
    strong, weak = [], []
    words = re.findall(r"\w+", _norm(seg))
    for i, w in enumerate(words):
        for stem, val in TASTES:
            if w.startswith(stem):
                (weak if i and words[i - 1] in WEAK else strong).append(val)
                break
    found = strong or weak
    return found[0] if found else None


def galen_quality(termika: str | None, wilgotnosc: str | None) -> list[str]:
    """['zimne', 'suche'] itd.; os neutralna albo sprzeczna -> pominieta."""
    out = []
    if termika and not _has(termika, ("neutral",)):
        hot, cold = _has(termika, HOT), _has(termika, COLD)
        if hot != cold:
            out.append("ciepłe" if hot else "zimne")
    if wilgotnosc and not _has(wilgotnosc, ("neutral", "stabiliz")):
        dry, wet = _has(wilgotnosc, DRY), _has(wilgotnosc, WET)
        if dry != wet:
            out.append("suche" if dry else "wilgotne")
    return out


def is_medicinal(rec: dict) -> bool:
    return str(rec.get("typ") or "").startswith("medyczne")


def main_plant(rec: dict) -> tuple[str | None, str | None]:
    """(id rosliny glownej, czesc rosliny w przepisie)."""
    pid = rec.get("roslina_id")
    skl = [s for s in rec.get("skladniki") or [] if isinstance(s, dict)]
    if not pid:
        pid = next((s.get("link_id") for s in skl if s.get("link_id")), None)
    part = None
    for s in skl:
        if s.get("link_id") == pid:
            part = part_of(s.get("czesc_rosliny")) or part_of(s.get("nazwa"))
            break
    return pid, part


def tradition(rec: dict, plant: dict | None) -> dict | None:
    """Ciekawostka dla przepisu albo None (przepis nieleczniczy, brak profilu, nic jednoznacznego)."""
    if not is_medicinal(rec) or not isinstance(plant, dict):
        return None
    prof = plant.get("profil_energetyczny")
    if not isinstance(prof, dict):
        return None
    _, part = main_plant(rec)
    name = plant.get("nazwa_pl") or plant.get("id") or ""
    out: dict = {"roslina": name, "roslina_id": plant.get("id"), "zrodlo": SOURCE, "uwaga": NOTE}
    if part:
        out["czesc"] = part
    t = taste(pick(str(prof.get("smak") or ""), part))
    if t:
        smak, przemiana, przemiany, narzad = t
        clock = ORGAN_CLOCK.get(narzad)
        out["wu_xing"] = {"smak": smak, "przemiana": przemiana,
                          "tekst": f"W tradycji Pięciu Przemian (Wu Xing) smak {smak} łączono z przemianą "
                                   f"{przemiany} i z {narzad}."
                                   + (f" Według chińskiego zegara narządów pora {clock[0]} to godz. {clock[1]}."
                                      if clock else "")}
    q = galen_quality(pick(str(prof.get("termika") or ""), part), pick(str(prof.get("wilgotnosc") or ""), part))
    if q:
        jak = " i ".join(q)
        przeciw = " i ".join(OPPOSITE[x] for x in q)
        out["galen"] = {"jakosc": jak,
                        "tekst": f"W tradycji Galena to zioło uchodziło za {jak}; takie zioła podawano przy "
                                 f"dolegliwościach {przeciw}."}
    return out if ("wu_xing" in out or "galen" in out) else None


def annotate(rec: dict, load_plant) -> dict:
    """Nowy rekord z polem 'ciekawostka_tradycyjna' (albo bez niego). load_plant(id) -> dane pliku rosliny."""
    out = {k: v for k, v in rec.items() if k != "ciekawostka_tradycyjna"}
    pid, _ = main_plant(rec)
    if not pid or not is_medicinal(rec):
        return out
    c = tradition(rec, load_plant(pid))
    if c:
        out["ciekawostka_tradycyjna"] = c
    return out
