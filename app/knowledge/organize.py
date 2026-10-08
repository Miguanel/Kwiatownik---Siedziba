"""Agregator wiedzy: z luznej listy informacji ("wiedza.fakty") robi uporzadkowane sekcje dla Kwiatownika.

Wynik (w pliku rosliny, obok surowych "fakty", ktore zostaja jako material zrodlowy):
  "wiedza": {
    "wersja": 2,
    "zrodla":  [{"nr": 1, "nazwa", "url", "jezyk", "typ": "wikipedia" | "strona"}],
    "sekcje":  [{"klucz", "tytul", "podsumowanie": "..." | null, "podsumowanie_zrodla": [1, 2],
                 "punkty": [{"tekst", "czesc": "liscie" | null, "zrodla": [1, 3], "fakty": ["s12", "s40"]}]}],
    "czesci":  {"liscie": ["s12", ...]},           # indeks: informacje o danej czesci rosliny
    "fakty":   [...]                                # surowe informacje (zawsze zachowane)
  }

Kroki:
1. deterministyczne: czesci rosliny do jednego slownika, numeracja zrodel (najpierw Wikipedia), kolejnosc sekcji;
2. LLM (redaktor): przenosi informacje do wlasciwych sekcji, laczy powtorzenia (zrodla sie sumuja),
   poprawia polszczyzne, uklada kolejnosc i pisze krotkie podsumowanie sekcji;
3. kontrola kodem: kazda informacja uzyta dokladnie raz, liczby tylko z oryginalow, tresc nie odbiega od
   oryginalow (wspolne rdzenie slow), zrodla tylko z laczonych informacji. Gdy LLM zawiedzie albo kontrola
   nie przejdzie - uklad deterministyczny (bez laczenia i podsumowan), nigdy utrata informacji.
"""
import json
import re

from app.knowledge.extract import numbers_supported
from app.knowledge.sections import PARTS, SECTION_TITLES_PL, SECTIONS, norm_part, norm_text  # noqa: F401

MAX_POINT = 600
HTML_RE = re.compile(r"<[a-z/!]", re.I)
MAX_SUMMARY = 700
MAX_LLM_FACTS = 160          # wiecej informacji -> uklad deterministyczny (odpowiedz LLM bylaby za dluga)


def number_sources(facts: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Lista zrodel z numerami (Wikipedia pierwsza, potem inne strony) i mapa url -> nr."""
    urls: dict[str, dict] = {}
    for f in facts:
        src = f.get("zrodlo") or {}
        url = src.get("url")
        if url and url not in urls:
            urls[url] = {"nazwa": src.get("nazwa") or url, "url": url, "jezyk": f.get("jezyk"),
                         "typ": "wikipedia" if "wikipedia.org" in url else "strona"}
    ordered = sorted(urls.values(), key=lambda z: (z["typ"] != "wikipedia", z["jezyk"] != "pl", z["url"]))
    for i, z in enumerate(ordered, 1):
        z["nr"] = i
    return ordered, {z["url"]: z["nr"] for z in ordered}


def deterministic(facts: list[dict]) -> dict:
    """Uklad bez LLM: sekcje w stalej kolejnosci, czesci znormalizowane, kazda informacja osobnym punktem."""
    sources, nr = number_sources(facts)
    by_sec: dict[str, list[dict]] = {}
    for f in facts:
        sec = f.get("sekcja") if f.get("sekcja") in SECTIONS else "ciekawostki"
        by_sec.setdefault(sec, []).append({
            "tekst": f["tekst"], "czesc": norm_part(f.get("czesc")),
            "zrodla": [nr[f["zrodlo"]["url"]]] if (f.get("zrodlo") or {}).get("url") in nr else [],
            "fakty": [f["id"]]})
    sekcje = [{"klucz": k, "tytul": SECTION_TITLES_PL[k], "podsumowanie": None, "podsumowanie_zrodla": [],
               "punkty": sorted(by_sec[k], key=lambda p: (p["czesc"] or "", p["zrodla"][:1]))}
              for k in SECTIONS if k in by_sec]
    return {"zrodla": sources, "sekcje": sekcje, "czesci": parts_index(sekcje)}


def parts_index(sekcje: list[dict]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for s in sekcje:
        for p in s["punkty"]:
            if p.get("czesc"):
                idx.setdefault(p["czesc"], []).extend(p["fakty"])
    return idx


SYSTEM_PROMPT = """Jestes redaktorem polskiego zielnika "Kwiatownik". Dostajesz ponumerowane informacje o jednej \
roslinie (ID, obecna sekcja, czesc rosliny, tekst). Uloz je w porzadna strukture:
- przenies informacje do WLASCIWEJ sekcji (np. jedzenie mlodych pedow to "ciekawostki" albo "historia",
  a nie "zastosowanie_lecznicze"; rozsiewanie nasion to "opis", a nie "kultura"),
- POLACZ informacje mowiace to samo (albo prawie to samo) w jeden punkt - podaj wszystkie ich ID,
- popraw polszczyzne (ortografia, skladnia, kalki z angielskiego, np. "sap" -> "sok"), NIE dodawaj nowych faktow,
  nazw ani liczb - tylko to, co jest w laczonych informacjach,
- uloz punkty w sekcji od najwazniejszych,
- dla sekcji z co najmniej 3 punktami napisz "podsumowanie": 1-3 zdania syntezy wylacznie z tych punktow.
Kazde ID musi wystapic DOKLADNIE RAZ. Czesc rosliny tylko z listy: """ + ", ".join(PARTS) + """ albo null.
Zwroc TYLKO JSON: {"sekcje": [{"klucz": "...", "podsumowanie": "... lub null",
 "punkty": [{"id": [3, 7], "tekst": "...", "czesc": "... lub null"}]}]}
Dozwolone klucze sekcji: """ + ", ".join(SECTIONS)


def build_prompt(plant_pl: str, latin: str | None, facts: list[dict]) -> str:
    lines = [f"ID {i} | {f.get('sekcja')} | {norm_part(f.get('czesc')) or '-'} | {f['tekst']}"
             for i, f in enumerate(facts, 1)]
    return f"ROSLINA: {plant_pl}" + (f" ({latin})" if latin else "") + "\n\nINFORMACJE:\n" + "\n".join(lines)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in norm_text(text).split() if len(w) > 3}


def _faithful(new: str, originals: list[str]) -> bool:
    """Tekst po redakcji: liczby tylko z oryginalow i wiekszosc slow nowego tekstu obecna w oryginalach."""
    joined = " ".join(originals)
    if not numbers_supported(new, joined):
        return False
    ns, os_ = _stems(new), _stems(joined)
    return bool(ns) and len(ns & os_) / len(ns) >= 0.6


def apply_llm_layout(raw: str, facts: list[dict]) -> tuple[dict | None, list[str]]:
    """Odpowiedz redaktora -> uklad wiedzy albo (None, bledy), gdy kontrola nie przejdzie."""
    errors: list[str] = []
    try:
        data = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except ValueError:
        return None, ["redaktor zwrocil niepoprawny JSON"]
    if not isinstance(data, dict):
        return None, ["redaktor zwrocil JSON, ktory nie jest obiektem"]
    raw_secs = data.get("sekcje") or []
    if isinstance(raw_secs, dict):        # {"opis": {...}} albo {"opis": [punkty]} zamiast listy sekcji
        raw_secs = [{"klucz": k, **v} if isinstance(v, dict) else {"klucz": k, "punkty": v}
                    for k, v in raw_secs.items()]
    if not isinstance(raw_secs, list):
        return None, ["redaktor zwrocil zly format sekcji"]
    sources, nr = number_sources(facts)
    used: list[int] = []
    sekcje: dict[str, dict] = {}
    for sec in raw_secs:
        if not isinstance(sec, dict):
            errors.append("sekcja nie jest obiektem")
            continue
        key = sec.get("klucz")
        if key not in SECTIONS:
            errors.append(f"nieznana sekcja {key}")
            continue
        out = sekcje.setdefault(key, {"klucz": key, "tytul": SECTION_TITLES_PL[key], "podsumowanie": None,
                                      "podsumowanie_zrodla": [], "punkty": []})
        punkty = sec.get("punkty") or []
        for p in punkty if isinstance(punkty, list) else []:
            if not isinstance(p, dict):
                errors.append("punkt nie jest obiektem")
                continue
            ids = p.get("id")
            ids = ids if isinstance(ids, list) else [ids]
            try:
                ids = [int(i) for i in ids]
            except (TypeError, ValueError):
                errors.append("zle ID punktu")
                continue
            if not ids or any(not (1 <= i <= len(facts)) for i in ids):
                errors.append(f"ID spoza listy: {ids}")
                continue
            used += ids
            orig = [facts[i - 1] for i in ids]
            text = re.sub(r"\s+", " ", str(p.get("tekst") or "")).strip()
            if not (20 <= len(text) <= MAX_POINT) or HTML_RE.search(text):
                errors.append(f"zly tekst punktu {ids}")
                continue
            if not _faithful(text, [f["tekst"] for f in orig]):
                errors.append(f"punkt {ids} odbiega od oryginalow")
                continue
            srcs = sorted({nr[f["zrodlo"]["url"]] for f in orig if (f.get("zrodlo") or {}).get("url") in nr})
            out["punkty"].append({"tekst": text, "czesc": norm_part(p.get("czesc")), "zrodla": srcs,
                                  "fakty": [f["id"] for f in orig]})
        summary = re.sub(r"\s+", " ", str(sec.get("podsumowanie") or "")).strip()
        if summary and summary.lower() != "null":
            texts = [pt["tekst"] for pt in out["punkty"]]
            if len(out["punkty"]) >= 3 and len(summary) <= MAX_SUMMARY and _faithful(summary, texts) \
                    and not HTML_RE.search(summary):
                out["podsumowanie"] = summary
                out["podsumowanie_zrodla"] = sorted({n for pt in out["punkty"] for n in pt["zrodla"]})
    missing = set(range(1, len(facts) + 1)) - set(used)
    doubled = {i for i in used if used.count(i) > 1}
    if missing:
        errors.append(f"pominiete informacje: {sorted(missing)[:10]}")
    if doubled:
        errors.append(f"informacje uzyte wiele razy: {sorted(doubled)[:10]}")
    if errors:
        return None, errors
    ordered = [sekcje[k] for k in SECTIONS if k in sekcje and sekcje[k]["punkty"]]
    return {"zrodla": sources, "sekcje": ordered, "czesci": parts_index(ordered)}, []


async def organize(llm, plant_pl: str, latin: str | None, facts: list[dict], log=None) -> tuple[dict, str]:
    """Uklad wiedzy dla Kwiatownika. Zwraca (uklad, opis: 'llm:model' albo 'deterministyczny: powod')."""
    say = log or (lambda m: None)
    if llm is None or len(facts) < 3 or len(facts) > MAX_LLM_FACTS:
        return deterministic(facts), "deterministyczny"
    try:
        res = await llm.complete(build_prompt(plant_pl, latin, facts), system=SYSTEM_PROMPT, json_mode=True)
    except Exception as exc:
        say(f"  redaktor niedostepny ({str(exc)[:80]}) - uklad bez laczenia")
        return deterministic(facts), "deterministyczny: brak LLM"
    try:
        layout, errors = apply_llm_layout(res.text, facts)
    except Exception as exc:  # noqa: BLE001 - nietypowa odpowiedz modelu nie moze zatrzymac zapisu wiedzy
        layout, errors = None, [f"nieoczekiwany format odpowiedzi ({type(exc).__name__})"]
    if layout is None:
        say("  redakcja odrzucona przez kontrole: " + "; ".join(errors[:4]))
        return deterministic(facts), "deterministyczny: " + errors[0]
    return layout, f"llm:{res.provider}/{res.model}"


def validate_layout(wiedza: dict) -> list[str]:
    """Kontrola ukladu w pliku (uzywana przez plantfile.validate_candidate)."""
    errors: list[str] = []
    fact_ids = {f.get("id") for f in wiedza.get("fakty") or []}
    nrs = {z.get("nr") for z in wiedza.get("zrodla") or [] if isinstance(z, dict)}
    for z in wiedza.get("zrodla") or []:
        if not re.match(r"^https?://", str(z.get("url") or "")):
            errors.append(f"zrodlo {z.get('nr')}: zly adres")
    seen: set[str] = set()
    nr_list = [z.get("nr") for z in wiedza.get("zrodla") or [] if isinstance(z, dict)]
    if len(nr_list) != len(set(nr_list)) or not all(isinstance(n, int) for n in nr_list):
        errors.append("zle numery zrodel")
    for s in wiedza.get("sekcje") or []:
        if s.get("klucz") not in SECTIONS:
            errors.append(f"nieznana sekcja {s.get('klucz')}")
        summary = s.get("podsumowanie") or ""
        if len(summary) > MAX_SUMMARY or HTML_RE.search(summary) \
                or not set(s.get("podsumowanie_zrodla") or []) <= nrs:
            errors.append(f"zle podsumowanie sekcji {s.get('klucz')}")
        for p in s.get("punkty") or []:
            if not (20 <= len(p.get("tekst") or "") <= MAX_POINT):
                errors.append("punkt o zlej dlugosci")
            if HTML_RE.search(p.get("tekst") or ""):
                errors.append("znaczniki HTML w punkcie")
            if p.get("czesc") is not None and p.get("czesc") not in PARTS:
                errors.append(f"nieznana czesc rosliny {p.get('czesc')}")
            if not set(p.get("zrodla") or []) <= nrs or not p.get("zrodla"):
                errors.append("punkt bez poprawnego zrodla")
            for fid in p.get("fakty") or []:
                if fid not in fact_ids:
                    errors.append(f"punkt wskazuje nieistniejaca informacje {fid}")
                if fid in seen:
                    errors.append(f"informacja {fid} w dwoch punktach")
                seen.add(fid)
    if wiedza.get("sekcje") is not None and fact_ids - seen:
        errors.append(f"informacje poza ukladem: {len(fact_ids - seen)}")
    return errors
