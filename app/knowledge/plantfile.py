"""Pliki roslin Kwiatownika: budowa kopii z nowa wiedza, kontrola kopii i bezpieczny zapis.

Wiedza z Siedziby trafia do bloku "wiedza" w pliku rosliny:
    "wiedza": {"zaktualizowano": "2026-09-30",
               "fakty": [{"id", "sekcja", "czesc", "tekst", "jezyk", "zrodlo": {"nazwa", "url"}}],   # surowe
               "zrodla": [{"nazwa", "url"}]}
Po agregatorze (app/knowledge/organize.py) blok ma "wersja": 2, ponumerowane "zrodla", "sekcje" i "czesci"
- to z nich korzysta strona; "fakty" zostaja jako material zrodlowy.
Reczne pola pliku (opis, czesci_rosliny, ciekawostki...) NIE sa zmieniane - kontrola kopii to sprawdza.
Wiedza wmontowana w rozdzialy (app/knowledge/merge.py) trafia do osobnego bloku "scalone" (z kopia
recznego tekstu w "oryginal") - strona Kwiatownika pokazuje ja w miejscu recznego pola.
Dla nowych roslin powstaje nowy plik z podstawowymi polami (+ "wiedza").
"""
import copy
import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path

from app.knowledge import merge, organize, photos
from app.knowledge.sections import SECTIONS, norm_part
from app.worker import snapshots

MAX_FACTS_PER_PLANT = 300
URL_RE = re.compile(r"^https?://[^\s<>\"']+$", re.I)


def plant_path(plants_dir: Path, pid: str) -> Path:
    return Path(plants_dir) / f"{pid}.json"


def read_plant(plants_dir: Path, pid: str) -> tuple[dict | None, bool]:
    """(dane, czy plik mial konce linii CRLF)."""
    path = plant_path(plants_dir, pid)
    if not path.exists() or path.stat().st_size == 0:
        return None, False
    raw = path.read_bytes().decode("utf-8-sig")
    return json.loads(raw), "\r\n" in raw


def skeleton(pid: str, nazwa_pl: str, nazwa_lat: str | None, rodzina: str = "", opis: str = "") -> dict:
    """Minimalny plik nowej rosliny w formacie Kwiatownika (pola, ktorych szablony strony oczekuja)."""
    return {"id": pid, "nazwa_pl": nazwa_pl, "nazwa_lat": nazwa_lat or "", "rodzina": rodzina, "opis": opis,
            "tagi": ["z sieci"], "ciekawostki": [], "ostrzezenia": "", "_siedziba_nowa": True}


def fact_entry(fact) -> dict:
    """PlantFact (baza Siedziby) -> wpis w pliku rosliny."""
    return {"id": f"s{fact.id}", "sekcja": fact.section, "czesc": norm_part(fact.part), "tekst": fact.text,
            "jezyk": fact.language, "zrodlo": {"nazwa": fact.source_name or fact.source_url, "url": fact.source_url}}


def build_candidate(original: dict | None, base: dict, facts: list, today: str | None = None) -> dict:
    """Kopia pliku rosliny z dopisana wiedza. `base` = szkielet dla rosliny bez pliku."""
    cand = copy.deepcopy(original) if original is not None else copy.deepcopy(base)
    wiedza = cand.get("wiedza") if isinstance(cand.get("wiedza"), dict) else {}
    old = [f for f in (wiedza.get("fakty") or []) if isinstance(f, dict)]
    have = {f.get("id") for f in old}
    new = [fact_entry(f) for f in facts if f"s{f.id}" not in have]
    all_facts = (old + new)[-MAX_FACTS_PER_PLANT:]
    seen, sources = set(), []
    for f in all_facts:
        src = f.get("zrodlo") or {}
        if src.get("url") and src["url"] not in seen:
            seen.add(src["url"])
            sources.append({"nazwa": src.get("nazwa") or src["url"], "url": src["url"]})
    cand["wiedza"] = {"zaktualizowano": today or date.today().isoformat(), "fakty": all_facts, "zrodla": sources}
    if original is None and not cand.get("opis"):
        first = next((f["tekst"] for f in all_facts if f.get("sekcja") == "opis"), None)
        cand["opis"] = first or ""
    return cand


def with_layout(cand: dict, layout: dict, how: str, today: str | None = None) -> dict:
    """Wstawia uporzadkowany uklad (organize) do bloku "wiedza" kopii. Surowe "fakty" zostaja."""
    w = cand["wiedza"]
    for f in w.get("fakty") or []:
        f["czesc"] = norm_part(f.get("czesc"))
    cand["wiedza"] = {"wersja": 2, "zaktualizowano": w.get("zaktualizowano") or today or date.today().isoformat(),
                      "uporzadkowano": today or date.today().isoformat(), "uklad": how,
                      "zrodla": layout["zrodla"], "sekcje": layout["sekcje"], "czesci": layout["czesci"],
                      "fakty": w.get("fakty") or []}
    return cand


def photos_owned(original: dict | None) -> bool:
    """Czy pole "url" (galeria zdjec) nalezy do Siedziby: plik bez zdjec albo zdjecia wstawione wczesniej przez nia.
    Reczna galeria (wpisana w Kwiatowniku) nigdy nie jest zmieniana - dostaje tylko opis autorow i licencji."""
    if not original or not original.get("url"):
        return True
    return bool((original.get("zdjecia_wiki") or {}).get("w_url"))


def with_photos(cand: dict, original: dict | None, photos: list[dict], credits: list[dict] | None = None,
                today: str | None = None) -> dict:
    """Zdjecia z Wikipedii/Commons -> "url" {podpis: adres} (tak jak reczne zdjecia Kwiatownika)
    + "zdjecia_wiki" z autorem, licencja i strona pliku (wymog licencji CC BY / BY-SA)."""
    own = photos_owned(original)
    if own and photos:
        cand["url"] = {p["podpis"]: p["url"] for p in photos}
        items = [dict(p) for p in photos]
    else:
        items = [dict(c) for c in credits or []]
    if not items:
        return cand
    cand["zdjecia_wiki"] = {"zaktualizowano": today or date.today().isoformat(), "w_url": bool(own and photos),
                            "zdjecia": items}
    return cand


def validate_photos(cand: dict) -> list[str]:
    errors: list[str] = []
    url = cand.get("url")
    zw = cand.get("zdjecia_wiki")
    if zw is None:
        return errors
    if zw.get("w_url"):
        if not isinstance(url, dict) or not (1 <= len(url) <= 8):
            errors.append("zla galeria zdjec (url)")
        else:
            for k, v in url.items():
                if not (1 <= len(str(k)) <= 80) or re.search(r"[<>\"'`]", str(k)):
                    errors.append(f"zly podpis zdjecia {str(k)[:30]}")
                if not isinstance(v, str) or not photos.UPLOAD_RE.match(v):
                    errors.append(f"zdjecie spoza upload.wikimedia.org: {str(v)[:60]}")
    for z in zw.get("zdjecia") or []:
        if not photos.FILEPAGE_RE.match(str(z.get("plik") or "")):
            errors.append("zdjecie bez strony pliku na Commons")
        if not z.get("licencja") or not z.get("autor"):
            errors.append("zdjecie bez autora lub licencji")
        if z.get("licencja_url") and not URL_RE.match(str(z["licencja_url"])):
            errors.append("zly adres licencji zdjecia")
        if re.search(r"<[a-z/!]", f"{z.get('autor')} {z.get('podpis')}", re.I):
            errors.append("znaczniki HTML w opisie zdjecia")
    return errors


def validate_candidate(original: dict | None, cand: dict, pid: str) -> list[str]:
    """Kontrola kopii przed zapisem (algorytm niezalezny od LLM). Pusta lista = mozna zapisac."""
    errors: list[str] = []
    try:
        json.loads(json.dumps(cand, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        return [f"kopia nie jest poprawnym JSON: {exc}"]
    if original is not None:
        own_photos = photos_owned(original)
        for key, val in original.items():
            if key in ("wiedza", "zdjecia_wiki", "scalone") or (key == "url" and own_photos):
                continue
            if key not in cand:
                errors.append(f"usunieto pole '{key}'")
            elif cand[key] != val:
                errors.append(f"zmieniono reczne pole '{key}'")
        old_ids = {f.get("id") for f in ((original.get("wiedza") or {}).get("fakty") or []) if isinstance(f, dict)}
        cand_facts = (cand.get("wiedza") or {}).get("fakty") or []
        new_ids = {f.get("id") for f in cand_facts}
        if len(cand_facts) < MAX_FACTS_PER_PLANT and not old_ids <= new_ids:
            errors.append("zniknely wczesniej zapisane informacje")
    else:
        for key in ("id", "nazwa_pl", "nazwa_lat", "opis", "tagi"):
            if key not in cand:
                errors.append(f"nowa roslina bez pola '{key}'")
        if not cand.get("nazwa_pl"):
            errors.append("nowa roslina bez nazwy")
    if cand.get("id", pid) != pid:
        errors.append(f"id w pliku ({cand.get('id')}) rozne od nazwy pliku ({pid})")
    ids = set()
    for f in (cand.get("wiedza") or {}).get("fakty") or []:
        fid, text = f.get("id"), f.get("tekst") or ""
        url = (f.get("zrodlo") or {}).get("url") or ""
        if fid in ids:
            errors.append(f"powtorzone id informacji {fid}")
        ids.add(fid)
        if f.get("sekcja") not in SECTIONS:
            errors.append(f"{fid}: nieznana sekcja {f.get('sekcja')}")
        if not (20 <= len(text) <= 600):
            errors.append(f"{fid}: zla dlugosc tekstu")
        if re.search(r"<[a-z/!]", text, re.I):
            errors.append(f"{fid}: znaczniki HTML w tekscie")
        if not URL_RE.match(url):
            errors.append(f"{fid}: brak poprawnego adresu zrodla")
    errors += validate_photos(cand)
    errors += merge.validate_merged(cand)
    if original is not None and not photos_owned(original) and cand.get("url") != original.get("url"):
        errors.append("zmieniono reczna galerie zdjec")
    wiedza = cand.get("wiedza") or {}
    if "sekcje" in wiedza:
        errors += organize.validate_layout(wiedza)
        for z in wiedza.get("zrodla") or []:
            if not URL_RE.match(str(z.get("url") or "")):
                errors.append(f"zrodlo {z.get('nr')}: niepoprawny adres")
    return errors


def diff_summary(original: dict | None, cand: dict) -> str:
    old = len(((original or {}).get("wiedza") or {}).get("fakty") or [])
    w = cand.get("wiedza") or {}
    zw = cand.get("zdjecia_wiki") or {}
    photos_txt = (f", zdjec w galerii {len(cand.get('url') or {})}" if zw.get("w_url")
                  else (f", opisy licencji {len(zw.get('zdjecia') or [])} zdjec" if zw else ""))
    merged = len(((cand.get("scalone") or {}).get("pola") or {}))
    return f"{'nowy plik' if original is None else 'plik uzupelniony'}: informacji {old} -> {len(w.get('fakty') or [])}, " \
           f"zrodel {len(w.get('zrodla') or [])}{photos_txt}" + (f", scalone podrozdzialy {merged}" if merged else "")


def write_plant(plants_dir: Path, pid: str, cand: dict, crlf: bool, backup_dir: Path,
                operation: str = "zapis pliku rosliny") -> Path | None:
    """Kopia zapasowa oryginalu + zapis atomowy (plik tymczasowy -> podmiana). Zwraca sciezke kopii.
    Tresc przed i po trafia do migawek (podglad w zakladce "Na zywo")."""
    path = plant_path(plants_dir, pid)
    before = snapshots.read_text(path)
    backup = None
    if path.exists():
        backup_dir = Path(backup_dir) / pid
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        shutil.copy2(path, backup)
    text = json.dumps(cand, ensure_ascii=False, indent=2) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)
    snapshots.record(path, before, text, f"{operation}: {pid}", link=f"/plants/{pid}")
    return backup
