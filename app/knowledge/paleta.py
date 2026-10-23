"""Paleta Barw Zielarza w Kwiatowniku2 (static/json/paleta.json) - uzupelnienia z wiedzy Siedziby.

Reczny plik palety (ziola do farbowania wlosow, model koloru) zostaje nietkniety. Siedziba laczy ziola palety
z roslinami ze swojej bazy po nazwie lacinskiej, wybiera ZWERYFIKOWANE informacje o farbowaniu i pielegnacji
WLOSOW (sekcje "barwienie" i "kosmetyka", ostrzezenia z "bezpieczenstwo") i zapisuje
static/json/paleta_siedziba.json. Przegladarka (static/js/paleta.js) tylko DOPISUJE z niego uwagi, ostrzezenia
i zrodla do kart ziol - nigdy nie zmienia kolorow, sily barwnika ani proporcji.

Tu sa same funkcje bez bazy (testy: tests/test_paleta.py); zapytania do bazy - app/worker/paleta.py.
"""
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.knowledge.sections import norm_text, similar

PALETA = "static/json/paleta.json"
WYJSCIE = "static/json/paleta_siedziba.json"
SEKCJE_UWAGI = ("barwienie", "kosmetyka")
SEKCJE_OSTRZEZEN = ("bezpieczenstwo",)
STATUSY = ("verified", "applied")
MAX_UWAG = 6
MAX_OSTRZEZEN = 3
MAX_ZRODEL = 8
DLUGOSC = (25, 400)

# informacja dotyczy wlosow (tekst po norm_text: male litery, bez polskich znakow)
_WLOS = re.compile(r"\b(wlos\w*|siw\w*|owlosien\w*|skor\w* glowy|plukank\w*|henn\w*|farbowan\w* glow\w*|brwi|rzes\w*)\b")
# a te mowia o tkaninach - bez wzmianki o wlosach nie pasuja do palety
_TKANINY = re.compile(r"\b(weln\w*|tkanin\w*|jedwab\w*|len|lnu|lnian\w*|bawelni\w*|przedz\w*|garbowan\w*|pisank\w*)\b")


def repo_dir() -> Path | None:
    """Kopia robocza Kwiatownika2 z plikiem palety: DEPLOY_REPO_DIR albo folder nad data/plants."""
    cands = [settings.deploy_repo_dir] if settings.deploy_repo_dir else []
    cands.append(Path(settings.kwiatownik_plants_dir).parent.parent)
    for c in cands:
        if c and (Path(c) / PALETA).is_file():
            return Path(c)
    return None


def load_herbs(repo: Path) -> list[dict]:
    data = json.loads((Path(repo) / PALETA).read_text(encoding="utf-8"))
    return [z for z in data.get("ziola") or [] if isinstance(z, dict) and z.get("id")]


def binomials(lat: str | None) -> set[str]:
    """'Senna italica (Cassia obovata)' -> {'senna italica', 'cassia obovata'}; 'Rheum spp.' -> {'rheum'} (sam rodzaj);
    "Alcea rosea 'Nigra'" -> {'alcea rosea'}; dopiski malymi literami (np. 'wg sprzedawcow') sa pomijane."""
    out = set()
    for m in re.finditer(r"(?<!['\"‘’])\b([A-Z][a-z]+)(?:\s+(?!spp?\.)(?:x\s+)?([a-z][a-z-]{2,}))?", lat or ""):
        genus, species = m.group(1).lower(), m.group(2)
        out.add(f"{genus} {species}" if species else genus)
    return out


def _plant_binomial(lat: str | None) -> str | None:
    words = re.findall(r"[A-Za-z-]+", lat or "")
    if not words:
        return None
    return " ".join(w.lower() for w in words[:2]) if len(words) > 1 and words[1].islower() else words[0].lower()


def match_plants(herbs: list[dict], plants: list[tuple[str, str | None]]) -> dict[str, list[str]]:
    """{id ziola palety: [id roslin Siedziby]} - ten sam gatunek (albo rodzaj, gdy w palecie jest np. 'Rheum spp.').
    Dodatkowo pole 'karta' ziola (id pliku rosliny w Kwiatowniku) zawsze laczy z ta rosline."""
    by_bin: dict[str, list[str]] = {}
    by_genus: dict[str, list[str]] = {}
    ids = {pid for pid, _ in plants}
    for pid, lat in plants:
        b = _plant_binomial(lat)
        if not b:
            continue
        by_bin.setdefault(b, []).append(pid)
        by_genus.setdefault(b.split(" ")[0], []).append(pid)
    out: dict[str, list[str]] = {}
    for z in herbs:
        found: list[str] = []
        for b in binomials(z.get("lat")):
            found += by_bin.get(b, []) if " " in b else by_genus.get(b, [])
        if z.get("karta") in ids:
            found.append(z["karta"])
        out[z["id"]] = sorted(set(found))
    return out


def is_hair(text: str) -> bool:
    t = norm_text(text)
    return bool(_WLOS.search(t))


def fits(fact: dict) -> bool:
    """Informacja nadaje sie do palety: o wlosach, rozsadnej dlugosci, ze zrodlem http(s)."""
    text = (fact.get("text") or "").strip()
    if not (DLUGOSC[0] <= len(text) <= DLUGOSC[1]):
        return False
    if not str(fact.get("source_url") or "").startswith(("https://", "http://")):
        return False
    if fact.get("status") not in STATUSY:
        return False
    if not is_hair(text):
        return False
    t = norm_text(text)
    return not (_TKANINY.search(t) and not re.search(r"\bwlos", t))


def _source(fact: dict) -> dict:
    url = fact["source_url"]
    host = urlparse(url).netloc.removeprefix("www.")
    return {"nazwa": (fact.get("source_name") or host)[:120], "url": url, "jezyk": fact.get("language") or None}


def _znane(herb: dict) -> list[str]:
    return [t for t in [herb.get("dziala"), herb.get("przygotowanie"), *(herb.get("uwagi") or []),
                        *(herb.get("ostrzezenia") or [])] if isinstance(t, str)]


def _pick(facts: list[dict], known: list[str], limit: int) -> list[dict]:
    """Bez powtorzen (podobne zdania z roznych zrodel) i bez tego, co jest juz w recznym pliku palety.
    Najpierw informacje juz w Kwiatowniku (applied), potem najkrotsze (konkretne)."""
    chosen: list[dict] = []
    texts = list(known)
    for f in sorted(facts, key=lambda f: (f.get("status") != "applied", len(f.get("text") or ""), f.get("id") or 0)):
        if any(similar(f["text"], t) >= 0.55 for t in texts):
            continue
        chosen.append(f)
        texts.append(f["text"])
        if len(chosen) >= limit:
            break
    return chosen


def build(herbs: list[dict], mapping: dict[str, list[str]], facts_by_plant: dict[str, list[dict]],
          stamp: str) -> dict:
    """Zawartosc paleta_siedziba.json. facts_by_plant: {id rosliny: [{id, section, text, source_url, source_name,
    language, status}]}."""
    uzup: dict[str, dict] = {}
    brak: list[str] = []
    for z in herbs:
        pids = mapping.get(z["id"]) or []
        if not pids:
            brak.append(z["id"])
            continue
        facts = [f for pid in pids for f in facts_by_plant.get(pid, []) if fits(f)]
        uwagi = _pick([f for f in facts if f.get("section") in SEKCJE_UWAGI], _znane(z), MAX_UWAG)
        ostrz = _pick([f for f in facts if f.get("section") in SEKCJE_OSTRZEZEN], _znane(z), MAX_OSTRZEZEN)
        if not uwagi and not ostrz:
            continue
        zrodla, seen = [], set()
        for f in uwagi + ostrz:
            if f["source_url"] not in seen and len(zrodla) < MAX_ZRODEL:
                seen.add(f["source_url"])
                zrodla.append(_source(f))
        uzup[z["id"]] = {
            "rosliny": pids,
            "uwagi": [f["text"].strip() for f in uwagi],
            "ostrzezenia": [f["text"].strip() for f in ostrz],
            "zrodla": zrodla,
            "fakty": [f.get("id") for f in uwagi + ostrz if f.get("id") is not None],
        }
    return {
        "wersja": 1,
        "opis": "Uzupełnienia Palety Barw z Siedziby Kwiatownika (zadanie 'paleta_export'). Plik nadpisuje Siedziba; "
                "ręczny static/json/paleta.json zostaje nietknięty. Przeglądarka tylko DOPISUJE stąd uwagi, "
                "ostrzeżenia i źródła do istniejących ziół.",
        "zaktualizowano": stamp,
        "uzupelnienia": uzup,
        "nowe_ziola": [],
        "bez_rosliny_w_siedzibie": brak,
    }


def _bez_daty(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "zaktualizowano"}


def write_if_changed(repo: Path, data: dict) -> bool:
    """Zapis atomowy tylko przy zmianie tresci (sama nowa data nie tworzy zmiany do commitu)."""
    path = Path(repo) / WYJSCIE
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if isinstance(old, dict) and _bez_daty(old) == _bez_daty(data):
        return False
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".paleta_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True
