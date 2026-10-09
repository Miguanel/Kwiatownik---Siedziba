"""Uklad strony rosliny: rozdzialy i podrozdzialy, do ktorych agent rozmieszcza wiedze z sieci (placement.py).

Wbudowane miejsca to placement.MIEJSCA i placement.CHAPTERS (odpowiadaja szablonowi strony w Kwiatowniku2).
Wlasne - dodane w Siedzibie (widok "Uklad strony", /uklad) albo zalozone przez agenta, gdy zaden podrozdzial nie
pasowal - sa w pliku data/uklad_strony.json:
  {"wersja": 1,
   "rozdzialy": [{"id": "obrzedy", "tytul": "Obrzędy i rytuały", "ikona": "ra-candle", "po": "tajemna",
                  "zrodlo": "uzytkownik", "utworzono": "2026-10-09"}],
   "podrozdzialy": [{"id": "tajemna/weterynaria", "tytul": "Weterynaria ludowa", "ikona": "ra-wolf-head",
                     "sekcje": {"zastosowanie_lecznicze": 2.0}, "slowa": ["zwierz", "bydl"], "opis": "dla LLM",
                     "zrodlo": "uzytkownik" | "agent", "zatwierdzony": false, "utworzono": "..."}],
   "wylaczone": ["cykl/biologia"]}
Kwiatownik nie czyta tego pliku: kazda wstawka w pliku rosliny niesie tytul i ikone swojego podrozdzialu oraz
rozdzialu (i "po" - miejsce nowego rozdzialu na stronie), wiec nowe rozdzialy pojawiaja sie bez zmian szablonu.
"""
import json
import os
import re
import threading
from datetime import date
from pathlib import Path

from app.config import settings
from app.knowledge.sections import SECTIONS, norm_text

FILE_NAME = "uklad_strony.json"
MAX_AGENT_NEW = 2            # tyle nowych podrozdzialow agent moze zalozyc przy jednej roslinie
_LOCK = threading.Lock()

# ikony RPG Awesome do wyboru w formularzu (Kwiatownik laduje te biblioteke)
ICONS = ("ra-leaf", "ra-flower", "ra-sprout", "ra-pine-tree", "ra-clover", "ra-trefoil", "ra-potion",
         "ra-fizzing-flask", "ra-health", "ra-meat", "ra-hammer", "ra-anvil", "ra-compass", "ra-moon-sun",
         "ra-burning-embers", "ra-cycle", "ra-sickle", "ra-hourglass", "ra-crystal-ball", "ra-quill-ink",
         "ra-scroll-unfurled", "ra-book", "ra-emerald", "ra-candle", "ra-wolf-head", "ra-bird-claw",
         "ra-bee", "ra-wooden-sign", "ra-skull", "ra-bleeding-eye", "ra-lightning-bolt", "ra-globe",
         "ra-feather-wing", "ra-cauldron", "ra-tower")


def path() -> Path:
    return Path(settings.data_dir) / FILE_NAME


def slug(text: str) -> str:
    s = norm_text(text).replace(" ", "_")
    return re.sub(r"_+", "_", s).strip("_")[:40] or "nowy"


def load() -> dict:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"wersja": 1,
            "rozdzialy": [r for r in data.get("rozdzialy") or [] if isinstance(r, dict) and r.get("id")],
            "podrozdzialy": [r for r in data.get("podrozdzialy") or [] if isinstance(r, dict) and r.get("id")],
            "wylaczone": [str(x) for x in data.get("wylaczone") or []]}


def save(data: dict) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------ odczyt dla algorytmu
def chapters(data: dict | None = None) -> list[dict]:
    """Rozdzialy w kolejnosci strony: wbudowane + wlasne wstawione po rozdziale "po"."""
    from app.knowledge.placement import CHAPTER_ICONS, CHAPTERS
    data = data or load()
    out = [{"id": k, "tytul": v, "ikona": CHAPTER_ICONS.get(k, "ra-leaf"), "wbudowany": True, "po": None}
           for k, v in CHAPTERS.items()]
    for r in data["rozdzialy"]:
        if any(c["id"] == r["id"] for c in out):
            continue
        row = {"id": r["id"], "tytul": r.get("tytul") or r["id"], "ikona": r.get("ikona") or "ra-leaf",
               "wbudowany": False, "po": r.get("po"), "zrodlo": r.get("zrodlo"), "utworzono": r.get("utworzono")}
        idx = next((i for i, c in enumerate(out) if c["id"] == r.get("po")), len(out) - 1)
        while idx + 1 < len(out) and not out[idx + 1]["wbudowany"] and out[idx + 1].get("po") == r.get("po"):
            idx += 1                                   # kilka wlasnych po tym samym rozdziale - w kolejnosci dodania
        out.insert(idx + 1, row)
    return out


def chapter(cid: str, data: dict | None = None) -> dict | None:
    return next((c for c in chapters(data) if c["id"] == cid), None)


def custom_places(data: dict | None = None) -> list:
    """Wlasne podrozdzialy jako placement.Place (tylko w istniejacych rozdzialach)."""
    from app.knowledge.placement import Place
    data = data or load()
    chaps = {c["id"]: c for c in chapters(data)}
    out = []
    for r in data["podrozdzialy"]:
        cid = r["id"].split("/", 1)[0]
        ch = chaps.get(cid)
        if ch is None:
            continue
        prior = {k: float(v) for k, v in (r.get("sekcje") or {}).items() if k in SECTIONS}
        words = tuple(norm_text(w) for w in r.get("slowa") or [] if norm_text(w))
        out.append(Place(r["id"], r.get("tytul") or r["id"], r.get("ikona") or "ra-leaf", False, prior, words,
                         r.get("opis") or "", chapter_title=ch["tytul"], chapter_icon=ch["ikona"],
                         chapter_after=ch.get("po") or "", source=r.get("zrodlo") or "uzytkownik"))
    return out


def disabled(data: dict | None = None) -> set[str]:
    return set((data or load())["wylaczone"])


# ------------------------------------------------------------------ zmiany (formularze /uklad i agent)
class CatalogError(ValueError):
    pass


def _title(t: str) -> str:
    t = re.sub(r"\s+", " ", str(t or "")).strip()
    if not (3 <= len(t) <= 60) or re.search(r"[<>{}]", t):
        raise CatalogError("tytul: 3-60 znakow, bez < > { }")
    return t


def _icon(i: str) -> str:
    return i if i in ICONS else "ra-leaf"


def add_chapter(tytul: str, ikona: str = "ra-leaf", po: str = "tajemna", zrodlo: str = "uzytkownik") -> dict:
    with _LOCK:
        data = load()
        t = _title(tytul)
        known = {c["id"] for c in chapters(data)}
        cid = slug(t)
        if cid in known or any(norm_text(c["tytul"]) == norm_text(t) for c in chapters(data)):
            raise CatalogError(f"rozdzial '{t}' juz jest")
        if po not in known:
            po = "tajemna"
        row = {"id": cid, "tytul": t, "ikona": _icon(ikona), "po": po, "zrodlo": zrodlo,
               "utworzono": date.today().isoformat()}
        data["rozdzialy"].append(row)
        save(data)
        return row


def add_subchapter(rozdzial: str, tytul: str, ikona: str = "ra-leaf", sekcje: dict | None = None,
                   slowa: list[str] | None = None, opis: str = "", zrodlo: str = "uzytkownik") -> dict:
    from app.knowledge.placement import MIEJSCA
    with _LOCK:
        data = load()
        if chapter(rozdzial, data) is None:
            raise CatalogError(f"nie ma rozdzialu '{rozdzial}'")
        t = _title(tytul)
        same = [p for p in list(MIEJSCA) + custom_places(data) if p.chapter_key == rozdzial
                and norm_text(p.title) == norm_text(t)]
        if same:
            raise CatalogError(f"podrozdzial '{t}' juz jest w tym rozdziale")
        pid = f"{rozdzial}/{slug(t)}"
        taken = {p.id for p in MIEJSCA} | {r["id"] for r in data["podrozdzialy"]}
        n = 2
        while pid in taken:
            pid = f"{rozdzial}/{slug(t)}_{n}"
            n += 1
        sek = {k: max(0.0, min(5.0, float(v))) for k, v in (sekcje or {}).items() if k in SECTIONS}
        words = [w.strip() for w in (slowa or []) if 2 <= len(w.strip()) <= 30][:30]
        row = {"id": pid, "tytul": t, "ikona": _icon(ikona), "sekcje": sek, "slowa": words,
               "opis": str(opis or "")[:200], "zrodlo": zrodlo, "zatwierdzony": zrodlo != "agent",
               "utworzono": date.today().isoformat()}
        data["podrozdzialy"].append(row)
        save(data)
        return row


def update_subchapter(pid: str, **changes) -> dict:
    with _LOCK:
        data = load()
        row = next((r for r in data["podrozdzialy"] if r["id"] == pid), None)
        if row is None:
            raise CatalogError("nie ma takiego wlasnego podrozdzialu")
        if "tytul" in changes:
            row["tytul"] = _title(changes["tytul"])
        if "ikona" in changes:
            row["ikona"] = _icon(changes["ikona"])
        if "sekcje" in changes:
            row["sekcje"] = {k: max(0.0, min(5.0, float(v))) for k, v in (changes["sekcje"] or {}).items() if k in SECTIONS}
        if "slowa" in changes:
            row["slowa"] = [w.strip() for w in changes["slowa"] or [] if 2 <= len(w.strip()) <= 30][:30]
        if "opis" in changes:
            row["opis"] = str(changes["opis"] or "")[:200]
        if changes.get("zatwierdz"):
            row["zatwierdzony"] = True
        save(data)
        return row


def remove(item_id: str) -> bool:
    """Usuwa wlasny rozdzial (razem z jego podrozdzialami) albo wlasny podrozdzial."""
    with _LOCK:
        data = load()
        before = len(data["rozdzialy"]) + len(data["podrozdzialy"])
        data["rozdzialy"] = [r for r in data["rozdzialy"] if r["id"] != item_id]
        data["podrozdzialy"] = [r for r in data["podrozdzialy"]
                                if r["id"] != item_id and r["id"].split("/", 1)[0] != item_id]
        if len(data["rozdzialy"]) + len(data["podrozdzialy"]) == before:
            return False
        save(data)
        return True


def toggle_builtin(pid: str, on: bool) -> None:
    """Wylacza/wlacza wbudowany NOWY podrozdzial (np. gdy nie chcemy "Pasza i gospodarstwo")."""
    with _LOCK:
        data = load()
        off = set(data["wylaczone"])
        off.discard(pid) if on else off.add(pid)
        data["wylaczone"] = sorted(off)
        save(data)
