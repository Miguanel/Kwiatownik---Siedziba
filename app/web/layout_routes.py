"""Widok "Uklad strony" (/uklad): rozdzialy i podrozdzialy strony rosliny, do ktorych agent rozmieszcza wiedze
z sieci (app/knowledge/placement.py). Tu dodaje sie nowe rozdzialy i podrozdzialy (app/knowledge/catalog.py),
zatwierdza podrozdzialy zalozone przez agenta i wylacza wbudowane nowe podrozdzialy."""
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from app.config import settings
from app.knowledge import catalog, placement
from app.knowledge.sections import SECTION_TITLES_PL
from app.web.templating import templates

router = APIRouter()


def _redirect(msg: str = "") -> RedirectResponse:
    return RedirectResponse("/uklad" + (f"?msg={quote(msg)}" if msg else ""), status_code=303)


def _usage() -> tuple[dict[str, int], dict[str, int], int]:
    """Ile informacji stoi w kazdym miejscu (wszystkie pliki roslin), ile roslin ma rozmieszczenie."""
    per_place: dict[str, int] = {}
    per_part = 0
    plants = 0
    folder = Path(settings.kwiatownik_plants_dir)
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        blk = data.get("rozmieszczenie") if isinstance(data, dict) else None
        if not isinstance(blk, dict) or not blk.get("wstawki"):
            continue
        plants += 1
        for w in blk["wstawki"]:
            n = len(w.get("punkty") or [])
            mid = str(w.get("miejsce") or "")
            if mid.startswith("surowce/czesc:"):
                per_part += n
            per_place[mid] = per_place.get(mid, 0) + n
    return per_place, {"surowce/czesc": per_part}, plants


def _view() -> list[dict]:
    data = catalog.load()
    off = catalog.disabled(data)
    custom = {r["id"]: r for r in data["podrozdzialy"]}
    rows: dict[str, list[dict]] = {}
    for pl in placement.MIEJSCA:
        rows.setdefault(pl.chapter_key, []).append({
            "id": pl.id, "tytul": pl.title, "ikona": pl.icon, "rodzaj": "szablon" if pl.builtin else "nowy",
            "wylaczony": pl.id in off, "sekcje": pl.prior, "slowa": list(pl.words), "opis": pl.hint})
    rows.setdefault("surowce", []).insert(0, {
        "id": "surowce/czesc:*", "tytul": "Części rośliny (liście, kwiaty, owoce, korzeń...)", "ikona": "ra-potion",
        "rodzaj": "czesci", "wylaczony": False, "sekcje": placement.PART_PRIOR, "slowa": list(placement.PART_WORDS),
        "opis": "punkt o konkretnej czesci rosliny trafia do jej podrozdzialu; czesc, ktorej nie ma w pliku, dostaje nowy"})
    for pl in catalog.custom_places(data):
        r = custom.get(pl.id, {})
        rows.setdefault(pl.chapter_key, []).append({
            "id": pl.id, "tytul": pl.title, "ikona": pl.icon, "rodzaj": "agent" if pl.source == "agent" else "wlasny",
            "zatwierdzony": r.get("zatwierdzony", True), "wylaczony": False, "sekcje": pl.prior,
            "slowa": list(r.get("slowa") or []), "opis": pl.hint, "utworzono": r.get("utworzono")})
    return [{**ch, "miejsca": rows.get(ch["id"], [])} for ch in catalog.chapters(data)]


@router.get("/uklad")
async def layout_page(request: Request):
    per_place, per_part, n_plants = _usage()
    chapters = _view()
    return templates.TemplateResponse(request, "uklad.html", {
        "chapters": chapters, "per_place": per_place, "per_part": per_part["surowce/czesc"], "n_plants": n_plants,
        "icons": catalog.ICONS, "sections": SECTION_TITLES_PL, "msg": request.query_params.get("msg", ""),
        "pending": sum(1 for c in chapters for m in c["miejsca"] if m["rodzaj"] == "agent" and not m.get("zatwierdzony")),
        "file": str(catalog.path())})


def _sections_from_form(form) -> dict[str, float]:
    out = {}
    for key in SECTION_TITLES_PL:
        v = str(form.get(f"sek_{key}") or "").strip()
        if v:
            try:
                out[key] = float(v)
            except ValueError:
                pass
    return out


def _words(text: str) -> list[str]:
    return [w.strip() for w in str(text or "").replace("\n", ",").split(",") if w.strip()]


@router.post("/uklad/rozdzial")
async def add_chapter(tytul: str = Form(...), ikona: str = Form("ra-leaf"), po: str = Form("tajemna")):
    try:
        row = catalog.add_chapter(tytul, ikona, po)
    except catalog.CatalogError as exc:
        return _redirect(f"Nie dodano rozdzialu: {exc}")
    return _redirect(f"Dodano rozdzial '{row['tytul']}' - dodaj w nim podrozdzialy, zeby agent mial gdzie rozmieszczac")


@router.post("/uklad/podrozdzial")
async def add_subchapter(request: Request):
    form = await request.form()
    try:
        row = catalog.add_subchapter(str(form.get("rozdzial") or ""), str(form.get("tytul") or ""),
                                     str(form.get("ikona") or "ra-leaf"), _sections_from_form(form),
                                     _words(str(form.get("slowa") or "")), str(form.get("opis") or ""))
    except catalog.CatalogError as exc:
        return _redirect(f"Nie dodano podrozdzialu: {exc}")
    return _redirect(f"Dodano podrozdzial '{row['tytul']}' ({row['id']}). Uruchom 'Rozmiesc ponownie', zeby agent go uzyl.")


@router.post("/uklad/podrozdzial/{pid:path}/zmien")
async def edit_subchapter(pid: str, request: Request):
    form = await request.form()
    try:
        catalog.update_subchapter(pid, tytul=str(form.get("tytul") or ""), ikona=str(form.get("ikona") or "ra-leaf"),
                                  sekcje=_sections_from_form(form), slowa=_words(str(form.get("slowa") or "")),
                                  opis=str(form.get("opis") or ""), zatwierdz=bool(form.get("zatwierdz")))
    except catalog.CatalogError as exc:
        return _redirect(f"Nie zapisano: {exc}")
    return _redirect("Zapisano podrozdzial")


@router.post("/uklad/zatwierdz/{pid:path}")
async def approve(pid: str):
    try:
        catalog.update_subchapter(pid, zatwierdz=True)
    except catalog.CatalogError as exc:
        return _redirect(str(exc))
    return _redirect("Zatwierdzono podrozdzial od agenta")


@router.post("/uklad/usun/{item_id:path}")
async def remove(item_id: str):
    done = catalog.remove(item_id)
    return _redirect("Usunieto. Strona pokaze to miejsce do czasu ponownego rozmieszczenia ('Rozmiesc ponownie')."
                     if done else "Nie ma takiego wlasnego wpisu")


@router.post("/uklad/przelacz/{pid:path}")
async def toggle(pid: str, on: str = Form("")):
    if pid not in placement.PLACE_BY_ID or placement.PLACE_BY_ID[pid].builtin:
        return _redirect("Mozna wylaczyc tylko nowe podrozdzialy (te spoza szablonu)")
    catalog.toggle_builtin(pid, bool(on))
    return _redirect(("Wlaczono " if on else "Wylaczono ") + pid)
