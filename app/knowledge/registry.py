"""Rejestr roslin: rosliny z Kwiatownika (pliki + archiwum) oraz nowe rosliny wykryte w przepisach."""
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

from sqlmodel import Session, col, select

from app.knowledge.sections import norm_text
from app.models import Item, ItemKind, ItemStatus, Plant, PlantAlias
from app.pipeline.translate import slugify

log = logging.getLogger(__name__)
LEGACY_DIRS = ("ziola", "drzewa", "krzewy", "bulwy", "cebule", "egzotyczne")

IDENTIFY_PROMPT = """Dostajesz nazwy skladnikow z przepisow zielarskich (po polsku). Dla kazdej ustal, czy chodzi \
o konkretna rosline lub grzyb (gatunek), a jesli tak - podaj polska nazwe gatunku i nazwe lacinska.
Nie-rosliny (cukier, woda, alkohol, miod, wosk, sol, olej bez nazwy rosliny, glinka...) -> roslina=false.
Produkty z konkretnej rosliny (np. "kwiaty czarnego bzu", "olej lniany") -> roslina=true i ten gatunek.
Zwroc TYLKO JSON: {"wyniki": [{"nazwa": "dokladnie jak na wejsciu", "roslina": true, "nazwa_pl": "Bez czarny",
"nazwa_lat": "Sambucus nigra"}]}"""


def _read(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8-sig"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def kwiatownik_plants(plants_dir: Path) -> list[dict]:
    """[{id, nazwa_pl, nazwa_lat, origin}] z plikow Kwiatownika (nowy format) i archiwum (podfoldery)."""
    d = Path(plants_dir)
    out, seen = [], set()
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        data = _read(f)
        if data:
            out.append({"id": f.stem, "nazwa_pl": data.get("nazwa_pl") or data.get("gatunek") or f.stem,
                        "nazwa_lat": data.get("nazwa_lat") or data.get("nazwa_lacinska"),
                        "origin": "nowa" if data.get("_siedziba_nowa") else "kwiatownik"})
            seen.add(f.stem)
    for cat in LEGACY_DIRS:
        for f in sorted((d / cat).glob("*.json")) if (d / cat).exists() else []:
            if f.stem in seen or f.stem[-1].isdigit():
                continue
            data = _read(f)
            if data:
                out.append({"id": f.stem, "nazwa_pl": data.get("gatunek") or f.stem.replace("_", " ").capitalize(),
                            "nazwa_lat": data.get("nazwa_lacinska"), "origin": "archiwum"})
                seen.add(f.stem)
    return out


def sync_kwiatownik(session: Session, plants_dir: Path) -> int:
    """Dopisuje/aktualizuje rosliny z plikow Kwiatownika. Zwraca liczbe nowych wpisow."""
    added = 0
    for p in kwiatownik_plants(plants_dir):
        row = session.get(Plant, p["id"])
        if row is None:
            session.add(Plant(**p))
            added += 1
        else:
            if p["nazwa_lat"] and row.nazwa_lat and _species(p["nazwa_lat"]) != _species(row.nazwa_lat):
                species_changed(session, row, p["nazwa_lat"])
            row.nazwa_pl, row.nazwa_lat = p["nazwa_pl"], p["nazwa_lat"] or row.nazwa_lat
            row.origin = p["origin"]
            session.add(row)
    session.commit()
    from app.config import settings
    merge_duplicate_species(session, plants_dir, Path(settings.backups_dir) / "plants")
    return added


def merge_duplicate_species(session: Session, plants_dir: Path, backup_dir: Path | None = None) -> list[str]:
    """Plik nowej rosliny, ktory zalozyla Siedziba ("_siedziba_nowa"), dla gatunku, ktory ma juz reczny plik
    w Kwiatowniku (np. kasztanowiec.json obok kasztanowiec_zwyczajny.json - Aesculus hippocastanum). Wiedza, zapytania
    i tropy przechodza do recznej rosliny (bez powtorek; zapisane wczesniej informacje wracaja do zapisu - brama
    sprawdzi je na recznym pliku), stare id zostaje aliasem, przepisy Siedziby wskazuja reczny plik, a plik Siedziby
    trafia do kopii zapasowej i znika z Kwiatownika. Reczne pliki nigdy nie sa ruszane. Zwraca opisy polaczen."""
    import shutil
    from datetime import datetime

    from app.knowledge.sections import fingerprint
    from app.models import FbIdea, MagicLead, PlantFact, PlantQuery
    groups: dict[str, list[dict]] = {}
    for p in kwiatownik_plants(plants_dir):
        if p["origin"] in ("kwiatownik", "nowa") and _species(p["nazwa_lat"]).count(" ") == 1:
            groups.setdefault(_species(p["nazwa_lat"]), []).append(p)
    done: list[str] = []
    for species, rows in groups.items():
        mains = [p for p in rows if p["origin"] == "kwiatownik"]
        dups = [p for p in rows if p["origin"] == "nowa"]
        if len(mains) != 1 or not dups:
            continue
        main = mains[0]["id"]
        have = {f.fingerprint for f in session.exec(select(PlantFact).where(PlantFact.plant_id == main)).all()}
        for d in dups:
            dup = d["id"]
            moved = dropped = 0
            for f in session.exec(select(PlantFact).where(PlantFact.plant_id == dup)).all():
                fp = fingerprint(main, f.text)
                if f.status != "rejected" and fp not in have:
                    f.plant_id, f.fingerprint = main, fp
                    f.status = "verified" if f.status in ("verified", "applied") else f.status
                    have.add(fp)
                    moved += 1
                else:
                    f.plant_id, f.fingerprint = main, fp + "~" + str(f.id)       # powtorka: zostaje jako odrzucona
                    if f.status != "rejected":
                        f.status, f.reason = "rejected", f"powtorka po polaczeniu {dup} z {main}"
                    dropped += 1
                session.add(f)
            for model in (PlantQuery, MagicLead, FbIdea):
                for r in session.exec(select(model).where(model.plant_id == dup)).all():
                    r.plant_id = main
                    session.add(r)
            for it in session.exec(select(Item).where(col(Item.data_json).contains(f'"{dup}"'))).all():
                data = json.loads(it.data_json)
                for sk in data.get("skladniki") or []:
                    if isinstance(sk, dict) and sk.get("link_id") == dup:
                        sk["link_id"] = main
                for k in ("roslina_id", "slug"):
                    if data.get(k) == dup:
                        data[k] = main
                it.data_json = json.dumps(data, ensure_ascii=False)
                session.add(it)
            for name in {norm_text(d["nazwa_pl"] or ""), norm_text(dup.replace("_", " "))} - {""}:
                session.merge(PlantAlias(name=name, plant_id=main))
            path = Path(plants_dir) / f"{dup}.json"
            if backup_dir is not None and path.exists():
                dest = Path(backup_dir) / dup
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-polaczona-z-{main}.json")
                path.unlink()
            row = session.get(Plant, dup)
            if row is not None:
                session.delete(row)
            msg = f"{dup} -> {main} ({species}): przeniesiono {moved} informacji, powtorek {dropped}"
            log.warning("Polaczono rosliny: %s", msg)
            done.append(msg)
    session.commit()
    return done


def _species(lat: str | None) -> str:
    return " ".join(str(lat or "").lower().split()[:2])


def species_changed(session: Session, row: Plant, new_lat: str) -> dict:
    """Plik rosliny opisuje teraz INNY gatunek niz wiedza zebrana pod tym id (np. tarnina.json byl kopia poziomki
    i Siedziba zbierala pod "tarnina" wiedze o poziomce). Wiedza starego gatunku przechodzi do rosliny, ktora go
    opisuje (bez powtorek), a dane identyfikacji (Wikidata, tytuly, zdjecia) sa czyszczone - nastepne zbieranie
    zacznie od wlasciwego gatunku. Zwraca {"przeniesiono", "odrzucono"}."""
    from app.models import PlantFact
    old = _species(row.nazwa_lat)
    target = next((p for p in session.exec(select(Plant).where(Plant.id != row.id)).all()
                   if _species(p.nazwa_lat) == old), None)
    have = {f.fingerprint for f in session.exec(select(PlantFact).where(PlantFact.plant_id == target.id)).all()} \
        if target else set()
    out = {"przeniesiono": 0, "odrzucono": 0}
    for f in session.exec(select(PlantFact).where(PlantFact.plant_id == row.id,
                                                  col(PlantFact.status) != "rejected")).all():
        if target is not None and f.fingerprint not in have:
            f.plant_id = target.id
            f.status = "verified" if f.status in ("verified", "applied") else f.status
            have.add(f.fingerprint)
            out["przeniesiono"] += 1
        else:
            f.status, f.reason = "rejected", f"dotyczy innego gatunku ({row.nazwa_lat}), nie {new_lat}"
            out["odrzucono"] += 1
        session.add(f)
    row.wikidata = row.wiki_titles_json = row.names_json = row.photos_json = None
    row.photos_at = row.last_research_at = None
    row.coverage, row.gaps_json, row.status = None, None, "idle"
    row.note = (f"gatunek w pliku zmienil sie z {row.nazwa_lat} na {new_lat}: przeniesiono {out['przeniesiono']} "
                f"informacji do {target.id if target else '-'}, odrzucono {out['odrzucono']}")[:300]
    log.warning("Roslina %s: %s", row.id, row.note)
    return out


def ingredient_candidates(session: Session) -> Counter:
    """Nazwy skladnikow bez powiazania z roslina (link_id) z przetlumaczonych przepisow."""
    rows = session.exec(select(Item.data_json).where(
        Item.kind == ItemKind.recipe, col(Item.data_json).is_not(None),
        col(Item.status).in_([ItemStatus.translated, ItemStatus.approved, ItemStatus.exported]))).all()
    names: Counter = Counter()
    for dj in rows:
        d = json.loads(dj)
        for s in d.get("skladniki") or []:
            if isinstance(s, dict) and s.get("nazwa") and not s.get("link_id"):
                names[_clean_name(s["nazwa"])] += 1
        if d.get("roslina") and not d.get("slug"):
            names[_clean_name(d["roslina"])] += 1
    return Counter({k: v for k, v in names.items() if k and len(k) >= 3})


def _clean_name(name: str) -> str:
    n = re.sub(r"\(.*?\)", " ", str(name)).lower()
    n = re.sub(r"\d+([.,]\d+)?\s*\w*", " ", n)                # ilosci typu "2 lyzki"
    return re.sub(r"\s+", " ", n).strip(" ,.;:-")


async def discover_new_plants(session: Session, llm, batch: int = 50, log_fn=None) -> list[str]:
    """LLM ustala, ktore nieznane skladniki to rosliny -> nowe wpisy Plant(origin='nowa'). Zwraca ich id."""
    say = log_fn or (lambda m: None)
    names = ingredient_candidates(session)
    known = {a.name for a in session.exec(select(PlantAlias)).all()}
    todo = [n for n, _ in names.most_common() if norm_text(n) not in known]
    if not todo:
        return []
    by_lat = {(p.nazwa_lat or "").lower(): p.id for p in session.exec(select(Plant)).all() if p.nazwa_lat}
    by_name = {norm_text(p.nazwa_pl): p.id for p in session.exec(select(Plant)).all()}
    created: list[str] = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = await llm.complete("SKLADNIKI:\n" + "\n".join(f"- {n}" for n in chunk), system=IDENTIFY_PROMPT,
                                 json_mode=True)
        from app.llm.base import InvalidOutputError, json_object
        try:
            data = json_object(res.text)
        except InvalidOutputError as exc:          # zla odpowiedz dla tej paczki - kolejne paczki dalej
            say(f"Skladniki {i + 1}-{i + len(chunk)}: odpowiedz modelu w zlym formacie ({exc}) - pomijam")
            continue
        for r in data.get("wyniki") or []:
            if not isinstance(r, dict):
                continue
            name = _clean_name(r.get("nazwa") or "")
            if name not in chunk:
                continue
            pid = None
            lat = str(r.get("nazwa_lat") or "").strip()
            pl = str(r.get("nazwa_pl") or "").strip()
            if r.get("roslina") is True and pl and re.match(r"^[A-Z][a-z]+ [a-z-]+", lat):
                lat = " ".join(lat.split()[:2])
                pid = by_lat.get(lat.lower()) or by_name.get(norm_text(pl))
                if pid is None:
                    pid = slugify(pl)
                    if session.get(Plant, pid) is None:
                        session.add(Plant(id=pid, nazwa_pl=pl[:1].upper() + pl[1:], nazwa_lat=lat, origin="nowa",
                                          mentions=names[name]))
                        created.append(pid)
                    by_lat[lat.lower()], by_name[norm_text(pl)] = pid, pid
                else:
                    row = session.get(Plant, pid)
                    if row and row.origin == "nowa":
                        row.mentions = (row.mentions or 0) + names[name]
                        session.add(row)
            session.merge(PlantAlias(name=norm_text(name), plant_id=pid))
        session.commit()
        say(f"Skladniki {i + 1}-{i + len(chunk)}: nowe rosliny {len(created)} ({res.provider}/{res.model})")
    return created


def plants_dir_ok(plants_dir: Path) -> bool:
    return Path(plants_dir).exists() and os.access(plants_dir, os.W_OK)
