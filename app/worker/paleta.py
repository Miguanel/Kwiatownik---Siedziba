"""Zadanie "paleta_export": uzupelnienia Palety Barw Zielarza w Kwiatowniku2 (static/json/paleta_siedziba.json).

Bez LLM i bez sieci - same zweryfikowane informacje z bazy (app/knowledge/paleta.py). Wywolywane:
- recznie z panelu /paleta (przycisk) albo jako zadanie (retry_job),
- automatycznie przez Wdrozeniowca przed sprawdzeniem zmian do commitu (refresh_quietly, najczesciej co 10 min),
dzieki czemu plik w repozytorium jest zawsze aktualny, a sama nowa data nie tworzy zmiany.
"""
import logging
import time
from datetime import datetime, timezone

from sqlmodel import Session, col, select

from app.db import engine
from app.knowledge import paleta as pal
from app.models import Plant, PlantFact
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner

log = logging.getLogger(__name__)
_last = {"t": 0.0}
QUIET_EVERY_S = 600


def facts_for(plant_ids: list[str]) -> dict[str, list[dict]]:
    if not plant_ids:
        return {}
    out: dict[str, list[dict]] = {}
    with Session(engine) as s:
        rows = s.exec(select(PlantFact).where(
            col(PlantFact.plant_id).in_(plant_ids), col(PlantFact.status).in_(pal.STATUSY),
            col(PlantFact.section).in_(pal.SEKCJE_UWAGI + pal.SEKCJE_OSTRZEZEN))).all()
        for f in rows:
            out.setdefault(f.plant_id, []).append({
                "id": f.id, "section": f.section, "text": f.text, "source_url": f.source_url,
                "source_name": f.source_name, "language": f.language, "status": f.status})
    return out


def overview() -> dict:
    """Stan palety dla panelu: ziola, dopasowane rosliny, liczby informacji (wszystkich i pasujacych do palety)."""
    repo = pal.repo_dir()
    if not repo:
        return {"repo": None, "herbs": [], "mapping": {}, "plants": {}, "counts": {}, "out": None, "missing": [],
                "all_pids": []}
    herbs = pal.load_herbs(repo)
    with Session(engine) as s:
        plants = s.exec(select(Plant.id, Plant.nazwa_lat)).all()
        names = dict(s.exec(select(Plant.id, Plant.nazwa_pl)).all())
        mapping = pal.match_plants(herbs, list(plants))
        pids = sorted({p for v in mapping.values() for p in v})
        rows = s.exec(select(PlantFact.plant_id, PlantFact.section, PlantFact.status, PlantFact.text,
                             PlantFact.source_url).where(
            col(PlantFact.plant_id).in_(pids),
            col(PlantFact.section).in_(pal.SEKCJE_UWAGI + pal.SEKCJE_OSTRZEZEN))).all() if pids else []
    counts: dict[str, dict] = {}
    for z in herbs:
        c = {"wszystkie": 0, "o_wlosach": 0, "do_weryfikacji": 0, "w_palecie": 0}
        for pid, sec, st, text, url in rows:
            if pid not in mapping.get(z["id"], []):
                continue
            c["wszystkie"] += 1
            if pal.is_hair(text):
                c["o_wlosach"] += 1
                if st == "new":
                    c["do_weryfikacji"] += 1
        counts[z["id"]] = c
    out = None
    try:
        import json
        out = json.loads((repo / pal.WYJSCIE).read_text(encoding="utf-8"))
        for zid, u in (out.get("uzupelnienia") or {}).items():
            if zid in counts:
                counts[zid]["w_palecie"] = len(u.get("uwagi") or []) + len(u.get("ostrzezenia") or [])
    except (OSError, ValueError):
        pass
    return {"repo": str(repo), "herbs": herbs, "mapping": mapping, "plants": names, "counts": counts, "out": out,
            "missing": [z for z in herbs if not mapping.get(z["id"])], "all_pids": pids}


def export_paleta(say=None) -> dict:
    """Przelicza i (gdy cos sie zmienilo) zapisuje paleta_siedziba.json. Zwraca statystyki."""
    say = say or (lambda m: None)
    repo = pal.repo_dir()
    if not repo:
        raise RuntimeError("Nie znaleziono Kwiatownika2 z plikiem static/json/paleta.json "
                           "(DEPLOY_REPO_DIR albo folder nad data/plants)")
    herbs = pal.load_herbs(repo)
    with Session(engine) as s:
        plants = list(s.exec(select(Plant.id, Plant.nazwa_lat)).all())
    mapping = pal.match_plants(herbs, plants)
    pids = sorted({p for v in mapping.values() for p in v})
    facts = facts_for(pids)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data = pal.build(herbs, mapping, facts, stamp)
    changed = pal.write_if_changed(repo, data)
    stats = {"ziola": len(herbs), "z_roslina": sum(1 for v in mapping.values() if v),
             "uzupelnione": len(data["uzupelnienia"]),
             "uwagi": sum(len(u["uwagi"]) for u in data["uzupelnienia"].values()),
             "ostrzezenia": sum(len(u["ostrzezenia"]) for u in data["uzupelnienia"].values()),
             "zmieniony": changed}
    say(f"Ziola palety: {stats['ziola']}, z roslina w Siedzibie: {stats['z_roslina']}, "
        f"uzupelnione: {stats['uzupelnione']} ({stats['uwagi']} uwag, {stats['ostrzezenia']} ostrzezen)")
    brak = data["bez_rosliny_w_siedzibie"]
    if brak:
        say("Bez rosliny w Siedzibie (nie ma skad brac wiedzy): " + ", ".join(brak))
    say(("Zapisano " if changed else "Bez zmian: ") + str(repo / pal.WYJSCIE))
    return stats


def refresh_quietly() -> bool | None:
    """Dla Wdrozeniowca: odswiezenie najczesciej co QUIET_EVERY_S, bledy tylko w logu (nie blokuja commitu)."""
    if time.monotonic() - _last["t"] < QUIET_EVERY_S:
        return None
    _last["t"] = time.monotonic()
    try:
        return export_paleta()["zmieniony"]
    except Exception as exc:  # brak pliku palety albo bazy - Wdrozeniowiec dziala dalej
        log.info("Paleta: pominieto odswiezenie (%s)", exc)
        return None


async def run_paleta_export(job_id: int, runner: JobRunner, llm=None) -> dict:
    activity.set(job_id, "paleta barw", "uzupelnienia z wiedzy o barwieniu wlosow")
    stats = export_paleta(lambda m: job_log(job_id, m))
    _last["t"] = time.monotonic()
    return stats
