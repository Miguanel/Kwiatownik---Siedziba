"""Zadanie 'import_k1': przepisy lecznicze z archiwum Kwiatownika 1 -> Siedziba (bez tlumaczenia).

Kazdy przepis staje sie Itemem zrodla "Kwiatownik 1 (archiwum)", przechodzi filtr (kulinarne / bez roslin),
sprawdzenie, czy juz jest w Kwiatowniku2, i trafia do zatwierdzenia. Opracowanie: osobne zadanie "Opracuj".
"""
import json
from pathlib import Path

from sqlmodel import Session, select

from app.config import settings
from app.llm.router import AllModelsFailedError
from app.models import Item, ItemKind, ItemStatus, Source
from app.pipeline.archive import convert_k1_recipe
from app.pipeline.catalog import load_kwiatownik_recipes, load_plants
from app.pipeline.matching import Candidate
from app.worker import activity
from app.worker import export as exp
from app.worker.jobs import job_log
from app.worker.runner import JobRunner

ARCHIVE_NAME = "Kwiatownik 1 (archiwum)"
ARCHIVE_URL = "archiwum://kwiatownik1/"


def archive_files(folder: Path) -> list[Path]:
    """Pliki z przepisami niekulinarnymi (kulinarne z archiwum sa juz w Kwiatowniku2 i nie sa celem)."""
    folder = Path(folder)
    return sorted(f for f in folder.glob("*.json") if "kulinar" not in f.name.lower()) if folder.exists() else []


def _source_id() -> int:
    with Session(exp.engine) as s:
        src = s.exec(select(Source).where(Source.base_url == ARCHIVE_URL)).first()
        if not src:
            src = Source(name=ARCHIVE_NAME, base_url=ARCHIVE_URL, language="pl", country="pl", active=False)
            s.add(src)
            s.commit()
        return src.id


async def run_import_k1(job_id: int, runner: JobRunner, llm, limit: int | None = None) -> dict:
    log = lambda m: job_log(job_id, m)  # noqa: E731
    files = archive_files(settings.kwiatownik1_przepisy_dir)
    if not files:
        raise RuntimeError(f"Brak plikow archiwum w {settings.kwiatownik1_przepisy_dir} "
                           "(w Dockerze: wolumen ../Kwiatownik/static/data/przepisy)")
    sid = _source_id()
    plants = load_plants(settings.kwiatownik_plants_dir)
    kw = load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir, exclude={settings.export_filename})
    kw_cands = [Candidate("kwiatownik", r.ref, r.title, r.ingredients) for r in kw]
    taken = {r.rid for r in load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir) if r.rid}
    stats = {"imported": 0, "exists": 0, "skipped": 0, "known": 0, "invalid": 0, "merged": 0, "translated": 0,
             "culinary": 0}
    todo = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        recs = data.get("przepisy") if isinstance(data, dict) else data
        todo += [(f.stem, i, r) for i, r in enumerate(recs or []) if isinstance(r, dict)]
    if limit:
        todo = todo[:limit]
    log(f"Archiwum: {len(files)} plikow, {len(todo)} przepisow; Kwiatownik2: {len(kw_cands)} przepisow")
    with Session(exp.engine) as s:
        known = set(s.exec(select(Item.url).where(Item.source_id == sid)).all())
    for n, (stem, i, r) in enumerate(todo, 1):
        if runner.should_stop(job_id):
            log("Zatrzymano")
            break
        url = f"{ARCHIVE_URL}{stem}/{i}"
        if url in known:
            stats["known"] += 1
            continue
        rec = convert_k1_recipe(r, plants, taken, f"{stem}_{i}")
        activity.set(job_id, f"importuje {n}/{len(todo)}", (r.get("tytul") or "")[:80])
        if rec is None:
            stats["invalid"] += 1
            log(f"[niekompletny] {r.get('tytul') or url}")
            continue
        taken.add(rec["id"])
        with Session(exp.engine) as s:
            item = Item(source_id=sid, url=url, kind=ItemKind.recipe, status=ItemStatus.raw, title=rec["tytul"],
                        language="pl", method="archiwum", confidence=1.0, reason="import z Kwiatownika 1",
                        structured_json=json.dumps(r, ensure_ascii=False))
            s.add(item)
            s.commit()
            iid = item.id
        try:
            outcome = await exp.finish_item(iid, rec, "archiwum (bez LLM)", llm, kw_cands, log,
                                            enrich=False, only_kwiatownik=True)
        except AllModelsFailedError:  # sedzia podobienstwa niedostepny - zostaw bez sprawdzenia LLM
            outcome = await exp.finish_item(iid, rec, "archiwum (bez LLM)", None, kw_cands, log,
                                            enrich=False, only_kwiatownik=True)
        stats["imported"] += 1
        stats[outcome] = stats.get(outcome, 0) + 1
        exp._progress(job_id, n, len(todo))
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in stats.items() if v))
    log("Dalej: 'Zadania do zrobienia' -> 'Do opracowania' (5 przemian, Kampo...) i 'Do zatwierdzenia'.")
    exp._save_stats(job_id, stats)
    return stats
