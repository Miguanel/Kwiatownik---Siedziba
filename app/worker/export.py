"""Zadania 'translate', 'enrich' i 'export':
tlumaczenie na polski -> pominiecie kulinarnych -> sprawdzenie, czy przepis juz istnieje -> opracowanie
(tradycyjne systemy) -> eksport."""
import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.exporters import store
from app.exporters.kwiatownik import source_entry, with_sources
from app.llm.base import LLMError
from app.llm.router import AllModelsFailedError
from app.models import Item, ItemKind, ItemStatus, Job, Source
from app.pipeline.catalog import load_kwiatownik_recipes, load_plants
from app.pipeline.matching import Candidate, find_match
from app.pipeline.enrich import enrich_recipe, parse_systems, strip_invented
from app.llm.base import InvalidOutputError
from app.mt.nllb import MTQualityError, MTUnavailable, get_translator
from app.pipeline.fast_translate import can_fast_translate, fast_translate_recipe
from app.pipeline.translate import NoRecipeError, skip_reason, translate_recipe
from app.worker import activity, snapshots
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


def _progress(job_id: int, done: int, total: int) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.progress, job.total = done, total
        s.add(job)
        s.commit()


def _kw_candidates() -> list[Candidate]:
    kw = load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir, exclude={settings.export_filename})
    return [Candidate("kwiatownik", r.ref, r.title, r.ingredients) for r in kw]


def _siedziba_candidates(s: Session, exclude_id: int) -> list[Candidate]:
    rows = s.exec(select(Item.id, Item.data_json).where(
        Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None), col(Item.data_json).is_not(None),
        Item.id != exclude_id, Item.status != ItemStatus.skipped)).all()
    out = []
    for iid, dj in rows:
        d = json.loads(dj)
        out.append(Candidate("siedziba", str(iid), d.get("tytul", ""), [x.get("nazwa", "") for x in d.get("skladniki", [])]))
    return out


def _taken_ids(kw_ids: set[str]) -> set[str]:
    with Session(engine) as s:
        ours = {json.loads(d).get("id") for d in s.exec(select(Item.data_json).where(col(Item.data_json).is_not(None))).all()}
    return kw_ids | {i for i in ours if i}


async def translate_one(item_id: int, llm, plants: dict, kw_cands: list[Candidate], kw_ids: set[str], log) -> str:
    with Session(engine) as s:
        item = s.get(Item, item_id)
        src = s.get(Source, item.source_id) if item.source_id else None
        recipe = json.loads(item.structured_json) if item.structured_json else None
        meta = {"item_id": item.id, "title": item.title, "language": item.language or (src.language if src else None),
                "country": src.country if src else None}
        raw = item.raw_text
        ignore_match = item.match_info == "ignored"
    rec = None
    mt = get_translator()
    if mt is not None and can_fast_translate(recipe, meta["language"]):
        # szybka sciezka bez LLM: struktura ze strony + NLLB + reguly
        try:
            rec, model = await fast_translate_recipe(mt, llm, recipe, meta, plants, _taken_ids(kw_ids))
        except (MTUnavailable, MTQualityError, InvalidOutputError) as exc:
            if not settings.mt_fallback_llm or llm is None:
                with Session(engine) as s:
                    item = s.get(Item, item_id)
                    item.status, item.reason = ItemStatus.error, f"tlumaczenie maszynowe: {exc}"[:300]
                    s.add(item)
                    s.commit()
                log(f"[blad MT] #{item_id} {meta['title'][:50]}: {exc}")
                return "error"
            log(f"[MT -> LLM] #{item_id} {meta['title'][:50]}: {str(exc)[:120]}")
    if rec is None and llm is None:
        log(f"[odlozone] #{item_id} {meta['title'][:50]}: brak danych strukturalnych dla tlumacza NLLB i brak LLM")
        return "postponed"
    try:
        if rec is None:
            rec, model = await translate_recipe(llm, recipe, raw, meta, plants, _taken_ids(kw_ids))
    except AllModelsFailedError as exc:
        # zaden model nie byl dostepny (limity) - to nie wina przepisu: zostaje "do tlumaczenia"
        log(f"[odlozone] #{item_id} {meta['title'][:50]}: wszystkie modele maja teraz limit ({str(exc)[:160]})")
        return "postponed"
    except NoRecipeError as exc:
        # lista przepisow / artykul ogolny - nie blad, tylko strona bez konkretnego przepisu
        with Session(engine) as s:
            item = s.get(Item, item_id)
            item.status, item.reason = ItemStatus.skipped, f"brak konkretnego przepisu: {exc}"[:300]
            s.add(item)
            s.commit()
        log(f"[pominiety] #{item_id} {meta['title'][:50]}: brak konkretnego przepisu")
        return "no_recipe"
    except (LLMError, ValueError) as exc:
        with Session(engine) as s:
            item = s.get(Item, item_id)
            item.status, item.reason = ItemStatus.error, f"tlumaczenie: {exc}"[:300]
            s.add(item)
            s.commit()
        log(f"[blad] #{item_id} {meta['title'][:50]}: {exc}")
        return "error"

    extras = rec.pop("_dodatkowe", [])
    outcome = await finish_item(item_id, rec, model, llm, kw_cands, log, raw=raw, ignore_match=ignore_match)
    for n, extra in enumerate(extras, 2):   # kolejne przepisy z tego samego artykulu -> osobne pozycje
        child_id = _child_item(item_id, n, extra)
        if child_id:
            await finish_item(child_id, extra, model, llm, kw_cands, log, raw=None)
    if extras:
        log(f"[podzial] #{item_id}: z artykulu wydzielono {len(extras) + 1} przepisow")
    return outcome


def _child_item(parent_id: int, n: int, rec: dict) -> int | None:
    """Osobny Item dla n-tego przepisu z artykulu (ten sam adres z kotwica #przepis-n)."""
    with Session(engine) as s:
        parent = s.get(Item, parent_id)
        url = f"{parent.url}#przepis-{n}"
        existing = s.exec(select(Item).where(Item.url == url)).first()
        if existing:
            return existing.id if existing.status in (ItemStatus.raw, ItemStatus.error) else None
        child = Item(source_id=parent.source_id, url=url, kind=ItemKind.recipe, status=ItemStatus.raw,
                     title=rec.get("tytul") or parent.title, language=parent.language, method="podzial-artykulu",
                     confidence=parent.confidence, reason=f"przepis {n} z artykulu #{parent_id}")
        s.add(child)
        s.commit()
        return child.id


async def finish_item(item_id: int, rec: dict, model: str, llm, kw_cands: list[Candidate], log, raw: str | None = None,
                      ignore_match: bool = False, enrich: bool | None = None, only_kwiatownik: bool = False) -> str:
    """Wspolny koniec dla przepisu po polsku (z tlumaczenia albo z archiwum): filtr -> duplikaty -> opracowanie."""
    why = skip_reason(rec, settings.skip_culinary, settings.require_key_plants)
    rec.pop("ziola_kluczowe", None)
    if why:
        with Session(engine) as s:
            item = s.get(Item, item_id)
            new = json.dumps(rec, ensure_ascii=False)
            snapshots.record_item(item, new, f"tlumaczenie ({model}) - pominiety: {why}")
            item.data_json = new
            item.translated_by = model
            item.status, item.reason = ItemStatus.skipped, why
            item.updated_at = datetime.now(timezone.utc)
            s.add(item)
            s.commit()
        log(f"[pominiety] {rec['tytul']}: {why}")
        return "culinary"

    names = [x["nazwa"] for x in rec["skladniki"]]
    with Session(engine) as s:
        cands = kw_cands + ([] if only_kwiatownik else _siedziba_candidates(s, item_id))
    match = None if ignore_match else await find_match(rec["tytul"], names, cands, llm)

    msg = f"[przetlumaczono] {rec['tytul']} ({model})"
    if match is None and llm is not None and (settings.enrich_recipes if enrich is None else enrich):
        rec, note = await _enrich(llm, rec, raw)
        msg += note
    with Session(engine) as s:
        item = s.get(Item, item_id)
        new = json.dumps(rec, ensure_ascii=False)
        snapshots.record_item(item, new, f"tlumaczenie ({model})" + (" + opracowanie" if "opracowanie" in msg else ""))
        item.data_json = new
        item.translated_by = model
        item.updated_at = datetime.now(timezone.utc)
        outcome = "translated"
        item.status = ItemStatus.translated
        if match and match.candidate.kind == "kwiatownik":
            item.status, outcome = ItemStatus.exists, "exists"
            item.match_ref = match.candidate.ref
            item.match_info = f"{match.how} (tytul {match.t}, skladniki {match.i}) {match.reason}".strip()
            msg = f"[juz jest w Kwiatowniku] {rec['tytul']} = {match.candidate.title} ({match.candidate.ref})"
        elif match:
            primary = int(match.candidate.ref)
            item.duplicate_of = primary
            item.match_ref = f"item#{primary}"
            item.match_info = f"{match.how} (tytul {match.t}, skladniki {match.i}) {match.reason}".strip()
            for child in s.exec(select(Item).where(Item.duplicate_of == item.id)).all():
                child.duplicate_of = primary
                s.add(child)
            outcome = "merged"
            msg = (f"[ten sam przepis z innej strony] {rec['tytul']} -> #{primary} {match.candidate.title} "
                   f"(dodane jako zrodlo)")
        s.add(item)
        s.commit()
    log(msg)  # dopiero po commit - log pisze w osobnej sesji, a SQLite pozwala tylko na jednego piszacego
    return outcome


async def _enrich(llm, rec: dict, raw: str | None) -> tuple[dict, str]:
    """Opracowanie w tradycyjnych systemach; blad nie psuje tlumaczenia - przepis zostaje bez opracowania."""
    try:
        rec2, model = await enrich_recipe(llm, rec, parse_systems(settings.enrich_systems), raw)
        return rec2, f" + opracowanie ({model})"
    except AllModelsFailedError:
        return rec, " (bez opracowania - modele maja limit; uzyj 'Opracuj' pozniej)"
    except (LLMError, ValueError) as exc:
        return rec, f" (bez opracowania: {str(exc)[:120]})"


async def enrich_one(item_id: int, llm, log) -> str:
    with Session(engine) as s:
        item = s.get(Item, item_id)
        if not item or not item.data_json:
            return "skipped"
        rec, raw = json.loads(item.data_json), item.raw_text
    try:
        rec2, model = await enrich_recipe(llm, rec, parse_systems(settings.enrich_systems), raw)
    except AllModelsFailedError as exc:
        log(f"[odlozone] #{item_id} {rec.get('tytul', '')[:50]}: modele maja limit ({str(exc)[:120]})")
        return "postponed"
    except (LLMError, ValueError) as exc:
        log(f"[blad opracowania] #{item_id} {rec.get('tytul', '')[:50]}: {exc}")
        return "error"
    with Session(engine) as s:
        item = s.get(Item, item_id)
        new = json.dumps(rec2, ensure_ascii=False)
        snapshots.record_item(item, new, f"opracowanie ({model})")
        item.data_json = new
        item.updated_at = datetime.now(timezone.utc)
        s.add(item)
        s.commit()
    log(f"[opracowano] {rec2.get('tytul')} ({model})")
    return "enriched"


def items_without_enrichment(limit: int | None = None, session: Session | None = None) -> list[int]:
    ok = [ItemStatus.translated, ItemStatus.approved, ItemStatus.exported]
    with (Session(engine) if session is None else nullcontext(session)) as s:
        rows = s.exec(select(Item.id, Item.data_json).where(
            Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None), col(Item.data_json).is_not(None),
            col(Item.status).in_(ok)).order_by(Item.id)).all()
    ids = [iid for iid, dj in rows if "opracowanie" not in json.loads(dj)]
    return ids[:limit] if limit else ids


async def run_enrich(job_id: int, runner: JobRunner, llm, limit: int | None = None,
                     item_ids: list[int] | None = None) -> dict:
    if llm is None:
        raise RuntimeError("Opracowanie wymaga LLM - ustaw klucze GEMINI/GROQ w .env")
    ids = list(item_ids) if item_ids else items_without_enrichment(limit or settings.translate_batch)
    log = lambda m: job_log(job_id, m)  # noqa: E731
    log(f"Do opracowania: {len(ids)} przepisow; systemy: {', '.join(parse_systems(settings.enrich_systems))}")
    stats = {"enriched": 0, "error": 0, "postponed": 0, "skipped": 0}
    postponed_in_row = 0
    for n, iid in enumerate(ids, 1):
        if runner.should_stop(job_id):
            log("Zatrzymano")
            break
        if postponed_in_row >= 2:
            log(f"Limity modeli wyczerpane - przerywam; {len(ids) - n + 1} przepisow czeka.")
            break
        activity.set(job_id, f"opracowuje {n}/{len(ids)}", f"#{iid}")
        outcome = await enrich_one(iid, llm, log)
        stats[outcome] += 1
        postponed_in_row = postponed_in_row + 1 if outcome == "postponed" else 0
        _progress(job_id, n, len(ids))
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    _save_stats(job_id, stats)
    return stats


async def run_translate(job_id: int, runner: JobRunner, llm, limit: int | None = None,
                        item_ids: list[int] | None = None) -> dict:
    mt = get_translator()
    if llm is None and not (mt and mt.model_present()):
        raise RuntimeError("Tlumaczenie wymaga LLM (klucze GEMINI/GROQ w .env) albo modelu NLLB (MT_ENGINE=nllb)")
    with Session(engine) as s:
        if item_ids:
            ids = list(item_ids)
        else:
            ids = list(s.exec(select(Item.id).where(
                Item.kind == ItemKind.recipe, Item.status == ItemStatus.raw, col(Item.duplicate_of).is_(None))
                .order_by(col(Item.confidence).desc(), col(Item.id)).limit(limit or settings.translate_batch)).all())
    log = lambda m: job_log(job_id, m)  # noqa: E731
    plants = load_plants(settings.kwiatownik_plants_dir)
    kw_recipes = load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir, exclude={settings.export_filename})
    kw_cands = [Candidate("kwiatownik", r.ref, r.title, r.ingredients) for r in kw_recipes]
    kw_ids = {r.rid for r in load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir) if r.rid}
    log(f"Do tlumaczenia: {len(ids)} przepisow; Kwiatownik: {len(kw_cands)} przepisow, {len(plants)} roslin")
    if mt is None:
        log("Tlumacz: LLM (MT_ENGINE=llm)")
    elif mt.available():
        log(f"Tlumacz: NLLB ({mt.model_dir.name}, {mt.device}) dla przepisow z danymi strukturalnymi; "
            "surowe artykuly - LLM")
    else:
        log(f"Tlumacz NLLB niedostepny ({mt.error or 'brak modelu - docker compose --profile mt run --rm nllb-init'})"
            " - tlumaczy LLM")
    stats = {"translated": 0, "exists": 0, "merged": 0, "error": 0, "postponed": 0, "culinary": 0, "no_recipe": 0}
    postponed_in_row = 0
    for n, iid in enumerate(ids, 1):
        if runner.should_stop(job_id):
            log("Zatrzymano")
            break
        if postponed_in_row >= 2:
            log(f"Limity modeli wyczerpane - przerywam; {len(ids) - n + 1} przepisow czeka. Ponow zadanie pozniej.")
            break
        with Session(engine) as s:
            it = s.get(Item, iid)
            title = (it.title or it.url) if it else f"#{iid}"
        activity.set(job_id, f"tlumacze i sprawdzam {n}/{len(ids)}", title)
        outcome = await translate_one(iid, llm, plants, kw_cands, kw_ids, log)
        stats[outcome] += 1
        postponed_in_row = postponed_in_row + 1 if outcome == "postponed" else 0
        _progress(job_id, n, len(ids))
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    _save_stats(job_id, stats)
    return stats


def recheck_filters() -> int:
    """Jednorazowo po zmianie profilu zbierania: przepisy zebrane zanim dzialal filtr kulinarny.
    Przetlumaczone/zatwierdzone/wyeksportowane z typem "kulinarne" oraz surowe, wyraznie kulinarne -> pominiete
    (kolejny eksport usunie je z pliku Kwiatownika)."""
    if not settings.skip_culinary:
        return 0
    from app.scrapers.classify import CULINARY_REASON, is_culinary
    from app.scrapers.page import PageData
    changed = 0
    with Session(engine) as s:
        rows = s.exec(select(Item).where(Item.kind == ItemKind.recipe, col(Item.status).in_(
            [ItemStatus.raw, ItemStatus.translated, ItemStatus.approved, ItemStatus.exported]))).all()
        for it in rows:
            culinary = False
            if it.data_json:
                culinary = skip_reason(json.loads(it.data_json), True, False) is not None
            elif it.status == ItemStatus.raw and it.method != "archiwum":
                st = json.loads(it.structured_json) if it.structured_json else {}
                page = PageData(url=it.url, canonical=None, title=it.title or "", lang=it.language,
                                text=(it.raw_text or "")[:3000], headings=[], list_items=[], jsonld=[], links=[])
                node = {"name": st.get("title"), "recipeCategory": st.get("category"), "keywords": st.get("keywords")}
                culinary = is_culinary(page, node)
            if culinary:
                it.status, it.reason = ItemStatus.skipped, CULINARY_REASON
                s.add(it)
                changed += 1
        s.commit()
    return changed


def build_export_records() -> tuple[list[dict], list[int]]:
    ok = [ItemStatus.approved, ItemStatus.exported] + ([] if settings.export_require_approval else [ItemStatus.translated])
    with Session(engine) as s:
        names = {x.id: x.name for x in s.exec(select(Source)).all()}
        primaries = s.exec(select(Item).where(Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None),
                                              col(Item.data_json).is_not(None), col(Item.status).in_(ok))
                           .order_by(Item.id)).all()
        records, ids = [], []
        for p in primaries:
            group = [p] + list(s.exec(select(Item).where(Item.duplicate_of == p.id).order_by(Item.id)).all())
            srcs = [source_entry(names.get(g.source_id, "?"), g.url, g.title, g.language, g.created_at) for g in group]
            records.append(with_sources(strip_invented(json.loads(p.data_json)), srcs))
            ids.append(p.id)
    return records, ids


async def run_export(job_id: int, runner: JobRunner, llm, translate_first: int = 0) -> dict:
    log = lambda m: job_log(job_id, m)  # noqa: E731
    if translate_first and llm is not None:
        log(f"Krok 1: tlumaczenie nieprzetlumaczonych przepisow (max {translate_first})")
        await run_translate(job_id, runner, llm, limit=translate_first)
    series = store.series_of(settings.export_filename)
    folder = Path(settings.kwiatownik_przepisy_dir)
    target = folder / store.dated_name(series)               # np. siedziba_przepisy_2026-10-05.json
    activity.set(job_id, "zapisuje plik dla Kwiatownika", target.name)
    records, ids = build_export_records()
    target.parent.mkdir(parents=True, exist_ok=True)
    # "przed" = plik tej serii, ktory Kwiatownik czytal do tej pory (poprzedni eksport, takze z innego dnia)
    prev = next((f for f in store.current_files(folder) if store.parse_name(f.name)[0] == series), None)
    before = snapshots.read_text(prev) if prev else None
    text = json.dumps(records, ensure_ascii=False, indent=2)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    snapshots.record(target, before, text, "eksport przepisow do Kwiatownika", job_id=job_id)
    with Session(engine) as s:
        for iid in ids:
            it = s.get(Item, iid)
            it.status = ItemStatus.exported
            s.add(it)
        s.commit()
    n_src = sum(len(r.get("zrodla", [])) for r in records)
    log(f"Zapisano {len(records)} przepisow ({n_src} zrodel) do {target}"
        + (f" (poprzedni plik serii: {prev.name} - Kwiatownik czyta teraz najnowszy)" if prev and prev != target else ""))
    if settings.export_require_approval:
        log("Eksportowane sa tylko przepisy zatwierdzone w panelu (EXPORT_REQUIRE_APPROVAL=true)")
    stats = {"exported": len(records), "sources": n_src}
    _save_stats(job_id, stats)
    return stats


def _save_stats(job_id: int, stats: dict) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        old = json.loads(job.stats_json) if job.stats_json else {}
        job.stats_json = json.dumps(old | stats)
        s.add(job)
        s.commit()
