"""Zadania 'translate', 'enrich' i 'export':
tlumaczenie na polski -> pominiecie kulinarnych -> sprawdzenie, czy przepis juz istnieje -> opracowanie
(tradycyjne systemy) -> eksport."""
import asyncio
import json
import re
from urllib.parse import urlparse
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
from app.pipeline.opracowanie import opracuj
from app.llm.base import InvalidOutputError
from app.mt.nllb import MTQualityError, MTUnavailable, get_translator, nllb_code
from app.knowledge.quotes import quote_leads
from app.pipeline.checks import load_foreign_names
from app.pipeline.extract import ExtractionCheckError, extract_recipes
from app.scrapers.clean import clean_text, site_boilerplate
from app.pipeline.fast_translate import can_fast_translate, fast_translate_recipe
from app.pipeline.translate import NoRecipeError, skip_reason, translate_recipe
from app.worker import activity, snapshots
from app.worker.jobs import checkpoint, job_log
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


async def _refresh_plant_names(log) -> None:
    """Nazwy roslin w obcych jezykach z Wikidata (slownik DeepL + kontrola tlumaczenia). Najwyzej 2 minuty."""
    from app.knowledge.plant_names import refresh_plant_names
    from app.knowledge.wiki import WikiClient
    from app.pipeline.checks import forget_plant_rows
    wiki = WikiClient(settings.user_agent)
    try:
        stats = await asyncio.wait_for(refresh_plant_names(engine, wiki, log, settings.plant_names_per_run), 120)
        if stats.get("found"):
            forget_plant_rows()
    except asyncio.TimeoutError:
        log("Nazwy roslin z Wikidata: przerwano po 2 min - reszta przy nastepnym tlumaczeniu")
        forget_plant_rows()
    except Exception as exc:  # noqa: BLE001 - nazwy to dodatek, tlumaczenie idzie dalej
        log(f"Nazwy roslin z Wikidata: {str(exc)[:120]}")
    finally:
        await wiki.aclose()


_BOILERPLATE: dict[int, tuple[int, set[str]]] = {}


def _site_boilerplate(source_id: int | None, item_id: int) -> set[str]:
    """Linie powtarzajace sie na wielu stronach tej samej witryny (do 40 ostatnich stron), pamietane na zrodlo."""
    if not source_id:
        return set()
    with Session(engine) as s:
        rows = s.exec(select(Item.raw_text).where(Item.source_id == source_id, col(Item.raw_text).is_not(None))
                      .order_by(col(Item.id).desc()).limit(40)).all()
    cached = _BOILERPLATE.get(source_id)
    if cached and cached[0] == len(rows):
        return cached[1]
    found = site_boilerplate(list(rows))
    _BOILERPLATE[source_id] = (len(rows), found)
    return found


def _mt_supports(mt, lang: str | None) -> bool:
    supports = getattr(mt, "supports", None)
    return supports(lang) if supports else nllb_code(lang) is not None


async def translate_one(item_id: int, llm, plants: dict, kw_cands: list[Candidate], kw_ids: set[str], log) -> str:
    with Session(engine) as s:
        item = s.get(Item, item_id)
        src = s.get(Source, item.source_id) if item.source_id else None
        recipe = json.loads(item.structured_json) if item.structured_json else None
        meta = {"item_id": item.id, "title": item.title, "language": item.language or (src.language if src else None),
                "country": src.country if src else None}
        raw = item.raw_text
        ignore_match = item.match_info == "ignored"
        source_id = item.source_id
    if raw:
        raw = clean_text(raw, _site_boilerplate(source_id, item_id))   # resztki nawigacji, reklam, spisu tresci
    rec, model = None, None
    mt = get_translator()
    lang = meta["language"]
    legacy = llm is not None and (mt is None or settings.translate_with_llm)
    taken = _taken_ids(kw_ids)

    def mark(status: ItemStatus, reason: str) -> None:
        with Session(engine) as s:
            item = s.get(Item, item_id)
            item.status, item.reason = status, reason[:300]
            s.add(item)
            s.commit()

    async def no_recipe(exc: NoRecipeError) -> str:
        # lista przepisow / artykul ogolny - nie blad, tylko strona bez konkretnego przepisu
        if exc.leads and raw:                          # opis wzmianki = przetlumaczony cytat ze strony, nie slowa LLM
            exc.leads = await asyncio.to_thread(quote_leads, exc.leads, raw, lang, mt)
        mark(ItemStatus.skipped, f"brak konkretnego przepisu: {exc}")
        with Session(engine) as s:
            url = s.get(Item, item_id).url
        log(f"[pominiety] #{item_id} {meta['title'][:50]}: brak konkretnego przepisu")
        if exc.leads:                                  # wzmianki o przepisach -> lista "Przepisy do odnalezienia"
            from app.pipeline.leads import save_leads
            new = save_leads(exc.leads, url, urlparse(url).netloc, meta.get("language"), item_id, None)
            if new:
                log(f"[wzmianka] #{item_id}: {len(new)} przepisow do odnalezienia: "
                    + ", ".join(w["tytul"][:40] for w in exc.leads[:3]))
        return "no_recipe"

    if mt is not None and not legacy:
        if not _mt_supports(mt, lang):
            mark(ItemStatus.error, f"tlumacz maszynowy nie obsluguje jezyka: {lang or 'nieznany'}")
            log(f"[blad MT] #{item_id} {meta['title'][:50]}: jezyk {lang or 'nieznany'} bez tlumacza")
            return "error"
        names = load_foreign_names(engine)
        try:
            if can_fast_translate(recipe, lang):
                # dane strukturalne ze strony -> tlumacz maszynowy -> kontrola kodem
                rec, model = await fast_translate_recipe(mt, llm, recipe, meta, plants, taken, names)
            elif raw and llm is not None:
                # surowy artykul: LLM tylko wskazuje doslowne fragmenty (kod je sprawdza), tlumaczy tlumacz maszynowy
                ex, emodel = await extract_recipes(llm, raw, meta, settings.extract_min_verbatim,
                                                   settings.extract_max_rejected)
                if ex.rejected:
                    log(f"[kontrola] #{item_id}: odrzucono {len(ex.rejected)} linii spoza tekstu strony "
                        f"(np. '{ex.rejected[0][:60]}')")
                rec, model = await fast_translate_recipe(mt, llm, ex.recipes[0], meta, plants, taken, names)
                rec["siedziba"]["wyciagnal"] = f"{emodel} (doslowne fragmenty, sprawdzone kodem)"
                extras = []
                for n, part in enumerate(ex.recipes[1:], 2):
                    try:
                        e, _ = await fast_translate_recipe(
                            mt, llm, part, dict(meta, item_id=f"{item_id}_{n}"), plants,
                            taken | {rec["id"]} | {x["id"] for x in extras}, names)
                    except MTQualityError as exc:
                        log(f"[kontrola] #{item_id}: pominieto przepis {n} z artykulu - {str(exc)[:120]}")
                        continue
                    e["siedziba"]["wyciagnal"] = rec["siedziba"]["wyciagnal"]
                    extras.append(e)
                if extras:
                    rec["_dodatkowe"] = extras
            elif not raw:
                mark(ItemStatus.error, "brak tekstu strony do tlumaczenia")
                return "error"
            else:
                log(f"[odlozone] #{item_id} {meta['title'][:50]}: artykul bez danych strukturalnych, "
                    "a LLM do wskazania fragmentow niedostepny")
                return "postponed"
        except MTUnavailable as exc:
            log(f"[odlozone] #{item_id} {meta['title'][:50]}: tlumacz maszynowy niedostepny ({str(exc)[:160]})")
            return "postponed"
        except AllModelsFailedError as exc:
            log(f"[odlozone] #{item_id} {meta['title'][:50]}: wszystkie modele maja teraz limit ({str(exc)[:160]})")
            return "postponed"
        except NoRecipeError as exc:
            return await no_recipe(exc)
        except (MTQualityError, ExtractionCheckError) as exc:
            mark(ItemStatus.error, str(exc) if str(exc).startswith("kontrola") else f"tlumaczenie maszynowe: {exc}")
            log(f"[kontrola] #{item_id} {meta['title'][:50]}: {str(exc)[:200]}")
            return "error"
        except (LLMError, ValueError) as exc:
            mark(ItemStatus.error, f"wyszukanie przepisu w tekscie: {exc}")
            log(f"[blad] #{item_id} {meta['title'][:50]}: {exc}")
            return "error"
    elif not legacy:
        log(f"[odlozone] #{item_id} {meta['title'][:50]}: brak tlumacza maszynowego")
        return "postponed"
    else:
        # dawny tryb (MT_ENGINE=llm albo TRANSLATE_WITH_LLM=true): LLM tlumaczy, kod sprawdza liczby
        try:
            rec, model = await translate_recipe(llm, recipe, raw, meta, plants, taken)
        except AllModelsFailedError as exc:
            # zaden model nie byl dostepny (limity) - to nie wina przepisu: zostaje "do tlumaczenia"
            log(f"[odlozone] #{item_id} {meta['title'][:50]}: wszystkie modele maja teraz limit ({str(exc)[:160]})")
            return "postponed"
        except NoRecipeError as exc:
            return await no_recipe(exc)
        except (LLMError, ValueError) as exc:
            mark(ItemStatus.error, f"tlumaczenie: {exc}")
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
    if match is None:                 # opracowanie bez LLM: karty roslin + zrodlo (decyzja 2026-11-02)
        rec = opracuj(rec, _plant_loader(), _source_text(item_id, raw))
        msg += " + opracowanie z kart roslin"
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


def _source_text(item_id: int, raw: str | None = None) -> str | None:
    """Tekst zrodla przepisu (strona + dane strukturalne) - sprawdzanie, czy liczby dawkowania sa w zrodle."""
    with Session(engine) as s:
        item = s.get(Item, item_id)
        parts = [raw or (item.raw_text if item else None), item.structured_json if item else None]
    text = " ".join(p for p in parts if p)
    return text or None


def enrich_one(item_id: int, log, load_plant=None) -> str:
    """Opracowanie przepisu z kart roslin (bez LLM): usuwa pola dopisane kiedys przez model, dodaje ostrzezenia
    i czas zbioru z kart roslin oraz dymek Wu Xing / Galen."""
    with Session(engine) as s:
        item = s.get(Item, item_id)
        if not item or not item.data_json:
            return "skipped"
        rec = json.loads(item.data_json)
    rec2 = opracuj(rec, load_plant or _plant_loader(), _source_text(item_id))
    with Session(engine) as s:
        item = s.get(Item, item_id)
        new = json.dumps(rec2, ensure_ascii=False)
        snapshots.record_item(item, new, "opracowanie z kart roslin (bez LLM)")
        item.data_json = new
        item.updated_at = datetime.now(timezone.utc)
        s.add(item)
        s.commit()
    log(f"[opracowano] {rec2.get('tytul')} (karty roslin, bez LLM)")
    return "enriched"


def items_without_enrichment(limit: int | None = None, session: Session | None = None) -> list[int]:
    ok = [ItemStatus.translated, ItemStatus.approved, ItemStatus.exported]
    with (Session(engine) if session is None else nullcontext(session)) as s:
        rows = s.exec(select(Item.id, Item.data_json).where(
            Item.kind == ItemKind.recipe, col(Item.duplicate_of).is_(None), col(Item.data_json).is_not(None),
            col(Item.status).in_(ok)).order_by(Item.id)).all()
    def old(dj: str) -> bool:                     # brak opracowania albo dawne opracowanie przez LLM
        op = json.loads(dj).get("opracowanie")
        return not (isinstance(op, dict) and str(op.get("jak") or "").startswith("bez LLM"))
    ids = [iid for iid, dj in rows if old(dj)]
    return ids[:limit] if limit else ids


def pending_items(ids: list[int], kind: str) -> list[int]:
    """Z listy wybranych przepisow tylko te, ktore dalej czekaja: do tlumaczenia surowe i nieudane, do opracowania
    przetlumaczone bez opracowania. Przepisy zrobione wczesniej (np. w przerwanym zadaniu) nie ida drugi raz."""
    if not ids:
        return []
    with Session(engine) as s:
        rows = {it.id: it for it in s.exec(select(Item).where(col(Item.id).in_(ids))).all()}
    if kind == "translate":
        ok = lambda it: it.status in (ItemStatus.raw, ItemStatus.error) and it.duplicate_of is None  # noqa: E731
    else:
        wait = set(items_without_enrichment())
        ok = lambda it: it.id in wait  # noqa: E731
    return [i for i in ids if i in rows and ok(rows[i])]


def _only_pending(ids: list[int], kind: str, log) -> list[int]:
    left = pending_items(ids, kind)
    if len(left) < len(ids):
        log(f"Pomijam {len(ids) - len(left)} z {len(ids)} wybranych przepisow - juz zrobione wczesniej "
            f"(albo pominiete / zatwierdzone). Zostaje {len(left)}.")
    return left


async def run_enrich(job_id: int, runner: JobRunner, llm, limit: int | None = None,
                     item_ids: list[int] | None = None, force: bool = False) -> dict:
    log = lambda m: job_log(job_id, m)  # noqa: E731
    ids = list(item_ids) if item_ids else items_without_enrichment(limit or 500)
    if item_ids and not force:
        ids = _only_pending(ids, "enrich", log)
    log(f"Do opracowania: {len(ids)} przepisow (karty roslin Kwiatownika, bez LLM)")
    stats = {"enriched": 0, "skipped": 0}
    load_plant = _plant_loader()                    # karty roslin czytane raz na zadanie
    for n, iid in enumerate(ids, 1):
        if runner.should_stop(job_id):
            log("Zatrzymano")
            break
        activity.set(job_id, f"opracowuje {n}/{len(ids)}", f"#{iid}")
        stats[enrich_one(iid, log, load_plant)] += 1
        checkpoint(job_id, iid)
        _progress(job_id, n, len(ids))
        await asyncio.sleep(0)                         # bez LLM - nie blokuj panelu przy setkach przepisow
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    _save_stats(job_id, stats)
    return stats


async def run_translate(job_id: int, runner: JobRunner, llm, limit: int | None = None,
                        item_ids: list[int] | None = None, force: bool = False) -> dict:
    """force=False: z wybranej listy tylko przepisy, ktore dalej czekaja (surowe / nieudane). force=True - reczne
    "Przetlumacz ponownie" konkretnego przepisu. Kazdy skonczony przepis to punkt kontrolny dla ponowienia."""
    mt = get_translator()
    legacy = llm is not None and (mt is None or settings.translate_with_llm)
    if not legacy and not (mt and mt.model_present()):
        raise RuntimeError("Tlumaczenie wymaga tlumacza maszynowego: DEEPL_API_KEY w .env albo modelu NLLB "
                           "(docker compose --profile mt run --rm nllb-init)")
    with Session(engine) as s:
        if item_ids:
            ids = list(item_ids)
        else:
            ids = list(s.exec(select(Item.id).where(
                Item.kind == ItemKind.recipe, Item.status == ItemStatus.raw, col(Item.duplicate_of).is_(None))
                .order_by(col(Item.confidence).desc(), col(Item.id)).limit(limit or settings.translate_batch)).all())
    log = lambda m: job_log(job_id, m)  # noqa: E731
    if item_ids and not force:
        ids = _only_pending(ids, "translate", log)
    plants = load_plants(settings.kwiatownik_plants_dir)
    kw_recipes = load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir, exclude={settings.export_filename})
    kw_cands = [Candidate("kwiatownik", r.ref, r.title, r.ingredients) for r in kw_recipes]
    kw_ids = {r.rid for r in load_kwiatownik_recipes(settings.kwiatownik_przepisy_dir) if r.rid}
    log(f"Do tlumaczenia: {len(ids)} przepisow; Kwiatownik: {len(kw_cands)} przepisow, {len(plants)} roslin")
    if not legacy and settings.plant_names_per_run > 0:
        await _refresh_plant_names(log)
    if legacy:
        log("Tlumacz: LLM (dawny tryb: MT_ENGINE=llm albo TRANSLATE_WITH_LLM=true)")
    elif mt.available():
        log(f"Tlumacz maszynowy: {' -> '.join(getattr(mt, 'names', [mt.model_dir.name]))}; surowe artykuly: LLM "
            "wskazuje doslowne fragmenty, kod je sprawdza, tlumaczy tlumacz maszynowy")
    else:
        log(f"Tlumacz maszynowy niedostepny ({mt.error or 'brak klucza DeepL i modelu NLLB'}) - przepisy czekaja")
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
        if outcome != "postponed":                     # odlozone (limity modeli) zostaja "do tlumaczenia"
            checkpoint(job_id, iid)
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


def _plant_loader():
    """load(id) -> dane pliku rosliny z Kwiatownika (z pamiecia na czas jednego eksportu)."""
    from app.knowledge import plantfile
    cache: dict[str, dict | None] = {}

    def load(pid: str) -> dict | None:
        if not isinstance(pid, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", pid):
            return None                                # id z przepisu (od LLM) - tylko nazwa pliku, bez sciezek
        if pid not in cache:
            try:
                cache[pid] = plantfile.read_plant(Path(settings.kwiatownik_plants_dir), pid)[0]
            except (OSError, ValueError):
                cache[pid] = None
        return cache[pid]
    return load


def build_export_records() -> tuple[list[dict], list[int]]:
    load_plant = _plant_loader()
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
            # Wu Xing / Galen: dymek-ciekawostka liczony z pliku rosliny (bez LLM), tylko przepisy lecznicze
            # bez pol dopisanych kiedys przez LLM; ostrzezenia, zbior i dymek z kart roslin (decyzja 2026-11-02)
            records.append(with_sources(opracuj(json.loads(p.data_json), load_plant,
                                                " ".join(x for x in (p.raw_text, p.structured_json) if x) or None),
                                        srcs))
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
    if settings.export_leads:                           # wzmianki o przepisach (bez potwierdzonych krokow)
        from app.pipeline.leads import export_leads
        try:
            name, n_leads = export_leads(folder, load_plants(settings.kwiatownik_plants_dir))
            log(f"Wzmianki o przepisach: {n_leads}" + (f" - zapisano {name}" if name else " - bez zmian"))
            stats["leads"] = n_leads
        except Exception as exc:  # noqa: BLE001 - wzmianki nie blokuja eksportu przepisow
            log(f"Wzmianki o przepisach: blad zapisu ({exc})")
    _save_stats(job_id, stats)
    return stats


def _save_stats(job_id: int, stats: dict) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        old = json.loads(job.stats_json) if job.stats_json else {}
        job.stats_json = json.dumps(old | stats)
        s.add(job)
        s.commit()
