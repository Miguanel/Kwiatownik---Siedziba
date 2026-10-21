"""Zadanie "plant_magic": magia roslin - czary, zabobony, obrzedy, rosliny czarownic, uroki, wrozby.

Na rosline: tropy (azymuty) z poprzednich stron -> zapytania tezaurusa w wielu jezykach -> glebokie archiwa
(app/knowledge/magic.py). Z kazdej znalezionej strony LLM wyciaga informacje (sekcja "magia" + rodzaj, kraj,
epoka, tradycja) i nowe tropy, ktore prowadza dalej. Na koniec zadania: przeszukiwanie "od praktyki"
(strony o obrzedzie/praktyce, z ktorych jedna daje wiedze o wielu roslinach). Potem zwykla droga wiedzy:
weryfikacja drugim modelem -> zapis do pliku rosliny -> porzadkowanie -> rozmieszczenie w rozdziale
"Magia i zabobony".
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.knowledge import magic as mg
from app.knowledge.magic_lexicon import CONCEPT_BY_KEY, CONCEPTS, FAMOUS_GENERA
from app.knowledge.sections import fingerprint, norm_text, similar
from app.llm.router import AllModelsFailedError
from app.models import MagicLead, Plant, PlantFact, PlantQuery
from app.worker import activity
from app.worker import knowledge as kw
from app.worker.jobs import job_log
from app.worker.runner import JobRunner

SWEEP_PLANT = "_magia"            # PlantQuery.plant_id zapytan "od praktyki" (bez konkretnej rosliny)
ACCEPT_LANGUAGE = "pl,de;q=0.9,ru;q=0.8,uk;q=0.8,en;q=0.7,cs;q=0.6,it;q=0.5,es;q=0.5,fr;q=0.5,ja;q=0.4,zh;q=0.4,*;q=0.3"
MIN_TEXT = {"zh": 300, "ja": 300}


def _utc(dt: datetime | None) -> datetime | None:
    return None if dt is None else (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))


def _langs() -> list[str]:
    return [x.strip() for x in settings.magic_langs.split(",") if x.strip()]


# ------------------------------------------------------------------ stan w bazie
def recent_queries(plant_id: str) -> tuple[set[str], set[str], set[str]]:
    """(klucze i tresci zapytan o magie z ostatnich MAGIC_REQUERY_DAYS dni, adresy juz uzyte, ich domeny)."""
    since = datetime.now(timezone.utc) - timedelta(days=settings.magic_requery_days)
    with Session(engine) as s:
        rows = s.exec(select(PlantQuery).where(PlantQuery.plant_id == plant_id,
                                               col(PlantQuery.slot).startswith("magia"))).all()
        urls = set(s.exec(select(PlantFact.source_url).where(PlantFact.plant_id == plant_id)).all())
    done: set[str] = set()
    for q in rows:
        if _utc(q.created_at) >= since:
            done.update((q.slot, q.query))
        if q.used_url:
            urls.add(q.used_url)
    return done, urls, {urlparse(u).netloc.removeprefix("www.") for u in urls if u}


def folk_names(plant_id: str) -> dict[str, list[str]]:
    """Ludowe nazwy rosliny z tropow typu "nazwa" (jezyk -> nazwy) - kolejne warianty do zapytan i do stron."""
    with Session(engine) as s:
        rows = s.exec(select(MagicLead).where(MagicLead.plant_id == plant_id, MagicLead.kind == "nazwa",
                                              MagicLead.status != "odrzucony")).all()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r.language, []).append(r.term)
    return out


def magic_langs_covered(plant_id: str) -> set[str]:
    with Session(engine) as s:
        return set(x for x in s.exec(select(PlantFact.language).where(
            PlantFact.plant_id == plant_id, PlantFact.section == "magia", PlantFact.status != "rejected")).all() if x)


def open_leads(plant_id: str | None, limit: int = 6) -> list[MagicLead]:
    with Session(engine) as s:
        q = select(MagicLead).where(MagicLead.status == "nowy", MagicLead.depth <= settings.magic_max_depth)
        q = q.where(MagicLead.plant_id == plant_id) if plant_id else q.where(col(MagicLead.plant_id).is_(None))
        rows = list(s.exec(q).all())
        for r in rows:
            s.expunge(r)
    covered = magic_langs_covered(plant_id) if plant_id else set()
    return mg.lead_order(rows, covered)[:limit]


def magic_counts() -> dict[str, int]:
    with Session(engine) as s:
        return dict(s.exec(select(PlantFact.plant_id, func.count()).where(
            PlantFact.section == "magia", PlantFact.status != "rejected").group_by(PlantFact.plant_id)).all())


def plants_for_magic(limit: int) -> list[str]:
    """Rosliny do zbadania: najpierw nigdy nie szukane pod katem magii, z rodzajow slynnych w magii ludowej,
    z otwartymi tropami, z najmniejsza wiedza o magii; na koncu najdawniej szukane."""
    counts = magic_counts()
    with Session(engine) as s:
        plants = s.exec(select(Plant.id, Plant.nazwa_lat)).all()
        last = dict(s.exec(select(PlantQuery.plant_id, func.max(PlantQuery.created_at))
                           .where(col(PlantQuery.slot).startswith("magia")).group_by(PlantQuery.plant_id)).all())
        leads = dict(s.exec(select(MagicLead.plant_id, func.count()).where(MagicLead.status == "nowy")
                            .group_by(MagicLead.plant_id)).all())
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)

    def key(row):
        pid, lat = row
        genus = (lat or "").split(" ")[0].lower()
        return (last.get(pid) is not None and not leads.get(pid), genus not in FAMOUS_GENERA,
                counts.get(pid, 0), _utc(last.get(pid)) or epoch, pid)
    return [pid for pid, _ in sorted(plants, key=key)[:limit]]


def log_query(plant_id: str, mq: mg.MQuery, results: int, used_url: str | None) -> int:
    with Session(engine) as s:
        row = PlantQuery(plant_id=plant_id, slot=mq.key[:120], language=mq.lang, query=mq.query[:300],
                         results=results, used_url=used_url)
        s.add(row)
        s.commit()
        return row.id


def lead_done(lead_id: int | None, found: bool, facts: int = 0) -> None:
    if not lead_id:
        return
    with Session(engine) as s:
        row = s.get(MagicLead, lead_id)
        if row:
            row.attempts += 1
            row.facts += facts
            row.status = "sprawdzony" if found else "wyczerpany"
            row.last_at = datetime.now(timezone.utc)
            s.add(row)
            s.commit()


def store_facts(plant_id: str, facts: list[mg.MagicFact], url: str, name: str, lang: str, model: str) -> list[int]:
    """Zapis nowych informacji o magii (bez powtorzen: ten sam odcisk albo bardzo podobne zdanie)."""
    ids = []
    with Session(engine) as s:
        existing = s.exec(select(PlantFact.fingerprint, PlantFact.text).where(PlantFact.plant_id == plant_id)).all()
        prints = {fp for fp, _ in existing}
        texts = [t for _, t in existing]
        rows = []
        for f in facts:
            fp = fingerprint(plant_id, f.text)
            if fp in prints or any(similar(f.text, t) >= 0.6 for t in texts):
                continue
            row = PlantFact(plant_id=plant_id, section="magia", part=f.part, text=f.text, quote=f.quote,
                            source_url=url, source_name=name, language=lang, fingerprint=fp, extracted_by=model,
                            meta_json=json.dumps(f.meta, ensure_ascii=False) if f.meta else None)
            s.add(row)
            rows.append(row)
            prints.add(fp)
            texts.append(f.text)
        s.commit()
        ids = [r.id for r in rows]
    return ids


def save_leads(plant_id: str | None, leads: list[dict], url: str, depth: int, parent_id: int | None,
               parent_yield: int, rodzaj: str | None = None, known_names: set[str] | None = None) -> int:
    """Nowe tropy (bez powtorzen dla tej rosliny; za gleboko = pomijane)."""
    if depth > settings.magic_max_depth or not leads:
        return 0
    known = {norm_text(n) for n in known_names or set()}
    added = 0
    with Session(engine) as s:
        for ld in leads:
            if ld["typ"] == "nazwa" and norm_text(ld["termin"]) in known:
                continue
            fp = mg.lead_fingerprint(plant_id, ld["termin"], ld["jezyk"])
            if s.exec(select(MagicLead.id).where(MagicLead.fingerprint == fp)).first():
                continue
            s.add(MagicLead(plant_id=plant_id, term=ld["termin"], language=ld["jezyk"], kind=ld["typ"],
                            rodzaj=rodzaj, note=ld.get("opis"), origin_url=url, depth=depth, parent_id=parent_id,
                            parent_yield=parent_yield, fingerprint=fp))
            added += 1
        s.commit()
    return added


def _known_magic(plant_id: str) -> list[str]:
    with Session(engine) as s:
        return list(s.exec(select(PlantFact.text).where(PlantFact.plant_id == plant_id, PlantFact.status != "rejected",
                                                        col(PlantFact.section).in_(["magia", "kultura"]))).all())


# ------------------------------------------------------------------ strony
async def find_page(mq: mg.MQuery, backends, fetcher, used_urls: set[str], name_keys: list[str], log,
                    min_magic: int = 2) -> tuple[int, object | None, str | None]:
    """(liczba wynikow wyszukiwarki, strona, adres) - pierwsza strona o tej roslinie i o magii."""
    from app.discovery.search import search_all
    from app.scrapers.page import parse_page
    results = await search_all(backends, mq.query, mq.lang, "", log_fn=log)
    for r in results[:6]:
        url = r.url
        if any(d in url for d in kw.SKIP_DOMAINS) or url in used_urls or url.lower().endswith((".pdf", ".djvu")):
            continue
        res = await fetcher.get(url)
        if not res.ok:
            continue
        page = await asyncio.to_thread(parse_page, res.text, res.url)
        if len(page.text or "") < MIN_TEXT.get(mq.lang, 600):
            continue
        ok, _why = mg.page_relevant(page.text, name_keys, min_magic)
        if ok:
            return len(results), page, res.url
    return len(results), None, None


# ------------------------------------------------------------------ jedna roslina
async def research_plant_magic(plant_id: str, llm, fetcher, backends, log, job_id: int = 0) -> dict:
    stats = {"zapytania": 0, "strony": 0, "fakty": 0, "odrzucone": 0, "tropy": 0}
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        s.expunge(plant)
    names = kw._names(plant)
    folk = folk_names(plant_id)
    name_keys = sorted(set(kw._name_keys(plant)) | {n.lower() for v in folk.values() for n in v if len(n) >= 3},
                       key=len, reverse=True)
    done, used_urls, used_domains = recent_queries(plant_id)
    leads = open_leads(plant_id)
    queries = mg.plant_queries(names, _langs(), done, leads, folk, used_domains, limit=settings.magic_queries)
    if not queries:
        log("  wszystkie zapytania o magie tej rosliny byly zadane niedawno - pomijam")
        return stats
    log(f"  zapytania: {len(queries)} (tropy: {sum(1 for q in queries if q.kind == 'trop')}, "
        f"archiwa: {sum(1 for q in queries if q.kind == 'archiwum')}), nazwy ludowe: "
        + (", ".join(f"{k}: {'/'.join(v[:2])}" for k, v in list(folk.items())[:4]) or "brak"))
    known = _known_magic(plant_id)
    for mq in queries:
        if stats["strony"] >= settings.magic_pages:
            break
        if job_id:
            activity.set(job_id, f"{plant.nazwa_pl}: magia ({mq.kind}, {mq.lang})", mq.query[:100])
        n_res, page, url = await find_page(mq, backends, fetcher, used_urls, name_keys, log)
        stats["zapytania"] += 1
        qid = log_query(plant_id, mq, n_res, url)
        if page is None:
            lead_done(mq.lead_id, False)
            log(f"  [{mq.lang}] {mq.kind}: {mq.query[:80]} - brak strony o magii tej rosliny")
            continue
        used_urls.add(url)
        stats["strony"] += 1
        domain = urlparse(url).netloc.removeprefix("www.")
        text = mg.focus_text(page.text, name_keys)
        hint = None
        if mq.kind == "trop":
            hint = mq.query
        try:
            res = await llm.complete(mg.build_prompt(plant.nazwa_pl, plant.nazwa_lat, names, domain,
                                                     page.lang or mq.lang, text, known, hint),
                                     system=mg.MAGIC_SYSTEM, json_mode=True)
            facts, leads_found, rejected = mg.parse_facts(res.text, text, settings.magic_include_modern)
            model = f"{res.provider}/{res.model}"
        except AllModelsFailedError:
            raise
        except Exception as exc:                       # zly JSON jednej strony nie zatrzymuje reszty
            lead_done(mq.lead_id, True)
            log(f"  [blad] {domain}: {str(exc)[:120]}")
            continue
        ids = store_facts(plant_id, facts, url, domain, (page.lang or mq.lang)[:5], model)
        kw._query_facts(qid, len(ids))
        lead_done(mq.lead_id, True, len(ids))
        depth = (mq.depth + 1) if mq.kind == "trop" else 1
        rodzaj = max((f.meta.get("rodzaj") for f in facts), key=[f.meta.get("rodzaj") for f in facts].count) \
            if facts else mq.rodzaj
        nl = save_leads(plant_id, leads_found, url, depth, mq.lead_id, len(ids), rodzaj,
                        set(names.values()) | {n for v in folk.values() for n in v})
        known += [f.text for f in facts]
        stats["fakty"] += len(ids)
        stats["odrzucone"] += len(rejected)
        stats["tropy"] += nl
        log(f"  [{mq.lang}] {mq.kind} {domain}: {len(ids)} nowych informacji o magii"
            + (f", tropy +{nl}" if nl else "") + (f", odrzucono {len(rejected)}" if rejected else ""))
    return stats


# ------------------------------------------------------------------ od praktyki (wiele roslin z jednej strony)
def plant_index() -> mg.PlantIndex:
    with Session(engine) as s:
        rows = s.exec(select(Plant)).all()
        data = []
        for p in rows:
            try:
                names = json.loads(p.names_json) if p.names_json else {}
            except ValueError:
                names = {}
            data.append((p.id, p.nazwa_pl, p.nazwa_lat, names if isinstance(names, dict) else {}))
    return mg.PlantIndex(data)


def general_lead_queries(done: set[str], limit: int) -> list[mg.MQuery]:
    """Tropy ogolne (z przeszukiwania od praktyki): termin + slowa rodzaju praktyki w jezyku tropu."""
    out = []
    for ld in open_leads(None, limit * 2):
        c = next((c for c in CONCEPTS if c.rodzaj == (ld.rodzaj or "zabobony") and ld.language in c.terms), None) \
            or CONCEPT_BY_KEY["zabobony"]
        q = f"\"{ld.term}\" {c.terms.get(ld.language) or c.terms.get('en', '')}".strip()
        key = f"magia:trop:{ld.id}"
        if key in done:
            continue
        out.append(mg.MQuery("trop", ld.language, q, key, ld.rodzaj, ld.id, ld.depth))
        if len(out) >= limit:
            break
    return out


async def sweep(llm, fetcher, backends, log, job_id: int = 0, limit: int = 2) -> tuple[dict, set[str]]:
    """Zapytania "od praktyki" (bez nazwy rosliny): z jednej strony informacje o wielu roslinach Kwiatownika."""
    from app.discovery.search import search_all
    from app.scrapers.page import parse_page
    stats = {"zapytania": 0, "strony": 0, "fakty": 0, "rosliny": 0, "spoza": 0, "tropy": 0}
    touched: set[str] = set()
    done, used_urls, _ = recent_queries(SWEEP_PLANT)
    queries = mg.sweep_queries(done, limit) + general_lead_queries(done, max(1, limit // 2))
    if not queries:
        log("Przeglad od praktyki: wszystkie zapytania byly zadane niedawno")
        return stats, touched
    index = plant_index()
    for mq in queries:
        if job_id:
            activity.set(job_id, f"magia: przeglad od praktyki ({mq.lang})", mq.query[:100])
        results = await search_all(backends, mq.query, mq.lang, "", log_fn=log)
        stats["zapytania"] += 1
        pages = []
        for r in results[:8]:
            if len(pages) >= 2:
                break
            if any(d in r.url for d in kw.SKIP_DOMAINS) or r.url in used_urls or r.url.lower().endswith(".pdf"):
                continue
            res = await fetcher.get(r.url)
            if not res.ok:
                continue
            page = await asyncio.to_thread(parse_page, res.text, res.url)
            if len(page.text or "") < MIN_TEXT.get(mq.lang, 800):
                continue
            ok, _ = mg.page_relevant(page.text, [], min_magic=3)
            if ok:
                pages.append((res.url, page))
                used_urls.add(res.url)
        log_query(SWEEP_PLANT, mq, len(results), pages[0][0] if pages else None)
        lead_done(mq.lead_id, bool(pages))
        if not pages:
            log(f"Przeglad [{mq.lang}] {mq.query[:70]}: brak stron o magii roslin")
            continue
        for url, page in pages:
            domain = urlparse(url).netloc.removeprefix("www.")
            text = page.text[:12000]
            try:
                res = await llm.complete(mg.sweep_prompt(domain, page.lang or mq.lang, text, mq.query),
                                         system=mg.SWEEP_SYSTEM, json_mode=True)
                facts, leads_found, _rej = mg.parse_facts(res.text, text, settings.magic_include_modern, sweep=True)
                model = f"{res.provider}/{res.model}"
            except AllModelsFailedError:
                raise
            except Exception as exc:
                log(f"  [blad] {domain}: {str(exc)[:120]}")
                continue
            stats["strony"] += 1
            by_plant: dict[str, list[mg.MagicFact]] = {}
            outside = 0
            for f in facts:
                pid = index.find(f.plant, f.latin)
                if pid:
                    by_plant.setdefault(pid, []).append(f)
                else:
                    outside += 1
            n_new = 0
            for pid, fs in by_plant.items():
                ids = store_facts(pid, fs, url, domain, (page.lang or mq.lang)[:5], model)
                if ids:
                    touched.add(pid)
                    n_new += len(ids)
            nl = save_leads(None, leads_found, url, (mq.depth + 1) if mq.kind == "trop" else 1, mq.lead_id, n_new,
                            mq.rodzaj)
            stats["fakty"] += n_new
            stats["spoza"] += outside
            stats["tropy"] += nl
            log(f"Przeglad [{mq.lang}] {domain}: {n_new} informacji o {len(by_plant)} roslinach Kwiatownika"
                + (f", {outside} o roslinach spoza Kwiatownika" if outside else "") + (f", tropy +{nl}" if nl else ""))
    stats["rosliny"] = len(touched)
    return stats, touched


# ------------------------------------------------------------------ zadanie
async def _verify_apply(pid: str, llm, log, job_id: int, total: dict) -> None:
    vs = await kw.verify_plant(pid, llm, log, job_id)
    total["zweryfikowane"] += vs["verified"]
    total["odrzucone_weryfikacja"] += vs["rejected"]
    if settings.knowledge_auto_apply and vs["verified"]:
        outcome = await kw.apply_plant(pid, log, llm)
        if outcome == "applied":
            total["zapisane_rosliny"] += 1


async def run_plant_magic(job_id: int, runner: JobRunner, llm, plant_ids: list[str] | None = None,
                          limit: int | None = None, sweeps: int | None = None) -> dict:
    if llm is None:
        raise RuntimeError("Magia roslin wymaga LLM - ustaw klucze GEMINI/GROQ w .env")
    from app.scrapers.fetcher import Fetcher
    from app.worker.discover import build_backends
    log = lambda m: job_log(job_id, m)  # noqa: E731
    total = {"rosliny": 0, "zapytania": 0, "strony": 0, "fakty": 0, "tropy": 0, "zweryfikowane": 0,
             "odrzucone_weryfikacja": 0, "zapisane_rosliny": 0, "przeglad_fakty": 0}
    backends = build_backends()
    if not backends:
        log("Brak wyszukiwarki (SEARXNG_URL / BRAVE_API_KEY) - nie ma gdzie szukac")
        return total
    ids = list(plant_ids) if plant_ids else plants_for_magic(limit or settings.magic_plants)
    n_sweeps = settings.magic_sweeps if sweeps is None else sweeps
    log(f"Magia roslin: {len(ids)} roslin, przeglad od praktyki: {n_sweeps} zapytan; jezyki: {settings.magic_langs}; "
        f"tropy do glebokosci {settings.magic_max_depth}")
    fetcher = Fetcher(settings.user_agent, settings.request_delay_s, accept_language=ACCEPT_LANGUAGE)
    stopped = False
    try:
        for n, pid in enumerate(ids, 1):
            if runner.should_stop(job_id):
                log("Zatrzymano")
                stopped = True
                break
            with Session(engine) as s:
                plant = s.get(Plant, pid)
                title = plant.nazwa_pl if plant else pid
            if not plant:
                continue
            if pid in kw._BUSY:
                log(f"[{n}/{len(ids)}] {title} - opracowywana teraz w innym zadaniu, pomijam")
                continue
            log(f"[{n}/{len(ids)}] {title}")
            activity.set(job_id, f"magia: roslina {n}/{len(ids)}", title)
            kw._BUSY.add(pid)
            before = kw.facts_dump(pid)
            try:
                st = await research_plant_magic(pid, llm, fetcher, backends, log, job_id)
                for k in ("zapytania", "strony", "fakty", "tropy"):
                    total[k] += st[k]
                await _verify_apply(pid, llm, log, job_id, total)
                kw._record_facts(pid, title, before, "magia roslin: zbieranie i weryfikacja")
            except AllModelsFailedError as exc:
                kw._record_facts(pid, title, before, "magia roslin (przerwane - limity modeli)")
                log(f"Limity modeli wyczerpane - przerywam ({str(exc)[:120]}). Zebrane informacje zostaja.")
                stopped = True
                break
            except Exception as exc:  # noqa: BLE001 - blad jednej rosliny nie przerywa zadania
                logging.getLogger(__name__).exception("Magia roslin: %s", pid)
                log(f"  BLAD ({type(exc).__name__}: {str(exc)[:160]}) - przechodze do nastepnej rosliny")
                continue
            finally:
                kw._BUSY.discard(pid)
            total["rosliny"] += 1
            kw._checkpoint(job_id, pid)
            kw._progress(job_id, n, len(ids))
        if not stopped and n_sweeps > 0 and not runner.should_stop(job_id):
            try:
                st, touched = await sweep(llm, fetcher, backends, log, job_id, n_sweeps)
                total["przeglad_fakty"] = st["fakty"]
                total["tropy"] += st["tropy"]
                for pid in sorted(touched):
                    if pid in kw._BUSY:
                        continue
                    kw._BUSY.add(pid)
                    try:
                        await _verify_apply(pid, llm, log, job_id, total)
                    finally:
                        kw._BUSY.discard(pid)
            except AllModelsFailedError as exc:
                log(f"Limity modeli wyczerpane w przegladzie od praktyki ({str(exc)[:120]})")
    finally:
        await fetcher.aclose()
        activity.clear(job_id)
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in total.items()))
    kw._save_stats(job_id, total)
    return total
