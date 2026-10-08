"""Zadania wiedzy o roslinach:
- 'plants_sync'  : rejestr roslin z Kwiatownika + nowe rosliny wykryte w skladnikach przepisow (LLM),
- 'plant_research': dla kazdej rosliny: Wikipedia (kilka jezykow) + strony z innych krajow + ciekawostki
  zebrane wczesniej -> LLM wyciaga informacje z cytatami -> kontrola cytatow -> weryfikacja drugim modelem
  -> kopia pliku rosliny -> agregator (uklad sekcji) -> kontrola kopii -> zapis do Kwiatownika (z kopia zapasowa),
- 'plant_organize': ponowne uporzadkowanie wiedzy juz zapisanej w plikach (bez nowego zbierania),
- 'plant_merge'   : scalenie wiedzy z sieci z rozdzialami pliku rosliny (blok "scalone", Bielik -> inne modele).
Zbieranie skupia sie na LUKACH: podrozdzialy schematu bez tresci (app/knowledge/schema.py) -> zapytania
w wielu jezykach, takze chinskim i japonskim (app/knowledge/gaps.py), z historia zapytan (PlantQuery).
"""
import asyncio
import copy
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.knowledge import gaps as gapmod
from app.knowledge import merge, organize, photos, plantfile, schema
from app.knowledge.extract import extract_facts
from app.knowledge.registry import discover_new_plants, sync_kwiatownik
from app.knowledge.sections import fingerprint, norm_text, similar
from app.knowledge.verify import BATCH, verify_facts
from app.knowledge.wiki import WikiClient
from app.llm.router import AllModelsFailedError
from app.models import Item, ItemKind, Job, JobStatus, Plant, PlantFact, PlantQuery
from app.worker import activity, snapshots
from app.worker.jobs import job_log
from app.worker.runner import JobRunner

# slowa do szukania ludowych zastosowan rosliny w roznych jezykach (strony spoza Wikipedii)
FOLK_WORDS = {"pl": "medycyna ludowa zastosowanie", "uk": "народна медицина застосування",
              "ru": "народная медицина применение", "de": "Volksmedizin Verwendung",
              "fr": "médecine populaire utilisation", "cs": "lidové léčitelství použití",
              "lt": "liaudies medicina", "sk": "ľudové liečiteľstvo", "hu": "népi gyógyászat",
              "ro": "medicina populară", "bg": "народна медицина", "it": "medicina popolare usi",
              "es": "medicina popular usos", "en": "folk medicine traditional uses"}
_BUSY: set[str] = set()   # rosliny opracowywane teraz (zeby dwa zadania nie badaly tej samej naraz)
SKIP_DOMAINS = ("wikipedia.org", "wikimedia.org", "wikidata.org", "youtube.com", "facebook.com", "instagram.com",
                "pinterest.", "amazon.", "allegro.", "ebay.", "aliexpress.",
                # sklepy i katalogi chemikaliow (wyniki zapytan zh/ja) - nie zrodla wiedzy zielarskiej
                "chemicalbook.", "alibaba.", "1688.com", "made-in-china.", "taobao.", "tmall.", "jd.com",
                "rakuten.", "mercari.", "yahoo.co.jp/auction", "shopping.", "zhihu.com/market")


def _save_stats(job_id: int, stats: dict) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.stats_json = json.dumps(stats)
        s.add(job)
        s.commit()


def _progress(job_id: int, done: int, total: int) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.progress, job.total = done, total
        s.add(job)
        s.commit()


# ============================================================ rejestr
async def run_plants_sync(job_id: int, runner: JobRunner, llm=None) -> dict:
    log = lambda m: job_log(job_id, m)  # noqa: E731
    with Session(engine) as s:
        added = sync_kwiatownik(s, settings.kwiatownik_plants_dir)
        total = s.exec(select(func.count()).select_from(Plant)).one()
    log(f"Rosliny z Kwiatownika: {total} w rejestrze ({added} nowych wpisow)")
    created: list[str] = []
    if llm is not None:
        with Session(engine) as s:
            activity.set(job_id, "LLM rozpoznaje rosliny w skladnikach przepisow")
            try:
                created = await discover_new_plants(s, llm, log_fn=log)
            except AllModelsFailedError as exc:
                log(f"Modele maja limit - rozpoznawanie skladnikow przerwane ({str(exc)[:120]})")
    log(f"Nowe rosliny z przepisow: {len(created)}" + (f" ({', '.join(created[:15])})" if created else ""))
    with Session(engine) as s:
        ids = list(s.exec(select(Plant.id)).all())
    for pid in ids:                                    # pokrycie schematu (widok Rosliny, kolejnosc skanow)
        try:
            update_coverage(pid)
        except (OSError, ValueError) as exc:
            log(f"  pokrycie {pid}: {str(exc)[:80]}")
    stats = {"plants": total, "added_from_kwiatownik": added, "new_from_recipes": len(created)}
    _save_stats(job_id, stats)
    return stats


# ============================================================ zbieranie wiedzy
def facts_dump(plant_id: str) -> str:
    """Wszystkie informacje o roslinie w bazie Siedziby (do podgladu "przed/po" w zakladce "Na zywo")."""
    with Session(engine) as s:
        rows = s.exec(select(PlantFact).where(PlantFact.plant_id == plant_id).order_by(PlantFact.id)).all()
        data = [{"id": r.id, "status": r.status, "sekcja": r.section, "czesc": r.part, "tekst": r.text,
                 "cytat": r.quote, "zrodlo": r.source_name or r.source_url, "url": r.source_url, "jezyk": r.language,
                 "powod": r.reason, "wyciagnal": r.extracted_by, "zweryfikowal": r.verified_by} for r in rows]
    return json.dumps({k: v for k, v in [("roslina", plant_id), ("informacje", data)]}, ensure_ascii=False, indent=2) + "\n"


def _record_facts(plant_id: str, title: str, before: str, operation: str) -> None:
    snapshots.record(f"baza Siedziby: wiedza o roslinie {plant_id}", before, facts_dump(plant_id),
                     f"{operation}: {title}", file=f"informacje o roslinie - {title}", link=f"/plants/{plant_id}")


def _known_texts(plant_id: str, file_data: dict | None) -> list[str]:
    with Session(engine) as s:
        texts = list(s.exec(select(PlantFact.text).where(PlantFact.plant_id == plant_id,
                                                          PlantFact.status != "rejected")).all())
    if file_data:
        for key in ("opis", "ostrzezenia", "interakcje"):
            if isinstance(file_data.get(key), str):
                texts.append(file_data[key])
        texts += [c for c in (file_data.get("ciekawostki") or []) if isinstance(c, str)]
    return texts


def _store_facts(plant_id: str, facts, src_url: str, src_name: str, lang: str, model: str) -> int:
    """Zapis nowych informacji (bez powtorzen: ten sam odcisk albo bardzo podobne zdanie)."""
    added = 0
    with Session(engine) as s:
        existing = s.exec(select(PlantFact.fingerprint, PlantFact.text).where(PlantFact.plant_id == plant_id)).all()
        prints = {fp for fp, _ in existing}
        texts = [t for _, t in existing]
        for f in facts:
            fp = fingerprint(plant_id, f.text)
            if fp in prints or any(similar(f.text, t) >= 0.6 for t in texts):
                continue
            s.add(PlantFact(plant_id=plant_id, section=f.section, part=f.part, text=f.text, quote=f.quote,
                            source_url=src_url, source_name=src_name, language=lang, fingerprint=fp,
                            extracted_by=model))
            prints.add(fp)
            texts.append(f.text)
            added += 1
        s.commit()
    return added


def _names(plant: Plant) -> dict[str, str]:
    try:
        labels = json.loads(plant.names_json) if plant.names_json else {}
        titles = json.loads(plant.wiki_titles_json) if plant.wiki_titles_json else {}
    except ValueError:
        labels, titles = {}, {}
    return gapmod.plant_names(plant.nazwa_lat, plant.nazwa_pl, labels, titles)


def _recent_queries(plant_id: str) -> tuple[set[str], set[str]]:
    """(zapytania zadane w ostatnich KNOWLEDGE_REQUERY_DAYS dniach, adresy juz uzyte dla tej rosliny)."""
    since = datetime.now(timezone.utc) - timedelta(days=settings.knowledge_requery_days)
    with Session(engine) as s:
        rows = s.exec(select(PlantQuery).where(PlantQuery.plant_id == plant_id)).all()
        urls = set(s.exec(select(PlantFact.source_url).where(PlantFact.plant_id == plant_id)).all())
    done = set()
    for q in rows:
        at = q.created_at if q.created_at.tzinfo else q.created_at.replace(tzinfo=timezone.utc)
        if at >= since:
            done.add(q.query)
        if q.used_url:
            urls.add(q.used_url)
    return done, urls


def _log_query(plant_id: str, gq, results: int, used_url: str | None) -> int:
    with Session(engine) as s:
        row = PlantQuery(plant_id=plant_id, slot=gq.gap.path, language=gq.lang, query=gq.query, results=results,
                         used_url=used_url)
        s.add(row)
        s.commit()
        return row.id


def _query_facts(qid: int, n: int) -> None:
    with Session(engine) as s:
        row = s.get(PlantQuery, qid)
        if row:
            row.facts = n
            s.add(row)
            s.commit()


async def _web_sources(plant: Plant, log, file_data: dict | None = None, slots: list[str] | None = None) -> list[tuple]:
    """(url, nazwa, jezyk, tekst, luki, id_zapytania) z innych stron: ciekawostki juz zebrane przez Siedzibe
    + wyszukiwarka skupiona na LUKACH rosliny (czego brakuje w schemacie) w wielu jezykach, takze zh/ja."""
    out: list[tuple] = []
    keys = [k for k in (plant.nazwa_lat, plant.nazwa_pl) if k]
    with Session(engine) as s:
        for it in s.exec(select(Item).where(Item.kind == ItemKind.fact, col(Item.raw_text).is_not(None))).all():
            text = it.raw_text or ""
            if any(k.lower() in text.lower() for k in keys):
                out.append((it.url, urlparse(it.url).netloc, it.language or "?", text[:8000], None, None))
                if len(out) >= 3:
                    break
    if settings.knowledge_web_pages <= 0 or not plant.nazwa_lat:
        return out
    from app.discovery.search import search_all
    from app.scrapers.fetcher import Fetcher
    from app.scrapers.page import parse_page
    from app.worker.discover import build_backends
    backends = build_backends()
    if not backends:
        return out
    cov = coverage_for(plant.id, file_data)
    gaps = gapmod.find_gaps(file_data, cov["sloty"])
    if slots:                                          # szukanie konkretnych podrozdzialow (z panelu)
        wanted = [g for g in gaps if g.path in slots or any(g.path.startswith(x + ".") for x in slots)]
        gaps = wanted or [gapmod.Gap(x, (schema.SLOT_BY_PATH[x].title if x in schema.SLOT_BY_PATH else x),
                                     gapmod.SLOT_TOPIC.get(x, "lecznicze")) for x in slots]
    if not gaps:
        log("  luki: brak - wszystkie podrozdzialy maja tresc; szukam ogolnie")
        gaps = [gapmod.Gap("zastosowanie.medyczne", "Praktyczne zastosowanie - Medycyna", "lecznicze"),
                gapmod.Gap("ciekawostki", "Wiedza tajemna - Ciekawostki", "folklor")]
    names = _names(plant)
    langs = [x.strip() for x in settings.knowledge_search_langs.split(",") if x.strip()]
    done, used_urls = _recent_queries(plant.id)
    queries = gapmod.build_queries(gaps, names, langs, done, max_gaps=settings.knowledge_gap_slots)
    log(f"  luki ({len(gaps)}): " + ", ".join(g.title for g in gaps[:6]) + ("..." if len(gaps) > 6 else ""))
    if not queries:
        log("  wszystkie zapytania o luki byly zadane niedawno - pomijam wyszukiwarke")
        return out
    fetcher = Fetcher(settings.user_agent, settings.request_delay_s,
                      accept_language="pl,en;q=0.8,de;q=0.6,zh;q=0.5,ja;q=0.5,*;q=0.3")
    found = 0
    topics_done: set[str] = set()
    try:
        for gq in queries:
            if found >= settings.knowledge_web_pages:
                break
            if gq.gap.topic in topics_done:            # temat juz ma strone - nastepna luka
                continue
            results = await search_all(backends, gq.query, gq.lang, "", log_fn=log)
            used = None
            for r in results[:6]:
                if any(d in r.url for d in SKIP_DOMAINS) or r.url in used_urls or any(r.url == o[0] for o in out):
                    continue
                res = await fetcher.get(r.url)
                if not res.ok:
                    continue
                page = await asyncio.to_thread(parse_page, res.text, res.url)
                min_len = 300 if gq.lang in ("zh", "ja") else 800
                if len(page.text) < min_len or not gapmod.page_about_plant(page.text, names, gq.lang):
                    continue
                used = res.url
                focus = gapmod.gaps_for_focus(gaps, gq.gap.topic)
                qid = _log_query(plant.id, gq, len(results), used)
                out.append((res.url, urlparse(res.url).netloc, page.lang or gq.lang, page.text[:8000], focus, qid))
                found += 1
                topics_done.add(gq.gap.topic)
                log(f"  [{gq.lang}] {gq.gap.title}: {urlparse(res.url).netloc}")
                break
            if used is None:
                _log_query(plant.id, gq, len(results), None)
    finally:
        await fetcher.aclose()
    return out


async def research_plant(plant_id: str, llm, wiki: WikiClient, log, job_id: int = 0,
                         slots: list[str] | None = None) -> dict:
    """Zbiera informacje o jednej roslinie (bez zapisu do pliku). Zwraca statystyki."""
    stats = {"sources": 0, "facts": 0, "rejected_quotes": 0}
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        s.expunge(plant)
    file_data, _ = plantfile.read_plant(settings.kwiatownik_plants_dir, plant_id)
    known = _known_texts(plant_id, file_data)
    sources: list[tuple[str, str, str, str]] = []

    # --- Wikipedia (przez Wikidata: jeden gatunek, wiele jezykow)
    try:
        wp = await wiki.find_plant(plant.nazwa_lat, plant.nazwa_pl)
    except Exception as exc:
        log(f"  Wikidata niedostepna: {str(exc)[:100]}")
        wp = None
    if wp:
        with Session(engine) as s:
            row = s.get(Plant, plant_id)
            row.wikidata, row.wiki_titles_json = wp.qid, json.dumps(wp.titles, ensure_ascii=False)
            if wp.labels:
                row.names_json = json.dumps(wp.labels, ensure_ascii=False)
            if not row.nazwa_lat and wp.taxon:
                row.nazwa_lat = wp.taxon
            s.add(row)
            s.commit()
        if settings.knowledge_photos:
            await collect_photos(plant_id, wp, wiki, log, file_data)
        langs = [x.strip() for x in settings.knowledge_languages.split(",") if x.strip() in wp.titles]
        langs = langs[:settings.knowledge_wiki_langs]
        langs += [x.strip() for x in settings.knowledge_asia_langs.split(",")       # chinska / japonska Wikipedia
                  if x.strip() in wp.titles and x.strip() not in langs]
        for lang in langs:
            try:
                art = await wiki.article(lang, wp.titles[lang])
            except Exception as exc:                   # jedna wersja jezykowa niedostepna - reszta dalej
                log(f"  Wikipedia ({lang}): {str(exc)[:100]}")
                continue
            if art:
                sources.append((art.url, f"Wikipedia ({lang}): {art.title}", lang, art.text, None, None))
        log(f"  Wikipedia: {wp.qid}, artykuly: {', '.join(src[2] for src in sources) or 'brak'}")
    else:
        log("  Wikipedia: nie znaleziono gatunku w Wikidata")
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        s.expunge(plant)
    sources += await _web_sources(plant, log, file_data, slots)

    for k, (url, name, lang, text, focus, qid) in enumerate(sources, 1):
        if job_id:
            activity.set(job_id, f"{plant.nazwa_pl}: zrodlo {k}/{len(sources)} - LLM wyciaga informacje", name)
        try:
            facts, rejected, model = await extract_facts(llm, plant.nazwa_pl, plant.nazwa_lat, name, lang, text, known,
                                                         focus)
        except AllModelsFailedError:
            raise
        except Exception as exc:                       # zly JSON jednego zrodla nie zatrzymuje reszty
            log(f"  [blad] {name}: {str(exc)[:120]}")
            continue
        added = _store_facts(plant_id, facts, url, name, lang, model)
        if qid:
            _query_facts(qid, added)
        known += [f.text for f in facts]
        stats["sources"] += 1
        stats["facts"] += added
        stats["rejected_quotes"] += len(rejected)
        log(f"  {name}: {added} nowych informacji" + (f", odrzucono {len(rejected)} (cytat/liczby)" if rejected else ""))
    return stats


async def verify_plant(plant_id: str, llm, log, job_id: int = 0) -> dict:
    """Weryfikacja nowych informacji drugim modelem (innym niz ten, ktory je wyciagnal)."""
    stats = {"verified": 0, "rejected": 0}
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        facts = s.exec(select(PlantFact).where(PlantFact.plant_id == plant_id, PlantFact.status == "new")).all()
        for f in facts:
            s.expunge(f)
        name, lat = plant.nazwa_pl, plant.nazwa_lat
    for i in range(0, len(facts), BATCH):
        batch = facts[i:i + BATCH]
        if job_id:
            activity.set(job_id, f"{name}: weryfikacja drugim modelem", f"informacje {i + 1}-{i + len(batch)} z {len(facts)}")
        avoid = {(f.extracted_by or "").replace("/", ":", 1) for f in batch if f.extracted_by}
        verdicts, model = await verify_facts(llm, name, lat, batch, avoid=avoid)
        with Session(engine) as s:
            for f in batch:
                if f.id not in verdicts:
                    continue
                ok, why = verdicts[f.id]
                row = s.get(PlantFact, f.id)
                row.status, row.reason, row.verified_by = ("verified" if ok else "rejected"), why or None, model
                s.add(row)
                stats["verified" if ok else "rejected"] += 1
            s.commit()
    if facts:
        log(f"  weryfikacja: potwierdzone {stats['verified']}, odrzucone {stats['rejected']}")
    return stats


async def _organized(cand: dict, name: str, latin: str | None, llm, log) -> dict:
    """Agregator: surowe informacje z kopii -> sekcje/punkty/zrodla (LLM + kontrola kodem, inaczej deterministycznie)."""
    facts = cand["wiedza"].get("fakty") or []
    layout, how = await organize.organize(llm, name, latin, facts, log)
    points = sum(len(sec["punkty"]) for sec in layout["sekcje"])
    log(f"  agregator ({how}): {len(facts)} informacji -> {points} punktow w {len(layout['sekcje'])} sekcjach")
    return plantfile.with_layout(cand, layout, how)


async def _merged(cand: dict, name: str, latin: str | None, llm, log) -> dict:
    """Scalanie wiedzy z sieci z rozdzialami pliku (blok "scalone"); Bielik 11B -> Bielik 4.5B -> inne modele."""
    from app.worker.ollama_pull import merge_models, missing_ollama_models
    prefer = merge_models()
    try:                                              # Bielik juz jest -> poprawia scalenia zrobione w chmurze
        absent = set(await missing_ollama_models(llm)) if llm is not None else {"?"}
        redo = bool(prefer) and not any(m.split(":", 1)[1] in absent for m in prefer if m.startswith("ollama:"))
    except Exception:
        redo = False
    try:
        sc = await merge.build_merged(llm, cand, name, latin, prefer=prefer, log=log, redo_other_models=redo,
                                      budget_s=settings.merge_plant_budget_s or None,
                                      prefer_timeout=settings.merge_timeout_s or None)
    except AllModelsFailedError:
        raise
    except Exception as exc:                          # scalanie nie moze zatrzymac zapisu wiedzy
        log(f"  scalanie z rozdzialami: blad ({str(exc)[:120]}) - zostaje poprzednie")
        return cand
    return merge.with_merged(cand, sc)


def web_section_counts(plant_id: str) -> dict[str, int]:
    """Zweryfikowane/zapisane informacje z sieci wg sekcji (do pokrycia schematu)."""
    with Session(engine) as s:
        rows = s.exec(select(PlantFact.section, func.count()).where(
            PlantFact.plant_id == plant_id, col(PlantFact.status).in_(["verified", "applied"]))
            .group_by(PlantFact.section)).all()
    return dict(rows)


def coverage_for(plant_id: str, data: dict | None = None) -> dict:
    """Pokrycie schematu rosliny: co jest w pliku (recznie / z sieci), czego brakuje."""
    if data is None:
        data, _ = plantfile.read_plant(settings.kwiatownik_plants_dir, plant_id)
    return schema.coverage(data, web_section_counts(plant_id))


def update_coverage(plant_id: str, data: dict | None = None) -> dict:
    cov = coverage_for(plant_id, data)
    with Session(engine) as s:
        row = s.get(Plant, plant_id)
        if row:
            row.coverage, row.gaps_json = cov["procent"], json.dumps(cov["puste"], ensure_ascii=False)
            s.add(row)
            s.commit()
    return cov


def _mark_error(plant_id: str, errors: list[str]) -> None:
    with Session(engine) as s:
        row = s.get(Plant, plant_id)
        row.status, row.note = "error", "; ".join(errors[:3])[:300]
        s.add(row)
        s.commit()


def _photos_of(plant) -> tuple[list[dict], list[dict]]:
    try:
        data = json.loads(plant.photos_json) if plant.photos_json else {}
    except ValueError:
        data = {}
    return data.get("zdjecia") or [], data.get("autorzy") or []


def _rejected(plant_id: str, original: dict | None, cand: dict, errors: list[str], operation: str, log) -> str:
    """Kopia odrzucona przez kontrole: plik bez zmian, ale migawka "przed/po" pokazuje, co chcielismy zapisac."""
    log("  KONTROLA KOPII NIE PRZESZLA - plik bez zmian: " + "; ".join(errors[:5]))
    _mark_error(plant_id, errors)
    snapshots.record_dict(plantfile.plant_path(settings.kwiatownik_plants_dir, plant_id), original, cand,
                          f"{operation}: {plant_id}", applied=False, note="; ".join(errors[:20]),
                          link=f"/plants/{plant_id}")
    return "invalid"


def _save_candidate(plant_id: str, original: dict | None, cand: dict, crlf: bool, log, facts=(),
                    operation: str = "dopisanie wiedzy") -> str:
    """Kontrola kopii -> kopia zapasowa -> zapis. Wspolne dla wiedzy i zdjec."""
    errors = plantfile.validate_candidate(original, cand, plant_id)
    if errors:
        return _rejected(plant_id, original, cand, errors, operation, log)
    if cand == original:
        return "nothing"
    plants_dir = Path(settings.kwiatownik_plants_dir)
    backup = plantfile.write_plant(plants_dir, plant_id, cand, crlf, Path(settings.backups_dir) / "plants",
                                   operation)
    with Session(engine) as s:
        for f in facts:
            row = s.get(PlantFact, f.id)
            row.status = "applied"
            s.add(row)
        row = s.get(Plant, plant_id)
        if row:
            row.status, row.last_apply_at, row.note = "applied", datetime.now(timezone.utc), None
            s.add(row)
        s.commit()
    log(f"  zapisano do Kwiatownika: {plantfile.diff_summary(original, cand)}"
        + (f" (kopia zapasowa: {backup.name})" if backup else ""))
    return "applied"


async def apply_plant(plant_id: str, log, llm=None) -> str:
    """Kopia pliku rosliny + zweryfikowane informacje -> agregator -> zdjecia -> kontrola kopii -> zapis."""
    plants_dir = Path(settings.kwiatownik_plants_dir)
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        facts = s.exec(select(PlantFact).where(PlantFact.plant_id == plant_id, PlantFact.status == "verified")
                       .order_by(PlantFact.id)).all()
        for f in facts:
            s.expunge(f)
        s.expunge(plant)
    pics, credits = _photos_of(plant)
    original, crlf = plantfile.read_plant(plants_dir, plant_id)
    if not facts and not (original and (pics or credits)):
        return "nothing"
    if original is None:
        applied_before = _count(plant_id, "applied")
        from_wiki = any("wikipedia.org" in f.source_url for f in facts)
        if len(facts) + applied_before < settings.knowledge_min_facts_new or not from_wiki:
            log(f"  plik nowej rosliny poczeka: {len(facts)} informacji"
                + ("" if from_wiki else ", brak potwierdzenia w Wikipedii"))
            return "waiting"
    if facts:
        base = plantfile.skeleton(plant_id, plant.nazwa_pl, plant.nazwa_lat)
        cand = plantfile.build_candidate(original, base, facts)
        cand = await _organized(cand, plant.nazwa_pl, plant.nazwa_lat, llm, log)
        if settings.knowledge_auto_merge:
            cand = await _merged(cand, plant.nazwa_pl, plant.nazwa_lat, llm, log)
    else:
        cand = copy.deepcopy(original)
    cand = plantfile.with_photos(cand, original, pics, credits)
    outcome = _save_candidate(plant_id, original, cand, crlf, log, facts,
                              "dopisanie wiedzy" if facts else "zdjecia")
    if outcome == "applied":
        _after_write(plant_id, cand)
    return outcome


def _after_write(plant_id: str, cand: dict) -> None:
    with Session(engine) as s:
        row = s.get(Plant, plant_id)
        if row and cand.get("scalone"):
            row.last_merge_at = datetime.now(timezone.utc)
            s.add(row)
            s.commit()
    update_coverage(plant_id, cand)


async def collect_photos(plant_id: str, wp, wiki: WikiClient, log, file_data: dict | None = None) -> int:
    """Zdjecia gatunku z Wikipedii/Commons -> Plant.photos_json (do zapisu w pliku przy apply)."""
    langs = [x.strip() for x in settings.knowledge_languages.split(",") if x.strip()]
    try:
        pics = await photos.find_photos(wiki, wp.taxon, wp.image, wp.commons_category, wp.titles, langs, log,
                                        settings.knowledge_max_photos)
        manual = (file_data or {}).get("url") if not plantfile.photos_owned(file_data) else None
        credits = await photos.credits_for_urls(wiki, manual) if isinstance(manual, dict) else []
    except Exception as exc:                           # zdjecia nie moga zatrzymac zbierania wiedzy
        log(f"  zdjecia: blad ({str(exc)[:100]})")
        return 0
    with Session(engine) as s:
        row = s.get(Plant, plant_id)
        row.photos_json = json.dumps({"zdjecia": [p.as_dict() for p in pics], "autorzy": credits}, ensure_ascii=False)
        row.photos_at = datetime.now(timezone.utc)
        s.add(row)
        s.commit()
    if credits:
        log(f"  zdjecia: galeria reczna zostaje, opisano autorow i licencje {len(credits)} zdjec")
    else:
        log(f"  zdjecia z Wikipedii: {len(pics)}" + (f" ({', '.join(p.podpis for p in pics)})" if pics else ""))
    return len(pics) or len(credits)


async def photos_plant(plant_id: str, wiki: WikiClient, log) -> str:
    """Tylko zdjecia (bez LLM): Wikidata -> Commons -> kontrola -> zapis w istniejacym pliku rosliny."""
    plants_dir = Path(settings.kwiatownik_plants_dir)
    original, crlf = plantfile.read_plant(plants_dir, plant_id)
    if original is None:
        return "nofile"
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        name, latin = (plant.nazwa_pl, plant.nazwa_lat) if plant else (original.get("nazwa_pl"),
                                                                        original.get("nazwa_lat"))
    wp = await wiki.find_plant(latin, name)
    if not wp:
        log("  nie znaleziono gatunku w Wikidata")
        return "notfound"
    if plant is None:
        return "nothing"
    if not await collect_photos(plant_id, wp, wiki, log, original):
        return "nothing"
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        s.expunge(plant)
    pics, credits = _photos_of(plant)
    cand = plantfile.with_photos(copy.deepcopy(original), original, pics, credits)
    return _save_candidate(plant_id, original, cand, crlf, log, operation="zdjecia")


async def organize_plant(plant_id: str, log, llm=None) -> str:
    """Tylko uporzadkowanie wiedzy juz zapisanej w pliku rosliny (bez zbierania). Zwraca wynik."""
    plants_dir = Path(settings.kwiatownik_plants_dir)
    original, crlf = plantfile.read_plant(plants_dir, plant_id)
    if not original or not ((original.get("wiedza") or {}).get("fakty")):
        return "nothing"
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        name, latin = (plant.nazwa_pl, plant.nazwa_lat) if plant else (original.get("nazwa_pl") or plant_id,
                                                                        original.get("nazwa_lat"))
    cand = await _organized(copy.deepcopy(original), name, latin, llm, log)
    if settings.knowledge_auto_merge:
        cand = await _merged(cand, name, latin, llm, log)
    errors = plantfile.validate_candidate(original, cand, plant_id)
    if errors:
        return _rejected(plant_id, original, cand, errors, "uporzadkowanie", log)
    backup = plantfile.write_plant(plants_dir, plant_id, cand, crlf, Path(settings.backups_dir) / "plants",
                                   "uporzadkowanie")
    log("  plik uporzadkowany" + (f" (kopia zapasowa: {backup.name})" if backup else ""))
    _after_write(plant_id, cand)
    return "organized"


async def merge_plant(plant_id: str, log, llm=None) -> str:
    """Tylko scalenie wiedzy z sieci (juz zapisanej w pliku) z rozdzialami i podrozdzialami strony."""
    plants_dir = Path(settings.kwiatownik_plants_dir)
    original, crlf = plantfile.read_plant(plants_dir, plant_id)
    if not original or not ((original.get("wiedza") or {}).get("fakty")):
        return "nothing"
    with Session(engine) as s:
        plant = s.get(Plant, plant_id)
        name, latin = (plant.nazwa_pl, plant.nazwa_lat) if plant else (original.get("nazwa_pl") or plant_id,
                                                                        original.get("nazwa_lat"))
    cand = copy.deepcopy(original)
    if not cand["wiedza"].get("sekcje"):               # wiedza sprzed agregatora - najpierw uklad
        cand = await _organized(cand, name, latin, llm, log)
    cand = await _merged(cand, name, latin, llm, log)
    if cand == original:
        log("  bez zmian")
        update_coverage(plant_id, cand)
        return "nothing"
    errors = plantfile.validate_candidate(original, cand, plant_id)
    if errors:
        return _rejected(plant_id, original, cand, errors, "scalenie z rozdzialami", log)
    backup = plantfile.write_plant(plants_dir, plant_id, cand, crlf, Path(settings.backups_dir) / "plants",
                                   "scalenie z rozdzialami")
    n = len((cand.get("scalone") or {}).get("pola") or {})
    log(f"  scalono z rozdzialami: {n} podrozdzialow" + (f" (kopia zapasowa: {backup.name})" if backup else ""))
    _after_write(plant_id, cand)
    return "merged"


def unmerge_plant(plant_id: str) -> bool:
    """Usuwa blok "scalone" (strona wraca do recznych tekstow; wiedza z sieci zostaje w swojej zakladce)."""
    plants_dir = Path(settings.kwiatownik_plants_dir)
    original, crlf = plantfile.read_plant(plants_dir, plant_id)
    if not original or "scalone" not in original:
        return False
    cand = copy.deepcopy(original)
    cand.pop("scalone")
    plantfile.write_plant(plants_dir, plant_id, cand, crlf, Path(settings.backups_dir) / "plants",
                          "cofniecie scalenia")
    update_coverage(plant_id, cand)
    return True


def organized_plant_ids() -> list[str]:
    """Rosliny, ktorych pliki maja wiedze z Siedziby (kandydaci do uporzadkowania)."""
    from app.knowledge.registry import kwiatownik_plants
    out = []
    for p in kwiatownik_plants(settings.kwiatownik_plants_dir):
        if p["origin"] == "archiwum":
            continue
        data, _ = plantfile.read_plant(settings.kwiatownik_plants_dir, p["id"])
        if data and (data.get("wiedza") or {}).get("fakty"):
            out.append(p["id"])
    return out


def _count(plant_id: str, status: str) -> int:
    with Session(engine) as s:
        return s.exec(select(func.count()).select_from(PlantFact).where(
            PlantFact.plant_id == plant_id, PlantFact.status == status)).one()


def plants_to_research(limit: int) -> list[str]:
    """Najpierw rosliny nigdy nie badane (z Kwiatownika, potem nowe z wieloma przepisami), potem te z
    najwiekszymi lukami w schemacie (najmniejsze pokrycie), na koncu najdawniej badane."""
    with Session(engine) as s:
        rows = s.exec(select(Plant)).all()
    never = sorted([p for p in rows if p.last_research_at is None], key=lambda p: -(p.mentions or 0))
    rest = sorted([p for p in rows if p.last_research_at is not None],
                  key=lambda p: (p.coverage if p.coverage is not None else 0, p.last_research_at))
    return [p.id for p in never + rest if p.id not in _BUSY][:limit]


async def run_plant_research(job_id: int, runner: JobRunner, llm, plant_ids: list[str] | None = None,
                             limit: int = 10, wiki: WikiClient | None = None, slots: list[str] | None = None) -> dict:
    """slots: szukaj tylko tych brakujacych podrozdzialow (np. ["zastosowanie.rzemieslnicze"])."""
    if llm is None:
        raise RuntimeError("Zbieranie wiedzy wymaga LLM - ustaw klucze GEMINI/GROQ w .env")
    log = lambda m: job_log(job_id, m)  # noqa: E731
    with Session(engine) as s:
        if not s.exec(select(func.count()).select_from(Plant)).one():
            sync_kwiatownik(s, settings.kwiatownik_plants_dir)
    ids = list(plant_ids) if plant_ids else plants_to_research(limit)
    own_wiki = wiki is None
    wiki = wiki or WikiClient(settings.user_agent)
    total = {"plants": 0, "sources": 0, "facts": 0, "verified": 0, "rejected": 0, "applied": 0, "waiting": 0,
             "invalid": 0}
    log(f"Rosliny do opracowania: {len(ids)}; jezyki Wikipedii: {settings.knowledge_languages} "
        f"(max {settings.knowledge_wiki_langs}), inne strony: {settings.knowledge_web_pages}")
    try:
        for n, pid in enumerate(ids, 1):
            if runner.should_stop(job_id):
                log("Zatrzymano")
                break
            with Session(engine) as s:
                plant = s.get(Plant, pid)
                title = plant.nazwa_pl if plant else pid
            if not plant:
                continue
            if pid in _BUSY:
                log(f"[{n}/{len(ids)}] {title} - opracowywana teraz w innym zadaniu, pomijam")
                continue
            activity.set(job_id, f"roslina {n}/{len(ids)}", title)
            log(f"[{n}/{len(ids)}] {title}")
            _BUSY.add(pid)
            facts_before = facts_dump(pid)
            try:
                st = await research_plant(pid, llm, wiki, log, job_id, slots=slots)
                vs = await verify_plant(pid, llm, log, job_id)
                _record_facts(pid, title, facts_before, "zbieranie i weryfikacja informacji")
                outcome = await apply_plant(pid, log, llm) if settings.knowledge_auto_apply else "nothing"
            except AllModelsFailedError as exc:
                _record_facts(pid, title, facts_before, "zbieranie informacji (przerwane - limity modeli)")
                log(f"Limity modeli wyczerpane - przerywam ({str(exc)[:120]}). Ponow zadanie pozniej "
                    "(zebrane informacje zostaja i zostana zweryfikowane przy nastepnym razie).")
                break
            except Exception as exc:  # noqa: BLE001 - blad jednej rosliny nie przerywa calego zadania
                logging.getLogger(__name__).exception("Zbieranie wiedzy: %s", pid)
                _record_facts(pid, title, facts_before, "zbieranie informacji (przerwane bledem)")
                _mark_error(pid, [f"{type(exc).__name__}: {exc}"[:300]])
                log(f"  BLAD przy tej roslinie ({type(exc).__name__}: {str(exc)[:160]}) - przechodze do nastepnej; "
                    "zebrane informacje zostaja w bazie")
                total["errors"] = total.get("errors", 0) + 1
                _progress(job_id, n, len(ids))
                continue
            finally:
                _BUSY.discard(pid)
            with Session(engine) as s:
                row = s.get(Plant, pid)
                row.last_research_at = datetime.now(timezone.utc)
                if row.status not in ("applied", "error"):
                    row.status = "researched"
                s.add(row)
                s.commit()
            total["plants"] += 1
            for k in ("sources", "facts"):
                total[k] += st[k]
            for k in ("verified", "rejected"):
                total[k] += vs[k]
            if outcome in total:
                total[outcome] += 1
            _progress(job_id, n, len(ids))
    finally:
        if own_wiki:
            await wiki.aclose()
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in total.items()))
    _save_stats(job_id, total)
    return total


async def run_plant_apply(job_id: int, runner: JobRunner, llm=None, plant_ids: list[str] | None = None) -> dict:
    """Zapis zweryfikowanych informacji do plikow roslin (np. gdy KNOWLEDGE_AUTO_APPLY=false)."""
    log = lambda m: job_log(job_id, m)  # noqa: E731
    with Session(engine) as s:
        ids = plant_ids or list(s.exec(select(PlantFact.plant_id).where(PlantFact.status == "verified")
                                       .distinct()).all())
    out = {"applied": 0, "waiting": 0, "invalid": 0, "nothing": 0}
    for pid in ids:
        log(pid)
        out[await apply_plant(pid, log, llm)] += 1
    _save_stats(job_id, out)
    return out


async def run_plant_photos(job_id: int, runner: JobRunner, llm=None, plant_ids: list[str] | None = None,
                           wiki: WikiClient | None = None) -> dict:
    """Zdjecia z Wikipedii dla roslin z plikami w Kwiatowniku (bez LLM - szybko)."""
    log = lambda m: job_log(job_id, m)  # noqa: E731
    from app.knowledge.registry import kwiatownik_plants
    with Session(engine) as s:
        sync_kwiatownik(s, settings.kwiatownik_plants_dir)
    ids = list(plant_ids) if plant_ids else [p["id"] for p in kwiatownik_plants(settings.kwiatownik_plants_dir)
                                             if p["origin"] != "archiwum"]
    own_wiki = wiki is None
    wiki = wiki or WikiClient(settings.user_agent)
    out = {"applied": 0, "nothing": 0, "invalid": 0, "notfound": 0, "nofile": 0, "error": 0}
    log(f"Zdjecia z Wikipedii dla {len(ids)} roslin (galerie wpisane recznie zostaja - dostaja opis autorow)")
    try:
        for n, pid in enumerate(ids, 1):
            if runner.should_stop(job_id):
                log("Zatrzymano")
                break
            activity.set(job_id, f"zdjecia {n}/{len(ids)}", pid)
            log(f"[{n}/{len(ids)}] {pid}")
            try:
                out[await photos_plant(pid, wiki, log)] += 1
            except Exception as exc:
                log(f"  [blad] {str(exc)[:120]}")
                out["error"] += 1
            _progress(job_id, n, len(ids))
    finally:
        if own_wiki:
            await wiki.aclose()
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    _save_stats(job_id, out)
    return out


async def run_plant_merge(job_id: int, runner: JobRunner, llm=None, plant_ids: list[str] | None = None) -> dict:
    """Scalanie wiedzy z sieci z rozdzialami plikow roslin (Bielik 11B -> Bielik 4.5B -> inne modele)."""
    from app.worker.ollama_pull import merge_models
    log = lambda m: job_log(job_id, m)  # noqa: E731
    ids = list(plant_ids) if plant_ids else organized_plant_ids()
    out = {"merged": 0, "invalid": 0, "nothing": 0, "busy": 0}
    log(f"Pliki do scalenia: {len(ids)}; modele scalania: {', '.join(merge_models()) or '-'}, potem zwykla lista"
        + ("" if llm else " (bez LLM - scalenie regulami)"))
    if llm is not None:
        from app.worker.ollama_pull import missing_ollama_models
        try:
            absent = set(await missing_ollama_models(llm))
        except Exception:
            absent = set()
        lacking = [m for m in merge_models() if m.startswith("ollama:") and m.split(":", 1)[1] in absent]
        if lacking:
            with Session(engine) as s:
                pulls = {j.kind.split(":", 1)[1]: j for j in s.exec(select(Job).where(
                    col(Job.kind).startswith("ollama_pull:"), col(Job.status).in_([JobStatus.queued, JobStatus.running]))).all()}
            for m in lacking:
                name = m.split(":", 1)[1]
                j = pulls.get(name)
                state = (f"pobiera sie teraz (zadanie #{j.id}: {j.progress or 0}/{j.total or '?'} MB)" if j else
                         "nie pobiera sie (tuz po starcie Siedziby pobieranie rusza samo po kilku sekundach; inaczej: "
                         "zakladka LLM, pole 'Pobierz model')")
                log(f"UWAGA: brak w Ollamie modelu scalania {name}: {state}")
            log("Do czasu pobrania scalaja modele z chmury (Gemini/Groq). Po pobraniu uruchom scalanie ponownie - "
                "podrozdzialy scalone w chmurze zostana poprawione Bielikiem.")
    for n, pid in enumerate(ids, 1):
        if runner.should_stop(job_id):
            log("Zatrzymano")
            break
        if pid in _BUSY:
            log(f"[{n}/{len(ids)}] {pid} - opracowywana w innym zadaniu, pomijam")
            out["busy"] += 1
            continue
        activity.set(job_id, f"scalanie {n}/{len(ids)}", pid)
        log(f"[{n}/{len(ids)}] {pid}")
        _BUSY.add(pid)
        try:
            out[await merge_plant(pid, log, llm)] += 1
        except AllModelsFailedError as exc:
            log(f"Limity modeli wyczerpane - przerywam ({str(exc)[:120]})")
            break
        finally:
            _BUSY.discard(pid)
        _progress(job_id, n, len(ids))
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    _save_stats(job_id, out)
    return out


async def run_plant_organize(job_id: int, runner: JobRunner, llm=None, plant_ids: list[str] | None = None) -> dict:
    """Agregator dla plikow, ktore juz maja wiedze (np. zebrana przed jego wprowadzeniem)."""
    log = lambda m: job_log(job_id, m)  # noqa: E731
    ids = list(plant_ids) if plant_ids else organized_plant_ids()
    out = {"organized": 0, "invalid": 0, "nothing": 0, "busy": 0}
    log(f"Pliki do uporzadkowania: {len(ids)}" + ("" if llm else " (bez LLM - uklad deterministyczny)"))
    for n, pid in enumerate(ids, 1):
        if runner.should_stop(job_id):
            log("Zatrzymano")
            break
        if pid in _BUSY:
            log(f"[{n}/{len(ids)}] {pid} - opracowywana w innym zadaniu, pomijam")
            out["busy"] += 1
            continue
        activity.set(job_id, f"porzadkowanie {n}/{len(ids)}", pid)
        log(f"[{n}/{len(ids)}] {pid}")
        _BUSY.add(pid)
        try:
            out[await organize_plant(pid, log, llm)] += 1
        finally:
            _BUSY.discard(pid)
        _progress(job_id, n, len(ids))
    log("Statystyki: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    _save_stats(job_id, out)
    return out
