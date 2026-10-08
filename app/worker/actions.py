"""Uruchamianie i ponawianie zadan (wspolne dla panelu, API, harmonogramu i wznawiania po restarcie).

Kazde zadanie zapisuje swoje parametry (Job.params_json), dzieki czemu mozna je "Ponowic".
Ponowiony skan kontynuuje od zapisanej kolejki adresow (Job.state_json) i pomija strony juz zebrane.
"""
import json
from dataclasses import asdict

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.discovery.finder import DiscoveryConfig
from app.models import Job, JobStatus, Source
from app.worker.agents import run_agent_audit, run_agent_deploy, run_agent_dispatch
from app.worker.archive_import import run_import_k1
from app.worker.discover import run_discover
from app.worker.export import run_enrich, run_export, run_translate
from app.worker.jobs import job_log
from app.worker.knowledge import (run_plant_apply, run_plant_merge, run_plant_organize, run_plant_photos,
                                  run_plant_research, run_plants_sync)
from app.worker.marketing import run_fb_ideas, run_fb_post, run_fb_review
from app.worker.ollama_pull import run_ollama_pull
from app.worker.scan import run_build_profile, run_fetch_list, run_harvest, run_scan, run_scan_pages

ACTIVE = [JobStatus.queued, JobStatus.running]


def _active_job(s: Session, source_id: int | None, kind: str) -> Job | None:
    return s.exec(select(Job).where(Job.source_id == source_id, Job.kind == kind,
                                    col(Job.status).in_(ACTIVE))).first()


def _create(state, source_id: int | None, kind: str, total: int | None, work_factory, dedupe: bool = True,
            params: dict | None = None, retry_of: int | None = None, bypass_queue: bool = False) -> int:
    with Session(engine) as s:
        existing = _active_job(s, source_id, kind) if dedupe else None
        if existing:
            return existing.id
        job = Job(kind=kind, source_id=source_id, total=total, status=JobStatus.queued,
                  params_json=json.dumps(params or {}, ensure_ascii=False), retry_of=retry_of)
        s.add(job)
        s.commit()
        jid = job.id
    runner = state.jobs
    runner.start(jid, lambda: work_factory(jid, runner, state.llm), bypass_queue=bypass_queue)
    return jid


def start_scan(state, source_id: int, max_pages: int | None = None, resume_from: int | None = None,
               retry_of: int | None = None) -> int:
    with Session(engine) as s:
        total = max_pages or s.get(Source, source_id).max_pages
    return _create(state, source_id, "scan", total,
                   lambda jid, runner, llm: run_scan(jid, source_id, runner, llm, max_pages, resume_from),
                   params={"max_pages": max_pages}, retry_of=retry_of)


def start_harvest(state, source_id: int, max_pages: int | None = None, resume_from: int | None = None,
                  retry_of: int | None = None) -> int:
    """Pelne pobieranie strony gotowym profilem scrapera (zadanie 'harvest')."""
    return _create(state, source_id, "harvest", max_pages or settings.harvest_max_pages,
                   lambda jid, runner, llm: run_harvest(jid, source_id, runner, llm, max_pages, resume_from),
                   params={"max_pages": max_pages}, retry_of=retry_of)


def start_plants_sync(state, retry_of: int | None = None) -> int:
    return _create(state, None, "plants_sync", None,
                   lambda jid, runner, llm: run_plants_sync(jid, runner, llm), params={}, retry_of=retry_of)


def start_plant_research(state, plant_ids: list[str] | None = None, limit: int = 10, slots: list[str] | None = None,
                         retry_of: int | None = None) -> int:
    """Zbieranie wiedzy o roslinach (Wikipedia + inne strony) -> weryfikacja -> zapis do plikow Kwiatownika."""
    return _create(state, None, "plant_research", len(plant_ids) if plant_ids else limit,
                   lambda jid, runner, llm: run_plant_research(jid, runner, llm, plant_ids, limit, slots=slots),
                   dedupe=not plant_ids and not slots, params={"plant_ids": plant_ids, "limit": limit, "slots": slots},
                   retry_of=retry_of)


def start_plant_apply(state, plant_ids: list[str] | None = None, retry_of: int | None = None) -> int:
    return _create(state, None, "plant_apply", None,
                   lambda jid, runner, llm: run_plant_apply(jid, runner, llm, plant_ids),
                   params={"plant_ids": plant_ids}, retry_of=retry_of)


def start_plant_photos(state, plant_ids: list[str] | None = None, retry_of: int | None = None) -> int:
    """Zdjecia z Wikipedii/Commons do galerii roslin (bez LLM)."""
    return _create(state, None, "plant_photos", len(plant_ids) if plant_ids else None,
                   lambda jid, runner, llm: run_plant_photos(jid, runner, llm, plant_ids),
                   dedupe=not plant_ids, params={"plant_ids": plant_ids}, retry_of=retry_of)


def start_plant_organize(state, plant_ids: list[str] | None = None, retry_of: int | None = None) -> int:
    """Agregator: porzadkuje wiedze juz zapisana w plikach roslin (sekcje, laczenie powtorzen, numerowane zrodla)."""
    return _create(state, None, "plant_organize", len(plant_ids) if plant_ids else None,
                   lambda jid, runner, llm: run_plant_organize(jid, runner, llm, plant_ids),
                   dedupe=not plant_ids, params={"plant_ids": plant_ids}, retry_of=retry_of)


def start_plant_merge(state, plant_ids: list[str] | None = None, retry_of: int | None = None) -> int:
    """Scalanie wiedzy z sieci z rozdzialami plikow roslin (Bielik -> inne modele)."""
    return _create(state, None, "plant_merge", len(plant_ids) if plant_ids else None,
                   lambda jid, runner, llm: run_plant_merge(jid, runner, llm, plant_ids),
                   dedupe=not plant_ids, params={"plant_ids": plant_ids}, retry_of=retry_of)


def start_fetch_list(state, source_id: int, urls: list[str], retry_of: int | None = None) -> int:
    return _create(state, source_id, "fetch_list", len(urls),
                   lambda jid, runner, llm: run_fetch_list(jid, source_id, urls, runner, llm),
                   params={"urls": urls}, retry_of=retry_of)


def start_scan_pages(state, source_id: int, urls: list[str], depth: int = 1, retry_of: int | None = None) -> int:
    return _create(state, source_id, "scan_pages", len(urls), dedupe=False,
                   work_factory=lambda jid, runner, llm: run_scan_pages(jid, source_id, urls, runner, llm, depth),
                   params={"urls": urls, "depth": depth}, retry_of=retry_of)


def start_build_profile(state, source_id: int, retry_of: int | None = None) -> int:
    return _create(state, source_id, "build_profile", None,
                   lambda jid, runner, llm: run_build_profile(jid, source_id, runner, llm), retry_of=retry_of)


def start_discover(state, country: str, cfg: DiscoveryConfig, auto_add: bool = False,
                   retry_of: int | None = None) -> int:
    return _create(state, None, f"discover:{country}", cfg.queries,
                   work_factory=lambda jid, runner, llm: run_discover(jid, runner, llm, country, cfg, auto_add),
                   params={"country": country, "cfg": asdict(cfg), "auto_add": auto_add}, retry_of=retry_of)


def start_translate(state, limit: int | None = None, item_ids: list[int] | None = None,
                    retry_of: int | None = None) -> int:
    return _create(state, None, "translate", len(item_ids) if item_ids else limit, dedupe=not item_ids,
                   work_factory=lambda jid, runner, llm: run_translate(jid, runner, llm, limit, item_ids),
                   params={"limit": limit, "item_ids": item_ids}, retry_of=retry_of)


def start_enrich(state, limit: int | None = None, item_ids: list[int] | None = None,
                 retry_of: int | None = None) -> int:
    return _create(state, None, "enrich", len(item_ids) if item_ids else limit, dedupe=not item_ids,
                   work_factory=lambda jid, runner, llm: run_enrich(jid, runner, llm, limit, item_ids),
                   params={"limit": limit, "item_ids": item_ids}, retry_of=retry_of)


def start_import_k1(state, limit: int | None = None, retry_of: int | None = None) -> int:
    return _create(state, None, "import_k1", limit, dedupe=True,
                   work_factory=lambda jid, runner, llm: run_import_k1(jid, runner, llm, limit),
                   params={"limit": limit}, retry_of=retry_of)


def start_export(state, translate_first: int = 0, retry_of: int | None = None) -> int:
    return _create(state, None, "export", None,
                   work_factory=lambda jid, runner, llm: run_export(jid, runner, llm, translate_first),
                   params={"translate_first": translate_first}, retry_of=retry_of)


def start_ollama_pull(state, model: str, retry_of: int | None = None) -> int:
    return _create(state, None, f"ollama_pull:{model}", None,
                   work_factory=lambda jid, runner, llm: run_ollama_pull(jid, runner, llm, model),
                   params={"model": model}, retry_of=retry_of, bypass_queue=True)  # nie czeka za skanami


def start_fb_post(state, kind: str = "auto", plant_id: str | None = None, hint: str | None = None,
                  count: int = 1, improve: int | None = None, mode: str = "nowa",
                  idea_id: int | None = None, retry_of: int | None = None) -> int:
    """Nowy post na fanpage (Lesny Dziadyga). Poza kolejka - krotkie zadanie, nie czeka za skanami."""
    return _create(state, None, "fb_post", count,
                   work_factory=lambda jid, runner, llm: run_fb_post(jid, runner, llm, kind, plant_id, hint, count, improve,
                                                                         mode, idea_id),
                   dedupe=False, params={"kind": kind, "plant_id": plant_id, "hint": hint, "count": count,
                                         "improve": improve, "mode": mode, "idea_id": idea_id}, retry_of=retry_of, bypass_queue=True)


def start_fb_review(state, post_ids: list[int] | None = None, retry_of: int | None = None) -> int:
    """Ocena postow jak specjalista (metryki + recenzja eksperta). None = wszystkie bez aktualnej oceny."""
    return _create(state, None, "fb_review", len(post_ids) if post_ids else None,
                   work_factory=lambda jid, runner, llm: run_fb_review(jid, runner, llm, post_ids),
                   dedupe=not post_ids, params={"post_ids": post_ids}, retry_of=retry_of, bypass_queue=True)


def start_fb_ideas(state, retry_of: int | None = None) -> int:
    """Zbieranie potencjalnych ciekawostek (wiedza o roslinach + artykuly z zagranicznych stron) i ich ocena."""
    return _create(state, None, "fb_ideas", None, work_factory=lambda jid, runner, llm: run_fb_ideas(jid, runner, llm),
                   params={}, retry_of=retry_of, bypass_queue=True)


def start_agent_audit(state, trigger: str = "reczny", retry_of: int | None = None) -> int:
    """Audytor wiedzy: raport oceny bazy wiedzy Kwiatownika + zalecenia (/agents). Krotkie - poza kolejka."""
    return _create(state, None, "agent_audit", None,
                   work_factory=lambda jid, runner, llm: run_agent_audit(jid, runner, llm, trigger),
                   params={"trigger": trigger}, retry_of=retry_of, bypass_queue=True)


def start_agent_dispatch(state, trigger: str = "reczny", max_tasks: int | None = None,
                         retry_of: int | None = None) -> int:
    """Zleceniodawca: zalecenia ostatniego audytu -> zadania Siedziby (kolejka)."""
    return _create(state, None, "agent_dispatch", None,
                   work_factory=lambda jid, runner, llm: run_agent_dispatch(jid, runner, state, trigger, max_tasks),
                   params={"trigger": trigger, "max_tasks": max_tasks}, retry_of=retry_of, bypass_queue=True)


def start_agent_deploy(state, trigger: str = "reczny", force: bool = False, dry_run: bool = False,
                       retry_of: int | None = None) -> int:
    """Wdrozeniowiec: commit + push nowych danych Kwiatownika2 (Render buduje strone sam)."""
    return _create(state, None, "agent_deploy", None,
                   work_factory=lambda jid, runner, llm: run_agent_deploy(jid, runner, llm, trigger, force, dry_run),
                   params={"trigger": trigger, "force": force, "dry_run": dry_run}, retry_of=retry_of,
                   bypass_queue=True)


def scan_all(state, only_auto: bool = False) -> list[int]:
    with Session(engine) as s:
        stmt = select(Source.id).where(Source.active == True)  # noqa: E712
        if only_auto:
            stmt = stmt.where(Source.auto_scan == True)  # noqa: E712
        ids = list(s.exec(stmt).all())
    return [start_scan(state, sid) for sid in ids]


def retry_job(state, job_id: int) -> int | None:
    """Uruchamia zadanie ponownie z tymi samymi parametrami. Skan kontynuuje od zapisanej kolejki."""
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if not job:
            return None
        if job.status in ACTIVE:
            return job.id
        kind, sid = job.kind, job.source_id
        p = json.loads(job.params_json) if job.params_json else {}
        has_frontier = bool(job.state_json and json.loads(job.state_json).get("frontier"))
    if kind == "scan" and sid:
        new = start_scan(state, sid, p.get("max_pages"), resume_from=job_id if has_frontier else None,
                         retry_of=job_id)
    elif kind == "plants_sync":
        new = start_plants_sync(state, retry_of=job_id)
    elif kind == "plant_research":
        new = start_plant_research(state, p.get("plant_ids"), p.get("limit") or 10, slots=p.get("slots"),
                                   retry_of=job_id)
    elif kind == "plant_apply":
        new = start_plant_apply(state, p.get("plant_ids"), retry_of=job_id)
    elif kind == "plant_photos":
        new = start_plant_photos(state, p.get("plant_ids"), retry_of=job_id)
    elif kind == "plant_organize":
        new = start_plant_organize(state, p.get("plant_ids"), retry_of=job_id)
    elif kind == "plant_merge":
        new = start_plant_merge(state, p.get("plant_ids"), retry_of=job_id)
    elif kind == "harvest" and sid:
        new = start_harvest(state, sid, p.get("max_pages"), resume_from=job_id if has_frontier else None,
                            retry_of=job_id)
    elif kind == "fetch_list" and sid and p.get("urls"):
        new = start_fetch_list(state, sid, p["urls"], retry_of=job_id)
    elif kind == "scan_pages" and sid and p.get("urls"):
        new = start_scan_pages(state, sid, p["urls"], p.get("depth", 1), retry_of=job_id)
    elif kind == "build_profile" and sid:
        new = start_build_profile(state, sid, retry_of=job_id)
    elif kind.startswith("discover:"):
        country = p.get("country") or kind.split(":", 1)[1]
        cfg = DiscoveryConfig(**p["cfg"]) if p.get("cfg") else DiscoveryConfig()
        new = start_discover(state, country, cfg, p.get("auto_add", False), retry_of=job_id)
    elif kind == "translate":
        new = start_translate(state, p.get("limit"), p.get("item_ids"), retry_of=job_id)
    elif kind == "import_k1":
        new = start_import_k1(state, p.get("limit"), retry_of=job_id)
    elif kind == "enrich":
        new = start_enrich(state, p.get("limit"), p.get("item_ids"), retry_of=job_id)
    elif kind.startswith("ollama_pull:"):
        new = start_ollama_pull(state, p.get("model") or kind.split(":", 1)[1], retry_of=job_id)
    elif kind == "fb_post":
        new = start_fb_post(state, p.get("kind") or "auto", p.get("plant_id"), p.get("hint"),
                            p.get("count") or 1, improve=p.get("improve"),
                            mode=p.get("mode") or "nowa", idea_id=p.get("idea_id"), retry_of=job_id)
    elif kind == "fb_ideas":
        new = start_fb_ideas(state, retry_of=job_id)
    elif kind == "fb_review":
        new = start_fb_review(state, p.get("post_ids"), retry_of=job_id)
    elif kind == "agent_audit":
        new = start_agent_audit(state, p.get("trigger") or "reczny", retry_of=job_id)
    elif kind == "agent_dispatch":
        new = start_agent_dispatch(state, p.get("trigger") or "reczny", p.get("max_tasks"), retry_of=job_id)
    elif kind == "agent_deploy":
        new = start_agent_deploy(state, p.get("trigger") or "reczny", bool(p.get("force")), bool(p.get("dry_run")),
                                 retry_of=job_id)
    elif kind == "export":
        new = start_export(state, p.get("translate_first", 0), retry_of=job_id)
    else:
        return None
    job_log(job_id, f"Ponowiono jako zadanie #{new}")
    return new
