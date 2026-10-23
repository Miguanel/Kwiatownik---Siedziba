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
                                  run_plant_place, run_plant_research, run_plants_sync)
from app.worker.marketing import run_fb_ideas, run_fb_post, run_fb_review
from app.worker.nn import run_nn_init, run_nn_train
from app.worker.leads import run_leads_extract, run_leads_search
from app.worker.magic import run_plant_magic
from app.worker.paleta import run_paleta_export
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


def start_plant_place(state, plant_ids: list[str] | None = None, force: bool = False,
                      retry_of: int | None = None) -> int:
    """Rozmieszczenie wiedzy z sieci w rozdzialach i podrozdzialach strony (Uklad strony)."""
    return _create(state, None, "plant_place", len(plant_ids) if plant_ids else None,
                   lambda jid, runner, llm: run_plant_place(jid, runner, llm, plant_ids, force),
                   dedupe=not plant_ids, params={"plant_ids": plant_ids, "force": force}, retry_of=retry_of)


def start_plant_magic(state, plant_ids: list[str] | None = None, limit: int | None = None, sweeps: int | None = None,
                      retry_of: int | None = None) -> int:
    """Magia roslin: czary, zabobony, obrzedy, rosliny czarownic - tezaurus, tropy, archiwa, przeglad od praktyki."""
    return _create(state, None, "plant_magic", len(plant_ids) if plant_ids else (limit or settings.magic_plants),
                   lambda jid, runner, llm: run_plant_magic(jid, runner, llm, plant_ids, limit, sweeps),
                   dedupe=not plant_ids, params={"plant_ids": plant_ids, "limit": limit, "sweeps": sweeps},
                   retry_of=retry_of)


def start_paleta_export(state, retry_of: int | None = None) -> int:
    """Paleta Barw w Kwiatowniku2: zweryfikowana wiedza o farbowaniu wlosow -> static/json/paleta_siedziba.json."""
    return _create(state, None, "paleta_export", None, lambda jid, runner, llm: run_paleta_export(jid, runner, llm),
                   params={}, retry_of=retry_of, bypass_queue=True)


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
                    retry_of: int | None = None, force: bool = False) -> int:
    """force=True: przetlumacz wybrane przepisy nawet, jesli byly juz tlumaczone (reczne "ponownie")."""
    return _create(state, None, "translate", len(item_ids) if item_ids else limit, dedupe=not item_ids,
                   work_factory=lambda jid, runner, llm: run_translate(jid, runner, llm, limit, item_ids, force),
                   params={"limit": limit, "item_ids": item_ids, "force": force}, retry_of=retry_of)


def start_enrich(state, limit: int | None = None, item_ids: list[int] | None = None,
                 retry_of: int | None = None, force: bool = False) -> int:
    return _create(state, None, "enrich", len(item_ids) if item_ids else limit, dedupe=not item_ids,
                   work_factory=lambda jid, runner, llm: run_enrich(jid, runner, llm, limit, item_ids, force),
                   params={"limit": limit, "item_ids": item_ids, "force": force}, retry_of=retry_of)


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


def start_nn_train(state, tasks: list[str] | None = None, encoder: str | None = None,
                   retry_of: int | None = None) -> int:
    """Trening sieci neuronowych (walidacja krzyzowa + mistrz/pretendent). Bez LLM - liczy procesor."""
    from app.nn.tasks import TASKS
    tasks = [t for t in (tasks or list(TASKS)) if t in TASKS]
    return _create(state, None, "nn_train", len(tasks),
                   work_factory=lambda jid, runner, llm: run_nn_train(jid, runner, llm, tasks, encoder),
                   params={"tasks": tasks, "encoder": encoder}, retry_of=retry_of)


def start_nn_init(state, retry_of: int | None = None) -> int:
    """Pobranie enkodera e5 (ONNX) do DATA_DIR/nn/encoder - poza kolejka, jak pobieranie modeli Ollamy."""
    return _create(state, None, "nn_init", None, work_factory=lambda jid, runner, llm: run_nn_init(jid, runner, llm),
                   params={}, retry_of=retry_of, bypass_queue=True)


def start_leads_extract(state, limit: int = 20, retry_of: int | None = None) -> int:
    """Wzmianki o przepisach w ciekawostkach (LLM) -> lista "Przepisy do odnalezienia"."""
    return _create(state, None, "leads_extract", limit,
                   work_factory=lambda jid, runner, llm: run_leads_extract(jid, runner, llm, limit),
                   params={"limit": limit}, retry_of=retry_of)


def start_leads_search(state, lead_ids: list[int] | None = None, limit: int = 5, retry_of: int | None = None) -> int:
    """Szukanie pelnych przepisow (skladniki + kroki) dla wzmianek w innych miejscach internetu."""
    return _create(state, None, "leads_search", len(lead_ids) if lead_ids else limit,
                   work_factory=lambda jid, runner, llm: run_leads_search(jid, runner, llm, lead_ids, limit),
                   dedupe=not lead_ids, params={"lead_ids": lead_ids, "limit": limit}, retry_of=retry_of)


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
        st = json.loads(job.state_json) if job.state_json else {}
        has_frontier = bool(st.get("frontier"))
    # punkty kontrolne (R4): rosliny skonczone w przerwanym zadaniu nie sa opracowywane drugi raz
    done = set(st.get("done") or [])
    if done and kind.startswith("plant_") and p.get("plant_ids"):
        left = [x for x in p["plant_ids"] if x not in done]
        if not left:
            job_log(job_id, f"Wszystkie rosliny tego zadania byly juz skonczone ({len(done)}) - nie ma czego ponawiac")
            return None
        job_log(job_id, f"Ponowienie od punktu kontrolnego: pomijam {len(p['plant_ids']) - len(left)} skonczonych roslin")
        p["plant_ids"] = left
    if done and kind in ("translate", "enrich") and p.get("item_ids"):
        left = [x for x in p["item_ids"] if x not in done]
        if not left:
            job_log(job_id, f"Wszystkie przepisy tego zadania byly juz zrobione ({len(done)}) - nie ma czego ponawiac")
            return None
        job_log(job_id, f"Ponowienie od punktu kontrolnego: pomijam {len(p['item_ids']) - len(left)} zrobionych przepisow")
        p["item_ids"] = left
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
    elif kind == "plant_place":
        new = start_plant_place(state, p.get("plant_ids"), bool(p.get("force")), retry_of=job_id)
    elif kind == "paleta_export":
        new = start_paleta_export(state, retry_of=job_id)
    elif kind == "plant_magic":
        new = start_plant_magic(state, p.get("plant_ids"), p.get("limit"), p.get("sweeps"), retry_of=job_id)
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
        new = start_translate(state, p.get("limit"), p.get("item_ids"), retry_of=job_id, force=bool(p.get("force")))
    elif kind == "import_k1":
        new = start_import_k1(state, p.get("limit"), retry_of=job_id)
    elif kind == "enrich":
        new = start_enrich(state, p.get("limit"), p.get("item_ids"), retry_of=job_id, force=bool(p.get("force")))
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
    elif kind == "leads_extract":
        new = start_leads_extract(state, p.get("limit") or 20, retry_of=job_id)
    elif kind == "leads_search":
        new = start_leads_search(state, p.get("lead_ids"), p.get("limit") or 5, retry_of=job_id)
    elif kind == "nn_train":
        new = start_nn_train(state, p.get("tasks"), p.get("encoder"), retry_of=job_id)
    elif kind == "nn_init":
        new = start_nn_init(state, retry_of=job_id)
    else:
        return None
    job_log(job_id, f"Ponowiono jako zadanie #{new}")
    return new
