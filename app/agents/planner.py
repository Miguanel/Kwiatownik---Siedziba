"""Planista ciaglosci: pilnuje, zeby Siedziba zawsze nad czyms pracowala.

Co AGENTS_CHECK_MINUTES (petla w tle) sprawdza kolejke. Uklada ranking propozycji kolejnego zadania:
  1. nowy audyt, gdy ostatni jest starszy niz AGENTS_AUDIT_HOURS (albo go nie ma),
  2. zalecenia audytu czekajace na zlecenie (wiedza trudno dostepna i opowiesci - cel Kwiatownika),
  3. domykanie potoku z listy "Zadania do zrobienia" (zapis i scalenie wiedzy, eksport, tlumaczenie, skany...),
  4. awaryjnie: zbieranie wiedzy o roslinach z najwiekszymi lukami (zeby kolejka nigdy nie stala).
W trybie automatycznym, gdy nie dziala zadne zadanie produkcyjne, zleca JEDNO - najlepsze z rankingu.
Po 3 nieudanych zleceniach z rzedu robi przerwe (AGENTS_FAIL_PAUSE_MIN). Decyzje sa w historii (/agents).
Zadania wymagajace decyzji czlowieka (zatwierdzanie przepisow, dodawanie/wylaczanie zrodel) nie sa zlecane.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import timedelta

from sqlmodel import Session, col, select

from app.agents import store, watchdog
from app.agents.tasks import (KINDS, NOT_PRODUCTION, PLANT_KINDS, describe, needs_llm, order, params, production_jobs,
                              recently_ordered, sync_tasks)
from app.config import settings
from app.db import engine
from app.models import AgentRun, AgentTask, Job, JobStatus


NO_EFFECT_HOURS = 6          # zadanie z listy, ktore niczego nie zmienilo - nie zlecaj tego samego przez tyle godzin


@dataclass
class Proposal:
    key: str                 # np. "task:12", "grupa:plants_merge", "audyt"
    kind: str                # rodzaj zadania (app/agents/tasks.KINDS)
    title: str
    why: str
    score: float             # wyzej = lepiej
    source: str              # audyt | potok | ciaglosc
    params: dict = field(default_factory=dict)
    task_id: int | None = None


# grupa z "Zadania do zrobienia" -> (rodzaj, ile pozycji w jednym zadaniu, ocena)
GROUPS: dict[str, tuple[str, int, float]] = {
    "plants_apply": ("zapis", 20, 80), "plants_organize": ("porzadkuj", 10, 78), "plants_merge": ("scal", 6, 76),
    "export": ("eksport", 0, 74), "plants_new": ("nowa_wiedza", 4, 70), "enrich": ("opracowanie", 10, 62),
    "translate": ("tlumaczenie", 0, 60), "retry": ("ponow", 1, 59), "harvest": ("pobierz_strone", 1, 58),
    "build_profile": ("scraper", 1, 56), "rebuild_profile": ("scraper", 1, 55), "scan_new": ("skan", 1, 52),
    "plants_photos": ("zdjecia", 10, 50), "scan_stale": ("skan", 1, 40), "discover": ("nowe_zrodla", 1, 35),
    "translate_errors": ("tlumaczenie", 10, 30),
}


def _audit_running(s: Session) -> bool:
    return bool(s.exec(select(Job).where(Job.kind == "agent_audit",
                                         col(Job.status).in_([JobStatus.queued, JobStatus.running]))).first())


def _failed_keys(s: Session, hours: int = 2) -> set[str]:
    """Propozycje, ktore planista zlecil niedawno i ktore sie nie powiodly - nie proponuj od razu ponownie."""
    since = store.now() - timedelta(hours=hours)
    rows = s.exec(select(AgentTask).where(AgentTask.agent == "planista", col(AgentTask.status).in_(["failed", "skipped"]),
                                          col(AgentTask.created_at) >= since)).all()
    return {params(t).get("klucz") for t in rows} - {None}


def _no_effect(s: Session, hours: int) -> dict[str, str]:
    """{klucz: stan listy} zadan planisty z listy "Zadania do zrobienia", ktore niedawno sie wykonaly. Gdy lista
    wyglada tak samo jak przed zleceniem (np. eksport, ktory nic nie zmienia), nie zlecamy go w kolko."""
    since = store.now() - timedelta(hours=hours)
    out: dict[str, str] = {}
    rows = s.exec(select(AgentTask).where(AgentTask.agent == "planista", AgentTask.status == "done",
                                          col(AgentTask.ordered_at) >= since)
                  .order_by(col(AgentTask.ordered_at))).all()
    for t in rows:
        p = params(t)
        if p.get("klucz") and p.get("stan"):
            out[p["klucz"]] = p["stan"]
    return out


def candidates(state, limit_items: int = 30, load: dict | None = None) -> list[Proposal]:
    from app.worker.jobs import retry_depth
    from app.worker.suggestions import build_groups

    has_llm = getattr(state, "llm", None) is not None
    out: list[Proposal] = []
    with Session(engine) as s:
        audit = store.last_run("audytor")
        failed = _failed_keys(s)
        done_same = _no_effect(s, NO_EFFECT_HOURS)
        if not _audit_running(s):
            age = (store.now() - store.utc(audit.created_at)) if audit else None
            if audit is None or age > timedelta(hours=settings.agents_audit_hours):
                out.append(Proposal("audyt", "audyt", "Nowy audyt bazy wiedzy Kwiatownika",
                                    "Jeszcze nie bylo audytu - bez niego nie wiadomo, czego brakuje." if audit is None else
                                    f"Ostatni audyt #{audit.id} ma {age.days * 24 + age.seconds // 3600} h - "
                                    "trzeba sprawdzic efekty zlecen i ustalic nowe priorytety.",
                                    95 if audit is None else 72, "ciaglosc"))
        # zalecenia audytu
        tasks = s.exec(select(AgentTask).where(AgentTask.agent == "audytor", AgentTask.status == "proposed")
                       .order_by(AgentTask.priority, col(AgentTask.id)).limit(8)).all()
        for t in tasks:
            if needs_llm(t.kind) and not has_llm:
                continue
            p = params(t)
            if t.kind in PLANT_KINDS and p.get("rosliny"):
                recent = recently_ordered(s, t.kind, settings.agents_reorder_days)
                if all(x in recent for x in p["rosliny"]):
                    continue
            # zalecenia audytu (cel: wiedza trudno dostepna, opowiesci) - za zapisem/porzadkowaniem/scaleniem juz
            # zebranej wiedzy (tanie, a od razu widoczne w Kwiatowniku), przed reszta potoku
            out.append(Proposal(f"task:{t.id}", t.kind, t.title, t.reason or "", 79 - 4 * t.priority, "audyt", p, t.id))
    # domykanie potoku (lista "Zadania do zrobienia")
    for g in build_groups(limit=limit_items):
        if g.key not in GROUPS or not g.items or f"grupa:{g.key}" in failed:
            continue
        kind, n, score = GROUPS[g.key]
        if needs_llm(kind) and not has_llm:
            continue
        free = [i for i in g.items if not i.status]
        if not free:
            continue
        if kind == "ponow":
            with Session(engine) as s:
                jobs = {j.id: j for j in s.exec(select(Job).where(col(Job.id).in_([int(i.id) for i in free]))).all()}
            # tylko nieudane - zadan zatrzymanych recznie (albo przez straznika jako zawieszone) nie wznawiamy sami
            free = [i for i in free if int(i.id) in jobs and jobs[int(i.id)].status == JobStatus.failed
                    and not jobs[int(i.id)].kind.startswith(NOT_PRODUCTION)
                    and retry_depth(int(i.id)) < settings.max_auto_retries]
            if not free:
                continue
        pick = free[:n] if n else free[:1]
        if kind in PLANT_KINDS:
            p = {"rosliny": [i.id for i in pick], "nazwy": [str(i.label or i.id) for i in pick]}
        elif kind == "nowe_zrodla":
            p = {"kraj": pick[0].id, "kraj_nazwa": str(pick[0].label or pick[0].id)}
        elif kind == "eksport":
            p = {"etykieta": pick[0].label}
        elif kind == "tlumaczenie" and g.key == "translate":
            p = {"limit": settings.translate_batch, "etykieta": f"{g.total} przepisow czeka"}
        else:
            p = {"ids": [i.id for i in pick], "etykieta": ", ".join(str(i.label or i.id) for i in pick[:3])
                 + (f" (+{len(pick) - 3})" if len(pick) > 3 else "")}
        p["klucz"] = f"grupa:{g.key}"
        p["stan"] = f"{g.total}|" + "|".join(str(i.label or i.id) for i in pick)     # jak wygladala lista przed zleceniem
        if done_same.get(p["klucz"]) == p["stan"]:
            continue                                   # to samo zlecenie juz bylo i niczego nie zmienilo
        out.append(Proposal(f"grupa:{g.key}", kind, describe(kind, p), f"{g.title}: {g.description}", score, "potok", p))
    if has_llm:
        p = {"limit": 3, "klucz": "ciaglosc:luki"}
        out.append(Proposal("ciaglosc:luki", "luki", "Zbieranie wiedzy o 3 roslinach z najwiekszymi lukami",
                            "Nic innego nie czeka - Siedziba szuka brakujacych informacji (takze po chinsku i japonsku), "
                            "zeby kolejka nie stala.", 10, "ciaglosc", p))
    load = load if load is not None else watchdog.llm_load(state)
    if load.get("overloaded"):                         # modele sie dlawia -> najpierw zadania bez LLM
        for p in out:
            job = KINDS[p.kind].job if p.kind in KINDS else ""
            if watchdog.is_llm_job(job):
                p.score -= 45
                p.why = f"{p.why} [odlozone: {load['powod']}]"
    seen, uniq = set(), []
    for p in sorted(out, key=lambda x: -x.score):
        if p.key not in seen:
            seen.add(p.key)
            uniq.append(p)
    return uniq


def paused_until(s: Session):
    """Przerwa po 3 nieudanych zleceniach planisty z rzedu (np. limity modeli) - zwraca czas konca albo None."""
    rows = s.exec(select(AgentTask).where(AgentTask.agent == "planista", col(AgentTask.ordered_at).is_not(None))
                  .order_by(col(AgentTask.ordered_at).desc()).limit(3)).all()
    if len(rows) < 3 or any(t.status != "failed" for t in rows):
        return None
    last = max(store.utc(t.finished_at or t.ordered_at) for t in rows)
    until = last + timedelta(minutes=settings.agents_fail_pause_min)
    return until if until > store.now() else None


def _execute(state, prop: Proposal, run_id: int) -> tuple[bool, str, int | None]:
    if prop.task_id is None:
        with Session(engine) as s:
            t = AgentTask(run_id=run_id, agent="planista", kind=prop.kind, title=prop.title, reason=prop.why[:1000],
                          priority=1, params_json=json.dumps({**prop.params, "klucz": prop.key}, ensure_ascii=False))
            s.add(t)
            s.commit()
            prop.task_id = t.id
    ok, msg = order(state, prop.task_id, run_id)
    return ok, msg, prop.task_id


def _signature(decision: str, props: list[Proposal], ordered, events=()) -> str:
    raw = json.dumps([decision.split(":")[0], [p.key for p in props[:3]], ordered,
                      [(e["job"], e["typ"], e["akcja"]) for e in events]], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _summary(report: dict) -> str:
    ev = report.get("straznik") or []
    if not ev:
        return report["decyzja"]
    stopped = [e for e in ev if e["typ"] == "zawieszone"]
    dead = [e for e in ev if e["typ"] == "martwe"]
    bits = []
    if stopped:
        bits.append("zawieszone: " + ", ".join(f"#{e['job']} ({e['akcja']})" for e in stopped))
    if dead:
        bits.append("martwe: " + ", ".join(f"#{e['job']}" for e in dead))
    return "; ".join(bits) + " | " + report["decyzja"]


def run_planner(state, trigger: str = "auto", execute: bool | None = None, force: bool = False,
                pick: str | None = None) -> int:
    """execute=None -> wg przelacznika "auto" z panelu. force=True: zlec mimo dzialajacych zadan (przycisk w panelu).
    pick: klucz propozycji do zlecenia (zamiast najlepszej). Zwraca id przebiegu (nowego albo powtorzonego)."""
    events = watchdog.check(state, act=True)        # zawieszone / martwe zadania - zanim ocenimy, czy kolejka stoi
    sync_tasks()
    auto = store.runtime().get("auto", True)
    execute = auto if execute is None else execute
    load = watchdog.llm_load(state)
    props = candidates(state, load=load)
    cap = store.job_limit(state)
    with Session(engine) as s:
        busy = production_jobs(s)
        pause = paused_until(s)
        active_keys = {params(t).get("klucz") for t in s.exec(select(AgentTask).where(
            col(AgentTask.status).in_(["ordered", "running"]))).all()} - {None}
    if busy:      # przy limicie > 1 planista doklada zadania - ale nie to samo, co juz dziala
        props = [p for p in props if (p.params or {}).get("klucz", p.key) not in active_keys
                 and p.key not in active_keys]
        if load.get("llm_jobs", 0) >= settings.agents_max_llm_jobs or load.get("cloud_down"):
            props = [p for p in props if not needs_llm(p.kind)]   # modele zajete - dokladamy tylko zadania bez LLM
    quiet = watchdog.job_silence(busy)
    state_info = {"auto": auto, "pracuje": len(busy), "max_naraz": cap,
                  "zadania": [f"#{j.id} {j.kind} ({j.status.value}"
                              + (f", cisza {quiet[j.id]} min" if quiet.get(j.id) else "") + ")" for j in busy[:8]]}
    ordered = None
    log_lines = []
    if cap == 0 and not force:
        decision = "wstrzymane: zadania w tle wstrzymane (limit 0 w panelu agentow) - nic nie zlecam"
    elif len(busy) >= cap and not force:
        decision = f"pracuje: Siedziba pracuje ({len(busy)}/{cap} zadan naraz) - nic nie zlecam"
    elif not execute:
        decision = "propozycje: tryb automatyczny wylaczony - tylko propozycje" if trigger == "auto" else \
            "propozycje: sprawdzenie bez zlecania"
    elif pause and not force:
        decision = f"przerwa: 3 ostatnie zlecenia nieudane - przerwa do {pause.astimezone(store._TZ):%H:%M}"
    elif not props:
        decision = "brak: nie ma czego zlecic (brak LLM i pustych zadan)"
    else:
        chosen = [p for p in props if p.key == pick] if pick else props
        decision = "brak: zadna propozycja nie dala sie zlecic"
        run_id = store.start_run("planista", trigger)
        for prop in chosen[:3]:
            ok, msg, tid = _execute(state, prop, run_id)
            log_lines.append(f"{prop.title}: {msg}")
            if ok:
                with Session(engine) as s:
                    jid = s.get(AgentTask, tid).job_id
                ordered = {"task": tid, "job": jid, "tytul": prop.title, "typ": prop.kind, "klucz": prop.key}
                decision = (f"zlecono: kolejka stala - zlecono '{prop.title}' (zadanie #{jid})" if not busy else
                            f"zlecono: zlecono recznie '{prop.title}' (zadanie #{jid})" if force else
                            f"zlecono: wolne miejsce ({len(busy)}/{cap}) - zlecono '{prop.title}' (zadanie #{jid})")
                break
        report = {"stan": state_info, "propozycje": [asdict(p) for p in props[:12]], "decyzja": decision.split(": ", 1)[1],
                  "zlecono": ordered, "proby": log_lines, "straznik": events, "llm": load,
                  "sygnatura": _signature(decision, props, ordered, events)}
        for line in log_lines + ([watchdog.summary(events)] if events else []):
            store.run_log(run_id, line)
        store.finish_run(run_id, "done", report, summary=_summary(report))
        return run_id

    report = {"stan": state_info, "propozycje": [asdict(p) for p in props[:12]], "decyzja": decision.split(": ", 1)[1],
              "zlecono": None, "proby": [], "straznik": events, "llm": load,
              "sygnatura": _signature(decision, props, None, events)}
    with Session(engine) as s:                     # to samo co ostatnio -> tylko licznik powtorzen (bez zasmiecania)
        last = s.exec(select(AgentRun).where(AgentRun.agent == "planista").order_by(col(AgentRun.id).desc())).first()
        if last and store.report(last).get("sygnatura") == report["sygnatura"] and trigger == "auto" \
                and last.trigger == "auto":
            last.repeats = (last.repeats or 0) + 1
            last.updated_at = store.now()
            last.report_json = json.dumps(report, ensure_ascii=False, default=str)
            s.add(last)
            s.commit()
            return last.id
    run_id = store.start_run("planista", trigger)
    if events:
        store.run_log(run_id, watchdog.summary(events))
    store.finish_run(run_id, "done", report, summary=_summary(report))
    return run_id
