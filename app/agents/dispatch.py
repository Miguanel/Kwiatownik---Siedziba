"""Zleceniodawca: zamienia zalecenia ostatniego audytu na zadania Siedziby (kolejka).

Zasady: od najpilniejszych (priorytet), najwyzej AGENTS_DISPATCH_MAX zlecen w jednym przebiegu, nie zleca, gdy
kolejka jest pelna (2 x MAX_CONCURRENT_JOBS czekajacych/dzialajacych), pomija rosliny/kraje, dla ktorych to samo
zlecono w ostatnich AGENTS_REORDER_DAYS dniach, i zadania wymagajace LLM, gdy modeli brak. Kazda decyzja
(zlecone / pominiete + powod) trafia do raportu przebiegu.
"""
import json

from sqlmodel import Session, col, select

from app.agents import store, watchdog
from app.agents.tasks import (KINDS, PLANT_KINDS, describe, needs_llm, order, params, production_jobs,
                              recently_ordered, sync_tasks)
from app.config import settings
from app.db import engine
from app.models import AgentTask


def _skip(task_id: int, note: str) -> None:
    with Session(engine) as s:
        t = s.get(AgentTask, task_id)
        t.status, t.note = "skipped", note
        s.add(t)
        s.commit()


def _narrow(task_id: int, p: dict, keep: list[str]) -> dict:
    """Zostawia w zadaniu tylko rosliny, ktorych nie zlecano niedawno."""
    names = dict(zip(p.get("rosliny") or [], p.get("nazwy") or []))
    p = {**p, "rosliny": keep, "nazwy": [names.get(i, i) for i in keep]}
    with Session(engine) as s:
        t = s.get(AgentTask, task_id)
        t.params_json = json.dumps(p, ensure_ascii=False)
        t.title = describe(t.kind, p)
        s.add(t)
        s.commit()
    return p


async def run_dispatch(state, trigger: str = "reczny", job_id: int | None = None, logf=None,
                       max_tasks: int | None = None) -> int:
    run_id = store.start_run("zleceniodawca", trigger, job_id)

    def say(msg: str) -> None:
        store.run_log(run_id, msg)
        if logf:
            logf(msg)

    try:
        sync_tasks()
        audit = store.last_run("audytor")
        limit = max_tasks or settings.agents_dispatch_max
        with Session(engine) as s:
            busy = production_jobs(s)
            tasks = s.exec(select(AgentTask).where(AgentTask.agent == "audytor", AgentTask.status == "proposed")
                           .order_by(AgentTask.priority, col(AgentTask.id))).all()
        report = {"audyt": audit.id if audit else None, "limit": limit, "kolejka": len(busy),
                  "zalecen": len(tasks), "zlecone": [], "pominiete": [], "czekaja": []}
        say(f"Zalecenia czekajace na zlecenie: {len(tasks)} (audyt #{audit.id if audit else '-'}); "
            f"w kolejce Siedziby: {len(busy)}; limit zlecen: {limit}")
        cap = store.job_limit(state)
        full = cap == 0 or len(busy) >= cap * 2
        if full:
            say("Zadania w tle wstrzymane (limit 0) - nic nie zlecam" if cap == 0 else
                "Kolejka pelna - nic nie zlecam (zalecenia czekaja na nastepny przebieg)")
        load = watchdog.llm_load(state)
        llm_running = load["llm_jobs"]
        report["llm"] = load
        if llm_running >= settings.agents_max_llm_jobs or load["cloud_down"]:
            say("Modele LLM zajete (" + (load["powod"] or f"{llm_running} zadan z LLM") + ") - zadania z LLM poczekaja")
        for t in tasks:
            entry = {"task": t.id, "typ": t.kind, "tytul": t.title, "priorytet": t.priority}
            if full or len(report["zlecone"]) >= limit:
                report["czekaja"].append({**entry, "powod": "kolejka pelna" if full else "limit zlecen w przebiegu"})
                continue
            p = params(t)
            if needs_llm(t.kind) and getattr(state, "llm", None) is None:
                report["czekaja"].append({**entry, "powod": "brak modeli LLM"})
                continue
            uses_llm = watchdog.is_llm_job(KINDS[t.kind].job) if t.kind in KINDS else False
            if uses_llm and (llm_running >= settings.agents_max_llm_jobs or load["cloud_down"]):
                report["czekaja"].append({**entry, "powod": "chmura LLM odpowiada bledami" if load["cloud_down"] else
                                          f"limit zadan z LLM naraz ({settings.agents_max_llm_jobs})"})
                continue
            with Session(engine) as s:
                recent = recently_ordered(s, t.kind, settings.agents_reorder_days)
            if t.kind in PLANT_KINDS and p.get("rosliny"):
                keep = [x for x in p["rosliny"] if x not in recent]
                if not keep:
                    why = f"wszystkie rosliny mialy to zadanie w ostatnich {settings.agents_reorder_days} dniach"
                    _skip(t.id, why)
                    report["pominiete"].append({**entry, "powod": why})
                    say(f"Pomijam: {t.title} - {why}")
                    continue
                if keep != p["rosliny"]:
                    p = _narrow(t.id, p, keep)
                    entry["tytul"] = describe(t.kind, p)
            if p.get("kraj") and p["kraj"] in recent:
                why = f"kraj przeszukiwany w ostatnich {settings.agents_reorder_days} dniach"
                _skip(t.id, why)
                report["pominiete"].append({**entry, "powod": why})
                continue
            ok, msg = order(state, t.id, run_id)
            if ok:
                with Session(engine) as s:
                    jid = s.get(AgentTask, t.id).job_id
                report["zlecone"].append({**entry, "job": jid})
                llm_running += uses_llm
                say(f"Zlecono [{KINDS[t.kind].label}] {entry['tytul']} -> {msg}")
            else:
                report["pominiete"].append({**entry, "powod": msg})
                say(f"Nie zlecono: {entry['tytul']} - {msg}")
        summary = (f"Zlecono {len(report['zlecone'])} z {len(tasks)} zalecen"
                   + (f" audytu #{audit.id}" if audit else "")
                   + (f"; pominieto {len(report['pominiete'])}" if report["pominiete"] else "")
                   + (f"; czeka {len(report['czekaja'])}" if report["czekaja"] else ""))
        store.finish_run(run_id, "done", report, summary=summary, parent_id=audit.id if audit else None)
        say(summary)
    except Exception as exc:
        say(f"BLAD: {type(exc).__name__}: {exc}")
        store.finish_run(run_id, "failed", summary=f"Zlecanie nieudane: {str(exc)[:200]}")
        raise
    return run_id
