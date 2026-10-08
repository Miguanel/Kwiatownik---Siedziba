"""Straznik kolejki (czesc planisty ciaglosci): zawieszone i "martwe" zadania, przeciazenie modeli LLM.

- Zawieszone: zadanie dziala, ale od AGENTS_STALL_MINUTES nie dalo znaku zycia (nowy krok, wpis w logu,
  zapytanie do modelu - app/worker/activity.beat). Planista je zatrzymuje (AGENTS_STALL_ACTION=stop) - lagodnie,
  a po 30 s sila - i zapisuje to w logu zadania i w historii agentow. Zatrzymane zadanie mozna ponowic
  (lista "Przerwane lub nieudane zadania"; skany kontynuuja od zapisanej kolejki).
- Martwe: w bazie "w kolejce"/"dziala", ale kolejka w pamieci go nie ma (np. blad poza zadaniem) - oznaczane
  jako nieudane, zeby nie blokowaly planisty ("Siedziba pracuje") w nieskonczonosc.
- Przeciazenie LLM: chmura (Gemini/Groq) odpowiada bledami albo lokalna Ollama ma kolejke - wtedy agenci
  odkladaja nowe zadania z LLM i wybieraja zadania bez modeli.
"""
import time
from datetime import datetime, timezone

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.models import Job, JobStatus
from app.worker import activity

_REPORTED: dict[int, float] = {}   # tryb "report": kiedy ostatnio zgloszono zadanie (zglaszamy raz na okres ciszy)
ZOMBIE_AFTER_S = 120          # tyle sekund po utworzeniu zadanie musi juz byc w kolejce w pamieci
# zadania, ktore pytaja modele LLM (ich liczbe naraz ogranicza AGENTS_MAX_LLM_JOBS)
LLM_JOBS = ("plant_research", "plant_apply", "plant_organize", "plant_merge", "translate", "enrich",
            "build_profile", "fb_ideas")


def _ts(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def silence_s(job: Job, now: float | None = None) -> float | None:
    """Ile sekund zadanie nie dalo znaku zycia (None = nie dziala)."""
    if job.status != JobStatus.running:
        return None
    now = now or time.time()
    last = activity.last_beat(job.id) or _ts(job.started_at) or _ts(job.created_at)
    return max(0.0, now - last) if last else None


def _note(job_id: int, message: str) -> None:
    """Wpis straznika w logu zadania - bez liczenia go jako znaku zycia zadania."""
    from app.worker.jobs import job_log
    prev = activity.last_beat(job_id)
    job_log(job_id, message)
    if prev is None:
        activity._BEAT.pop(job_id, None)
    else:
        activity._BEAT[job_id] = prev


def is_llm_job(kind: str) -> bool:
    return kind in LLM_JOBS


def check(state, act: bool = True, now: float | None = None) -> list[dict]:
    """Szuka zawieszonych i martwych zadan; act=True - zatrzymuje / oznacza je. Zwraca liste zdarzen."""
    from app.worker.jobs import set_status
    runner = getattr(state, "jobs", None)
    if runner is None or not hasattr(runner, "tasks"):
        return []
    now = now or time.time()
    limit = max(1, settings.agents_stall_minutes) * 60
    stop = settings.agents_stall_action.lower() != "report"
    out: list[dict] = []
    with Session(engine) as s:
        rows = s.exec(select(Job).where(col(Job.status).in_([JobStatus.queued, JobStatus.running]))).all()
    for job in rows:
        if job.kind.startswith("ollama_pull"):     # pobieranie modelu: dlugie i bez krokow - nie ruszamy
            continue
        age = now - (_ts(job.created_at) or now)
        if job.id not in runner.tasks:
            if age < ZOMBIE_AFTER_S:
                continue
            ev = {"job": job.id, "kind": job.kind, "typ": "martwe", "cisza_min": round(age / 60),
                  "opis": f"w bazie '{job.status.value}', ale nie ma go w kolejce", "akcja": "oznaczono jako nieudane"}
            if act:
                _note(job.id, "Planista ciaglosci: zadanie nie dziala w kolejce (martwe) - oznaczam jako nieudane; "
                                "mozna je ponowic")
                set_status(job.id, "failed")
            else:
                ev["akcja"] = "do oznaczenia"
            out.append(ev)
            continue
        if job.id not in runner.active:
            continue                               # czeka w kolejce na wolne miejsce - to nie zawieszenie
        quiet = silence_s(job, now)
        if quiet is None or quiet < limit:
            continue
        act_info = activity.get(job.id) or {}
        ev = {"job": job.id, "kind": job.kind, "typ": "zawieszone", "cisza_min": round(quiet / 60),
              "opis": f"brak znaku zycia od {round(quiet / 60)} min"
                      + (f" (ostatni krok: {act_info.get('step')} {act_info.get('detail') or ''})".rstrip()
                         if act_info else ""),
              "akcja": "zatrzymano" if stop else "tylko zgloszono"}
        if act:
            if job.id in runner.stop_flags:
                ev["akcja"] = "zatrzymywanie w toku"
            elif stop:
                _note(job.id, f"Planista ciaglosci: brak postepu od {round(quiet / 60)} min "
                                f"(limit {settings.agents_stall_minutes} min) - zatrzymuje zadanie")
                runner.request_stop(job.id, grace=30)
            elif now - _REPORTED.get(job.id, 0) >= limit:
                _note(job.id, f"Planista ciaglosci: brak postepu od {round(quiet / 60)} min - zadanie moze byc zawieszone")
                _REPORTED[job.id] = now
            else:
                continue                           # juz zgloszone w tym okresie ciszy
        out.append(ev)
    return out


def llm_load(state, window_s: int = 600) -> dict:
    """Stan modeli: ile zadan z LLM dziala, czy chmura odpowiada, czy lokalna Ollama ma kolejke."""
    with Session(engine) as s:
        running = s.exec(select(Job).where(col(Job.status).in_([JobStatus.queued, JobStatus.running]))).all()
    llm_jobs = [j for j in running if is_llm_job(j.kind)]
    router = getattr(state, "llm", None)
    recent = [r for r in list(getattr(router, "recent", []) or [])
              if isinstance(r, dict) and r.get("ts", 0) >= time.time() - window_s]
    cloud = [r for r in recent if not str(r.get("model", "")).startswith("ollama:")]
    cloud_ok = sum(1 for r in cloud if r.get("ok"))
    inflight = dict(getattr(router, "inflight", {}) or {})
    local_queue = sum(v for k, v in inflight.items() if str(k).startswith("ollama:"))
    cloud_down = len(cloud) >= 6 and cloud_ok / len(cloud) < 0.3
    overloaded = len(llm_jobs) >= settings.agents_max_llm_jobs or cloud_down or local_queue >= 2
    reasons = []
    if len(llm_jobs) >= settings.agents_max_llm_jobs:
        reasons.append(f"{len(llm_jobs)} zadan z LLM naraz (limit {settings.agents_max_llm_jobs})")
    if cloud_down:
        reasons.append(f"chmura odpowiada bledami ({cloud_ok}/{len(cloud)} udanych w {window_s // 60} min)")
    if local_queue >= 2:
        reasons.append(f"lokalna Ollama ma kolejke ({local_queue} zapytan)")
    return {"llm_jobs": len(llm_jobs), "limit": settings.agents_max_llm_jobs, "cloud_calls": len(cloud),
            "cloud_ok": cloud_ok, "cloud_down": cloud_down, "local_queue": local_queue, "overloaded": overloaded,
            "powod": "; ".join(reasons)}


def job_silence(jobs: list[Job]) -> dict[int, int]:
    """{job_id: minuty ciszy} dla dzialajacych zadan (pasek stanu w panelu)."""
    now = time.time()
    out = {}
    for j in jobs:
        q = silence_s(j, now)
        if q is not None:
            out[j.id] = int(q // 60)
    return out


def summary(events: list[dict]) -> str:
    if not events:
        return ""
    parts = [f"#{e['job']} {e['kind']} ({e['typ']}, {e['cisza_min']} min): {e['akcja']}" for e in events]
    return "Straznik kolejki: " + "; ".join(parts)


