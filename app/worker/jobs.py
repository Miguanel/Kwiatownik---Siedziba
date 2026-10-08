"""Integracja kolejki zadan z baza (statusy i logi w tabeli Job)."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import Job, JobStatus
from app.worker import activity
from app.worker.runner import JobRunner

_TZ = ZoneInfo(settings.timezone)

__all__ = ["JobRunner", "job_log", "set_status", "make_runner", "mark_interrupted_jobs", "retry_depth"]


def job_log(job_id: int, message: str) -> None:
    activity.beat(job_id)
    stamp = datetime.now(_TZ).strftime("%H:%M:%S")  # czas lokalny (w Dockerze domyslnie bylby UTC)
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job:
            job.log = (job.log or "") + f"{stamp} {message}\n"
            s.add(job)
            s.commit()


def set_status(job_id: int, status: str | JobStatus, **fields) -> None:
    status = JobStatus(status)
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if not job:
            return
        job.status = status
        if status == JobStatus.running:
            activity.beat(job_id)
            job.started_at = datetime.now(timezone.utc)
        for k, v in fields.items():
            setattr(job, k, v)
        if status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
            job.finished_at = datetime.now(timezone.utc)
        s.add(job)
        s.commit()


def make_runner(max_concurrent: int) -> JobRunner:
    return JobRunner(max_concurrent, on_status=lambda j, st: set_status(j, st), on_log=job_log)


def mark_interrupted_jobs() -> list[int]:
    """Po restarcie aplikacji zadania 'running'/'queued' z poprzedniego uruchomienia sa juz martwe.
    Zwraca ich id (do automatycznego ponowienia)."""
    ids = []
    with Session(engine) as s:
        for job in s.exec(select(Job).where(Job.status.in_([JobStatus.running, JobStatus.queued]))).all():
            job.status = JobStatus.failed
            job.log = (job.log or "") + "Przerwane przez restart aplikacji\n"
            s.add(job)
            ids.append(job.id)
        s.commit()
    return ids


def retry_depth(job_id: int) -> int:
    """Ile razy z rzedu to zadanie bylo juz ponawiane (lancuch retry_of)."""
    depth = 0
    with Session(engine) as s:
        job = s.get(Job, job_id)
        while job and job.retry_of and depth < 50:
            depth += 1
            job = s.get(Job, job.retry_of)
    return depth
