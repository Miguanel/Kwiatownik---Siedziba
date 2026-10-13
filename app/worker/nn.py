"""Zadania sieci neuronowych w kolejce: 'nn_train' (trening) i 'nn_init' (pobranie enkodera e5)."""
import asyncio
import json

from sqlmodel import Session

from app.db import engine
from app.models import Job
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


def _stats(job_id: int, stats: dict) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job:
            job.stats_json = json.dumps(stats, ensure_ascii=False)
            s.add(job)
            s.commit()


def _progress(job_id: int, done: int, total: int) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job:
            job.progress, job.total = done, total
            s.add(job)
            s.commit()


async def run_nn_train(job_id: int, runner: JobRunner, llm=None, tasks: list[str] | None = None,
                       encoder: str | None = None) -> None:
    """Trenuje wybrane zadania po kolei (CPU w osobnym watku - panel dziala dalej)."""
    from app.nn.tasks import TASKS
    from app.nn.train import NotEnoughData, train_task
    todo = [t for t in (tasks or list(TASKS)) if t in TASKS]
    stats: dict = {"zadania": {}}
    _progress(job_id, 0, len(todo))
    for i, task in enumerate(todo):
        if runner.should_stop(job_id):
            job_log(job_id, "Zatrzymano")
            break
        title = TASKS[task].title
        activity.set(job_id, f"trening sieci: {title}", "przygotowanie zbioru")

        def prog(done, total, _t=title):
            activity.set(job_id, f"trening sieci: {_t}", f"osadzenia {done}/{total}")

        try:
            row, note = await asyncio.to_thread(
                train_task, task, encoder, lambda m: job_log(job_id, m), lambda: runner.should_stop(job_id), prog,
                job_id)
            m = json.loads(row.metrics_json or "{}")
            stats["zadania"][task] = {"wersja": row.version, "status": row.status, "enkoder": row.encoder,
                                      "trafnosc": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
                                      "pokrycie": (m.get("effective") or {}).get("coverage")}
            job_log(job_id, f"{title}: {note}")
        except NotEnoughData as exc:
            stats["zadania"][task] = {"pominiete": str(exc)}
            job_log(job_id, f"Pominieto - za malo danych. {exc}")
        except InterruptedError:
            job_log(job_id, f"{title}: zatrzymano")
            break
        _progress(job_id, i + 1, len(todo))
        _stats(job_id, stats)
    activity.clear(job_id)


async def run_nn_init(job_id: int, runner: JobRunner, llm=None) -> None:
    """Pobiera enkoder multilingual-e5-small (ONNX) z Hugging Face."""
    from app.nn.encoders import download_e5, get_encoder
    state = {"last": -1}

    def progress(fname: str, done: int, total: int) -> None:
        mb, tmb = done // 1_000_000, total // 1_000_000
        activity.set(job_id, "pobieram enkoder e5", f"{fname} {mb}/{tmb} MB" if total else f"{fname} {mb} MB")
        if mb - state["last"] >= 20:
            state["last"] = mb
            _progress(job_id, mb, tmb or mb)

    job_log(job_id, "Pobieram enkoder multilingual-e5-small (ONNX, ~120 MB) z Hugging Face")
    got = await download_e5(progress)
    for fname, src in got.items():
        job_log(job_id, f"{fname}: {src}")
    enc = get_encoder("e5")
    vec = await asyncio.to_thread(enc.encode, ["Napar z kwiatów rumianku łagodzi trawienie."])
    job_log(job_id, f"Enkoder dziala: wektor {vec.shape[1]} wymiarow. Nastepny trening uzyje e5 (i porowna z n-gramami).")
    _stats(job_id, {"pliki": got})
    activity.clear(job_id)
