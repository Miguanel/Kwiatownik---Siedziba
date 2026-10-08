"""Kolejka zadan w tle z limitem jednoczesnie dzialajacych (bez bazy - testowalna osobno)."""
import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.worker import activity, snapshots

log = logging.getLogger(__name__)

StatusFn = Callable[[int, str], None]   # (job_id, "running" | "done" | "failed" | "cancelled")
LogFn = Callable[[int, str], None]


class _NoLimit:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class JobRunner:
    def __init__(self, max_concurrent: int = 3, on_status: StatusFn | None = None, on_log: LogFn | None = None):
        self.max_concurrent = max(1, max_concurrent)
        self._sem: asyncio.Semaphore | None = None
        self.on_status = on_status or (lambda j, s: None)
        self.on_log = on_log or (lambda j, m: None)
        self.tasks: dict[int, asyncio.Task] = {}
        self.stop_flags: set[int] = set()
        self.active: set[int] = set()
        self.shutting_down = False

    @property
    def sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        return self._sem

    def start(self, job_id: int, work: Callable[[], Awaitable[None]], bypass_queue: bool = False) -> None:
        """Dodaje zadanie do kolejki; ruszy, gdy zwolni sie miejsce (status 'queued' -> 'running').
        bypass_queue=True: rusza od razu, poza limitem (np. pobieranie modelu Ollamy, na ktory czekaja inne)."""
        async def wrapper():
            snapshots.current_job.set(job_id)
            try:
                async with (_NoLimit() if bypass_queue else self.sem):
                    if job_id in self.stop_flags:
                        self.on_status(job_id, "cancelled")
                        return
                    self.active.add(job_id)
                    self.on_status(job_id, "running")
                    activity.set(job_id, "start")
                    await work()
                    self.on_status(job_id, "cancelled" if job_id in self.stop_flags else "done")
            except asyncio.CancelledError:
                if job_id in self.stop_flags and not self.shutting_down:
                    # "Zatrzymaj" nie zadzialalo lagodnie w czasie grace -> zadanie przerwane sila
                    self.on_log(job_id, "Zatrzymano (przerwane - zadanie nie zakonczylo biezacego kroku na czas)")
                    self.on_status(job_id, "cancelled")
                    return
                if self.shutting_down:
                    # zamykanie aplikacji: status zostaje 'running'/'queued', wiec po starcie
                    # zadanie zostanie rozpoznane jako przerwane i (opcjonalnie) wznowione
                    self.on_log(job_id, "Przerwane - zamykanie aplikacji")
                else:
                    self.on_log(job_id, "Przerwano")
                    self.on_status(job_id, "cancelled")
                raise
            except Exception as exc:
                log.exception("Zadanie %s nieudane", job_id)
                self.on_log(job_id, f"BLAD: {type(exc).__name__}: {exc}")
                self.on_status(job_id, "failed")
            finally:
                activity.clear(job_id)
                self.active.discard(job_id)
                self.tasks.pop(job_id, None)
                self.stop_flags.discard(job_id)

        self.tasks[job_id] = asyncio.create_task(wrapper())

    def request_stop(self, job_id: int, grace: float = 15.0) -> bool:
        """Zatrzymanie: zadanie w kolejce nie wystartuje, dzialajace konczy biezacy krok; jesli po `grace`
        sekundach nadal dziala (np. czeka na LLM), zostaje przerwane sila. False = tego zadania tu nie ma."""
        task = self.tasks.get(job_id)
        if task is None:
            return False
        self.stop_flags.add(job_id)
        if job_id in self.active:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return True
            loop.call_later(grace, lambda: None if task.done() else task.cancel())
        return True

    def should_stop(self, job_id: int) -> bool:
        return job_id in self.stop_flags

    def is_running(self, job_id: int) -> bool:
        return job_id in self.tasks

    @property
    def waiting(self) -> int:
        return len(self.tasks) - len(self.active)

    async def shutdown(self) -> None:
        self.shutting_down = True
        for t in list(self.tasks.values()):
            t.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
