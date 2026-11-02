"""Kolejka zadan w tle z limitem jednoczesnie dzialajacych (bez bazy - testowalna osobno)."""
import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.worker import activity, snapshots

log = logging.getLogger(__name__)

StatusFn = Callable[[int, str], None]   # (job_id, "running" | "done" | "failed" | "cancelled")
LogFn = Callable[[int, str], None]



def error_place(exc: BaseException) -> str:
    """' (app/pipeline/x.py:212 w f)' - ostatnie miejsce bledu w kodzie Siedziby (bez bibliotek)."""
    import traceback
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/app/" in f.filename.replace("\\", "/")]
    if not frames:
        return ""
    f = frames[-1]
    path = f.filename.replace("\\", "/")
    return f" ({path[path.rindex('/app/') + 1:]}:{f.lineno} w {f.name})"


class _NoLimit:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


MAX_LIMIT = 6   # najwiecej zadan naraz, jakie mozna ustawic w panelu (/agents)

# Zadania, ktore wiekszosc czasu czekaja na LLM i moga trafic do lokalnej Ollamy (etap 3 diagnostyki 2026-11-02).
# Gdy chmura (Gemini/Groq) ma limity, zostaje jedno miejsce w Ollamie - wtedy tylko tyle z nich rusza naraz,
# a reszta miejsc idzie na zadania bez LLM (zdjecia, NLLB, trening sieci, eksport, skan na regulach).
# Skan nie jest tu liczony: klasyfikuje strony tylko chmura (cloud_only), bez niej - regulami i siecia.
LLM_HEAVY_KINDS = frozenset({"plant_research", "translate", "enrich", "plant_magic", "plant_merge", "plant_organize",
                             "plant_place", "leads_search", "build_profile"})
RECHECK_S = 60.0   # co tyle sekund kolejka sprawdza ponownie, czy chmura juz odpowiada


class JobRunner:
    """Limit zadan naraz mozna zmieniac w trakcie pracy (set_limit, 0-6). 0 = wstrzymane: dzialajace koncza
    swoja prace, nowe czekaja w kolejce. Zadania z bypass_queue (agenci, fanpage, pobieranie modelu) - poza limitem."""

    def __init__(self, max_concurrent: int = 3, on_status: StatusFn | None = None, on_log: LogFn | None = None):
        self.max_concurrent = max(0, min(MAX_LIMIT, int(max_concurrent)))
        self.on_status = on_status or (lambda j, s: None)
        self.on_log = on_log or (lambda j, m: None)
        self.tasks: dict[int, asyncio.Task] = {}
        self.stop_flags: set[int] = set()
        self.active: set[int] = set()
        self.limited: set[int] = set()       # dzialajace zadania liczone do limitu
        self.queue: list[int] = []           # czekajace na miejsce (kolejnosc zlecenia)
        self._wake: asyncio.Event | None = None
        self.shutting_down = False
        self.heavy: set[int] = set()         # zadania z LLM_HEAVY_KINDS (w kolejce i dzialajace)
        # ile zadan z LLM naraz pozwalaja modele: None = bez dodatkowego limitu (chmura dziala); ustawia main.py
        self.llm_capacity: Callable[[], int | None] = lambda: None

    def set_limit(self, n: int) -> int:
        """Nowy limit zadan naraz (0-6) - dziala od razu: przy wiekszym ruszaja czekajace zadania."""
        self.max_concurrent = max(0, min(MAX_LIMIT, int(n)))
        self._notify()
        return self.max_concurrent

    def _notify(self) -> None:
        if self._wake is not None:
            self._wake.set()
            self._wake = None

    def llm_cap(self) -> int | None:
        try:
            return self.llm_capacity()
        except Exception:  # noqa: BLE001 - blad oceny modeli nie blokuje kolejki
            return None

    def _may_start(self, job_id: int) -> bool:
        if job_id in self.stop_flags:
            return True                      # zatrzymane w kolejce - wychodzi od razu (status cancelled)
        free = self.max_concurrent - len(self.limited)
        if free <= 0:
            return False
        cap = self.llm_cap()
        if cap is None:
            return job_id in self.queue[:free]
        # tylko lokalny model: zadania z LLM czekaja na wolne miejsce w nim, zadania bez LLM je wyprzedzaja
        heavy_running = len(self.limited & self.heavy)
        eligible = [j for j in self.queue if j not in self.heavy or heavy_running < cap]
        return job_id in eligible[:free]

    def waiting_for_llm(self) -> list[int]:
        """Zadania z LLM czekajace w kolejce, bo chmura ma limity, a miejsce w lokalnym modelu jest zajete."""
        cap = self.llm_cap()
        if cap is None:
            return []
        if len(self.limited & self.heavy) < cap:
            return []
        return [j for j in self.queue if j in self.heavy and j not in self.stop_flags]

    async def _acquire(self, job_id: int) -> None:
        self.queue.append(job_id)
        try:
            while not self._may_start(job_id):
                if self._wake is None:
                    self._wake = asyncio.Event()
                wake = self._wake
                try:                         # limit modeli zmienia sie sam (koniec przerwy chmury) - co minute
                    await asyncio.wait_for(wake.wait(), RECHECK_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            if job_id in self.queue:
                self.queue.remove(job_id)
        if job_id not in self.stop_flags:
            self.limited.add(job_id)
        self._notify()                       # nastepny w kolejce sprawdzi, czy jest miejsce

    def _release(self, job_id: int) -> None:
        if job_id in self.limited:
            self.limited.discard(job_id)
            self._notify()

    def start(self, job_id: int, work: Callable[[], Awaitable[None]], bypass_queue: bool = False,
              kind: str | None = None) -> None:
        """Dodaje zadanie do kolejki; ruszy, gdy zwolni sie miejsce (status 'queued' -> 'running').
        bypass_queue=True: rusza od razu, poza limitem (np. pobieranie modelu Ollamy, na ktory czekaja inne).
        kind z LLM_HEAVY_KINDS: przy samym lokalnym modelu rusza tylko, gdy jest w nim miejsce."""
        if kind in LLM_HEAVY_KINDS:
            self.heavy.add(job_id)
        async def wrapper():
            snapshots.current_job.set(job_id)
            try:
                if not bypass_queue:
                    await self._acquire(job_id)
                async with _NoLimit():
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
                self.on_log(job_id, f"BLAD: {type(exc).__name__}: {exc}{error_place(exc)}")
                self.on_status(job_id, "failed")
            finally:
                self._release(job_id)
                self.heavy.discard(job_id)
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
        self._notify()                       # czekajace w kolejce (np. przy limicie 0) wyjda od razu
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

    @property
    def running_limited(self) -> int:
        return len(self.limited)

    async def shutdown(self) -> None:
        self.shutting_down = True
        for t in list(self.tasks.values()):
            t.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
