"""Inteligentny router modeli LLM.

1. refresh(): pyta kazdego providera o liste dostepnych modeli, odrzuca nietekstowe,
   ocenia je (scoring.py), normalizuje do 0-100 i uklada jedna wspolna liste - najlepsze na gorze.
2. complete(): bierze PIERWSZY dostepny model z listy. Jesli zawiedzie -> punkt karny, cooldown
   i zapytanie idzie do nastepnego modelu. Kolejne zapytanie znowu zaczyna od gory listy.
   - blad twardy (zly model/opcja, zly JSON, zly klucz) -> od razu NA KONIEC listy
   - blad chwilowy (429, 503 "high demand", timeout) -> model zostaje na swoim miejscu
     (tylko pauzuje na czas cooldownu); na koniec spada dopiero po `demote_after` porazkach z rzedu
3. Stan (kolejnosc + statystyki) zapisuje sie w data/llm_models.json i przetrwa restart.
"""
import asyncio
import json
import logging
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.llm.base import (AuthError, InvalidOutputError, LLMError, LLMProvider, LLMResult, QuotaExceededError,
                          RateLimiter, RateLimitError, RequestTooLargeError, ServerError)
from app.llm.scoring import SCORERS

log = logging.getLogger(__name__)


class AllModelsFailedError(LLMError):
    pass


@dataclass
class ModelEntry:
    provider: str
    model: str
    score: float                    # 0-100 z heurystyki (100 = najlepszy u danego providera)
    context_window: int | None = None
    pinned: bool = False            # wymuszony przez LLM_MODEL_PRIORITY
    successes: int = 0
    failures: int = 0               # punkty karne
    streak: int = 0                 # porazki z rzedu
    avg_latency: float | None = None
    last_error: str | None = None
    last_used: float | None = None  # epoch
    cooldown_until: float = 0.0     # epoch
    limit_hits: int = 0             # ile razy model wyczerpal limit (429 / kwota)
    fallback: bool = False          # model zapasowy (Ollama) - zawsze na koncu listy
    _limiter: RateLimiter | None = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def rating(self) -> float:
        """Biezaca ocena: ocena z nazwy x niezawodnosc (nowy model = pelna ocena, porazki i limity obnizaja)."""
        total = self.successes + self.failures
        reliability = (self.successes + 1) / (total + 2) * 2
        return round(self.score * min(1.0, reliability) / (1 + 0.15 * self.limit_hits), 1)

    def cooling(self, now: float | None = None) -> float:
        return max(0.0, self.cooldown_until - (now or time.time()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_limiter")
        return d


class ModelRouter:
    def __init__(self, providers: list[LLMProvider], state_path: Path | None = None,
                 priority: list[str] | None = None, blocklist: list[str] | None = None,
                 max_per_provider: int = 5, max_attempts: int = 0, demote_after: int = 3,
                 max_wait_s: float = 0.0):
        self.providers = {p.name: p for p in providers}
        self.state_path = state_path
        self.priority = [x.strip() for x in (priority or []) if x.strip()]
        self.blocklist = [x.strip().lower() for x in (blocklist or []) if x.strip()]
        self.max_per_provider = max_per_provider
        self.max_attempts = max_attempts
        self.demote_after = max(1, demote_after)
        self.max_wait_s = max_wait_s   # ile najdluzej czekac na odnowienie limitu, gdy wszystkie modele pauzuja
        self.order: list[ModelEntry] = []
        self.discovery_errors: dict[str, str] = {}
        self.last_refresh: float | None = None
        self.inflight: Counter = Counter()            # zapytania w toku per model (widok "Na zywo")
        self.recent: deque = deque(maxlen=30)         # ostatnie wywolania: czas, model, wynik
        self._server_errors: dict[str, deque] = {}    # provider -> [(czas, model)] bledow 5xx/timeout
        self.provider_pause: dict[str, float] = {}    # provider -> do kiedy wstrzymany (awaria po stronie API)
        self._prefer_skip: dict[str, float] = {}      # model preferowany -> do kiedy pomijany (po bledzie)
        self._load()

    # ------------------------------------------------------------ odkrywanie
    async def refresh(self) -> list[ModelEntry]:
        names = list(self.providers)
        found = await asyncio.gather(*(self.providers[n].list_models() for n in names), return_exceptions=True)
        fresh: list[ModelEntry] = []
        self.discovery_errors = {}
        for name, res in zip(names, found):
            if isinstance(res, Exception):
                self.discovery_errors[name] = str(res)
                log.warning("Nie udalo sie pobrac modeli %s: %s", name, res)
                fresh += [e for e in self.order if e.provider == name]  # zostaw znane
                continue
            fresh += self._rank_provider(name, res)
        self._merge(fresh)
        self.last_refresh = time.time()
        self._save()
        return self.order

    def _rank_provider(self, name: str, infos) -> list[ModelEntry]:
        scorer = SCORERS.get(name, lambda _: 50.0)
        scored = []
        for info in infos:
            key = f"{name}:{info.model}"
            if any(b in info.model.lower() for b in self.blocklist):
                continue
            s = scorer(info)
            if s is None and key not in self.priority:
                continue
            scored.append((s or 0.0, info))
        if not scored:
            return []
        top = max(s for s, _ in scored) or 1.0
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [x for x in scored if f"{name}:{x[1].model}" in self.priority]
        keep += [x for x in scored if x not in keep][: max(0, self.max_per_provider - len(keep))]
        fallback = getattr(self.providers.get(name), "fallback", False)
        return [ModelEntry(name, i.model, round(100 * s / top, 1), i.context_window,
                           pinned=f"{name}:{i.model}" in self.priority, fallback=fallback) for s, i in keep]

    def _merge(self, fresh: list[ModelEntry]) -> None:
        """Zachowuje dotychczasowa kolejnosc i statystyki; nowe modele wstawia wg oceny."""
        old = {e.key: e for e in self.order}
        fresh_keys = {e.key for e in fresh}
        merged = [e for e in self.order if e.key in fresh_keys]
        for e in fresh:
            if e.key in old:
                o = old[e.key]
                o.score, o.context_window, o.pinned = e.score, e.context_window, e.pinned
                continue
            # nowy model: przed pierwszym "zdrowym" modelem o nizszej ocenie
            pos = next((i for i, x in enumerate(merged) if x.streak == 0 and not x.pinned and x.score < e.score),
                       len(merged))
            merged.insert(pos, e)
        # przypiete zawsze na gorze, w kolejnosci z LLM_MODEL_PRIORITY
        pinned = sorted([e for e in merged if e.pinned], key=lambda e: self.priority.index(e.key))
        self.order = pinned + [e for e in merged if not e.pinned]
        self._fallbacks_last()

    def _fallbacks_last(self) -> None:
        """Modele zapasowe (lokalna Ollama) zawsze na samym koncu - uzywane dopiero, gdy chmura zawiedzie."""
        self.order = [e for e in self.order if not e.fallback] + [e for e in self.order if e.fallback]

    def reset(self) -> None:
        """Czysci punkty karne i uklada liste od nowa wg oceny."""
        for e in self.order:
            e.successes = e.failures = e.streak = e.limit_hits = 0
            e.cooldown_until, e.last_error, e.avg_latency = 0.0, None, None
        self.order.sort(key=lambda e: (not e.pinned, self.priority.index(e.key) if e.pinned else 0, -e.score))
        self._fallbacks_last()
        self._save()

    # ------------------------------------------------------------ zapytania
    def _limiter(self, e: ModelEntry) -> RateLimiter:
        if e._limiter is None:
            e._limiter = RateLimiter(self.providers[e.provider].rpm)
        return e._limiter

    def _pick(self, tried: set[str]) -> tuple[ModelEntry, float] | None:
        """Pierwszy model z listy, ktory nie ma cooldownu i ma wolny limit.
        Gdy wszystkie maja limit zajety - ten z najkrotszym czekaniem."""
        now = time.time()
        candidates = [e for e in self.order if e.key not in tried and e.provider in self.providers
                      and e.cooling(now) == 0]
        if not candidates:
            return None
        for e in candidates:
            if self._limiter(e).delay() == 0:
                return e, self._limiter(e).reserve()
        e = min(candidates, key=lambda x: self._limiter(x).delay())
        return e, self._limiter(e).reserve()

    async def complete(self, prompt: str, system: str | None = None, json_mode: bool = False,
                       avoid: set[str] | None = None, prefer: list[str] | None = None,
                       prefer_timeout: float | None = None, cloud_only: bool = False,
                       prefer_only: bool = False) -> LLMResult:
        """avoid: modele ("provider:model"), ktorych lepiej nie uzywac (np. weryfikacja innym modelem niz
        ten, ktory tekst napisal). Gdy zaden inny nie jest dostepny - uzyje ich mimo wszystko.
        prefer: modele do sprobowania NAJPIERW, po kolei (np. Bielik 11B -> Bielik 4.5B do scalania tekstow
        po polsku); gdy zawioda - zwykla lista modeli. prefer_timeout: limit sekund jednego zapytania do modelu
        preferowanego - model za wolny na tym komputerze jest pomijany przez PREFER_SLOW_PAUSE_S.
        cloud_only: bez modeli zapasowych (lokalna Ollama) - gdy wolny lokalny model nie ma sensu (np. scalanie:
        lepiej scalic regulami niz czekac kilkanascie minut na model liczacy na CPU)."""
        job = _beat()                                  # zadanie czeka na model = zyje (wykrywanie zawieszonych)
        try:
            return await self._complete_all(prompt, system, json_mode, avoid, prefer, prefer_timeout, cloud_only,
                                            prefer_only)
        finally:
            _beat(job)

    async def _complete_all(self, prompt, system, json_mode, avoid, prefer, prefer_timeout, cloud_only,
                            prefer_only) -> LLMResult:
        for key in prefer or []:
            res = await self._try_preferred(key, prompt, system, json_mode, prefer_timeout)
            if res is not None:
                return res
        if prefer_only:
            raise AllModelsFailedError("modele preferowane zawiodly (" + ", ".join(prefer or []) + ")")
        skip = {e.key for e in self.order if e.fallback} if cloud_only else set()
        if avoid:
            try:
                return await self._complete(prompt, system, json_mode, set(avoid) | skip)
            except AllModelsFailedError:
                pass
        return await self._complete(prompt, system, json_mode, skip)

    async def _complete(self, prompt: str, system: str | None, json_mode: bool, avoid: set[str]) -> LLMResult:
        if not self.order:
            await self.refresh()
        attempts: list[str] = []
        tried: set[str] = set(avoid)  # modele juz probowane w tym zapytaniu (+ te, ktorych unikamy)
        limited: set[str] = set()    # ...z nich te, ktore trafily na limit (moga wrocic po przerwie)
        waited = 0.0
        while not (self.max_attempts and len(attempts) >= self.max_attempts):
            picked = self._pick(tried)
            if picked is None:
                # wszystkie modele wyprobowane albo na przerwie: poczekaj na najblizsze odnowienie limitu
                cooling = [e.cooling() for e in self.order if e.provider in self.providers
                           and (e.key not in tried or e.key in limited) and e.cooling() > 0]
                if not cooling or waited + min(cooling) > self.max_wait_s:
                    break
                pause = min(cooling) + 0.5
                log.warning("Wszystkie modele chwilowo niedostepne - czekam %.0f s na odnowienie limitu", pause)
                await asyncio.sleep(pause)
                waited += pause
                tried -= limited
                limited.clear()
                continue
            entry, wait = picked
            tried.add(entry.key)
            if wait:
                await asyncio.sleep(wait)
            start = time.monotonic()
            self.inflight[entry.key] += 1
            try:
                text = await self.providers[entry.provider].generate(entry.model, prompt, system, json_mode)
                if json_mode:
                    _validate_json(text, entry)
            except LLMError as exc:
                self._on_failure(entry, exc)
                attempts.append(f"{entry.key}: {exc}")
                if isinstance(exc, RateLimitError) and not isinstance(exc, QuotaExceededError):
                    limited.add(entry.key)
                self.recent.append({"ts": time.time(), "model": entry.key, "ok": False,
                                    "s": round(time.monotonic() - start, 2), "error": type(exc).__name__})
                continue
            finally:
                self.inflight[entry.key] -= 1
                if self.inflight[entry.key] <= 0:
                    del self.inflight[entry.key]
            took = round(time.monotonic() - start, 2)
            self.recent.append({"ts": time.time(), "model": entry.key, "ok": True, "s": took})
            self._on_success(entry, took)
            return LLMResult(text, entry.provider, entry.model, took, attempts)
        if not attempts:
            attempts.append("wszystkie modele maja przerwe (limity)")
        raise AllModelsFailedError(" | ".join(attempts[-6:]))

    PREFER_PAUSE_S = 300.0   # model preferowany, ktory zawiodl (np. brak modelu w Ollamie), pomijany przez 5 min

    PREFER_SLOW_PAUSE_S = 1800.0   # model preferowany przekroczyl limit czasu -> pomijany przez 30 min

    async def _try_preferred(self, key: str, prompt: str, system: str | None, json_mode: bool,
                             timeout: float | None = None) -> LLMResult | None:
        provider_name, _, model = key.partition(":")
        provider = self.providers.get(provider_name)
        now = time.time()
        if not provider or not model or self._prefer_skip.get(key, 0) > now \
                or self.provider_pause.get(provider_name, 0) > now:
            return None
        start = time.monotonic()
        self.inflight[key] += 1
        try:
            gen = provider.generate(model, prompt, system, json_mode)
            text = await (asyncio.wait_for(gen, timeout) if timeout else gen)
            if json_mode:
                cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                json.loads(cleaned)
        except asyncio.TimeoutError:
            self._prefer_skip[key] = now + self.PREFER_SLOW_PAUSE_S
            self.recent.append({"ts": time.time(), "model": key, "ok": False,
                                "s": round(time.monotonic() - start, 2), "error": "Timeout"})
            log.warning("Model preferowany %s: brak odpowiedzi w %.0f s - pomijany przez %d min", key, timeout,
                        int(self.PREFER_SLOW_PAUSE_S // 60))
            return None
        except (LLMError, ValueError) as exc:
            if not isinstance(exc, InvalidOutputError) and not isinstance(exc, ValueError):
                self._prefer_skip[key] = now + self.PREFER_PAUSE_S   # brak modelu / Ollama nie dziala
            self.recent.append({"ts": time.time(), "model": key, "ok": False,
                                "s": round(time.monotonic() - start, 2), "error": type(exc).__name__})
            log.warning("Model preferowany %s zawiodl (%s: %s) - probuje nastepnego", key, type(exc).__name__,
                        str(exc)[:120])
            return None
        finally:
            self.inflight[key] -= 1
            if self.inflight[key] <= 0:
                del self.inflight[key]
        took = round(time.monotonic() - start, 2)
        self.recent.append({"ts": time.time(), "model": key, "ok": True, "s": took})
        return LLMResult(text, provider_name, model, took, [])

    async def map(self, prompts: list[str], system: str | None = None, json_mode: bool = False):
        """Wiele zapytan rownolegle - limity sprawiaja, ze rozlozą sie na kolejne modele z listy."""
        return await asyncio.gather(*(self.complete(p, system, json_mode) for p in prompts),
                                    return_exceptions=True)

    # ------------------------------------------------------------ punktacja
    def _on_success(self, e: ModelEntry, took: float) -> None:
        e.successes += 1
        e.streak = 0
        e.last_used = time.time()
        e.avg_latency = took if e.avg_latency is None else round(0.8 * e.avg_latency + 0.2 * took, 2)
        self._save()

    def _on_failure(self, e: ModelEntry, exc: LLMError) -> None:
        now = time.time()
        e.last_used = now
        e.last_error = str(exc)[:300]
        if isinstance(exc, RequestTooLargeError):
            # zapytanie za duze dla limitu TEGO modelu - pomijamy go tylko teraz, bez kary
            log.info("Model %s: zapytanie za duze dla jego limitu - probuje nastepnego", e.key)
            return
        e.failures += 1
        e.streak += 1
        e.cooldown_until = now + exc.cooldown_s if exc.cooldown_s else 0.0
        victims = [e]
        if isinstance(exc, AuthError):  # zly klucz -> caly provider w dol listy
            victims = [x for x in self.order if x.provider == e.provider]
            for x in victims:
                x.cooldown_until = e.cooldown_until
        if isinstance(exc, RateLimitError):
            e.limit_hits += 1   # limit modelu wyczerpany -> ocena w dol i od razu na koniec listy
        transient = isinstance(exc, ServerError)
        if transient:
            self._check_outage(e, now)
        if transient and e.streak < self.demote_after:
            log.warning("Model %s: chwilowy blad (%s: %s), pauza %ss - zostaje na miejscu (%d/%d)",
                        e.key, type(exc).__name__, str(exc)[:120], int(exc.cooldown_s), e.streak, self.demote_after)
        else:
            for v in victims:  # przeniesienie zepsutego modelu na koniec listy
                if v in self.order:
                    self.order.remove(v)
                    self.order.append(v)
            log.warning("Model %s zawiodl (%s: %s) - przeniesiony na koniec listy, przerwa %ss", e.key,
                        type(exc).__name__, str(exc)[:120], int(exc.cooldown_s))
        self._fallbacks_last()
        self._save()

    OUTAGE_WINDOW_S = 180.0   # okno obserwacji bledow 5xx
    OUTAGE_MODELS = 3         # tyle ROZNYCH modeli jednego providera z bledem 5xx w oknie = awaria API
    OUTAGE_PAUSE_S = 300.0    # na tyle wstrzymujemy caly provider

    def _check_outage(self, e: ModelEntry, now: float) -> None:
        """Gdy kilka roznych modeli jednego providera naraz zwraca 5xx/timeout, to awaria po stronie API
        (np. Gemini "overloaded") - wstrzymujemy caly provider, zamiast probowac model po modelu co 30 s."""
        errs = self._server_errors.setdefault(e.provider, deque(maxlen=20))
        errs.append((now, e.model))
        models = {m for t, m in errs if now - t <= self.OUTAGE_WINDOW_S}
        if len(models) >= self.OUTAGE_MODELS and self.provider_pause.get(e.provider, 0) <= now:
            until = now + self.OUTAGE_PAUSE_S
            self.provider_pause[e.provider] = until
            for x in self.order:
                if x.provider == e.provider:
                    x.cooldown_until = max(x.cooldown_until, until)
            errs.clear()
            log.warning("Provider %s: bledy serwera na %d modelach w %ds - wstrzymany na %d min (awaria API)",
                        e.provider, len(models), int(self.OUTAGE_WINDOW_S), int(self.OUTAGE_PAUSE_S // 60))

    # ------------------------------------------------------------ stan
    def snapshot(self) -> list[dict]:
        now = time.time()
        return [e.to_dict() | {"position": i + 1, "cooldown_s": round(e.cooling(now)), "rating": e.rating}
                for i, e in enumerate(self.order)]

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"order": [e.to_dict() for e in self.order]}, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.order = [ModelEntry(**d) for d in data.get("order", [])]
            self._fallbacks_last()
        except (ValueError, TypeError) as exc:
            log.warning("Nie wczytano stanu routera (%s) - zaczynam od zera", exc)

    async def aclose(self) -> None:
        await asyncio.gather(*(p.aclose() for p in self.providers.values()))


def _beat(job_id: int | None = None) -> int | None:
    """Znak zycia zadania, ktore pyta model (biezace zadanie z kolejki - contextvar)."""
    try:
        from app.worker import activity, snapshots
        job_id = job_id if job_id is not None else snapshots.current_job.get()
        activity.beat(job_id)
        return job_id
    except Exception:  # noqa: BLE001 - router dziala tez poza aplikacja (testy, skrypty)
        return None


def _validate_json(text: str, entry: ModelEntry) -> None:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        json.loads(cleaned)
    except ValueError as exc:
        raise InvalidOutputError(f"{entry.key}: niepoprawny JSON") from exc
