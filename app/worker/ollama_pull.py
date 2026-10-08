"""Zadanie 'ollama_pull': pobranie lokalnego modelu do kontenera Ollama (z postepem w "Na zywo")."""
import json

from sqlmodel import Session

from app.config import settings
from app.db import engine
from app.llm.ollama import OllamaProvider
from app.models import Job
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


async def run_ollama_pull(job_id: int, runner: JobRunner, llm, model: str) -> None:
    provider = (llm.providers.get("ollama") if llm else None) or OllamaProvider(settings.ollama_url)
    job_log(job_id, f"Pobieram model {model} do Ollamy ({settings.ollama_url}) - to moze potrwac kilka minut")
    state = {"last_mb": -1, "status": ""}

    def progress(status: str, done: int, total: int) -> None:
        mb, total_mb = done // 1_000_000, total // 1_000_000
        activity.set(job_id, f"pobieram model {model}", f"{status} {mb}/{total_mb} MB" if total else status)
        if total and (mb - state["last_mb"] >= 100 or mb == total_mb):   # zapis do bazy co ~100 MB
            state["last_mb"] = mb
            with Session(engine) as s:
                job = s.get(Job, job_id)
                job.progress, job.total = mb, total_mb
                s.add(job)
                s.commit()
        if status != state["status"] and not total:
            state["status"] = status
            job_log(job_id, status)

    await provider.pull(model, progress)
    job_log(job_id, f"Model {model} gotowy")
    if llm:
        await llm.refresh()   # nowy model trafia na liste (jako zapasowy - na koniec)
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.stats_json = json.dumps({"model": model})
        s.add(job)
        s.commit()


async def ensure_ollama(state, attempts: int = 20, every_s: float = 15) -> None:
    """W tle po starcie: czeka, az kontener Ollamy wstanie, dopisuje jego modele do listy
    i pobiera brakujace z OLLAMA_MODELS."""
    import asyncio
    import logging

    from app.worker import actions
    log = logging.getLogger(__name__)
    llm = getattr(state, "llm", None)
    provider = llm.providers.get("ollama") if llm else None
    if not provider:
        return
    for _ in range(attempts):
        try:
            await provider.list_models()
        except Exception:
            await asyncio.sleep(every_s)   # Ollama jeszcze startuje
            continue
        if "ollama" in llm.discovery_errors or not any(e.provider == "ollama" for e in llm.order):
            await llm.refresh()
        for model in await missing_ollama_models(llm):
            log.warning("Ollama: brak modelu %s - pobieram (zadanie #%s)", model, actions.start_ollama_pull(state, model))
        return
    log.warning("Ollama niedostepna pod %s - pomijam model zapasowy", settings.ollama_url)


def merge_models() -> list[str]:
    return [m.strip() for m in settings.merge_models.split(",") if m.strip()]


async def missing_ollama_models(llm) -> list[str]:
    """Modele z OLLAMA_MODELS, ktorych jeszcze nie ma w Ollamie (pusta lista, gdy Ollama niedostepna)."""
    provider = llm.providers.get("ollama") if llm else None
    if not provider:
        return []
    try:
        have = {m.model for m in await provider.list_models()}
    except Exception:
        return []
    wanted = [m.strip() for m in settings.ollama_models.split(",") if m.strip()]
    if settings.merge_pull_models:      # modele do scalania wiedzy (Bielik) - "ollama:nazwa:tag"
        wanted += [m.split(":", 1)[1] for m in merge_models() if m.startswith("ollama:") and m.split(":", 1)[1] not in wanted]
    return [m for m in wanted if m not in have and f"{m}:latest" not in have]
