"""Zadania agentow w kolejce: 'agent_audit' (Audytor wiedzy) i 'agent_dispatch' (Zleceniodawca)."""
from app.agents.audit import run_audit
from app.agents.deploy import run_deploy
from app.agents.dispatch import run_dispatch
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


async def run_agent_audit(job_id: int, runner: JobRunner, llm=None, trigger: str = "reczny") -> int:
    activity.set(job_id, "audyt bazy wiedzy", "")
    return await run_audit(llm, trigger=trigger, job_id=job_id, logf=lambda m: job_log(job_id, m))


async def run_agent_dispatch(job_id: int, runner: JobRunner, state, trigger: str = "reczny",
                             max_tasks: int | None = None) -> int:
    activity.set(job_id, "zlecanie zadan z audytu", "")
    return await run_dispatch(state, trigger=trigger, job_id=job_id, logf=lambda m: job_log(job_id, m),
                              max_tasks=max_tasks)


async def run_agent_deploy(job_id: int, runner: JobRunner, llm=None, trigger: str = "reczny", force: bool = False,
                           dry_run: bool = False) -> int:
    activity.set(job_id, "publikacja nowej wersji Kwiatownika" if not dry_run else "sprawdzanie nowych danych", "")
    return await run_deploy(trigger=trigger, job_id=job_id, logf=lambda m: job_log(job_id, m), force=force,
                            dry_run=dry_run)
