"""Zadanie 'fb_post': nowy post na fanpage Kwiatownika."""
from app.config import settings
from app.marketing.service import collect_ideas, generate_post, improve_post, review_posts
from app.worker import activity
from app.worker.jobs import job_log
from app.worker.runner import JobRunner


async def run_fb_post(job_id: int, runner: JobRunner, llm=None, kind: str = "auto", plant_id: str | None = None,
                      hint: str | None = None, count: int = 1, improve: int | None = None,
                      mode: str = "nowa", idea_id: int | None = None) -> list[int]:
    """count > 1: posty po kolei (kazdy kolejny wie o poprzednich - np. 1. i 2. post o Siedzibie).
    improve: id posta do przepisania wedlug jego oceny."""
    out = []
    if improve:
        activity.set(job_id, "redakcja posta" if mode == "redaguj" else "poprawa posta", f"#{improve}")
        pid = await improve_post(llm, improve, log=lambda m: job_log(job_id, m), job_id=job_id, mode=mode)
        return [pid]   # ocena juz zapisana w petli poprawek
    for n in range(1, max(1, count) + 1):
        if runner.should_stop(job_id):
            job_log(job_id, "Zatrzymano")
            break
        activity.set(job_id, f"pisanie posta {n}/{count}", f"{kind} {plant_id or ''}".strip())
        pid = await generate_post(llm, kind, plant_id, hint, log=lambda m: job_log(job_id, m), job_id=job_id,
                                  idea_id=idea_id)
        out.append(pid)
        if settings.marketing_auto_review:   # od razu ocena specjalisty (innym modelem niz autor)
            activity.set(job_id, f"ocena posta {n}/{count}", f"#{pid}")
            await review_posts(llm, [pid], log=lambda m: job_log(job_id, m))
    return out


async def run_fb_review(job_id: int, runner: JobRunner, llm=None, post_ids: list[int] | None = None) -> int:
    activity.set(job_id, "ocena postow", "")
    n = await review_posts(llm, post_ids, log=lambda m: job_log(job_id, m),
                           should_stop=lambda: runner.should_stop(job_id))
    job_log(job_id, f"Ocenione: {n}")
    return n


async def run_fb_ideas(job_id: int, runner: JobRunner, llm=None) -> dict:
    activity.set(job_id, "zbieranie ciekawostek", "")
    out = await collect_ideas(llm, log=lambda m: job_log(job_id, m), should_stop=lambda: runner.should_stop(job_id))
    job_log(job_id, "Statystyki: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    return out
