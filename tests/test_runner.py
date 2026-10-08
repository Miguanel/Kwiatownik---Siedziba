import asyncio

from app.worker.runner import JobRunner


def test_queue_limits_concurrency_and_reports_statuses():
    statuses: dict[int, list[str]] = {}
    running_now, peak = 0, 0

    async def work():
        nonlocal running_now, peak
        running_now += 1
        peak = max(peak, running_now)
        await asyncio.sleep(0.02)
        running_now -= 1

    async def main():
        r = JobRunner(2, on_status=lambda j, s: statuses.setdefault(j, []).append(s))
        for j in range(1, 6):
            r.start(j, work)
        r.request_stop(5)            # jeszcze w kolejce -> nie wystartuje
        await asyncio.gather(*r.tasks.values())

    asyncio.run(main())
    assert peak == 2
    assert all(statuses[j] == ["running", "done"] for j in range(1, 5))
    assert statuses[5] == ["cancelled"]


def test_failure_is_reported():
    statuses, logs = {}, []

    async def boom():
        raise ValueError("x")

    async def main():
        r = JobRunner(1, on_status=lambda j, s: statuses.setdefault(j, []).append(s), on_log=lambda j, m: logs.append(m))
        r.start(1, boom)
        await asyncio.gather(*r.tasks.values())

    asyncio.run(main())
    assert statuses[1] == ["running", "failed"] and "ValueError" in logs[0]


def test_shutdown_leaves_status_for_resume():
    statuses, logs = {}, []

    async def slow():
        await asyncio.sleep(10)

    async def main():
        r = JobRunner(1, on_status=lambda j, s: statuses.setdefault(j, []).append(s), on_log=lambda j, m: logs.append(m))
        r.start(1, slow)
        r.start(2, slow)
        await asyncio.sleep(0.01)
        await r.shutdown()

    asyncio.run(main())
    assert statuses == {1: ["running"]}          # brak "cancelled" -> po restarcie wykryte jako przerwane
    assert any("zamykanie" in m for m in logs)


def test_activity_visible_while_running_and_cleared_after():
    from app.worker import activity
    seen = {}

    async def work():
        activity.set(7, "pobieram", "https://x.example/a")
        seen["during"] = activity.get(7)

    async def main():
        r = JobRunner(1)
        r.start(7, work)
        await asyncio.gather(*r.tasks.values())

    asyncio.run(main())
    assert seen["during"]["step"] == "pobieram" and seen["during"]["detail"] == "https://x.example/a"
    assert activity.get(7) is None                                   # po zakonczeniu znika
    assert activity.recent_events(5)[0]["step"] == "pobieram"


def test_bypass_queue_starts_immediately_even_when_queue_is_full():
    order = []

    async def slow():
        await asyncio.sleep(0.05)
        order.append("slow")

    async def pull():
        order.append("pull")

    async def main():
        r = JobRunner(1)
        r.start(1, slow)
        r.start(2, slow)
        r.start(3, pull, bypass_queue=True)
        await asyncio.gather(*r.tasks.values())

    asyncio.run(main())
    assert order[0] == "pull"


def test_tests_use_isolated_database():
    from app.config import settings
    assert "siedziba-test-" in settings.database_url and settings.llm_providers == ""


def test_stop_forces_cancel_when_job_hangs():
    """'Zatrzymaj' na zadaniu, ktore czeka (np. na LLM) i nie sprawdza flagi -> po grace przerwane sila."""
    statuses, logs = {}, []

    async def hang():
        await asyncio.sleep(30)

    async def main():
        r = JobRunner(1, on_status=lambda j, s: statuses.setdefault(j, []).append(s), on_log=lambda j, m: logs.append(m))
        r.start(1, hang)
        await asyncio.sleep(0.01)
        assert r.request_stop(1, grace=0.05) and not r.request_stop(99)
        await asyncio.wait_for(asyncio.gather(*r.tasks.values()), 2)

    asyncio.run(main())
    assert statuses[1] == ["running", "cancelled"] and "Zatrzymano" in logs[-1]
