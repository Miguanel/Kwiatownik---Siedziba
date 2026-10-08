import asyncio
import json

import httpx
import pytest

from app.llm import router as router_mod
from app.llm.base import (LLMProvider, ModelError, ModelInfo, QuotaExceededError, RateLimitError,
                          RequestTooLargeError, ServerError, classify_limit, parse_retry_seconds)
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.router import AllModelsFailedError, ModelRouter
from app.llm.scoring import score_gemini, score_groq


# ---------------------------------------------------------------- scoring
def test_scoring_filters_and_orders():
    assert score_gemini("text-embedding-004") is None
    assert score_gemini("gemini-3.5-flash-preview-tts") is None
    assert score_gemini("gemini-flash-latest") is None
    assert score_gemini("gemini-3.8-flash") > score_gemini("gemini-3.5-flash") > score_gemini("gemini-3.5-flash-lite")
    assert score_gemini("gemini-3.5-flash") > score_gemini("gemini-3.5-flash-preview-05-20")
    assert score_groq("whisper-large-v3") is None
    assert score_groq("openai/gpt-oss-120b") > score_groq("llama-3.3-70b-versatile") > score_groq("llama-3.1-8b-instant")


# ---------------------------------------------------------------- odkrywanie przez HTTP
def _run(coro):
    return asyncio.run(coro)


def test_gemini_and_groq_list_models_parsing():
    def gemini(req: httpx.Request):
        return httpx.Response(200, json={"models": [
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"], "inputTokenLimit": 1000},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        ]})

    def groq(req: httpx.Request):
        return httpx.Response(200, json={"data": [
            {"id": "llama-3.3-70b-versatile", "active": True, "context_window": 131072},
            {"id": "old-model", "active": False},
        ]})

    g = GeminiProvider("k", client=httpx.AsyncClient(transport=httpx.MockTransport(gemini)))
    q = GroqProvider("k", client=httpx.AsyncClient(transport=httpx.MockTransport(groq)))
    assert [m.model for m in _run(g.list_models())] == ["gemini-3.5-flash"]
    assert [m.model for m in _run(q.list_models())] == ["llama-3.3-70b-versatile"]


# ---------------------------------------------------------------- router na atrapach
class FakeProvider(LLMProvider):
    def __init__(self, name, models, broken=(), rate_limited=(), bad_json=(), overloaded=(), quota=(), too_large=()):
        super().__init__("key", rpm=0, concurrency=5)
        self.name = name
        self.models = models
        self.broken, self.rate_limited, self.bad_json = set(broken), set(rate_limited), set(bad_json)
        self.overloaded, self.quota, self.too_large = set(overloaded), set(quota), set(too_large)
        self.calls: list[str] = []

    async def list_models(self):
        return [ModelInfo(self.name, m) for m in self.models]

    async def _generate(self, model, prompt, system, json_mode):
        self.calls.append(model)
        if model in self.broken:
            raise ModelError("404")
        if model in self.rate_limited:
            raise RateLimitError("429", retry_after=5)
        if model in self.overloaded:
            raise ServerError("503 high demand")
        if model in self.quota:
            raise QuotaExceededError("429 quota per day")
        if model in self.too_large:
            raise RequestTooLargeError("413")
        if model in self.bad_json:
            return "to nie jest json"
        return json.dumps({"model": model})


def _router(tmp_path=None, **kw):
    gem = FakeProvider("gemini", ["gemini-3.5-flash", "gemini-3.8-flash", "text-embedding-004"], **kw.pop("gem", {}))
    groq = FakeProvider("groq", ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "whisper-large-v3"], **kw.pop("groq", {}))
    r = ModelRouter([gem, groq], state_path=(tmp_path / "state.json") if tmp_path else None, **kw)
    _run(r.refresh())
    return r, gem, groq


def keys(r):
    return [e.key for e in r.order]


def test_refresh_builds_ranking_without_non_text_models():
    r, *_ = _router()
    assert set(keys(r)) == {"gemini:gemini-3.8-flash", "gemini:gemini-3.5-flash",
                            "groq:openai/gpt-oss-120b", "groq:llama-3.3-70b-versatile"}
    top2 = keys(r)[:2]
    assert set(top2) == {"gemini:gemini-3.8-flash", "groq:openai/gpt-oss-120b"}  # najlepsi obu providerow


def test_failed_model_goes_to_end_and_next_request_starts_from_new_first():
    r, gem, _ = _router(gem={"broken": ["gemini-3.8-flash"]}, priority=["gemini:gemini-3.8-flash"])
    assert keys(r)[0] == "gemini:gemini-3.8-flash"

    res = _run(r.complete("json", json_mode=True))
    assert res.model != "gemini-3.8-flash" and len(res.attempts) == 1
    assert keys(r)[-1] == "gemini:gemini-3.8-flash"          # zepsuty na koncu
    assert r.order[-1].failures == 1 and r.order[-1].cooldown_until > 0

    first = keys(r)[0]
    res2 = _run(r.complete("json", json_mode=True))
    assert f"{res2.provider}:{res2.model}" == first and res2.attempts == []
    assert gem.calls.count("gemini-3.8-flash") == 1          # nie probowany ponownie


def test_bad_json_counts_as_failure():
    r, *_ = _router(groq={"bad_json": ["openai/gpt-oss-120b"]}, priority=["groq:openai/gpt-oss-120b"])
    res = _run(r.complete("json", json_mode=True))
    assert res.model != "openai/gpt-oss-120b"
    assert keys(r)[-1] == "groq:openai/gpt-oss-120b"


def test_all_fail_raises(monkeypatch):
    r, *_ = _router(gem={"broken": ["gemini-3.5-flash", "gemini-3.8-flash"]},
                    groq={"rate_limited": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]})
    with pytest.raises(AllModelsFailedError):
        _run(r.complete("json", json_mode=True))
    assert all(e.failures == 1 for e in r.order)
    with pytest.raises(AllModelsFailedError, match="przerwe"):
        _run(r.complete("json"))  # wszystkie na przerwie


def test_state_survives_restart_and_reset(tmp_path):
    r, *_ = _router(tmp_path, gem={"broken": ["gemini-3.8-flash"]}, priority=["gemini:gemini-3.8-flash"])
    _run(r.complete("json", json_mode=True))
    saved = keys(r)

    r2 = ModelRouter([FakeProvider("gemini", []), FakeProvider("groq", [])], state_path=tmp_path / "state.json")
    assert keys(r2) == saved and r2.order[-1].failures == 1

    r.reset()
    assert keys(r)[0] == "gemini:gemini-3.8-flash" and all(e.failures == 0 for e in r.order)


def test_parallel_requests_spread_over_models_when_rate_limited(monkeypatch):
    gem = FakeProvider("gemini", ["gemini-3.8-flash", "gemini-3.5-flash"])
    groq = FakeProvider("groq", ["openai/gpt-oss-120b"])
    gem.rpm = groq.rpm = 1                                     # 1 zapytanie / minute na model
    r = ModelRouter([gem, groq])
    _run(r.refresh())

    async def no_sleep(_):
        return None
    monkeypatch.setattr(router_mod.asyncio, "sleep", no_sleep)
    results = _run(r.map(["json"] * 3, json_mode=True))
    assert len({res.model for res in results}) == 3            # kazde zapytanie do innego modelu


def test_transient_error_keeps_position_until_demote_after():
    r, gem, _ = _router(gem={"overloaded": ["gemini-3.8-flash"]}, priority=["gemini:gemini-3.8-flash"],
                        demote_after=3)
    for i in range(1, 4):
        r.order[0].cooldown_until = 0                         # udajemy, ze przerwa minela
        res = _run(r.complete("json", json_mode=True))
        assert res.model != "gemini-3.8-flash"
        entry = next(e for e in r.order if e.model == "gemini-3.8-flash")
        assert entry.streak == i
        if i < 3:
            assert keys(r)[0] == "gemini:gemini-3.8-flash"    # chwilowy blad - zostaje na gorze
    assert keys(r)[-1] == "gemini:gemini-3.8-flash"           # 3. porazka z rzedu - spada na koniec


def test_router_tracks_inflight_and_recent_calls():
    r, *_ = _router()

    async def run():
        res = await r.complete("json", json_mode=True)
        return res
    asyncio.run(run())
    assert dict(r.inflight) == {} and r.recent[-1]["ok"] is True


def test_limit_errors_are_parsed_from_real_messages():
    otpm = '{"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` on output tokens per minute (OTPM): Limit 1000, Used 787. Please try again in 11.2s."}}'
    daily = '{"error":{"code":429,"message":"You exceeded your current quota, please check your plan and billing details.","details":[{"violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}'
    minute = '{"error":{"code":429,"message":"You exceeded your current quota","details":[{"violations":[{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},{"retryDelay":"34s"}]}}'
    e1 = classify_limit("x", otpm, parse_retry_seconds(otpm))
    e2 = classify_limit("x", daily, parse_retry_seconds(daily))
    e3 = classify_limit("x", minute, parse_retry_seconds(minute))
    assert type(e1) is RateLimitError and e1.cooldown_s == 11.2
    assert type(e2) is QuotaExceededError and e2.cooldown_s >= 3600
    assert type(e3) is RateLimitError and e3.cooldown_s == 34.0
    assert parse_retry_seconds("try again in 1m23.5s") == 83.5 and parse_retry_seconds("in 450ms") is None


def test_rate_limited_model_is_rated_down_and_moved_to_end_at_once():
    r, *_ = _router(groq={"rate_limited": ["openai/gpt-oss-120b"]}, priority=["groq:openai/gpt-oss-120b"])
    before = r.order[0].rating
    res = _run(r.complete("json", json_mode=True))
    e = next(x for x in r.order if x.model == "openai/gpt-oss-120b")
    assert res.model != "openai/gpt-oss-120b" and keys(r)[-1] == "groq:openai/gpt-oss-120b"   # od razu na koniec
    assert e.limit_hits == 1 and e.rating < before and 0 < e.cooling() <= 5


def test_quota_exhausted_gets_long_pause_and_too_large_is_skipped_without_penalty():
    r, *_ = _router(gem={"quota": ["gemini-3.8-flash"]}, groq={"too_large": ["openai/gpt-oss-120b"]},
                    priority=["groq:openai/gpt-oss-120b", "gemini:gemini-3.8-flash"])
    res = _run(r.complete("json", json_mode=True))
    assert res.model not in ("openai/gpt-oss-120b", "gemini-3.8-flash")
    big = next(x for x in r.order if x.model == "openai/gpt-oss-120b")
    quota = next(x for x in r.order if x.model == "gemini-3.8-flash")
    assert big.failures == 0 and big.cooling() == 0 and keys(r)[0] == "groq:openai/gpt-oss-120b"
    assert quota.cooling() > 3600 and quota.limit_hits == 1


def test_router_waits_for_limit_renewal_instead_of_failing(monkeypatch):
    import app.llm.router as router_mod

    clock = {"t": 1000.0}

    class FakeTime:
        @staticmethod
        def time():
            return clock["t"]

        @staticmethod
        def monotonic():
            return clock["t"]

    async def fake_sleep(s):
        clock["t"] += s

    gem = FakeProvider("gemini", ["gemini-3.8-flash"])
    calls = {"n": 0}
    orig = gem._generate

    async def flaky(model, prompt, system, json_mode):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("429 per minute", retry_after=20)
        return await orig(model, prompt, system, json_mode)

    gem._generate = flaky
    monkeypatch.setattr(router_mod, "time", FakeTime)
    monkeypatch.setattr(router_mod.asyncio, "sleep", fake_sleep)
    r = ModelRouter([gem], max_wait_s=90)
    _run(r.refresh())
    res = _run(r.complete("json", json_mode=True))
    assert res.model == "gemini-3.8-flash" and calls["n"] == 2 and clock["t"] >= 1020


# ---------------------------------------------------------------- Ollama (lokalny zapas)
def test_ollama_provider_list_generate_and_pull():
    from app.llm.ollama import OllamaProvider

    def handler(req: httpx.Request):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}]})
        if req.url.path == "/api/chat":
            body = json.loads(req.content)
            assert body["format"] == "json" and body["think"] is False and body["options"]["num_ctx"] == 8192
            return httpx.Response(200, json={"message": {"content": '<think>hmm</think>{"ok": 1}'}})
        if req.url.path == "/api/pull":
            lines = [{"status": "pulling manifest"}, {"status": "downloading", "completed": 50, "total": 100},
                     {"status": "success"}]
            return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))
        return httpx.Response(404)

    p = OllamaProvider("http://ollama:11434", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert [m.model for m in _run(p.list_models())] == ["qwen3:8b"]
    assert _run(p.generate("qwen3:8b", "json", json_mode=True)) == '{"ok": 1}'
    seen = []
    _run(p.pull("qwen3:8b", lambda st, d, t: seen.append((st, d, t))))
    assert seen[1] == ("downloading", 50, 100) and seen[-1][0] == "success"


def test_ollama_is_last_resort():
    local = FakeProvider("ollama", ["qwen3:8b"])
    local.fallback = True
    gem = FakeProvider("gemini", ["gemini-3.8-flash"], broken=["gemini-3.8-flash"])
    groq = FakeProvider("groq", ["openai/gpt-oss-120b"], rate_limited=["openai/gpt-oss-120b"])
    r = ModelRouter([local, gem, groq])
    _run(r.refresh())
    assert keys(r)[-1] == "ollama:qwen3:8b"                        # zapas zawsze na koncu listy
    res = _run(r.complete("json", json_mode=True))
    assert res.provider == "ollama" and len(res.attempts) == 2     # dopiero po porazce chmury
    assert keys(r)[-1] == "ollama:qwen3:8b"                        # zepsute modele chmury nie spadaja za niego
    assert gem.calls and groq.calls and local.calls == ["qwen3:8b"]
    r.reset()
    assert keys(r)[-1] == "ollama:qwen3:8b"


def test_provider_outage_pauses_whole_provider():
    """5xx na 3 roznych modelach Gemini -> caly Gemini wstrzymany, zapytania ida od razu do Groq."""
    gem = FakeProvider("gemini", ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.8-flash"],
                       overloaded=["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.8-flash"])
    groq = FakeProvider("groq", ["openai/gpt-oss-120b"], rate_limited=["openai/gpt-oss-120b"])
    r = ModelRouter([gem, groq])
    _run(r.refresh())
    for e in [x for x in r.order if x.provider == "gemini"]:
        r._on_failure(e, ServerError("503 high demand"))
    assert r.provider_pause.get("gemini", 0) > 0
    assert all(e.cooling() > 200 for e in r.order if e.provider == "gemini")
    groq.rate_limited.clear()
    gem.calls.clear()
    res = _run(r.complete("y"))
    assert res.provider == "groq" and gem.calls == []     # Gemini pominiety w czasie przerwy


def test_prefer_tries_listed_models_in_order_then_normal_list():
    """Scalanie po polsku: Bielik 11B -> Bielik 4.5B -> zwykla lista (np. brak modelu w Ollamie)."""
    local = FakeProvider("ollama", [], broken=["bielik-11b"], bad_json=[])
    r, gem, groq = _router()
    r.providers["ollama"] = local
    prefer = ["ollama:bielik-11b", "ollama:bielik-4.5b"]
    res = _run(r.complete("json", json_mode=True, prefer=prefer))
    assert (res.provider, res.model) == ("ollama", "bielik-4.5b")
    assert local.calls == ["bielik-11b", "bielik-4.5b"]
    # zepsuty model preferowany jest pomijany przez chwile (bez ponownego pytania)
    _run(r.complete("json", json_mode=True, prefer=prefer))
    assert local.calls.count("bielik-11b") == 1
    # oba zawodza (zly JSON nie blokuje modelu na dluzej) -> zwykla lista modeli
    local.bad_json.add("bielik-4.5b")
    res = _run(r.complete("json", json_mode=True, prefer=prefer))
    assert res.provider in ("gemini", "groq")
    assert keys(r)[0] != "ollama:bielik-4.5b"                # preferowane nie psuja rankingu


def test_prefer_timeout_skips_slow_local_model():
    """Bielik za wolny na tym komputerze: po limicie czasu zapytanie idzie do chmury, a model jest pomijany."""
    class Slow(FakeProvider):
        async def _generate(self, model, prompt, system, json_mode):
            self.calls.append(model)
            await asyncio.sleep(5)
            return json.dumps({"model": model})
    local = Slow("ollama", [])
    r, gem, groq = _router()
    r.providers["ollama"] = local
    res = _run(r.complete("json", json_mode=True, prefer=["ollama:bielik-11b"], prefer_timeout=0.05))
    assert res.provider in ("gemini", "groq")
    _run(r.complete("json", json_mode=True, prefer=["ollama:bielik-11b"], prefer_timeout=0.05))
    assert local.calls == ["bielik-11b"]                       # drugi raz juz nie czekamy


def test_ollama_merge_models_get_smaller_context():
    from app.llm.ollama import OllamaProvider
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen[body["model"]] = body["options"]["num_ctx"]
        return httpx.Response(200, json={"message": {"content": "{}"}})
    p = OllamaProvider("http://o", num_ctx=8192, model_ctx={"SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0": 4096},
                       client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    _run(p.generate("SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0", "x", json_mode=True))
    _run(p.generate("qwen3:8b", "x", json_mode=True))
    assert seen == {"SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0": 4096, "qwen3:8b": 8192}


def test_cloud_only_never_uses_local_fallback():
    r, gem, groq = _router(gem={"rate_limited": ["gemini-3.5-flash", "gemini-3.8-flash"]},
                           groq={"rate_limited": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]})
    local = FakeProvider("ollama", ["qwen3:8b"])
    local.fallback = True
    r.providers["ollama"] = local
    _run(r.refresh())
    assert any(e.provider == "ollama" for e in r.order)
    with pytest.raises(AllModelsFailedError):
        _run(r.complete("json", json_mode=True, cloud_only=True))
    assert local.calls == []
    assert _run(r.complete("json", json_mode=True)).provider == "ollama"     # zwykle zapytanie: zapas dziala
