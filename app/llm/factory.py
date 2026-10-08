from app.config import settings
from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.ollama import OllamaProvider
from app.llm.router import ModelRouter


def _split(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_provider(name: str) -> LLMProvider | None:
    s = settings
    if name == "gemini" and s.gemini_api_key:
        return GeminiProvider(s.gemini_api_key, rpm=s.gemini_rpm, concurrency=s.gemini_concurrency)
    if name == "groq" and s.groq_api_key:
        return GroqProvider(s.groq_api_key, rpm=s.groq_rpm, concurrency=s.groq_concurrency)
    if name == "ollama" and s.ollama_url.strip():
        merge_ctx = {m.split(":", 1)[1]: s.merge_num_ctx for m in _split(s.merge_models)
                     if m.startswith("ollama:")} if s.merge_num_ctx else {}
        return OllamaProvider(s.ollama_url, num_ctx=s.ollama_num_ctx, concurrency=s.ollama_concurrency,
                              model_ctx=merge_ctx)
    return None  # nieznany provider albo brak klucza


def build_router() -> ModelRouter | None:
    providers = [p for p in map(build_provider, _split(settings.llm_providers.lower())) if p]
    if not providers:
        return None
    return ModelRouter(
        providers,
        state_path=settings.data_dir / "llm_models.json",
        priority=_split(settings.llm_model_priority),
        blocklist=_split(settings.llm_model_blocklist),
        max_per_provider=settings.llm_max_models_per_provider,
        max_attempts=settings.llm_max_attempts,
        demote_after=settings.llm_demote_after,
        max_wait_s=settings.llm_max_wait_s,
    )
