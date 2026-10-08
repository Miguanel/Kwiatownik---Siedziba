"""Ollama - lokalny model uruchamiany w Dockerze. Ostatnia deska ratunku: router uzywa go dopiero wtedy,
gdy wszystkie modele w chmurze (Gemini, Groq) zawioda albo maja limit."""
import json
import re
from collections.abc import Callable

import httpx

from app.llm.base import InvalidOutputError, LLMProvider, ModelInfo


class OllamaProvider(LLMProvider):
    name = "ollama"
    fallback = True   # zawsze na koncu listy modeli

    def __init__(self, url: str, num_ctx: int = 8192, concurrency: int = 1, model_ctx: dict[str, int] | None = None,
                 client: httpx.AsyncClient | None = None, timeout: float = 900):
        super().__init__("", rpm=0, concurrency=concurrency, client=client, timeout=timeout)
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        # mniejszy kontekst dla wybranych modeli (np. Bielik do scalania - krotkie prompty): mniej pamieci
        # na kontekst = wieksza czesc modelu miesci sie w GPU = duzo szybciej
        self.model_ctx = dict(model_ctx or {})

    async def list_models(self) -> list[ModelInfo]:
        data = self._check(await self.client.get(f"{self.url}/api/tags", timeout=10))
        out = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if not name or "embed" in name:
                continue
            out.append(ModelInfo(self.name, name, None))
        return out

    async def _generate(self, model: str, prompt: str, system: str | None, json_mode: bool) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        # Bielik: producent zaleca niska temperature (0.1)
        temp = 0.1 if "bielik" in model.lower() else 0.3
        body: dict = {"model": model, "messages": messages, "stream": False, "think": False,
                      "options": {"num_ctx": self.model_ctx.get(model, self.num_ctx), "temperature": temp}}
        if json_mode:
            body["format"] = "json"
        r = await self.client.post(f"{self.url}/api/chat", json=body)
        if r.status_code == 400 and "think" in r.text.lower():    # model bez trybu "myslenia"
            body.pop("think")
            r = await self.client.post(f"{self.url}/api/chat", json=body)
        data = self._check(r)
        text = (data.get("message") or {}).get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # starsze modele "myslace"
        if not text:
            raise InvalidOutputError(f"ollama/{model}: pusta odpowiedz")
        return text

    async def pull(self, model: str, on_progress: Callable[[str, int, int], None] | None = None) -> None:
        """Pobiera model (np. 'qwen3:8b'); on_progress(status, pobrane_bajty, wszystkie_bajty)."""
        async with self.client.stream("POST", f"{self.url}/api/pull", json={"model": model, "stream": True},
                                      timeout=None) as r:
            if r.status_code >= 400:
                raise InvalidOutputError(f"ollama pull {model}: HTTP {r.status_code}: {(await r.aread())[:200]!r}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise InvalidOutputError(f"ollama pull {model}: {msg['error']}")
                if on_progress:
                    on_progress(msg.get("status", ""), int(msg.get("completed") or 0), int(msg.get("total") or 0))
