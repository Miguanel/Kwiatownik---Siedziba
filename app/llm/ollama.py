"""Ollama - lokalny model uruchamiany w Dockerze. Ostatnia deska ratunku: router uzywa go dopiero wtedy,
gdy wszystkie modele w chmurze (Gemini, Groq) zawioda albo maja limit."""
import json
import logging
import re
import time
from collections.abc import Callable

import httpx

from app.llm.base import InvalidOutputError, LLMProvider, ModelInfo

log = logging.getLogger(__name__)


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
        # R8: model, ktory nie miesci sie w pamieci karty, liczy czesc warstw na CPU (dawniej 790 s na odpowiedz).
        # Po wolnej odpowiedzi sprawdzamy /api/ps; gdy w VRAM jest < 90% modelu, kontekst tego modelu jest
        # zmniejszany o polowe (do MIN_CTX) - mniej pamieci na kontekst = wiecej warstw w GPU.
        self.vram: dict[str, dict] = {}            # model -> {"procent_vram", "num_ctx", "sprawdzono"}

    SLOW_S = 60.0
    MIN_CTX = 2048

    def ctx_for(self, model: str) -> int:
        return self.model_ctx.get(model, self.num_ctx)

    async def check_offload(self, model: str) -> dict | None:
        """Ile modelu jest w pamieci karty (Ollama /api/ps). Zmniejsza kontekst modelu, ktory sie nie miesci."""
        try:
            data = self._check(await self.client.get(f"{self.url}/api/ps", timeout=10))
        except Exception as exc:  # noqa: BLE001 - diagnostyka nie moze psuc odpowiedzi
            log.debug("ollama /api/ps: %s", exc)
            return None
        row = next((m for m in data.get("models", []) if (m.get("name") or m.get("model")) == model), None)
        if not row or not row.get("size"):
            return None
        pct = round(100.0 * float(row.get("size_vram") or 0) / float(row["size"]), 1)
        ctx = self.ctx_for(model)
        if pct < 90 and ctx > self.MIN_CTX:
            new = max(self.MIN_CTX, ctx // 2)
            self.model_ctx[model] = new
            log.warning("Ollama %s: w pamieci karty tylko %.0f%% modelu - liczy na CPU; kontekst %d -> %d", model,
                        pct, ctx, new)
        self.vram[model] = {"procent_vram": pct, "num_ctx": self.ctx_for(model), "sprawdzono": time.time()}
        return self.vram[model]

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
                      "options": {"num_ctx": self.ctx_for(model), "temperature": temp}}
        start = time.monotonic()
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
        if time.monotonic() - start > self.SLOW_S:
            await self.check_offload(model)
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
