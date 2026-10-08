from app.llm.base import InvalidOutputError, LLMProvider, ModelInfo

API_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    name = "groq"

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def list_models(self) -> list[ModelInfo]:
        data = self._check(await self.client.get(f"{API_URL}/models", headers=self._headers))
        return [ModelInfo(self.name, m["id"], m.get("context_window"))
                for m in data.get("data", []) if m.get("active", True)]

    async def _generate(self, model: str, prompt: str, system: str | None, json_mode: bool) -> str:
        if json_mode and "json" not in f"{system or ''} {prompt}".lower():
            system = (system + "\n" if system else "") + "Respond with valid JSON only."
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict = {"model": model, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._check(await self.client.post(f"{API_URL}/chat/completions", headers=self._headers, json=body))
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise InvalidOutputError(f"groq/{model}: nieoczekiwana odpowiedz")
