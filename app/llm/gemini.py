from app.llm.base import InvalidOutputError, LLMProvider, ModelInfo

API_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    name = "gemini"

    @property
    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key}

    async def list_models(self) -> list[ModelInfo]:
        models, token = [], None
        while True:
            params = {"pageSize": 1000} | ({"pageToken": token} if token else {})
            data = self._check(await self.client.get(f"{API_URL}/models", headers=self._headers, params=params))
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    models.append(ModelInfo(self.name, m["name"].removeprefix("models/"), m.get("inputTokenLimit")))
            token = data.get("nextPageToken")
            if not token:
                return models

    async def _generate(self, model: str, prompt: str, system: str | None, json_mode: bool) -> str:
        body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        data = self._check(await self.client.post(
            f"{API_URL}/models/{model}:generateContent", headers=self._headers, json=body))
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            reason = (data.get("candidates") or [{}])[0].get("finishReason") or data.get("promptFeedback")
            raise InvalidOutputError(f"gemini/{model}: brak tresci ({reason})")
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
