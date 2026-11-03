"""Lancuch tlumaczy maszynowych: pierwszy dostepny, obslugujacy jezyk (DeepL -> Azure -> NLLB). Bez LLM.
Kazde udane tlumaczenie zapisuje liczbe wyslanych znakow w liczniku (app/mt/usage.py)."""
from pathlib import Path

from app.mt.nllb import MTUnavailable


class ChainTranslator:
    def __init__(self, engines: list, knowledge_local: bool = False, usage=None):
        self.engines = [e for e in engines if e is not None]
        self.usage = usage
        self.knowledge_local = knowledge_local       # wiedza tez przez tlumacze slabszej jakosci (NLLB)
        self.last_name: str | None = None
        self.error: str | None = None

    # zgodnosc z dawnym NLLBTranslator (logi, /llm)
    @property
    def model_dir(self) -> Path:
        for e in self.engines:
            if hasattr(e, "model_dir"):
                return e.model_dir
        return Path(self.engines[0].label if self.engines else "brak")

    @property
    def device(self) -> str:
        return next((e.device for e in self.engines if hasattr(e, "device")), "api")

    @property
    def names(self) -> list[str]:
        return [e.label for e in self.engines]

    def model_present(self) -> bool:
        return any(e.model_present() for e in self.engines)

    def available(self) -> bool:
        return any(e.available() for e in self.engines)

    def supports(self, lang: str | None) -> bool:
        return any(e.supports(lang) for e in self.engines)

    def status(self) -> dict:
        use = self.usage.summary() if self.usage else {}
        return {"engines": [dict(e.status(), name=e.label, usage=use.get(e.label, {})) for e in self.engines],
                "last": self.last_name}

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str = "pl", purpose: str = "przepis") -> list[str]:
        """purpose: "przepis" (wszystkie tlumacze) albo "wiedza" (tylko wysokiej jakosci, z budzetem limitu DeepL,
        chyba ze KNOWLEDGE_MT_LOCAL=true)."""
        if (src_lang or "")[:2].lower() == (tgt_lang or "pl")[:2].lower():
            self.last_name = "bez tlumaczenia (ten sam jezyk)"       # zrodlo po polsku: nic do wysylania
            return list(texts)
        errors = []
        for e in self.engines:
            if purpose != "przepis" and not self.knowledge_local and getattr(e, "quality", "niska") != "wysoka":
                errors.append(f"{e.label}: wiedza tylko przez tlumacza wysokiej jakosci")
                continue
            if not e.supports(src_lang) or not e.available():
                errors.append(f"{e.label}: {'jezyk nieobslugiwany' if not e.supports(src_lang) else e.error or 'niedostepny'}")
                continue
            before = getattr(e, "chars_sent", None)
            try:
                out = (e.translate(texts, src_lang, tgt_lang, purpose=purpose) if getattr(e, "takes_purpose", False)
                       else e.translate(texts, src_lang, tgt_lang))
            except MTUnavailable as exc:
                errors.append(f"{e.label}: {exc}")
                continue
            self.last_name, self.error = e.label, None
            if self.usage is not None:
                sent = (e.chars_sent - before) if isinstance(before, int) else sum(len(x) for x in texts)
                self.usage.add(e.label, purpose, sent)
            return out
        self.error = "; ".join(errors) or "brak tlumacza"
        raise MTUnavailable(self.error)
