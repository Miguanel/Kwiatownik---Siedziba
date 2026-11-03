"""Azure AI Translator (darmowy poziom F0: 2 mln znakow miesiecznie) - drugi tlumacz w chmurze, po DeepL.

Klucz: AZURE_TRANSLATOR_KEY (+ AZURE_TRANSLATOR_REGION, np. "westeurope", gdy zasob jest regionalny).
Azure nie podaje zuzycia przez API - budzet liczy app/mt/usage.py (AZURE_TRANSLATOR_MONTHLY_CHARS).
Nazwy roslin: slownik dynamiczny Azure (<mstrans:dictionary translation="...">nazwa</mstrans:dictionary>).
"""
import html
import logging
import re
import threading
import time
from collections import OrderedDict

import httpx

from app.mt.nllb import MTUnavailable

log = logging.getLogger(__name__)

AZURE_SOURCE = {"ar", "bg", "bs", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "ga", "he",
                "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "ka", "ko", "lt", "lv", "mk", "nb", "nl", "pl",
                "pt", "ro", "ru", "sk", "sl", "sq", "sr", "sv", "th", "tr", "uk", "vi", "zh"}
CODES = {"zh": "zh-Hans", "nb": "nb", "no": "nb", "sr": "sr-Cyrl"}
MAX_TEXTS, MAX_CHARS = 100, 40000          # limit API: 1000 tekstow i 50 000 znakow na zapytanie
PAUSE_S = 6 * 3600
MARKUP = re.compile(r"</?mstrans:dictionary[^>]*>")


class AzureTranslator:
    name = "azure"
    quality = "wysoka"
    takes_purpose = True

    def __init__(self, key: str, region: str = "", url: str = "", monthly_chars: int = 2_000_000,
                 knowledge_share: float = 1.0, usage=None, glossary=None, client: httpx.Client | None = None,
                 timeout: float = 30.0, cache_size: int = 5000):
        self.key, self.region = (key or "").strip(), (region or "").strip()
        self.url = (url or "https://api.cognitive.microsofttranslator.com").rstrip("/")
        self.monthly_chars, self.knowledge_share = monthly_chars, knowledge_share
        self.usage, self.glossary = usage, glossary
        self.error: str | None = None
        self.paused_until = 0.0
        self.chars_sent = 0
        self._client, self.timeout = client, timeout
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._cache_size = cache_size

    label = "azure/translator"

    @staticmethod
    def code(lang: str | None) -> str | None:
        c = (lang or "").strip().lower()[:2]
        c = "nb" if c == "no" else c
        return c if c in AZURE_SOURCE else None

    def supports(self, lang: str | None) -> bool:
        return self.code(lang) is not None

    def model_present(self) -> bool:
        return bool(self.key)

    def available(self) -> bool:
        return bool(self.key) and time.time() >= self.paused_until

    def used(self) -> int:
        return self.usage.month(self.label) if self.usage else self.chars_sent

    def budget_left(self, purpose: str, extra: int = 0) -> bool:
        if not self.monthly_chars:
            return True
        cap = self.monthly_chars * (1.0 if purpose == "przepis" else self.knowledge_share)
        return self.used() + extra <= cap

    def status(self) -> dict:
        return {"engine": "azure", "url": self.url, "present": bool(self.key), "loaded": self.available(),
                "paused_until": self.paused_until or None, "chars_sent": self.chars_sent, "error": self.error,
                "used": self.used(), "limit": self.monthly_chars, "knowledge_share": self.knowledge_share}

    # ------------------------------------------------------------ slownik nazw roslin
    def _with_dictionary(self, text: str, terms: dict[str, str]) -> str:
        if not terms:
            return text
        out = text
        for term in sorted(terms, key=len, reverse=True):
            pat = re.compile(rf"(?<![\w>]){re.escape(term)}(?!\w)")
            if pat.search(out):
                tr = html.escape(terms[term], quote=True)
                out = pat.sub(lambda m: f'<mstrans:dictionary translation="{tr}">{m.group(0)}</mstrans:dictionary>',
                              out, count=1)
        return out

    # ------------------------------------------------------------ API
    def _request(self, payload: list[dict], src: str, tgt: str) -> list[str]:
        headers = {"Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/json"}
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        params = {"api-version": "3.0", "from": CODES.get(src, src), "to": tgt}
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            for attempt in range(3):
                try:
                    r = client.post(self.url + "/translate", params=params, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    if attempt == 2:
                        raise MTUnavailable(f"Azure nieosiagalny: {type(exc).__name__}") from exc
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code in (401, 403):
                    self.paused_until = time.time() + PAUSE_S
                    self.error = ("Azure: wyczerpany limit znakow" if "403001" in r.text or "quota" in r.text.lower()
                                  else "Azure odrzucil klucz (AZURE_TRANSLATOR_KEY / REGION)")
                    raise MTUnavailable(self.error)
                if r.status_code == 429 or r.status_code >= 500:
                    if attempt == 2:
                        raise MTUnavailable(f"Azure przeciazony (HTTP {r.status_code})")
                    time.sleep(3 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    raise MTUnavailable(f"Azure: HTTP {r.status_code} {r.text[:120]}")
                self.error = None
                return [MARKUP.sub("", ((x.get("translations") or [{}])[0]).get("text", "")) for x in r.json()]
        finally:
            if self._client is None:
                client.close()
        raise MTUnavailable("Azure: brak odpowiedzi")

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str = "pl", purpose: str = "przepis") -> list[str]:
        src = self.code(src_lang)
        if not self.key:
            raise MTUnavailable("brak AZURE_TRANSLATOR_KEY")
        if not src:
            raise MTUnavailable(f"jezyk nieobslugiwany przez Azure: {src_lang}")
        tgt = (tgt_lang or "pl")[:2].lower()
        if src == tgt:
            return list(texts)
        if time.time() < self.paused_until:
            raise MTUnavailable(self.error or "Azure chwilowo wylaczony")
        todo = list(dict.fromkeys(t for t in texts if t.strip() and (src, t) not in self._cache))
        if not self.budget_left(purpose, sum(len(t) for t in todo)):
            raise MTUnavailable("budzet Azure " + ("na wiedze " if purpose != "przepis" else "") + "wyczerpany")
        terms = {}
        if self.glossary:
            try:
                terms = self.glossary(src) or {}
            except Exception:  # noqa: BLE001 - slownik nigdy nie zatrzymuje tlumaczenia
                terms = {}
        batches, cur, size = [], [], 0
        for t in todo:
            if cur and (len(cur) >= MAX_TEXTS or size + len(t) > MAX_CHARS):
                batches.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += len(t)
        if cur:
            batches.append(cur)
        with self._lock:
            for batch in batches:
                outs = self._request([{"Text": self._with_dictionary(t, terms)} for t in batch], src, tgt)
                if len(outs) != len(batch):
                    raise MTUnavailable("Azure zwrocil inna liczbe tlumaczen")
                self.chars_sent += sum(len(t) for t in batch)
                for t, o in zip(batch, outs):
                    self._cache[(src, t)] = o
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return [self._cache.get((src, t), "") if t.strip() else t for t in texts]
