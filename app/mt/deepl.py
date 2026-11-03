"""DeepL API (darmowe konto "DeepL API Free": 500 000 znakow miesiecznie) - tlumacz maszynowy bez LLM.

Klucz: DEEPL_API_KEY w .env (klucz konta Free konczy sie na ":fx" -> api-free.deepl.com, inaczej api.deepl.com).
Limit znakow wyczerpany (HTTP 456) -> DeepL odpoczywa kilka godzin, tlumaczy zapasowy NLLB.
Pamiec podreczna fragmentow: ten sam skladnik/krok nie jest wysylany drugi raz (oszczednosc limitu).

Slownik (glossary): nazwy roslin katalogu w jezyku zrodla -> polskie nazwy Kwiatownika (z Wikidata, bez LLM),
np. de "Spitzwegerich" -> "babka lancetowata". Jeden slownik na pare jezykow, nazwany skrotem tresci
("kwiatownik-de-pl-<skrot>"); zmiana nazw w bazie = nowy slownik, stary jest usuwany. Blad slownika nie
zatrzymuje tlumaczenia - DeepL tlumaczy wtedy bez niego.
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict

import httpx

from app.mt.nllb import MTUnavailable

log = logging.getLogger(__name__)

# jezyki zrodlowe DeepL (ISO 639-1, wg dokumentacji API); pozostale -> NLLB
DEEPL_SOURCE = {"ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he", "hu", "id", "it", "ja",
                "ko", "lt", "lv", "nb", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "th", "tr", "uk", "vi", "zh"}
ALIASES = {"no": "nb"}
MAX_TEXTS, MAX_CHARS = 50, 60000          # na jedno zapytanie (limit API: 50 tekstow, 128 KiB)
QUOTA_PAUSE_S = 6 * 3600


class DeepLTranslator:
    name = "deepl"
    quality = "wysoka"
    takes_purpose = True

    def __init__(self, key: str, url: str | None = None, timeout: float = 30.0, cache_size: int = 5000,
                 client: httpx.Client | None = None, glossary=None, knowledge_share: float = 1.0):
        self.key = (key or "").strip()
        self.url = (url or ("https://api-free.deepl.com" if self.key.endswith(":fx") else "https://api.deepl.com"))
        self.url = self.url.rstrip("/")
        self.timeout = timeout
        self.error: str | None = None
        self.paused_until = 0.0
        self.chars_sent = 0
        self._client = client
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._cache_size = cache_size
        self.knowledge_share = knowledge_share    # cel "wiedza" moze zuzyc tyle limitu miesiecznego (przepisy - calosc)
        self._usage: tuple[float, int, int, int] | None = None   # (kiedy, zuzyte, limit, chars_sent wtedy)
        self.glossary = glossary                  # callable(jezyk_zrodla) -> {termin: polska nazwa} albo None
        self.glossary_error: str | None = None
        self._glossaries: dict[tuple[str, str], tuple[str, str | None, int]] = {}   # para -> (skrot, id, wpisow)
        self._pairs: set[tuple[str, str]] | None = None

    @property
    def label(self) -> str:
        return "deepl/api-free" if "api-free" in self.url else "deepl/api"

    @staticmethod
    def code(lang: str | None) -> str | None:
        c = (lang or "").strip().lower()[:2]
        c = ALIASES.get(c, c)
        return c if c in DEEPL_SOURCE else None

    def supports(self, lang: str | None) -> bool:
        return self.code(lang) is not None

    def model_present(self) -> bool:
        return bool(self.key)

    def available(self) -> bool:
        return bool(self.key) and time.time() >= self.paused_until

    def status(self) -> dict:
        return {"engine": "deepl", "url": self.url, "present": bool(self.key), "loaded": self.available(),
                "paused_until": self.paused_until or None, "chars_sent": self.chars_sent, "error": self.error,
                "used": self._usage[1] + (self.chars_sent - self._usage[3]) if self._usage else None,
                "limit": self._usage[2] if self._usage else None, "knowledge_share": self.knowledge_share,
                "glossaries": {f"{s}-{t}": n for (s, t), (_h, gid, n) in self._glossaries.items() if gid},
                "glossary_error": self.glossary_error}

    # ------------------------------------------------------------ API
    def _post(self, path: str, payload: dict | None = None, method: str = "POST") -> dict:
        headers = {"Authorization": f"DeepL-Auth-Key {self.key}", "User-Agent": "Siedziba-Kwiatownika/1.0"}
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            for attempt in range(3):
                try:
                    r = client.request(method, self.url + path, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    if attempt == 2:
                        raise MTUnavailable(f"DeepL nieosiagalny: {type(exc).__name__}") from exc
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code == 456:
                    self.paused_until = time.time() + QUOTA_PAUSE_S
                    self.error = "wyczerpany miesieczny limit znakow DeepL"
                    raise MTUnavailable(self.error)
                if r.status_code in (401, 403):
                    self.error = "DeepL odrzucil klucz (DEEPL_API_KEY)"
                    self.paused_until = time.time() + QUOTA_PAUSE_S
                    raise MTUnavailable(self.error)
                if r.status_code == 429 or r.status_code >= 500:
                    if attempt == 2:
                        raise MTUnavailable(f"DeepL przeciazony (HTTP {r.status_code})")
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    raise MTUnavailable(f"DeepL: HTTP {r.status_code} {r.text[:120]}")
                self.error = None
                return r.json() if r.content else {}
        finally:
            if self._client is None:
                client.close()
        raise MTUnavailable("DeepL: brak odpowiedzi")

    # ------------------------------------------------------------ slownik nazw roslin
    def _glossary_id(self, src: str, tgt: str) -> str | None:
        if not self.glossary:
            return None
        try:
            entries = self.glossary(src) or {}
        except Exception as exc:  # noqa: BLE001 - slownik nigdy nie zatrzymuje tlumaczenia
            self.glossary_error = f"slownik: {exc}"[:200]
            return None
        entries = {k.strip(): v.strip() for k, v in entries.items()
                   if k.strip() and v.strip() and "\t" not in k + v and "\n" not in k + v}
        if not entries:
            return None
        tsv = "\n".join(f"{k}\t{v}" for k, v in sorted(entries.items()))
        digest = hashlib.sha1(tsv.encode()).hexdigest()[:10]
        cached = self._glossaries.get((src, tgt))
        if cached and cached[0] == digest:
            return cached[1]
        try:
            if self._pairs is None:
                pairs = self._post("/v2/glossary-language-pairs", method="GET").get("supported_languages", [])
                self._pairs = {(p.get("source_lang", "").lower(), p.get("target_lang", "").lower()) for p in pairs}
            if (src, tgt) not in self._pairs:
                self._glossaries[(src, tgt)] = (digest, None, 0)
                return None
            prefix = f"kwiatownik-{src}-{tgt}-"
            name = prefix + digest
            existing = self._post("/v2/glossaries", method="GET").get("glossaries", [])
            gid = next((g["glossary_id"] for g in existing if g.get("name") == name and g.get("ready", True)), None)
            if gid is None:
                made = self._post("/v2/glossaries", {"name": name, "source_lang": src, "target_lang": tgt,
                                                      "entries": tsv, "entries_format": "tsv"})
                gid = made.get("glossary_id")
            for g in existing:                                    # stare wersje slownika tej pary
                if g.get("name", "").startswith(prefix) and g.get("glossary_id") != gid:
                    try:
                        self._post(f"/v2/glossaries/{g['glossary_id']}", method="DELETE")
                    except MTUnavailable:
                        pass
        except MTUnavailable as exc:
            self.glossary_error = f"slownik {src}->{tgt}: {exc}"[:200]
            if self.paused_until > time.time():
                raise
            return None
        self.glossary_error = None
        self._glossaries[(src, tgt)] = (digest, gid, len(entries))
        log.info("DeepL: slownik %s->%s, %d nazw roslin", src, tgt, len(entries))
        return gid

    def usage(self) -> dict:
        return self._post("/v2/usage", method="GET")

    def used(self) -> tuple[int, int]:
        """(zuzyte znaki w okresie rozliczeniowym, limit) - z /v2/usage co 15 min + to, co wyslano od tamtej pory.
        Blad -> (0, 0) = bez ograniczenia budzetem."""
        now = time.time()
        if self._usage is None or now - self._usage[0] > 900:
            try:
                u = self.usage()
                self._usage = (now, int(u.get("character_count") or 0), int(u.get("character_limit") or 0),
                               self.chars_sent)
            except MTUnavailable:
                if self._usage is None:
                    return 0, 0
        at, count, limit, sent_then = self._usage
        return count + (self.chars_sent - sent_then), limit

    def budget_left(self, purpose: str) -> bool:
        if purpose == "przepis" or self.knowledge_share >= 1:
            return True
        used, limit = self.used()
        return not limit or used < self.knowledge_share * limit

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str = "pl", purpose: str = "przepis") -> list[str]:
        src = self.code(src_lang)
        if not self.key:
            raise MTUnavailable("brak DEEPL_API_KEY")
        if not src:
            raise MTUnavailable(f"jezyk nieobslugiwany przez DeepL: {src_lang}")
        tgt = (tgt_lang or "pl")[:2].lower()
        if src == tgt:
            return list(texts)
        if time.time() < self.paused_until:
            raise MTUnavailable(self.error or "DeepL chwilowo wylaczony")
        if not self.budget_left(purpose):
            raise MTUnavailable(f"budzet DeepL na wiedze ({int(self.knowledge_share * 100)}% miesiecznego limitu) "
                                "wyczerpany - przepisy maja pierwszenstwo")
        todo = list(dict.fromkeys(t for t in texts if t.strip() and (src, t) not in self._cache))
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
            gid = self._glossary_id(src, tgt) if batches else None
            for batch in batches:
                payload = {"text": batch, "source_lang": src.upper(), "target_lang": tgt.upper(),
                           "preserve_formatting": True}
                if gid:
                    payload["glossary_id"] = gid
                data = self._post("/v2/translate", payload)
                outs = [x.get("text", "") for x in data.get("translations", [])]
                if len(outs) != len(batch):
                    raise MTUnavailable("DeepL zwrocil inna liczbe tlumaczen")
                self.chars_sent += sum(len(t) for t in batch)
                for t, o in zip(batch, outs):
                    self._cache[(src, t)] = o
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return [self._cache.get((src, t), "") if t.strip() else t for t in texts]
