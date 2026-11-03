"""Drugi tlumacz w chmurze (Azure Translator), licznik znakow i budzet (etap: limit DeepL)."""
import json
from datetime import date

import httpx
import pytest

from app.mt.azure import AzureTranslator
from app.mt.chain import ChainTranslator
from app.mt.nllb import MTUnavailable
from app.mt.usage import UsageBook


def azure_client(log, status=200):
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        log.append((dict(req.url.params), dict(req.headers), body))
        if status != 200:
            return httpx.Response(status, text='{"error":{"code":403001,"message":"quota"}}')
        return httpx.Response(200, json=[{"translations": [{"text": "PL:" + x["Text"], "to": "pl"}]} for x in body])
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_azure_translates_with_dictionary_and_region(tmp_path):
    log = []
    book = UsageBook(tmp_path / "u.json")
    az = AzureTranslator("k", "westeurope", usage=book, client=azure_client(log),
                         glossary=lambda src: {"Spitzwegerich": "babka lancetowata"})
    out = az.translate(["Spitzwegerich-Sirup", "Spitzwegerich ist gut", ""], "de")
    params, headers, body = log[0]
    assert params == {"api-version": "3.0", "from": "de", "to": "pl"} and headers["ocp-apim-subscription-region"] == "westeurope"
    assert body[1]["Text"] == '<mstrans:dictionary translation="babka lancetowata">Spitzwegerich</mstrans:dictionary> ist gut'
    assert out[1] == "PL:Spitzwegerich ist gut" and out[2] == ""                  # znaczniki usuniete z wyniku
    assert az.supports("sr") and not az.supports("xx") and AzureTranslator("k").code("no") == "nb"


def test_azure_quota_pauses_and_budget_from_book(tmp_path):
    q = AzureTranslator("k", client=azure_client([], status=403))
    with pytest.raises(MTUnavailable, match="limit"):
        q.translate(["x"], "de")
    assert not q.available()
    book = UsageBook(tmp_path / "u.json")
    book.add("azure/translator", "wiedza", 1_250_000)
    az = AzureTranslator("k", usage=book, monthly_chars=2_000_000, knowledge_share=0.6, client=azure_client([]))
    assert az.translate(["abc"], "de", purpose="przepis") == ["PL:abc"]
    with pytest.raises(MTUnavailable, match="wiedze"):
        az.translate(["def"], "de", purpose="wiedza")                            # 1,25 mln > 60% z 2 mln


def test_usage_book_days_months_and_chain_records(tmp_path):
    book = UsageBook(tmp_path / "u.json")
    book.add("deepl/api-free", "przepis", 100, date(2026, 11, 3))
    book.add("deepl/api-free", "wiedza", 50, date(2026, 11, 3))
    book.add("deepl/api-free", "przepis", 7, date(2026, 11, 4))
    for m in (8, 9, 10):
        book.add("deepl/api-free", "przepis", 1, date(2026, m, 1))
    again = UsageBook(tmp_path / "u.json")                                        # z pliku
    assert again.summary(date(2026, 11, 3)) == {"deepl/api-free": {"dzis": 150, "miesiac": 157, "wiedza": 50}}
    assert again.month("deepl/api-free", "wiedza", date(2026, 11, 3)) == 50
    assert sorted(json.loads((tmp_path / "u.json").read_text()))[0] == "2026-09"   # tylko 3 miesiace

    class Eng:
        label, quality, takes_purpose, error = "deepl/x", "wysoka", True, None
        chars_sent = 0

        def supports(self, lang):
            return True

        def available(self):
            return True

        def status(self):
            return {}

        def translate(self, texts, src, tgt="pl", purpose="przepis"):
            self.chars_sent += 4                                                  # tylko to, co poszlo do API
            return texts
    b2 = UsageBook(None)
    chain = ChainTranslator([Eng()], usage=b2)
    chain.translate(["abcdefgh"], "de", purpose="wiedza")
    assert b2.month("deepl/x") == 4 and b2.month("deepl/x", "wiedza") == 4
    assert chain.status()["engines"][0]["usage"]["miesiac"] == 4
