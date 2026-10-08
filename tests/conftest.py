"""Testy NIGDY nie dotykaja prawdziwych danych Siedziby.

Bez tego test_health (TestClient uruchamia caly start aplikacji) pracowalby na data/siedziba.db
dzialajacej w Dockerze: oznaczalby jej zadania jako przerwane, ponawial je, odpytywal prawdziwe API LLM
i nadpisywal data/llm_models.json. Zmienne srodowiskowe maja pierwszenstwo przed .env.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="siedziba-test-"))
(_TMP / "przepisy").mkdir()
(_TMP / "plants").mkdir()

os.environ.update({
    "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    "DATA_DIR": str(_TMP),
    "LLM_PROVIDERS": "",                 # bez prawdziwych modeli (Gemini/Groq/Ollama)
    "GEMINI_API_KEY": "",
    "GROQ_API_KEY": "",
    "RESUME_INTERRUPTED_JOBS": "false",
    "PHASED_SCAN": "false",              # testy skanu sprawdzaja zwykly tryb; etapy maja wlasny test
    "AUTO_SCAN_TIME": "",
    "AGENTS_ENABLED": "false",           # bez petli planisty w tle
    "SEARXNG_URL": "",
    "BRAVE_API_KEY": "",
    "KWIATOWNIK_PRZEPISY_DIR": str(_TMP / "przepisy"),
    "KWIATOWNIK_PLANTS_DIR": str(_TMP / "plants"),
    "KWIATOWNIK_EXPORT_DIR": str(_TMP / "export"),
})
