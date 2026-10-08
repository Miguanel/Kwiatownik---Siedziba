# ============================================================
#  Siedziba Kwiatownika - szkielet projektu (PowerShell / PyCharm)
#  Uruchamiaj w katalogu: C:\Users\micha\PycharmProjects\KwiatownikSiedziba
# ============================================================

# --- KROK 0: helper do zapisu plikow w UTF-8 bez BOM (PowerShell 5 domyslnie dodaje BOM / UTF-16) ---
function W([string]$Path, [string]$Text) {
    $full = Join-Path (Get-Location) $Path
    New-Item -ItemType Directory -Force -Path (Split-Path $full) | Out-Null
    [IO.File]::WriteAllText($full, $Text.TrimStart("`r","`n"), (New-Object Text.UTF8Encoding $false))
    Write-Host "  + $Path"
}

# --- KROK 1: katalogi ---
$dirs = @(
  "app\web\templates", "app\web\static",
  "app\scrapers\sites", "app\llm", "app\pipeline", "app\exporters", "app\worker", "app\schemas",
  "data\raw", "data\processed", "data\export",
  "tests"
)
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

# puste __init__.py w pakietach
"app","app\web","app\scrapers","app\scrapers\sites","app\llm","app\pipeline","app\exporters","app\worker","tests" |
  ForEach-Object { W "$_\__init__.py" "" }

# .gitkeep, zeby git widzial puste katalogi danych
"data\raw","data\processed","data\export","app\web\static" | ForEach-Object { W "$_\.gitkeep" "" }

# --- KROK 2: pliki konfiguracyjne ---
W "requirements.txt" @'
fastapi>=0.115
uvicorn[standard]>=0.30
jinja2>=3.1
python-multipart>=0.0.9
sqlmodel>=0.0.22
pydantic-settings>=2.4
httpx>=0.27
beautifulsoup4>=4.12
lxml>=5.2
trafilatura>=1.12
tenacity>=8.5
apscheduler>=3.10
pytest>=8.0
'@

W ".env.example" @'
APP_NAME=Siedziba Kwiatownika
# DATABASE_URL=sqlite:///data/siedziba.db
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
REQUEST_DELAY_S=2.0
# Gdzie trafiaja gotowe JSON-y dla Kwiatownika
KWIATOWNIK_EXPORT_DIR=data/export
'@

W ".gitignore" @'
venv/
.venv/
__pycache__/
*.pyc
.idea/
.env
data/*.db
data/raw/*
data/processed/*
data/export/*
!data/**/.gitkeep
'@

W ".dockerignore" @'
venv/
.venv/
.idea/
.git/
__pycache__/
data/*.db
.env
'@

# --- KROK 3: rdzen aplikacji ---
W "app\config.py" @'
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Siedziba Kwiatownika"
    data_dir: Path = BASE_DIR / "data"
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'siedziba.db').as_posix()}"
    kwiatownik_export_dir: Path = BASE_DIR / "data" / "export"

    llm_provider: str = "ollama"  # ollama | openai | anthropic
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    user_agent: str = "SiedzibaKwiatownika/0.1 (+https://kwiatownik.onrender.com)"
    request_delay_s: float = 2.0


settings = Settings()
'@

W "app\db.py" @'
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

settings.data_dir.mkdir(parents=True, exist_ok=True)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)


def init_db() -> None:
    from app import models  # noqa: F401  (rejestracja tabel)
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
'@

W "app\models.py" @'
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ItemKind(str, Enum):
    recipe = "recipe"      # przepis
    fact = "fact"          # ciekawostka


class ItemStatus(str, Enum):
    raw = "raw"                # pobrany HTML/tekst
    extracted = "extracted"    # LLM wyciagnal strukture
    translated = "translated"  # przetlumaczony na PL
    approved = "approved"      # zatwierdzony recznie w UI
    exported = "exported"      # wyslany do Kwiatownika
    error = "error"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Source(SQLModel, table=True):
    """Zagraniczna strona, z ktorej zbieramy tresci."""
    id: int | None = Field(default=None, primary_key=True)
    name: str
    base_url: str = Field(index=True, unique=True)
    language: str = "en"
    scraper: str = "generic"   # nazwa scrapera z app/scrapers/registry.py
    active: bool = True
    created_at: datetime = Field(default_factory=_now)


class Item(SQLModel, table=True):
    """Pojedynczy przepis / ciekawostka przechodzacy przez pipeline."""
    id: int | None = Field(default=None, primary_key=True)
    source_id: int | None = Field(default=None, foreign_key="source.id")
    url: str = Field(index=True, unique=True)
    kind: ItemKind = ItemKind.recipe
    status: ItemStatus = ItemStatus.raw
    title: str | None = None
    raw_text: str | None = None
    data_json: str | None = None   # JSON zgodny z wzorcem Kwiatownika
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Job(SQLModel, table=True):
    """Zadanie uruchamiane z panelu (scrape / extract / translate / export)."""
    id: int | None = Field(default=None, primary_key=True)
    kind: str
    source_id: int | None = Field(default=None, foreign_key="source.id")
    status: JobStatus = JobStatus.queued
    log: str = ""
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
'@

W "app\main.py" @'
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.web.routes import router as web_router

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
app.include_router(web_router)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}
'@

# --- KROK 4: panel w przegladarce (FastAPI + Jinja2 + HTMX) ---
W "app\web\routes.py" @'
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, func, select

from app.config import settings
from app.db import get_session
from app.models import Item, ItemStatus, Job, Source

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
router = APIRouter()


@router.get("/")
def dashboard(request: Request, session: Session = Depends(get_session)):
    counts = {
        s.value: session.exec(select(func.count()).select_from(Item).where(Item.status == s)).one()
        for s in ItemStatus
    }
    jobs = session.exec(select(Job).order_by(Job.id.desc()).limit(10)).all()
    return templates.TemplateResponse(
        request, "dashboard.html", {"app_name": settings.app_name, "counts": counts, "jobs": jobs}
    )


@router.get("/sources")
def sources_list(request: Request, session: Session = Depends(get_session)):
    sources = session.exec(select(Source).order_by(Source.id)).all()
    return templates.TemplateResponse(
        request, "sources.html", {"app_name": settings.app_name, "sources": sources}
    )


@router.post("/sources")
def sources_add(
    name: str = Form(...),
    base_url: str = Form(...),
    language: str = Form("en"),
    session: Session = Depends(get_session),
):
    session.add(Source(name=name, base_url=base_url.rstrip("/"), language=language))
    session.commit()
    return RedirectResponse("/sources", status_code=303)
'@

W "app\web\templates\base.html" @'
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ app_name }}</title>
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #f6f7f2; color: #1f2a1f; }
    nav { background: #2f5d3a; padding: 12px 20px; }
    nav a { color: #fff; margin-right: 16px; text-decoration: none; font-weight: 600; }
    main { max-width: 1000px; margin: 24px auto; padding: 0 16px; }
    table { width: 100%; border-collapse: collapse; background: #fff; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e3e6dc; text-align: left; }
    .cards { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { background: #fff; padding: 14px 18px; border-radius: 8px; min-width: 110px; }
    .card b { font-size: 1.6em; display: block; }
    form { background: #fff; padding: 14px; border-radius: 8px; display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    input, button { padding: 6px 10px; }
  </style>
</head>
<body>
  <nav>
    <a href="/">🌿 {{ app_name }}</a>
    <a href="/sources">Zrodla</a>
    <a href="/docs">API</a>
  </nav>
  <main>{% block content %}{% endblock %}</main>
</body>
</html>
'@

W "app\web\templates\dashboard.html" @'
{% extends "base.html" %}
{% block content %}
<h1>Panel</h1>
<div class="cards">
  {% for status, n in counts.items() %}
    <div class="card"><b>{{ n }}</b>{{ status }}</div>
  {% endfor %}
</div>
<h2>Ostatnie zadania</h2>
<table>
  <tr><th>#</th><th>Typ</th><th>Status</th><th>Utworzone</th></tr>
  {% for j in jobs %}
    <tr><td>{{ j.id }}</td><td>{{ j.kind }}</td><td>{{ j.status.value }}</td><td>{{ j.created_at.strftime("%Y-%m-%d %H:%M") }}</td></tr>
  {% else %}
    <tr><td colspan="4">Brak zadan.</td></tr>
  {% endfor %}
</table>
{% endblock %}
'@

W "app\web\templates\sources.html" @'
{% extends "base.html" %}
{% block content %}
<h1>Zrodla</h1>
<form method="post" action="/sources">
  <input name="name" placeholder="Nazwa" required>
  <input name="base_url" placeholder="https://..." required size="40">
  <input name="language" value="en" size="4">
  <button type="submit">Dodaj</button>
</form>
<table>
  <tr><th>#</th><th>Nazwa</th><th>URL</th><th>Jezyk</th><th>Scraper</th><th>Aktywne</th></tr>
  {% for s in sources %}
    <tr><td>{{ s.id }}</td><td>{{ s.name }}</td><td><a href="{{ s.base_url }}" target="_blank">{{ s.base_url }}</a></td>
        <td>{{ s.language }}</td><td>{{ s.scraper }}</td><td>{{ "tak" if s.active else "nie" }}</td></tr>
  {% else %}
    <tr><td colspan="6">Brak zrodel - dodaj pierwsza strone powyzej.</td></tr>
  {% endfor %}
</table>
{% endblock %}
'@

# --- KROK 5: szkielety modulow (scrapery, LLM, eksport) ---
W "app\scrapers\base.py" @'
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScrapedPage:
    url: str
    title: str | None
    text: str
    html: str | None = None


class BaseScraper(ABC):
    """Kazdy scraper umie: znalezc linki do tresci i pobrac z nich tekst."""

    name: str = "base"

    def __init__(self, base_url: str, user_agent: str, delay_s: float = 2.0):
        self.base_url = base_url
        self.user_agent = user_agent
        self.delay_s = delay_s

    @abstractmethod
    def discover_urls(self, limit: int = 20) -> list[str]: ...

    @abstractmethod
    def fetch(self, url: str) -> ScrapedPage: ...
'@

W "app\scrapers\generic.py" @'
import time
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from app.scrapers.base import BaseScraper, ScrapedPage


class GenericScraper(BaseScraper):
    """Uniwersalny scraper: linki z tej samej domeny + ekstrakcja tresci trafilatura."""

    name = "generic"
    KEYWORDS = ("recipe", "remedy", "herb", "tea", "tincture", "salve", "rezept", "receta")

    def _get(self, url: str) -> str:
        time.sleep(self.delay_s)
        r = httpx.get(url, headers={"User-Agent": self.user_agent}, timeout=30, follow_redirects=True)
        r.raise_for_status()
        return r.text

    def discover_urls(self, limit: int = 20) -> list[str]:
        soup = BeautifulSoup(self._get(self.base_url), "lxml")
        domain = urlparse(self.base_url).netloc
        found: list[str] = []
        for a in soup.select("a[href]"):
            url = urljoin(self.base_url, a["href"]).split("#")[0]
            if urlparse(url).netloc == domain and any(k in url.lower() for k in self.KEYWORDS):
                if url not in found:
                    found.append(url)
            if len(found) >= limit:
                break
        return found

    def fetch(self, url: str) -> ScrapedPage:
        html = self._get(url)
        text = trafilatura.extract(html, include_comments=False) or ""
        title = BeautifulSoup(html, "lxml").title
        return ScrapedPage(url=url, title=title.get_text(strip=True) if title else None, text=text, html=html)
'@

W "app\scrapers\registry.py" @'
from app.scrapers.base import BaseScraper
from app.scrapers.generic import GenericScraper

SCRAPERS: dict[str, type[BaseScraper]] = {
    GenericScraper.name: GenericScraper,
    # "nazwa_strony": KlasaScrapera,  <- dedykowane scrapery z app/scrapers/sites/
}


def get_scraper(name: str) -> type[BaseScraper]:
    return SCRAPERS.get(name, GenericScraper)
'@

W "app\llm\base.py" @'
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Wspolny interfejs dla dowolnego modelu (Ollama, OpenAI, Anthropic...)."""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...
'@

W "app\llm\ollama.py" @'
import httpx

from app.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        r = httpx.post(f"{self.url}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["response"]
'@

W "app\llm\factory.py" @'
from app.config import settings
from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider


def get_llm(name: str | None = None) -> LLMProvider:
    name = name or settings.llm_provider
    if name == "ollama":
        return OllamaProvider(settings.ollama_url, settings.ollama_model)

    raise ValueError(f"Nieznany provider LLM: {name}")
'@

W "app\exporters\kwiatownik.py" @'
import json
from pathlib import Path

from app.config import settings


def export_items(items: list[dict], filename: str, target_dir: Path | None = None) -> Path:
    """Zapisuje liste przepisow w formacie Kwiatownika (lista obiektow JSON)."""
    target_dir = Path(target_dir or settings.kwiatownik_export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
'@

W "tests\test_health.py" @'
from fastapi.testclient import TestClient

from app.main import app


def test_health_and_pages():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        assert client.get("/sources").status_code == 200
'@

# --- KROK 6: Docker ---
W "Dockerfile" @'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
'@

W "docker-compose.yml" @'
services:
  siedziba:
    build: .
    container_name: siedziba
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      OLLAMA_URL: http://ollama:11434
      KWIATOWNIK_EXPORT_DIR: /kwiatownik/export
    volumes:
      - ./data:/app/data
      # gotowe JSON-y trafiaja do projektu Kwiatownik2
      - ../Kwiatownik2/data/przepisy/siedziba:/kwiatownik/export
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    container_name: siedziba-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama

volumes:
  ollama_models:
'@

W "README.md" @'
# Siedziba Kwiatownika

Lokalna "fabryka" przepisow i ciekawostek dla [Kwiatownika](https://kwiatownik.onrender.com):
scrapuje zagraniczne strony, przetwarza tresci modelami LLM i eksportuje JSON-y do projektu Kwiatownik2.

Pipeline: `zrodlo -> scrape (raw) -> ekstrakcja LLM -> tlumaczenie PL -> akceptacja w panelu -> eksport JSON`

## Start lokalnie
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    uvicorn app.main:app --reload     # http://localhost:8000

## Docker
    docker compose up --build         # panel: http://localhost:8000
    docker exec -it siedziba-ollama ollama pull llama3.1
'@

Write-Host "`nGotowe. Szkielet utworzony." -ForegroundColor Green
