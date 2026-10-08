from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ItemKind(str, Enum):
    recipe = "recipe"      # przepis
    fact = "fact"          # ciekawostka
    other = "other"        # strona bez wartosci (zapamietana, zeby jej nie pobierac ponownie)


class ItemStatus(str, Enum):
    raw = "raw"                # pobrany tekst strony
    extracted = "extracted"    # LLM wyciagnal strukture
    translated = "translated"  # przetlumaczony na PL
    exists = "exists"          # taki przepis juz jest w Kwiatowniku - nie eksportujemy
    approved = "approved"      # zatwierdzony recznie w UI
    exported = "exported"      # wyslany do Kwiatownika
    skipped = "skipped"        # odrzucony / nieprzydatny
    error = "error"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class Source(SQLModel, table=True):
    """Zagraniczna strona, z ktorej zbieramy tresci."""
    id: int | None = Field(default=None, primary_key=True)
    name: str
    base_url: str = Field(index=True, unique=True)
    language: str = "en"
    country: str | None = None           # kod kraju (ua, de...) - decyduje o VPN
    scraper: str = "auto"
    active: bool = True
    keywords: str | None = None          # dodatkowe slowa kluczowe, po przecinku
    max_pages: int | None = 50           # limit stron na jedno skanowanie
    max_depth: int | None = 3
    needs_js: bool | None = False        # wykryto strone renderowana w JavaScript
    last_scan_at: datetime | None = None
    pages_scanned: int | None = 0
    auto_scan: bool | None = False       # skanowanie cykliczne (np. co noc)
    scraper_profile: str | None = None   # JSON profilu scrapera zbudowanego przez LLM
    profile_status: str | None = None    # ready | failed | degraded
    created_at: datetime = Field(default_factory=_now)


class Item(SQLModel, table=True):
    """Pojedyncza strona / przepis / ciekawostka przechodzaca przez pipeline."""
    id: int | None = Field(default=None, primary_key=True)
    source_id: int | None = Field(default=None, foreign_key="source.id", index=True)
    url: str = Field(index=True, unique=True)
    kind: ItemKind = ItemKind.recipe
    status: ItemStatus = ItemStatus.raw
    title: str | None = None
    raw_text: str | None = None
    structured_json: str | None = None   # JSON-LD schema.org/Recipe, jesli strona go ma
    data_json: str | None = None         # JSON zgodny z wzorcem Kwiatownika
    confidence: float | None = None
    method: str | None = None            # jsonld | heuristic | llm:<model>
    reason: str | None = None
    language: str | None = None
    pattern: str | None = None           # wzorzec URL, np. /recipes/{slug}
    duplicate_of: int | None = Field(default=None, index=True)  # ten sam przepis z innej strony -> id glownego
    fingerprint: str | None = None       # znormalizowany tytul + skladniki (wykrywanie duplikatow)
    translated_by: str | None = None     # provider/model, ktory przetlumaczyl
    match_ref: str | None = None         # z czym sie pokrywa (plik Kwiatownika#id albo item #id)
    match_info: str | None = None        # jak ustalono (podobienstwo/llm) + wynik
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class SitePattern(SQLModel, table=True):
    """Czego crawler nauczyl sie o strukturze danego serwisu."""
    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    pattern: str
    hits: int = 0
    misses: int = 0
    discovered: int = 0


class Job(SQLModel, table=True):
    """Zadanie uruchamiane z panelu (scan / extract / translate / export)."""
    id: int | None = Field(default=None, primary_key=True)
    kind: str
    source_id: int | None = Field(default=None, foreign_key="source.id")
    status: JobStatus = JobStatus.queued
    log: str = ""
    progress: int | None = 0
    total: int | None = 0
    stats_json: str | None = None
    params_json: str | None = None       # parametry uruchomienia - do "Ponow"
    state_json: str | None = None        # stan do wznowienia (np. kolejka adresow crawlera)
    retry_of: int | None = None          # to zadanie jest ponowieniem zadania #...
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None   # kiedy faktycznie ruszylo (po czekaniu w kolejce)
    finished_at: datetime | None = None


class DiscoveredSite(SQLModel, table=True):
    """Strona znaleziona w wyszukiwarce (kazda domena tylko raz)."""
    id: int | None = Field(default=None, primary_key=True)
    domain: str = Field(index=True, unique=True)
    country: str = Field(index=True)
    url: str                              # podstrona znaleziona w wyszukiwarce
    title: str | None = None
    snippet: str | None = None
    query: str | None = None              # pierwsze zapytanie, ktore ja znalazlo
    engines: str | None = None
    status: str = "new"                   # new | verified | maybe | rejected | unreachable | added
    score: float | None = 0.0
    language: str | None = None
    reason: str | None = None
    hits: int | None = 1                  # ile razy pojawila sie w wynikach
    source_id: int | None = None          # po dodaniu jako zrodlo
    created_at: datetime = Field(default_factory=_now)


class SearchQuery(SQLModel, table=True):
    """Uzyte zapytania - zeby ich nie powtarzac i wiedziec, ktore daja nowe strony."""
    id: int | None = Field(default=None, primary_key=True)
    country: str = Field(index=True)
    text: str
    results: int = 0
    new_sites: int = 0
    created_at: datetime = Field(default_factory=_now)


class Plant(SQLModel, table=True):
    """Roslina, o ktorej Siedziba zbiera wiedze (z Kwiatownika albo nowa - np. z przepisow)."""
    id: str = Field(primary_key=True)                 # id pliku w Kwiatowniku, np. "krwawnik_pospolity"
    nazwa_pl: str
    nazwa_lat: str | None = Field(default=None, index=True)
    origin: str = "kwiatownik"                        # kwiatownik | archiwum | nowa
    mentions: int = 0                                 # ile przepisow wspomina te rosline (dla nowych)
    wikidata: str | None = None                       # np. "Q25408"
    wiki_titles_json: str | None = None               # {"pl": "Krwawnik pospolity", "en": "Achillea millefolium"}
    names_json: str | None = None                     # nazwy w jezykach z Wikidata {"zh": "蓍", "ja": "セイヨウノコギリソウ"}
    coverage: int | None = None                       # % podrozdzialow schematu z trescia (reczna albo z sieci)
    gaps_json: str | None = None                      # brakujace podrozdzialy (sciezki) - cel kolejnych skanow
    last_merge_at: datetime | None = None             # ostatnie scalenie wiedzy z rozdzialami
    photos_json: str | None = None                    # zdjecia z Wikipedii/Commons (app/knowledge/photos.py)
    photos_at: datetime | None = None
    status: str = "idle"                              # idle | researched | applied | error
    last_research_at: datetime | None = None
    last_apply_at: datetime | None = None
    note: str | None = None
    created_at: datetime = Field(default_factory=_now)


class PlantFact(SQLModel, table=True):
    """Pojedyncza informacja o roslinie ze zrodlem. Trafia do pliku rosliny dopiero po weryfikacji."""
    id: int | None = Field(default=None, primary_key=True)
    plant_id: str = Field(index=True, foreign_key="plant.id")
    section: str                                      # patrz app/knowledge/sections.py
    part: str | None = None                           # czesc rosliny (liscie, korzen...)
    text: str                                         # po polsku
    quote: str | None = None                          # fragment zrodla, na ktorym opiera sie informacja
    source_url: str
    source_name: str | None = None
    language: str | None = None
    fingerprint: str = Field(index=True)
    status: str = "new"                               # new | verified | rejected | applied
    reason: str | None = None                         # powod odrzucenia / uwagi weryfikatora
    extracted_by: str | None = None
    verified_by: str | None = None
    created_at: datetime = Field(default_factory=_now)


class PlantQuery(SQLModel, table=True):
    """Historia zapytan o brakujace informacje (zeby kolejne skany nie powtarzaly tych samych)."""
    id: int | None = Field(default=None, primary_key=True)
    plant_id: str = Field(index=True)
    slot: str                                         # podrozdzial schematu, ktorego brakowalo
    language: str
    query: str
    results: int = 0                                  # ile wynikow dala wyszukiwarka
    used_url: str | None = None                       # strona, z ktorej wyciagnieto informacje
    facts: int = 0                                    # ile nowych informacji dala
    created_at: datetime = Field(default_factory=_now)


class PlantAlias(SQLModel, table=True):
    """Nazwa skladnika z przepisu -> roslina (albo 'to nie roslina'); zeby nie pytac LLM drugi raz."""
    name: str = Field(primary_key=True)               # znormalizowana nazwa skladnika
    plant_id: str | None = None                       # None = to nie roslina (cukier, woda, wodka...)
    checked_at: datetime = Field(default_factory=_now)


class FbPost(SQLModel, table=True):
    """Post na fanpage Kwiatownika (dzial marketingu, /posts). Publikacja na razie recznie - kopiuj i wklej."""
    id: int | None = Field(default=None, primary_key=True)
    status: str = Field(default="draft", index=True)  # draft | approved | published | rejected
    kind: str | None = Field(default=None, index=True)  # siedziba | pora_roku | ciekawostka | przepis
    ref: str | None = None                            # wykorzystane dane, np. "fact:12,13" / "item:2837"
    persona: str | None = None                        # imie postaci, np. "Lesny Dziadyga"
    season: str | None = None                         # np. "poczatek pazdziernika (jesien)"
    plants_json: str | None = None                    # [{id, nazwa_pl, nazwa_lat, url}] - pierwsza = glowna
    hint: str | None = None                           # zyczenie redaktora (temat)
    body: str = ""                                    # sama tresc od modelu
    text: str = ""                                    # gotowy tekst do wklejenia (edytowalny)
    model: str | None = None
    problems: str | None = None                       # uwagi z kontroli kodem (gdy nie udalo sie poprawic)
    job_id: int | None = None
    edited: bool = False
    published_at: datetime | None = None
    source_json: str | None = None                    # dane i polecenie dla modelu (kontrola merytoryczna w ocenie)
    revision_json: str | None = None                  # poprawka wg raportu: z ktorego posta, rundy, co wdrozono
    parent_id: int | None = Field(default=None, index=True)  # post, z ktorego powstal przez poprawke wg raportu
    root_id: int | None = Field(default=None, index=True)    # pierwszy post w rodzinie wersji (historia edycji)
    # wyniki z Facebooka (wpisywane recznie po publikacji) - do statystyk "ocena vs rzeczywistosc"
    reach: int | None = None
    reactions: int | None = None
    comments: int | None = None
    shares: int | None = None
    created_at: datetime = Field(default_factory=_now)


class FbPostReview(SQLModel, table=True):
    """Ocena posta: metryki kodem + recenzja eksperta (LLM, inny model niz autor). Najnowsza = aktualna."""
    id: int | None = Field(default=None, primary_key=True)
    post_id: int = Field(index=True, foreign_key="fbpost.id")
    text_hash: str                                    # ktora wersje tekstu oceniono (po edycji ocena jest nieaktualna)
    overall: float                                    # 0-100
    code_score: float                                 # 0-100 z metryk
    expert_score: float | None = None                 # 0-100 z recenzji LLM (None = recenzja sie nie udala)
    verdict: str = "popraw"                           # publikuj | popraw | odrzuc
    metrics_json: str = "{}"                          # app/marketing/metrics.compute
    scores_json: str = "{}"                           # kryterium -> 1-10
    review_json: str = "{}"                           # podsumowanie, mocne/slabe strony, sugestie, bledy...
    model: str | None = None
    created_at: datetime = Field(default_factory=_now)


class FbPostVersion(SQLModel, table=True):
    """Historia tekstu posta (cala rodzina wersji: wygenerowany, edycje reczne, poprawki wg raportu, rundy)."""
    id: int | None = Field(default=None, primary_key=True)
    post_id: int = Field(index=True)
    root_id: int = Field(index=True)
    text: str
    kind: str = "edit"        # generated | edit | rewrite | improve | round | restore | initial
    note: str | None = None
    created_at: datetime = Field(default_factory=_now)


class FbIdea(SQLModel, table=True):
    """Kandydat na ciekawostke do postow (pula zbierana zadaniem 'fb_ideas', podsuwana modelom piszacym)."""
    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(index=True, unique=True)       # fact:<id> | item:<id>#<n> | item:<id> (artykul bez ciekawostek)
    source_type: str = "fact"                       # fact (PlantFact) | item (artykul z zagranicznej strony)
    source_id: int | None = None
    plant_id: str | None = Field(default=None, index=True)
    plant_name: str | None = None
    text: str = ""                                  # ciekawostka po polsku (podstawa posta)
    original: str | None = None                     # doslowny fragment zrodla (cytat)
    source_url: str | None = None
    source_name: str | None = None
    language: str | None = None
    category: str | None = None
    teaser: str | None = None                       # zajawka od oceniajacego modelu
    months_json: str | None = None                  # miesiace, w ktorych temat jest na czasie
    code_score: float = 0.0
    llm_score: float | None = None                  # 1-10 (srednia: ciekawosc, zaskoczenie, zrozumialosc)
    score: float = Field(default=0.0, index=True)   # 0-100 koncowa
    risk: bool = False                              # obietnice zdrowotne / ryzykowne - nie podsuwac
    rated_by: str | None = None
    extracted_by: str | None = None
    status: str = Field(default="new", index=True)  # new | rated | rejected | empty
    created_at: datetime = Field(default_factory=_now)
    rated_at: datetime | None = None


class AgentRun(SQLModel, table=True):
    """Przebieg agenta (/agents): audytor wiedzy, zleceniodawca, planista ciaglosci. Historia wszystkich przebiegow."""
    id: int | None = Field(default=None, primary_key=True)
    agent: str = Field(index=True)                  # audytor | zleceniodawca | planista
    status: str = "running"                         # running | done | failed
    trigger: str = "reczny"                         # reczny | auto (petla w tle) | planista (zlecony przez planiste)
    job_id: int | None = None                       # zadanie w kolejce (audyt, zlecanie); planista dziala bez zadania
    summary: str | None = None                      # jedno zdanie do listy historii
    score: float | None = None                      # audyt: ocena koncowa bazy wiedzy 0-100
    code_score: float | None = None                 # audyt: ocena z metryk (kod)
    expert_score: float | None = None               # audyt: ocena eksperta LLM 0-100 (None = bez LLM)
    model: str | None = None
    report_json: str | None = None                  # pelny raport / decyzje / propozycje
    log: str = ""
    parent_id: int | None = None                    # zleceniodawca: z ktorego audytu bral zalecenia
    repeats: int = 0                                # planista: ile kolejnych sprawdzen dalo ten sam wynik
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime | None = None
    finished_at: datetime | None = None


class AgentTask(SQLModel, table=True):
    """Zadanie zaproponowane przez agenta i (po zleceniu) wykonywane przez Siedzibe - z wynikiem."""
    id: int | None = Field(default=None, primary_key=True)
    run_id: int | None = Field(default=None, index=True)   # przebieg, ktory zaproponowal (audyt / planista)
    dispatch_run_id: int | None = None                     # przebieg, ktory zlecil (zleceniodawca / planista)
    agent: str = "audytor"                                 # kto zaproponowal
    kind: str = Field(index=True)                          # opowiesci | wschod | barwienie | luki | scal | ... (app/agents/tasks.py)
    title: str = ""
    reason: str | None = None                              # uzasadnienie (regula albo ekspert)
    priority: int = 3                                      # 1 = najpilniejsze ... 5
    params_json: str = "{}"                                # {rosliny, sloty, kraj, grupa, ids...}
    status: str = Field(default="proposed", index=True)    # proposed | ordered | running | done | failed | skipped | expired
    job_id: int | None = Field(default=None, index=True)
    job_kind: str | None = None
    result_json: str | None = None                         # statystyki zakonczonego zadania (Job.stats_json)
    note: str | None = None                                # np. powod pominiecia
    created_at: datetime = Field(default_factory=_now)
    ordered_at: datetime | None = None
    finished_at: datetime | None = None


class SiteStat(SQLModel, table=True):
    """Kopia licznikow strony z backendu Kwiatownika (odslony roslin, plant.id, rozdzialy, zrodla, wyszukiwania).
    Siedziba trzyma ja, bo darmowy serwer na Render gubi dysk przy restarcie - wtedy Siedziba odtwarza liczniki."""
    day: str = Field(primary_key=True)          # RRRR-MM-DD
    kind: str = Field(primary_key=True)         # plant_view, plantid_click, plant_section, ...
    key: str = Field(default="", primary_key=True)
    n: int = 0
