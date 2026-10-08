from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Siedziba Kwiatownika"
    timezone: str = "Europe/Warsaw"
    data_dir: Path = BASE_DIR / "data"
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'siedziba.db').as_posix()}"
    kwiatownik_export_dir: Path = BASE_DIR / "data" / "export"
    # Kwiatownik2: stad czytamy istniejace przepisy i katalog roslin, tu zapisujemy eksport
    kwiatownik_przepisy_dir: Path = BASE_DIR.parent / "Kwiatownik2" / "data" / "przepisy"
    kwiatownik_plants_dir: Path = BASE_DIR.parent / "Kwiatownik2" / "data" / "plants"
    # Archiwum starego Kwiatownika (1) - przepisy do zaimportowania (tylko odczyt)
    kwiatownik1_przepisy_dir: Path = BASE_DIR.parent / "Kwiatownik" / "static" / "data" / "przepisy"
    export_filename: str = "siedziba_przepisy.json"
    export_require_approval: bool = True     # eksportuj tylko przepisy zatwierdzone w panelu
    translate_batch: int = 20                # ile przepisow tlumaczyc w jednym zadaniu
    # Tlumaczenie maszynowe bez LLM (NLLB-200 w CTranslate2) dla przepisow z danymi strukturalnymi
    mt_engine: str = "nllb"                  # "nllb" albo "llm" (wszystko przez LLM, jak dawniej)
    mt_model_dir: Path = BASE_DIR / "models" / "nllb-200-distilled-600M"
    mt_device: str = "cpu"                   # "cpu", "cuda" albo "auto"
    mt_compute_type: str = "int8"            # int8 (CPU, najszybciej) / int8_float16 / float16 (GPU)
    mt_threads: int = 4                      # watki CPU dla tlumacza
    mt_beam_size: int = 2                    # 1 = najszybciej, 4 = odrobine lepiej
    mt_fallback_llm: bool = True             # gdy tlumaczenie maszynowe zawiedzie - LLM

    # --- Profil zbierania: Kwiatownik = przepisy NIEKULINARNE ---
    skip_culinary: bool = True               # pomijaj zwykle przepisy kulinarne (ciasta, dania, zupy)
    require_key_plants: bool = True          # pomijaj przepisy, w ktorych rosliny nie sa kluczowym skladnikiem
    # Opracowanie przepisu w tradycyjnych systemach (LLM) po tlumaczeniu
    enrich_recipes: bool = True
    enrich_systems: str = "wu_xing,kampo,doktryna_sygnatur,humory,spagiryka,chronoterapia"

    # --- LLM: router sam wykrywa modele u providerow i uklada ranking ---
    llm_providers: str = "gemini,groq"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    # Modele wymuszone na poczatek listy, np. "groq:openai/gpt-oss-120b,gemini:gemini-3.5-flash"
    llm_model_priority: str = ""
    # Fragmenty nazw modeli do pominiecia, np. "preview,8b"
    llm_model_blocklist: str = ""
    llm_max_models_per_provider: int = 5  # ile najlepszych modeli od kazdego providera trafia na liste
    llm_max_attempts: int = 0             # ile prob w jednym zapytaniu (0 = wszystkie modele z listy)
    llm_max_wait_s: int = 90              # gdy wszystkie modele maja limit: ile max czekac na jego odnowienie
    llm_demote_after: int = 3             # po ilu chwilowych bledach z rzedu model spada na koniec

    # Limity (per model) i wspolbieznosc (per provider)
    gemini_rpm: int = 10
    gemini_concurrency: int = 3
    groq_rpm: int = 30
    groq_concurrency: int = 3

    # Ollama - lokalny model zapasowy (uzywany, gdy Gemini i Groq zawioda lub maja limity)
    ollama_url: str = "http://localhost:11434"
    ollama_models: str = "qwen3:8b"      # pobierane automatycznie przy starcie, jesli ich brak
    ollama_num_ctx: int = 8192           # dlugosc kontekstu (prompty tlumaczenia sa dlugie)
    ollama_concurrency: int = 1          # lokalny model liczy jedno zapytanie naraz

    # --- zadania w tle ---
    max_concurrent_jobs: int = 3          # ile skanow naraz; kolejne czekaja w kolejce
    auto_scan_time: str = "03:00"         # codzienny skan zrodel z wlaczonym "auto" (puste = wylaczone)
    profile_samples: int = 5              # ile stron z przepisami pokazac LLM przy budowie scrapera
    # Skan etapowy (strony bez profilu): 1) rozpoznanie do N przepisow i N ciekawostek, 2) profil, 3) pobieranie
    phased_scan: bool = True
    probe_recipes: int = 5
    probe_facts: int = 5
    probe_max_pages: int = 150            # limit stron rozpoznania (gdy strona ma malo przepisow)
    harvest_max_pages: int = 2000         # "Pobierz cala strone" z gotowym profilem - domyslny limit stron

    # --- Wiedza o roslinach (Wikipedia + strony z innych krajow -> blok "wiedza" w plikach roslin) ---
    knowledge_languages: str = "pl,en,de,uk,ru,fr,cs,lt,sk,hu,ro,bg,it,es"  # kolejnosc = priorytet
    knowledge_wiki_langs: int = 5         # z ilu wersji jezykowych Wikipedii czytac artykul
    knowledge_web_pages: int = 3          # ile innych stron (wyszukiwarka, rozne kraje) na rosline
    knowledge_auto_apply: bool = True     # po weryfikacji od razu zapisuj do plikow roslin Kwiatownika
    knowledge_photos: bool = True         # przy zbieraniu wiedzy pobieraj tez adresy zdjec z Wikipedii/Commons
    knowledge_max_photos: int = 5         # ile zdjec w galerii rosliny (po jednym: pokroj, kwiaty, liscie, owoce...)
    knowledge_min_facts_new: int = 5      # nowa roslina dostaje plik dopiero po tylu zweryfikowanych informacjach
    # Scalanie wiedzy z sieci z rozdzialami pliku rosliny (blok "scalone"; reczne pola zostaja nietkniete)
    knowledge_auto_merge: bool = True     # po zapisie wiedzy od razu scal ja z rozdzialami
    # Modele do scalania tekstow po polsku - probowane po kolei, potem zwykla lista modeli (Gemini/Groq/...)
    merge_models: str = ("ollama:SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M,"
                         "ollama:SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0")
    merge_pull_models: bool = True        # pobierz brakujace modele scalania do Ollamy przy starcie
    merge_timeout_s: int = 120            # limit jednego zapytania do modelu scalania (Bielik); potem chmura,
                                          # a model za wolny na tym komputerze pomijany przez 30 min
    merge_plant_budget_s: int = 600       # limit czasu modeli lokalnych na jedna rosline (reszta w chmurze)
    merge_num_ctx: int = 4096             # kontekst modeli scalania w Ollamie (prompty scalania sa krotkie);
                                          # mniej pamieci -> wieksza czesc modelu w GPU (0 = jak OLLAMA_NUM_CTX)
    # Szukanie brakujacych informacji (luki w schemacie rosliny) - zapytania w tych jezykach, w tym Chiny/Japonia
    knowledge_search_langs: str = "en,de,ru,uk,fr,zh,ja"
    knowledge_gap_slots: int = 4          # ile brakujacych podrozdzialow szukac na rosline w jednym przebiegu
    knowledge_requery_days: int = 45      # to samo zapytanie powtarzaj najwczesniej po tylu dniach
    knowledge_asia_langs: str = "zh,ja"   # Wikipedie czytane zawsze (poza KNOWLEDGE_WIKI_LANGS), jesli artykul jest
    backups_dir: Path = BASE_DIR / "data" / "backups"
    # Podglad "przed/po" kazdego zapisu pliku JSON (zakladka "Na zywo"); puste = DATA_DIR/snapshots
    snapshots_dir: Path | None = None
    snapshots_keep: int = 300             # ile ostatnich migawek trzymac (0 = wylaczone)
    resume_interrupted_jobs: bool = True  # po restarcie automatycznie ponow przerwane zadania
    max_auto_retries: int = 3             # ile razy z rzedu mozna automatycznie ponawiac to samo zadanie

    # --- szukanie nowych stron ---
    searxng_url: str = "http://searxng:8080"   # lokalny SearXNG (docker compose); puste = wylaczony
    brave_api_key: str | None = None           # opcjonalnie Brave Search API
    # VPN per kraj: "ua=http://vpn-ua:8888,de=http://vpn-de:8888" (proxy HTTP kontenerow gluetun)
    country_proxies: str = ""

    def proxy_for(self, country: str | None) -> str | None:
        if not country:
            return None
        for pair in self.country_proxies.split(","):
            code, _, url = pair.strip().partition("=")
            if code.strip().lower() == country.lower() and url.strip():
                return url.strip()
        return None

    marketing_auto_review: bool = True   # po napisaniu posta od razu ocena specjalisty (/posts)
    marketing_improve_rounds: int = 2    # poprawki wg raportu: ile rund "popraw -> ocen" najwyzej
    marketing_target_score: float = 72   # ...konczy wczesniej, gdy wersja osiagnie te ocene (= "gotowy")
    ideas_items_per_run: int = 10        # zbieranie ciekawostek: ile artykulow z zagranicznych stron przejrzec naraz
    ideas_rate_per_run: int = 48         # ...ile kandydatow ocenic modelem w jednym zadaniu
    ideas_rate_batch: int = 8            # ...kandydatow w jednym zapytaniu do modelu

    # --- Agenci (/agents): audytor bazy wiedzy Kwiatownika, zleceniodawca, planista ciaglosci ---
    agents_enabled: bool = True          # petla planisty w tle (sprawdza kolejke co AGENTS_CHECK_MINUTES)
    agents_auto_continue: bool = True    # domyslnie: gdy kolejka stoi, planista sam zleca jedno zadanie (przelacznik w panelu)
    agents_check_minutes: int = 5        # co ile minut planista sprawdza, czy Siedziba nad czyms pracuje
    agents_audit_hours: int = 24         # nowy audyt, gdy ostatni starszy niz tyle godzin
    agents_dispatch_max: int = 3         # ile zalecen audytu zleca jeden przebieg zleceniodawcy
    agents_plants_per_task: int = 4      # ile roslin w jednym zleconym zadaniu zbierania wiedzy
    agents_reorder_days: int = 7         # ta sama roslina + rodzaj zadania - najwczesniej po tylu dniach
    agents_expert_review: bool = True    # ocena audytu przez eksperta LLM (bez LLM - same metryki)
    agents_fail_pause_min: int = 30      # po 3 nieudanych zleceniach z rzedu planista czeka tyle minut
    agents_stall_minutes: int = 30       # zadanie bez znaku zycia (log, krok, zapytanie do LLM) tak dlugo = zawieszone
    agents_stall_action: str = "stop"    # stop = zatrzymaj zawieszone zadanie; report = tylko zglos w historii
    agents_max_llm_jobs: int = 2         # agenci nie zlecaja nowych zadan z LLM, gdy tyle juz dziala (modele sie dlawia)

    # --- Backend Kwiatownika na Render (liczniki strony, papirus na zywo) ---
    backend_url: str = ""                # np. https://kwiatownik-backend.onrender.com (puste = wylaczone)
    backend_token: str = ""              # SIEDZIBA_TOKEN z panelu Render
    backend_sync_minutes: int = 5        # co ile minut heartbeat + kopia licznikow (Render usypia po 15 min)

    # --- Wdrozeniowiec: publikacja nowych danych w repozytorium Kwiatownika2 (Render buduje strone sam) ---
    deploy_enabled: bool = True          # agent dziala, gdy repozytorium jest podpiete (DEPLOY_REPO_DIR) i jest token
    deploy_repo_dir: Path | None = None  # cale repozytorium Kwiatownika2 (w Dockerze /kwiatownik2)
    deploy_branch: str = "main"
    github_token: str = ""               # token GitHub z prawem zapisu do repozytorium Kwiatownika2 (contents: write)
    deploy_min_points: int = 30          # publikuj, gdy jest tyle nowych informacji o roslinach...
    deploy_min_recipes: int = 10         # ...albo tyle nowych przepisow...
    deploy_max_hours: int = 48           # ...albo cokolwiek nowego, a ostatnia publikacja byla tak dawno
    deploy_min_hours: int = 6            # najczesciej co tyle godzin (kazda publikacja = nowy build na Render)
    deploy_git_name: str = "Siedziba Kwiatownika"
    deploy_git_email: str = "siedziba-kwiatownika@users.noreply.github.com"

    user_agent: str = "SiedzibaKwiatownika/0.1 (+https://kwiatownik.onrender.com)"
    request_delay_s: float = 2.0


settings = Settings()