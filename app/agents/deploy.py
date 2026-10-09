"""Wdrozeniowiec: publikuje nowa wersje Kwiatownika, gdy Siedziba nazbiera dosc nowych danych.

Kwiatownik2 to strona statyczna na Render, ktora buduje sie sama po kazdym pushu do repozytorium. Siedziba zapisuje
wiedze o roslinach (data/plants/*.json) i przepisy (data/przepisy/*.json) prosto w kopii roboczej repozytorium.
Agent:
  1. sprawdza (git status) zmienione pliki TYLKO w data/przepisy, data/plants i data/changelog.json,
  2. liczy, co jest nowego wzgledem ostatniego commita: informacje o roslinach (punkty bloku "wiedza"), nowe rosliny,
     zdjecia, nowe przepisy (po id) - i sprawdza, czy kazdy plik JSON jest poprawny,
  3. publikuje, gdy: >= DEPLOY_MIN_POINTS informacji albo >= DEPLOY_MIN_RECIPES przepisow, albo cokolwiek nowego,
     a ostatnia publikacja byla ponad DEPLOY_MAX_HOURS temu; najczesciej co DEPLOY_MIN_HOURS (kazda publikacja = build),
  4. dopisuje wpis do data/changelog.json (kronika - papirus na stronie glownej), robi commit tylko tych plikow
     (`git commit -- <pliki>`: inne zmiany w repozytorium zostaja nietkniete) i push na DEPLOY_BRANCH.
Przy kazdej publikacji zapisuje tez data/siedziba_stan.json: nad czym Siedziba pracuje, ostatnie wpisy dziennika
i statystyki strony (z kopii licznikow backendu). Kwiatownik pokazuje go w papirusie od razu, bez pytania backendu
(szybkie ladowanie na telefonie); swiezsze dane doklada w tle papirus.js, gdy backend odpowie. Bez nowych danych
sam stan jest odswiezany najczesciej co DEPLOY_STATUS_HOURS (osobny, maly commit).
Tryb DEPLOY_MODE=reczny (domyslny): agent niczego sam nie wypycha. Co pol minuty liczy, co czeka na commit (licznik
w pasku Siedziby), a gdy progi sa spelnione albo jest nowe rozmieszczenie wiedzy w rozdzialach, pokazuje na kazdej
stronie komunikat "Commit gotowy" - commit i push robi dopiero przycisk (/commit).
Nie publikuje, gdy: repozytorium jest na innej galezi, trwa merge/rebase, git jest zajety (index.lock) albo plik
JSON jest uszkodzony. Token GitHub nie trafia do logow ani do konfiguracji repozytorium.
"""
import asyncio
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from app.agents import store
from app.config import settings
from app.db import engine
from app.models import AgentRun, Job, JobStatus

TZ = ZoneInfo(settings.timezone)
DATA_PATHS = ("data/przepisy", "data/plants", "data/changelog.json")
CHANGELOG = "data/changelog.json"
STATUS_FILE = "data/siedziba_stan.json"
CHANGELOG_KEEP = 200
CHECK_EVERY_MIN = 30                      # automatyczne sprawdzenie (bez zapisu w historii) najczesciej co tyle
PENDING_MAX_AGE_S = 60                    # licznik "czeka na commit" liczony najczesciej co tyle sekund
_last_check = {"t": None}
_pending: dict = {"t": None, "state": None, "ready_since": None}
JEZYKI = {"zh": "chińskie", "ja": "japońskie", "uk": "ukraińskie", "ru": "rosyjskie", "de": "niemieckie",
          "fr": "francuskie", "en": "angielskie", "cs": "czeskie", "it": "włoskie", "es": "hiszpańskie"}


class DeployError(Exception):
    pass


def odmiana(n: int, jeden: str, kilka: str, wiele: str) -> str:
    if n == 1:
        return jeden
    return kilka if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else wiele


# ------------------------------------------------------------------ git
def repo_dir() -> Path | None:
    """Repozytorium Kwiatownika2: DEPLOY_REPO_DIR albo folder nad data/plants (uruchomienie bez Dockera)."""
    cands = [settings.deploy_repo_dir] if settings.deploy_repo_dir else []
    cands.append(Path(settings.kwiatownik_plants_dir).parent.parent)
    for c in cands:
        if c and (Path(c) / ".git").exists():
            return Path(c)
    return None


def git(repo: Path, *args: str, check: bool = True, timeout: int = 180) -> subprocess.CompletedProcess:
    cmd = ["git", "-c", "core.autocrlf=true", "-c", "core.filemode=false", "-c", "safe.directory=*",
           "-c", "core.quotepath=false", "-C", str(repo), *args]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C.UTF-8"}
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                         env=env)
    if check and res.returncode != 0:
        raise DeployError(mask(f"git {' '.join(args[:2])}: {(res.stderr or res.stdout).strip()[:400]}"))
    return res


def mask(text: str) -> str:
    tok = settings.github_token.strip()
    text = text.replace(tok, "***") if tok else text
    return re.sub(r"(https://)[^@/\s]+@", r"\1***@", text)


def readiness() -> tuple[Path | None, str | None]:
    """(repozytorium, powod, dla ktorego nie da sie publikowac - albo None)."""
    if not settings.deploy_enabled:
        return None, "wdrozenia wylaczone (DEPLOY_ENABLED=false)"
    repo = repo_dir()
    if repo is None:
        return None, "brak repozytorium Kwiatownika2 (ustaw DEPLOY_REPO_DIR i podepnij ../Kwiatownik2 w docker-compose)"
    try:
        git(repo, "--version", timeout=20)
    except (OSError, DeployError) as exc:
        return repo, f"brak programu git ({exc})"
    gd = repo / ".git"
    if (gd / "index.lock").exists():
        return repo, "repozytorium zajete (.git/index.lock) - git dziala w innym programie"
    if any((gd / x).exists() for x in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD")):
        return repo, "w repozytorium trwa merge/rebase - dokoncz go recznie"
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != settings.deploy_branch:
        return repo, f"repozytorium jest na galezi '{branch}', a publikujemy z '{settings.deploy_branch}'"
    return repo, None


def changed_files(repo: Path) -> list[tuple[str, str]]:
    """[(status, sciezka)] zmienionych plikow danych (git status), np. ('M', 'data/plants/lipa.json')."""
    res = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *DATA_PATHS)
    out = []
    for entry in res.stdout.split("\0"):
        if len(entry) < 4:
            continue
        status, path = entry[:2].strip() or "?", entry[3:]
        if path.endswith(".json"):
            out.append((status, path))
    return out


def _head_json(repo: Path, path: str):
    res = git(repo, "show", f"HEAD:{path}", check=False)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout.lstrip("﻿"))
    except ValueError:
        return None


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _points(data) -> int:
    w = data.get("wiedza") if isinstance(data, dict) and isinstance(data.get("wiedza"), dict) else {}
    pts = sum(len(s.get("punkty") or []) for s in w.get("sekcje") or [] if isinstance(s, dict))
    return pts or len(w.get("fakty") or [])


def _langs(data) -> set[str]:
    w = data.get("wiedza") if isinstance(data, dict) and isinstance(data.get("wiedza"), dict) else {}
    return {str(z.get("jezyk") or "")[:2].lower() for z in w.get("zrodla") or [] if isinstance(z, dict)} - {"", "pl"}


def _placed(data) -> set[str]:
    """Informacje rozmieszczone w rozdzialach (blok "rozmieszczenie")."""
    blk = data.get("rozmieszczenie") if isinstance(data, dict) and isinstance(data.get("rozmieszczenie"), dict) else {}
    return {f"{w.get('miejsce')}|{f}" for w in blk.get("wstawki") or [] if isinstance(w, dict)
            for p in w.get("punkty") or [] if isinstance(p, dict) for f in p.get("fakty") or []}


def _merged_slots(data) -> dict:
    sc = data.get("scalone") if isinstance(data, dict) and isinstance(data.get("scalone"), dict) else {}
    return {k: json.dumps(v.get("tresc"), ensure_ascii=False, sort_keys=True)
            for k, v in (sc.get("pola") or {}).items() if isinstance(v, dict)}


def _photos(data) -> int:
    zw = data.get("zdjecia_wiki") if isinstance(data, dict) and isinstance(data.get("zdjecia_wiki"), dict) else {}
    return len(zw.get("zdjecia") or [])


def _recipe_rows(data) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("przepisy") if isinstance(data.get("przepisy"), list) else [data]
    return [r for r in data or [] if isinstance(r, dict)]


def _recipe_key(r: dict) -> str:
    return str(r.get("id") or "") or re.sub(r"\s+", " ", str(r.get("tytul") or "")).strip().lower()


def _domain(r: dict) -> str:
    for z in r.get("zrodla") or []:
        url = z.get("url") if isinstance(z, dict) else z
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            d = url.split("//", 1)[1].split("/", 1)[0]
            return d[4:] if d.startswith("www.") else d
    return ""


def summarize(repo: Path, files: list[tuple[str, str]]) -> dict:
    """Co nowego wzgledem HEAD + bledy plikow (uszkodzony JSON blokuje publikacje)."""
    errors: list[str] = []
    plants: list[dict] = []
    new_plants = photos = points = placed = slots = 0
    recipe_files_changed = False
    for status, path in files:
        full = repo / path
        if status == "D" or not full.exists():
            continue
        try:
            data = _load(full)
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: niepoprawny JSON ({str(exc)[:80]})")
            continue
        if path.startswith("data/plants/") and path.count("/") == 2:
            if not isinstance(data, dict) or not (data.get("nazwa_pl") or data.get("gatunek")):
                errors.append(f"{path}: plik rosliny bez nazwy_pl")
                continue
            old = _head_json(repo, path)
            added = max(0, _points(data) - _points(old or {}))
            photos += max(0, _photos(data) - _photos(old or {}))
            moved = len({k.split("|", 1)[1] for k in _placed(data) - _placed(old or {})})
            old_slots, new_slots = _merged_slots(old or {}), _merged_slots(data)
            changed_slots = sum(1 for k, v in new_slots.items() if old_slots.get(k) != v)
            if old is None:
                new_plants += 1
            if added or old is None or moved or changed_slots:
                plants.append({"id": Path(path).stem, "nazwa": data.get("nazwa_pl") or Path(path).stem, "nowe": added,
                               "jezyki": sorted(_langs(data) - _langs(old or {})) or sorted(_langs(data))[:4],
                               "nowa": old is None, "rozmieszczone": moved, "rozdzialy": changed_slots})
                points += added
                placed += moved
                slots += changed_slots
        elif path.startswith("data/przepisy/"):
            if not isinstance(data, (list, dict)):
                errors.append(f"{path}: plik przepisow nie jest lista")
                continue
            recipe_files_changed = True
    new_recipes: list[dict] = []
    if recipe_files_changed:
        head_keys: set[str] = set()
        listing = git(repo, "ls-tree", "--name-only", "HEAD", "data/przepisy/", check=False).stdout.split("\n")
        for p in listing:
            if p.endswith(".json"):
                head_keys |= {_recipe_key(r) for r in _recipe_rows(_head_json(repo, p))}
        seen: set[str] = set()
        for f in sorted((repo / "data" / "przepisy").glob("*.json")):
            try:
                rows = _recipe_rows(_load(f))
            except (OSError, ValueError):
                continue
            for r in rows:
                k = _recipe_key(r)
                if k and k not in head_keys and k not in seen:
                    seen.add(k)
                    new_recipes.append({"tytul": str(r.get("tytul") or k)[:120], "zrodlo": _domain(r)})
    plants.sort(key=lambda x: -x["nowe"])
    return {"informacje": points, "rosliny": plants, "nowe_rosliny": new_plants, "zdjecia": photos,
            "rozmieszczone": placed, "rozdzialy": slots,
            "przepisy": len(new_recipes), "przepisy_lista": new_recipes[:5], "pliki": len(files), "bledy": errors}


def last_publication() -> datetime | None:
    with Session(engine) as s:
        runs = s.exec(select(AgentRun).where(AgentRun.agent == "wdrozeniowiec", AgentRun.status == "done")
                      .order_by(col(AgentRun.id).desc()).limit(50)).all()
    for r in runs:
        if store.report(r).get("opublikowano"):
            return store.utc(r.finished_at or r.created_at)
    return None


def decide(summary: dict, last: datetime | None, now: datetime | None = None,
           manual: bool = False) -> tuple[bool, str]:
    """(publikowac?, powod). manual=True (tryb reczny): "gotowe do commita" bez limitu DEPLOY_MIN_HOURS -
    o czestotliwosci decyduje czlowiek przyciskiem."""
    now = now or store.now()
    if summary["bledy"]:
        return False, "uszkodzone pliki: " + "; ".join(summary["bledy"][:3])
    layout = summary.get("rozmieszczone", 0) or summary.get("rozdzialy", 0)
    anything = summary["informacje"] or summary["przepisy"] or summary["zdjecia"] or summary["nowe_rosliny"] or layout
    if not anything and not summary["pliki"]:
        return False, "brak nowych danych"
    hours = (now - last).total_seconds() / 3600 if last else None
    if not manual and hours is not None and hours < settings.deploy_min_hours:
        return False, f"ostatnia publikacja {hours:.1f} h temu (najczesciej co {settings.deploy_min_hours} h)"
    if summary.get("rozmieszczone"):
        return True, f"nowe rozmieszczenie wiedzy w rozdzialach ({summary['rozmieszczone']} informacji)"
    progress = (f"{summary['informacje']}/{settings.deploy_min_points} informacji, "
                f"{summary['przepisy']}/{settings.deploy_min_recipes} przepisow")
    if summary["informacje"] >= settings.deploy_min_points or summary["przepisy"] >= settings.deploy_min_recipes:
        return True, f"dosc nowych danych ({progress})"
    if summary["nowe_rosliny"]:
        return True, f"nowe rosliny w Kwiatowniku ({summary['nowe_rosliny']})"
    if hours is None or hours >= settings.deploy_max_hours:
        if anything or summary["pliki"]:
            return True, ("pierwsza publikacja agenta" if hours is None else
                          f"ostatnia publikacja {hours:.0f} h temu - publikuje to, co jest ({progress})")
    return False, f"czekam: {progress}"


# ------------------------------------------------------------------ kronika
def changelog_entry(summary: dict, when: datetime | None = None) -> dict:
    when = (when or store.now()).astimezone(TZ)
    n_pl, n_pt, n_rc = len(summary["rosliny"]), summary["informacje"], summary["przepisy"]
    bits = []
    if n_pl and n_pt:
        bits.append(f"nowa wiedza o {n_pl} {odmiana(n_pl, 'roślinie', 'roślinach', 'roślinach')}")
    if n_rc:
        bits.append(f"{n_rc} {odmiana(n_rc, 'nowy przepis', 'nowe przepisy', 'nowych przepisów')}")
    n_mv = summary.get("rozmieszczone", 0)
    if n_mv and not bits:
        bits.append("wiedza z sieci rozłożona po rozdziałach")
    if summary["zdjecia"] and not bits:
        bits.append("nowe zdjęcia roślin")
    tytul = (" i ".join(bits) or "Porządki w danych Kwiatownika")
    tytul = tytul[:1].upper() + tytul[1:]
    langs = sorted({j for r in summary["rosliny"] for j in r.get("jezyki") or []})
    opis = []
    if n_pt:
        opis.append(f"Siedziba zebrała {n_pt} {odmiana(n_pt, 'sprawdzoną informację', 'sprawdzone informacje', 'sprawdzonych informacji')}"
                    + (f" (źródła: {', '.join(JEZYKI.get(j, j) for j in langs[:5])})" if langs else "") + ".")
    if summary["nowe_rosliny"]:
        opis.append(f"Nowe rośliny w zielniku: {summary['nowe_rosliny']}.")
    if n_mv:
        opis.append(f"{n_mv} {odmiana(n_mv, 'informacja z sieci trafiła', 'informacje z sieci trafiły', 'informacji z sieci trafiło')}"
                    " do właściwych rozdziałów i podrozdziałów stron roślin.")
    if summary["zdjecia"]:
        opis.append(f"Dodano {summary['zdjecia']} {odmiana(summary['zdjecia'], 'zdjęcie', 'zdjęcia', 'zdjęć')} z Wikimedia Commons.")
    return {"data": when.isoformat(timespec="seconds"), "tytul": tytul, "opis": " ".join(opis),
            "liczby": {"informacje": n_pt, "przepisy": n_rc, "rosliny": n_pl, "nowe_rosliny": summary["nowe_rosliny"],
                       "zdjecia": summary["zdjecia"], "rozmieszczone": n_mv},
            "rosliny": [{"id": r["id"], "nazwa": r["nazwa"], "nowe": r["nowe"], "jezyki": r.get("jezyki") or []}
                        for r in summary["rosliny"][:12]],
            "przepisy": summary["przepisy_lista"]}


def _day(iso: str) -> str | None:
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(TZ).date().isoformat()
    except (TypeError, ValueError):
        return None


def summary_from_entries(entries: list[dict]) -> dict:
    """Wpisy kroniki z jednego dnia -> jedno podsumowanie (format jak dla changelog_entry).
    Ta sama roslina w kilku publikacjach: informacje sie sumuja, jezyki zrodel lacza."""
    plants: dict[str, dict] = {}
    recipes: list[dict] = []
    total = {"informacje": 0, "przepisy": 0, "nowe_rosliny": 0, "zdjecia": 0, "rozmieszczone": 0}
    for e in entries:
        liczby = e.get("liczby") if isinstance(e.get("liczby"), dict) else {}
        for k in total:
            try:
                total[k] += int(liczby.get(k) or 0)
            except (TypeError, ValueError):
                pass
        for r in e.get("rosliny") or []:
            if not isinstance(r, dict) or not r.get("id"):
                continue
            p = plants.setdefault(r["id"], {"id": r["id"], "nazwa": r.get("nazwa") or r["id"], "nowe": 0, "jezyki": []})
            p["nowe"] += int(r.get("nowe") or 0)
            p["jezyki"] = sorted(set(p["jezyki"]) | set(r.get("jezyki") or []))
        for rc in e.get("przepisy") or []:
            if isinstance(rc, dict) and rc.get("tytul") and all(x.get("tytul") != rc["tytul"] for x in recipes):
                recipes.append(rc)
    return {"rosliny": sorted(plants.values(), key=lambda r: -r["nowe"]), "informacje": total["informacje"],
            "przepisy": total["przepisy"], "przepisy_lista": recipes, "nowe_rosliny": total["nowe_rosliny"],
            "zdjecia": total["zdjecia"], "rozmieszczone": total["rozmieszczone"]}


def merge_day(entry: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    """Kilka publikacji tego samego dnia = jeden wpis kroniki (zamiast "Nowa wiedza o 3 roslinach" i "o 2
    roslinach" jeden pod drugim). Nowy wpis laczy sie z najnowszym, jesli jest z tego samego dnia (czas lokalny)."""
    day = _day(entry.get("data"))
    same = [r for r in rows if isinstance(r, dict) and day and _day(r.get("data")) == day]
    if not same:
        return entry, rows
    parts = [entry] + same
    merged = changelog_entry(summary_from_entries(parts), datetime.fromisoformat(entry["data"]))
    count = sum(int(p.get("aktualizacje") or 1) for p in parts)
    times = sorted({t for p in parts for t in (p.get("godziny") or [str(p.get("data"))[11:16]]) if t})
    merged.update(aktualizacje=count, godziny=times)
    return merged, [r for r in rows if r not in same]


def write_changelog(repo: Path, entry: dict) -> None:
    path = repo / CHANGELOG
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = {}
    rows = data.get("wpisy") if isinstance(data, dict) and isinstance(data.get("wpisy"), list) else []
    entry, rows = merge_day(entry, rows)                  # ten sam dzien -> jeden wpis kroniki
    data = {"wersja": 1, "opis": "Kronika Siedziby Kwiatownika - pisze ja agent wdrozen (Siedziba).",
            "wpisy": ([entry] + rows)[:CHANGELOG_KEEP]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ stan Siedziby dla strony (papirus)
def status_snapshot(when: datetime | None = None) -> dict:
    """Stan Siedziby + ostatnie wpisy + statystyki strony (format czyta Kwiatownik2: app/utils/kronika.py)."""
    from app.agents import backend_sync
    when = (when or store.now()).astimezone(TZ)
    st = backend_sync.status_payload()
    return {"wersja": 1,
            "opis": "Stan Siedziby Kwiatownika i statystyki strony - zapisuje agent wdrozen (Siedziba). "
                    "Strona pokazuje go w papirusie bez pytania backendu.",
            "zaktualizowano": when.isoformat(timespec="seconds"),
            "siedziba": {"status": st.get("status"), "zadania": st.get("zadania") or [], "info": st.get("info") or {}},
            "wpisy": backend_sync.recent_events(),
            "statystyki": backend_sync.site_summary()}


def write_status(repo: Path, when: datetime | None = None) -> dict:
    data = status_snapshot(when)
    path = repo / STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return data


def status_age_hours(repo: Path, now: datetime | None = None) -> float | None:
    """Ile godzin temu opublikowano stan (plik w HEAD); None = jeszcze nigdy."""
    data = _head_json(repo, STATUS_FILE)
    try:
        when = datetime.fromisoformat(str((data or {}).get("zaktualizowano")))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return ((now or store.now()) - when).total_seconds() / 3600


def status_due(repo: Path, now: datetime | None = None) -> bool:
    """Czas odswiezyc sam stan na stronie (bez nowych danych)?"""
    if settings.deploy_status_hours <= 0:
        return False
    age = status_age_hours(repo, now)
    return age is None or age >= settings.deploy_status_hours


def push_url(repo: Path) -> str:
    remote = git(repo, "remote", "get-url", "origin").stdout.strip()
    token = settings.github_token.strip()
    m = re.match(r"^(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:)(.+?)(?:\.git)?/?$", remote)
    if not m:
        raise DeployError(f"nieobslugiwany adres repozytorium: {mask(remote)}")
    if not token:
        raise DeployError("brak GITHUB_TOKEN w .env (token z prawem zapisu do repozytorium Kwiatownika2)")
    return f"https://x-access-token:{token}@github.com/{m.group(1)}.git"


def commit_and_push(repo: Path, files: list[str], message: str) -> tuple[str, bool, str]:
    """Commit TYLKO podanych plikow + push. Zwraca (sha, czy_wypchnieto, komunikat)."""
    ident = ["-c", f"user.name={settings.deploy_git_name}", "-c", f"user.email={settings.deploy_git_email}"]
    git(repo, "add", "--", *files)
    res = git(repo, *ident, "commit", "-m", message, "--only", "--", *files, check=False)
    if res.returncode != 0:
        raise DeployError(mask("git commit: " + (res.stderr or res.stdout).strip()[:400]))
    sha = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    url = push_url(repo)
    pushed = git(repo, "push", url, f"HEAD:{settings.deploy_branch}", check=False, timeout=240)
    if pushed.returncode != 0:
        return sha, False, mask((pushed.stderr or pushed.stdout).strip()[:400])
    git(repo, "fetch", "--quiet", url, f"{settings.deploy_branch}:refs/remotes/origin/{settings.deploy_branch}",
        check=False, timeout=120)                    # origin/main w kopii uzytkownika = to, co na GitHubie
    return sha, True, "ok"


# ------------------------------------------------------------------ przebieg
async def run_deploy(trigger: str = "reczny", job_id: int | None = None, logf=None, force: bool = False,
                     dry_run: bool = False) -> int:
    run_id = store.start_run("wdrozeniowiec", trigger, job_id)

    def say(msg: str) -> None:
        store.run_log(run_id, mask(msg))
        if logf:
            logf(mask(msg))

    report: dict = {"opublikowano": False, "proba": dry_run}
    try:
        repo, why_not = await asyncio.to_thread(readiness)
        report["repozytorium"] = str(repo) if repo else None
        if why_not:
            say(f"Nie moge publikowac: {why_not}")
            store.finish_run(run_id, "done", {**report, "powod": why_not}, summary=f"Wstrzymane: {why_not}")
            return run_id
        files = await asyncio.to_thread(changed_files, repo)
        summary = await asyncio.to_thread(summarize, repo, files)
        last = last_publication()
        go, reason = decide(summary, last)
        if force and not summary["bledy"] and (files or summary["informacje"]):
            go, reason = True, "commit zatwierdzony recznie (przycisk)" if trigger == "przycisk" else "publikacja na zadanie (przycisk)"
        report.update(zmiany=summary, pliki=[p for _, p in files], ostatnia=last, powod=reason)
        say(f"Zmienione pliki danych: {len(files)}; nowe informacje: {summary['informacje']}, przepisy: "
            f"{summary['przepisy']}, nowe rosliny: {summary['nowe_rosliny']}, zdjecia: {summary['zdjecia']}, "
            f"rozmieszczone w rozdzialach: {summary['rozmieszczone']}, scalone podrozdzialy: {summary['rozdzialy']}")
        for e in summary["bledy"]:
            say(f"BLAD: {e}")
        if not go and not summary["bledy"] and not manual_mode() and await asyncio.to_thread(status_due, repo):
            # nic do publikacji, ale stan Siedziby na stronie jest stary - maly commit samego stanu
            if dry_run:
                say(f"Proba: {reason}; stan Siedziby na stronie do odswiezenia")
                store.finish_run(run_id, "done", {**report, "stan_do_odswiezenia": True},
                                 summary=f"{reason}; stan Siedziby do odswiezenia")
                return run_id
            await asyncio.to_thread(write_status, repo)
            say(f"{reason} - odswiezam tylko stan Siedziby i statystyki na stronie")
            sha, pushed, msg = await asyncio.to_thread(
                commit_and_push, repo, [STATUS_FILE],
                "Siedziba: stan pracy i statystyki strony\n\nAgent wdrozen Siedziby Kwiatownika (papirus).")
            report.update(commit=sha, stan_opublikowano=pushed, push=msg)
            if pushed:
                say(f"Wypchnieto commit {sha} (stan Siedziby)")
                store.finish_run(run_id, "done", report, summary=f"Odswiezono stan Siedziby na stronie ({sha})")
            else:
                say(f"Commit {sha} zapisany lokalnie, ale push sie nie udal: {msg}")
                store.finish_run(run_id, "failed", report, summary=f"Push stanu nieudany (commit {sha} czeka lokalnie): {msg[:120]}")
            return run_id
        if not go or dry_run:
            say(("Proba: " if dry_run and go else "") + reason)
            store.finish_run(run_id, "done", report, summary=("Gotowe do publikacji: " if go else "") + reason)
            return run_id
        entry = changelog_entry(summary)
        await asyncio.to_thread(write_changelog, repo, entry)
        try:
            await asyncio.to_thread(write_status, repo)
            status_paths = {STATUS_FILE}
        except Exception as exc:  # noqa: BLE001 - stan to dodatek, publikacja danych idzie dalej
            say(f"Nie udalo sie zapisac stanu Siedziby: {type(exc).__name__}: {exc}")
            status_paths = set()
        paths = sorted({p for s, p in files if s != "D"} | {CHANGELOG} | status_paths)
        message = f"Siedziba: {entry['tytul']}\n\n{entry['opis']}\n\nAgent wdrozen Siedziby Kwiatownika ({reason})."
        say(f"Publikuje: {entry['tytul']} ({len(paths)} plikow)")
        sha, pushed, msg = await asyncio.to_thread(commit_and_push, repo, paths, message)
        report.update(commit=sha, wpis=entry, opublikowano=pushed, push=msg)
        if pushed:
            say(f"Wypchnieto commit {sha} na {settings.deploy_branch} - Render zbuduje nowa wersje strony")
            store.finish_run(run_id, "done", report, summary=f"Opublikowano {sha}: {entry['tytul']}")
            from app.agents import backend_sync
            await backend_sync.push_event(f"Opublikowano nową wersję Kwiatownika: {entry['tytul']}", "wdrozenie",
                                          eid=f"run:{run_id}")
        else:
            say(f"Commit {sha} zapisany lokalnie, ale push sie nie udal: {msg}")
            store.finish_run(run_id, "failed", report, summary=f"Push nieudany (commit {sha} czeka lokalnie): {msg[:120]}")
    except Exception as exc:  # noqa: BLE001 - kazdy blad trafia do historii agenta
        say(f"BLAD: {type(exc).__name__}: {mask(str(exc))}")
        store.finish_run(run_id, "failed", report, summary=f"Publikacja nieudana: {mask(str(exc))[:200]}")
        if not isinstance(exc, DeployError):
            raise
    finally:
        invalidate_pending()
    return run_id


def should_check(now: datetime | None = None) -> bool:
    now = now or store.now()
    t = _last_check["t"]
    return t is None or now - t >= timedelta(minutes=CHECK_EVERY_MIN)


def _check() -> bool:
    repo, why_not = readiness()
    if why_not:
        return False
    files = changed_files(repo)
    if files:
        summary = summarize(repo, files)
        go, _reason = decide(summary, last_publication())
        if go:
            return True
        if summary["bledy"]:
            return False
    return status_due(repo)


# ------------------------------------------------------------------ tryb reczny: licznik i "Commit gotowy"
def manual_mode() -> bool:
    return str(settings.deploy_mode or "").strip().lower() not in ("auto", "automatyczny")


def invalidate_pending() -> None:
    _pending["t"] = None


def _deploy_running() -> bool:
    with Session(engine) as s:
        return bool(s.exec(select(Job).where(Job.kind == "agent_deploy",
                                             col(Job.status).in_([JobStatus.queued, JobStatus.running]))).first())


def compute_pending() -> dict:
    """Co czeka na commit (licznik w pasku Siedziby) i czy commit jest gotowy. Bez zapisu w historii agenta."""
    out = {"tryb": "reczny" if manual_mode() else "auto", "gotowy": False, "w_toku": False, "blokada": None,
           "powod": "", "pliki": [], "zmiany": None, "licznik": 0, "sprawdzono": store.now()}
    repo, why_not = readiness()
    if why_not:
        out["blokada"] = why_not
        return out
    files = changed_files(repo)
    summary = summarize(repo, files)
    go, reason = decide(summary, last_publication(), manual=manual_mode())
    out.update(pliki=[p for _, p in files], zmiany=summary, powod=reason, gotowy=go,
               licznik=summary["informacje"] + summary["przepisy"] + summary["zdjecia"] + summary["rozmieszczone"]
               + summary["nowe_rosliny"] + summary["rozdzialy"],
               progi={"informacje": settings.deploy_min_points, "przepisy": settings.deploy_min_recipes},
               commit=changelog_entry(summary) if files else None)
    return out


def pending(refresh: bool = False) -> dict:
    """compute_pending z pamieci (najwyzej PENDING_MAX_AGE_S sekund) + "gotowy od" + czy commit wlasnie trwa."""
    now = store.now()
    t = _pending["t"]
    if refresh or t is None or (now - t).total_seconds() >= PENDING_MAX_AGE_S or _pending["state"] is None:
        try:
            st = compute_pending()
        except Exception as exc:  # noqa: BLE001 - licznik nie moze wywrocic stron Siedziby
            st = {"tryb": "reczny" if manual_mode() else "auto", "gotowy": False, "blokada": f"blad: {mask(str(exc))[:160]}",
                  "pliki": [], "zmiany": None, "licznik": 0, "powod": "", "sprawdzono": now}
        if st["gotowy"] and not _pending["ready_since"]:
            _pending["ready_since"] = now
        elif not st["gotowy"]:
            _pending["ready_since"] = None
        _pending.update(t=now, state=st)
    st = dict(_pending["state"])
    st["gotowy_od"] = _pending["ready_since"]
    try:
        st["w_toku"] = _deploy_running()
    except Exception:  # noqa: BLE001
        st["w_toku"] = False
    return st


async def maybe_start(state) -> int | None:
    """Wolane przez petle agentow: gdy sa nowe dane i warunki publikacji spelnione - uruchamia zadanie agent_deploy.
    Samo sprawdzenie (git status, liczenie zmian) nie zapisuje nic w historii, zeby jej nie zasmiecac.
    W trybie recznym tylko odswieza licznik i komunikat "Commit gotowy" - commit robi przycisk."""
    if not settings.deploy_enabled:
        return None
    if manual_mode():
        await asyncio.to_thread(pending, True)
        return None
    if not should_check():
        return None
    _last_check["t"] = store.now()
    with Session(engine) as s:
        if s.exec(select(Job).where(Job.kind == "agent_deploy",
                                    col(Job.status).in_([JobStatus.queued, JobStatus.running]))).first():
            return None
    if not await asyncio.to_thread(_check):
        return None
    from app.worker import actions
    return actions.start_agent_deploy(state, trigger="auto")
