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
CHANGELOG_KEEP = 200
CHECK_EVERY_MIN = 30                      # automatyczne sprawdzenie (bez zapisu w historii) najczesciej co tyle
_last_check = {"t": None}
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
    new_plants = photos = points = 0
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
            if old is None:
                new_plants += 1
            if added or old is None:
                plants.append({"id": Path(path).stem, "nazwa": data.get("nazwa_pl") or Path(path).stem, "nowe": added,
                               "jezyki": sorted(_langs(data) - _langs(old or {})) or sorted(_langs(data))[:4],
                               "nowa": old is None})
                points += added
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
            "przepisy": len(new_recipes), "przepisy_lista": new_recipes[:5], "pliki": len(files), "bledy": errors}


def last_publication() -> datetime | None:
    with Session(engine) as s:
        runs = s.exec(select(AgentRun).where(AgentRun.agent == "wdrozeniowiec", AgentRun.status == "done")
                      .order_by(col(AgentRun.id).desc()).limit(50)).all()
    for r in runs:
        if store.report(r).get("opublikowano"):
            return store.utc(r.finished_at or r.created_at)
    return None


def decide(summary: dict, last: datetime | None, now: datetime | None = None) -> tuple[bool, str]:
    now = now or store.now()
    if summary["bledy"]:
        return False, "uszkodzone pliki: " + "; ".join(summary["bledy"][:3])
    anything = summary["informacje"] or summary["przepisy"] or summary["zdjecia"] or summary["nowe_rosliny"]
    if not anything and not summary["pliki"]:
        return False, "brak nowych danych"
    hours = (now - last).total_seconds() / 3600 if last else None
    if hours is not None and hours < settings.deploy_min_hours:
        return False, f"ostatnia publikacja {hours:.1f} h temu (najczesciej co {settings.deploy_min_hours} h)"
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
    if summary["zdjecia"]:
        opis.append(f"Dodano {summary['zdjecia']} {odmiana(summary['zdjecia'], 'zdjęcie', 'zdjęcia', 'zdjęć')} z Wikimedia Commons.")
    return {"data": when.isoformat(timespec="seconds"), "tytul": tytul, "opis": " ".join(opis),
            "liczby": {"informacje": n_pt, "przepisy": n_rc, "rosliny": n_pl, "nowe_rosliny": summary["nowe_rosliny"],
                       "zdjecia": summary["zdjecia"]},
            "rosliny": [{"id": r["id"], "nazwa": r["nazwa"], "nowe": r["nowe"], "jezyki": r.get("jezyki") or []}
                        for r in summary["rosliny"][:12]],
            "przepisy": summary["przepisy_lista"]}


def write_changelog(repo: Path, entry: dict) -> None:
    path = repo / CHANGELOG
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = {}
    rows = data.get("wpisy") if isinstance(data, dict) and isinstance(data.get("wpisy"), list) else []
    data = {"wersja": 1, "opis": "Kronika Siedziby Kwiatownika - pisze ja agent wdrozen (Siedziba).",
            "wpisy": ([entry] + rows)[:CHANGELOG_KEEP]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
            go, reason = True, "publikacja na zadanie (przycisk)"
        report.update(zmiany=summary, pliki=[p for _, p in files], ostatnia=last, powod=reason)
        say(f"Zmienione pliki danych: {len(files)}; nowe informacje: {summary['informacje']}, przepisy: "
            f"{summary['przepisy']}, nowe rosliny: {summary['nowe_rosliny']}, zdjecia: {summary['zdjecia']}")
        for e in summary["bledy"]:
            say(f"BLAD: {e}")
        if not go or dry_run:
            say(("Proba: " if dry_run and go else "") + reason)
            store.finish_run(run_id, "done", report, summary=("Gotowe do publikacji: " if go else "") + reason)
            return run_id
        entry = changelog_entry(summary)
        await asyncio.to_thread(write_changelog, repo, entry)
        paths = sorted({p for s, p in files if s != "D"} | {CHANGELOG})
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
    if not files:
        return False
    go, _reason = decide(summarize(repo, files), last_publication())
    return go


async def maybe_start(state) -> int | None:
    """Wolane przez petle agentow: gdy sa nowe dane i warunki publikacji spelnione - uruchamia zadanie agent_deploy.
    Samo sprawdzenie (git status, liczenie zmian) nie zapisuje nic w historii, zeby jej nie zasmiecac."""
    if not settings.deploy_enabled or not should_check():
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
