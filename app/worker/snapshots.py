"""Migawki plikow JSON "przed i po" kazdej operacji zapisu (podglad w zakladce "Na zywo").

Kazdy zapis pliku JSON (plik rosliny Kwiatownika, eksport przepisow) wola record(); powstaje katalog
data/snapshots/<id>/ z meta.json, before.json (brak = plik nie istnial) i after.json.
Kopie odrzucone przez kontrole (plik bez zmian) tez sa zapisywane - z bledami kontroli w "note".
Trzymanych jest SNAPSHOTS_KEEP najnowszych migawek, starsze sa usuwane.
"""
import contextvars
import difflib
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

# Zadanie, w ktorym dziala biezacy kod (ustawiane przez JobRunner; kazde zadanie asyncio ma wlasna kopie)
current_job: contextvars.ContextVar[int | None] = contextvars.ContextVar("current_job", default=None)

ID_RE = re.compile(r"^\d{8}-\d{6}-\d{6}(-\d+)?$")
MAX_DIFF_LINES = 4000
_LOCK = threading.Lock()


def _root() -> Path:
    return Path(settings.snapshots_dir or Path(settings.data_dir) / "snapshots")


def _norm(text: str | None) -> str | None:
    return None if text is None else text.replace("\r\n", "\n")


def _diff_counts(before: str | None, after: str) -> tuple[int, int]:
    added = removed = 0
    for ln in difflib.unified_diff((before or "").splitlines(), after.splitlines(), lineterm="", n=0):
        if ln.startswith("+") and not ln.startswith("+++"):
            added += 1
        elif ln.startswith("-") and not ln.startswith("---"):
            removed += 1
    return added, removed


def record(path: Path | str, before: str | None, after: str, operation: str, *, applied: bool = True,
           note: str = "", job_id: int | None = None, file: str | None = None, link: str | None = None,
           before_label: str | None = None) -> str | None:
    """Zapisuje migawke. Nigdy nie przerywa operacji (bledy migawek sa tylko ignorowane). Zwraca id."""
    try:
        if settings.snapshots_keep <= 0:
            return None
        before, after = _norm(before), _norm(after)
        if applied and before == after:
            return None
        root = _root()
        with _LOCK:
            root.mkdir(parents=True, exist_ok=True)
            sid = base = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            n = 1
            while (root / sid).exists():
                sid, n = f"{base}-{n}", n + 1
            d = root / sid
            d.mkdir()
        if before is not None:
            (d / "before.json").write_text(before, encoding="utf-8")
        (d / "after.json").write_text(after, encoding="utf-8")
        added, removed = _diff_counts(before, after)
        meta = {"id": sid, "ts": datetime.now(timezone.utc).isoformat(), "path": str(path),
                "file": file or Path(path).name, "link": link, "before_label": before_label,
                "operation": operation, "applied": applied, "note": note[:1000],
                "job_id": job_id if job_id is not None else current_job.get(), "new_file": before is None,
                "size_before": len(before.encode("utf-8")) if before is not None else 0,
                "size_after": len(after.encode("utf-8")), "added": added, "removed": removed}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _prune(root)
        return sid
    except Exception:                                   # migawka to tylko podglad - zapis pliku jest wazniejszy
        return None


def record_dict(path: Path | str, before: dict | list | None, after: dict | list, operation: str, **kw) -> str | None:
    """Jak record(), ale dla danych (kopia odrzucona przez kontrole - nie ma tekstu pliku)."""
    dump = lambda x: json.dumps(x, ensure_ascii=False, indent=2) + "\n"  # noqa: E731
    return record(path, dump(before) if before is not None else None, dump(after), operation, **kw)


def _pretty(text: str | None) -> str | None:
    """Zapis z bazy (JSON w jednej linii) -> wciecia, zeby roznice byly czytelne linia po linii."""
    if text is None:
        return None
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2) + "\n"
    except (TypeError, ValueError):
        return text if text.endswith("\n") else text + "\n"


def record_item(item, new_json: str, operation: str, **kw) -> str | None:
    """Przepis w bazie Siedziby (Item.data_json) przed i po operacji (tlumaczenie, opracowanie...).
    Gdy przepis nie mial jeszcze tresci po polsku, "przed" = oryginal ze strony (dane strukturalne albo tekst)."""
    before, label = item.data_json, None
    if before is None and item.structured_json:
        before, label = item.structured_json, "oryginal ze strony (dane strukturalne)"
    elif before is None and item.raw_text:
        before, label = item.raw_text[:60000], "oryginal ze strony (tekst)"
    title = (item.title or "").strip()[:60]
    return record(f"baza Siedziby: przepis #{item.id}", _pretty(before), _pretty(new_json),
                  f"{operation}: #{item.id}", file=f"przepis #{item.id}" + (f" - {title}" if title else ""),
                  link=f"/items/{item.id}", before_label=label, **kw)


def read_text(path: Path | str) -> str | None:
    p = Path(path)
    try:
        return p.read_bytes().decode("utf-8-sig") if p.exists() else None
    except OSError:
        return None


def _prune(root: Path) -> None:
    dirs = sorted(x for x in root.iterdir() if x.is_dir() and ID_RE.match(x.name))
    for old in dirs[:max(0, len(dirs) - settings.snapshots_keep)]:
        shutil.rmtree(old, ignore_errors=True)


def recent(n: int = 15) -> list[dict]:
    root = _root()
    if not root.exists():
        return []
    out = []
    for d in sorted((x for x in root.iterdir() if x.is_dir() and ID_RE.match(x.name)), reverse=True)[:n]:
        try:
            out.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def for_link(link: str, n: int = 20) -> list[dict]:
    """Historia zmian jednej pozycji (przepisu, rosliny) - najnowsze pierwsze."""
    root = _root()
    if not root.exists():
        return []
    out = []
    for d in sorted((x for x in root.iterdir() if x.is_dir() and ID_RE.match(x.name)), reverse=True):
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("link") == link:
            try:
                meta["ts"] = datetime.fromisoformat(meta["ts"])
            except (KeyError, TypeError, ValueError):
                pass
            out.append(meta)
            if len(out) >= n:
                break
    return out


def get(sid: str) -> dict | None:
    """Migawka z tresciami: meta + before/after (tekst) + diff."""
    if not ID_RE.match(sid or ""):
        return None
    d = _root() / sid
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        after = (d / "after.json").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    before = (d / "before.json").read_text(encoding="utf-8") if (d / "before.json").exists() else None
    return {**meta, "before": before, "after": after}


def raw(sid: str, which: str) -> str | None:
    if not ID_RE.match(sid or "") or which not in ("before", "after"):
        return None
    p = _root() / sid / f"{which}.json"
    return p.read_text(encoding="utf-8") if p.exists() else None


def diff_lines(before: str | None, after: str, context: int = 3) -> tuple[list[dict], bool]:
    """Diff do wyswietlenia: [{"t": "add"|"del"|"hunk"|"ctx", "s": linia}], czy obciety."""
    out, cut = [], False
    for ln in difflib.unified_diff((before or "").splitlines(), after.splitlines(), "przed", "po",
                                   lineterm="", n=context):
        if ln.startswith(("---", "+++")):
            continue
        t = "hunk" if ln.startswith("@@") else "add" if ln.startswith("+") else "del" if ln.startswith("-") else "ctx"
        out.append({"t": t, "s": ln})
        if len(out) >= MAX_DIFF_LINES:
            cut = True
            break
    return out, cut
