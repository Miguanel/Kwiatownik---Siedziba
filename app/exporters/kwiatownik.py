import json
from datetime import datetime
from pathlib import Path

from app.config import settings


def export_items(items: list[dict], filename: str, target_dir: Path | None = None) -> Path:
    """Zapisuje liste przepisow w formacie Kwiatownika (lista obiektow JSON)."""
    target_dir = Path(target_dir or settings.kwiatownik_export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    from app.worker import snapshots          # lokalnie: unikamy importu cyklicznego
    before = snapshots.read_text(path)
    text = json.dumps(items, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    snapshots.record(path, before, text, "eksport przepisow")
    return path


def source_entry(source_name: str, url: str, title: str | None, language: str | None,
                 fetched_at: datetime | None) -> dict:
    """Jeden wpis pola "zrodla" w JSON-ie Kwiatownika."""
    return {
        "nazwa": source_name,
        "url": url,
        "tytul_oryginalny": title,
        "jezyk": language,
        "data_pobrania": fetched_at.date().isoformat() if fetched_at else None,
    }


def with_sources(record: dict, sources: list[dict]) -> dict:
    """Dopisuje do rekordu Kwiatownika liste zrodel: strony WWW (bez duplikatow URL, glowne pierwsze),
    potem zrodla, ktore rekord mial wlasne (np. bibliografia przepisow z archiwum Kwiatownika 1)."""
    seen, uniq = set(), []
    for s in sources:
        if str(s.get("url") or "").startswith("http") and s["url"] not in seen:
            seen.add(s["url"])
            uniq.append(s)
    own = record.get("zrodla") or []
    own = own if isinstance(own, list) else [own]
    return {**record, "zrodla": uniq + [z for z in own if z and z not in uniq]}
