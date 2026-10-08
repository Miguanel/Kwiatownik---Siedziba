"""Odczyt danych Kwiatownika2: istniejace przepisy (do sprawdzania duplikatow) i katalog roslin (link_id)."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.exporters import store

log = logging.getLogger(__name__)


@dataclass
class KwRecipe:
    ref: str                 # "plik.json#id" albo "plik.json#tytul"
    title: str
    ingredients: list[str]
    typ: str | None = None
    rid: str | None = None


def _names(skladniki) -> list[str]:
    out = []
    for s in skladniki or []:
        if isinstance(s, dict):
            out.append(str(s.get("nazwa") or ""))
        elif isinstance(s, str):
            out.append(s)
    return [x for x in out if x]


def load_kwiatownik_recipes(przepisy_dir: Path, exclude: set[str] | None = None) -> list[KwRecipe]:
    """Przepisy z magazynu data/przepisy (tak jak czyta je Kwiatownik: najnowszy plik z kazdej serii),
    bez serii z `exclude` (nazwy plikow albo serii, np. {"siedziba_przepisy.json"} - nasz wlasny eksport)."""
    out: list[KwRecipe] = []
    d = Path(przepisy_dir)
    if not d.exists():
        log.warning("Brak katalogu przepisow Kwiatownika: %s", d)
        return out
    skip = {store.series_of(x) for x in (exclude or set())}
    for f in store.current_files(d):
        if store.parse_name(f.name)[0] in skip:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        items = data.get("przepisy", [data]) if isinstance(data, dict) else data
        for r in items:
            if isinstance(r, dict) and r.get("tytul"):
                rid = r.get("id") or r.get("slug")
                out.append(KwRecipe(f"{f.name}#{rid or r['tytul']}", r["tytul"], _names(r.get("skladniki")),
                                    r.get("typ"), rid))
    return out


def load_plants(plants_dir: Path) -> dict[str, str]:
    """{id_rosliny: 'Nazwa polska (Nazwa lacinska)'} z data/plants/*.json (tez podkatalogi)."""
    out: dict[str, str] = {}
    d = Path(plants_dir)
    if not d.exists():
        return out
    for f in sorted(d.rglob("*.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(p, dict):
            pid = p.get("id") or f.stem
            name = p.get("nazwa_pl") or pid.replace("_", " ")
            out.setdefault(pid, f"{name} ({p['nazwa_lat']})" if p.get("nazwa_lat") else name)
    return out


def existing_ids(przepisy_dir: Path) -> set[str]:
    return {r.rid for r in load_kwiatownik_recipes(przepisy_dir) if r.rid}
