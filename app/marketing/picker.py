"""Dobor roslin do posta: z plikow Kwiatownika (kalendarz_ogrodnika) wedlug biezacego miesiaca."""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from app.marketing.season import Season

HARVEST_WORDS = ("zbiór", "zbior", "zbieranie", "zbieramy", "kopanie", "wykop")


@dataclass
class PlantPick:
    id: str
    nazwa_pl: str
    nazwa_lat: str | None
    url: str
    tasks: list[dict] = field(default_factory=list)       # czynnosci z kalendarza na ten czas
    facts: list[str] = field(default_factory=list)        # ciekawostki
    recipes: list[str] = field(default_factory=list)      # tytuly przepisow
    uses: dict[str, str] = field(default_factory=dict)    # zastosowanie (medyczne, kulinarne...)
    warning: str | None = None
    score: float = 0.0

    def brief(self) -> dict:
        return {"id": self.id, "nazwa_pl": self.nazwa_pl, "nazwa_lat": self.nazwa_lat, "url": self.url}


def load_plants(plants_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(plants_dir).glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("nazwa_pl"):
            out[data.get("id") or p.stem] = data
    return out


def _months(season: Season) -> set[int]:
    months = {season.month}
    if season.phase == "koniec":
        months.add(season.month % 12 + 1)  # pod koniec miesiaca dziadek zapowiada, co bedzie dalej
    return months


def season_tasks(data: dict, season: Season) -> list[dict]:
    months = _months(season)
    tasks = (data.get("kalendarz_ogrodnika") or {}).get("zadania") or []
    return [t for t in tasks if isinstance(t, dict) and months & set(t.get("miesiace") or [])]


def to_pick(pid: str, data: dict, season: Season, site_url: str, rnd: random.Random) -> PlantPick:
    tasks = season_tasks(data, season)
    facts = [f for f in (data.get("ciekawostki") or []) if isinstance(f, str)]
    rnd.shuffle(facts)
    recipes = [r.get("tytul") for key in ("przepisy_medyczne", "przepisy_kulinarne")
               for r in (data.get(key) or []) if isinstance(r, dict) and r.get("tytul")]
    uses = {k: v[:300] for k, v in (data.get("zastosowanie") or {}).items() if isinstance(v, str) and v.strip()}
    warning = data.get("ostrzezenia")
    pick = PlantPick(
        id=pid, nazwa_pl=data["nazwa_pl"], nazwa_lat=data.get("nazwa_lat"),
        url=f"{site_url.rstrip('/')}/plant/{pid}/",
        tasks=[{k: t.get(k) for k in ("czynnosc", "opis", "pora_dnia", "faza_ksiezyca") if t.get(k)} for t in tasks],
        facts=facts[:2], recipes=recipes[:3], uses=uses,
        warning=warning[:350] if isinstance(warning, str) and warning.strip() else None)
    harvest = any(any(w in (t.get("czynnosc") or "").lower() for w in HARVEST_WORDS) for t in tasks)
    pick.score = (3 if tasks else 0) + (2 if harvest else 0) + (1 if recipes else 0) + (0.5 if facts else 0)
    return pick


def candidates(plants: dict[str, dict], season: Season, site_url: str, recent: set[str] | None = None,
               seed: int | None = None) -> list[PlantPick]:
    """Rosliny, ktore maja cos do roboty w tym czasie (kalendarz), najlepsze na gorze.
    Rosliny z ostatnich postow (recent) spadaja na koniec - zeby dziadek sie nie powtarzal."""
    rnd = random.Random(seed)
    recent = recent or set()
    picks = [to_pick(pid, d, season, site_url, rnd) for pid, d in plants.items()]
    picks = [p for p in picks if p.tasks]
    for p in picks:
        p.score += rnd.random() * 1.5          # odrobina losowosci - nie zawsze te same
        if p.id in recent:
            p.score -= 10
    return sorted(picks, key=lambda p: -p.score)


def choose(plants: dict[str, dict], season: Season, site_url: str, plant_id: str | None = None,
           recent: set[str] | None = None, count: int = 2, seed: int | None = None) -> list[PlantPick]:
    """Glowna roslina (wybrana recznie albo najlepsza z kalendarza) + ewentualnie jedna poboczna."""
    rnd = random.Random(seed)
    ranked = candidates(plants, season, site_url, recent, seed)
    chosen: list[PlantPick] = []
    if plant_id and plant_id in plants:
        chosen.append(to_pick(plant_id, plants[plant_id], season, site_url, rnd))
    for p in ranked:
        if len(chosen) >= count:
            break
        if all(p.id != c.id for c in chosen):
            chosen.append(p)
    if not chosen and plants:  # zaden plik nie ma kalendarza na ten miesiac - losowa roslina
        pid = rnd.choice(sorted(plants))
        chosen.append(to_pick(pid, plants[pid], season, site_url, rnd))
    return chosen
