"""Proponowane zadania: co warto teraz zrobic, pogrupowane w podobne zadania (mozna wybrac wiele pozycji).

Kazda grupa ma liste pozycji (zrodla, przepisy, kraje, zadania...) z aktualnym statusem
oraz akcje, ktore uruchamiaja zadania w tle (asynchronicznie, przez kolejke).
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.discovery.countries import COUNTRIES
from app.discovery.finder import DiscoveryConfig
from app.models import (DiscoveredSite, Item, ItemKind, ItemStatus, Job, JobStatus, Plant, PlantFact, SearchQuery,
                        Source)
from app.worker.export import items_without_enrichment

ACTIVE = [JobStatus.queued, JobStatus.running]


def _plants_without_photos() -> list[str]:
    from pathlib import Path
    d = Path(settings.kwiatownik_plants_dir)
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and not data.get("url"):
            out.append(f.stem)
    return out


def _unorganized_plants() -> list[tuple[str, int]]:
    """(id, liczba informacji) plikow roslin z blokiem "wiedza" bez uporzadkowanych sekcji."""
    from pathlib import Path
    d = Path(settings.kwiatownik_plants_dir)
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            raw = f.read_text(encoding="utf-8-sig")
            if '"wiedza"' not in raw:
                continue
            w = json.loads(raw).get("wiedza") or {}
        except (OSError, ValueError, AttributeError):
            continue
        if w.get("fakty") and not w.get("sekcje"):
            out.append((f.stem, len(w["fakty"])))
    return out


def _unmerged_plants() -> list[tuple[str, int]]:
    """(id, liczba punktow) plikow roslin z uporzadkowana wiedza z sieci, jeszcze nie scalona z rozdzialami."""
    from pathlib import Path
    d = Path(settings.kwiatownik_plants_dir)
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            raw = f.read_text(encoding="utf-8-sig")
            if '"sekcje"' not in raw:
                continue
            data = json.loads(raw)
        except (OSError, ValueError):
            continue
        w = data.get("wiedza") or {}
        if w.get("sekcje") and not data.get("scalone"):
            out.append((f.stem, sum(len(x.get("punkty") or []) for x in w["sekcje"])))
    return out


CATEGORIES = {"sources": "Zrodla i skany", "recipes": "Przepisy i eksport", "plants": "Wiedza o roslinach",
              "discover": "Odkrywanie", "jobs": "Zadania"}


@dataclass
class TaskItem:
    id: str
    label: str
    detail: str = ""
    link: str | None = None
    status: str | None = None      # np. "w kolejce", "dziala" - gdy juz jest zadanie dla tej pozycji


@dataclass
class TaskGroup:
    key: str
    category: str
    title: str
    description: str
    actions: list[tuple[str, str]]              # (klucz akcji, etykieta przycisku)
    items: list[TaskItem] = field(default_factory=list)
    total: int = 0
    priority: int = 50                           # mniejsze = wyzej na liscie

    @property
    def busy(self) -> int:
        return sum(1 for i in self.items if i.status)


def _utc(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt and dt.tzinfo is None else dt


def _active_jobs(s: Session) -> list[Job]:
    return list(s.exec(select(Job).where(col(Job.status).in_(ACTIVE))).all())


def _status_label(job: Job) -> str:
    return "dziala" if job.status == JobStatus.running else "w kolejce"


def build_groups(limit: int = 150) -> list[TaskGroup]:
    now = datetime.now(timezone.utc)
    stale = now - timedelta(days=7)
    groups: list[TaskGroup] = []
    with Session(engine) as s:
        active = _active_jobs(s)
        by_source = {(j.source_id, j.kind): _status_label(j) for j in active if j.source_id}
        by_kind = {j.kind: _status_label(j) for j in active}
        in_translation: dict[int, str] = {}
        for j in active:
            if j.kind == "translate" and j.params_json:
                for iid in json.loads(j.params_json).get("item_ids") or []:
                    in_translation[iid] = _status_label(j)
        sources = s.exec(select(Source).where(Source.active == True).order_by(Source.name)).all()  # noqa: E712
        recipes_per_source = dict(s.exec(select(Item.source_id, func.count()).where(Item.kind == ItemKind.recipe)
                                          .group_by(Item.source_id)).all())

        # ---------------- zrodla
        new = [x for x in sources if not x.last_scan_at]
        groups.append(TaskGroup("scan_new", "sources", "Nieskanowane zrodla",
                                "Zrodla dodane, ale jeszcze nigdy nie przeskanowane.", [("scan", "Skanuj")],
                                [TaskItem(str(x.id), x.name, x.base_url, f"/sources/{x.id}", by_source.get((x.id, "scan")))
                                 for x in new], len(new), 10))
        old = [x for x in sources if x.last_scan_at and _utc(x.last_scan_at) < stale]
        groups.append(TaskGroup("scan_stale", "sources", "Dawno nieskanowane (ponad 7 dni)",
                                "Skan pobierze tylko nowe strony - znane sa pomijane.", [("scan", "Skanuj ponownie")],
                                [TaskItem(str(x.id), x.name, f"ostatnio: {_utc(x.last_scan_at):%Y-%m-%d}",
                                          f"/sources/{x.id}", by_source.get((x.id, "scan"))) for x in old], len(old), 40))
        prof = [x for x in sources if x.profile_status in (None, "failed") and recipes_per_source.get(x.id, 0) >= 2]
        groups.append(TaskGroup("build_profile", "sources", "Zbuduj scraper (LLM)",
                                "Zrodla z min. 2 znalezionymi przepisami bez dzialajacego profilu - po zbudowaniu "
                                "skany pobieraja przepisy bez LLM.", [("build_profile", "Zbuduj scraper")],
                                [TaskItem(str(x.id), x.name, f"{recipes_per_source.get(x.id, 0)} przepisow"
                                          + (" - poprzednia proba nieudana" if x.profile_status == "failed" else ""),
                                          f"/sources/{x.id}", by_source.get((x.id, "build_profile"))) for x in prof],
                                len(prof), 30))
        harvested = {j.source_id for j in s.exec(select(Job).where(Job.kind == "harvest")).all()}
        ready = [x for x in sources if x.profile_status == "ready" and x.id not in harvested]
        groups.append(TaskGroup("harvest", "sources", "Gotowe do pobrania calej strony",
                                "Strony z gotowym profilem scrapera, ktorych jeszcze nie pobrano w calosci - "
                                "pobieranie idzie selektorami, bez zapytan do LLM.", [("harvest", "Pobierz cala strone")],
                                [TaskItem(str(x.id), x.name, x.base_url, "/scrapers",
                                          by_source.get((x.id, "harvest"))) for x in ready], len(ready), 12))
        deg = [x for x in sources if x.profile_status == "degraded"]
        groups.append(TaskGroup("rebuild_profile", "sources", "Scraper do przebudowy",
                                "Profil coraz czesciej zawodzi - strona mogla zmienic wyglad.",
                                [("build_profile", "Przebuduj scraper")],
                                [TaskItem(str(x.id), x.name, x.base_url, f"/sources/{x.id}",
                                          by_source.get((x.id, "build_profile"))) for x in deg], len(deg), 20))

        # zrodla, ktore daja glownie przepisy kulinarne albo karty produktow (sklepy) - do wylaczenia
        reasons = s.exec(select(Item.source_id, Item.kind, Item.reason, func.count()).where(
            col(Item.source_id).is_not(None)).group_by(Item.source_id, Item.kind, Item.reason)).all()
        misfit: dict[int, list[int]] = {}   # source_id -> [niepasujace, przepisy_ok]
        for sid, kind, reason, n in reasons:
            r = (reason or "").lower()
            m = misfit.setdefault(sid, [0, 0])
            if "kulinarn" in r or "produktu" in r or "sklep" in r:
                m[0] += n
            elif kind == ItemKind.recipe and "pominiety" not in r:
                m[1] += n
        bad = [x for x in sources if (misfit.get(x.id, [0, 0])[0] >= 5
                                      and misfit[x.id][0] >= 3 * max(misfit[x.id][1], 1))
               or ((x.pages_scanned or 0) >= 40 and misfit.get(x.id, [0, 0])[1] == 0)]
        groups.append(TaskGroup("misfit", "sources", "Zrodla niepasujace do Kwiatownika",
                                "Strony, z ktorych prawie wszystko to przepisy kulinarne albo produkty w sklepie, "
                                "albo po 40+ stronach nie dalo sie znalezc zadnego przepisu ziolowego. "
                                "Wylaczenie oszczedza skany i limity LLM (dane zostaja w bazie).",
                                [("deactivate", "Wylacz zrodlo")],
                                [TaskItem(str(x.id), x.name, f"niepasujace: {misfit.get(x.id, [0, 0])[0]}, przepisy ziolowe: "
                                          f"{misfit.get(x.id, [0, 0])[1]}, stron: {x.pages_scanned or 0}",
                                          f"/sources/{x.id}") for x in bad], len(bad), 18))

        # ---------------- wiedza o roslinach
        fresh = s.exec(select(Plant).where(col(Plant.last_research_at).is_(None))
                       .order_by(col(Plant.mentions).desc(), Plant.nazwa_pl)).all()
        groups.append(TaskGroup("plants_new", "plants", "Rosliny bez wiedzy z sieci",
                                "Siedziba jeszcze nie zbierala o nich informacji (Wikipedia w kilku jezykach + strony "
                                "z innych krajow). Zweryfikowane informacje trafia do plikow roslin ze zrodlami.",
                                [("plant_research", "Zbierz wiedze")],
                                [TaskItem(p.id, p.nazwa_pl, (p.nazwa_lat or "") + (" - nowa" if p.origin == "nowa" else ""),
                                          f"/plants/{p.id}", by_kind.get("plant_research")) for p in fresh[:limit]],
                                len(fresh), 45))
        waiting = s.exec(select(PlantFact.plant_id, func.count()).where(PlantFact.status == "verified")
                         .group_by(PlantFact.plant_id)).all()
        groups.append(TaskGroup("plants_apply", "plants", "Zweryfikowana wiedza czeka na zapis",
                                "Informacje potwierdzone przez drugi model, jeszcze nie zapisane w plikach roslin.",
                                [("plant_apply", "Zapisz do Kwiatownika")],
                                [TaskItem(pid, pid, f"{n} informacji", f"/plants/{pid}") for pid, n in waiting],
                                len(waiting), 42))
        no_photos = _plants_without_photos()
        groups.append(TaskGroup("plants_photos", "plants", "Rosliny bez zdjec",
                                "Pliki roslin bez galerii. Siedziba pobierze zdjecia z Wikipedii / Wikimedia Commons "
                                "(z autorem i licencja) - bez LLM, szybko.",
                                [("plant_photos", "Pobierz zdjecia")],
                                [TaskItem(pid, pid, "", f"/plants/{pid}") for pid in no_photos[:limit]],
                                len(no_photos), 43))
        messy = _unorganized_plants()
        groups.append(TaskGroup("plants_organize", "plants", "Wiedza w plikach do uporzadkowania",
                                "Pliki roslin z wiedza zapisana jako luzna lista informacji (bez sekcji). Agregator "
                                "ulozy ja w sekcje, polaczy powtorzenia i ponumeruje zrodla.",
                                [("plant_organize", "Uporzadkuj")],
                                [TaskItem(pid, pid, f"{n} informacji", f"/plants/{pid}") for pid, n in messy[:limit]],
                                len(messy), 41))

        unmerged = _unmerged_plants()
        groups.append(TaskGroup("plants_merge", "plants", "Wiedza z sieci do scalenia z rozdzialami",
                                "Pliki roslin z wiedza z sieci, ktora nie jest jeszcze wmontowana w rozdzialy strony "
                                "(Medycyna, Surowce, Rzemioslo...). Bielik polaczy ja z tekstem Kwiatownika, "
                                "z przypisami do zrodel; reczny tekst zostaje w pliku.",
                                [("plant_merge", "Scal z rozdzialami")],
                                [TaskItem(pid, pid, f"{n} punktow", f"/plants/{pid}") for pid, n in unmerged[:limit]],
                                len(unmerged), 40))

        # ---------------- przepisy
        def recipe_items(status: ItemStatus, order=None) -> tuple[list[Item], int]:
            base = select(Item).where(Item.kind == ItemKind.recipe, Item.status == status,
                                      col(Item.duplicate_of).is_(None))
            total = s.exec(select(func.count()).select_from(base.subquery())).one()
            rows = s.exec(base.order_by(order if order is not None else col(Item.id).desc()).limit(limit)).all()
            return list(rows), total

        raw, n_raw = recipe_items(ItemStatus.raw, col(Item.confidence).desc())
        groups.append(TaskGroup("translate", "recipes", "Przepisy do przetlumaczenia",
                                "Tlumaczenie na polski + sprawdzenie, czy taki przepis juz istnieje.",
                                [("translate", "Przetlumacz i sprawdz"), ("reject", "Odrzuc")],
                                [TaskItem(str(i.id), i.title or i.url, i.language or "", f"/items/{i.id}",
                                          in_translation.get(i.id)) for i in raw], n_raw, 20))
        err, n_err = recipe_items(ItemStatus.error)
        groups.append(TaskGroup("translate_errors", "recipes", "Nieudane tlumaczenia",
                                "Mozna sprobowac ponownie (np. innym modelem z listy).",
                                [("translate", "Przetlumacz ponownie"), ("reject", "Odrzuc")],
                                [TaskItem(str(i.id), i.title or i.url, (i.reason or "")[:80], f"/items/{i.id}",
                                          in_translation.get(i.id)) for i in err], n_err, 35))
        tr, n_tr = recipe_items(ItemStatus.translated)
        groups.append(TaskGroup("approve", "recipes", "Do zatwierdzenia",
                                "Przetlumaczone i niepowtarzajace sie przepisy - po zatwierdzeniu trafia do eksportu.",
                                [("approve", "Zatwierdz"), ("reject", "Odrzuc")],
                                [TaskItem(str(i.id), json.loads(i.data_json).get("tytul", i.title) if i.data_json else i.title,
                                          i.title or "", f"/items/{i.id}") for i in tr], n_tr, 15))
        if settings.enrich_recipes:
            to_enrich = items_without_enrichment(session=s)
            if to_enrich:
                rows = s.exec(select(Item).where(col(Item.id).in_(to_enrich[:50]))).all()
                groups.append(TaskGroup("enrich", "recipes", "Do opracowania",
                                        "Przetlumaczone przepisy bez opracowania w tradycyjnych systemach "
                                        "(Piec Przemian, Kampo, ajurweda...) - np. gdy modele mialy limit.",
                                        [("enrich", "Opracuj")],
                                        [TaskItem(str(i.id), json.loads(i.data_json).get("tytul", i.title),
                                                  i.title or "", f"/items/{i.id}") for i in rows],
                                        len(to_enrich), 20))
        # duplikaty (duplicate_of) eksport dolacza jako 2. zrodlo przepisu glownego - nie czekaja na eksport
        n_appr = s.exec(select(func.count()).select_from(Item).where(Item.status == ItemStatus.approved,
                                                                     col(Item.duplicate_of).is_(None))).one()
        groups.append(TaskGroup("export", "recipes", "Eksport do Kwiatownika",
                                "Zapisuje zatwierdzone przepisy (ze zrodlami) do Kwiatownik2.",
                                [("export", "Eksportuj")],
                                [TaskItem("all", f"{n_appr} zatwierdzonych przepisow czeka na eksport",
                                          settings.export_filename, "/export", by_kind.get("export"))] if n_appr else [],
                                1 if n_appr else 0, 25))

        # ---------------- odkrywanie
        for st, key, title, desc, prio in [
            ("verified", "add_verified", "Zweryfikowane strony do dodania", "Maja przepisy - dodanie = trafia do Zrodel.", 45),
            ("maybe", "review_maybe", "Strony do przejrzenia", "Moga miec przepisy albo ciekawostki.", 60),
        ]:
            base = select(DiscoveredSite).where(DiscoveredSite.status == st)
            total = s.exec(select(func.count()).select_from(base.subquery())).one()
            rows = s.exec(base.order_by(col(DiscoveredSite.score).desc()).limit(limit)).all()
            groups.append(TaskGroup(key, "discover", title, desc, [("add_source", "Dodaj do zrodel"), ("reject_site", "Odrzuc")],
                                    [TaskItem(str(r.id), r.domain, f"{r.country} - {(r.title or '')[:70]}", r.url)
                                     for r in rows], total, prio))
        last = dict(s.exec(select(SearchQuery.country, func.max(SearchQuery.created_at)).group_by(SearchQuery.country)).all())
        countries = [(c, last.get(code)) for code, c in COUNTRIES.items()
                     if not last.get(code) or _utc(last[code]) < now - timedelta(days=14)]
        groups.append(TaskGroup("discover", "discover", "Szukaj nowych stron",
                                "Kraje jeszcze nieprzeszukane albo przeszukane ponad 14 dni temu.",
                                [("discover", "Szukaj (5 zapytan)")],
                                [TaskItem(c.code, c.name, f"ostatnio: {_utc(t):%Y-%m-%d}" if t else "jeszcze nie szukano",
                                          None, by_kind.get(f"discover:{c.code}")) for c, t in countries],
                                len(countries), 70))

        # ---------------- zadania
        retried = set(s.exec(select(Job.retry_of).where(col(Job.retry_of).is_not(None))).all())
        failed = [j for j in s.exec(select(Job).where(col(Job.status).in_([JobStatus.failed, JobStatus.cancelled]))
                                    .order_by(col(Job.id).desc()).limit(100)).all()
                  if j.id not in retried and _utc(j.created_at) > now - timedelta(days=7)]
        names = {x.id: x.name for x in s.exec(select(Source)).all()}
        groups.append(TaskGroup("retry", "jobs", "Przerwane lub nieudane zadania",
                                "Ponowienie uruchamia zadanie z tymi samymi ustawieniami; skany kontynuuja od "
                                "zapisanej kolejki.", [("retry", "Ponow")],
                                [TaskItem(str(j.id), f"#{j.id} {j.kind}" + (f" - {names.get(j.source_id, '')}" if j.source_id else ""),
                                          f"{j.status.value}, {j.progress or 0}/{j.total or '?'}", f"/jobs/{j.id}")
                                 for j in failed], len(failed), 5))
    return sorted(groups, key=lambda g: (g.total == 0, g.priority))


def run_task(state, group: str, action: str, ids: list[str]) -> str:
    """Wykonuje akcje dla wybranych pozycji grupy. Zwraca komunikat dla uzytkownika."""
    from app.worker import actions
    from app.worker.discover import add_site_as_source

    if not ids:
        return "Nic nie zaznaczono."
    int_ids = [int(i) for i in ids if str(i).isdigit()]
    if action == "scan":
        jobs = [actions.start_scan(state, sid) for sid in int_ids]
        return f"Uruchomiono {len(jobs)} skanow (kolejka: max {settings.max_concurrent_jobs} naraz)."
    if action == "harvest":
        jobs = [actions.start_harvest(state, sid) for sid in int_ids]
        return f"Uruchomiono pelne pobieranie {len(jobs)} stron (kolejka: max {settings.max_concurrent_jobs} naraz)."
    if action == "plant_research":
        jid = actions.start_plant_research(state, [str(i) for i in ids])
        return f"Zbieranie wiedzy o {len(ids)} roslinach - zadanie #{jid}."
    if action == "plant_apply":
        jid = actions.start_plant_apply(state, [str(i) for i in ids])
        return f"Zapis wiedzy {len(ids)} roslin - zadanie #{jid}."
    if action == "plant_photos":
        jid = actions.start_plant_photos(state, [str(i) for i in ids])
        return f"Zdjecia z Wikipedii dla {len(ids)} roslin - zadanie #{jid}."
    if action == "plant_organize":
        jid = actions.start_plant_organize(state, [str(i) for i in ids])
        return f"Porzadkowanie wiedzy {len(ids)} roslin - zadanie #{jid}."
    if action == "plant_merge":
        jid = actions.start_plant_merge(state, [str(i) for i in ids])
        return f"Scalanie wiedzy {len(ids)} roslin z rozdzialami - zadanie #{jid}."
    if action == "build_profile":
        jobs = [actions.start_build_profile(state, sid) for sid in int_ids]
        return f"Uruchomiono budowe {len(jobs)} scraperow."
    if action == "translate":
        jid = actions.start_translate(state, item_ids=int_ids)
        return f"Tlumaczenie {len(int_ids)} przepisow - zadanie #{jid}."
    if action == "enrich":
        jid = actions.start_enrich(state, item_ids=int_ids)
        return f"Opracowanie {len(int_ids)} przepisow - zadanie #{jid}."
    if action == "export":
        jid = actions.start_export(state)
        return f"Eksport - zadanie #{jid}."
    if action == "discover":
        jobs = [actions.start_discover(state, code, DiscoveryConfig()) for code in ids if code in COUNTRIES]
        return f"Szukanie stron w {len(jobs)} krajach."
    if action == "retry":
        jobs = [j for j in (actions.retry_job(state, i) for i in int_ids) if j]
        return f"Ponowiono {len(jobs)} zadan."
    if action == "add_source":
        added = [add_site_as_source(i) for i in int_ids]
        return f"Dodano {sum(1 for a in added if a)} stron do zrodel."
    with Session(engine) as s:
        if action == "deactivate":
            for src in s.exec(select(Source).where(col(Source.id).in_(int_ids))).all():
                src.active = False
                s.add(src)
            s.commit()
            return f"Wylaczono {len(int_ids)} zrodel (nie beda skanowane; mozna je wlaczyc na stronie zrodla)."
        if action in ("approve", "reject"):
            for it in s.exec(select(Item).where(col(Item.id).in_(int_ids))).all():
                if action == "approve" and not it.data_json:
                    continue
                it.status = ItemStatus.approved if action == "approve" else ItemStatus.skipped
                s.add(it)
            s.commit()
            return f"{'Zatwierdzono' if action == 'approve' else 'Odrzucono'} {len(int_ids)} przepisow."
        if action == "reject_site":
            for r in s.exec(select(DiscoveredSite).where(col(DiscoveredSite.id).in_(int_ids))).all():
                r.status = "rejected"
                s.add(r)
            s.commit()
            return f"Odrzucono {len(int_ids)} stron."
    return f"Nieznana akcja: {action}"
