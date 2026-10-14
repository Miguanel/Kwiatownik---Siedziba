"""Kampanie "od A do Z" (R13): planista prowadzi kilka roslin i jedno zrodlo przez caly potok, zanim zacznie nastepne.

Dawniej planista na kazde wolne miejsce bral najlepiej oceniona propozycje - a w kazdej grupie z listy "Zadania do
zrobienia" pierwsze pozycje. Efekt: zapis dla roslin A, scalanie dla B, zdjecia dla C, badania dla D; skan jednego
zrodla, pobieranie drugiego, scraper trzeciego. Duzo zaczetej pracy, malo domknietej (i ta sama praca bez efektu
zlecana w kolko). Kampania to fokus:

  * kampania roslin - AGENTS_FOCUS_PLANTS roslin przechodzi po kolei: badania -> zapis -> porzadkowanie -> scalanie
    -> zdjecia. Najpierw wybierane sa rosliny w polowie drogi (czekajaca wiedza), potem z zalecen audytu, potem nowe.
    Zadania dla roslin z kampanii ida wyzej, a w zadaniach zbiorczych (zapis 20 roslin) rosliny z kampanii sa
    pierwsze. Zbieranie wiedzy o roslinach SPOZA kampanii czeka (nizsza ocena), dopoki kampania trwa.
  * kampania zrodla - jedno zrodlo: skan -> scraper -> pobranie calej strony (i ponowienia jego przerwanych zadan).
    Skany i pobieranie innych zrodel ida nizej.
  * limit pracy w toku - gdy na tlumaczenie czeka AGENTS_TRANSLATE_WIP przepisow, nowe skany, pobieranie i szukanie
    stron czekaja (najpierw tlumaczenie tego, co juz jest).
  * bez efektu - roslina, dla ktorej ten sam rodzaj zadania niedawno sie wykonal, a ona dalej jest na liscie
    (np. zdjecia, ktorych nie ma w Wikipedii), nie jest zlecana ponownie przez NO_EFFECT_HOURS.

Kampania konczy sie, gdy jej rosliny / zrodlo zejda z listy zadan; po AGENTS_FOCUS_HOURS ustepuje nastepnej
(zrodlo, ktore nie dalo sie domknac, czeka SKIP_SOURCE_DAYS dni). Stan jest w data/agents_settings.json,
historia ostatnich kampanii - w raporcie planisty i na stronie /agents.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from sqlmodel import Session, col, select

from app.agents import store
from app.agents.tasks import PLANT_KINDS, params
from app.config import settings
from app.models import AgentTask, Job, JobStatus, Plant, Source

# etapy potoku roslin (grupa z listy "Zadania do zrobienia") - w tej kolejnosci roslina przechodzi przez potok
PLANT_STAGES = [("badania", "plants_new"), ("zapis", "plants_apply"), ("porzadkowanie", "plants_organize"),
                ("scalanie", "plants_merge"), ("zdjecia", "plants_photos")]
DOWNSTREAM = ("plants_apply", "plants_organize", "plants_merge", "plants_photos")
SOURCE_STAGES = [("pobranie calej strony", "harvest"), ("budowa scrapera", "build_profile"),
                 ("przebudowa scrapera", "rebuild_profile"), ("skan", "scan_new"), ("ponowny skan", "scan_stale")]
SOURCE_KINDS = ("skan", "pobierz_strone", "scraper")
COLLECTING = ("skan", "pobierz_strone")                     # dokladaja surowych przepisow do tlumaczenia
RESEARCH_JOB = "plant_research"
NO_EFFECT_HOURS = {"zdjecia": 72}                           # inne rodzaje: DEFAULT_NO_EFFECT_H
DEFAULT_NO_EFFECT_H = 4
SKIP_SOURCE_DAYS = 3
HISTORY = 8
BONUS, OUTSIDE, OUTSIDE_RESEARCH, WIP_PENALTY, WIP_BONUS = 18, 6, 22, 30, 25


@dataclass
class Focus:
    on: bool = True
    plants: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)       # roslina -> etap (albo "gotowe")
    plants_since: str | None = None
    tasks: list[int] = field(default_factory=list)            # zalecenia audytu, dla ktorych wybrano rosliny
    source: int | None = None
    source_name: str = ""
    source_stage: str = ""
    source_since: str | None = None
    backlog: int = 0                                          # przepisy czekajace na tlumaczenie
    wip_hold: bool = False
    history: list[dict] = field(default_factory=list)
    events: list[str] = field(default_factory=list)          # co sie zmienilo w tym przebiegu

    def report(self) -> dict:
        out = asdict(self)
        out["plants_hours"] = round(_age_h(self.plants_since), 1) if self.plants_since else None
        out["source_hours"] = round(_age_h(self.source_since), 1) if self.source_since else None
        out["wip_limit"] = settings.agents_translate_wip
        return out


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _age_h(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return (store.now() - store.utc(datetime.fromisoformat(iso))).total_seconds() / 3600
    except ValueError:
        return 0.0


def tried(s: Session, kind: str) -> set[str]:
    """Rosliny, dla ktorych zadanie tego rodzaju niedawno sie wykonalo (albo zatrzymalo w bramie). Jesli dalej
    sa na liscie, to zadanie nie dalo efektu - nie zlecamy go znowu przez NO_EFFECT_HOURS."""
    since = store.now() - timedelta(hours=NO_EFFECT_HOURS.get(kind, DEFAULT_NO_EFFECT_H))
    out: set[str] = set()
    for t in s.exec(select(AgentTask).where(AgentTask.kind == kind, col(AgentTask.status).in_(["done", "blocked"]),
                                            col(AgentTask.finished_at).is_not(None))).all():
        if store.utc(t.finished_at) >= since:
            out.update(str(x) for x in params(t).get("rosliny") or [])
    return out


def _active_plants(s: Session) -> set[str]:
    rows = s.exec(select(AgentTask).where(col(AgentTask.status).in_(["ordered", "running"]))).all()
    return {str(x) for t in rows if t.kind in PLANT_KINDS for x in params(t).get("rosliny") or []}


def _membership(groups: dict, skip: dict[str, set[str]]) -> dict[str, list[str]]:
    """roslina -> lista grup roslin (w kolejnosci potoku), w ktorych czeka - bez prob zakonczonych bez efektu."""
    from app.agents.planner import GROUPS
    out: dict[str, list[str]] = {}
    for _, key in PLANT_STAGES:
        g = groups.get(key)
        if not g:
            continue
        kind = GROUPS[key][0]
        for i in g.items:
            if i.id not in skip.get(kind, set()):
                out.setdefault(i.id, []).append(key)
    return out


def _stage_label(keys: list[str]) -> str:
    names = dict((k, n) for n, k in PLANT_STAGES)
    return names[keys[0]] if keys else "gotowe"


def _pick_plants(s: Session, groups: dict, member: dict, audit_tasks: list[AgentTask]) -> tuple[list[str], list[int]]:
    """Nowa kampania roslin: najpierw rosliny w polowie potoku, potem z najpilniejszego zalecenia audytu, potem nowe."""
    from app.agents.tasks import recently_ordered
    k = max(1, settings.agents_focus_plants)
    chosen: list[str] = []
    order = {}
    for key in DOWNSTREAM:
        for n, i in enumerate(groups[key].items if key in groups else []):
            order.setdefault(i.id, n)
    mid = [pid for pid, keys in member.items() if any(x in DOWNSTREAM for x in keys)]
    # najpierw rosliny ze swiezo zebrana wiedza (czeka na zapis - poczatek reszty potoku), potem dalsze etapy;
    # przy tym samym etapie - wiecej czekajacych etapow, potem kolejnosc na liscie
    stage_no = {key: n for n, (_, key) in enumerate(PLANT_STAGES)}
    first = {p: min(stage_no[x] for x in member[p] if x in DOWNSTREAM) for p in mid}
    mid.sort(key=lambda p: (first[p], -len(member[p]), order.get(p, 999)))
    chosen += mid[:k]
    task_ids: list[int] = []
    for t in audit_tasks:
        if len(chosen) >= k:
            break
        if t.kind not in PLANT_KINDS:
            continue
        recent = recently_ordered(s, t.kind, settings.agents_reorder_days)
        new = [str(x) for x in params(t).get("rosliny") or [] if str(x) not in recent and str(x) not in chosen]
        if new:
            chosen += new[:k - len(chosen)]
            task_ids.append(t.id)
    for i in groups["plants_new"].items if "plants_new" in groups else []:
        if len(chosen) >= k:
            break
        if i.id not in chosen:
            chosen.append(i.id)
    return chosen, task_ids


def _active_sources(s: Session) -> set[int]:
    """Zrodla, dla ktorych dziala (albo czeka w kolejce) jakies zadanie."""
    return {j.source_id for j in s.exec(select(Job).where(
        col(Job.status).in_([JobStatus.queued, JobStatus.running]), col(Job.source_id).is_not(None))).all()}


def _source_stage(sid: int, groups: dict, active: set[int], failed_src: set[int]) -> str:
    if sid in active:
        return "w toku"
    for label, key in SOURCE_STAGES:
        g = groups.get(key)
        if g and any(i.id == str(sid) for i in g.items):
            return label
    if sid in failed_src:
        return "ponowienie przerwanego zadania"
    return "gotowe"


def _pick_source(s: Session, groups: dict, skipped: dict, active: set[int],
                 failed: list[int]) -> tuple[int | None, str]:
    """Nowa kampania zrodla: zrodlo, nad ktorym juz cos dziala, potem przerwane (do ponowienia), potem najdalej
    w potoku (gotowy scraper -> pobranie), scraper, skan nowego zrodla, ponowny skan. Zrodla, ktorych kampania sie
    nie domknela, czekaja kilka dni."""
    rows = [int(i.id) for _, key in SOURCE_STAGES for i in (groups[key].items if key in groups else [])
            if i.id.isdigit()]
    rows = list(dict.fromkeys(failed + rows))
    rows = [r for r in rows if not (skipped.get(str(r)) and _age_h(skipped[str(r)]) < 24 * SKIP_SOURCE_DAYS)]
    rows.sort(key=lambda r: r not in active)                      # stabilnie: dzialajace najpierw
    if not rows:
        return None, ""
    src = s.get(Source, rows[0])
    return rows[0], (src.name if src else str(rows[0]))


def failed_sources(s: Session, groups: dict) -> dict[str, int]:
    """Nieudane zadania z listy "Przerwane lub nieudane", ktore planista moze ponowic -> ich zrodlo (zatrzymanych
    recznie albo przez straznika nie wznawiamy, wiec nie trzymaja kampanii)."""
    from app.worker.jobs import retry_depth
    g = groups.get("retry")
    ids = [int(i.id) for i in (g.items if g else []) if i.id.isdigit()]
    if not ids:
        return {}
    jobs = {j.id: j for j in s.exec(select(Job).where(col(Job.id).in_(ids))).all()}
    return {str(i): jobs[i].source_id for i in ids if i in jobs and jobs[i].source_id
            and jobs[i].status == JobStatus.failed and retry_depth(i) < settings.max_auto_retries}


def build(s: Session, groups: dict, audit_tasks: list[AgentTask]) -> Focus:
    """Stan kampanii na ten przebieg planisty: konczy domkniete (albo za dlugie) kampanie i zaczyna nastepne."""
    f = Focus(on=settings.agents_focus)
    rt = store.runtime()
    f.history = list(rt.get("kampanie_historia") or [])
    g = groups.get("translate")
    f.backlog = g.total if g else 0
    f.wip_hold = f.backlog >= settings.agents_translate_wip > 0
    if not f.on:
        return f
    skip = {kind: tried(s, kind) for kind in ("zapis", "porzadkuj", "scal", "zdjecia")}
    member = _membership(groups, skip)
    busy = _active_plants(s)
    changes: dict = {}

    # ---------------- kampania roslin
    camp = rt.get("kampania_roslin") or {}
    plants = [str(p) for p in camp.get("rosliny") or []]
    waiting_tasks = []
    if camp.get("zadania"):
        waiting_tasks = [t.id for t in s.exec(select(AgentTask).where(col(AgentTask.id).in_(camp["zadania"]),
                                                                      AgentTask.status == "proposed")).all()]

    def pending(pid: str) -> bool:
        return bool(member.get(pid)) or pid in busy

    if plants and (not any(pending(p) for p in plants) and not waiting_tasks
                   or _age_h(camp.get("od")) >= settings.agents_focus_hours):
        done = [p for p in plants if not pending(p)]
        why = "domknieta" if len(done) == len(plants) and not waiting_tasks else \
            f"limit {settings.agents_focus_hours} h (domkniete {len(done)}/{len(plants)})"
        f.history.insert(0, {"typ": "rosliny", "co": plants, "od": camp.get("od"), "do": _iso(store.now()),
                             "wynik": why, "godzin": round(_age_h(camp.get("od")), 1)})
        f.events.append(f"kampania roslin {', '.join(plants)}: {why}")
        plants, waiting_tasks = [], []
    if not plants:
        plants, waiting_tasks = _pick_plants(s, groups, member, audit_tasks)
        if plants:
            camp = {"rosliny": plants, "od": _iso(store.now()), "zadania": waiting_tasks}
            f.events.append("nowa kampania roslin: " + ", ".join(plants))
        else:
            camp = {}
        changes["kampania_roslin"] = camp
    f.plants, f.tasks, f.plants_since = plants, waiting_tasks, camp.get("od")
    if plants:
        f.names = dict(s.exec(select(Plant.id, Plant.nazwa_pl).where(col(Plant.id).in_(plants))).all())
        for p in plants:
            f.stages[p] = ("w toku" if p in busy and not member.get(p) else _stage_label(member.get(p, []))) \
                if pending(p) else ("zalecenie audytu" if waiting_tasks else "gotowe")

    # ---------------- kampania zrodla
    active = _active_sources(s)
    failed_order = list(dict.fromkeys(failed_sources(s, groups).values()))
    failed_src = set(failed_order)
    sc = rt.get("kampania_zrodla") or {}
    skipped = dict(rt.get("kampanie_pominiete_zrodla") or {})
    sid = sc.get("zrodlo")
    if sid is not None:
        stage = _source_stage(int(sid), groups, active, failed_src)
        too_long = _age_h(sc.get("od")) >= settings.agents_focus_hours
        if stage == "gotowe" or too_long:
            why = "domknieta" if stage == "gotowe" else f"limit {settings.agents_focus_hours} h (etap: {stage})"
            if too_long and stage != "gotowe":
                skipped[str(sid)] = _iso(store.now())
                changes["kampanie_pominiete_zrodla"] = {k: v for k, v in skipped.items()
                                                        if _age_h(v) < 24 * SKIP_SOURCE_DAYS}
            f.history.insert(0, {"typ": "zrodlo", "co": [sc.get("nazwa") or str(sid)], "od": sc.get("od"),
                                 "do": _iso(store.now()), "wynik": why, "godzin": round(_age_h(sc.get("od")), 1)})
            f.events.append(f"kampania zrodla {sc.get('nazwa') or sid}: {why}")
            sid, sc = None, {}
    if sid is None:
        new, name = _pick_source(s, groups, skipped, active, failed_order)
        sc = {"zrodlo": new, "nazwa": name, "od": _iso(store.now())} if new is not None else {}
        if new is not None:
            f.events.append(f"nowa kampania zrodla: {name}")
        changes["kampania_zrodla"] = sc
        sid = new
    if sid is not None:
        f.source, f.source_name, f.source_since = int(sid), sc.get("nazwa") or "", sc.get("od")
        f.source_stage = _source_stage(int(sid), groups, active, failed_src)
        if f.wip_hold and f.source_stage in ("skan", "ponowny skan", "pobranie calej strony"):
            f.source_stage += f" - czeka, az przetlumacza sie przepisy ({f.backlog})"
        if not f.source_name:
            src = s.get(Source, int(sid))
            f.source_name = src.name if src else str(sid)

    if f.events:
        changes["kampanie_historia"] = f.history[:HISTORY]
    f.history = f.history[:HISTORY]
    if changes:
        store.set_runtime(**changes)
    return f


def order_items(key: str, items: list, f: Focus | None, retry_src: dict[str, int] | None = None) -> list:
    """Pozycje grupy z kampania na poczatku (zadanie zbiorcze najpierw obejmie rosliny / zrodlo z kampanii)."""
    if not f or not f.on:
        return items
    if key in dict((k, n) for n, k in PLANT_STAGES):
        first = [i for i in items if i.id in f.plants]
    elif key == "retry":
        first = [i for i in items if f.source is not None and (retry_src or {}).get(i.id) == f.source]
    elif key in dict((k, n) for n, k in SOURCE_STAGES):
        first = [i for i in items if f.source is not None and i.id == str(f.source)]
    else:
        return items
    return first + [i for i in items if i not in first]


def adjust(props: list, f: Focus, retry_src: dict[str, int] | None = None) -> list:
    """Oceny propozycji wg kampanii: praca kampanii wyzej, nowe badania i skany spoza kampanii nizej. Przy duzej
    kolejce tlumaczen nowe skany i pobieranie odpadaja (nie dokladamy surowych przepisow), a tlumaczenie idzie
    na poczatek. Zwraca liste propozycji po zmianach."""
    if not f.on:
        return props
    if f.wip_hold:
        props = [p for p in props if p.kind not in COLLECTING]
    plant_names = ", ".join(f.names.get(p, p) for p in f.plants)
    for p in props:
        par = p.params or {}
        tag = ""
        if p.kind in PLANT_KINDS and f.plants:
            inside = [x for x in par.get("rosliny") or [] if str(x) in f.plants]
            if inside:
                p.score += BONUS
                tag = f"kampania roslin: {plant_names}"
            elif _is_research(p.kind):
                p.score -= OUTSIDE_RESEARCH
                tag = f"poza kampania - najpierw domkniecie: {plant_names}"
            else:
                p.score -= OUTSIDE
        elif p.kind in SOURCE_KINDS and f.source is not None:
            if str(f.source) in [str(x) for x in par.get("ids") or []]:
                p.score += BONUS
                tag = f"kampania zrodla: {f.source_name}"
            else:
                p.score -= BONUS
                tag = f"poza kampania zrodla ({f.source_name})"
        elif p.kind == "ponow" and f.source is not None and retry_src:
            src = [retry_src.get(str(x)) for x in par.get("ids") or []]
            if f.source in src:
                p.score += BONUS
                tag = f"kampania zrodla: {f.source_name}"
            elif any(src):
                p.score -= BONUS
                tag = f"poza kampania zrodla ({f.source_name})"
        if f.wip_hold and p.kind == "tlumaczenie":
            p.score += WIP_BONUS
            tag = f"najpierw tlumaczenie ({f.backlog} czeka) - nowe skany wstrzymane"
        elif f.wip_hold and p.kind == "nowe_zrodla":
            p.score -= WIP_PENALTY
            tag = f"pozniej: {f.backlog} przepisow czeka na tlumaczenie"
        if tag:
            p.why = f"{p.why} [{tag}]"
    return props


def _is_research(kind: str) -> bool:
    from app.agents.tasks import KINDS
    return kind in KINDS and KINDS[kind].job == RESEARCH_JOB
