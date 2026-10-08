"""Slownik zadan, ktore agenci moga zlecic Siedzibie, ich wykonanie (kolejka) i sledzenie wyniku."""
import json
from dataclasses import dataclass

from sqlmodel import Session, col, select

from app.agents.store import now, utc
from app.db import engine
from app.models import AgentTask, Job, JobStatus


@dataclass(frozen=True)
class Kind:
    label: str
    job: str                       # rodzaj zadania Siedziby (Job.kind)
    slots: tuple = ()              # podrozdzialy schematu, na ktorych skupia sie wyszukiwanie (plant_research)
    llm: bool = False              # wymaga modeli LLM
    hint: str = ""                 # opis dla eksperta LLM / panelu


KINDS: dict[str, Kind] = {
    # --- wiedza trudno dostepna i opowiesci (cel Kwiatownika)
    "opowiesci": Kind("Opowiesci, legendy i wierzenia", "plant_research", ("ciekawostki",), True,
                      "legendy, wierzenia, obrzedy, historia i nazwy ludowe z zagranicznych zrodel (uk, ru, ja, zh, de, fr)"),
    "wschod": Kind("Medycyna Wschodu (TCM, kampo)", "plant_research",
                   ("profil_energetyczny.opis", "profil_energetyczny.smak", "profil_energetyczny.termika"), True,
                   "natura, smak, meridiany i wskazania w medycynie chinskiej i kampo - zrodla chinskie i japonskie"),
    "barwienie": Kind("Barwienie i rzemioslo", "plant_research", ("zastosowanie.rzemieslnicze",), True,
                      "barwienie tkanin, welny i wlosow (草木染め, Pflanzenfarben) i inne rzemiosla"),
    "kosmetyka": Kind("Kosmetyka naturalna", "plant_research", ("zastosowanie.kosmetyczne",), True,
                      "pielegnacja skory i wlosow, kosmetyki ziolowe"),
    "lecznicze": Kind("Ziololecznictwo i surowce", "plant_research",
                      ("zastosowanie.medyczne", "czesci_rosliny"), True,
                      "dzialanie lecznicze, surowce, substancje czynne, medycyna ludowa roznych krajow"),
    "bezpieczenstwo": Kind("Toksycznosc i interakcje", "plant_research", ("ostrzezenia", "interakcje"), True,
                           "toksycznosc, przeciwwskazania, interakcje z lekami"),
    "luki": Kind("Uzupelnienie luk w schemacie", "plant_research", (), True,
                 "najwazniejsze puste podrozdzialy rosliny (Siedziba sama dobiera tematy i jezyki)"),
    "nowa_wiedza": Kind("Pierwsze zbieranie wiedzy", "plant_research", (), True,
                        "rosliny, o ktorych Siedziba jeszcze nic nie zebrala"),
    # --- domykanie potoku wiedzy
    "zapis": Kind("Zapis zweryfikowanej wiedzy", "plant_apply"),
    "porzadkuj": Kind("Porzadkowanie wiedzy w plikach", "plant_organize"),
    "scal": Kind("Scalenie wiedzy z rozdzialami", "plant_merge", hint="wmontowanie wiedzy z sieci w rozdzialy strony"),
    "zdjecia": Kind("Zdjecia z Wikipedii", "plant_photos", hint="galeria roslin bez zdjec (bez LLM)"),
    # --- przepisy i zrodla
    "nowe_zrodla": Kind("Szukanie nowych stron", "discover", hint="nowe zagraniczne strony z przepisami ziolowymi w kraju"),
    "skan": Kind("Skan zrodla", "scan"),
    "pobierz_strone": Kind("Pobranie calej strony", "harvest"),
    "scraper": Kind("Budowa scrapera", "build_profile", llm=True),
    "tlumaczenie": Kind("Tlumaczenie przepisow", "translate"),
    "opracowanie": Kind("Opracowanie przepisow", "enrich", llm=True),
    "eksport": Kind("Eksport przepisow do Kwiatownika", "export"),
    "ponow": Kind("Ponowienie przerwanego zadania", "retry"),
    "audyt": Kind("Audyt bazy wiedzy", "agent_audit", hint="nowy raport oceny bazy wiedzy Kwiatownika"),
}
# rodzaje, ktore moze zaproponowac ekspert LLM w audycie (dotycza roslin)
EXPERT_KINDS = ("opowiesci", "wschod", "barwienie", "kosmetyka", "lecznicze", "bezpieczenstwo", "luki", "scal", "zdjecia")
PLANT_KINDS = {k for k, v in KINDS.items() if v.job.startswith("plant")}
ACTIVE = [JobStatus.queued, JobStatus.running]
NOT_PRODUCTION = ("agent_", "fb_", "ollama_pull")      # zadania, ktore nie licza sie jako "Siedziba pracuje"
STATUS_LABELS = {"proposed": "zaproponowane", "ordered": "zlecone", "running": "w toku", "done": "wykonane",
                 "failed": "nieudane", "skipped": "pominiete", "expired": "zastapione"}


def params(task: AgentTask) -> dict:
    try:
        return json.loads(task.params_json or "{}")
    except ValueError:
        return {}


def describe(kind: str, p: dict) -> str:
    k = KINDS.get(kind)
    label = k.label if k else kind
    if p.get("rosliny"):
        names = p.get("nazwy") or p["rosliny"]
        return f"{label}: " + ", ".join(names[:6]) + ("..." if len(names) > 6 else "")
    if p.get("kraj"):
        return f"{label}: {p.get('kraj_nazwa') or p['kraj']}"
    if p.get("etykieta"):
        return f"{label}: {p['etykieta']}"
    return label


def needs_llm(kind: str) -> bool:
    return bool(KINDS.get(kind) and KINDS[kind].llm)


def execute(state, kind: str, p: dict) -> int:
    """Zleca zadanie Siedzibie (kolejka). Zwraca id zadania (Job). ValueError - nie da sie zlecic."""
    from app.discovery.countries import COUNTRIES
    from app.discovery.finder import DiscoveryConfig
    from app.worker import actions

    k = KINDS.get(kind)
    if not k:
        raise ValueError(f"nieznany rodzaj zadania: {kind}")
    if k.llm and getattr(state, "llm", None) is None:
        raise ValueError("brak modeli LLM (ustaw klucze GEMINI/GROQ)")
    plants = [str(x) for x in p.get("rosliny") or []]
    ids = [int(x) for x in p.get("ids") or [] if str(x).isdigit()]
    if k.job == "plant_research":
        slots = list(p.get("sloty") or k.slots) or None
        if kind == "nowa_wiedza" and not plants:
            return actions.start_plant_research(state, None, int(p.get("limit") or 3))
        if not plants:
            return actions.start_plant_research(state, None, int(p.get("limit") or 3), slots=slots)
        return actions.start_plant_research(state, plants, len(plants), slots=slots)
    if k.job in ("plant_apply", "plant_organize", "plant_merge", "plant_photos"):
        fn = {"plant_apply": actions.start_plant_apply, "plant_organize": actions.start_plant_organize,
              "plant_merge": actions.start_plant_merge, "plant_photos": actions.start_plant_photos}[k.job]
        return fn(state, plants or None)
    if kind == "nowe_zrodla":
        code = (p.get("kraj") or "").lower()
        if code not in COUNTRIES:
            raise ValueError(f"nieznany kraj: {code}")
        return actions.start_discover(state, code, DiscoveryConfig())
    if kind in ("skan", "pobierz_strone", "scraper"):
        if not ids:
            raise ValueError("brak zrodla")
        fn = {"skan": actions.start_scan, "pobierz_strone": actions.start_harvest,
              "scraper": actions.start_build_profile}[kind]
        return fn(state, ids[0])
    if kind == "tlumaczenie":
        return actions.start_translate(state, p.get("limit"), ids or None)
    if kind == "opracowanie":
        return actions.start_enrich(state, p.get("limit"), ids or None)
    if kind == "eksport":
        return actions.start_export(state)
    if kind == "audyt":
        return actions.start_agent_audit(state, trigger=p.get("trigger") or "planista")
    if kind == "ponow":
        new = actions.retry_job(state, ids[0]) if ids else None
        if not new:
            raise ValueError("tego zadania nie da sie ponowic")
        return new
    raise ValueError(f"nieobslugiwany rodzaj: {kind}")


def order(state, task_id: int, dispatch_run_id: int | None = None) -> tuple[bool, str]:
    """Zleca zaproponowane zadanie (AgentTask). Zwraca (czy zlecono, komunikat)."""
    with Session(engine) as s:
        t = s.get(AgentTask, task_id)
        if not t:
            return False, "nie ma takiego zadania"
        if t.status not in ("proposed", "failed", "skipped", "expired"):
            return False, f"zadanie jest juz {STATUS_LABELS.get(t.status, t.status)}"
        kind, p = t.kind, params(t)
    try:
        jid = execute(state, kind, p)
    except Exception as exc:   # noqa: BLE001 - zapisujemy powod w historii
        with Session(engine) as s:
            t = s.get(AgentTask, task_id)
            t.status, t.note = "skipped", f"nie zlecono: {exc}"[:400]
            s.add(t)
            s.commit()
        return False, str(exc)
    with Session(engine) as s:
        t = s.get(AgentTask, task_id)
        job = s.get(Job, jid)
        t.status, t.job_id, t.ordered_at = "ordered", jid, now()
        t.job_kind = job.kind if job else KINDS[kind].job
        t.dispatch_run_id = dispatch_run_id or t.dispatch_run_id
        t.note = None
        s.add(t)
        s.commit()
    return True, f"zadanie #{jid}"


def sync_tasks() -> int:
    """Statusy zleconych zadan z kolejki (Job) -> AgentTask, z wynikiem po zakonczeniu. Zwraca liczbe zmian."""
    changed = 0
    with Session(engine) as s:
        rows = s.exec(select(AgentTask).where(col(AgentTask.status).in_(["ordered", "running"]),
                                              col(AgentTask.job_id).is_not(None))).all()
        for t in rows:
            job = s.get(Job, t.job_id)
            if not job:
                t.status, t.note = "failed", "zadanie zniknelo z bazy"
            elif job.status == JobStatus.running:
                if t.status == "running":
                    continue
                t.status = "running"
            elif job.status == JobStatus.queued:
                continue
            else:
                t.status = "done" if job.status == JobStatus.done else "failed"
                t.result_json = job.stats_json
                t.finished_at = job.finished_at or now()
                if job.status != JobStatus.done:
                    t.note = (job.log or "").strip().splitlines()[-1][:300] if job.log else job.status.value
            s.add(t)
            changed += 1
        s.commit()
    return changed


def production_jobs(s: Session) -> list[Job]:
    """Dzialajace / czekajace zadania Siedziby (bez agentow, fanpage i pobierania modeli)."""
    rows = s.exec(select(Job).where(col(Job.status).in_(ACTIVE))).all()
    return [j for j in rows if not j.kind.startswith(NOT_PRODUCTION)]


def recently_ordered(s: Session, kind: str, days: int) -> set[str]:
    """Rosliny / kraje / zrodla, dla ktorych zadanie tego rodzaju zlecono w ostatnich `days` dniach."""
    from datetime import timedelta
    since = now() - timedelta(days=days)
    out: set[str] = set()
    rows = s.exec(select(AgentTask).where(AgentTask.kind == kind, col(AgentTask.ordered_at).is_not(None),
                                          col(AgentTask.status).in_(["ordered", "running", "done"]))).all()
    for t in rows:
        if utc(t.ordered_at) < since:
            continue
        p = params(t)
        out.update(str(x) for x in p.get("rosliny") or [])
        if p.get("kraj"):
            out.add(p["kraj"])
        out.update(str(x) for x in p.get("ids") or [])
    return out
