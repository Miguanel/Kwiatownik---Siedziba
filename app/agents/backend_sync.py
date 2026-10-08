"""Polaczenie z backendem Kwiatownika na Render (repozytorium KwiatownikBackend).

Co BACKEND_SYNC_MINUTES (petla agentow):
  1. heartbeat: czy Siedziba pracuje i nad czym (ludzkim jezykiem) + nowe wpisy dziennika na zywo
     (zakonczone zadania: zebrana wiedza, zdjecia, tlumaczenia, audyty, publikacje) -> papirus na stronie glownej,
  2. kopia licznikow strony (odslony roslin, plant.id, rozdzialy, zrodla, wyszukiwania) -> tabela SiteStat,
  3. gdy serwer wystartowal od nowa i zgubil dysk (darmowy Render) -> odeslanie kopii licznikow (restore).
Heartbeat co 5 min przy okazji nie pozwala serwerowi zasnac, dopoki Siedziba dziala.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
from sqlmodel import Session, col, func, select

from app.config import settings
from app.db import engine
from app.models import AgentRun, Job, JobStatus, Plant, PlantFact, SiteStat
from app.worker import activity

log = logging.getLogger(__name__)
TZ = ZoneInfo(settings.timezone)
EVENT_KINDS = ("plant_research", "plant_apply", "plant_merge", "plant_photos", "translate", "enrich", "export",
               "scan", "harvest")
JOB_TEXT = {
    "plant_research": "zbiera wiedzę o roślinach", "plant_apply": "zapisuje zebraną wiedzę do Kwiatownika",
    "plant_merge": "wplata wiedzę z sieci w rozdziały roślin", "plant_organize": "porządkuje wiedzę o roślinach",
    "plant_photos": "pobiera zdjęcia z Wikimedia Commons", "plants_sync": "porządkuje rejestr roślin",
    "scan": "przegląda zagraniczne strony z przepisami", "harvest": "pobiera przepisy z zagranicznej strony",
    "scan_pages": "czyta wybrane strony", "fetch_list": "czyta wybrane strony",
    "build_profile": "uczy się czytać nową stronę z przepisami", "translate": "tłumaczy przepisy na polski",
    "enrich": "opracowuje przepisy w tradycyjnych systemach", "export": "przygotowuje przepisy do publikacji",
    "import_k1": "przenosi przepisy ze starego Kwiatownika", "agent_audit": "ocenia bazę wiedzy Kwiatownika",
    "agent_deploy": "publikuje nową wersję Kwiatownika", "fb_post": "pisze post na fanpage",
    "fb_ideas": "wyszukuje ciekawostki na fanpage",
}


def configured() -> bool:
    return bool(settings.backend_url.strip() and settings.backend_token.strip())


def _state_path():
    return settings.data_dir / "backend_sync.json"


def load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    try:
        _state_path().write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except OSError:
        log.exception("Nie udalo sie zapisac stanu synchronizacji z backendem")


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _names(ids) -> dict[str, str]:
    ids = [str(i) for i in ids or []]
    if not ids:
        return {}
    with Session(engine) as s:
        return {p.id: p.nazwa_pl for p in s.exec(select(Plant).where(col(Plant.id).in_(ids))).all()}


def _job_label(job: Job) -> str:
    kind = job.kind.split(":", 1)[0]
    text = JOB_TEXT.get(kind) or ("szuka nowych stron z przepisami" if kind == "discover" else "pracuje")
    if kind == "discover" and ":" in job.kind:
        from app.discovery.countries import COUNTRIES
        c = COUNTRIES.get(job.kind.split(":", 1)[1])
        text += f" ({c.name})" if c else ""
    act = activity.get(job.id) or {}
    detail = str(act.get("detail") or "").strip()
    if kind in ("plant_research", "plant_merge", "plant_photos", "plant_apply") and detail and len(detail) < 60:
        text += f": {detail}"
    return text


def status_payload() -> dict:
    """Stan Siedziby dla papirusu: co teraz robi (bez szczegolow technicznych)."""
    from app.agents.tasks import production_jobs
    with Session(engine) as s:
        jobs = production_jobs(s)
        running = [j for j in jobs if j.status == JobStatus.running]
        info = {
            "rosliny": s.exec(select(func.count()).select_from(Plant)).one(),
            "informacje_zapisane": s.exec(select(func.count()).select_from(PlantFact)
                                          .where(PlantFact.status == "applied")).one(),
            "czeka_w_kolejce": len(jobs) - len(running),
        }
    zadania = [{"opis": _job_label(j), "od": _iso(j.started_at)} for j in running[:4]]
    return {"status": "pracuje" if running else "czeka", "zadania": zadania, "info": info}


def _event_for_job(job: Job) -> dict | None:
    try:
        st = json.loads(job.stats_json or "{}")
        p = json.loads(job.params_json or "{}")
    except ValueError:
        return None
    if not isinstance(st, dict):
        return None
    kind = job.kind.split(":", 1)[0]
    ids = [str(i) for i in (p.get("plant_ids") or [])]
    names = _names(ids)
    plants_txt = ", ".join(names.get(i, i) for i in ids[:4]) + ("…" if len(ids) > 4 else "")
    url = f"/plant/{ids[0]}/" if len(ids) == 1 else None
    text, ev_kind = None, "info"
    if kind == "plant_research":
        n = int(st.get("verified") or 0)
        if n:
            text = (f"Zebrano {n} sprawdzonych informacji" + (f" o: {plants_txt}" if plants_txt else "")
                    + " – trafią na stronę przy najbliższej publikacji")
            ev_kind = "wiedza"
    elif kind == "plant_photos" and st.get("applied"):
        text, ev_kind = f"Nowe zdjęcia z Wikimedia Commons dla {st['applied']} roślin", "zdjecia"
    elif kind == "plant_merge" and st.get("merged"):
        text = f"Wiedza z sieci wpleciona w rozdziały" + (f": {plants_txt}" if plants_txt else f" {st['merged']} roślin")
        ev_kind = "wiedza"
    elif kind == "translate" and st.get("translated"):
        text, ev_kind = f"Przetłumaczono {st['translated']} przepisów z zagranicznych stron", "przepisy"
    elif kind == "enrich" and st.get("enriched"):
        text, ev_kind = f"Opracowano {st['enriched']} przepisów w tradycyjnych systemach", "przepisy"
    elif kind in ("scan", "harvest"):
        found = int(st.get("recipe") or 0) + sum(int((st.get(k) or {}).get("recipe") or 0)
                                                  for k in ("rozpoznanie", "pobieranie") if isinstance(st.get(k), dict))
        if found:
            text, ev_kind = f"Znaleziono {found} nowych przepisów na zagranicznych stronach", "przepisy"
    if not text:
        return None
    return {"eid": f"job:{job.id}", "ts": _iso(job.finished_at), "kind": ev_kind, "text": text, "url": url,
            "plant_id": ids[0] if len(ids) == 1 else None}


def _event_for_run(run: AgentRun) -> dict | None:
    from app.agents import store
    rep = store.report(run)
    if run.agent == "audytor" and run.score is not None:
        return {"eid": f"run:{run.id}", "ts": _iso(run.finished_at or run.created_at), "kind": "audyt",
                "text": f"Audyt bazy wiedzy: ocena {run.score:.0f}/100 – Siedziba wie, czego szukać dalej"}
    if run.agent == "wdrozeniowiec" and rep.get("opublikowano"):
        wpis = rep.get("wpis") or {}
        return {"eid": f"run:{run.id}", "ts": _iso(run.finished_at or run.created_at), "kind": "wdrozenie",
                "text": "Opublikowano nową wersję Kwiatownika: " + (wpis.get("tytul") or "nowe dane")}
    return None


def new_events(st: dict, limit: int = 30) -> tuple[list[dict], int, int]:
    """Wpisy dziennika od ostatniej synchronizacji. Zwraca (wpisy, nowy kursor zadan, nowy kursor przebiegow)."""
    last_job, last_run = int(st.get("last_job_id") or 0), int(st.get("last_run_id") or 0)
    out: list[dict] = []
    with Session(engine) as s:
        if not last_job and not last_run:          # pierwsze polaczenie: tylko ostatnia doba
            since = datetime.now(timezone.utc) - timedelta(days=1)
            jobs = s.exec(select(Job).where(Job.status == JobStatus.done, col(Job.finished_at) >= since)
                          .order_by(Job.id)).all()
            runs = s.exec(select(AgentRun).where(AgentRun.status == "done", AgentRun.created_at >= since)
                          .order_by(AgentRun.id)).all()
        else:
            jobs = s.exec(select(Job).where(Job.id > last_job, col(Job.status).in_(
                [JobStatus.done, JobStatus.failed, JobStatus.cancelled])).order_by(Job.id).limit(200)).all()
            runs = s.exec(select(AgentRun).where(AgentRun.id > last_run, AgentRun.status == "done")
                          .order_by(AgentRun.id).limit(100)).all()
        active_min = s.exec(select(func.min(Job.id)).where(col(Job.status).in_(
            [JobStatus.queued, JobStatus.running]))).one()
    for j in jobs:
        if j.status == JobStatus.done and j.kind.split(":", 1)[0] in EVENT_KINDS:
            ev = _event_for_job(j)
            if ev:
                out.append(ev)
    for r in runs:
        ev = _event_for_run(r)
        if ev:
            out.append(ev)
    # kursor zadan: najwyzsze zakonczone id, ale nie dalej niz najstarsze wciaz dzialajace (zeby go nie pominac;
    # wpisy wyslane drugi raz backend pomija po "eid")
    done_ids = [j.id for j in jobs]
    new_job = max(done_ids, default=last_job)
    if active_min is not None:
        new_job = min(new_job, active_min - 1) if new_job >= active_min else new_job
    new_run = max((r.id for r in runs), default=last_run)
    return out[-limit:], max(last_job, new_job), new_run


# ------------------------------------------------------------------ liczniki strony (kopia w Siedzibie)
def save_counters(rows: list[dict]) -> int:
    n = 0
    with Session(engine) as s:
        for r in rows:
            try:
                day, kind, key, val = str(r["day"])[:10], str(r["kind"])[:32], str(r.get("key") or "")[:120], int(r["n"])
            except (KeyError, TypeError, ValueError):
                continue
            row = s.get(SiteStat, (day, kind, key))
            if row is None:
                s.add(SiteStat(day=day, kind=kind, key=key, n=val))
            elif row.n != val:
                row.n = val
                s.add(row)
            n += 1
        s.commit()
    return n


def restore_rows(days: int = 31) -> list[dict]:
    """Kopia dla serwera po restarcie: ostatnie `days` dni dzien po dniu + starsze zsumowane jako "archiwum"."""
    cut = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
    out: list[dict] = []
    with Session(engine) as s:
        for r in s.exec(select(SiteStat).where(SiteStat.day >= cut, SiteStat.day != "archiwum")).all():
            out.append({"day": r.day, "kind": r.kind, "key": r.key, "n": r.n})
        old = s.exec(select(SiteStat.kind, SiteStat.key, func.sum(SiteStat.n)).where(
            (SiteStat.day < cut) | (SiteStat.day == "archiwum")).group_by(SiteStat.kind, SiteStat.key)).all()
    out += [{"day": "archiwum", "kind": k, "key": key, "n": int(n)} for k, key, n in old if n]
    return out


async def sync(client: httpx.AsyncClient | None = None) -> dict:
    """Jedna synchronizacja z backendem. Zwraca podsumowanie (zapisywane tez w data/backend_sync.json)."""
    if not configured():
        return {"ok": False, "powod": "brak BACKEND_URL / BACKEND_TOKEN w .env"}
    st = load_state()
    own = client is None
    client = client or httpx.AsyncClient(base_url=settings.backend_url.rstrip("/"), timeout=80,
                                         headers={"Authorization": f"Bearer {settings.backend_token}"})
    out: dict = {"ok": False, "kiedy": datetime.now(timezone.utc).isoformat()}
    try:
        events, job_cur, run_cur = new_events(st)
        payload = {**status_payload(), "wpisy": events}
        r = await client.post("/api/siedziba/heartbeat", json=payload)
        r.raise_for_status()
        hb = r.json()
        st.update(last_job_id=job_cur, last_run_id=run_cur, boot=hb.get("boot"))
        out.update(wpisy=len(events), boot=hb.get("boot"))
        if hb.get("potrzebna_kopia"):
            rows = restore_rows()
            rr = await client.post("/api/siedziba/restore", json={"wiersze": rows})
            if rr.status_code not in (200, 409):
                rr.raise_for_status()
            out["odtworzono"] = len(rows)
        since = (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
        rc = await client.get("/api/siedziba/counters", params={"since": since})
        rc.raise_for_status()
        out["liczniki"] = save_counters(rc.json().get("wiersze") or [])
        out["ok"] = True
        st.update(last_ok=out["kiedy"], last_error=None)
    except (httpx.HTTPError, ValueError) as exc:
        out["blad"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        st.update(last_error=out["blad"], last_error_at=out["kiedy"])
        log.warning("Backend Kwiatownika: synchronizacja nieudana: %s", out["blad"])
    finally:
        if own:
            await client.aclose()
        save_state(st)
    return out


async def push_event(text: str, kind: str = "info", eid: str | None = None, url: str | None = None) -> None:
    """Pojedynczy wpis na zywo od razu (np. publikacja) - bez czekania na kolejna synchronizacje."""
    if not configured():
        return
    try:
        async with httpx.AsyncClient(base_url=settings.backend_url.rstrip("/"), timeout=80,
                                     headers={"Authorization": f"Bearer {settings.backend_token}"}) as c:
            await c.post("/api/siedziba/heartbeat", json={**status_payload(), "wpisy": [
                {"eid": eid, "ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "text": text, "url": url}]})
    except httpx.HTTPError as exc:
        log.warning("Backend Kwiatownika: wpis nieudany: %s", exc)


def site_views(days: int = 30) -> dict[str, int]:
    """Odslony roslin na stronie z ostatnich `days` dni (do audytu: popularne rosliny najpierw)."""
    cut = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
    with Session(engine) as s:
        rows = s.exec(select(SiteStat.key, func.sum(SiteStat.n)).where(
            SiteStat.kind == "plant_view", SiteStat.day >= cut, SiteStat.day != "archiwum").group_by(SiteStat.key)).all()
    return {k: int(n) for k, n in rows}
