"""Zapis przebiegow agentow (AgentRun) i ustawienia zmieniane w panelu (data/agents_settings.json)."""
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from app.config import settings
from app.db import engine
from app.models import AgentRun

log = logging.getLogger(__name__)
_TZ = ZoneInfo(settings.timezone)
AGENTS = {"audytor": "Audytor wiedzy", "zleceniodawca": "Zleceniodawca", "planista": "Planista ciaglosci",
          "wdrozeniowiec": "Wdrozeniowiec"}


def _settings_path():
    return settings.data_dir / "agents_settings.json"


def runtime() -> dict:
    """Ustawienia z panelu (nadpisuja .env): {"auto": bool}."""
    out = {"auto": settings.agents_auto_continue}
    try:
        out.update(json.loads(_settings_path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return out


def set_runtime(**values) -> dict:
    cur = runtime()
    cur.update(values)
    try:
        _settings_path().write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        log.exception("Nie udalo sie zapisac ustawien agentow")
    return cur


def now() -> datetime:
    return datetime.now(timezone.utc)


def utc(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt and dt.tzinfo is None else dt


def start_run(agent: str, trigger: str = "reczny", job_id: int | None = None, parent_id: int | None = None) -> int:
    with Session(engine) as s:
        run = AgentRun(agent=agent, trigger=trigger, job_id=job_id, parent_id=parent_id)
        s.add(run)
        s.commit()
        return run.id


def run_log(run_id: int, message: str) -> None:
    stamp = datetime.now(_TZ).strftime("%H:%M:%S")
    with Session(engine) as s:
        run = s.get(AgentRun, run_id)
        if run:
            run.log = (run.log or "") + f"{stamp} {message}\n"
            s.add(run)
            s.commit()


def finish_run(run_id: int, status: str = "done", report: dict | None = None, **fields) -> None:
    with Session(engine) as s:
        run = s.get(AgentRun, run_id)
        if not run:
            return
        run.status = status
        run.finished_at = run.updated_at = now()
        if report is not None:
            run.report_json = json.dumps(report, ensure_ascii=False, default=str)
        for k, v in fields.items():
            setattr(run, k, v)
        s.add(run)
        s.commit()


def last_run(agent: str, status: str | None = "done") -> AgentRun | None:
    with Session(engine) as s:
        stmt = select(AgentRun).where(AgentRun.agent == agent)
        if status:
            stmt = stmt.where(AgentRun.status == status)
        return s.exec(stmt.order_by(col(AgentRun.id).desc())).first()


def report(run: AgentRun | None) -> dict:
    if not run or not run.report_json:
        return {}
    try:
        return json.loads(run.report_json)
    except ValueError:
        return {}
