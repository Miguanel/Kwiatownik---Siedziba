import logging

from sqlalchemy import event, inspect, text
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

log = logging.getLogger(__name__)
settings.data_dir.mkdir(parents=True, exist_ok=True)

_sqlite = settings.database_url.startswith("sqlite")
# timeout: przy zajetej bazie czekaj do 30 s zamiast od razu zglaszac "database is locked"
_connect_args = {"check_same_thread": False, "timeout": 30} if _sqlite else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)

if _sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        # WAL: czytanie nie blokuje zapisu - wazne przy kilku zadaniach w tle naraz
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()


def _add_missing_columns() -> None:
    """Prosta automatyczna migracja: dodaje kolumny, ktore pojawily sie w modelach."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                coltype = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}'))
                log.info("Migracja: dodano kolumne %s.%s", table.name, col.name)


def init_db() -> None:
    from app import models  # noqa: F401  (rejestracja tabel)
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def get_session():
    with Session(engine) as session:
        yield session
