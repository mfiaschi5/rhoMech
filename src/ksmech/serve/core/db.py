"""The curation database (`data/curation/curation.sqlite`): SQLModel tables and the engine.

`Mechanism` and `CuratedStep` carry exactly the brief's columns; JSON is stored as text.
`Recheck`, `ExportRecord` and `ExportedStep` are additive app tables (design §9.1).

Datetimes are stored as naive UTC (SQLite has no time zone type) and read back as aware UTC,
so a value read from the database compares equal to the `utcnow()` that wrote it.

This module imports no other serve module.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, event
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, Session, SQLModel, create_engine

log = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 30000


def utcnow() -> datetime:
    """The current time, timezone-aware UTC."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """A DateTime column that stores naive UTC and returns aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime {value!r}: use db.utcnow() or an aware datetime")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Mechanism(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, index=True, sa_type=UTCDateTime)
    reactants: str  # JSON list[str]: the input SMILES as entered
    stoichiometry: str  # JSON list[int], aligned with reactants
    conditions: str | None = None
    predictor: str = Field(index=True)  # floor | gbdt | network | user (spec v3: stub, ...)
    # the predictor_version string (embeds the model sha256[:16] for gbdt and the
    # physics fingerprint for floor/gbdt); never null
    checkpoint_hash: str
    spec_version: str = Field(index=True)
    status: str = Field(index=True)  # predicted | validated | corrected | rejected
    notes: str = ""


class CuratedStep(SQLModel, table=True):
    mechanism_id: int = Field(foreign_key="mechanism.id", primary_key=True)
    step_index: int = Field(primary_key=True)
    order_group: int
    reactant_pool_key: str = Field(index=True)
    product_pool_key: str = Field(index=True)
    arrows_json: str  # StepDoc (design §9.2)
    was_corrected: bool = False
    original_arrows_json: str | None = None  # StepDoc of the predicted edge it replaced


class Recheck(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    mechanism_id: int = Field(foreign_key="mechanism.id", index=True)
    step_index: int
    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime)
    predictor: str
    predictor_version: str
    spec_version: str
    physics_fingerprint: str
    aligned: bool
    first_arrow_rank: int | None = None
    n_valid: int | None = None
    top1: bool | None = None
    top5: bool | None = None
    product_match: str
    top1_reproduces_product: bool | None = None
    reason: str | None = None


class ExportRecord(SQLModel, table=True):
    export_id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime)
    path: str
    sha256: str
    n_steps: int
    report_json: str


class ExportedStep(SQLModel, table=True):
    step_key: str = Field(primary_key=True)  # JSON [reactant_pool_key, product_pool_key]
    step_id: str
    mechanism_id: int
    step_index: int
    first_export_id: str
    last_export_id: str


TABLES = (Mechanism, CuratedStep, Recheck, ExportRecord, ExportedStep)


def _set_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def make_engine(db_path: Path) -> Engine:
    """A SQLite engine on `db_path` (parent created): WAL, 30 s busy timeout, any thread."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": BUSY_TIMEOUT_MS / 1000},
    )
    event.listen(engine, "connect", _set_pragmas)
    log.debug("curation db at %s", db_path)
    return engine


def init_db(engine: Engine) -> None:
    """Create the curation tables that do not exist yet (idempotent)."""
    SQLModel.metadata.create_all(engine, tables=[t.__table__ for t in TABLES])


@contextmanager
def session(engine: Engine) -> Iterator[Session]:
    """One unit of work: commits on normal exit, rolls back and re-raises on an exception.

    `expire_on_commit=False`, so rows stay readable after the block.
    """
    with Session(engine, expire_on_commit=False) as s:
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
