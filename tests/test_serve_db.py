"""The curation database: tables, keys, pragmas and the session unit of work (tmp SQLite)."""

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlmodel import select

from ksmech.serve import db

pytestmark = pytest.mark.serve


@pytest.fixture
def engine(tmp_path):
    eng = db.make_engine(tmp_path / "curation" / "curation.sqlite")
    db.init_db(eng)
    yield eng
    eng.dispose()


def _mechanism(**kw) -> db.Mechanism:
    base = dict(
        reactants=json.dumps(["C[O-]", "CCl"]),
        stoichiometry=json.dumps([1, 1]),
        predictor="floor",
        checkpoint_hash="floor/ks3+ks2/d2.3/0123456789ab",
        spec_version="v3+0123456789ab;tok7;smp6;ori5;qm1",
        status="predicted",
    )
    return db.Mechanism(**{**base, **kw})


def _step(mid: int, i: int, **kw) -> db.CuratedStep:
    base = dict(
        mechanism_id=mid,
        step_index=i,
        order_group=i,
        reactant_pool_key="C[O-].CCl",
        product_pool_key="COC.[Cl-]",
        arrows_json=json.dumps({"format": "ksmech-serve/step-v1"}),
    )
    return db.CuratedStep(**{**base, **kw})


def test_tables_are_created(engine, tmp_path):
    assert (tmp_path / "curation" / "curation.sqlite").is_file()
    names = set(inspect(engine).get_table_names())
    assert names == {"mechanism", "curatedstep", "recheck", "exportrecord", "exportedstep"}
    db.init_db(engine)  # idempotent
    cols = {c["name"] for c in inspect(engine).get_columns("curatedstep")}
    assert cols == {
        "mechanism_id",
        "step_index",
        "order_group",
        "reactant_pool_key",
        "product_pool_key",
        "arrows_json",
        "was_corrected",
        "original_arrows_json",
    }
    cols = {c["name"] for c in inspect(engine).get_columns("mechanism")}
    assert cols == {
        "id",
        "created_at",
        "reactants",
        "stoichiometry",
        "conditions",
        "predictor",
        "checkpoint_hash",
        "spec_version",
        "status",
        "notes",
    }


def test_curated_step_has_a_composite_primary_key(engine):
    pk = inspect(engine).get_pk_constraint("curatedstep")["constrained_columns"]
    assert pk == ["mechanism_id", "step_index"]
    fks = inspect(engine).get_foreign_keys("curatedstep")
    assert [(f["referred_table"], f["referred_columns"]) for f in fks] == [("mechanism", ["id"])]
    with db.session(engine) as s:
        m = _mechanism()
        s.add(m)
        s.flush()
        s.add_all([_step(m.id, 0), _step(m.id, 1)])
    with pytest.raises(IntegrityError), db.session(engine) as s:
        s.add(_step(m.id, 1, order_group=5))  # same (mechanism_id, step_index)
    with db.session(engine) as s:
        rows = s.exec(select(db.CuratedStep).order_by(db.CuratedStep.step_index)).all()
        assert [(r.mechanism_id, r.step_index, r.order_group) for r in rows] == [
            (m.id, 0, 0),
            (m.id, 1, 1),
        ]
        assert s.get(db.CuratedStep, (m.id, 1)).product_pool_key == "COC.[Cl-]"


def test_pragmas_wal_and_busy_timeout(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 30000


def test_engine_is_usable_from_another_thread(engine):
    import threading

    out = []

    def work():
        with db.session(engine) as s:
            s.add(_mechanism(notes="from a thread"))
        with db.session(engine) as s:
            out.append(s.exec(select(db.Mechanism)).one().notes)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert out == ["from a thread"]


def test_session_commits_on_exit(engine):
    with db.session(engine) as s:
        m = _mechanism(notes="kept")
        s.add(m)
    assert m.id is not None and m.notes == "kept"  # readable after the block
    with db.session(engine) as s:
        assert s.get(db.Mechanism, m.id).notes == "kept"


def test_session_rolls_back_on_exception(engine):
    with pytest.raises(RuntimeError, match="boom"), db.session(engine) as s:
        s.add(_mechanism(notes="lost"))
        s.flush()
        raise RuntimeError("boom")
    with db.session(engine) as s:
        assert s.exec(select(db.Mechanism)).all() == []


def test_datetimes_round_trip_as_aware_utc(engine):
    t = datetime(2026, 10, 21, 12, 30, tzinfo=timezone(timedelta(hours=2)))
    with db.session(engine) as s:
        m = _mechanism(created_at=t)
        auto = _mechanism()
        s.add_all([m, auto])
    with db.session(engine) as s:
        got = s.get(db.Mechanism, m.id).created_at
        assert got == t and got.tzinfo is UTC
        assert s.get(db.Mechanism, auto.id).created_at.tzinfo is UTC
    assert db.utcnow().tzinfo is UTC


def test_naive_datetimes_are_refused(engine):
    with pytest.raises(StatementError, match="naive datetime"), db.session(engine) as s:
        s.add(_mechanism(created_at=datetime(2026, 10, 21)))


def test_side_tables_store_rows(engine):
    with db.session(engine) as s:
        m = _mechanism()
        s.add(m)
        s.flush()
        s.add(
            db.Recheck(
                mechanism_id=m.id,
                step_index=0,
                predictor="floor",
                predictor_version="floor/ks3+ks2/d2.3/0123456789ab",
                spec_version=m.spec_version,
                physics_fingerprint="0123456789ab",
                aligned=True,
                product_match="exact",
            )
        )
        s.add(
            db.ExportRecord(export_id="x1", path="p", sha256="0" * 64, n_steps=1, report_json="{}")
        )
        s.add(
            db.ExportedStep(
                step_key=json.dumps(["C[O-].CCl", "COC.[Cl-]"]),
                step_id="user_curated-0123456789abcdef",
                mechanism_id=m.id,
                step_index=0,
                first_export_id="x1",
                last_export_id="x1",
            )
        )
    with db.session(engine) as s:
        r = s.exec(select(db.Recheck)).one()
        assert r.top1 is None and r.first_arrow_rank is None and r.created_at.tzinfo is UTC
        assert s.get(db.ExportRecord, "x1").n_steps == 1
        assert s.exec(select(db.ExportedStep)).one().mechanism_id == m.id
