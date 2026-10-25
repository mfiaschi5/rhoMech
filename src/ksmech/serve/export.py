"""The PMechDB-format download of curated steps (spec v9): a file for reading and upstream
contribution, never training data (nothing is written under `data/processed`, and no source of
the data layer reads it).

`pmechdb_export(engine, statuses)` takes the steps of every mechanism whose status is requested
(validated and corrected by default), in pathway order (mechanism creation time, mechanism id,
order group, step index), and writes one raw PMechDB record per step: the mapped reaction
SMILES `reactants>>products` (every H an atom, as the pool states hold them) and the arrow
code of the step's arrow reading (`serve.reading.pmech_code`: `a=b`, `a,c=a,b`, `a,c=c` on map
numbers), in the headerless one-column CSV that `data.pmechdb.load` reads (`pmech_format`).

Each record is checked before it is written: its arrow code is parsed back
(`data.pmechdb.parse_arrow_code`) and applied as one edit (`arrows.apply.apply_arrows`), and the
product must have the sigma key of the curated product. Every step left out is counted with a
reason code: SPEC_V3_STEP (a step-v1 document, no reaction matrix), NO_ARROW_READING,
ROUNDTRIP_FAILED, DUPLICATE_IN_EXPORT (the same record already written).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlmodel import col, select

from ksmech.serve import db

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

DEFAULT_STATUSES = ("validated", "corrected")


def step_rows(session, statuses: Sequence[str]) -> list[tuple]:
    """(mechanism, step, StepDoc) of every mechanism with a status in `statuses`, in order."""
    q = (
        select(db.CuratedStep, db.Mechanism)
        .join(db.Mechanism, col(db.Mechanism.id) == col(db.CuratedStep.mechanism_id))
        .where(col(db.Mechanism.status).in_(list(statuses)))
        .order_by(
            col(db.Mechanism.created_at),
            col(db.Mechanism.id),
            col(db.CuratedStep.order_group),
            col(db.CuratedStep.step_index),
        )
    )
    return [(m, s, json.loads(s.arrows_json)) for s, m in session.exec(q).all()]


def roundtrip(doc: dict) -> bool:
    """The record's arrow code, applied to its reactants, gives its product (sigma key)."""
    from ksmech.arrows.apply import apply_arrows
    from ksmech.data import chem
    from ksmech.data.pmechdb import parse_arrow_code
    from ksmech.data.resonance import sigma_key
    from ksmech.data.validation import StepRejected

    try:
        arrows = parse_arrow_code(str(doc["pmechdb_code"]))
        res = apply_arrows(chem.parse_mapped(doc["reactants_mapped"]), arrows)
        want = sigma_key(chem.parse_mapped(doc["products_mapped"]))
    except (StepRejected, NotImplementedError, ValueError, KeyError):
        return False
    return res.ok and res.key == want


def _reason(doc: dict, seen: set[str]) -> str | None:
    if not doc.get("matrix"):
        return "SPEC_V3_STEP"
    if not doc.get("pmechdb_code"):
        return "NO_ARROW_READING"
    if record(doc) in seen:
        return "DUPLICATE_IN_EXPORT"
    if not roundtrip(doc):
        return "ROUNDTRIP_FAILED"
    return None


def record(doc: dict) -> str:
    return f"{doc['reactants_mapped']}>>{doc['products_mapped']} {doc['pmechdb_code']}"


def pmechdb_export(engine: Engine, statuses: Sequence[str] = DEFAULT_STATUSES) -> tuple[str, dict]:
    """(the CSV text, the report {statuses, n_mechanisms, n_steps_considered,
    n_steps_written, skipped})."""
    with db.session(engine) as s:
        rows = step_rows(s, statuses)
    skipped: Counter = Counter()
    lines: list[str] = []
    for m, step, doc in rows:
        why = _reason(doc, set(lines))
        if why is not None:
            skipped[why] += 1
            log.warning("export: mechanism %s step %s skipped [%s]", m.id, step.step_index, why)
            continue
        lines.append(record(doc))
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerows([r] for r in lines)
    report = {
        "statuses": list(statuses),
        "n_mechanisms": len({m.id for m, _, _ in rows}),
        "n_steps_considered": len(rows),
        "n_steps_written": len(lines),
        "skipped": dict(sorted(skipped.items())),
    }
    return buf.getvalue(), report
