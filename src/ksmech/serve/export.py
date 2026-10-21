"""Export of the curated steps (design 9.6): the `export` job handler.

Every export is a full, deterministic regeneration from the curation database:

1. the PMechDB pairs (the injected `PairsProvider`; None: no PMechDB dedupe, a warning);
2. the steps of every mechanism whose status is requested, in pathway order (mechanism
   creation time, mechanism id, order group, step index);
3. per step, a raw row in the canonical schema, unless it has no dataset arrows
   (ARROW_ENCODING_FAILED), its (reactant pool key, product pool key) is a PMechDB step
   (DUPLICATE_OF_PMECHDB) or was already emitted in this export (DUPLICATE_IN_EXPORT; a pair
   counts as emitted once one of its rows passes `user_curated.check_row`, so a row the ingest
   rejects does not block a valid copy of the same step);
4. `data/curation/export_raw/steps.parquet` (STEP_SCHEMA), re-ingested through
   `ksmech.data.user_curated.load` (build_step per row, counted rejections) and written with
   `write_outputs` under `data/processed/user_curated/` only;
5. `pmechdb_arrow_codes.csv`, the PMechDB raw format for upstream contribution, from H-compacted
   steps (`compact_step`; an uncompactable step is written as is, counted COMPACT_FALLBACK);
6. the `ExportedStep` ledger (changed step ids, withdrawn keys), `export_report.json` and an
   `ExportRecord` row with the sha256 of `steps.parquet`.

Step ids are content hashes and the rows are written in pathway order, so re-exporting an
unchanged database gives byte-identical parquet files. Every file is written atomically.

API side (the batch job thread): the data layer and pyarrow are imported inside the functions,
so importing this module loads only RDKit, sqlmodel and serve modules.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem
from sqlmodel import Session, col, select

from ksmech.serve import db
from ksmech.serve.contracts import JobContext, PairsProvider
from ksmech.serve.errors import Code
from ksmech.serve.schemas import ExportIn, ExportReportOut
from ksmech.serve.settings import ServeSettings, assert_writable
from ksmech.serve.versions import file_sha256, spec_version

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ksmech.data.ingest import IngestResult
    from ksmech.data.schema import ElementaryStep

log = logging.getLogger(__name__)

USER_CURATED = "user_curated"  # data.schema.Source.USER_CURATED
INVALID_ARROWS = "<invalid arrows>"  # data.user_curated.INVALID_ARROWS
PMECHDB_UNAVAILABLE = "PMECHDB_UNAVAILABLE"
CODES_FILE = "pmechdb_arrow_codes.csv"
REPORT_FILE = "export_report.json"
ARROW_FIELDS = ("source_kind", "source_atoms", "sink_kind", "sink_atoms")


@dataclass
class CuratedRow:
    """One curated step of an exportable mechanism, with its parsed StepDoc."""

    mechanism: db.Mechanism
    step: db.CuratedStep
    doc: dict

    @property
    def key(self) -> tuple[str, str]:
        return (self.step.reactant_pool_key, self.step.product_pool_key)

    @property
    def where(self) -> dict:
        return {"mechanism_id": int(self.mechanism.id), "step_index": int(self.step.step_index)}


@dataclass
class _Plan:
    raw: list[dict] = field(default_factory=list)  # STEP_SCHEMA rows, in pathway order
    entries: list[dict] = field(default_factory=list)  # one per considered step (report)
    by_step_id: dict[str, tuple[CuratedRow, dict]] = field(default_factory=dict)
    skipped: Counter = field(default_factory=Counter)
    mechanisms: set[int] = field(default_factory=set)


def step_rows(session: Session, statuses: Sequence[str]) -> list[CuratedRow]:
    """The steps of every mechanism with a status in `statuses`, in pathway order."""
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
    return [CuratedRow(m, s, json.loads(s.arrows_json)) for s, m in session.exec(q).all()]


# ---- the arrow-code CSV


def _compact_side(smiles: str, keep: set[int]) -> str:
    """`smiles` with every plain H atom whose map is not in `keep` folded into its heavy
    neighbour's H count."""
    from ksmech.data import chem

    mol = Chem.RWMol(chem.parse_mapped(smiles))
    drop = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetAtomMapNum() in keep or atom.GetIsotope():
            continue
        nbrs = list(atom.GetNeighbors())
        if len(nbrs) != 1 or nbrs[0].GetAtomicNum() == 1:
            continue
        nbrs[0].SetNumExplicitHs(nbrs[0].GetNumExplicitHs() + 1)
        drop.append(atom.GetIdx())
    for idx in sorted(drop, reverse=True):
        mol.RemoveAtom(idx)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return chem.canonical_mapped_smiles(mol)


def compact_step(step: ElementaryStep) -> ElementaryStep | None:
    """The step with every H atom the arrows do not name removed (its neighbour's H count
    raised instead), re-canonicalized and re-verified with `build_step`; None when that
    fails. The parquet keeps explicit H; this is for the upstream CSV only."""
    from ksmech.data import chem
    from ksmech.data.validation import StepRejected, build_step

    keep = {int(m) for a in step.arrows for m in a.atoms}
    try:
        r = _compact_side(step.reactants_smiles, keep)
        p = _compact_side(step.products_smiles, keep)
        return build_step(
            step_id=step.step_id,
            source=step.source,
            rxn_smiles=f"{r}>>{p}",
            arrows=list(step.arrows),
            pathway_id=step.pathway_id,
            step_index=step.step_index,
            mechanism_class=step.mechanism_class,
            conditions=step.conditions,
        )
    except (StepRejected, chem.SmilesParseError, chem.SanitizeError) as exc:
        log.warning("cannot compact %s: %s", step.step_id, exc)
        return None


# ---- planning


def _code_of(arrows: list[dict], doc: dict) -> str:
    from ksmech.data.pmechdb import format_arrow_code
    from ksmech.data.schema import Arrow

    if doc.get("pmechdb_code"):
        return str(doc["pmechdb_code"])
    try:
        return format_arrow_code([Arrow.from_dict(a) for a in arrows])
    except (ValueError, KeyError, TypeError):
        return INVALID_ARROWS


def _raw_row(row: CuratedRow) -> dict:
    from ksmech.data.validation import make_step_id

    doc = row.doc
    arrows = [{k: a.get(k) for k in ARROW_FIELDS} for a in doc["dataset_arrows"]]
    rxn = f"{doc['reactants_mapped']}>>{doc['products_mapped']}"
    return {
        "step_id": make_step_id(USER_CURATED, f"{rxn} {_code_of(arrows, doc)}"),
        "source": USER_CURATED,
        "pathway_id": f"user_curated-path-{row.mechanism.id}",
        "step_index": int(row.step.order_group),
        "reactants_smiles": doc["reactants_mapped"],
        "products_smiles": doc["products_mapped"],
        "arrows": arrows,
        "mechanism_class": None,
        "conditions": row.mechanism.conditions,
        "resonance_step": False,
    }


def _provenance(doc: dict) -> dict:
    keys = ("predictor", "predictor_version", "spec_version", "physics_fingerprint")
    out = {k: doc.get(k) for k in keys}
    out |= {"job_id": doc.get("job_id"), "edge_id": doc.get("edge_id")}
    return out | {"edge_kind": doc.get("edge_kind")}


def _skip_reason(row: CuratedRow, pmechdb, emitted) -> tuple[str, str] | None:
    if row.doc.get("dataset_arrows") is None:
        return Code.ARROW_ENCODING_FAILED, str(row.doc.get("encode_reason") or "")
    if pmechdb is not None and row.key in pmechdb:
        return Code.DUPLICATE_OF_PMECHDB, str(pmechdb[row.key])
    if row.key in emitted:
        mid, k = emitted[row.key]
        return Code.DUPLICATE_IN_EXPORT, f"({mid}, {k})"
    return None


def _same_record(plan: _Plan, raw: dict) -> tuple[str, str] | None:
    """DUPLICATE_IN_EXPORT when a planned row has the same step id (the same record text): a
    second copy of a row the ingest rejects, which `_plan` does not mark as emitted."""
    first = plan.by_step_id.get(raw["step_id"])
    if first is None:
        return None
    return Code.DUPLICATE_IN_EXPORT, f"({first[0].mechanism.id}, {first[0].step.step_index})"


def _plan(ctx: JobContext, rows: list[CuratedRow], pmechdb: Mapping | None) -> _Plan:
    """The raw rows and report entries. A (reactant, product) pair counts as emitted only once
    one of its rows passes `user_curated.check_row`, so a row the ingest will reject never
    blocks a valid duplicate from a later mechanism; the rejected row is still written to the
    raw parquet, where the ingest reports it."""
    from ksmech.data.user_curated import check_row

    plan, emitted = _Plan(), {}
    for row in rows:
        plan.mechanisms.add(int(row.mechanism.id))
        entry = {**row.where, "step_id": None, "code": None, "detail": ""}
        entry["provenance"] = _provenance(row.doc)
        skip = _skip_reason(row, pmechdb, emitted)
        raw = _raw_row(row) if skip is None else None
        skip = skip or _same_record(plan, raw)
        if skip is None:
            plan.raw.append(raw)
            plan.by_step_id[raw["step_id"]] = (row, entry)
            entry["step_id"] = raw["step_id"]
            if check_row(raw) is None:
                emitted[row.key] = (row.mechanism.id, row.step.step_index)
        else:
            code, detail = skip
            entry.update(code=str(code), detail=detail)
            plan.skipped[str(code)] += 1
            ctx.skip(code, detail, **row.where)
        plan.entries.append(entry)
    return plan


# ---- files (atomic)


def _tmp_of(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def _write_raw(s: ServeSettings, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ksmech.data.schema import STEP_SCHEMA

    path = s.export_raw
    assert_writable(path, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(path)
    pq.write_table(pa.Table.from_pylist(rows, schema=STEP_SCHEMA), tmp)
    os.replace(tmp, path)
    return path


def _write_processed(s: ServeSettings, result: IngestResult) -> dict[str, Path]:
    """`write_outputs` into a scratch directory, then each file moved into
    `processed/user_curated/` (never the pmechdb directory)."""
    from ksmech.data.ingest import write_outputs

    out = s.processed / USER_CURATED
    scratch = s.processed / f".user_curated-{os.getpid()}-{secrets.token_hex(4)}"
    for path in (out, scratch):
        assert_writable(path, s)
    try:
        written = write_outputs(result, scratch)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        for kind, path in written.items():
            paths[kind] = out / path.name
            os.replace(path, paths[kind])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return paths


def _write_codes(ctx: JobContext, s: ServeSettings, result: IngestResult) -> tuple[Path, int]:
    """pmechdb_arrow_codes.csv from the compacted accepted steps; returns (path, fallbacks)."""
    from ksmech.data.pmechdb import write_records

    steps, fallbacks = [], 0
    for step in result.steps:
        compact = compact_step(step)
        if compact is None:
            fallbacks += 1
            ctx.skip(Code.COMPACT_FALLBACK, "written with explicit H", step_id=step.step_id)
        steps.append(compact or step)
    path = s.processed / USER_CURATED / CODES_FILE
    assert_writable(path, s)
    tmp = _tmp_of(path)
    write_records(steps, tmp)
    os.replace(tmp, path)
    return path, fallbacks


def _write_json(s: ServeSettings, path: Path, data: dict) -> Path:
    assert_writable(path, s)
    tmp = _tmp_of(path)
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---- ingest results and the ledger


def _apply_ingest(ctx: JobContext, plan: _Plan, result: IngestResult) -> list[tuple]:
    """Mark ingest rejections on the report entries; returns the written (row, entry)."""
    for rej in result.rejections:
        row, entry = plan.by_step_id.get(rej.step_id, (None, None))
        if entry is None:
            continue
        entry.update(code=str(rej.reason), detail=rej.detail)
        ctx.skip(str(rej.reason), rej.detail, **row.where)
    return [plan.by_step_id[s.step_id] for s in result.steps if s.step_id in plan.by_step_id]


def _ledger(row: CuratedRow, entry: dict) -> dict:
    return {"step_id": entry["step_id"], **row.where}


def _withdrawn(key: str, e: db.ExportedStep) -> dict:
    out = {"step_key": key, "step_id": e.step_id, "mechanism_id": e.mechanism_id}
    return out | {"step_index": e.step_index, "last_export_id": e.last_export_id}


def _update_ledger(session: Session, written: list[tuple], export_id: str) -> dict:
    """Upsert one ExportedStep per written key; count changed ids and withdrawn keys."""
    existing = {e.step_key: e for e in session.exec(select(db.ExportedStep)).all()}
    seen, changed = set(), 0
    for row, entry in written:
        key = json.dumps(list(row.key))
        seen.add(key)
        e = existing.get(key)
        if e is None:
            e = db.ExportedStep(step_key=key, first_export_id=export_id, **_ledger(row, entry))
        elif e.step_id != entry["step_id"]:
            changed += 1
        for name, value in _ledger(row, entry).items():
            setattr(e, name, value)
        e.last_export_id = export_id
        session.add(e)
    withdrawn = [_withdrawn(k, e) for k, e in sorted(existing.items()) if k not in seen]
    return {"changed_ids": changed, "withdrawn": len(withdrawn), "withdrawn_steps": withdrawn}


# ---- the job handler


def run_export(ctx: JobContext, request: dict, *, engine: Engine, pairs: PairsProvider) -> dict:
    """The `export` job: regenerate the user_curated export from the database (module doc).
    Returns the ExportReportOut fields (plus `warnings`, `statuses`, `withdrawn_steps`)."""
    from ksmech.data import user_curated

    req = ExportIn.model_validate(request or {})
    s = ctx.settings
    export_id, created = "x" + secrets.token_hex(8), db.utcnow()
    ctx.progress(phase="export", message="PMechDB pairs")
    pmechdb = pairs(ctx)
    warnings = []
    if pmechdb is None:
        warnings.append(f"{PMECHDB_UNAVAILABLE}: no PMechDB steps; no PMechDB de-duplication")
        log.warning("export %s: %s", export_id, warnings[-1])
    with db.session(engine) as ses:
        rows = step_rows(ses, req.statuses)
    ctx.progress(phase="export", done=0, total=len(rows), message="planning")
    plan = _plan(ctx, rows, pmechdb)
    raw_path = _write_raw(s, plan.raw)
    result = user_curated.load(raw_path)
    written = _apply_ingest(ctx, plan, result)
    paths = {"export_raw": raw_path, **_write_processed(s, result)}
    paths["pmechdb_arrow_codes"], fallbacks = _write_codes(ctx, s, result)
    if fallbacks:
        plan.skipped[str(Code.COMPACT_FALLBACK)] += fallbacks
        warnings.append(f"{Code.COMPACT_FALLBACK}: {fallbacks} CSV rows kept explicit H")
    paths["export_report"] = s.processed / USER_CURATED / REPORT_FILE
    report = {
        "export_id": export_id,
        "created_at": created.isoformat(),
        "spec_version": spec_version(s.spec_path),
        "paths": {k: str(v) for k, v in paths.items()},
        "n_mechanisms": len(plan.mechanisms),
        "n_steps_considered": len(rows),
        "n_steps_written": len(written),
        "skipped": dict(sorted(plan.skipped.items())),
        "ingest_rejections": dict(sorted(result.reason_counts().items())),
        "duplicates_pmechdb": plan.skipped[str(Code.DUPLICATE_OF_PMECHDB)],
        "duplicates_in_export": plan.skipped[str(Code.DUPLICATE_IN_EXPORT)],
        "steps": plan.entries,
        "sha256": file_sha256(paths["steps"]),
        "warnings": warnings,
        "statuses": list(req.statuses),
    }
    return _finish(ctx, engine, report, written, paths["export_report"])


def _finish(ctx: JobContext, engine: Engine, report: dict, written, report_path: Path) -> dict:
    """Ledger, export_report.json and the ExportRecord row, in one unit of work."""
    with db.session(engine) as ses:
        report |= _update_ledger(ses, written, report["export_id"])
        out = ExportReportOut.model_validate(report).model_dump(mode="json")
        out |= {k: report[k] for k in ("warnings", "statuses", "withdrawn_steps")}
        _write_json(ctx.settings, report_path, out)
        ses.add(
            db.ExportRecord(
                export_id=out["export_id"],
                created_at=datetime.fromisoformat(out["created_at"]),
                path=out["paths"]["steps"],
                sha256=out["sha256"],
                n_steps=out["n_steps_written"],
                report_json=json.dumps(out, sort_keys=True),
            )
        )
    n, written_n = out["n_steps_considered"], out["n_steps_written"]
    ctx.progress(phase="export", done=n, total=n, message=f"wrote {written_n} steps")
    log.info(
        "export %s: %d of %d steps written, skipped %s",
        out["export_id"],
        written_n,
        n,
        out["skipped"],
    )
    return out
