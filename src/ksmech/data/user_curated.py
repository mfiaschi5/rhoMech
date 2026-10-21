"""Loader of the steps curated in ksmech-serve (`Source.USER_CURATED`).

The serve export (`ksmech.serve.export`) writes the validated and corrected steps as a raw
parquet in the canonical schema (`STEP_SCHEMA`), by default `RAW_FILE`. This loader re-ingests
it through the same path as every other source: each row goes through `build_step`, so the
arrows are replayed onto the reactants and must reproduce the products, and every rejection
carries a reason code and is counted in the `IngestResult`.

Row checks before `build_step`:

- `source` must be `user_curated`, else `MALFORMED_RECORD`;
- every arrow must parse (`Arrow.from_dict`), else `ARROW_INVALID`;
- radical arrows are not exportable (`ARROW_INVALID`): PMechDB has no code for them and the
  replay does not implement them.

The step id is the row's own `step_id` (the export's content hash), or one derived from the
record text when the row has none. Curated steps are train-only (`splits.TRAIN_ONLY_SOURCES`).

`check_row` runs the same checks on one row without an `IngestResult`: the export uses it to
mark a (reactant, product) pair as emitted only once a row of it will pass the ingest.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq

from ksmech.data.ingest import IngestResult
from ksmech.data.pmechdb import format_arrow_code
from ksmech.data.schema import STEP_SCHEMA, Arrow, ElementaryStep, Source, SourceKind
from ksmech.data.validation import RejectReason, StepRejected, build_step, make_step_id

log = logging.getLogger(__name__)

RAW_FILE = Path("data/curation/export_raw/steps.parquet")
INVALID_ARROWS = "<invalid arrows>"


def resolve_input(raw: Path | None = None) -> Path:
    """The raw parquet: `raw` itself, `<raw>/steps.parquet` for a directory, or `RAW_FILE`.
    Raises FileNotFoundError when it does not exist."""
    path = RAW_FILE if raw is None else Path(raw)
    if path.is_dir():
        path = path / "steps.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no user_curated export at {path} (run the serve export first)")
    return path


def load(raw: Path | None = None) -> IngestResult:
    """Every row of the export parquet, validated by `build_step`. FileNotFoundError when the
    file is missing (the CLI returns 2)."""
    path = resolve_input(raw)
    result = IngestResult(Source.USER_CURATED, inputs=[str(path)])
    log.info("reading %s", path)
    for row in pq.read_table(path, schema=STEP_SCHEMA).to_pylist():
        _add_row(result, row)
    return result


def _arrows(row: dict) -> list[Arrow]:
    """The row's arrows. Raises StepRejected(ARROW_INVALID)."""
    try:
        arrows = [Arrow.from_dict(a) for a in row["arrows"] or []]
    except (ValueError, KeyError, TypeError) as exc:
        raise StepRejected(RejectReason.ARROW_INVALID, f"{type(exc).__name__}: {exc}") from exc
    if any(a.source_kind is SourceKind.RADICAL for a in arrows):
        raise StepRejected(RejectReason.ARROW_INVALID, "radical arrows are not exportable")
    return arrows


def _raise(exc: StepRejected) -> ElementaryStep:
    raise exc


def _record(row: dict) -> tuple[str, str, list[Arrow] | StepRejected]:
    """(step id, raw record text, the arrows or the row's rejection before `build_step`)."""
    rxn = f"{row['reactants_smiles']}>>{row['products_smiles']}"
    error: StepRejected | None = None
    try:
        arrows = _arrows(row)
        code = format_arrow_code(arrows)
    except StepRejected as exc:
        arrows, code, error = [], INVALID_ARROWS, exc
    raw_text = f"{rxn} {code}"
    step_id = row.get("step_id") or make_step_id(Source.USER_CURATED, raw_text)
    if row.get("source") != str(Source.USER_CURATED):
        detail = f"source {row.get('source')!r}, expected {str(Source.USER_CURATED)!r}"
        error = StepRejected(RejectReason.MALFORMED_RECORD, detail)
    return step_id, raw_text, arrows if error is None else error


def _make(row: dict, step_id: str, arrows: list[Arrow]) -> partial:
    """The `build_step` call of one row (arrows already checked)."""
    return partial(
        build_step,
        step_id=step_id,
        source=Source.USER_CURATED,
        rxn_smiles=f"{row['reactants_smiles']}>>{row['products_smiles']}",
        arrows=arrows,
        pathway_id=row.get("pathway_id"),
        step_index=row.get("step_index"),
        mechanism_class=row.get("mechanism_class"),
        conditions=row.get("conditions"),
    )


def check_row(row: dict) -> StepRejected | None:
    """The rejection `load` gives this row on its own, or None when it would be accepted.

    The same checks as `load` (source, arrows, arrow code, `build_step`), so the serve export can
    tell a row that will be written from one the ingest rejects. DUPLICATE_STEP is not checked:
    it depends on the other rows of the file.
    """
    step_id, _, checked = _record(row)
    if isinstance(checked, StepRejected):
        return checked
    try:
        _make(row, step_id, checked)()
    except StepRejected as exc:
        return exc
    return None


def _add_row(result: IngestResult, row: dict) -> None:
    """Route one row through `result.add`: the step, or its counted rejection."""
    step_id, raw_text, checked = _record(row)
    if isinstance(checked, StepRejected):
        result.add(step_id, raw_text, partial(_raise, checked))
    else:
        result.add(step_id, raw_text, _make(row, step_id, checked))
