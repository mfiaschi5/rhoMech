"""PMechDB loader. The raw files come from the PMechRP data release.

Raw format, as inspected and documented in docs/data_schema.md: a headerless one-column CSV.
Each line is one double-quoted field, `"<reactants>>products> <arrow code>"`. The arrow code is
`;`-separated arrows of the form `source=sink`, where each side is one atom-map number (an atom)
or two comma-separated map numbers (a bond).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from functools import partial
from pathlib import Path

from ksmech.data.ingest import IngestResult
from ksmech.data.pathways import assign_pathways
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, Source, SourceKind
from ksmech.data.validation import RejectReason, StepRejected, build_step, make_step_id

log = logging.getLogger(__name__)

RAW_DIR_CANDIDATES = (Path("data/raw/pmechdb"), Path("data/raw/pmechrp"))
# The curated ("mc_only") set. Every fold holds the same 13,104 records split differently, so
# fold0 train+val+test is the whole set.
CURATED_FILES = tuple(
    f"two_stage_models/mc_only_splits/fold0/raw/{split}.txt" for split in ("train", "val", "test")
)
_SOURCE_KINDS = {1: SourceKind.LONE_PAIR, 2: SourceKind.BOND}
_SINK_KINDS = {1: SinkKind.ATOM, 2: SinkKind.BOND}


def resolve_inputs(raw: Path | None = None) -> list[Path]:
    """A single file is read as is. A directory must contain the PMechRP curated files."""
    if raw is None:
        found = [d for d in RAW_DIR_CANDIDATES if d.is_dir()]
        if not found:
            raise FileNotFoundError(f"no PMechDB raw dir among {RAW_DIR_CANDIDATES}")
        raw = found[0]
    raw = Path(raw)
    if raw.is_file():
        return [raw]
    files = [raw / f for f in CURATED_FILES]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"PMechDB raw files missing: {missing}")
    return files


def _atoms(text: str, token: str) -> tuple[int, ...]:
    parts = text.split(",")
    if not all(p.isdigit() for p in parts):
        raise StepRejected(RejectReason.ARROW_PARSE, f"non-integer atom in arrow {token!r}")
    return tuple(int(p) for p in parts)


def parse_arrow(token: str) -> Arrow:
    """`a=b`, `a,b=c` or `a,b=c,d`. One atom is a lone pair (source) or an atom (sink)."""
    src_text, sep, snk_text = token.partition("=")
    if not sep or "=" in snk_text:
        raise StepRejected(RejectReason.ARROW_PARSE, f"arrow needs exactly one '=': {token!r}")
    src, snk = _atoms(src_text, token), _atoms(snk_text, token)
    if len(src) not in _SOURCE_KINDS or len(snk) not in _SINK_KINDS:
        raise StepRejected(RejectReason.ARROW_PARSE, f"arrow end has >2 atoms: {token!r}")
    try:
        return Arrow(_SOURCE_KINDS[len(src)], src, _SINK_KINDS[len(snk)], snk)
    except ValueError as exc:
        raise StepRejected(RejectReason.ARROW_INVALID, f"{token!r}: {exc}") from exc


def strip_stray_comma(code: str) -> tuple[str, bool]:
    """Drop the stray trailing ',' that 1,155 raw records carry (e.g. `10=20,`)."""
    stripped = code.rstrip(",")
    return stripped, stripped != code


def parse_arrow_code(code: str) -> list[Arrow]:
    code, _ = strip_stray_comma(code.strip())
    if not code:
        raise StepRejected(RejectReason.ARROW_PARSE, "empty arrow code")
    return [parse_arrow(tok) for tok in code.split(";")]


def parse_record(raw: str, step_id: str) -> ElementaryStep:
    fields = raw.split()
    if len(fields) != 2:
        raise StepRejected(RejectReason.MALFORMED_RECORD, f"expected 2 fields, got {len(fields)}")
    rxn, code = fields
    arrows = parse_arrow_code(code)
    return build_step(step_id=step_id, source=Source.PMECHDB, rxn_smiles=rxn, arrows=arrows)


def read_records(path: Path) -> Iterator[list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.reader(fh)


def load(raw: Path | None = None) -> IngestResult:
    inputs = resolve_inputs(raw)
    result = IngestResult(Source.PMECHDB, inputs=[str(p) for p in inputs])
    for path in inputs:
        log.info("reading %s", path)
        for row in read_records(path):
            if not row:
                result.notes["blank_line_skipped"] += 1
                continue
            text = ",".join(row).strip()  # one field expected; rejoin so nothing is dropped
            step_id = make_step_id(Source.PMECHDB, text)
            result.notes["stray_trailing_comma"] += strip_stray_comma(text)[1]
            if len(row) == 1:
                result.add(step_id, text, partial(parse_record, text, step_id))
            else:
                result.add(step_id, text, partial(_malformed, len(row)))
    result.steps, report = assign_pathways(result.steps)  # the release has no pathway ids
    pathways = result.sections["pathways"] = report.as_dict()
    log.info(
        "pathways: %d groups hold %d of %d steps",
        pathways["n_pathways"],
        pathways["n_steps_in_pathways"],
        pathways["n_steps"],
    )
    return result


def _malformed(n_fields: int) -> ElementaryStep:
    raise StepRejected(RejectReason.MALFORMED_RECORD, f"{n_fields} CSV fields, expected 1")
