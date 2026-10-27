"""The PMechDB arrow-code writer, the inverse of `data.pmechdb.parse_arrow_code` (adapter).

Raw format (docs/data_schema.md): a headerless one-column CSV, each line one double-quoted
field `"<reactants>>products> <arrow code>"`; the arrow code is `;`-separated `source=sink`
arrows, each side one atom-map number (an atom) or two comma-separated map numbers (a bond).
Browse shows the code of every step and the (disabled) export wrote curated steps in it. The
writer lives in serve: upstream's data layer only reads the format.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ksmech.data.schema import Arrow, ElementaryStep

log = logging.getLogger(__name__)


def _atoms_text(atoms: Iterable[int]) -> str:
    return ",".join(str(a) for a in atoms)


def format_arrow(arrow: Arrow) -> str:
    """The `source=sink` token of one arrow, the inverse of `parse_arrow`.

    A lone pair aimed at the bond it forms (`a=a,b`) is written in the dataset's own form `a=b`,
    which parses back to the equivalent lone-pair-to-atom arrow (same destination); every other
    arrow round-trips exactly. Radical arrows have no PMechDB code and raise ValueError.
    """
    from ksmech.data.schema import SinkKind, SourceKind

    if arrow.source_kind is SourceKind.RADICAL:
        raise ValueError(f"radical arrows have no PMechDB arrow code: {arrow}")
    sink = arrow.sink_atoms
    if arrow.source_kind is SourceKind.LONE_PAIR and arrow.sink_kind is SinkKind.BOND:
        sink = tuple(a for a in arrow.sink_atoms if a not in arrow.source_atoms)
    return f"{_atoms_text(arrow.source_atoms)}={_atoms_text(sink)}"


def format_arrow_code(arrows: Sequence[Arrow]) -> str:
    """Arrows joined by `;` in order, with no trailing comma."""
    return ";".join(format_arrow(a) for a in arrows)


def format_record(step: ElementaryStep) -> str:
    """One raw record, `<reactants>>products> <arrow code>`, as `data.pmechdb.load` reads it."""
    return f"{step.reactants_smiles}>>{step.products_smiles} {format_arrow_code(step.arrows)}"


def write_records(steps: Iterable[ElementaryStep], path: Path) -> Path:
    """Write `steps` as a headerless one-column CSV, every field double-quoted (`load` input).

    Raises ValueError on a step with a radical arrow before anything is written.
    """
    rows = [format_record(s) for s in steps]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerows([r] for r in rows)
    log.info("wrote %d PMechDB records to %s", len(rows), path)
    return path
