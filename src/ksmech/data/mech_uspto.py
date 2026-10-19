"""mech-USPTO-31K loader. untested_on_real_data=True

mech-USPTO-31K: Chen, Babazade, Kim, Han, Jung, "A large-scale reaction dataset of mechanistic
pathways of organic reactions", Sci. Data 11, 863 (2024), doi:10.1038/s41597-024-03709-y.
Data: doi:10.6084/m9.figshare.24797220. Code: https://github.com/snu-micc/MechFinder.

This is the held-out benchmark: never train on it. No raw file exists in data/raw/mech_uspto/,
so this module follows the *published* description and has not been run on the real file:

- The file is `mech-USPTO-31k.csv`: comma-separated with a header. MechFinder's Demo.ipynb
  writes it with the columns `original_reactions`, `updated_reaction`, `mechanistic_class` and
  `mechanistic_label`. The Figshare header itself is unconfirmed, so `read_mechanisms` checks the
  header and fails with the columns it actually found.
- There is one row per *reaction*, not per step. `mechanistic_label` is a Python-literal list of
  (source, sink) pairs over the whole mechanism, e.g.
  `[(26, 1), ([1, 9], 9), (9, [9, 1]), ([1, 101], 101)]`. An int is a lone pair (as source) or
  an atom (as sink), and `[a, b]` is a bond. A float `a.k` is the k-th implicit hydrogen on the
  atom mapped `a`. Map numbers in the hundreds appear to be added reagents, so the arrows are
  read against `updated_reaction` (an inference, unconfirmed).

The label carries no elementary-step boundaries and refers to implicit hydrogens. Splitting it
into balanced, fully mapped `ElementaryStep`s means replaying the arrows on the bond-electron
matrix, and that engine lands with `ksmech.arrows`. Until then, `read_mechanisms` parses and
validates the file, and `load` raises NotImplementedError.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

from ksmech.data.ingest import IngestResult
from ksmech.data.schema import Arrow, SinkKind, Source, SourceKind
from ksmech.data.validation import RejectReason, StepRejected, make_step_id

UNTESTED_ON_REAL_DATA = True
RAW_FILE = Path("data/raw/mech_uspto/mech-USPTO-31k.csv")
REQUIRED_COLUMNS = ("updated_reaction", "mechanistic_class", "mechanistic_label")


@dataclass(frozen=True)
class HydrogenRef:
    """The k-th (1-based) implicit hydrogen on the atom with map number `parent_map`."""

    parent_map: int
    k: int


AtomRef = int | HydrogenRef
End = AtomRef | tuple[AtomRef, AtomRef]


@dataclass(frozen=True)
class LabelArrow:
    source: End
    sink: End


@dataclass(frozen=True)
class Mechanism:
    """One mech-USPTO row: a whole reaction plus its flat arrow list."""

    record_id: str
    rxn_smiles: str
    mechanistic_class: str
    arrows: list[LabelArrow]


def _atom_ref(x: object) -> AtomRef:
    if isinstance(x, bool) or not isinstance(x, int | float):
        raise StepRejected(RejectReason.ARROW_PARSE, f"atom ref must be int/float, got {x!r}")
    if isinstance(x, int):
        return x
    parent, _, k = repr(x).partition(".")
    if not (parent.isdigit() and k.isdigit() and int(k) >= 1):
        raise StepRejected(RejectReason.ARROW_PARSE, f"bad hydrogen ref {x!r}")
    return HydrogenRef(int(parent), int(k))


def _end(x: object) -> End:
    if isinstance(x, list | tuple):
        if len(x) != 2:
            raise StepRejected(RejectReason.ARROW_PARSE, f"bond needs 2 atoms, got {x!r}")
        return (_atom_ref(x[0]), _atom_ref(x[1]))
    return _atom_ref(x)


def parse_mechanistic_label(text: str) -> list[LabelArrow]:
    try:
        pairs = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise StepRejected(
            RejectReason.ARROW_PARSE, f"not a Python literal: {text[:80]!r}"
        ) from exc
    if not isinstance(pairs, list) or not pairs:
        raise StepRejected(RejectReason.ARROW_PARSE, "label must be a non-empty list")
    if not all(isinstance(p, list | tuple) and len(p) == 2 for p in pairs):
        raise StepRejected(RejectReason.ARROW_PARSE, "every arrow must be a (source, sink) pair")
    return [LabelArrow(_end(src), _end(snk)) for src, snk in pairs]


def to_arrow(label: LabelArrow) -> Arrow:
    """Convert a label arrow without hydrogen references into our `Arrow`."""
    ends = [label.source, label.sink]
    flat = [a for e in ends for a in (e if isinstance(e, tuple) else (e,))]
    if any(isinstance(a, HydrogenRef) for a in flat):
        raise ValueError(f"{label} refers to an implicit H; needs explicit-H mapping first")
    src_bond, snk_bond = (isinstance(e, tuple) for e in ends)
    return Arrow(
        SourceKind.BOND if src_bond else SourceKind.LONE_PAIR,
        label.source if src_bond else (label.source,),
        SinkKind.BOND if snk_bond else SinkKind.ATOM,
        label.sink if snk_bond else (label.sink,),
    )


def read_mechanisms(path: Path = RAW_FILE) -> tuple[list[Mechanism], IngestResult]:
    """Parse every row. Bad rows are recorded as rejections in the returned IngestResult."""
    result = IngestResult(Source.MECH_USPTO, inputs=[str(path)])
    mechanisms: list[Mechanism] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; found {reader.fieldnames}")
        for row in reader:
            raw = row["updated_reaction"] + " " + row["mechanistic_label"]
            rid = make_step_id(Source.MECH_USPTO, raw)
            result.n_records += 1
            try:
                arrows = parse_mechanistic_label(row["mechanistic_label"])
            except StepRejected as exc:
                result.reject(rid, raw, exc)
                continue
            mechanisms.append(
                Mechanism(rid, row["updated_reaction"], row["mechanistic_class"], arrows)
            )
    return mechanisms, result


def load(raw: Path | None = None) -> IngestResult:
    raise NotImplementedError(
        "mech-USPTO-31K stores whole-reaction arrow lists with implicit-H references and no "
        "step boundaries; turning them into ElementaryStep needs arrow replay (ksmech.arrows, a "
        "later phase). Use mech_uspto.read_mechanisms() to parse the file meanwhile."
    )
