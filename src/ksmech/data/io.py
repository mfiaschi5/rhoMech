"""Parquet persistence for `ElementaryStep` tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ksmech.data.resonance import training_steps
from ksmech.data.schema import STEP_SCHEMA, ElementaryStep, Source

DEFAULT_PROCESSED_DIR = Path("data/processed")


def source_dir(source: Source | str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return Path(processed_dir) / str(Source(source))


def steps_path(source: Source | str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return source_dir(source, processed_dir) / "steps.parquet"


def steps_to_table(steps: Iterable[ElementaryStep]) -> pa.Table:
    return pa.Table.from_pylist([s.to_dict() for s in steps], schema=STEP_SCHEMA)


def write_steps(steps: Iterable[ElementaryStep], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(steps_to_table(steps), path)
    return path


RESONANCE_PAIR_SCHEMA = pa.schema(
    [("step_id", pa.string()), ("reactants_smiles", pa.string()), ("products_smiles", pa.string())]
)


def write_resonance_pairs(steps: Iterable[ElementaryStep], path: Path) -> Path:
    """The resonance steps inside the closed-shell domain (two sides with one sigma key), as
    (step id, reactant drawing, product drawing): equivalence examples, kept out of the
    training target."""
    rows = [
        {
            "step_id": s.step_id,
            "reactants_smiles": s.reactants_smiles,
            "products_smiles": s.products_smiles,
        }
        for s in steps
        if s.resonance_step and s.domain_exclusion is None
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=RESONANCE_PAIR_SCHEMA), path)
    return path


def read_steps(path: Path) -> list[ElementaryStep]:
    """Every accepted step, resonance steps included (flagged `resonance_step`)."""
    table = pq.read_table(path, schema=STEP_SCHEMA)
    return [ElementaryStep.from_dict(row) for row in table.to_pylist()]


def read_training_steps(path: Path) -> list[ElementaryStep]:
    """The training target: resonance steps dropped, `step_index` renumbered per pathway."""
    return training_steps(read_steps(path))
