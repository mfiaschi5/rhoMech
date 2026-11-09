"""Pathway benchmarks for the search towards a product (reaction-network plan, items G and 9).

A `Case` is a start pool, a target, and what the benchmark records about the way between:

- ReactMech chains (`data.pathways`, the sidecar of the ID_CRM and OOD mechanism files): every
  chain whose steps are all in the source's steps parquet, the start the first step's
  reactants, the target the last step's products, `recorded` the pool keys along the chain
  (start first), `part` the split part of its steps ('mixed' when they straddle parts). The
  target keeps its atom maps only when every boundary of the chain kept the same numbers
  (`maps_to_next` 'same'), so a mapped target is always on the start pool's numbering;
- the PMechRP human benchmark (`data.human_benchmark`, evaluation only): the reactants, the final
  product without its maps, and the target of each step (`step_targets`).

A chain that is not ingested whole is counted as CHAIN_NOT_INGESTED, a benchmark row whose start
or final product does not parse as BENCHMARK_UNPARSED.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase

from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, steps_path

CHAIN_NOT_INGESTED, BENCHMARK_UNPARSED = "CHAIN_NOT_INGESTED", "BENCHMARK_UNPARSED"


@dataclass(frozen=True)
class Case:
    case_id: str
    start: str
    target: str
    recorded: tuple[str, ...] | None = None  # pool keys along the recorded pathway, start first
    step_targets: tuple[str | None, ...] = ()  # the target of each recorded step
    part: str = ""  # the split part of the recorded steps


def unmapped(smiles: str) -> str | None:
    """The SMILES without atom maps, None when RDKit cannot read it."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _chains(table_rows: list[dict]) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = defaultdict(list)
    for row in table_rows:
        chains[row["pathway_id"]].append(row)
    return {k: sorted(v, key=lambda r: r["step_index"]) for k, v in sorted(chains.items())}


def _part(step_ids: list[str], parts: dict[str, str]) -> str:
    found = {parts.get(s, "none") for s in step_ids}
    return found.pop() if len(found) == 1 else "mixed"


def reactmech_cases(
    processed_dir: Path = DEFAULT_PROCESSED_DIR, split: str = "molecule",
    counts: Counter | None = None,
) -> list[Case]:  # fmt: skip
    """The ReactMech chains ingested whole, as cases (module docstring)."""
    from ksmech.arrows.steps import load_split_file, splits_path
    from ksmech.data.pathways import load_pathways
    from ksmech.eval.siblings import parts_of
    from ksmech.network.pools import pool_state

    counts = Counter() if counts is None else counts
    steps = {s.step_id: s for s in read_steps(steps_path("reactmech", processed_dir))}
    split_file = splits_path("reactmech", split, processed_dir)
    parts = parts_of(load_split_file(split_file)) if split_file.is_file() else {}
    out = []
    for pid, rows in _chains(load_pathways(processed_dir).to_pylist()).items():
        ids = [r["step_id"] for r in rows]
        if any(i not in steps for i in ids):
            counts[CHAIN_NOT_INGESTED] += 1
            continue
        last = steps[ids[-1]].products_smiles
        same = all(r["maps_to_next"] == "same" for r in rows[:-1])
        target = last if same else unmapped(last)
        start = steps[ids[0]].reactants_smiles
        recorded = (pool_state(start).key, *(pool_state(steps[i].products_smiles).key for i in ids))
        out.append(Case(pid, start, target, recorded, (), _part(ids, parts)))
    return out


def human_benchmark_cases(
    processed_dir: Path = DEFAULT_PROCESSED_DIR, counts: Counter | None = None
) -> list[Case]:
    """The human benchmark's pathways as cases (module docstring), read from
    `<processed_dir>/human_benchmark/` (`data.human_benchmark`)."""
    from ksmech.data import human_benchmark as hb

    counts = Counter() if counts is None else counts
    out = []
    for p in hb.load_pathways(Path(processed_dir) / hb.NAME):
        target = unmapped(p.final_product)
        if target is None or unmapped(p.reactants) is None:
            counts[BENCHMARK_UNPARSED] += 1
            continue
        out.append(Case(p.pathway_id, p.reactants, target, None, tuple(p.targets), "benchmark"))
    return out
