"""Deterministic train/val/test splits. Each one returns sorted step_id lists for a given seed.

Every strategy assigns whole *groups* of steps to one set:

- random by pathway: one group per pathway; a step with no pathway is its own group;
- leave one mechanism class out: the held-out class is the test set;
- scaffold: one group per Bemis-Murcko scaffold of the largest reactant.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from ksmech.data import chem
from ksmech.data.schema import ElementaryStep

Fractions = tuple[float, float, float]


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]
    excluded: list[str] = field(default_factory=list)  # steps the strategy cannot place

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test, "excluded": self.excluded}


def _check_fractions(fractions: Fractions) -> None:
    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must be 3 non-negative numbers summing to 1, got {fractions}")


def _pathway_key(step: ElementaryStep) -> str:
    return f"pathway:{step.pathway_id}" if step.pathway_id is not None else f"step:{step.step_id}"


def _assign_groups(groups: dict[str, list[str]], fractions: Fractions, seed: int) -> Split:
    """Shuffle groups with `seed`, then fill train, val, test greedily, putting large groups
    first (as in chemprop's balanced scaffold split) so they land in train, not test."""
    _check_fractions(fractions)
    keys = sorted(groups)
    keys = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    n = sum(len(v) for v in groups.values())
    big = max(fractions[1], fractions[2]) * n / 2
    order = [k for k in keys if len(groups[k]) > big] + [k for k in keys if len(groups[k]) <= big]
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in order:
        ids = groups[key]
        if len(out["train"]) + len(ids) <= fractions[0] * n:
            out["train"] += ids
        elif len(out["val"]) + len(ids) <= fractions[1] * n:
            out["val"] += ids
        else:
            out["test"] += ids
    return Split(sorted(out["train"]), sorted(out["val"]), sorted(out["test"]))


def split_by_pathway(
    steps: Sequence[ElementaryStep], fractions: Fractions = (0.8, 0.1, 0.1), seed: int = 0
) -> Split:
    """Random split that never puts two steps of one pathway into different sets."""
    groups: dict[str, list[str]] = defaultdict(list)
    for s in steps:
        groups[_pathway_key(s)].append(s.step_id)
    return _assign_groups(groups, fractions, seed)


def split_leave_one_class_out(
    steps: Sequence[ElementaryStep], held_out: str, val_fraction: float = 0.1, seed: int = 0
) -> Split:
    """Test = every step of class `held_out`. The other classes are split train/val by pathway.

    Excluded: steps with no class, and other-class steps that share a pathway with a held-out
    step (keeping them would leak that pathway into training).
    """
    unlabelled = [s.step_id for s in steps if s.mechanism_class is None]
    labelled = [s for s in steps if s.mechanism_class is not None]
    if not labelled:
        raise ValueError("no step has a mechanism_class; leave-one-class-out is impossible")
    if held_out not in {s.mechanism_class for s in labelled}:
        raise ValueError(f"class {held_out!r} not present")
    test = [s for s in labelled if s.mechanism_class == held_out]
    test_pathways = {s.pathway_id for s in test if s.pathway_id is not None}
    others = [s for s in labelled if s.mechanism_class != held_out]
    leaky = [s.step_id for s in others if s.pathway_id in test_pathways]
    rest = [s for s in others if s.pathway_id not in test_pathways]
    # test fraction 0: groups that overflow val spill into "test" and are folded back into val
    tv = split_by_pathway(rest, (1.0 - val_fraction, val_fraction, 0.0), seed)
    return Split(
        tv.train,
        sorted(tv.val + tv.test),
        sorted(s.step_id for s in test),
        sorted(unlabelled + leaky),
    )


def leave_one_class_out_splits(
    steps: Sequence[ElementaryStep], val_fraction: float = 0.1, seed: int = 0
) -> dict[str, Split]:
    classes = sorted({s.mechanism_class for s in steps if s.mechanism_class is not None})
    return {c: split_leave_one_class_out(steps, c, val_fraction, seed) for c in classes}


@cache
def largest_reactant_scaffold(reactants_smiles: str) -> str:
    """Bemis-Murcko scaffold (aromatic canonical SMILES, '' if acyclic) of the reactant fragment
    with the most heavy atoms. Ties go to the lexicographically smallest unmapped SMILES."""
    frags = chem.fragments(chem.parse_mapped(reactants_smiles))
    keyed = [(-chem.heavy_atom_count(f), chem.unmapped_canonical(f)) for f in frags]
    _, smiles = min(keyed)
    with rdBase.BlockLogs():
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))


def split_by_scaffold(
    steps: Sequence[ElementaryStep], fractions: Fractions = (0.8, 0.1, 0.1), seed: int = 0
) -> Split:
    """Group by the scaffold of the largest reactant. A step with a known pathway takes the
    scaffold of its pathway's first step, so a pathway never straddles two sets."""
    first: dict[str, ElementaryStep] = {}
    for s in steps:
        if s.pathway_id is None:
            continue
        cur = first.get(s.pathway_id)
        if cur is None or (s.step_index or 0, s.step_id) < (cur.step_index or 0, cur.step_id):
            first[s.pathway_id] = s
    groups: dict[str, list[str]] = defaultdict(list)
    for s in steps:
        anchor = first.get(s.pathway_id, s) if s.pathway_id is not None else s
        groups[largest_reactant_scaffold(anchor.reactants_smiles)].append(s.step_id)
    return _assign_groups(groups, fractions, seed)
