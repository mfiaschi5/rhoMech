"""Deterministic train/val/test splits. Each one returns sorted step_id lists for a given seed.

Every strategy assigns whole *groups* of steps to one set:

- random by pathway: one group per pathway; a step with no pathway is its own group;
- leave one mechanism class out: the held-out class is the test set;
- scaffold: one group per Bemis-Murcko scaffold of the largest reactant;
- molecule: steps that share any non-trivial molecule, on either side, form one group.

`default_split(source)` names the strategy each source uses unless told otherwise.

Frozen splits (`frozen`, step id -> part, from a `split_frozen_<strategy>.json` snapshot) keep
every existing step in its part when steps are added: a group holding frozen steps keeps their
part, a group whose frozen steps sit in two parts keeps them and sends its new steps to
`conflict` (SPLIT_CONFLICT, counted), and a group with no frozen step is placed by
sha256(seed, group key) against the fractions instead of the global permutation, which would
reshuffle val and test whenever one group is added.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from ksmech.data import chem
from ksmech.data.schema import ElementaryStep, Source

log = logging.getLogger(__name__)

Fractions = tuple[float, float, float]


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]
    excluded: list[str] = field(default_factory=list)  # steps the strategy cannot place
    conflict: list[str] = field(default_factory=list)  # new steps of a group frozen in two parts

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test, "excluded": self.excluded}


Frozen = Mapping[str, str]  # step id -> "train" | "val" | "test"


def _check_fractions(fractions: Fractions) -> None:
    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must be 3 non-negative numbers summing to 1, got {fractions}")


def _pathway_key(step: ElementaryStep) -> str:
    return f"pathway:{step.pathway_id}" if step.pathway_id is not None else f"step:{step.step_id}"


def _hash_part(key: str, fractions: Fractions, seed: int) -> str:
    """The part of a group no frozen step pins: sha256(seed, group key) read as a uniform
    number in [0, 1) against the fractions, so placing one group moves no other."""
    u = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:15], 16) / 16**15
    if u < fractions[0]:
        return "train"
    return "val" if u < fractions[0] + fractions[1] else "test"


def _assign_frozen(groups: dict[str, list[str]], fractions: Fractions, seed: int, frozen: Frozen):
    """The frozen path of `_assign_groups` (module docstring)."""
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    conflict: list[str] = []
    for key in sorted(groups):
        ids = groups[key]
        parts = {frozen[i] for i in ids if i in frozen}
        if len(parts) > 1:
            for i in ids:
                (out[frozen[i]] if i in frozen else conflict).append(i)
            continue
        out[parts.pop() if parts else _hash_part(key, fractions, seed)] += ids
    if conflict:
        log.warning("%d new steps join groups frozen in two parts: SPLIT_CONFLICT", len(conflict))
    parts = (sorted(out[p]) for p in ("train", "val", "test"))
    return Split(*parts, conflict=sorted(conflict))


def _assign_groups(
    groups: dict[str, list[str]], fractions: Fractions, seed: int, frozen: Frozen | None = None
) -> Split:
    """Shuffle groups with `seed`, then fill train, val, test greedily, putting large groups
    first (as in chemprop's balanced scaffold split) so they land in train, not test. With
    `frozen`, existing steps keep their parts instead (module docstring).

    A group larger than the whole train target still goes to train (the target is overshot,
    with a warning): placing it in val or test would make a degenerate split."""
    _check_fractions(fractions)
    if frozen is not None:
        return _assign_frozen(groups, fractions, seed, frozen)
    keys = sorted(groups)
    keys = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    n = sum(len(v) for v in groups.values())
    big = max(fractions[1], fractions[2]) * n / 2
    order = [k for k in keys if len(groups[k]) > big] + [k for k in keys if len(groups[k]) <= big]
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in order:
        ids = groups[key]
        if len(ids) > fractions[0] * n:
            log.warning("group of %d steps > train target %.0f", len(ids), fractions[0] * n)
            out["train"] += ids
        elif len(out["train"]) + len(ids) <= fractions[0] * n:
            out["train"] += ids
        elif len(out["val"]) + len(ids) <= fractions[1] * n:
            out["val"] += ids
        else:
            out["test"] += ids
    return Split(sorted(out["train"]), sorted(out["val"]), sorted(out["test"]))


def split_by_pathway(
    steps: Sequence[ElementaryStep],
    fractions: Fractions = (0.8, 0.1, 0.1),
    seed: int = 0,
    frozen: Frozen | None = None,
) -> Split:
    """Random split that never puts two steps of one pathway into different sets."""
    groups: dict[str, list[str]] = defaultdict(list)
    for s in steps:
        groups[_pathway_key(s)].append(s.step_id)
    return _assign_groups(groups, fractions, seed, frozen)


def split_leave_one_class_out(
    steps: Sequence[ElementaryStep],
    held_out: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    frozen: Frozen | None = None,
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
    tv = split_by_pathway(rest, (1.0 - val_fraction, val_fraction, 0.0), seed, frozen)
    return Split(
        tv.train,
        sorted(tv.val + tv.test),
        sorted(s.step_id for s in test),
        sorted(unlabelled + leaky),
        tv.conflict,
    )


def leave_one_class_out_splits(
    steps: Sequence[ElementaryStep], val_fraction: float = 0.1, seed: int = 0
) -> dict[str, Split]:
    classes = sorted({s.mechanism_class for s in steps if s.mechanism_class is not None})
    return {c: split_leave_one_class_out(steps, c, val_fraction, seed) for c in classes}


def split_leave_one_class_out_all(
    steps: Sequence[ElementaryStep],
    held_out: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    frozen: Frozen | None = None,
) -> Split:
    """Test = every step of class `held_out`; every other step, unlabelled ones included, is
    split train/val by molecule (`split_by_molecule`), so val is an in-distribution set.

    Excluded: the steps that share a non-trivial molecule or a pathway with a held-out step
    (they would leak the held-out reactions into training)."""
    if held_out not in {s.mechanism_class for s in steps}:
        raise ValueError(f"class {held_out!r} not present")
    test = [s for s in steps if s.mechanism_class == held_out]
    tokens = {f"pathway:{s.pathway_id}" for s in test if s.pathway_id is not None}
    for s in test:
        tokens |= step_molecules(s.reactants_smiles, s.products_smiles)

    def leaks(s: ElementaryStep) -> bool:
        return f"pathway:{s.pathway_id}" in tokens or bool(
            tokens & step_molecules(s.reactants_smiles, s.products_smiles)
        )

    others = [s for s in steps if s.mechanism_class != held_out]
    rest = [s for s in others if not leaks(s)]
    # test fraction 0: groups that overflow val spill into "test" and are folded back into val
    tv = split_by_molecule(rest, (1.0 - val_fraction, val_fraction, 0.0), seed, frozen)
    leaky = sorted({s.step_id for s in others} - {s.step_id for s in rest})
    test_ids = sorted(s.step_id for s in test)
    return Split(tv.train, sorted(tv.val + tv.test), test_ids, leaky, tv.conflict)


def leave_one_class_out_all_splits(
    steps: Sequence[ElementaryStep], val_fraction: float = 0.1, seed: int = 0
) -> dict[str, Split]:
    classes = sorted({s.mechanism_class for s in steps if s.mechanism_class is not None})
    return {c: split_leave_one_class_out_all(steps, c, val_fraction, seed) for c in classes}


@cache
def step_molecules(reactants_smiles: str, products_smiles: str) -> frozenset[str]:
    """Non-trivial map-free molecules (more than 3 heavy atoms) on either side of a step."""
    return frozenset(
        chem.nontrivial_species(reactants_smiles) + chem.nontrivial_species(products_smiles)
    )


def molecule_groups(steps: Sequence[ElementaryStep]) -> dict[str, list[str]]:
    """Connected components of the step-molecule graph, keyed by their smallest step_id. Steps of
    one pathway are joined too, so a known pathway never straddles two sets."""
    parent = {s.step_id: s.step_id for s in steps}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}  # molecule or pathway token -> first step that holds it
    for s in steps:
        tokens = set(step_molecules(s.reactants_smiles, s.products_smiles))
        if s.pathway_id is not None:
            tokens.add(f"pathway:{s.pathway_id}")
        for tok in sorted(tokens):
            a, b = find(owner.setdefault(tok, s.step_id)), find(s.step_id)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in parent:
        groups[find(sid)].append(sid)
    return dict(groups)


def split_by_molecule(
    steps: Sequence[ElementaryStep],
    fractions: Fractions = (0.8, 0.1, 0.1),
    seed: int = 0,
    frozen: Frozen | None = None,
) -> Split:
    """No non-trivial molecule (maps stripped, stereo kept, more than 3 heavy atoms), as reactant
    or as product, appears in more than one of train/val/test. Products count too: otherwise
    step A (X -> Y) and step B (Y -> Z) of one reaction share no reactant and can be split.

    Common reagents (HSO4-, AcO-, TsO-, Et3NH+, pyridine...) chain most of PMechDB into one
    group, which lands in train; val and test are the steps that share no molecule with it."""
    return _assign_groups(molecule_groups(steps), fractions, seed, frozen)


def _flower(steps: Sequence[ElementaryStep], suffix: str) -> list[ElementaryStep]:
    if any(s.flower_suffix is None for s in steps):
        raise ValueError("a FlowER split needs flower_suffix on every step (rmechdb only)")
    return [s for s in steps if s.flower_suffix == suffix]


def leak_guarded(
    candidates: Sequence[ElementaryStep], held: Sequence[ElementaryStep]
) -> tuple[list[str], list[str]]:
    """(kept ids, LEAK_GUARD ids): a candidate training step that shares a non-trivial
    molecule (`step_molecules`) with a held-out step leaves training."""
    seen = frozenset().union(*(step_molecules(s.reactants_smiles, s.products_smiles) for s in held))
    keep, leak = [], []
    for s in candidates:
        mols = step_molecules(s.reactants_smiles, s.products_smiles)
        (leak if mols & seen else keep).append(s.step_id)
    return keep, leak


def split_rc_molecule(
    steps: Sequence[ElementaryStep],
    fractions: Fractions = (0.8, 0.1, 0.1),
    seed: int = 0,
    frozen: Frozen | None = None,
) -> Split:
    """rmechdb's default (open-shell plan P8 step 4): the molecule split over the RC steps;
    every RS step is training data (RS cannot be split by molecule: almost all of it is one
    group), except the RS steps that share a non-trivial molecule with an RC val or test step,
    excluded as LEAK_GUARD (the split's `excluded`). `frozen` pins the RC parts."""
    rc, rs = _flower(steps, "RC"), _flower(steps, "RS")
    sp = split_by_molecule(rc, fractions, seed, frozen)
    held_ids = set(sp.val) | set(sp.test)
    keep, leak = leak_guarded(rs, [s for s in rc if s.step_id in held_ids])
    train = sorted(sp.train + keep)
    return Split(train, sp.val, sp.test, sorted(leak), sp.conflict)


def split_flower_given(
    steps: Sequence[ElementaryStep],
    val_fraction: float = 0.1,
    seed: int = 0,
    frozen: Frozen | None = None,
) -> Split:
    """FlowER's own train/test (`flower_part`), RC and RS alike, leaky by construction: for
    the RMechRP comparison only. Test is every step FlowER's test holds (a reaction in both
    parts, `train+test`, included); val is `val_fraction` of the molecule groups of the
    FlowER-train RC steps; the other FlowER-train steps are train."""
    _flower(steps, "RC")
    test = [s for s in steps if s.flower_part and "test" in s.flower_part.split("+")]
    rest = [s for s in steps if s not in test]
    rc_train = [s for s in rest if s.flower_suffix == "RC"]
    # test fraction 0: groups that overflow val spill into "test" and are folded back into val
    tv = split_by_molecule(rc_train, (1.0 - val_fraction, val_fraction, 0.0), seed, frozen)
    val = set(tv.val) | set(tv.test)
    train = sorted(s.step_id for s in rest if s.step_id not in val)
    return Split(train, sorted(val), sorted(s.step_id for s in test), [], tv.conflict)


SplitFn = Callable[..., Split]
DEFAULT_SPLIT: dict[Source, SplitFn] = {
    Source.PMECHDB: split_by_molecule,
    Source.RMECHDB: split_rc_molecule,
}


def default_split(source: Source | str) -> SplitFn:
    """The split strategy a source uses by default (see docs/decisions.md, 2026-10-19)."""
    return DEFAULT_SPLIT.get(Source(source), split_by_pathway)


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
    steps: Sequence[ElementaryStep],
    fractions: Fractions = (0.8, 0.1, 0.1),
    seed: int = 0,
    frozen: Frozen | None = None,
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
    return _assign_groups(groups, fractions, seed, frozen)
