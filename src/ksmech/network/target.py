"""Arrival at a target product, and the change still to make (reaction-network plan, item D).

Arrival is tested on two levels, reported with the pathway:

- EXACT: every species of the target (identity key v2, `data.resonance.identity_key`: the sigma
  key of a closed shell, with the unpaired counts of an open shell) is in the pool, with its
  multiplicity; the pool may hold by-products besides;
- SKELETON: every target species' heavy-atom skeleton (its heavy atoms with their bond orders,
  no H, no charges, no radicals, aromaticity perceived) is in the pool, so a product drawn after
  work-up (neutral, or protonated) is reached up to proton transfers that move no bond: an acid
  and its carboxylate agree, a ketone and its alcohol, or two tautomers, do not (decided
  2026-11-10; until then the bond orders were erased and those merged). The level is always
  reported.

When the target is atom-mapped on the start pool's map numbers (a dataset's chain keeps them),
the remaining change dB = B_target - B_now on the target's heavy atoms gives a lower bound on the
steps left (`steps_lower_bound`): one step changes at most `max_bonds` bond entries by 1 and
`max_atoms` atom entries by 2 (the decoder's caps), and each entry by at most that much, so the
bound is admissible and A* with `step_cost` x the bound stays exact. Entries on H, or on atoms
outside the target, are left out, which only lowers the bound. A target without maps, or with a
map number the pool lacks, has no bound (MAPPING_FAILED): the search runs without heuristic.
"""

from __future__ import annotations

import contextlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rdkit import Chem, rdBase

from ksmech.arrows.apply import TYPE_TO_ORDER, nonbonding_electrons
from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key

EXACT, SKELETON = "EXACT", "SKELETON"
LEVELS = (EXACT, SKELETON)
MAPPING_FAILED = "MAPPING_FAILED"


def fragments(mol: Chem.Mol) -> list[Chem.Mol]:
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def skeleton_key(frag: Chem.Mol) -> str:
    """The canonical SMILES of a fragment's heavy atoms with their bond orders, no charge, no H,
    no radical and no map number; aromaticity is perceived on it, so the Kekule forms of one
    ring system give one key (decided 2026-11-10: an acid and its carboxylate agree, a ketone
    and its alcohol, or two tautomers, do not)."""
    rw = Chem.RWMol(frag)
    for atom in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        rw.RemoveAtom(atom)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetAtomMapNum(0)
        atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        bond.SetIsAromatic(False)
    mol = rw.GetMol()
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        with contextlib.suppress(Exception):  # an exotic ring system keeps its Kekule form
            Chem.SetAromaticity(mol)
        return Chem.MolToSmiles(mol, canonical=True)


def heavy_state(mol: Chem.Mol) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    """(bond order between mapped heavy atoms, N of each mapped heavy atom), on map numbers."""
    maps = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    heavy = {i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1 and maps[i]}
    orders = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in heavy and j in heavy:
            m, n = sorted((maps[i], maps[j]))
            orders[(m, n)] = TYPE_TO_ORDER.get(b.GetBondType(), 1)
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
        lone = {maps[i]: nonbonding_electrons(mol.GetAtomWithIdx(i)) for i in heavy}
    return orders, lone


@dataclass(frozen=True)
class Target:
    species: Counter  # identity keys of the target's fragments
    skeletons: Counter  # their skeleton keys
    orders: Mapping[tuple[int, int], int] | None  # mapped heavy-atom bond orders (None: no maps)
    nonbonding: Mapping[int, int] | None  # N of each mapped heavy atom


def target_from_smiles(smiles: str) -> Target:
    """The target of a search from a SMILES (one product or several, '.' separated, atom-mapped
    on the start pool's map numbers or not mapped at all)."""
    with rdBase.BlockLogs():
        mol = parse_mapped(smiles)
        frags = fragments(mol)
        species = Counter(identity_key(f) for f in frags)
    skeletons = Counter(skeleton_key(f) for f in frags)
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy or any(a.GetAtomMapNum() == 0 for a in heavy):
        return Target(species, skeletons, None, None)
    orders, lone = heavy_state(mol)
    return Target(species, skeletons, orders, lone)


def arrival(
    target: Target, species: Sequence[str], skeletons: Sequence[str], levels=LEVELS
) -> str | None:
    """The first of `levels` at which the pool holds the target, None when none."""
    if EXACT in levels and not target.species - Counter(species):
        return EXACT
    if SKELETON in levels and not target.skeletons - Counter(skeletons):
        return SKELETON
    return None


def steps_lower_bound(
    target: Target, orders: Mapping, nonbonding: Mapping, max_bonds: int, max_atoms: int
) -> int | None:
    """A lower bound on the elementary steps from a pool (its mapped heavy-atom bond orders and
    N) to the target (module docstring); None when the bound is unavailable (MAPPING_FAILED)."""
    if target.orders is None or target.nonbonding is None:
        return None
    if not set(target.nonbonding) <= set(nonbonding):
        return None
    atoms = set(target.nonbonding)
    pairs = {p for p in (*target.orders, *orders) if p[0] in atoms and p[1] in atoms}
    d_bonds = [abs(target.orders.get(p, 0) - orders.get(p, 0)) for p in pairs]
    d_atoms = [abs(target.nonbonding[m] - nonbonding[m]) for m in atoms]
    bounds = [
        math.ceil(sum(d_bonds) / max(max_bonds, 1)),
        math.ceil(sum(d_atoms) / (2 * max(max_atoms, 1))),
        max(d_bonds, default=0),
        math.ceil(max(d_atoms, default=0) / 2),
    ]
    return max(bounds)
