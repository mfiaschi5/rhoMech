"""Reservoir species: always available, never consumed (reaction-network plan, item E).

A reservoir species (the solvent, a reagent in excess) is named by its identity key v2, as a
pool state names its species (`PoolState.species`). `Reservoir` maps each key to the SMILES of
one molecule and keeps `copies` of it in every pool, so the model always sees it and a pool
does not run out of it after one bimolecular step.

`ReservoirExpander` wraps an expander over `network.pools.PoolState`s: every child is brought
back to `copies` of each reservoir species (`normalize_pool`: a copy the step consumed is
replaced by a fresh molecule with new map numbers; a copy beyond `copies`, which a step
released, is removed, never one the step touches, so the reverse step stays on the pool's
atoms), and its proposal records in `info["reservoir_draw"]` the copies the normalization took
from the reservoir (species key -> count, negative when it gave copies back).

The draws enter the G balance at the species' chemical potential: along a path the draws add
up to D_x, and the potential of a state is phi = G(state) - sum_x D_x mu_x (`network.chain`).
A step A + H2O -> B, with the water replenished, costs G(B + H2O) - G(A + H2O) - mu(H2O) =
G(B) - G(A) - mu(H2O): the replenishment itself costs nothing, since a copy moves between the
reservoir and the pool at the same chemical potential.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from rdkit import Chem, rdBase

from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key
from ksmech.network.pools import PoolState, pool_state, state_of_mol
from ksmech.network.search import Expansion, Proposal

RESERVOIR_DRAW = "reservoir_draw"
KEPT_TOUCHED = "RESERVOIR_KEPT_TOUCHED"  # a released copy the step touches stays in the pool


@dataclass(frozen=True)
class Reservoir:
    smiles: Mapping[str, str] = field(default_factory=dict)  # species key -> one molecule
    copies: int = 1  # copies of each kept in every pool

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.smiles)


def reservoir_from_smiles(smiles: Iterable[str], copies: int = 1) -> Reservoir:
    """The reservoir of the given molecules (one species each), keyed by identity key v2."""
    out = {}
    for smi in smiles:
        species = pool_state(smi).species
        if len(species) != 1:
            raise ValueError(f"a reservoir species is one molecule, got {smi!r}")
        out[species[0]] = smi
    return Reservoir(out, copies)


def _fragments(mol: Chem.Mol) -> list[tuple[str, tuple[int, ...], frozenset[int]]]:
    """(identity key, atom indices, map numbers) of each fragment."""
    mapping: list = []
    with rdBase.BlockLogs():
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False,
                                 fragsMolAtomMapping=mapping)  # fmt: skip
        keys = [identity_key(f) for f in frags]
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    return [
        (k, tuple(idx), frozenset(maps[i] for i in idx))
        for k, idx in zip(keys, mapping, strict=True)
    ]


def _added(mol: Chem.Mol, smiles: str) -> Chem.Mol:
    """mol with one molecule of `smiles` added, its map numbers after mol's."""
    top = max((a.GetAtomMapNum() for a in mol.GetAtoms()), default=0)
    add = parse_mapped(pool_state(smiles).smiles)
    for atom in add.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + top)
    return Chem.CombineMols(mol, add)


def normalize_pool(
    state: PoolState, reservoir: Reservoir, keep: frozenset[int] = frozenset(),
    skips: Counter | None = None,
) -> tuple[PoolState, Counter]:  # fmt: skip
    """(the pool with `copies` of every reservoir species, the copies taken from the
    reservoir); copies whose map numbers meet `keep` are never removed (KEPT_TOUCHED)."""
    mol = parse_mapped(state.smiles)
    frags = _fragments(mol)
    drawn: Counter = Counter()
    drop: list[int] = []
    for key in sorted(reservoir.keys):
        have = [f for f in frags if f[0] == key]
        extra = len(have) - reservoir.copies
        free = sorted((f for f in have if not f[2] & keep), key=lambda f: -max(f[2]))
        if extra > len(free) and skips is not None:
            skips[KEPT_TOUCHED] += 1
        for f in free[: max(extra, 0)]:
            drop.extend(f[1])
            drawn[key] -= 1
    if drop:
        rw = Chem.RWMol(mol)
        for i in sorted(drop, reverse=True):
            rw.RemoveAtom(i)
        mol = rw.GetMol()
    for key in sorted(reservoir.keys):
        for _ in range(reservoir.copies - sum(f[0] == key for f in frags)):
            mol = _added(mol, reservoir.smiles[key])
            drawn[key] += 1
    if not drop and not drawn:
        return state, Counter()
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
    return state_of_mol(mol), Counter({k: n for k, n in drawn.items() if n})


class ReservoirExpander:
    """An expander whose children keep the reservoir's copies (module docstring); `key`,
    `reverse_logp` and the sink test's `rescan` go to the inner expander."""

    def __init__(self, inner, reservoir: Reservoir) -> None:
        self.inner, self.reservoir = inner, reservoir
        for hook in ("rescan", "reverse_logp"):  # absent on the inner expander: absent here
            if not hasattr(inner, hook):
                setattr(self, hook, None)

    def key(self, state) -> str:
        return self.inner.key(state)

    def start(self, state: PoolState) -> PoolState:
        """The start pool with the reservoir's copies."""
        return normalize_pool(state, self.reservoir)[0]

    def _normalized(self, ex: Expansion) -> Expansion:
        out = []
        for p in ex.proposals:
            child, drawn = normalize_pool(p.child, self.reservoir, p.touched, ex.skips)
            out.append(replace(p, child=child, info={**p.info, RESERVOIR_DRAW: dict(drawn)}))
        return Expansion(out, ex.skips, ex.failure)

    def expand(self, state) -> Expansion:
        return self._normalized(self.inner.expand(state))

    def rescan(self, state, factor: int) -> Expansion:
        return self._normalized(self.inner.rescan(state, factor))

    def reverse_logp(self, proposal: Proposal):
        return self.inner.reverse_logp(proposal)
