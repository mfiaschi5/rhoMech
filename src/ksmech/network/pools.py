"""The engine's pool states and the model's expander over them (reaction-network plan, items C
and D).

A pool state is a Kekule, explicit-H SMILES with a map number on every atom, as the web app's
pool states are: the map numbers follow every atom through a pathway, so a step's atoms are
named the same way in every pool (the mechanism classes) and a mapped target can be compared
with the pool (the remaining change, `network.target`). `pool_state` keeps the map numbers a
SMILES has and numbers the other atoms after them, hydrogens included.

`ModelExpander` expands a state with the trained model: the tokens of its fragments (computed
with GFN2-xTB first when `build_missing`, else a missing one fails the node as MISSING_TOKENS,
STALE_ARTIFACT or NO_BOND_INDEX), the sample with its Lewis structure, `model.decode.decode_pool`
with the stored samples' checks (`eval.decode.lewis_apply`), and B + R written on the pool's
mapped molecule (`arrows.apply.apply_edit`; a product RDKit cannot sanitize is PRODUCT_INVALID).
The decoder merges completions by the child's pool key. The END_MECH head is not read.
`ModelExpander.step_logp` scores a step given on map numbers at a state, `reverse_logp` the
reverse -R of a proposal at its child (microscopic reversibility, item E), both with
`model.decode.matrix_logp`. The samples of the `SAMPLE_CACHE` states used last are kept.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from rdkit import Chem, rdBase

from ksmech.arrows.apply import (
    ArrowApplyError,
    BEEdit,
    apply_edit,
    nonbonding_electrons,
    sanitize_product,
)
from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key
from ksmech.network.keys import pool_key_of
from ksmech.network.search import Expansion, Proposal
from ksmech.network.target import fragments, heavy_state, skeleton_key

SAMPLE_CACHE = 256  # states whose (ReactantTokens, sample) an expander keeps


@dataclass(frozen=True)
class PoolState:
    smiles: str  # Kekule, explicit H, every atom mapped
    species: tuple[str, ...]  # identity key v2 of each fragment, sorted
    skeletons: tuple[str, ...]  # heavy-atom skeleton key of each fragment, sorted
    orders: Mapping[tuple[int, int], int]  # bond orders between heavy atoms, on map numbers
    nonbonding: Mapping[int, int]  # N of each heavy atom, by map number

    @property
    def key(self) -> str:
        return pool_key_of(self.species)


def _numbered(mol: Chem.Mol) -> Chem.Mol:
    """Explicit hydrogens and a map number on every atom, the given ones kept."""
    with rdBase.BlockLogs():
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
    used = {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    nxt = max(used, default=0) + 1
    for atom in mol.GetAtoms():
        if not atom.GetAtomMapNum():
            atom.SetAtomMapNum(nxt)
            nxt += 1
    return mol


def state_of_mol(mol: Chem.Mol) -> PoolState:
    """The state of a mapped, explicit-H, Kekule molecule. Its SMILES lists the fragments in
    the order of their identity keys (ties by their mapped SMILES), so the fragment order the
    model sees does not depend on the map numbers: the model reads a fragment index, and a pool
    drawn in two orders scores differently (decisions.md, 2026-11-09)."""
    frags = fragments(mol)
    with rdBase.BlockLogs():
        rows = sorted((identity_key(f), Chem.MolToSmiles(f, canonical=True)) for f in frags)
    smiles = ".".join(text for _, text in rows)
    species = tuple(key for key, _ in rows)
    skeletons = tuple(sorted(skeleton_key(f) for f in frags))
    orders, lone = heavy_state(mol)
    return PoolState(smiles, species, skeletons, orders, lone)


def pool_state(smiles: str) -> PoolState:
    """The state of a pool given as SMILES, mapped or not (module docstring)."""
    with rdBase.BlockLogs():
        return state_of_mol(_numbered(parse_mapped(smiles)))


def apply_on_mol(mol: Chem.Mol, bonds, lone) -> PoolState:
    """B + R on the mapped molecule's atom indices: bond orders and N changed, the formal
    charges following (dq = -(dN + dB)), every changed atom's radical count set to the parity
    of its N. Raises ArrowApplyError when the edit or the sanitization fails."""
    edit = BEEdit()
    for i, j, d in bonds:
        edit.bond_delta[(min(i, j), max(i, j))] += int(d)
    for i, d in lone:
        edit.lone_delta[int(i)] += int(d)
    rw = apply_edit(mol, edit)
    for i in edit.atoms():
        atom = rw.GetAtomWithIdx(i)
        atom.SetNumRadicalElectrons(nonbonding_electrons(atom) % 2)
    return state_of_mol(sanitize_product(rw))


def signature(bonds, lone, maps: Mapping[int, int]) -> tuple[str, frozenset[int]]:
    """(the step on map numbers as a canonical text, the map numbers it touches)."""
    b = sorted((*sorted((maps[i], maps[j])), int(d)) for i, j, d in bonds)
    a = sorted((maps[i], int(d)) for i, d in lone)
    text = (
        ";".join(f"{m}-{n}:{d:+d}" for m, n, d in b) + "|" + ";".join(f"{m}:{d:+d}" for m, d in a)
    )
    return text, frozenset(m for m, n, _ in b for m in (m, n)) | {m for m, _ in a}


class ModelExpander:
    """`network.search.Expander` over `PoolState`s with a trained model (module docstring)."""

    def __init__(self, model, loader, cfg=None, scope: str = "all", build_missing: bool = False):
        from ksmech.model.decode import DecodeConfig

        self.model, self.loader, self.scope = model, loader, scope
        self.cfg = cfg or DecodeConfig()
        self.build_missing = build_missing
        self._samples: dict[str, tuple] = {}

    def key(self, state: PoolState) -> str:
        return state.key

    def _sample(self, state: PoolState):
        """(ReactantTokens, sample) of a state, or (None, the failure code); the last
        `SAMPLE_CACHE` states used are kept (least recently used out first)."""
        hit = self._samples.pop(state.smiles, None)
        if hit is None:
            if len(self._samples) >= SAMPLE_CACHE:
                self._samples.pop(next(iter(self._samples)))
            hit = self._featurize(state)
        self._samples[state.smiles] = hit  # most recent last
        return hit

    def _featurize(self, state: PoolState):
        from ksmech.arrows.lewis import MissingBondIndex, reactant_lewis
        from ksmech.arrows.reactants import MissingTokens, ReactantTokens
        from ksmech.arrows.steps import reactant_sample, with_lewis

        try:
            if self.build_missing:
                from ksmech.eval.embedding import ensure_pool
                from ksmech.physics.textbook import fragment_keys

                ensure_pool(fragment_keys(state.smiles), self.loader)
            rt = ReactantTokens.from_smiles(state.smiles, self.loader)
            return rt, with_lewis(reactant_sample("network", "network", rt), reactant_lewis(rt))
        except MissingTokens as exc:
            return None, exc.reason
        except MissingBondIndex:
            return None, "NO_BOND_INDEX"

    def expand(self, state: PoolState) -> Expansion:
        from ksmech.eval.decode import lewis_apply
        from ksmech.model.decode import decode_pool

        rt, sample = self._sample(state)
        if rt is None:
            return Expansion([], Counter(), sample)
        mol = parse_mapped(state.smiles)
        check, children, atoms = lewis_apply(sample), {}, set(rt.mol_atom)

        def apply(bonds, lone):
            _, why = check(bonds, lone)
            if why is not None:
                return None, why
            if not {a for i, j, _ in bonds for a in (i, j)} | {i for i, _ in lone} <= atoms:
                return None, "STEP_OUTSIDE_POOL"
            on_mol = [(rt.mol_atom[i], rt.mol_atom[j], d) for i, j, d in bonds]
            try:
                child = apply_on_mol(mol, on_mol, [(rt.mol_atom[i], d) for i, d in lone])
            except ArrowApplyError:
                return None, "PRODUCT_INVALID"
            children[(child.key, bonds, lone)] = (child, on_mol)
            return child.key, None

        skips: Counter = Counter()
        found = decode_pool(self.model, sample, self.cfg, apply, self.scope, skips)
        maps = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
        out = []
        for c in found:
            child, on_mol = children[(c.key, c.bonds, c.lone)]
            text, touched = signature(on_mol, [(rt.mol_atom[i], d) for i, d in c.lone], maps)
            info = {"anchor": c.anchor, **on_maps(on_mol, c.lone, rt, maps)}
            out.append(Proposal(child, c.logp, text, touched, info))
        return Expansion(out, skips)

    def step_logp(self, state: PoolState, bonds, lone):
        """`model.decode.MatrixScore` of a step given on map numbers ((m, n, d), (m, d)) at
        `state`, or the failure code (the state's, or STEP_OUTSIDE_POOL)."""
        from ksmech.model.decode import matrix_logp

        rt, sample = self._sample(state)
        if rt is None:
            return sample
        atom = {a.GetAtomMapNum(): rt.global_atom.get(a.GetIdx()) for a in rt.mol.GetAtoms()}
        try:
            on_global = [(atom[m], atom[n], d) for m, n, d in bonds]
            lone_global = [(atom[m], d) for m, d in lone]
        except KeyError:
            return "STEP_OUTSIDE_POOL"
        if any(x is None for b in on_global for x in b[:2]) or any(
            a is None for a, _ in lone_global
        ):
            return "STEP_OUTSIDE_POOL"
        return matrix_logp(self.model, sample, on_global, lone_global)

    def reverse_logp(self, proposal: Proposal):
        """The score of -R at the proposal's child: how likely the child reverts."""
        bonds = [(m, n, -d) for m, n, d in proposal.info["bonds"]]
        lone = [(m, -d) for m, d in proposal.info["lone"]]
        return self.step_logp(proposal.child, bonds, lone)


def on_maps(on_mol, lone_global, rt, maps: Mapping[int, int]) -> dict:
    """A step's bonds and atom entries on map numbers (`bonds`, `lone`), from its bonds on the
    mol's atoms and its atom entries on the global atoms."""
    bonds = tuple(sorted((*sorted((maps[i], maps[j])), int(d)) for i, j, d in on_mol))
    lone = tuple(sorted((maps[rt.mol_atom[i]], int(d)) for i, d in lone_global))
    return {"bonds": bonds, "lone": lone}
