"""Turn predicted (donor, donor site, acceptor, acceptor site) arrows into a graph edit.

Only the sites and the descriptors of the chosen tokens are used, never the dataset arrows
and never the classifier's orbital type. Each arrow, in order:

1. source: if a carries a negative charge in the current state, or a one-centre occupied
   token (`n_centers == 1`) sits on a, and the current Lewis structure gives a a lone pair
   (at least 2 nonbonding electrons), the pair is a lone pair of a. Otherwise it comes from
   a bond a-c, chosen in this order: a multiple bond to a centre of the donor token; the
   same after switching to another Kekulé structure of the molecule; a single bond to a
   centre of the donor token (a sigma donor, such as C-H in an E2); a drawn bond to a that
   no occupied token spans (the wavefunction holds that pair on a: a beta-silicate C-Si);
   a multiple bond to any neighbour; c is never the acceptor site b. That bond loses one
   order and c takes the positive charge. When no bond qualifies but a has a lone pair in
   the current Lewis structure (a delocalized amine N, drawn neutral), that pair is used;
2. the bond a-b forms (or gains one order);
3. sink: if b had an open valence shell before the new bond (fewer than 8 electrons, 2 for
   H: a carbocation, H+, a borane), nothing else changes. Otherwise a bond b-d breaks and d
   takes the leaving pair. d is a centre of the acceptor token bonded to b, when the
   acceptor is antibonding (negative overlap population): multiple bonds first (pi*), then
   by weight, never the donor site a. When the acceptor names no such centre, d is the
   neighbour whose bond pair already sits mostly off b (smallest weight on b over the
   occupied tokens spanning b-d; 0 for a drawn bond no token spans).

The spec's rule for step 3 ("a two-centre antibonding acceptor on (b, d) breaks b-d") is the
special case of a full-shell b with a two-centre acceptor. Taken literally it breaks a C-C
single bond when a carbocation's empty p is conjugated onto a neighbour (a two-centre token,
C 0.60 / C 0.25, overlap -0.19) and breaks nothing when a protonated carbonyl pi* spreads over
three centres; see docs/decisions.md.

Arrows are applied in sequence, so an atom that took a leaving pair from an earlier arrow of
the step is negative when a later arrow starts there (the E2 and conjugate-addition relays).
Each electron pair moves once per elementary step: a bond already decremented by an earlier
arrow of the step is not decremented again. The result is sanitized and reduced to its
resonance key (`ApplyResult.key`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.arrows.align import ArrowLabel, representatives
from ksmech.arrows.apply import (
    TYPE_TO_ORDER,
    ApplyReason,
    ApplyResult,
    ArrowApplyError,
    BEEdit,
    apply_edit,
    nonbonding_electrons,
    sanitize_product,
)
from ksmech.arrows.reactants import ReactantTokens

MAX_KEKULE = 64  # Kekulé structures tried when a donor site needs a multiple bond


@dataclass(frozen=True)
class PredictedArrow:
    """Global token ids and global atom indices of a `ReactantTokens` index space."""

    donor: int
    donor_site: int
    acceptor: int
    acceptor_site: int


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


@dataclass
class EditState:
    """A base molecule plus the edit accumulated by the arrows applied so far."""

    base: Chem.Mol
    edit: BEEdit = field(default_factory=BEEdit)
    consumed: set = field(default_factory=set)  # bonds already decremented in this step

    def order(self, i: int, j: int) -> int:
        bond = self.base.GetBondBetweenAtoms(i, j)
        old = 0 if bond is None else TYPE_TO_ORDER.get(bond.GetBondType(), 1)
        return old + self.edit.bond_delta[_pair(i, j)]

    def charge(self, i: int) -> int:
        return self.base.GetAtomWithIdx(i).GetFormalCharge() + self.edit.charge_delta()[i]

    def nonbonding(self, i: int) -> int:
        """Bond moves change charges, not nonbonding electrons: N = N_base + dN."""
        return nonbonding_electrons(self.base.GetAtomWithIdx(i)) + self.edit.lone_delta[i]

    def neighbours(self, i: int) -> list[int]:
        """Current neighbours of atom i whose bond to it was not decremented this step."""
        n_atoms = self.base.GetNumAtoms()
        return [
            j
            for j in range(n_atoms)
            if j != i and self.order(i, j) >= 1 and _pair(i, j) not in self.consumed
        ]

    def decrement(self, i: int, j: int) -> None:
        self.edit.add((i, j), -1)
        self.consumed.add(_pair(i, j))


def _mol_index(tokens: ReactantTokens, site: int) -> int:
    idx = tokens.mol_atom.get(site)
    if idx is None:
        raise ArrowApplyError(
            ApplyReason.UNKNOWN_ARROW, f"site {site} is an implicit hydrogen of the drawn molecule"
        )
    return idx


def has_one_centre_donor(tokens: ReactantTokens, site: int) -> bool:
    """A one-centre occupied token sits on `site` (a global atom)."""
    return any(
        tokens.occupied[t - tokens.n_atoms] and tokens.orbital(t).descriptors.n_centers == 1
        for t in tokens.tokens_on_atom.get(site, [])
    )


def _kekule_switch(state: EditState, a: int, centres: set[int]) -> int | None:
    """Swap the base for another Kekulé structure (same charges) in which a has a multiple
    bond to one of `centres`, keeping every accumulated bond change valid. Returns that c."""
    flags = int(rdchem.ResonanceFlags.KEKULE_ALL)
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(state.base, flags, MAX_KEKULE)
        structures = [supplier[k] for k in range(len(supplier))]
    charges = [x.GetFormalCharge() for x in state.base.GetAtoms()]
    for mol in structures:
        if mol is None or [x.GetFormalCharge() for x in mol.GetAtoms()] != charges:
            continue
        trial = EditState(Chem.Mol(mol), state.edit, state.consumed)
        trial.base.UpdatePropertyCache(strict=False)
        Chem.Kekulize(trial.base, clearAromaticFlags=True)
        valid = all(0 <= trial.order(i, j) <= 3 for (i, j) in state.edit.bond_delta)
        hits = [c for c in trial.neighbours(a) if c in centres and trial.order(a, c) >= 2]
        if valid and hits:
            state.base = trial.base
            return hits[0]
    return None


def bond_weight_on(tokens: ReactantTokens, b: int, x: int) -> float:
    """Smallest weight on b over the occupied tokens spanning the global atoms b and x; 0 when
    no occupied token spans them (the drawn bond's pair sits on one atom)."""
    spans = [
        tokens.centres(t)[b]
        for t in tokens.tokens_on_atom.get(b, [])
        if tokens.occupied[t - tokens.n_atoms] and x in tokens.centres(t)
    ]
    return min(spans, default=0.0)


def source_partner(
    state: EditState, tokens: ReactantTokens, a: int, b: int, weights: dict[int, float]
) -> int | None:
    """The atom c whose bond to a donates the pair (see the module docstring for the order),
    never the acceptor site b. `weights`: mol atom index -> population of the donor token's
    centres."""
    nbrs = [c for c in state.neighbours(a) if c != b]
    by_weight = sorted((c for c in nbrs if c in weights), key=lambda c: -weights[c])
    multiple = [c for c in by_weight if state.order(a, c) >= 2]
    if multiple:
        return multiple[0]
    switched = _kekule_switch(state, a, set(weights) - {a}) if len(weights) > 1 else None
    if switched is not None:
        return switched
    if by_weight:
        return by_weight[0]
    glob = tokens.global_atom
    tokenless = [c for c in nbrs if bond_weight_on(tokens, glob[a], glob[c]) == 0.0]
    if tokenless:
        return tokenless[0]
    any_multiple = [c for c in nbrs if state.order(a, c) >= 2]
    return any_multiple[0] if any_multiple else None


def _apply_source(
    state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, a: int, b: int
) -> None:
    """Take the pair from a lone pair of a or from a bond a-c. When no bond qualifies, a lone
    pair of the current Lewis structure (a delocalized amine N drawn neutral) is used."""
    has_pair = state.nonbonding(a) >= 2
    lone = state.charge(a) < 0 or has_one_centre_donor(tokens, arrow.donor_site)
    if lone and has_pair:
        state.edit.add((a,), -1)
        return
    weights = {
        tokens.mol_atom[g]: w
        for g, w in tokens.centres(arrow.donor).items()
        if g in tokens.mol_atom
    }
    c = source_partner(state, tokens, a, b, weights)
    if c is not None:
        state.decrement(a, c)
    elif has_pair:
        state.edit.add((a,), -1)
    else:
        raise ArrowApplyError(ApplyReason.VALENCE, f"donor site {a} has no pair to give")


def open_shell(state: EditState, i: int) -> bool:
    """Fewer than 8 valence electrons (2 for H) around atom i in the current state:
    nonbonding electrons plus two per bond order, hydrogens included."""
    atom = state.base.GetAtomWithIdx(i)
    bonds = sum(state.order(i, j) for j in state.neighbours(i)) + atom.GetTotalNumHs()
    capacity = 2 if atom.GetAtomicNum() == 1 else 8
    return state.nonbonding(i) + 2 * bonds < capacity


def sink_partner(
    state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, a: int, b: int
) -> int | None:
    """The atom d whose bond to the full-shell sink b breaks (see the module docstring)."""
    nbrs = [x for x in state.neighbours(b) if x != a]
    glob = tokens.global_atom
    centres = tokens.centres(arrow.acceptor)
    if tokens.orbital(arrow.acceptor).descriptors.overlap_population < 0:
        named = [x for x in nbrs if glob.get(x) in centres and glob[x] != arrow.acceptor_site]
        named.sort(key=lambda x: (state.order(b, x) < 2, -centres[glob[x]]))
        if named:
            return named[0]
    ranked = sorted(nbrs, key=lambda x: bond_weight_on(tokens, glob[b], glob[x]))
    return ranked[0] if ranked else None


def _apply_sink(
    state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, a: int, b: int, room: bool
) -> None:
    if room:
        return
    d = sink_partner(state, tokens, arrow, a, b)
    if d is not None:
        state.decrement(b, d)
        state.edit.add((d,), +1)


def apply_predicted_arrows(
    mol: Chem.Mol, arrows: list[PredictedArrow], tokens: ReactantTokens
) -> ApplyResult:
    """Apply predicted arrows to `mol`, the molecule `tokens` was built from (same atom
    indices). Returns the product or the reason code."""
    if mol.GetNumAtoms() != tokens.mol.GetNumAtoms():
        return ApplyResult(None, ApplyReason.UNKNOWN_ARROW, "mol is not the tokens' molecule")
    state = EditState(Chem.Mol(mol))
    try:
        for arrow in arrows:
            a = _mol_index(tokens, arrow.donor_site)
            b = _mol_index(tokens, arrow.acceptor_site)
            room = open_shell(state, b)
            _apply_source(state, tokens, arrow, a, b)
            state.edit.add((a, b), +1)
            _apply_sink(state, tokens, arrow, a, b, room)
        return ApplyResult(sanitize_product(apply_edit(state.base, state.edit)))
    except ArrowApplyError as exc:
        return ApplyResult(None, exc.reason, exc.detail)


def labeled_arrows(tokens: ReactantTokens, labels: list[ArrowLabel]) -> list[PredictedArrow]:
    """The teacher representatives (r*, a*, s*, b*) of aligned labels, as predicted arrows."""
    out = []
    for label in labels:
        r, s = representatives(tokens, label)
        out.append(PredictedArrow(r, label.donor_site, s, label.acceptor_site))
    return out
