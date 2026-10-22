"""Turn predicted (donor, donor site, acceptor, acceptor site) arrows into a graph edit.

Only the sites and the centres and descriptors of the chosen tokens are used, never the dataset
arrows and never the classifier's orbital type (docs/decisions.md 2026-10-22, round-trip
diagnosis). The arrows of a step build one bond-electron edit, sanitized once; each arrow
reads the state left by the earlier ones (an atom that took a pair is negative for the next
arrow: the E2 and enolization relays), and each electron pair moves once per step.

Source, for donor r at site a:
1. a has a pair and r is one-centre (a lone pair) or a is negative now: a lone pair of a;
2. else the bond to the largest-weight centre c of r (c != a, b) bonded to a, if that bond is
   multiple or r is a sigma pair (chi >= `SIGMA_CHI`: C-H in an E2, C-Si);
3. else a drawn bond a-c that no occupied token spans (its pair sits on a: Na-H);
4. else a lone pair of a; else, last, a Kekule structure in which a has a multiple bond
   (flagged APPLY_KEKULE_FALLBACK).

The bond a-b forms (or gains one order). Sink, for acceptor s at site b:
5. b had an open shell before the new bond (a carbocation, H+, a borane): nothing breaks;
6. b is in period 3 or beyond and s is one-centre (an empty d / p of its own, TiCl4, SnR3+):
   nothing breaks (the octet may expand);
7. else, if s is antibonding (negative overlap population), the bond to its centre d bonded to
   b breaks (multiple bonds first, then weight) and d takes the pair; with no such centre, the
   bond whose pair sits mostly off b (smallest weight on b over the occupied tokens spanning
   it) breaks.

Rule 7 is not decided by the tuple when b can expand its octet and s is a sigma*: PMechDB adds
to Si without a leaving group (pentacoordinate silicates) but substitutes at Br and Cl, and P,
S and I go both ways. Those steps stay below the round-trip gate (docs/decisions.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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
    breaks_bd: int | None = None  # the decoder's bond-break flag; None: the physical rules decide


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


KEKULE_FALLBACK = "APPLY_KEKULE_FALLBACK"
SIGMA_CHI = 0.5  # chi at or above: a sigma bond pair (chi 1 sigma, 0 pi)


def _mol_index(tokens: ReactantTokens, site: int) -> int:
    idx = tokens.mol_atom.get(site)
    if idx is None:
        raise ArrowApplyError(
            ApplyReason.UNKNOWN_ARROW, f"site {site} is an implicit hydrogen of the drawn molecule"
        )
    return idx


def _bonded_centre(
    tokens: ReactantTokens,
    state: EditState,
    tok: int,
    site: int,
    skip: int,
    sigma_ok: bool,
    multiple_first: bool = False,
):
    """The largest-weight centre c of token `tok` (not `site`, not `skip`) bonded to `site` in
    the current state, with a bond this step has not consumed yet; with `sigma_ok` False only a
    multiple bond qualifies; with `multiple_first` a multiple bond wins over any single one.
    Mol indices; None when there is none."""
    i = tokens.mol_atom[site]
    found = []
    for g, w in tokens.centres(tok).items():
        j = tokens.mol_atom.get(g)
        if g in (site, skip) or j is None or j not in state.neighbours(i):
            continue
        if sigma_ok or state.order(i, j) >= 2:
            found.append((multiple_first and state.order(i, j) < 2, -w, g, j))
    return min(found)[3] if found else None


def bond_weight_on(tokens: ReactantTokens, b: int, x: int) -> float:
    """Smallest weight on b over the occupied tokens spanning the global atoms b and x; 0 when
    no occupied token spans them (the drawn bond's pair sits on one atom: an ionic Na-H)."""
    spans = [
        tokens.centres(t)[b]
        for t in tokens.tokens_on_atom.get(b, [])
        if tokens.occupied[t - tokens.n_atoms] and x in tokens.centres(t)
    ]
    return min(spans, default=0.0)


def _tokenless_bond(state: EditState, tokens: ReactantTokens, a: int, b: int) -> int | None:
    """A drawn bond a-c (c != b) that no occupied token spans: its pair already sits on a."""
    glob = tokens.global_atom
    for c in state.neighbours(a):
        if c != b and bond_weight_on(tokens, glob[a], glob[c]) == 0.0:
            return c
    return None


def _kekule_switch(state: EditState, a: int, b: int) -> int | None:
    """Switch the base to another Kekule structure (same charges) in which a has a multiple
    bond to a neighbour other than b, keeping the accumulated edit valid; returns it."""
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
        hits = [c for c in trial.neighbours(a) if c != b and trial.order(a, c) >= 2]
        if valid and hits:
            state.base = trial.base
            return hits[0]
    return None


def _source(state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, flags: set) -> None:
    """Rules 1-3 of the module docstring for the donor side of one arrow."""
    a, b = _mol_index(tokens, arrow.donor_site), _mol_index(tokens, arrow.acceptor_site)
    has_pair = state.nonbonding(a) >= 2
    one_centre = len(tokens.centres(arrow.donor)) == 1
    if has_pair and (one_centre or state.charge(a) < 0):
        state.edit.add((a,), -1)
        return
    sigma = tokens.orbital(arrow.donor).descriptors.cos_character() >= SIGMA_CHI
    site, skip = arrow.donor_site, arrow.acceptor_site
    c = _bonded_centre(tokens, state, arrow.donor, site, skip, sigma_ok=sigma)
    c = c if c is not None else _tokenless_bond(state, tokens, a, b)
    if c is not None:
        state.decrement(a, c)
    elif has_pair:
        state.edit.add((a,), -1)
    else:
        c = _kekule_switch(state, a, b)
        if c is None:
            raise ArrowApplyError(ApplyReason.VALENCE, f"donor site {a}: no bond and no lone pair")
        flags.add(KEKULE_FALLBACK)
        state.decrement(a, c)


def open_shell(state: EditState, i: int) -> bool:
    """Fewer than 8 valence electrons (2 for H) around atom i in the current state:
    nonbonding electrons plus two per bond order, hydrogens included."""
    atom = state.base.GetAtomWithIdx(i)
    bonds = sum(state.order(i, j) for j in state.neighbours(i)) + atom.GetTotalNumHs()
    capacity = 2 if atom.GetAtomicNum() == 1 else 8
    return state.nonbonding(i) + 2 * bonds < capacity


def _sink(state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, room: bool) -> None:
    """Rules 5-7 of the module docstring for the acceptor side of one arrow; `room`: the
    acceptor site had an open shell before the new bond. A set `arrow.breaks_bd` decides
    whether a bond at b breaks; rule 7 still says which one."""
    b = _mol_index(tokens, arrow.acceptor_site)
    if arrow.breaks_bd == 0:
        return
    if arrow.breaks_bd is None:
        if room:
            return
        expands = state.base.GetAtomWithIdx(b).GetAtomicNum() > 10  # period 3+: can exceed 8
        if expands and len(tokens.centres(arrow.acceptor)) == 1:
            return  # an empty orbital of its own (Ti, Sn d / p): the bond only forms
    site, skip = arrow.acceptor_site, arrow.donor_site
    d = None
    if tokens.orbital(arrow.acceptor).descriptors.overlap_population < 0:
        d = _bonded_centre(tokens, state, arrow.acceptor, site, skip, True, multiple_first=True)
    if d is None:  # a full shell with no partner named: the bond whose pair sits mostly off b
        a = _mol_index(tokens, arrow.donor_site)
        glob = tokens.global_atom
        nbrs = [x for x in state.neighbours(b) if x != a]
        d = min(nbrs, key=lambda x: bond_weight_on(tokens, glob[b], glob[x]), default=None)
    if d is not None:
        state.decrement(b, d)
        state.edit.add((d,), +1)


def apply_predicted_arrows(
    mol: Chem.Mol, arrows: list[PredictedArrow], tokens: ReactantTokens
) -> ApplyResult:
    """Apply predicted arrows to `mol`, the molecule `tokens` was built from (same atom
    indices), as one edit. Returns the product or the reason code."""
    if mol.GetNumAtoms() != tokens.mol.GetNumAtoms():
        return ApplyResult(None, ApplyReason.UNKNOWN_ARROW, "mol is not the tokens' molecule")
    state = EditState(Chem.Mol(mol))
    flags: set[str] = set()
    try:
        for arrow in arrows:
            b = _mol_index(tokens, arrow.acceptor_site)
            room = open_shell(state, b)
            _source(state, tokens, arrow, flags)
            state.edit.add((_mol_index(tokens, arrow.donor_site), b), +1)
            _sink(state, tokens, arrow, room)
        product = sanitize_product(apply_edit(state.base, state.edit))
        return ApplyResult(product, flags=frozenset(flags))
    except ArrowApplyError as exc:
        return ApplyResult(None, exc.reason, exc.detail, flags=frozenset(flags))


def bond_break_label(tokens: ReactantTokens, label: ArrowLabel, s: int) -> tuple[int | None, bool]:
    """(breaks_bd, forced) of an aligned arrow with acceptor s: does the bond from b to s's
    partner d break in this step? Physics fixes it where it can:
    - s is one-centre (an empty orbital of b's own) or b has an open shell: 0, forced;
    - b is a full-shell atom of period 1-2 (H, C, N, O, ...): 1, forced;
    - b is in period 3+ and s is antibonding with a centre d bonded to b: learned, 1 when the
      annotated step breaks b-d (`label.sink_partners`), else 0;
    - b is in period 3+ and s is antibonding but its partner is the donor's own atom (the
      arrow turns that bond multiple): learned, 1 when the annotated step breaks any bond at b
      (rule 7's weakest bond then says which), else 0;
    - otherwise None (s not antibonding, or b bonded to nothing but the donor's atom)."""
    b = tokens.mol_atom.get(label.acceptor_site)
    if b is None:
        return None, False
    state = EditState(Chem.Mol(tokens.mol))
    if len(tokens.centres(s)) == 1 or open_shell(state, b):
        return 0, True
    if state.base.GetAtomWithIdx(b).GetAtomicNum() <= 10:
        return 1, True
    if tokens.orbital(s).descriptors.overlap_population >= 0:
        return None, False
    site, skip = label.acceptor_site, label.donor_site
    d = _bonded_centre(tokens, state, s, site, skip, True, multiple_first=True)
    if d is None:
        a = tokens.mol_atom.get(skip)
        if not any(x != a for x in state.neighbours(b)):
            return None, False
        return int(bool(label.sink_partners)), False
    d_global = next(g for g, i in tokens.mol_atom.items() if i == d)
    return int(d_global in label.sink_partners), False


def labeled_arrows(
    tokens: ReactantTokens, labels: list[ArrowLabel], valid=None, with_breaks: bool = True
) -> list[PredictedArrow]:
    """The teacher representatives (r*, a*, s*, b*) of aligned labels, as predicted arrows;
    with `valid(r, s)` the teacher pair is chosen among valid pairs, as the samples do; with
    `with_breaks` each arrow carries its labeled bond-break flag where it is learned; a forced
    flag is left to the physical rules, which apply it in the state the earlier arrows of the
    step leave (`bond_break_label`)."""
    out = []
    for label in labels:
        r, s = representatives(tokens, label, valid)
        flag, forced = bond_break_label(tokens, label, s) if with_breaks else (None, True)
        flag = None if forced else flag
        out.append(PredictedArrow(r, label.donor_site, s, label.acceptor_site, flag))
    return out


def valid_pair_test(tokens: ReactantTokens):
    """valid(r, s) over global token ids: the samples' `pair_valid` (bonded-site filter and
    antibonding-partner rule) of this reactant side."""
    from ksmech.physics.pairs import valid_pairs

    valid = valid_pairs([f.tokens for f in tokens.fragments], tokens.candidate)
    occ = tokens.n_atoms + np.flatnonzero(tokens.occupied)
    virt = tokens.n_atoms + np.flatnonzero(~tokens.occupied)
    row = {int(t): i for i, t in enumerate(occ)}
    col = {int(t): i for i, t in enumerate(virt)}
    return lambda r, s: r in row and s in col and bool(valid[row[r], col[s]])
