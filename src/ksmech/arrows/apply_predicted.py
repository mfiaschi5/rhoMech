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

The bond a-b forms (or gains one order). Sink, for acceptor s at site b: the arrow's leaving
bond lambda (spec v7.5, Leaving bond) breaks, and its other atom x takes the pair; NONE breaks
nothing. lambda is chosen among NONE and the Lewis bonds of b other than the one being formed,
as the edited state has them when the arrow is read, under the physical mask
(`leaving_choices`):
5. b had an open shell before the new bond (a carbocation, H+, a borane, an empty d / p
   shell of Ti, Sn): NONE only;
6. b is a full-shell atom of period 1-2 (H, C, N, O): NONE excluded, a bond must give way;
7. otherwise (b in period 3+, which can expand its octet: Si, P, S, I) the choice is learned.
`leaving_labels` labels lambda for a dataset step with the same reading: the bond b-x whose
leaving arrow the step has, else NONE. Each candidate carries the pointer head's features:
whether it is the partner bond of s, its Wiberg index, its order, the element and charge of x.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.arrows.align import (
    NONE_LEAVING,
    AlignFlag,
    ArrowLabel,
    LeavingCandidate,
    representatives,
)
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
    leaving: int = NONE_LEAVING  # x of the bond b-x that breaks (global atom), or NONE_LEAVING


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


def acceptor_partner(state: EditState, tokens: ReactantTokens, arrow: PredictedArrow):
    """The partner d of s at b (mol index): the largest-weight centre of an antibonding,
    multi-centre s bonded to b in the current state (multiple bonds first, never the donor
    site); None when s names none."""
    s = arrow.acceptor
    if len(tokens.centres(s)) < 2 or tokens.orbital(s).descriptors.overlap_population >= 0:
        return None
    site, skip = arrow.acceptor_site, arrow.donor_site
    return _bonded_centre(tokens, state, s, site, skip, True, multiple_first=True)


def leaving_choices(
    state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, room: bool
) -> tuple[list[LeavingCandidate], bool]:
    """(the Lewis bonds at b the mask allows, NONE allowed) in the current state (rules 5-7).
    `room`: b had an open shell before the new bond."""
    if room:
        return [], True
    b, a = _mol_index(tokens, arrow.acceptor_site), _mol_index(tokens, arrow.donor_site)
    partner, glob = acceptor_partner(state, tokens, arrow), tokens.global_atom
    out = []
    for x in state.neighbours(b):
        if x == a:
            continue
        wbi = tokens.bond_index(glob[b], glob[x])
        element = state.base.GetAtomWithIdx(x).GetAtomicNum()
        order, charge = state.order(b, x), state.charge(x)
        out.append(LeavingCandidate(glob[x], x == partner, wbi or 0.0, order, element, charge))
    return out, state.base.GetAtomWithIdx(b).GetAtomicNum() > 10


def _leave(state: EditState, tokens: ReactantTokens, b: int, leaving: int) -> None:
    """Break the leaving bond b-x (x takes the pair); NONE_LEAVING breaks nothing."""
    if leaving == NONE_LEAVING:
        return
    x = tokens.mol_atom.get(leaving)
    if x is None or x not in state.neighbours(b):
        raise ArrowApplyError(
            ApplyReason.UNKNOWN_ARROW, f"leaving bond {b}-{leaving} is not a bond at the sink"
        )
    state.decrement(b, x)
    state.edit.add((x,), +1)


def _step(state: EditState, tokens: ReactantTokens, arrow: PredictedArrow, flags: set):
    """Source and new bond of one arrow; returns (b, whether b had an open shell before)."""
    b = _mol_index(tokens, arrow.acceptor_site)
    room = open_shell(state, b)
    _source(state, tokens, arrow, flags)
    state.edit.add((_mol_index(tokens, arrow.donor_site), b), +1)
    return b, room


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
            b, _ = _step(state, tokens, arrow, flags)
            _leave(state, tokens, b, arrow.leaving)
        product = sanitize_product(apply_edit(state.base, state.edit))
        return ApplyResult(product, flags=frozenset(flags))
    except ArrowApplyError as exc:
        return ApplyResult(None, exc.reason, exc.detail, flags=frozenset(flags))


def _consumed(state: EditState, tokens: ReactantTokens, b: int, x: int) -> bool:
    j = tokens.mol_atom.get(x)
    return j is not None and _pair(b, j) in state.consumed


@dataclass(frozen=True)
class LeavingLabel:
    leaving: int  # x of the annotated leaving bond, or NONE_LEAVING
    candidates: tuple[LeavingCandidate, ...]
    none_allowed: bool
    in_mask: bool  # the annotated choice is one the mask allows
    breaks_bd: int | None  # diagnostic: the leaving bond is s's partner bond


def _leaving_label(cands, none_ok: bool, wanted: list[int]) -> LeavingLabel:
    atoms = [c.atom for c in cands]
    hit = [x for x in wanted if x in atoms]
    leaving = hit[0] if hit else NONE_LEAVING
    in_mask = (bool(hit) or not wanted) and (leaving != NONE_LEAVING or none_ok)
    partner = next((c.atom for c in cands if c.is_partner), None)
    breaks = None if partner is None else int(leaving == partner)
    return LeavingLabel(leaving, tuple(cands), none_ok, in_mask, breaks)


def leaving_labels(
    tokens: ReactantTokens, arrows: list[PredictedArrow], labels: list[ArrowLabel]
) -> list[LeavingLabel | None]:
    """The leaving bond of every arrow of a dataset step (`arrows`: its teacher tuples, in the
    order of `labels`), read as `apply_predicted_arrows` reads them: the bond b-x whose leaving
    arrow the step has (`label.sink_partners`), else NONE, with the candidates and the mask of
    the edited state. A partner bond that an earlier arrow of the step has already consumed
    (a relay: the C=C pair of an enolate that alkylates has moved before O- adds to C=O) is
    not this arrow's leaving bond. The state advances with these labels. An arrow the edit
    cannot read (and every later one) gets None."""
    state = EditState(Chem.Mol(tokens.mol))
    out: list[LeavingLabel | None] = []
    try:
        for arrow, label in zip(arrows, labels, strict=True):
            b, room = _step(state, tokens, arrow, set())
            cands, none_ok = leaving_choices(state, tokens, arrow, room)
            gone = {x for x in label.sink_partners if _consumed(state, tokens, b, x)}
            wanted = [x for x in label.sink_partners if x not in gone]
            out.append(_leaving_label(cands, none_ok, wanted))
            _leave(state, tokens, b, out[-1].leaving)
    except ArrowApplyError:
        pass
    return out + [None] * (len(labels) - len(out))


def with_leaving(label: ArrowLabel, lab: LeavingLabel | None) -> ArrowLabel:
    """The label with its leaving bond (LEAVING_OUTSIDE_MASK when the mask excludes it)."""
    if lab is None:
        return label
    flags = label.flags | ({AlignFlag.LEAVING_OUTSIDE_MASK} if not lab.in_mask else set())
    return replace(
        label,
        leaving_bond=lab.leaving,
        leaving_candidates=lab.candidates,
        leaving_none_allowed=lab.none_allowed,
        breaks_bd=lab.breaks_bd,
        flags=frozenset(flags),
    )


def labeled_arrows(
    tokens: ReactantTokens, labels: list[ArrowLabel], valid=None, with_leaving: bool = True
) -> list[PredictedArrow]:
    """The teacher representatives (r*, a*, s*, b*) of aligned labels, as predicted arrows;
    with `valid(r, s)` the teacher pair is chosen among valid pairs, as the samples do. With
    `with_leaving` each arrow carries its labeled leaving bond (`leaving_labels`)."""
    arrows = [
        PredictedArrow(r, lab.donor_site, s, lab.acceptor_site)
        for lab in labels
        for r, s in [representatives(tokens, lab, valid)]
    ]
    if not with_leaving:
        return arrows
    leaving = leaving_labels(tokens, arrows, labels)
    return [
        a if lab is None else replace(a, leaving=lab.leaving)
        for a, lab in zip(arrows, leaving, strict=True)
    ]


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
