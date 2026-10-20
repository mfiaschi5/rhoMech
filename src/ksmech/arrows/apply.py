"""Apply the arrows of one elementary step as a bond-electron-matrix edit (Ugi-Dugundji).

Every arrow moves one electron pair from a source (a lone pair on an atom, or a bond) to a
destination (a lone pair on an atom, or a bond, new or existing). The arrows of a step are
summed into one net edit before anything touches the molecule, so the order in which they are
listed does not matter and an atom that gains a bond and loses another in the same step ends
up with the right charge. The formal charge of every atom follows from electron counting:

    q = V - N - B

with V the valence electrons of the element, N its nonbonding electrons and B the sum of its
bond orders (hydrogens included). A move takes 2 from N or 1 from a bond order at the source
and adds 2 to N or 1 to a bond order at the destination, so each atom's charge changes by
`-(dN + dB)`. Examples (PMechDB arrow code in brackets):

- lone pair a -> atom b (`a=b`): bond a-b +1; a +1, b -1;
- bond (a, c) -> atom c (`a,c=c`, heterolysis): bond a-c -1; a +1, c -1;
- bond (a, c) -> bond (a, b) (`a,c=a,b`): bond a-c -1, bond a-b +1; c +1, b -1;
- lone pair a -> bond (b, d) with a outside it (`Displacement`): bond a-b +1 and bond b-d -1;
  a +1, d -1. PMechDB never writes this shape; the predicted-arrow path produces it.

Hydrogen counts of the touched atoms are frozen before the edit, so RDKit never re-derives
them from valence. Radical (half-headed) arrows are not implemented yet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

from rdkit import Chem, rdBase

from ksmech.data import chem
from ksmech.data.resonance import ResonanceKey, canonical_resonance_key
from ksmech.data.schema import Arrow, SourceKind

ORDER_TO_TYPE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
TYPE_TO_ORDER = {v: k for k, v in ORDER_TO_TYPE.items()}
_SANITIZE = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY

Site = tuple[int, ...]  # (a,) a lone pair on atom a, (a, b) the bond a-b; map numbers


class ApplyReason(StrEnum):
    VALENCE = "VALENCE"  # impossible electron count or bond order, or RDKit valence error
    SANITIZE = "SANITIZE"  # RDKit rejects the edited molecule for another reason
    UNKNOWN_ARROW = "UNKNOWN_ARROW"  # arrow shape not understood, or names a missing atom/bond


class ArrowApplyError(Exception):
    def __init__(self, reason: ApplyReason, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = ApplyReason(reason)
        self.detail = detail


@dataclass(frozen=True)
class Displacement:
    """A lone pair on `donor` attacks `attacked` and pushes the pair of bond
    (`attacked`, `leaving`) onto `leaving`, drawn as one arrow (concerted sigma displacement).
    Atoms are map numbers; the order is significant, unlike in `Arrow`."""

    donor: int
    attacked: int
    leaving: int


@dataclass(frozen=True)
class Move:
    """One electron pair moving from `src` to `dst` (each a lone-pair or a bond site)."""

    src: Site
    dst: Site


def _bond(a: int, b: int) -> Site:
    return (a, b) if a < b else (b, a)


def arrow_moves(arrow: Arrow | Displacement) -> list[Move]:
    """The electron-pair moves an arrow stands for."""
    if isinstance(arrow, Displacement):
        a, b, d = arrow.donor, arrow.attacked, arrow.leaving
        if len({a, b, d}) != 3:
            raise ArrowApplyError(ApplyReason.UNKNOWN_ARROW, f"degenerate {arrow}")
        return [Move((a,), _bond(a, b)), Move(_bond(b, d), (d,))]
    if not isinstance(arrow, Arrow):
        raise ArrowApplyError(ApplyReason.UNKNOWN_ARROW, f"not an arrow: {arrow!r}")
    if arrow.source_kind is SourceKind.RADICAL:
        raise NotImplementedError("radical (half-headed) arrows: RMechDB extension")
    _, dst = arrow.destination()
    return [Move(tuple(arrow.source_atoms), tuple(dst))]


@dataclass
class BEEdit:
    """Net change of an atomic bond-electron-matrix edit, keyed by atom index."""

    bond_delta: Counter = field(default_factory=Counter)  # (i, j), i < j -> order change
    lone_delta: Counter = field(default_factory=Counter)  # i -> nonbonding-electron change

    def add(self, site: tuple[int, ...], sign: int) -> None:
        if len(site) == 1:
            self.lone_delta[site[0]] += 2 * sign
            return
        i, j = sorted(site)
        if i == j:
            raise ArrowApplyError(ApplyReason.UNKNOWN_ARROW, f"bond from atom {i} to itself")
        self.bond_delta[(i, j)] += sign

    def charge_delta(self) -> Counter:
        """dq = -(dN + dB) per atom."""
        dq: Counter = Counter()
        for i, d in self.lone_delta.items():
            dq[i] -= d
        for (i, j), d in self.bond_delta.items():
            dq[i] -= d
            dq[j] -= d
        return dq

    def atoms(self) -> set[int]:
        return set(self.lone_delta) | {i for pair in self.bond_delta for i in pair}


def edit_from_moves(moves: list[Move], index: dict[int, int]) -> BEEdit:
    """Sum moves (map-number sites) into one edit on atom indices of the molecule."""
    edit = BEEdit()
    for move in moves:
        for site, sign in ((move.src, -1), (move.dst, +1)):
            missing = [m for m in site if m not in index]
            if missing:
                raise ArrowApplyError(ApplyReason.UNKNOWN_ARROW, f"atoms {missing} not in mol")
            edit.add(tuple(index[m] for m in site), sign)
    return edit


def nonbonding_electrons(atom: Chem.Atom) -> int:
    """N = V - q - B, B counting every bond order and every hydrogen."""
    valence = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
    bonds = sum(TYPE_TO_ORDER.get(b.GetBondType(), 1) for b in atom.GetBonds())
    return valence - atom.GetFormalCharge() - bonds - atom.GetTotalNumHs()


def freeze_hydrogens(mol: Chem.RWMol, atoms: set[int]) -> None:
    """Pin the H count of `atoms`, so RDKit does not refill valences after the edit."""
    for i in atoms:
        atom = mol.GetAtomWithIdx(i)
        if not atom.GetNoImplicit():
            atom.SetNumExplicitHs(atom.GetTotalNumHs())
            atom.SetNoImplicit(True)


def _set_bond_order(mol: Chem.RWMol, i: int, j: int, delta: int) -> None:
    bond = mol.GetBondBetweenAtoms(i, j)
    old = 0 if bond is None else TYPE_TO_ORDER.get(bond.GetBondType())
    if old is None:
        raise ArrowApplyError(ApplyReason.UNKNOWN_ARROW, f"bond {i}-{j} is {bond.GetBondType()}")
    new = old + delta
    if new < 0:
        what = "missing" if bond is None else f"order {old}"
        raise ArrowApplyError(ApplyReason.VALENCE, f"bond {i}-{j} {what}, change {delta}")
    if new > 3:
        raise ArrowApplyError(ApplyReason.VALENCE, f"bond {i}-{j} order {new}")
    if new == 0:
        mol.RemoveBond(i, j)
    elif bond is None:
        mol.AddBond(i, j, ORDER_TO_TYPE[new])
    else:
        bond.SetBondType(ORDER_TO_TYPE[new])
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)


def apply_edit(mol: Chem.Mol, edit: BEEdit) -> Chem.RWMol:
    """The edited molecule, unsanitized. Raises ArrowApplyError (VALENCE) for a lone-pair
    source on an atom without one, or a bond order outside 0..3."""
    rw = Chem.RWMol(mol)
    touched = edit.atoms()
    freeze_hydrogens(rw, touched)
    for i, d in edit.lone_delta.items():
        if nonbonding_electrons(rw.GetAtomWithIdx(i)) + d < 0:
            raise ArrowApplyError(ApplyReason.VALENCE, f"atom {i} has no lone pair to give")
    degree = {i: rw.GetAtomWithIdx(i).GetDegree() for i in touched}
    for (i, j), d in sorted(edit.bond_delta.items()):
        if d:
            _set_bond_order(rw, i, j, d)
    for i, d in edit.charge_delta().items():
        atom = rw.GetAtomWithIdx(i)
        atom.SetFormalCharge(atom.GetFormalCharge() + d)
        if atom.GetDegree() != degree[i]:  # neighbour set changed: the parity tag is void
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return rw


def sanitize_product(mol: Chem.RWMol) -> Chem.Mol:
    """Sanitize without aromaticity perception (the Kekulé form stays as edited)."""
    out = mol.GetMol()
    try:
        with rdBase.BlockLogs():
            out.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(out, _SANITIZE)
    except Chem.rdchem.AtomValenceException as exc:
        raise ArrowApplyError(ApplyReason.VALENCE, str(exc)) from exc
    except Chem.rdchem.MolSanitizeException as exc:
        raise ArrowApplyError(ApplyReason.SANITIZE, f"{type(exc).__name__}: {exc}") from exc
    return out


@dataclass
class ApplyResult:
    """The product of an edit, or the reason it failed."""

    mol: Chem.Mol | None
    reason: ApplyReason | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.mol is not None

    @cached_property
    def smiles(self) -> str | None:
        """Canonical Kekulé SMILES with maps: an exact (not modulo resonance) identity."""
        return None if self.mol is None else chem.canonical_mapped_smiles(self.mol)

    @cached_property
    def key(self) -> ResonanceKey | None:
        return None if self.mol is None else canonical_resonance_key(self.mol)


def apply_moves(mol: Chem.Mol, moves: list[Move]) -> ApplyResult:
    """Apply map-number moves as one edit, then sanitize."""
    try:
        edit = edit_from_moves(moves, chem.map_to_index(mol))
        return ApplyResult(sanitize_product(apply_edit(mol, edit)))
    except ArrowApplyError as exc:
        return ApplyResult(None, exc.reason, exc.detail)


def apply_arrows(reactants: Chem.Mol, arrows: list[Arrow | Displacement]) -> ApplyResult:
    """Products of one step: every arrow summed into one BE-matrix edit of `reactants`
    (atom-mapped, as from `chem.parse_mapped`), then sanitized. `result.key` is the
    canonical resonance key. Raises NotImplementedError for radical arrows."""
    try:
        moves = [m for arrow in arrows for m in arrow_moves(arrow)]
    except ArrowApplyError as exc:
        return ApplyResult(None, exc.reason, exc.detail)
    return apply_moves(reactants, moves)
