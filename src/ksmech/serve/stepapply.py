"""Check and apply one elementary step given as a reaction matrix (spec v9, "Applying a step").

A step is R = B_prod - B_reac on the pool atoms (the atom indices of
`parse_mapped(state.mapped_smiles)`, every H an atom): bond-order changes (i < j) and changes of
non-bonding electrons. It is applied by adding it to B on an RDKit molecule
(`arrows.apply.apply_edit`, H counts frozen) and sanitizing (`sanitize_product`, Kekule form
kept); formal charges follow from electron counting. Nothing is inferred: the leaving bond, the
source of every pair and every relay are entries of R.

`check_matrix` reports every problem with a code and the atoms involved, before and after the
edit (`ISSUE_CODES`); `apply_matrix` returns the product state, or the issues. A matrix whose
product has the parent's pool key changes no sigma bond or H: a resonance interconversion, not a
step (`RESONANCE_ONLY_STEP`, spec "Resonance").

Serve may only import upstream: `ReactionMatrix`, `apply_edit`, `sanitize_product`,
`nonbonding_electrons` are used as they are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rdkit import Chem

from ksmech.serve import keys
from ksmech.serve.errors import ServeError

if TYPE_CHECKING:
    from ksmech.arrows.delta_be import ReactionMatrix
    from ksmech.serve.contracts import PoolState

EMPTY_MATRIX = "EMPTY_MATRIX"
MATRIX_MALFORMED = "MATRIX_MALFORMED"  # an index outside the pool, i >= j, a delta not allowed
CHARGE_NOT_CONSERVED = "CHARGE_NOT_CONSERVED"
BOND_ORDER_RANGE = "BOND_ORDER_RANGE"  # a bond order would leave 0..3
BREAK_UNBONDED = "BREAK_UNBONDED"  # a negative bond change on a pair without a bond
LONE_PAIR_NEGATIVE = "LONE_PAIR_NEGATIVE"  # non-bonding electrons would go below 0
NO_LONE_PAIR = "NO_LONE_PAIR"  # a -2 on an atom without a lone pair
VALENCE_EXCEEDED = "VALENCE_EXCEEDED"  # RDKit valence error, or more than an octet (2 for H)
SANITIZE_FAILED = "SANITIZE_FAILED"
RESONANCE_ONLY_STEP = "RESONANCE_ONLY_STEP"
NO_ARROW_READING = "NO_ARROW_READING"  # R has no decomposition into electron-pair moves
ISSUE_CODES = (
    EMPTY_MATRIX,
    MATRIX_MALFORMED,
    CHARGE_NOT_CONSERVED,
    BOND_ORDER_RANGE,
    BREAK_UNBONDED,
    LONE_PAIR_NEGATIVE,
    NO_LONE_PAIR,
    VALENCE_EXCEEDED,
    SANITIZE_FAILED,
    RESONANCE_ONLY_STEP,
    NO_ARROW_READING,
)
BOND_DELTAS = (-3, -2, -1, 1, 2, 3)  # PMechDB uses +-1; larger ones are allowed by the spec
LONE_DELTAS = (-6, -4, -2, 2, 4, 6)
_ATOM_IN_MESSAGE = re.compile(r"atom # ?(\d+)")


@dataclass(frozen=True)
class StepIssue:
    code: str
    message: str
    atoms: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "atoms": list(self.atoms)}


@dataclass
class Applied:
    """The product of a matrix, or why there is none."""

    product: Chem.Mol | None
    state: PoolState | None
    issues: list[StepIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.product is not None and not self.issues


def matrix_of(d: Any) -> ReactionMatrix:
    """A `ReactionMatrix` from `MatrixIO` ({bonds: [[i, j, d]], lone: [[i, d]]}), its
    `to_list()` form or a ReactionMatrix; pairs are sorted (i < j), entries summed and zero
    entries dropped. Raises ServeError(MATRIX_MALFORMED) on a shape it cannot read."""
    from ksmech.arrows.delta_be import ReactionMatrix

    if isinstance(d, ReactionMatrix):
        return d
    if hasattr(d, "model_dump"):
        d = d.model_dump()
    try:
        bonds, lone = (d["bonds"], d["lone"]) if isinstance(d, dict) else (d[0], d[1])
        b: dict[tuple[int, int], int] = {}
        for i, j, x in bonds:
            key = (min(int(i), int(j)), max(int(i), int(j)))
            b[key] = b.get(key, 0) + int(x)
        n: dict[int, int] = {}
        for i, x in lone:
            n[int(i)] = n.get(int(i), 0) + int(x)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ServeError(MATRIX_MALFORMED, f"unreadable matrix: {exc}", status=422) from exc
    return ReactionMatrix(
        tuple(sorted((i, j, x) for (i, j), x in b.items() if x)),
        tuple(sorted((i, x) for i, x in n.items() if x)),
    )


def matrix_io(R: ReactionMatrix) -> dict:
    """`MatrixIO` of a matrix."""
    return {"bonds": [list(b) for b in R.bonds], "lone": [list(x) for x in R.lone]}


def matrix_atoms(R: ReactionMatrix) -> list[int]:
    return sorted({a for i, j, _ in R.bonds for a in (i, j)} | {i for i, _ in R.lone})


def _order(mol: Chem.Mol, i: int, j: int) -> int:
    from ksmech.arrows.apply import TYPE_TO_ORDER

    bond = mol.GetBondBetweenAtoms(i, j)
    return 0 if bond is None else TYPE_TO_ORDER.get(bond.GetBondType(), 1)


def _shape_issues(mol: Chem.Mol, R: ReactionMatrix) -> list[StepIssue]:
    n = mol.GetNumAtoms()
    out = []
    for i, j, d in R.bonds:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            out.append(
                StepIssue(
                    MATRIX_MALFORMED,
                    f"bond ({i}, {j}) is not a pair of pool atoms 0..{n - 1}",
                    (i, j),
                )
            )
        elif d not in BOND_DELTAS:
            out.append(
                StepIssue(
                    MATRIX_MALFORMED, f"bond ({i}, {j}) change {d} outside {BOND_DELTAS}", (i, j)
                )
            )
    for i, d in R.lone:
        if not 0 <= i < n:
            out.append(StepIssue(MATRIX_MALFORMED, f"atom {i} is not a pool atom 0..{n - 1}", (i,)))
        elif d not in LONE_DELTAS:
            out.append(
                StepIssue(MATRIX_MALFORMED, f"atom {i} lone-electron change {d} is not even", (i,))
            )
    return out


def _charge_issue(R: ReactionMatrix) -> list[StepIssue]:
    total = sum(d for _, d in R.lone) + 2 * sum(d for *_, d in R.bonds)
    if total == 0:
        return []
    msg = f"the matrix changes the electron count by {total} (sum R_ii + 2 sum R_ij must be 0)"
    return [StepIssue(CHARGE_NOT_CONSERVED, msg, tuple(matrix_atoms(R)))]


def _bond_issues(mol: Chem.Mol, R: ReactionMatrix) -> list[StepIssue]:
    out = []
    for i, j, d in R.bonds:
        old = _order(mol, i, j)
        if old == 0 and d < 0:
            out.append(StepIssue(BREAK_UNBONDED, f"atoms {i} and {j} are not bonded", (i, j)))
        elif not 0 <= old + d <= 3:
            msg = f"bond {i}-{j} of order {old} would become {old + d}"
            out.append(StepIssue(BOND_ORDER_RANGE, msg, (i, j)))
    return out


def _lone_issues(mol: Chem.Mol, R: ReactionMatrix) -> list[StepIssue]:
    from ksmech.arrows.apply import nonbonding_electrons

    out = []
    for i, d in R.lone:
        n = nonbonding_electrons(mol.GetAtomWithIdx(i))
        if d < 0 and n < 2:
            out.append(StepIssue(NO_LONE_PAIR, f"atom {i} has no lone pair to give", (i,)))
        elif n + d < 0:
            msg = f"atom {i} has {n} non-bonding electrons, the change is {d}"
            out.append(StepIssue(LONE_PAIR_NEGATIVE, msg, (i,)))
    return out


def pre_issues(mol: Chem.Mol, R: ReactionMatrix) -> list[StepIssue]:
    """Every issue that needs no edit: shape, charge, bond orders, lone pairs."""
    if R.size == 0:
        return [StepIssue(EMPTY_MATRIX, "the matrix changes nothing")]
    out = _shape_issues(mol, R)
    if out:
        return out
    return _charge_issue(R) + _bond_issues(mol, R) + _lone_issues(mol, R)


def _octet_issues(prod: Chem.Mol, atoms: list[int]) -> list[StepIssue]:
    """More than 8 valence electrons (2 for H) on a period 1-2 atom of the product."""
    from ksmech.arrows.apply import nonbonding_electrons

    out = []
    for i in atoms:
        atom = prod.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        if z > 10:
            continue
        bonds = sum(_order(prod, i, n.GetIdx()) for n in atom.GetNeighbors()) + atom.GetTotalNumHs()
        electrons = nonbonding_electrons(atom) + 2 * bonds
        cap = 2 if z <= 2 else 8
        if electrons > cap:
            msg = (
                f"atom {i} ({atom.GetSymbol()}) would carry {electrons} valence electrons (> {cap})"
            )
            out.append(StepIssue(VALENCE_EXCEEDED, msg, (i,)))
    return out


def _edit(mol: Chem.Mol, R: ReactionMatrix) -> tuple[Chem.Mol | None, list[StepIssue]]:
    from ksmech.arrows.apply import ApplyReason, ArrowApplyError, apply_edit, sanitize_product

    try:
        prod = sanitize_product(apply_edit(mol, R.to_edit()))
    except ArrowApplyError as exc:
        found = [int(x) for x in _ATOM_IN_MESSAGE.findall(exc.detail)]
        atoms = tuple(found) or tuple(matrix_atoms(R))
        code = VALENCE_EXCEEDED if exc.reason == ApplyReason.VALENCE else SANITIZE_FAILED
        return None, [StepIssue(code, f"{exc.reason}: {exc.detail}", atoms)]
    return prod, _octet_issues(prod, matrix_atoms(R))


def apply_matrix(state: PoolState, R: ReactionMatrix) -> Applied:
    """B + R on the state's molecule, checked; the product state when nothing is wrong."""
    from ksmech.data import chem
    from ksmech.serve.pools import state_from_mol

    mol = chem.parse_mapped(state.mapped_smiles)
    issues = pre_issues(mol, R)
    if issues:
        return Applied(None, None, issues)
    prod, issues = _edit(mol, R)
    if prod is None or issues:
        return Applied(prod, None, issues)
    try:
        child = state_from_mol(prod)
    except ServeError as exc:
        return Applied(
            prod,
            None,
            [StepIssue(SANITIZE_FAILED, f"{exc.code}: {exc.message}", tuple(matrix_atoms(R)))],
        )
    if child.pool_key == state.pool_key:
        msg = f"the product has the parent's pool key {state.pool_key}: a resonance interconversion"
        return Applied(prod, child, [StepIssue(RESONANCE_ONLY_STEP, msg, tuple(matrix_atoms(R)))])
    return Applied(prod, child, [])


def formal_charges(prod: Chem.Mol, R: ReactionMatrix) -> list[list[int]]:
    """[atom, charge] of every product atom that R touches or that carries a charge."""
    atoms = set(matrix_atoms(R))
    atoms |= {a.GetIdx() for a in prod.GetAtoms() if a.GetFormalCharge()}
    return [[i, int(prod.GetAtomWithIdx(i).GetFormalCharge())] for i in sorted(atoms)]


def product_keys(child: PoolState) -> tuple[list[str], list[str]]:
    """(token keys, sigma keys) of the product's fragments, in fragment order."""
    from ksmech.data import chem

    frags = keys.fragment_keys(chem.parse_mapped(child.mapped_smiles))
    return [f.token_key for f in frags], [f.species_key for f in frags]


def touched_maps(state: PoolState, R: ReactionMatrix) -> list[int]:
    """Map numbers of the atoms R touches (the curation independence check)."""
    from ksmech.data import chem

    mol = chem.parse_mapped(state.mapped_smiles)
    return sorted(int(mol.GetAtomWithIdx(i).GetAtomMapNum()) for i in matrix_atoms(R))
