"""Edit labels (spec v8, "Edit labels"): the reaction matrix R of an annotated step and its
parsimonious resonance class.

`R_data` is the dataset's arrows applied as one bond-electron edit of the drawn reactants
(`arrows.apply`), checked against the annotated products modulo resonance. The label is the
parsimonious class of `R_data`: every R whose product `B_reac + R` is a resonance structure of
the annotated product, chosen first by the fewest formal charges in that product structure (its
dominant resonance contributor) and then by the fewest changed entries (`parsimonious_class`).
Counting entries alone would prefer charge-separated forms, which a drawn reactant often reaches
in fewer changes (a hydrazone as N(-)-C=OH(+): 441 PMechDB steps; docs/decisions.md 2026-10-24).
The product's structures are enumerated with the resonance key's settings, capped at
`RESONANCE_CAP`, and `R_data` is always a candidate. Arrows are used for nothing else.

A ReactionMatrix holds the non-zero entries: bond-order changes (i < j, delta +-1) and changes
of non-bonding electrons (delta +-2), on reactant-mol atom indices; `to_global` moves them to
the atoms of the reactant tokens. An anchor (spec v8, Edit decoder) is a bond order that rises
(a +1 entry: a new bond, or a pi bond on a bonded pair), or, in a step without one, a bond order
that falls.

The class is built in order of cost among the matrices whose product passes the resonance-key
check that evaluation uses (`reproduces`). The key comes from a capped enumeration that is not
perfectly form-invariant: from some charge-separated forms it does not find its way back, so a
resonance form of the product can fail it (262 PMechDB steps). Such members are dropped and
counted; `R_data` always passes, so the class is never empty.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum

from rdkit import Chem

from ksmech.arrows.apply import (
    TYPE_TO_ORDER,
    ApplyResult,
    ArrowApplyError,
    BEEdit,
    apply_edit,
    arrow_moves,
    edit_from_moves,
    nonbonding_electrons,
    sanitize_product,
)
from ksmech.arrows.compare import MATCH_STATUSES, match_products
from ksmech.data import chem
from ksmech.data.resonance import frozen_hydrogens, resonance_forms
from ksmech.data.schema import ElementaryStep

RESONANCE_CAP = 256  # product resonance structures enumerated for the label class


class EditSkip(StrEnum):
    EDIT_FAILED = "EDIT_FAILED"  # the dataset arrows do not apply as one edit
    PRODUCT_MISMATCH = "PRODUCT_MISMATCH"  # the edit does not give the annotated product
    RESONANCE_STEP = "RESONANCE_STEP"  # the parsimonious class is the empty matrix
    NO_ANCHOR = "NO_ANCHOR"  # no bond order changes (never expected)
    ATOM_WITHOUT_TOKEN = "ATOM_WITHOUT_TOKEN"  # an edited atom has no token atom


def _order(mol: Chem.Mol, i: int, j: int) -> int:
    bond = mol.GetBondBetweenAtoms(i, j)
    return 0 if bond is None else TYPE_TO_ORDER.get(bond.GetBondType(), 1)


@dataclass(frozen=True, order=True)
class ReactionMatrix:
    bonds: tuple[tuple[int, int, int], ...]  # (i, j, delta) with i < j, delta != 0
    lone: tuple[tuple[int, int], ...]  # (i, delta) non-bonding electron change, delta != 0

    @property
    def size(self) -> int:
        return len(self.bonds) + len(self.lone)

    def to_edit(self) -> BEEdit:
        edit = BEEdit()
        for i, j, d in self.bonds:
            edit.bond_delta[(i, j)] += d
        for i, d in self.lone:
            edit.lone_delta[i] += d
        return edit

    def mapped(self, index: dict[int, int]) -> ReactionMatrix:
        """The same matrix on other atom indices (`index[i]` for every atom)."""
        bonds = tuple(sorted((*sorted((index[i], index[j])), d) for i, j, d in self.bonds))
        return ReactionMatrix(bonds, tuple(sorted((index[i], d) for i, d in self.lone)))

    def anchors(self) -> tuple[tuple[int, int, str], ...]:
        """Admissible anchors: the bond orders that rise, else those that fall."""
        rising = tuple((i, j, "form") for i, j, d in self.bonds if d > 0)
        return rising or tuple((i, j, "break") for i, j, d in self.bonds if d < 0)

    def to_list(self) -> list:
        return [[list(b) for b in self.bonds], [list(x) for x in self.lone]]

    @classmethod
    def from_list(cls, v: list) -> ReactionMatrix:
        return cls(tuple(tuple(b) for b in v[0]), tuple(tuple(x) for x in v[1]))


def matrix_between(reac: Chem.Mol, prod: Chem.Mol) -> ReactionMatrix:
    """R = B(prod) - B(reac) for two molecules on the same atom indices."""
    pairs = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for m in (reac, prod)
        for b in m.GetBonds()
    }
    bonds = [(i, j, _order(prod, i, j) - _order(reac, i, j)) for i, j in sorted(pairs)]
    lone = [
        (
            a,
            nonbonding_electrons(prod.GetAtomWithIdx(a))
            - nonbonding_electrons(reac.GetAtomWithIdx(a)),
        )
        for a in range(reac.GetNumAtoms())
    ]
    return ReactionMatrix(tuple(b for b in bonds if b[2]), tuple(x for x in lone if x[1]))


Anchor = tuple[int, int, str]  # (i, j, "form" | "break"), i < j


@dataclass(frozen=True)
class EditLabel:
    variants: tuple[ReactionMatrix, ...]  # the parsimonious class
    anchors: tuple[tuple[Anchor, ...], ...]  # admissible anchors of each variant
    data_parsimonious: bool  # R_data itself is in the class
    n_structures: int  # product resonance structures enumerated
    truncated: bool  # the enumeration hit RESONANCE_CAP
    n_rejected: int = 0  # cheaper resonance variants the key check rejected

    def atoms(self) -> set[int]:
        return {a for v in self.variants for i, j, _ in v.bonds for a in (i, j)} | {
            i for v in self.variants for i, _ in v.lone
        }

    def mapped(self, index: dict[int, int]) -> EditLabel:
        """The label on other atom indices (`index[i]` for every edited atom)."""
        anchors = tuple(
            tuple(sorted((*sorted((index[i], index[j])), s) for i, j, s in a)) for a in self.anchors
        )
        return replace(
            self, variants=tuple(v.mapped(index) for v in self.variants), anchors=anchors
        )

    def to_dict(self) -> dict:
        return {
            "variants": [v.to_list() for v in self.variants],
            "anchors": [[list(x) for x in a] for a in self.anchors],
            "data_parsimonious": self.data_parsimonious,
            "n_structures": self.n_structures,
            "truncated": self.truncated,
            "n_rejected": self.n_rejected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EditLabel:
        anchors = tuple(tuple((int(i), int(j), str(s)) for i, j, s in a) for a in d["anchors"])
        variants = tuple(ReactionMatrix.from_list(v) for v in d["variants"])
        flags = (d["data_parsimonious"], d["n_structures"], d["truncated"], d.get("n_rejected", 0))
        return cls(variants, anchors, *flags)


def dataset_matrix(step: ElementaryStep) -> tuple[Chem.Mol, Chem.Mol, ReactionMatrix] | EditSkip:
    """(reactants, product, R_data) of an annotated step, or why it has none."""
    mol = chem.parse_mapped(step.reactants_smiles)
    try:
        moves = [m for arrow in step.arrows for m in arrow_moves(arrow)]
        prod = sanitize_product(apply_edit(mol, edit_from_moves(moves, chem.map_to_index(mol))))
    except (ArrowApplyError, NotImplementedError):
        return EditSkip.EDIT_FAILED
    want = chem.parse_mapped(step.products_smiles)
    if match_products(ApplyResult(prod), want, step.products_smiles)[0] not in MATCH_STATUSES:
        return EditSkip.PRODUCT_MISMATCH
    return mol, prod, matrix_between(mol, prod)


def _charges(mol: Chem.Mol) -> int:
    return sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())


def _costs(reac: Chem.Mol, prod: Chem.Mol, r_data: ReactionMatrix) -> tuple[dict, int]:
    """(R -> (formal charges of its product structure, entries), structures enumerated). The
    structures are the resonance forms of the product with its H frozen, Kekule structures as
    drawn (`resonance.resonance_forms`); a product that does not sanitize has only `r_data`."""
    cost = {r_data: (_charges(prod), r_data.size)}
    frozen = frozen_hydrogens(prod)
    forms, n = ([], 0) if frozen is None else resonance_forms(frozen, RESONANCE_CAP)
    for form in forms:
        r = matrix_between(reac, form)
        cost[r] = min(cost.get(r, (_charges(form), r.size)), (_charges(form), r.size))
    return cost, n


def parsimonious_class(
    reac: Chem.Mol, prod: Chem.Mol, r_data: ReactionMatrix, accept=lambda r: True
) -> tuple[tuple[ReactionMatrix, ...], int, bool, int]:
    """(the parsimonious class sorted, structures enumerated, truncated, members rejected) of
    `r_data`: the R of lowest cost (fewest formal charges in the product structure, then fewest
    entries) that `accept` (the key check) accepts; `r_data` is always accepted."""
    cost, n = _costs(reac, prod, r_data)
    rejected = 0
    for level in sorted(set(cost.values())):
        members = sorted(r for r, c in cost.items() if c == level)
        ok = [r for r in members if r == r_data or accept(r)]
        rejected += len(members) - len(ok)
        if ok:
            return tuple(ok), n, n >= RESONANCE_CAP, rejected
    raise AssertionError("r_data is always accepted")


def reproduces(step: ElementaryStep, mol: Chem.Mol, r: ReactionMatrix) -> bool:
    """`B_reac + R` is the annotated product modulo resonance."""
    try:
        prod = sanitize_product(apply_edit(mol, r.to_edit()))
    except ArrowApplyError:
        return False
    want = chem.parse_mapped(step.products_smiles)
    return match_products(ApplyResult(prod), want, step.products_smiles)[0] in MATCH_STATUSES


def edit_label(step: ElementaryStep) -> EditLabel | EditSkip:
    """The edit label of an annotated step on reactant-mol indices, or why it has none."""
    out = dataset_matrix(step)
    if isinstance(out, EditSkip):
        return out
    mol, prod, r_data = out
    accept = lambda r: reproduces(step, mol, r)  # noqa: E731
    variants, n, truncated, rejected = parsimonious_class(mol, prod, r_data, accept)
    if variants[0].size == 0:
        return EditSkip.RESONANCE_STEP
    anchors = tuple(v.anchors() for v in variants)
    if not all(anchors):
        return EditSkip.NO_ANCHOR
    return EditLabel(variants, anchors, r_data in variants, n, truncated, rejected)


def to_global(label: EditLabel, global_atom: dict[int, int]) -> EditLabel | EditSkip:
    """The label on the global atoms of the reactant tokens (`ReactantTokens.global_atom`)."""
    if not label.atoms() <= global_atom.keys():
        return EditSkip.ATOM_WITHOUT_TOKEN
    return label.mapped(global_atom)


def step_label(step: ElementaryStep, loader) -> EditLabel | str:
    """The edit label of a step on the global atoms of its reactant tokens, or the reason code
    it has none (an EditSkip, or MISSING_TOKENS / CORRESPONDENCE from the tokens)."""
    from ksmech.arrows.reactants import CorrespondenceError, MissingTokens, ReactantTokens

    label = edit_label(step)
    if isinstance(label, EditSkip):
        return str(label)
    try:
        rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
    except MissingTokens:
        return "MISSING_TOKENS"
    except CorrespondenceError:
        return "CORRESPONDENCE"
    out = to_global(label, rt.global_atom)
    return str(out) if isinstance(out, EditSkip) else out


def label_report(labels: dict[str, EditLabel | str]) -> dict:
    """Coverage and shape of the edit labels of many steps (step id -> label or reason)."""
    ok = [x for x in labels.values() if isinstance(x, EditLabel)]
    skipped = Counter(x for x in labels.values() if isinstance(x, str))
    sizes = Counter(min(len(x.variants), 4) for x in ok)
    kinds = Counter(a[0][2] for x in ok for a in x.anchors[:1])
    return {
        "n_steps": len(labels),
        "n_labeled": len(ok),
        "labeled_fraction": len(ok) / len(labels) if labels else 0.0,
        "skipped": dict(skipped.most_common()),
        "class_size": {("4+" if k == 4 else str(k)): v for k, v in sorted(sizes.items())},
        "data_parsimonious": sum(x.data_parsimonious for x in ok),
        "resonance_truncated": sum(x.truncated for x in ok),
        "steps_with_rejected_variants": sum(x.n_rejected > 0 for x in ok),
        "anchor_kind": dict(kinds),
        "bond_changes": dict(sorted(Counter(len(x.variants[0].bonds) for x in ok).items())),
    }
