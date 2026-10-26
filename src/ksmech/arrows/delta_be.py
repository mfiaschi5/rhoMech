"""Edit labels (spec v8, "Edit labels"): the reaction matrix R of an annotated step and its
parsimonious resonance class.

`R_data` is the dataset's arrows applied as one bond-electron edit of the drawn reactants
(`arrows.apply`), checked against the annotated products modulo resonance. The label is the
parsimonious class of `R_data`: every R whose product `B_reac + R` is a resonance structure of
the annotated product, chosen first by the fewest formal charges in that product structure (its
dominant resonance contributor) and then by the fewest changed entries (`parsimonious_class`).
Counting entries alone would prefer charge-separated forms, which a drawn reactant often reaches
in fewer changes (a hydrazone as N(-)-C=OH(+): 441 PMechDB steps; docs/decisions.md 2026-10-24).
The product's structures are its resonance forms (`data.resonance.resonance_forms`, H
frozen), capped at `RESONANCE_CAP`, and `R_data` is always a candidate. Arrows are used for
nothing else.

A ReactionMatrix holds the non-zero entries: bond-order changes (i < j, delta +-1) and changes
of non-bonding electrons (delta +-2), on reactant-mol atom indices; `to_global` moves them to
the atoms of the reactant tokens. An anchor (spec v8, Edit decoder) is a bond order that rises
(a +1 entry: a new bond, or a pi bond on a bonded pair), or, in a step without one, a bond order
that falls.

The class is built in order of cost among the matrices whose product passes the check that
evaluation uses (`reproduces`: the same sigma key as the annotated product). Every resonance
form has the product's sigma key, so a member fails only when its edit cannot be applied; such
members are dropped and counted, and `R_data` always passes, so the class is never empty.
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


MAX_CYCLE_SEARCH = 20_000  # path extensions tried per alternating-cycle search


def _alternating_cycle(edges: dict[tuple[int, int], int]) -> list[tuple[int, int]] | None:
    """A cycle of `edges` (pair -> +1 or -1) whose signs alternate all the way round, as its
    pairs; None when there is none (or the search gives up after MAX_CYCLE_SEARCH steps)."""
    adj: dict[int, list[tuple[int, int]]] = {}
    for (i, j), d in edges.items():
        adj.setdefault(i, []).append((j, d))
        adj.setdefault(j, []).append((i, d))
    budget = MAX_CYCLE_SEARCH
    for start in sorted(adj):
        stack = [(start, 0, (start,), ())]
        while stack and budget > 0:
            v, last, atoms, path = stack.pop()
            for w, d in adj[v]:
                budget -= 1
                e = (min(v, w), max(v, w))
                if d == last or e in path:
                    continue
                if w == start and len(path) >= 3 and d != edges[path[0]]:
                    return [*path, e]
                if w not in atoms:
                    stack.append((w, d, (*atoms, w), (*path, e)))
    return None


def without_kekule_cycles(r: ReactionMatrix, reac: Chem.Mol) -> ReactionMatrix:
    """`r` with every alternating cycle of pi-bond changes removed: a Kekule swap of a ring that
    the change does not need. Along such a cycle each atom gains one bond order and loses one,
    so removing it keeps every atom's bond total, charge and lone pairs, and `reac + r` stays a
    valid structure with the same sigma bonds; only entries on bonds of `reac` that stay bonds
    (order >= 1 before and after) take part. Used on the resonance drawings of a reactant
    (`arrows.drawings.kekule_closest`), not on the label candidates: there it would also make a
    carbocation form as cheap as the iminium the dataset draws, which the label's cost (formal
    charges, then entries) cannot tell apart (docs/decisions.md, 2026-10-26)."""
    pi = {
        (i, j): d
        for i, j, d in r.bonds
        if abs(d) == 1 and _order(reac, i, j) >= 1 and _order(reac, i, j) + d >= 1
    }
    removed: set[tuple[int, int]] = set()
    while (cycle := _alternating_cycle(pi)) is not None:
        removed.update(cycle)
        for e in cycle:
            del pi[e]
    if not removed:
        return r
    return ReactionMatrix(tuple(b for b in r.bonds if (b[0], b[1]) not in removed), r.lone)


def product_forms(prod: Chem.Mol) -> tuple[list[Chem.Mol], int]:
    """(the resonance forms of the product with its H frozen, Kekule structures as drawn
    (`resonance.resonance_forms`), the structures RDKit counted); none when the product does
    not sanitize."""
    frozen = frozen_hydrogens(prod)
    return ([], 0) if frozen is None else resonance_forms(frozen, RESONANCE_CAP)


def _costs(reac: Chem.Mol, prod: Chem.Mol, r_data: ReactionMatrix, structures) -> dict:
    """R -> (formal charges of its product structure, entries), over `r_data` and the matrices
    that take `reac` to each of `structures`."""
    cost = {r_data: (_charges(prod), r_data.size)}
    for form in structures:
        r = matrix_between(reac, form)
        cost[r] = min(cost.get(r, (_charges(form), r.size)), (_charges(form), r.size))
    return cost


def parsimonious_class(
    reac: Chem.Mol,
    prod: Chem.Mol,
    r_data: ReactionMatrix,
    accept=lambda r: True,
    forms: tuple[list[Chem.Mol], int] | None = None,
    extra: tuple[Chem.Mol, ...] = (),
) -> tuple[tuple[ReactionMatrix, ...], int, bool, int]:
    """(the parsimonious class sorted, structures enumerated, truncated, members rejected) of
    `r_data`: the R of lowest cost (fewest formal charges in the product structure, then fewest
    entries) that `accept` (the key check) accepts; `r_data` is always accepted. The product
    structures are `forms` (`product_forms(prod)` unless given) and `extra`, more resonance
    structures of the product."""
    structures, n = product_forms(prod) if forms is None else forms
    cost = _costs(reac, prod, r_data, (*structures, *extra))
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


@dataclass(frozen=True)
class StepEdit:
    """The dataset's edit of a step: the reactants as drawn, the product, R_data and the
    product's resonance forms (`product_forms`), enumerated once for every drawing."""

    mol: Chem.Mol
    prod: Chem.Mol
    r_data: ReactionMatrix
    forms: tuple[list[Chem.Mol], int]


def step_edit(step: ElementaryStep) -> StepEdit | EditSkip:
    out = dataset_matrix(step)
    if isinstance(out, EditSkip):
        return out
    return StepEdit(*out, product_forms(out[1]))


def _label(step: ElementaryStep, mol: Chem.Mol, edit: StepEdit, r_data, extra=()):
    accept = lambda r: reproduces(step, mol, r)  # noqa: E731
    variants, n, truncated, rejected = parsimonious_class(
        mol, edit.prod, r_data, accept, edit.forms, extra
    )
    if variants[0].size == 0:
        return EditSkip.RESONANCE_STEP
    anchors = tuple(v.anchors() for v in variants)
    if not all(anchors):
        return EditSkip.NO_ANCHOR
    return EditLabel(variants, anchors, r_data in variants, n, truncated, rejected)


def edit_label(step: ElementaryStep, edit: StepEdit | None = None) -> EditLabel | EditSkip:
    """The edit label of an annotated step on reactant-mol indices, or why it has none."""
    edit = step_edit(step) if edit is None else edit
    if isinstance(edit, EditSkip):
        return edit
    return _label(step, edit.mol, edit, edit.r_data)


def drawn_label(
    step: ElementaryStep, edit: StepEdit, label: EditLabel, drawing: Chem.Mol
) -> EditLabel | EditSkip:
    """The edit label of `step` on another resonance drawing of its reactants (reactant-mol
    indices; spec v9, "Resonance augmentation"): the parsimonious class with the drawing's B in
    place of the drawn one. The drawn edit's product plays the part of R_data (the drawing
    reaches it with `matrix_between`), and the structures that the members of the drawn
    `label` reach from the drawing are candidates too: a drawing that differs far from the
    reaction centre (a carboxylate written with its other O-) keeps the drawn class even
    where RDKit's capped enumeration lacks the matching product structure."""
    reached = []
    for r in label.variants:
        try:
            reached.append(sanitize_product(apply_edit(drawing, r.to_edit())))
        except ArrowApplyError:
            continue
    return _label(step, drawing, edit, matrix_between(drawing, edit.prod), tuple(reached))


def to_global(label: EditLabel, global_atom: dict[int, int]) -> EditLabel | EditSkip:
    """The label on the global atoms of the reactant tokens (`ReactantTokens.global_atom`)."""
    if not label.atoms() <= global_atom.keys():
        return EditSkip.ATOM_WITHOUT_TOKEN
    return label.mapped(global_atom)


def step_label(step: ElementaryStep, loader) -> EditLabel | str:
    """The edit label of a step on the global atoms of its reactant tokens, or the reason code
    it has none (an EditSkip, or from the tokens MISSING_TOKENS / STALE_ARTIFACT /
    NO_BOND_INDEX (`MissingTokens.reason`) or CORRESPONDENCE)."""
    from ksmech.arrows.reactants import CorrespondenceError, MissingTokens, ReactantTokens

    label = edit_label(step)
    if isinstance(label, EditSkip):
        return str(label)
    try:
        rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
    except MissingTokens as exc:
        return exc.reason
    except CorrespondenceError:
        return "CORRESPONDENCE"
    out = to_global(label, rt.global_atom)
    return str(out) if isinstance(out, EditSkip) else out


def class_shape(labels: list[EditLabel]) -> dict:
    """The class sizes (1, 2, 3, 4+) and the kind of the first anchor of `labels`."""
    sizes = Counter(min(len(x.variants), 4) for x in labels)
    return {
        "class_size": {("4+" if k == 4 else str(k)): v for k, v in sorted(sizes.items())},
        "anchor_kind": dict(Counter(x.anchors[0][0][2] for x in labels).most_common()),
    }


def label_report(labels: dict[str, EditLabel | str]) -> dict:
    """Coverage and shape of the edit labels of many steps (step id -> label or reason)."""
    ok = [x for x in labels.values() if isinstance(x, EditLabel)]
    skipped = Counter(x for x in labels.values() if isinstance(x, str))
    return {
        "n_steps": len(labels),
        "n_labeled": len(ok),
        "labeled_fraction": len(ok) / len(labels) if labels else 0.0,
        "skipped": dict(skipped.most_common()),
        **class_shape(ok),
        "data_parsimonious": sum(x.data_parsimonious for x in ok),
        "resonance_truncated": sum(x.truncated for x in ok),
        "steps_with_rejected_variants": sum(x.n_rejected > 0 for x in ok),
        "bond_changes": dict(sorted(Counter(len(x.variants[0].bonds) for x in ok).items())),
    }
