"""Edit labels (spec v8, "Edit labels"): the reaction matrix R of an annotated step and its
parsimonious resonance class.

`R_data` is the dataset's arrows applied as one bond-electron edit of the drawn reactants
(`arrows.apply`), checked against the annotated products modulo resonance; a source without
arrows (ReactMech) gives it as the difference of its two mapped sides (`mapped_matrix`), with
the hydrogens that move written as atoms first (`arrows.hydrogens`) and the product redrawn in
the Kekule structure closest to the reactant (`mapped_closest`, rule KEKULE_CLOSEST, spec
v9.1): the two sides are kekulized independently, and an unchanged ring written the other way
round would enter R as a swap the step does not make. The label is the
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
from ksmech.arrows.hydrogens import explicit_moving_hydrogens
from ksmech.data import chem
from ksmech.data.resonance import (
    balanced_flow,
    frozen_hydrogens,
    radical_forms,
    resonance_forms,
)
from ksmech.data.schema import ElementaryStep, SourceKind

RESONANCE_CAP = 256  # product resonance structures enumerated for the label class


class EditSkip(StrEnum):
    EDIT_FAILED = "EDIT_FAILED"  # the dataset arrows do not apply as one edit
    PRODUCT_MISMATCH = "PRODUCT_MISMATCH"  # the edit does not give the annotated product
    RESONANCE_STEP = "RESONANCE_STEP"  # the parsimonious class is the empty matrix
    NO_ANCHOR = "NO_ANCHOR"  # no bond order changes (never expected)
    ATOM_WITHOUT_TOKEN = "ATOM_WITHOUT_TOKEN"  # an edited atom has no token atom
    NOT_ELEMENTARY = "NOT_ELEMENTARY"  # no candidate is an elementary step (`is_elementary`)
    H_TRANSFER_AMBIGUOUS = "H_TRANSFER_AMBIGUOUS"  # no unique donor -> acceptor pairing of the
    #   hydrogens that move (`arrows.hydrogens`; steps without arrows only)
    RADICAL_ARROW = "RADICAL_ARROW"  # a half-headed arrow: one-electron edits are not built yet
    SET_STEP = "SET_STEP"  # R has atom entries only: an electron transfer, no bond changes (P6)
    DISSOCIATIVE_ET_STEP = "DISSOCIATIVE_ET_STEP"  # bond entries, but the electron flow is
    #   unbalanced in every variant of the class: an electron transfer coupled to a cleavage


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

    def is_elementary(self, open_shell: bool = False) -> bool:
        """Every bond order changes by 1 and every atom's non-bonding electrons by 2: a step of
        spec v9, and all the decoder can write ({-1, 0, +1} per pair, {-2, 0, +2} per atom).
        With `open_shell` (the source's label rule, `domain.open_shell_labels`) an atom entry
        may also be +-1: homolysis, recombination, radical addition and transfer."""
        atoms = (1, 2) if open_shell else (2,)
        return all(abs(d) == 1 for *_, d in self.bonds) and all(
            abs(d) in atoms for _, d in self.lone
        )

    def balanced(self) -> bool:
        """The electron-flow law holds (`data.resonance.balanced_flow`)."""
        return balanced_flow(self.bonds, self.lone)

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
    n_symmetric: int = 0  # members added as images under the pool's automorphisms
    symmetry_capped: bool = False  # that closure was cut short (`arrows.symmetry`)
    kekule_closest: int = 0  # entries R_data lost as Kekule swaps (rule KEKULE_CLOSEST)

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
            "n_symmetric": self.n_symmetric,
            "symmetry_capped": self.symmetry_capped,
            "kekule_closest": self.kekule_closest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EditLabel:
        anchors = tuple(tuple((int(i), int(j), str(s)) for i, j, s in a) for a in d["anchors"])
        variants = tuple(ReactionMatrix.from_list(v) for v in d["variants"])
        flags = (d["data_parsimonious"], d["n_structures"], d["truncated"], d.get("n_rejected", 0))
        symmetry = (d.get("n_symmetric", 0), d.get("symmetry_capped", False))
        return cls(variants, anchors, *flags, *symmetry, d.get("kekule_closest", 0))


def mapped_matrix(reac: Chem.Mol, prod: Chem.Mol) -> ReactionMatrix:
    """R between the two mapped sides of a step without arrows, on the reactant indices: every
    mapped atom is read on its product partner (the hydrogens that move must be mapped atoms:
    `hydrogens.explicit_moving_hydrogens`). Unmapped atoms are taken as unchanged; the product
    check of `dataset_matrix` rejects a step where they are not."""
    partner = {a.GetAtomMapNum(): a.GetIdx() for a in prod.GetAtoms() if a.GetAtomMapNum()}
    index = {
        a.GetIdx(): partner[a.GetAtomMapNum()]
        for a in reac.GetAtoms()
        if a.GetAtomMapNum() in partner
    }
    back = {p: i for i, p in index.items()}
    pairs = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in reac.GetBonds()}
    pairs |= {
        tuple(sorted((back[b.GetBeginAtomIdx()], back[b.GetEndAtomIdx()])))
        for b in prod.GetBonds()
        if b.GetBeginAtomIdx() in back and b.GetEndAtomIdx() in back
    }
    bonds = [
        (i, j, _order(prod, index[i], index[j]) - _order(reac, i, j))
        for i, j in sorted(pairs)
        if i in index and j in index
    ]
    lone = [
        (
            i,
            nonbonding_electrons(prod.GetAtomWithIdx(p))
            - nonbonding_electrons(reac.GetAtomWithIdx(i)),
        )
        for i, p in sorted(index.items())
    ]
    return ReactionMatrix(tuple(b for b in bonds if b[2]), tuple(x for x in lone if x[1]))


def mapped_closest(reac: Chem.Mol, prod: Chem.Mol) -> tuple[ReactionMatrix, int]:
    """(`mapped_matrix` with the product in the Kekule structure closest to the reactant, the
    entries this removed): every alternating cycle of pi-bond changes on bonds that stay bonds
    is undone (`without_kekule_cycles`, the helper `drawings.kekule_closest` uses), so a ring
    the step does not change never enters R, however each side was kekulized. Rule
    KEKULE_CLOSEST (spec v9.1); the product's charges, lone pairs and sigma bonds are kept."""
    r = mapped_matrix(reac, prod)
    closest = without_kekule_cycles(r, reac)
    return closest, r.size - closest.size


def dataset_matrix(
    step: ElementaryStep,
) -> tuple[Chem.Mol, Chem.Mol, ReactionMatrix, int] | EditSkip:
    """(reactants, product, R_data, the entries KEKULE_CLOSEST removed) of an annotated step,
    or why it has none: its arrows applied as one edit, or for a step without arrows the edit
    between its mapped sides with the product closest to the reactant (`mapped_closest`), so
    the product applied, checked and enumerated is that Kekule structure."""
    if any(a.source_kind is SourceKind.RADICAL for a in step.arrows):
        return EditSkip.RADICAL_ARROW
    mol = chem.parse_mapped(step.reactants_smiles)
    want = chem.parse_mapped(step.products_smiles)
    removed = 0
    try:
        if step.arrows:
            moves = [m for arrow in step.arrows for m in arrow_moves(arrow)]
            edit = edit_from_moves(moves, chem.map_to_index(mol))
        else:
            r, removed = mapped_closest(mol, want)
            edit = r.to_edit()
        prod = sanitize_product(apply_edit(mol, edit))
    except (ArrowApplyError, NotImplementedError):
        return EditSkip.EDIT_FAILED
    if match_products(ApplyResult(prod), want, step.products_smiles)[0] not in MATCH_STATUSES:
        return EditSkip.PRODUCT_MISMATCH
    return mol, prod, matrix_between(mol, prod), removed


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
    (`arrows.drawings.kekule_closest`) and on R_data of a step without arrows
    (`mapped_closest`), not on the label candidates: there it would also make a carbocation
    form as cheap as the iminium the dataset draws, which the label's cost (formal charges,
    then entries) cannot tell apart (docs/decisions.md, 2026-10-26)."""
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
    (`resonance.resonance_forms`), then a radical product's forms through the anion proxy
    (`resonance.radical_forms`, P6), each on the product's atom indices; the structures RDKit
    counted); none when the product does not sanitize."""
    frozen = frozen_hydrogens(prod)
    if frozen is None:
        return [], 0
    forms, n = resonance_forms(frozen, RESONANCE_CAP)
    return forms + radical_forms(frozen, RESONANCE_CAP), n


def _costs(reac: Chem.Mol, prod: Chem.Mol, r_data: ReactionMatrix, structures) -> dict:
    """R -> (formal charges of its product structure, entries), over `r_data` and the matrices
    that take `reac` to each of `structures`."""
    cost = {r_data: (_charges(prod), r_data.size)}
    for form in structures:
        r = matrix_between(reac, form)
        cost[r] = min(cost.get(r, (_charges(form), r.size)), (_charges(form), r.size))
    return cost


def _cheapest(cost: dict, ok) -> tuple[tuple[ReactionMatrix, ...], int]:
    """(the members `ok` keeps at the lowest cost level where it keeps any, the members it
    dropped up to that level); ((), all dropped) when it keeps none."""
    rejected = 0
    for level in sorted(set(cost.values())):
        members = sorted(r for r, c in cost.items() if c == level)
        kept = [r for r in members if ok(r)]
        rejected += len(members) - len(kept)
        if kept:
            return tuple(kept), rejected
    return (), rejected


def parsimonious_class(
    reac: Chem.Mol,
    prod: Chem.Mol,
    r_data: ReactionMatrix,
    accept=lambda r: True,
    forms: tuple[list[Chem.Mol], int] | None = None,
    extra: tuple[Chem.Mol, ...] = (),
    open_shell: bool = False,
) -> tuple[tuple[ReactionMatrix, ...], int, bool, int]:
    """(the parsimonious class sorted, structures enumerated, truncated, members rejected) of
    `r_data`: the elementary R (`ReactionMatrix.is_elementary`) of lowest cost (fewest formal
    charges in the product structure, then fewest entries) whose electron flow is balanced
    (`ReactionMatrix.balanced`, open-shell plan P6) and that `accept` (the key check)
    accepts; `r_data` is always accepted when it is elementary. Only when no candidate has a
    balanced flow is the class taken without that law (`flow_skip` then names it); it is empty
    when no candidate is elementary. The product structures are `forms` (`product_forms(prod)`
    unless given) and `extra`, more resonance structures of the product. A drawing of the
    reactants with a triple bond that the product needs single has only matrices with a -2
    entry there. `open_shell` is the source's elementary rule. The flow law matters where a
    product form is cheaper by charges but reached only by an electron jump (NO2. as O=N-O.
    from an O-NO2 homolysis)."""
    structures, n = product_forms(prod) if forms is None else forms
    cost = _costs(reac, prod, r_data, (*structures, *extra))

    def ok(r: ReactionMatrix, flow: bool) -> bool:
        if not r.is_elementary(open_shell) or (flow and not r.balanced()):
            return False
        return r == r_data or accept(r)

    variants, rejected = _cheapest(cost, lambda r: ok(r, True))
    if not variants:
        variants, rejected = _cheapest(cost, lambda r: ok(r, False))
    return variants, n, n >= RESONANCE_CAP, rejected


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
    product's resonance forms (`product_forms`), enumerated once for every drawing, and the
    entries KEKULE_CLOSEST removed from R_data (`mapped_closest`)."""

    mol: Chem.Mol
    prod: Chem.Mol
    r_data: ReactionMatrix
    forms: tuple[list[Chem.Mol], int]
    kekule_closest: int = 0


def step_edit(step: ElementaryStep) -> StepEdit | EditSkip:
    out = dataset_matrix(step)
    if isinstance(out, EditSkip):
        return out
    mol, prod, r_data, removed = out
    return StepEdit(mol, prod, r_data, product_forms(prod), removed)


def flow_skip(variants: tuple[ReactionMatrix, ...]) -> EditSkip | None:
    """The flow codes of a non-empty class (open-shell plan P6): SET_STEP when R has atom
    entries only, DISSOCIATIVE_ET_STEP when every variant breaks the electron-flow law."""
    if not variants[0].bonds:
        return EditSkip.SET_STEP
    if not any(v.balanced() for v in variants):
        return EditSkip.DISSOCIATIVE_ET_STEP
    return None


def _label(step: ElementaryStep, mol: Chem.Mol, edit: StepEdit, r_data, extra=()):
    from ksmech.data.domain import open_shell_labels

    accept = lambda r: reproduces(step, mol, r)  # noqa: E731
    variants, n, truncated, rejected = parsimonious_class(
        mol, edit.prod, r_data, accept, edit.forms, extra, open_shell_labels(step.source)
    )
    if not variants:
        return EditSkip.NOT_ELEMENTARY
    if variants[0].size == 0:
        return EditSkip.RESONANCE_STEP
    if (skip := flow_skip(variants)) is not None:
        return skip
    anchors = tuple(v.anchors() for v in variants)
    if not all(anchors):
        return EditSkip.NO_ANCHOR
    flags = (r_data in variants, n, truncated, rejected)
    return EditLabel(variants, anchors, *flags, kekule_closest=edit.kekule_closest)


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
    from ksmech.arrows.drawings import source_label
    from ksmech.arrows.reactants import CorrespondenceError, MissingTokens, ReactantTokens

    step = explicit_moving_hydrogens(step)
    if step is None:
        return str(EditSkip.H_TRANSFER_AMBIGUOUS)
    edit = step_edit(step)
    found = edit if isinstance(edit, EditSkip) else source_label(step, edit)
    if isinstance(found, EditSkip):
        return str(found)
    label = found[2]
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


def smallest_variant(label: EditLabel) -> ReactionMatrix:
    """The member with the fewest entries, then the fewest bond entries. The members of a
    class have one size; where they split it differently between bonds and atoms (280 PMechDB
    training labels in v17), taking fewer bonds gave the smallest caps that still reach every
    label (PMechDB 6 bonds and 2 atoms per part, not 7 and 2)."""
    return min(label.variants, key=lambda v: (v.size, len(v.bonds), v))


def label_counts(labels: list[EditLabel]) -> dict:
    """The rules and the size of `labels`: how many KEKULE_CLOSEST re-made, how many had their
    product's enumeration cut at RESONANCE_CAP, and the label maxima, the largest bond and
    atom entry counts of a label's smallest variant (the decoder caps of `ksmech-eval decode
    --caps labels`, read from a training part)."""
    smallest = [smallest_variant(x) for x in labels]
    return {
        "kekule_closest": sum(x.kekule_closest > 0 for x in labels),
        "resonance_truncated": sum(x.truncated for x in labels),
        "label_maxima": {
            "bonds": max((len(v.bonds) for v in smallest), default=0),
            "atoms": max((len(v.lone) for v in smallest), default=0),
        },
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
        **label_counts(ok),
        "steps_with_rejected_variants": sum(x.n_rejected > 0 for x in ok),
        "bond_changes": dict(sorted(Counter(len(x.variants[0].bonds) for x in ok).items())),
    }
