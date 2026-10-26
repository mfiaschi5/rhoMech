"""Resonance drawings of a sample's reactant pool (spec v9, "Resonance augmentation").

A pool is drawn as one of its resonance drawings. The formal charges of the atom tokens, the
matrix B on which R is decoded and the pair feature `bond_order` belong to the drawing; the
wavefunction, so every orbital token and every other feature, does not. In training each
molecule of the pool is drawn as a random one of its forms and the label is recomputed on that
drawing (`delta_be.drawn_label`), so the model learns the step, not the drawing.

A molecule's forms are its resonance forms (`data.resonance.resonance_forms`: H frozen, odd
structures dropped) with no more formal charges (the sum of |q|) than the molecule as drawn
(user decision, 2026-10-26): an enolate's charge on C or O, a carboxylate's on either O, the
ends of an allyl ion. The charge-separated minor forms are never drawn: RDKit returns 50 or more
structures for 9% of the PMechDB reactant molecules, nearly all of them charge-separated, so
uniform draws over every form would mostly be zwitterions. A form keeps the drawn Kekule
structure wherever it does not need another (`kekule_closest`): RDKit's structure of a cation
delocalized into one ring of a trityl also swaps the other two, and the label would then swap
them back.

`resonance_drawings` keeps up to MAX_DRAWINGS drawings of the pool per sample, the dataset's own
counted: every combination of one form per molecule when there are no more, else a random choice
seeded by the step id, so a build is reproducible. A drawing whose label cannot be recomputed,
or is no valid edit of it (`lewis.check_edit`), is dropped and counted by reason.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from ksmech.arrows.apply import ORDER_TO_TYPE
from ksmech.arrows.delta_be import (
    EditLabel,
    EditSkip,
    StepEdit,
    _order,
    drawn_label,
    matrix_between,
    to_global,
    without_kekule_cycles,
)
from ksmech.arrows.lewis import check_edit, lewis_bonds, lewis_lone_pairs
from ksmech.arrows.reactants import ReactantTokens
from ksmech.data.resonance import frozen_hydrogens, resonance_forms
from ksmech.data.schema import ElementaryStep

MAX_DRAWINGS = 8  # pool drawings per sample, the dataset's own included
FORM_CAP = 256  # structures RDKit enumerates per molecule


@dataclass(frozen=True, eq=False)
class Drawing:
    """Another resonance drawing of a sample's reactant pool, on its global atoms."""

    charges: np.ndarray  # (A,) int8 formal charge of every atom token
    orders: np.ndarray  # (E,) int8 bond order of every row of the sample's `bonds`
    lone_pairs: np.ndarray  # (A,) int8
    edit_label: EditLabel  # the label class recomputed on this drawing


def _charges(mol: Chem.Mol) -> int:
    return sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())


def _structure(mol: Chem.Mol) -> tuple:
    bonds = sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        + (str(b.GetBondType()),)
        for b in mol.GetBonds()
    )
    return tuple(bonds), tuple(a.GetFormalCharge() for a in mol.GetAtoms())


def kekule_closest(drawn: Chem.Mol, form: Chem.Mol) -> Chem.Mol:
    """`form` (a resonance form of `drawn` on its atom indices) with its Kekule cycles relative
    to `drawn` undone (`delta_be.without_kekule_cycles`): its charges and lone pairs, the drawn
    bond orders wherever the form does not need others."""
    r = without_kekule_cycles(matrix_between(drawn, form), drawn)
    rw = Chem.RWMol(drawn)
    for i, j, d in r.bonds:
        rw.GetBondBetweenAtoms(i, j).SetBondType(ORDER_TO_TYPE[_order(drawn, i, j) + d])
    for a in form.GetAtoms():
        rw.GetAtomWithIdx(a.GetIdx()).SetFormalCharge(a.GetFormalCharge())
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def molecule_forms(frag: Chem.Mol) -> list[Chem.Mol]:
    """The forms one molecule may be drawn in: as drawn first (H frozen), then its other
    resonance forms with no more formal charges, each closest to the drawn Kekule structure
    (`kekule_closest`) and each structure once, in RDKit's order. Only the drawn one when the
    molecule does not sanitize."""
    frozen = frozen_hydrogens(frag)
    if frozen is None:
        return [frag]
    out, seen, q = [frozen], {_structure(frozen)}, _charges(frozen)
    for form in resonance_forms(frozen, FORM_CAP)[0]:
        if any(b.GetIsAromatic() for b in form.GetBonds()) or _charges(form) > q:
            continue
        form = kekule_closest(frozen, form)
        s = _structure(form)
        if s not in seen:
            seen.add(s)
            out.append(form)
    return out


def combinations(counts: list[int], limit: int, rng: np.random.Generator) -> list[tuple]:
    """Up to `limit` distinct choices of one form per molecule, the all-drawn choice first:
    all of them when there are no more, else the rest uniformly at random."""
    if math.prod(counts) <= limit:
        return list(itertools.product(*(range(n) for n in counts)))
    out = [tuple(0 for _ in counts)]
    seen = set(out)
    while len(out) < limit:
        choice = tuple(int(rng.integers(n)) for n in counts)
        if choice not in seen:
            seen.add(choice)
            out.append(choice)
    return out


def pool_drawing(base: Chem.Mol, parts: list[tuple[tuple[int, ...], Chem.Mol]]) -> Chem.Mol:
    """`base` (the pool with its H frozen) with each molecule's bond types and formal charges
    taken from a form, `parts` holding (the molecule's pool atom indices, the form)."""
    rw = Chem.RWMol(base)
    for atoms, form in parts:
        for b in form.GetBonds():
            i, j = atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()]
            rw.GetBondBetweenAtoms(i, j).SetBondType(b.GetBondType())
        for a in form.GetAtoms():
            rw.GetAtomWithIdx(atoms[a.GetIdx()]).SetFormalCharge(a.GetFormalCharge())
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def _seed(step_id: str) -> int:
    return int.from_bytes(hashlib.sha256(step_id.encode()).digest()[:8], "little")


def drawing_of(
    rt: ReactantTokens, mol: Chem.Mol, label: EditLabel, bonds: np.ndarray, charges: np.ndarray
) -> Drawing | str:
    """The Drawing of the pool drawing `mol` with its label on the global atoms, or why the
    label is no valid edit of it. `bonds` and `charges` are the sample's (the drawn pool)."""
    new = lewis_bonds(rt, mol)
    if not np.array_equal(new[:, :2], bonds[:, :2]):
        raise ValueError("a resonance drawing changed the sigma bonds")
    lone = lewis_lone_pairs(rt, mol)
    problem = check_edit(label, new, lone)
    if problem is not None:
        return problem
    q = charges.astype(np.int8).copy()
    for m, g in rt.global_atom.items():
        q[g] = mol.GetAtomWithIdx(m).GetFormalCharge()
    return Drawing(q, new[:, 2].astype(np.int8), lone, label)


def resonance_drawings(
    step: ElementaryStep,
    rt: ReactantTokens,
    edit: StepEdit,
    label: EditLabel,
    bonds: np.ndarray,
    charges: np.ndarray,
    limit: int = MAX_DRAWINGS,
) -> tuple[list[Drawing], Counter]:
    """The other drawings of the step's reactant pool (at most `limit - 1`) and the drawings
    dropped, by reason. `label` is the drawn step's label on the reactant-mol atoms, `bonds`
    and `charges` the sample's."""
    base = frozen_hydrogens(rt.mol)
    if base is None:
        return [], Counter()
    mapping: list = []
    frags = Chem.GetMolFrags(rt.mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    forms = [molecule_forms(f) for f in frags]
    rng = np.random.default_rng(_seed(step.step_id))
    out, dropped = [], Counter()
    for choice in combinations([len(f) for f in forms], limit, rng)[1:]:
        parts = [(tuple(a), f[k]) for a, f, k in zip(mapping, forms, choice, strict=True)]
        mol = pool_drawing(base, parts)
        lab = drawn_label(step, edit, label, mol)
        if not isinstance(lab, EditSkip):
            lab = to_global(lab, rt.global_atom)
        if isinstance(lab, EditSkip):
            dropped[str(lab)] += 1
            continue
        drawing = drawing_of(rt, mol, lab, bonds, charges)
        if isinstance(drawing, str):
            dropped[drawing] += 1
        else:
            out.append(drawing)
    return out, dropped
