"""The label class closed under the reactant pool's automorphisms (`arrows.symmetry`), on the
fixture molecules."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from ksmech.arrows.apply import nonbonding_electrons
from ksmech.arrows.delta_be import EditLabel, ReactionMatrix
from ksmech.arrows.lewis import check_edit
from ksmech.arrows.symmetry import symmetric_closure
from tests.fixtures.molecules import MOLECULES


def pool(smiles: str):
    """(mol, elements, formal charges, bonds (i, j, order), lone pairs): every H an atom, the
    Kekule structure as RDKit draws it."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    q = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])
    order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
    bonds = np.array(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                order[b.GetBondType()],
            )
            for b in mol.GetBonds()
        ),
        dtype=np.int16,
    ).reshape(-1, 3)
    lone = np.array([nonbonding_electrons(a) // 2 for a in mol.GetAtoms()], dtype=np.int8)
    return mol, z, q, bonds, lone


def label(bonds: list, lone: list) -> EditLabel:
    r = ReactionMatrix(tuple(sorted(bonds)), tuple(sorted(lone)))
    return EditLabel((r,), (r.anchors(),), True, 1, False)


def first(mol, element: int, bonded_to: int | None = None) -> int:
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == element and (
            bonded_to is None or any(n.GetIdx() == bonded_to for n in a.GetNeighbors())
        ):
            return a.GetIdx()
    raise LookupError(element)


def close(smiles: str, make) -> tuple[EditLabel, np.ndarray, np.ndarray]:
    mol, z, q, bonds, lone = pool(smiles)
    lab = symmetric_closure(make(mol), z, q, bonds)
    for v, anchors in zip(lab.variants, lab.anchors, strict=True):
        one = EditLabel((v,), (anchors,), True, 1, False)
        assert check_edit(one, bonds, lone) is None  # an automorphism keeps every edit valid
        assert anchors == v.anchors()
    return lab, z, bonds


def abstraction(heavy: int):
    """OH- takes a proton from the first H on the first atom of element `heavy`."""

    def make(mol):
        o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == -1)
        c = first(mol, heavy)
        h = first(mol, 1, c)
        return label([(*sorted((o, h)), 1), (*sorted((c, h)), -1)], [(o, -2), (c, 2)])

    return make


@pytest.mark.parametrize(
    ("name", "heavy", "n"),
    [("ethane", 6, 6), ("acetone", 6, 6), ("methylamine", 7, 2), ("ethanol", 8, 1)],
)
def test_a_proton_on_equivalent_atoms_gets_every_equivalent_variant(name, heavy, n):
    """Ethane's six H and acetone's two methyls are one class each; methylamine's two N-H are
    not its C-H; ethanol's O-H is unique."""
    lab, z, _ = close(MOLECULES[name] + ".[OH-]", abstraction(heavy))
    assert len(lab.variants) == n and lab.n_symmetric == n - 1 and not lab.symmetry_capped
    moved = {i for v in lab.variants for i, j, d in v.bonds if d < 0 for i in (i, j) if z[i] == 1}
    assert len(moved) == n


def test_an_sn2_without_symmetric_atoms_is_unchanged():
    def make(mol):
        c, cl = first(mol, 6), first(mol, 17)
        o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == -1)
        return label([(*sorted((o, c)), 1), (*sorted((c, cl)), -1)], [(o, -2), (cl, 2)])

    lab, _, _ = close(MOLECULES["chloromethane"] + ".[OH-]", make)
    assert len(lab.variants) == 1 and lab.n_symmetric == 0


def test_a_kekule_drawing_keeps_only_the_symmetry_it_has():
    """H+ adds to a carbon of Kekule benzene and takes that carbon's double bond: the drawing's
    alternation leaves six such attacks (each carbon, with its own double bond), not twelve."""

    def make(mol):
        h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
        c = first(mol, 6)
        b = next(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(c).GetNeighbors()
            if mol.GetBondBetweenAtoms(c, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        )
        return label([(*sorted((h, c)), 1), (*sorted((c, b)), -1)], [])

    lab, _, bonds = close(MOLECULES["benzene"] + ".[H+]", make)
    double = {(int(i), int(j)) for i, j, o in bonds if o == 2}
    assert len(lab.variants) == 6
    assert all({(i, j) for i, j, d in v.bonds if d < 0} <= double for v in lab.variants)


def test_hydronium_gives_any_of_its_three_protons():
    def make(mol):
        ox = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
        ow = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8 and a.GetIdx() != ox)
        h = first(mol, 1, ox)
        return label([(*sorted((ow, h)), 1), (*sorted((ox, h)), -1)], [(ow, -2), (ox, 2)])

    lab, _, _ = close("[OH3+].O", make)
    assert len(lab.variants) == 3


def test_the_e2_mini_step_takes_a_proton_from_either_methyl(mini_steps, token_loader):
    """2-bromopropane: the dataset names one H of one methyl; its six H are one class."""
    from ksmech.arrows.steps import sample_for_step

    s = sample_for_step(mini_steps["e2"], token_loader)
    lab = s.edit_label
    assert len(lab.variants) == 6 and lab.n_symmetric == 5 and not lab.symmetry_capped
    for v, anchors in zip(lab.variants, lab.anchors, strict=True):
        one = EditLabel((v,), (anchors,), True, 1, False)
        assert check_edit(one, s.bonds, s.lone_pairs) is None
