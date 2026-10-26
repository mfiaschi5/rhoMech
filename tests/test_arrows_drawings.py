"""Resonance drawings of a reactant pool (spec v9, "Resonance augmentation"): the forms a molecule
is drawn in, the label recomputed on a drawing, and the sample in that drawing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rdkit import Chem

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.delta_be import (
    EditLabel,
    ReactionMatrix,
    drawn_label,
    edit_label,
    step_edit,
    without_kekule_cycles,
)
from ksmech.arrows.drawings import combinations, kekule_closest, molecule_forms
from ksmech.arrows.lewis import pair_bond_orders
from ksmech.arrows.steps import MechanismSample, load_samples, sample_for_step
from ksmech.data import chem
from ksmech.data.schema import Arrow, ElementaryStep
from ksmech.physics.stage_a import A
from tests.fixtures.molecules import MOLECULES
from tests.test_arrows_steps import _stored

# acetone's enolate, drawn with its charge on O, alkylated on C by chloromethane
ALKYLATION = ElementaryStep(
    "enolate-alkylation",
    "pmechdb",
    None,
    None,
    "[CH3:1][C:20](=[CH2:21])[O-:22].[CH3:30][Cl:31]",
    "[CH3:1][C:20](=[O:22])[CH2:21][CH3:30].[Cl-:31]",
    [
        Arrow("lone_pair", (22,), "atom", (20,)),
        Arrow("bond", (20, 21), "bond", (21, 30)),
        Arrow("bond", (30, 31), "atom", (31,)),
    ],
)


def _forms(smiles: str) -> list[str]:
    return [Chem.MolToSmiles(m) for m in molecule_forms(chem.parse_mapped(smiles))]


def _by_map(mol: Chem.Mol, r: ReactionMatrix) -> tuple[set, set]:
    m = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    return {(*sorted((m[i], m[j])), d) for i, j, d in r.bonds}, {(m[i], d) for i, d in r.lone}


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_a_neutral_molecule_is_drawn_only_as_drawn(name):
    """No fixture molecule has a second form without charge separation (acrolein's
    [CH2+]C=C[O-], acetonitrile's C[C+]=[N-] and benzene's zwitterions are never drawn)."""
    assert len(_forms(MOLECULES[name])) == 1


def test_an_enolate_and_a_carboxylate_have_their_charge_on_either_end():
    assert _forms("C=C(C)[O-]") == ["C=C(C)[O-]", "[CH2-]C(C)=O"]
    forms = molecule_forms(chem.parse_mapped("CC(=O)[O-]"))
    assert [[a.GetIdx() for a in m.GetAtoms() if a.GetFormalCharge()] for m in forms] == [[3], [2]]


def test_a_benzyl_cation_keeps_the_drawn_ring_wherever_it_can():
    """The charge of a 3-phenylbenzyl cation moves round its own ring (ortho, ortho', para),
    never onto the phenyl. RDKit's structures swap the phenyl's Kekule structure in three of
    the four forms; each form keeps the drawn one, so it differs from the drawing only along
    the charge's path."""
    drawn = chem.parse_mapped("[CH2+]C1=CC(=CC=C1)C1=CC=CC=C1")
    forms = molecule_forms(drawn)
    assert len(forms) == 4
    other = [b.GetIdx() for b in drawn.GetBonds() if b.GetBeginAtomIdx() >= 7]
    for form in forms:
        kept = [form.GetBondWithIdx(k).GetBondType() for k in other]
        assert kept == [drawn.GetBondWithIdx(k).GetBondType() for k in other]


def test_a_kekule_swap_is_undone_but_a_pericyclic_ring_is_not():
    benzene = chem.parse_mapped("C1=CC=CC=C1")
    swapped = Chem.RWMol(benzene)
    for b in swapped.GetBonds():
        single = b.GetBondType() == Chem.BondType.SINGLE
        b.SetBondType(Chem.BondType.DOUBLE if single else Chem.BondType.SINGLE)
    assert kekule_closest(benzene, swapped.GetMol()).GetBondWithIdx(0).GetBondType() == (
        benzene.GetBondWithIdx(0).GetBondType()
    )
    # Diels-Alder: an alternating cycle, but two of its bonds form, so it is the step itself
    diene = chem.parse_mapped("[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]")
    r = ReactionMatrix(((0, 1, -1), (0, 5, 1), (1, 2, 1), (2, 3, -1), (3, 4, 1), (4, 5, -1)), ())
    assert without_kekule_cycles(r, diene) == r


def test_the_label_is_recomputed_on_the_carbanion_drawing():
    """From the O- drawing the step moves the C=C electrons and an O lone pair; from the C-
    drawing it only uses the C lone pair."""
    edit = step_edit(ALKYLATION)
    label = edit_label(ALKYLATION, edit)
    assert isinstance(label, EditLabel)
    assert _by_map(edit.mol, label.variants[0]) == (
        {(20, 21, -1), (20, 22, 1), (21, 30, 1), (30, 31, -1)},
        {(22, -2), (31, 2)},
    )
    mapping: list = []
    frags = Chem.GetMolFrags(
        edit.mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping
    )
    carbanion = molecule_forms(frags[0])[1]
    drawing = Chem.RWMol(edit.mol)
    for b in carbanion.GetBonds():
        i, j = mapping[0][b.GetBeginAtomIdx()], mapping[0][b.GetEndAtomIdx()]
        drawing.GetBondBetweenAtoms(i, j).SetBondType(b.GetBondType())
    for a in carbanion.GetAtoms():
        drawing.GetAtomWithIdx(mapping[0][a.GetIdx()]).SetFormalCharge(a.GetFormalCharge())
    mol = drawing.GetMol()
    mol.UpdatePropertyCache(strict=False)
    other = drawn_label(ALKYLATION, edit, label, mol)
    assert isinstance(other, EditLabel) and len(other.variants) == 1
    assert _by_map(mol, other.variants[0]) == ({(21, 30, 1), (30, 31, -1)}, {(21, -2), (31, 2)})


def test_the_sample_in_its_other_drawing(token_loader):
    s = sample_for_step(ALKYLATION, token_loader)
    assert isinstance(s, MechanismSample) and s.n_drawings == 2 and not s.drawing_skips
    d = s.drawn(1)
    assert s.drawn(0) is s and d.drawings == []
    assert int(np.abs(d.atom_categorical[:, 1]).sum()) == int(
        np.abs(s.atom_categorical[:, 1]).sum()
    )
    assert not np.array_equal(d.atom_categorical[:, 1], s.atom_categorical[:, 1])
    assert (
        np.array_equal(d.bonds[:, :2], s.bonds[:, :2])
        and d.bonds[:, 2].sum() == s.bonds[:, 2].sum()
    )
    assert d.lone_pairs.sum() == s.lone_pairs.sum()
    assert sum(v.size for v in d.edit_label.variants) == 4
    assert sum(v.size for v in s.edit_label.variants) == 6
    for x in (s, d):  # bond_order read on each drawing's own structure
        orders = pair_bond_orders(x.bonds, x.incidence, x.orbital_occupied, x.n_atoms)
        col = x.pair_z[..., A["bond_order"]]
        assert np.array_equal(col, np.where(x.pair_defined[..., A["bond_order"]], orders, 0))
    assert not np.array_equal(d.pair_z, s.pair_z)
    rest = [k for k in range(s.pair_z.shape[-1]) if k != A["bond_order"]]
    assert np.array_equal(d.pair_z[..., rest], s.pair_z[..., rest])
    b = collate_samples([s, d])
    assert not torch.equal(b["bond_order"][0], b["bond_order"][1])
    assert not torch.equal(b["can_lose_pair"][0], b["can_lose_pair"][1])


def test_drawings_round_trip_through_the_store(token_loader, tmp_path):
    s = sample_for_step(ALKYLATION, token_loader)
    (back,) = load_samples(_stored(tmp_path, {s.step_id: s}))
    assert back.n_drawings == s.n_drawings == 2
    for a, b in zip(back.drawings, s.drawings, strict=True):
        assert a.edit_label == b.edit_label
        for name in ("charges", "orders", "lone_pairs"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.dtype == y.dtype == np.int8 and np.array_equal(x, y), name
    valid = s.pair_valid
    assert np.allclose(back.drawn(1).pair_z[valid], s.drawn(1).pair_z[valid], rtol=1e-3, atol=1e-6)


def test_combinations_take_every_choice_or_a_random_few():
    rng = np.random.default_rng(0)
    assert combinations([2, 1, 3], 8, rng) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ]
    few = combinations([4, 4], 8, rng)
    assert few[0] == (0, 0) and len(set(few)) == 8 and all(max(c) < 4 for c in few)


def test_the_training_loader_draws_each_sample_in_its_drawings(token_loader):
    from ksmech.train.loop import TrainConfig, make_loader

    s = sample_for_step(ALKYLATION, token_loader)
    cfg = TrainConfig(workers=0, cell_budget=10**7)
    loader = make_loader([s] * 8, cfg, True, draw=True)
    seen = set()
    for epoch in range(3):
        loader.batch_sampler.set_epoch(epoch)
        for b in loader:
            seen |= {tuple(row.tolist()) for row in b["atom_categorical"][:, :, 1]}
    assert seen == {tuple(s.atom_categorical[:, 1]), tuple(s.drawings[0].charges)}
    (b,) = list(make_loader([s], cfg, False))  # validation: the dataset's drawing
    assert b["atom_categorical"][0, :, 1].tolist() == s.atom_categorical[:, 1].tolist()


def test_a_class_holds_only_elementary_steps():
    """A triple bond that the product needs single changes its order by 2, which no decoder can
    write: such a matrix never enters a class (a class left empty is NOT_ELEMENTARY),
    `check_edit` refuses it, and the trainer drops a stored drawing that has one."""
    from ksmech.arrows.delta_be import matrix_between, parsimonious_class
    from ksmech.arrows.drawings import Drawing
    from ksmech.arrows.lewis import check_edit
    from ksmech.train.loop import drop_undecodable_drawings

    reac = chem.parse_mapped("[CH:1]#[CH:2]")
    r = matrix_between(reac, chem.parse_mapped("[CH:1][CH:2]"))
    assert r == ReactionMatrix(((0, 1, -2),), ((0, 2), (1, 2))) and not r.is_elementary()
    assert parsimonious_class(reac, reac, r, forms=([], 0))[0] == ()
    label = EditLabel((r,), (r.anchors(),), True, 0, False)
    bonds = np.array([[0, 1, 3]], dtype=np.int16)
    assert check_edit(label, bonds, np.zeros(2, dtype=np.int8)) == "NOT_ELEMENTARY"
    z = np.zeros(2, dtype=np.int8)
    bad = Drawing(z, np.ones(1, dtype=np.int8), z, label)

    class Stub:
        step_id, drawings = "s", [bad]

    s = Stub()
    assert drop_undecodable_drawings([s], "train") == 1 and s.drawings == []
