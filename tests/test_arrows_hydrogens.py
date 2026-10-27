"""Steps without arrows (ReactMech): R from the two mapped sides, the moving hydrogens as atoms."""

from dataclasses import replace

from ksmech.arrows.delta_be import EditLabel, EditSkip, edit_label, reproduces
from ksmech.arrows.hydrogens import explicit_moving_hydrogens, hydrogen_moves
from ksmech.arrows.steps import READING_NO_ARROWS, MechanismSample, sample_for_step
from ksmech.data import chem
from ksmech.data.schema import ElementaryStep
from tests.test_arrows_delta_be import _by_map, _conserves_charge


def _step(reactants: str, products: str) -> ElementaryStep:
    return ElementaryStep("t", "reactmech", None, None, reactants, products)


# ethanol protonates methylamine, the hydrogens written as counts (as ReactMech draws them)
PROTON = _step("[CH3:1][CH2:2][OH:3].[CH3:4][NH2:5]", "[CH3:1][CH2:2][O-:3].[CH3:4][NH3+:5]")
# methoxide on chloromethane: nothing moves but heavy atoms
SN2 = _step("[CH3:1][O-:2].[CH3:3][Cl:4]", "[CH3:1][O:2][CH3:3].[Cl-:4]")


def test_a_moving_proton_becomes_a_mapped_atom_on_both_sides():
    step = explicit_moving_hydrogens(PROTON)
    assert "[H:6]" in step.reactants_smiles and "[H:6]" in step.products_smiles
    reac, prod = (chem.parse_mapped(s) for s in (step.reactants_smiles, step.products_smiles))
    for mol, owner in ((reac, 3), (prod, 5)):
        (h,) = (a for a in mol.GetAtoms() if a.GetAtomMapNum() == 6)
        assert [n.GetAtomMapNum() for n in h.GetNeighbors()] == [owner]
    # the hydrogen counts of every mapped heavy atom are unchanged, the moving H included
    for before, after in ((PROTON.reactants_smiles, reac), (PROTON.products_smiles, prod)):
        old = {a.GetAtomMapNum(): a.GetTotalNumHs() for a in chem.parse_mapped(before).GetAtoms()}
        new = {
            a.GetAtomMapNum(): a.GetTotalNumHs(includeNeighbors=True)
            for a in after.GetAtoms()
            if a.GetAtomMapNum() in old
        }
        assert new == old


def test_a_proton_transfer_without_arrows_breaks_o_h_and_forms_n_h():
    step = explicit_moving_hydrogens(PROTON)
    label = edit_label(step)
    assert isinstance(label, EditLabel) and label.data_parsimonious
    (r,) = label.variants
    assert _by_map(step, r) == ({(3, 6, -1), (5, 6, 1)}, {(3, 2), (5, -2)})
    assert _conserves_charge(r) and reproduces(step, chem.parse_mapped(step.reactants_smiles), r)
    ((anchor,),) = label.anchors
    assert anchor[2] == "form"


def test_a_step_where_no_hydrogen_moves_is_left_as_it_is():
    assert explicit_moving_hydrogens(SN2) is SN2
    label = edit_label(SN2)
    assert isinstance(label, EditLabel)
    (r,) = label.variants
    assert _by_map(SN2, r) == ({(2, 3, 1), (3, 4, -1)}, {(2, -2), (4, 2)})


def test_unmapped_spectators_are_taken_as_unchanged():
    step = _step(PROTON.reactants_smiles + ".CC=O", PROTON.products_smiles + ".CC=O")
    label = edit_label(explicit_moving_hydrogens(step))
    assert isinstance(label, EditLabel)
    (r,) = label.variants
    assert _by_map(explicit_moving_hydrogens(step), r)[0] == {(3, 6, -1), (5, 6, 1)}


def test_a_change_on_an_unmapped_atom_is_a_product_mismatch():
    step = _step(PROTON.reactants_smiles + ".CC=O", PROTON.products_smiles + ".CCO")
    assert edit_label(explicit_moving_hydrogens(step)) == EditSkip.PRODUCT_MISMATCH


def test_two_donors_and_one_acceptor_pair_uniquely():
    reac = chem.parse_mapped("[CH3:1][OH:2].[CH3:3][OH:4].[NH2-:5]")
    prod = chem.parse_mapped("[CH3:1][O-:2].[CH3:3][O-:4].[NH4+:5]")
    assert sorted(hydrogen_moves(reac, prod)) == [(2, 5), (4, 5)]


def test_several_donors_and_several_acceptors_are_ambiguous():
    step = _step(
        "[CH3:1][OH:2].[CH3:3][OH:4].[NH3:5].[NH3:6]",
        "[CH3:1][O-:2].[CH3:3][O-:4].[NH4+:5].[NH4+:6]",
    )
    assert explicit_moving_hydrogens(step) is None


def test_steps_with_arrows_name_their_own_hydrogens(mini_steps):
    for step in mini_steps.values():
        assert explicit_moving_hydrogens(step) is step


def test_the_mapped_sides_give_the_label_the_arrows_give(mini_steps):
    """PMechDB writes its moving hydrogens as atoms: without the arrows, the difference of the
    mapped sides is the same reaction matrix class."""
    for name, step in mini_steps.items():
        with_arrows = edit_label(step)
        without = edit_label(explicit_moving_hydrogens(replace(step, arrows=[])))
        if isinstance(with_arrows, EditSkip):
            assert without == with_arrows, name
            continue
        assert (without.variants, without.anchors) == (with_arrows.variants, with_arrows.anchors)


def test_a_step_without_arrows_is_a_sample_anchored_on_the_moving_hydrogen(token_loader):
    s = sample_for_step(PROTON, token_loader)
    assert isinstance(s, MechanismSample)
    assert s.orbital_reading == READING_NO_ARROWS and s.labels == []
    anchors = {a for v in s.edit_label.anchors for a in v}
    maps = {(*sorted((int(s.atom_map[i]), int(s.atom_map[j]))), kind) for i, j, kind in anchors}
    assert maps == {(5, 6, "form")}
    (h,) = (g for g in range(s.n_atoms) if s.atom_map[g] == 6)
    assert s.atom_categorical[h, 0] == 1
