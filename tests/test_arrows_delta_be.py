"""Edit labels (spec v8): the reaction matrix of a step and its parsimonious resonance class."""

import pytest

from ksmech.arrows.delta_be import (
    EditLabel,
    EditSkip,
    ReactionMatrix,
    dataset_matrix,
    edit_label,
    reproduces,
    to_global,
)
from ksmech.arrows.reactants import ReactantTokens
from ksmech.data import chem
from ksmech.data.schema import Arrow, ElementaryStep
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401


def _step(reactants: str, products: str, arrows: list[Arrow]) -> ElementaryStep:
    return ElementaryStep("t", "pmechdb", None, None, reactants, products, arrows)


def _by_map(step: ElementaryStep, r: ReactionMatrix) -> tuple[set, set]:
    """The entries of `r` on map numbers (unmapped atoms as None)."""
    mol = chem.parse_mapped(step.reactants_smiles)
    m = {a.GetIdx(): a.GetAtomMapNum() or None for a in mol.GetAtoms()}
    bonds = {(*sorted((m[i] or 0, m[j] or 0)), d) for i, j, d in r.bonds}
    return bonds, {(m[i], d) for i, d in r.lone}


def _conserves_charge(r: ReactionMatrix) -> bool:
    return sum(d for _, d in r.lone) + 2 * sum(d for *_, d in r.bonds) == 0


def test_sn2_is_one_forming_and_one_breaking_bond(mini_steps):
    step = mini_steps["sn2"]
    label = edit_label(step)
    assert isinstance(label, EditLabel) and label.data_parsimonious
    (r,) = label.variants
    assert _by_map(step, r) == ({(10, 20, 1), (20, 21, -1)}, {(10, -2), (21, 2)})
    assert _conserves_charge(r)
    ((anchor,),) = label.anchors
    assert anchor[2] == "form"


def test_every_fixture_step_has_a_label_that_reproduces_it(mini_steps):
    for name, step in mini_steps.items():
        label = edit_label(step)
        if name == "resonance":  # carbanion <-> enolate: the class holds the empty matrix
            assert label == EditSkip.RESONANCE_STEP
            continue
        assert isinstance(label, EditLabel), name
        mol = chem.parse_mapped(step.reactants_smiles)
        for r, anchors in zip(label.variants, label.anchors, strict=True):
            assert reproduces(step, mol, r) and _conserves_charge(r) and anchors, name
            assert {abs(d) for *_, d in r.bonds} <= {1} and {abs(d) for _, d in r.lone} <= {2}


def test_a_heterolysis_anchors_on_the_bond_that_breaks():
    step = _step("[CH3:10][Br:11]", "[CH3+:10].[Br-:11]", [Arrow("bond", (10, 11), "atom", (11,))])
    label = edit_label(step)
    assert isinstance(label, EditLabel)
    ((anchor,),) = label.anchors
    assert anchor[2] == "break"


def test_an_ionic_bond_leaving_needs_no_choice_between_pair_and_lone_pair():
    """A lithium alkoxide protonated by H3O+: the O-Li pair forms O-H. As arrows, the O lone
    pair and the O-Li pair are one localized orbital; as a reaction matrix there is one answer."""
    step = _step(
        "[H:20][OH2+:21].[Li:11][O:10]C(C)(C)C",
        "[H:20][O:10]C(C)(C)C.[Li+:11].[OH2:21]",
        [Arrow("bond", (10, 11), "bond", (10, 20)), Arrow("bond", (20, 21), "atom", (21,))],
    )
    label = edit_label(step)
    assert isinstance(label, EditLabel)
    (r,) = label.variants
    assert _by_map(step, r) == ({(10, 11, -1), (10, 20, 1), (20, 21, -1)}, {(21, 2)})


def test_the_mitsunobu_step_keeps_its_neutral_product():
    """Counting changed entries alone would label the hydrazone as N(-)-C=OH(+), which the drawn
    reactant reaches in fewer changes; the fewest-charges rule keeps the neutral product."""
    step = _step(
        "CCO[P:11](C)(C)(C)[N:10](NC(=O)OC)[C:20](OC)=[OH+:21]",
        "CCO[P+:11](C)(C)C.COC(=O)N[N:10]=[C:20](OC)[OH:21]",
        [Arrow("bond", (10, 11), "bond", (10, 20)), Arrow("bond", (20, 21), "atom", (21,))],
    )
    label = edit_label(step)
    assert isinstance(label, EditLabel) and label.data_parsimonious
    (r,) = label.variants
    bonds, lone = _by_map(step, r)
    assert bonds == {(10, 11, -1), (10, 20, 1), (20, 21, -1)} and lone == {(21, 2)}


def test_the_neutral_product_form_wins_over_a_charge_separated_one():
    """Acetate alkylated through a drawn pi relay: O12 pushes into C=O11 and O11 attacks. The
    charge-separated ester C(-O12-)=O11(+)CH3 is reached in fewer changes from the drawn
    reactant (4 entries against 6), but the label is the neutral ester the dataset draws."""
    step = _step(
        "C[C:10](=[O:11])[O-:12].[CH3:20][Cl:21]",
        "C[C:10](=[O:12])[O:11][CH3:20].[Cl-:21]",
        [
            Arrow("lone_pair", (12,), "atom", (10,)),
            Arrow("bond", (10, 11), "bond", (11, 20)),
            Arrow("bond", (20, 21), "atom", (21,)),
        ],
    )
    label = edit_label(step)
    assert isinstance(label, EditLabel) and label.data_parsimonious
    (r,) = label.variants
    bonds, lone = _by_map(step, r)
    assert bonds == {(10, 11, -1), (10, 12, 1), (11, 20, 1), (20, 21, -1)}
    assert lone == {(12, -2), (21, 2)}


def test_a_mismatched_annotation_is_rejected():
    step = _step(
        "[OH-:10].[CH3:20][Cl:21]",
        "[OH:10][CH2:20]C.[Cl-:21]",  # the fixture's broken line: a carbon appears
        [Arrow("lone_pair", (10,), "atom", (20,)), Arrow("bond", (20, 21), "atom", (21,))],
    )
    assert dataset_matrix(step) == EditSkip.PRODUCT_MISMATCH
    assert edit_label(step) == EditSkip.PRODUCT_MISMATCH


def test_labels_serialize_and_map_onto_the_token_atoms(mini_steps, token_loader):
    step = mini_steps["e2"]
    label = edit_label(step)
    assert isinstance(label, EditLabel)
    assert EditLabel.from_dict(label.to_dict()) == label
    rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
    glob = to_global(label, rt.global_atom)
    assert isinstance(glob, EditLabel)
    back = {g: m for m, g in rt.global_atom.items()}
    for r, g in zip(label.variants, glob.variants, strict=True):
        assert sorted((*sorted((back[i], back[j])), d) for i, j, d in g.bonds) == sorted(r.bonds)
        for i, j, _ in g.bonds:  # the token atoms are the drawn atoms
            for a in (i, j):
                f = int(rt.atom_fragment[a])
                atom = rt.fragments[f].tokens.atoms[a - int(rt.atom_offsets[f])]
                assert atom.atomic_number == rt.mol.GetAtomWithIdx(back[a]).GetAtomicNum()
    assert to_global(label, {}) == EditSkip.ATOM_WITHOUT_TOKEN


@pytest.mark.parametrize("delta", [1, -1])
def test_reaction_matrix_round_trips_through_lists(delta):
    r = ReactionMatrix(((0, 3, delta),), ((0, -2 * delta),))
    assert ReactionMatrix.from_list(r.to_list()) == r
    assert r.mapped({0: 7, 3: 2}) == ReactionMatrix(((2, 7, delta),), ((7, -2 * delta),))


def test_edit_labels_cli_reports_coverage_and_gates_it(
    mini_processed,  # noqa: F811
    token_loader,
    monkeypatch,
):
    import json

    import ksmech.arrows.cli as cli

    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: missing_iodide(token_loader))
    args = ["edit-labels", "--source", "pmechdb", "--processed-dir", str(mini_processed)]
    code = cli.main(args)
    report = json.loads((mini_processed / "pmechdb" / "edit_labels_report.json").read_text())
    assert report["n_steps"] == 5 and report["n_labeled"] == 4  # the resonance step is not one
    assert report["skipped"] == {"MISSING_TOKENS": 1}
    assert code == 1  # 80% labeled: below the 95% gate


def test_a_pi_bond_that_forms_is_an_anchor():
    """An amine lone pair closing an iminium on a carbocation forms no new sigma bond: the C-N
    order rises from 1 to 2, and that entry is the anchor."""
    step = _step(
        "S[CH+:20][NH2:10]", "S[CH:20]=[NH2+:10]", [Arrow("lone_pair", (10,), "atom", (20,))]
    )
    label = edit_label(step)
    assert isinstance(label, EditLabel)
    (r,) = label.variants
    assert _by_map(step, r) == ({(10, 20, 1)}, {(10, -2)})
    ((anchor,),) = label.anchors
    assert anchor[2] == "form"


def test_variants_the_key_check_rejects_give_way_to_the_next_cost_level():
    """A thiolate next to the carbocation: the product's dominant contributor is the neutral
    thioformamide S=C-NH2, reached in fewer changes than the drawn S(-)/iminium. If the key
    check rejected it, the class falls back level by level, at worst to the drawn R."""
    from ksmech.arrows.delta_be import parsimonious_class

    step = _step(
        "[S-][CH+:20][NH2:10]", "[S-][CH:20]=[NH2+:10]", [Arrow("lone_pair", (10,), "atom", (20,))]
    )
    mol, prod, r_data = dataset_matrix(step)
    best, _, _, rejected = parsimonious_class(mol, prod, r_data)
    assert r_data not in best and rejected == 0 and all(_conserves_charge(r) for r in best)
    fallback, _, _, rejected = parsimonious_class(mol, prod, r_data, accept=lambda r: False)
    assert r_data in fallback and rejected >= len(best)


@pytest.mark.parametrize(
    ("reactants", "products", "arrows"),
    [
        (  # methyl azide formed
            "[N-:1]=[N+:2]=[N-:3].[CH3:4][Br:5]",
            "[N-:1]=[N+:2]=[N:3][CH3:4].[Br-:5]",
            [Arrow("lone_pair", (3,), "atom", (4,)), Arrow("bond", (4, 5), "atom", (5,))],
        ),
        (  # an azide ion looking on at an SN2
            "[N-]=[N+]=[N-].[OH-:1].[CH3:2][Cl:3]",
            "[N-]=[N+]=[N-].[OH:1][CH3:2].[Cl-:3]",
            [Arrow("lone_pair", (1,), "atom", (2,)), Arrow("bond", (2, 3), "atom", (3,))],
        ),
    ],
)
def test_a_radical_azide_never_wins_the_fewest_charges_rule(reactants, products, arrows):
    """RDKit's enumeration also returns an azide as the diradical N=N[N], with fewer formal
    charges than N=[N+]=[N-]. It is not a resonance form of the closed-shell product, so the
    drawn R stays the one parsimonious label."""
    label = edit_label(_step(reactants, products, arrows))
    assert isinstance(label, EditLabel) and label.data_parsimonious
    assert len(label.variants) == 1
