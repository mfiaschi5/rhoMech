"""Open-shell plan P6: electron bookkeeping (N and u per atom), the electron-flow law and its
codes, the per-source elementary rule, the identity key v2, radical resonance forms through the
anion proxy and the per-drawing atom categorical (Z, charge, u)."""

from dataclasses import fields, replace

import numpy as np
import pytest
from rdkit import Chem

from ksmech.arrows.apply import apply_edit, sanitize_product
from ksmech.arrows.delta_be import (
    EditLabel,
    EditSkip,
    ReactionMatrix,
    edit_label,
    product_forms,
    step_edit,
)
from ksmech.arrows.drawings import molecule_forms, redrawn_label
from ksmech.arrows.hydrogens import explicit_moving_hydrogens
from ksmech.arrows.lewis import check_edit, lewis_unpaired
from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.steps import MechanismSample, reactant_sample, sample_for_step
from ksmech.data import chem, spin
from ksmech.data.resonance import (
    StepKind,
    balanced_flow,
    classify_step,
    frozen_hydrogens,
    identity_key,
    mapped_changes,
    radical_forms,
    resonance_forms,
    sigma_key,
)
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.validation import build_step
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL

P = chem.parse_mapped


def _step(rxn: str, source: Source = Source.RMECHDB, sid: str = "t") -> ElementaryStep:
    r, p = rxn.split(">>")
    return ElementaryStep(sid, source, None, None, chem.canonical_mapped_smiles(P(r)),
                          chem.canonical_mapped_smiles(P(p)))  # fmt: skip


def _label(rxn: str, source: Source = Source.RMECHDB):
    step = explicit_moving_hydrogens(_step(rxn, source))
    return edit_label(step)


# ---- step 1: N and u per atom


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_a_closed_shell_has_even_n_and_no_unpaired_electron(name):
    mol = Chem.MolFromSmiles(MOLECULES[name])
    u, per_atom = spin.spin_of_drawing(mol)
    assert u == 0 and not any(per_atom)


@pytest.mark.parametrize("name", sorted(OPEN_SHELL))
def test_u_counts_the_odd_n_atoms_raised_only_by_the_table(name):
    """u from N: the radicals drawn with one odd atom are doublets, O2 a triplet ([O][O], a Hund
    row); the drawn carbenes and CO pair their electrons (even N), while [N]=[N] and the ozone
    drawings put one on each of two atoms (odd N: u is never lowered; P8 pairs them); Cu(II)
    has odd N; the species table raises this FeBr4- drawing to its high-spin 5."""
    want = {"triplet_o2": 2, "dichlorocarbene": 0, "co_drawn": 0, "n2_drawn": 2,
            "ozone_drawn": 2, "ozone_charge_separated": 2, "copper_dichloride": 1,
            "tetrabromoferrate": 5}  # fmt: skip
    u, per_atom = spin.spin_of_drawing(Chem.MolFromSmiles(OPEN_SHELL[name][0]))
    assert u == want.get(name, 1) and sum(per_atom) == u


def test_closed_shell_samples_have_n_twice_the_lone_pairs_and_no_u(mini_steps, token_loader):
    for step in mini_steps.values():
        s = sample_for_step(step, token_loader)
        if not isinstance(s, MechanismSample):
            continue
        assert np.array_equal(s.nonbonding, 2 * s.lone_pairs.astype(np.int8))
        assert not s.unpaired.any() and not s.atom_categorical[:, 2].any()
        assert s.atom_categorical.shape == (s.n_atoms, 3)
        for k in range(1, s.n_drawings):
            d = s.drawn(k)
            assert np.array_equal(d.nonbonding, 2 * d.lone_pairs) and not d.unpaired.any()


def test_a_live_sample_without_n_fails_collate(mini_steps, token_loader):
    from ksmech.arrows.batches import collate_samples

    step = next(iter(mini_steps.values()))
    rt = ReactantTokens.from_smiles(step.reactants_smiles, token_loader)
    sample = reactant_sample("live", "serve", rt)
    assert len(sample.nonbonding) == sample.n_atoms  # every live caller goes through here
    batch = collate_samples([sample])
    assert (batch["can_lose_pair"][0, : sample.n_atoms].numpy() == (sample.nonbonding >= 2)).all()
    with pytest.raises(ValueError, match="N has 0 atoms"):
        collate_samples([replace(sample, nonbonding=np.zeros(0, dtype=np.int8))])


def test_a_recombination_at_a_carbon_radical_passes_check_edit():
    """Two methyl radicals: N = 1 on each C; the -1 entries keep N + d >= 0 under the
    one-electron rule, and the polar rule refuses them as NOT_ELEMENTARY."""
    r = ReactionMatrix(((0, 4, 1),), ((0, -1), (4, -1)))
    label = EditLabel((r,), (r.anchors(),), True, 0, False)
    bonds = np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [4, 5, 1], [4, 6, 1], [4, 7, 1]], np.int16)
    n = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    assert check_edit(label, bonds, n, open_shell=True) is None
    assert check_edit(label, bonds, n) == "NOT_ELEMENTARY"
    assert check_edit(label, bonds, np.zeros(8, np.int8), open_shell=True) == "LONE_PAIRS"


@pytest.mark.parametrize(
    "reactant, r, want",
    [
        ("[Br:1][Br:2]", ReactionMatrix(((0, 1, -1),), ((1, 2),)), "[Br+]"),
        ("[OH2:1]", None, "[OH+]"),
        ("[CH3-:1]", None, "[CH2]"),
    ],
)
def test_a_plus_two_lone_pair_change_leaves_u_zero(reactant, r, want):
    """[Br+], [OH+] and a singlet [CH2] made by a +2 atom entry have even N, so u = 0, where
    RDKit's radical count gives 2."""
    mol = Chem.AddHs(P(reactant)) if r is None else P(reactant)
    if r is None:  # an X-H heterolysis toward the H: the H leaves as hydride
        h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        r = ReactionMatrix(((0, h, -1),), ((h, 2),))
    prod = sanitize_product(apply_edit(mol, r.to_edit()))
    frag = next(f for f in chem.fragments(prod) if chem.unmapped_canonical(f) == want)
    assert spin.rdkit_unpaired(frag) == 2 and spin.spin_of_drawing(frag)[0] == 0
    assert identity_key(frag) == sigma_key(frag)


# ---- step 2: the flow law and its codes


def test_both_allyl_cation_drawings_are_balanced_and_resonance():
    a, b = P("[CH2:1]=[CH:2][CH2+:3]"), P("[CH2+:1][CH:2]=[CH2:3]")
    assert balanced_flow(*mapped_changes(a, b)) and balanced_flow(*mapped_changes(b, a))
    assert classify_step(a, b) is StepKind.RESONANCE


def test_a_charge_jump_across_saturated_atoms_is_a_nonconjugated_shift():
    a, b = P("[CH2+:1][CH2:2][CH2:3][CH2-:4]"), P("[CH2-:1][CH2:2][CH2:3][CH2+:4]")
    assert sigma_key(a, keep_maps=True) == sigma_key(b, keep_maps=True)
    assert classify_step(a, b) is StepKind.NONCONJUGATED_SHIFT
    step = build_step(
        step_id="x",
        source=Source.REACTMECH,
        rxn_smiles=f"{chem.canonical_mapped_smiles(a)}>>{chem.canonical_mapped_smiles(b)}",
        arrows=[],
        require_arrows=False,
    )  # noqa: E501
    assert step.domain_exclusion == "NONCONJUGATED_SHIFT" and not step.resonance_step


def test_identical_sides_are_a_null_step_and_no_resonance_pair(tmp_path):
    from ksmech.data.io import write_resonance_pairs

    rxn = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
    step = build_step(
        step_id="n", source=Source.REACTMECH, rxn_smiles=rxn, arrows=[], require_arrows=False
    )
    assert step.domain_exclusion == "NULL_STEP" and not step.resonance_step
    import pandas as pd

    path = write_resonance_pairs([step], tmp_path / "pairs.parquet")
    assert pd.read_parquet(path).empty


def test_an_electron_transfer_without_bond_change_is_a_set_step():
    """D. + A >> D+ + A- (an ethyl radical gives its electron to a Cl atom): atom entries only,
    and a real step (the fragments' charges and spins change), not resonance."""
    rxn = "[CH3:1][CH2:2].[Cl:3]>>[CH3:1][CH2+:2].[Cl-:3]"
    assert _label(rxn) is EditSkip.SET_STEP
    assert classify_step(*map(P, rxn.split(">>"))) is StepKind.STEP


def test_an_electron_transfer_with_a_cleavage_is_dissociative():
    """R-X + D. >> R. + X- + D+ in one step: bond entries, every variant unbalanced."""
    rxn = "[CH3:1][Br:2].[CH3:3][CH2:4]>>[CH3:1].[Br-:2].[CH3:3][CH2+:4]"
    assert _label(rxn) is EditSkip.DISSOCIATIVE_ET_STEP


def test_no3_abstraction_labels_through_the_drawing_with_the_radical_there():
    """NO3. + Et2O with the H landing on the O drawn as O-: no flow on the drawn NO3 balances
    (DISSOCIATIVE_ET_STEP); the NO3 drawing that carries the radical on that O does."""
    rxn = ("[CH3:1][CH2:2][O:3][CH2:4][CH3:5].[O:6][N+:7](=[O:8])[O-:9]>>"
           "[CH3:1][CH:2][O:3][CH2:4][CH3:5].[O-:6][N+:7](=[O:8])[OH:9]")  # fmt: skip
    step = explicit_moving_hydrogens(_step(rxn))
    edit = step_edit(step)
    assert edit_label(step, edit) is EditSkip.DISSOCIATIVE_ET_STEP
    k, mol, label = redrawn_label(step, edit)
    assert k > 0 and isinstance(label, EditLabel) and all(v.balanced() for v in label.variants)
    o9 = next(a for a in mol.GetAtoms() if a.GetAtomMapNum() == 9)
    assert o9.GetNumRadicalElectrons() == 1 and o9.GetFormalCharge() == 0


# ---- step 3: the per-source elementary rule

RADICAL_SHAPES = {
    "homolysis": "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2].[Br:3]",
    "recombination": "[CH3:1].[CH3:2]>>[CH3:1][CH3:2]",
    "addition": "[Br:1].[CH2:2]=[CH:3][CH3:4]>>[Br:1][CH2:2][CH:3][CH3:4]",
    "hat": "[Br:1].[H:5][CH2:2][CH3:3]>>[Br:1][H:5].[CH2:2][CH3:3]",
    "beta_scission": "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5]>>[CH3:1].[CH3:3][C:2]([CH3:4])=[O:5]",
    "sh2": "[CH3:1].[Br:2][Br:3]>>[CH3:1][Br:2].[Br:3]",
    "disproportionation": "[CH3:1][CH2:2].[CH3:3][CH2:4]>>[CH3:1][CH3:2].[CH2:3]=[CH2:4]",
    "pi_homolysis": "[CH2:1]=[CH2:2]>>[CH2:1][CH2:2]",
}


@pytest.mark.parametrize("name", sorted(RADICAL_SHAPES))
def test_the_radical_shapes_label_and_apply(name):
    """Under the one-electron rule (rmechdb) each shape labels, its flow balances, and B + R is
    the annotated product; the polar rule (PMechDB) refuses every one."""
    rxn = RADICAL_SHAPES[name]
    label = _label(rxn)
    assert isinstance(label, EditLabel), label
    step = explicit_moving_hydrogens(_step(rxn))
    mol, want = P(step.reactants_smiles), P(step.products_smiles)
    for v in label.variants:
        assert v.is_elementary(open_shell=True) and v.balanced()
        got = sanitize_product(apply_edit(mol, v.to_edit()))
        assert identity_key(got) == identity_key(want)
    assert _label(rxn, Source.PMECHDB) in (EditSkip.NOT_ELEMENTARY, EditSkip.RESONANCE_STEP)


def test_no_batch_key_and_no_model_option_carries_the_source(mini_steps, token_loader):
    from ksmech.arrows.batches import collate_samples
    from ksmech.model.config import ModelConfig

    s = next(x for x in (sample_for_step(m, token_loader) for m in mini_steps.values())
             if isinstance(x, MechanismSample))  # fmt: skip
    a, b = collate_samples([s]), collate_samples([replace(s, source="")])
    assert not [k for k in a if "source" in k]
    assert not [f.name for f in fields(ModelConfig) if "source" in f.name]
    for key, value in a.items():
        other = b[key]
        same = value == other if not hasattr(value, "shape") else (value == other).all()
        assert bool(same), key


# ---- step 4: the identity key v2


def test_o2_drawings_key_as_one_triplet_through_the_hund_row(monkeypatch):
    assert identity_key(Chem.MolFromSmiles("O=O")) == identity_key(Chem.MolFromSmiles("[O][O]"))
    rows = {k: [r for r in v if r[0] != "[O][O]"] for k, v in spin._raising_rows().items()}
    monkeypatch.setattr(spin, "_raising_rows", lambda: rows)
    assert identity_key(Chem.MolFromSmiles("O=O")) != identity_key(Chem.MolFromSmiles("[O][O]"))


def test_a_drawing_row_holds_only_for_its_drawing(monkeypatch):
    """C=C never inherits a [CH2][CH2] row, and C=C >> [CH2][CH2] is a real step."""
    base = spin._raising_rows()
    ethene = spin._formula(Chem.MolFromSmiles("C=C"))
    rows = base | {ethene: [("[CH2][CH2]", 2, False)]}
    monkeypatch.setattr(spin, "_raising_rows", lambda: rows)
    assert identity_key(Chem.MolFromSmiles("C=C")) == sigma_key(Chem.MolFromSmiles("C=C"))
    assert classify_step(P("[CH2:1]=[CH2:2]"), P("[CH2:1][CH2:2]")) is StepKind.STEP


@pytest.mark.parametrize(
    "a, b, equal",
    [
        ("[C]=O", "[C-]#[O+]", True),
        ("C=C[CH2]", "[CH2]C=C", True),
        ("Cl[C]Cl", "Cl[C]Cl", True),
        ("C[O-].[CH3+]", "C[O].[CH3]", False),
        ("[CH2]C.Cl", "[CH2+]C.[Cl-]", False),
    ],
)
def test_the_v2_key_separates_spin_and_fragment_charges(a, b, equal):
    ka, kb = identity_key(Chem.MolFromSmiles(a)), identity_key(Chem.MolFromSmiles(b))
    assert (ka == kb) is equal
    if a == "Cl[C]Cl":
        assert ka == sigma_key(Chem.MolFromSmiles(a))  # a paired carbene keys as a closed shell


def test_ionic_and_radical_splits_of_one_bond_are_not_resonance():
    a, b = P("[CH3:1][O-:2].[CH3+:3]"), P("[CH3:1][O:2].[CH3:3]")
    assert classify_step(a, b) is StepKind.STEP
    d, e = P("[CH3:1][CH2:2].[Cl:3]"), P("[CH3:1][CH2+:2].[Cl-:3]")
    assert classify_step(d, e) is StepKind.STEP


def test_closed_shell_keys_are_byte_identical():
    assert identity_key(Chem.MolFromSmiles("CC")) == "[CH3][CH3]|0"
    assert identity_key(Chem.MolFromSmiles("CC(C)=O")) == "[CH3][C]([CH3])[O]|0"
    for smiles in MOLECULES.values():
        mol = Chem.MolFromSmiles(smiles)
        assert identity_key(mol) == sigma_key(mol)


@pytest.mark.parametrize("smiles", ["[CH3]", "[O][O]", "C=C[CH2]", "[O]c1ccccc1", "[O][O-]"])
def test_a_v2_species_key_has_no_dot_and_pool_keys_round_trip(smiles):
    from ksmech.serve.core.keys import pool_key_of, split_pool_key

    key = identity_key(Chem.MolFromSmiles(smiles))
    assert "." not in key and "|" in key and key.count(",") == 2
    pool = pool_key_of([key, key, "[CH3][CH3]|0"])
    assert split_pool_key(pool) == {key: 2, "[CH3][CH3]|0": 1}


# ---- step 6: radical forms through the anion proxy


@pytest.mark.parametrize(
    "smiles, n",
    [
        ("[CH2]c1ccccc1", 4),
        ("[O]c1ccccc1", 4),
        ("C=CC=C[CH2]", 3),
        ("C=C[CH2]", 2),
        ("CC1(C)CCCC(C)(C)N1[O]", 1),
    ],  # fmt: skip
)
def test_radical_forms_on_mapped_atoms(smiles, n):
    frozen = frozen_hydrogens(P(smiles))
    assert 1 + len(radical_forms(frozen, 256)) == n
    assert len(molecule_forms(P(smiles))) == n


def test_no3_has_three_drawings_one_per_oxygen():
    forms = molecule_forms(P("[O:1][N+:2](=[O:3])[O-:4]"))
    sites = {next(a.GetAtomMapNum() for a in f.GetAtoms() if a.GetNumRadicalElectrons())
             for f in forms}  # fmt: skip
    assert len(forms) == 3 and sites == {1, 3, 4}


def test_both_allylic_drawings_of_the_propene_hat_product_give_one_label():
    """Propene + Br. >> HBr + allyl.: whichever end the dataset draws the radical on, the
    anion proxy gives the product's other form, and the class is the cheap HAT at C4 (the
    drawing at C2 alone would make R_data shift the pi bond too)."""
    labels = []
    for prod in ("[CH2:2]=[CH:3][CH2:4]", "[CH2:2][CH:3]=[CH2:4]"):
        label = _label(f"[Br:1].[CH2:2]=[CH:3][CH3:4]>>[BrH:1].{prod}")
        assert isinstance(label, EditLabel)
        labels.append(label.variants)
    assert labels[0] == labels[1] and len(labels[0][0].bonds) == 2


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_closed_shell_form_lists_are_unchanged(name):
    mol = P(MOLECULES[name])
    frozen = frozen_hydrogens(mol)
    assert radical_forms(frozen, 256) == []
    forms, n = product_forms(mol)
    old, n_old = resonance_forms(frozen, 256)
    assert n == n_old and [Chem.MolToSmiles(f) for f in forms] == [Chem.MolToSmiles(f) for f in old]


# ---- step 7: the per-drawing atom categorical


def test_the_allyl_radical_drawings_put_u_on_either_end(token_loader):
    rt = ReactantTokens.from_smiles("[CH2:1]=[CH:2][CH2:3]", token_loader)
    us = [lewis_unpaired(rt, f) for f in molecule_forms(rt.mol)]
    ends = [{rt.mol.GetAtomWithIdx(m).GetAtomMapNum() for m, g in rt.global_atom.items() if u[g]}
            for u in us]  # fmt: skip
    assert len(us) == 2 and ends == [{3}, {1}]
