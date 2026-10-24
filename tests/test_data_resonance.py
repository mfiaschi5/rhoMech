"""The sigma key, resonance forms, the ingestion flag and the training-target filter."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.data import chem
from ksmech.data.resonance import (
    _FLAGS,
    frozen_hydrogens,
    is_resonance_step,
    resonance_forms,
    sigma_key,
    training_steps,
)
from ksmech.data.schema import ElementaryStep, Source
from ksmech.data.stats import compute_stats
from tests.fixtures.molecules import MOLECULES

KEKULE_A = "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1"
KEKULE_B = "[CH:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=1"
ENOLATE = "[CH2:1]=[C:2]([CH3:4])[O-:3]"
CARBANION = "[CH2-:1][C:2]([CH3:4])=[O:3]"
FORMS = 256  # enumeration cap of these tests, as the edit labels'


def key(smiles: str) -> str:
    return sigma_key(chem.parse_mapped(smiles))


def sides(a: str, b: str) -> bool:
    return is_resonance_step(chem.parse_mapped(a), chem.parse_mapped(b))


def test_kekule_structures_share_a_sigma_key():
    assert key(KEKULE_A) == key(KEKULE_B) == key("c1ccccc1")


def test_enolate_and_carbanion_share_a_sigma_key():
    assert key(ENOLATE) == key(CARBANION) and sides(ENOLATE, CARBANION)
    assert not sides(ENOLATE, "[CH3:1][C:2]([CH3:4])=[O:3]")  # an H more: a protonation


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("CC(=O)C", "CC(O)=C"),  # a tautomer moves an H
        ("[NH4+]", "N"),  # a proton more, a charge more
        ("C=C[O-]", "C=CO"),  # the charge, not only where it sits
        ("CC=O", "C1CO1"),  # another sigma skeleton
        ("CC[O-:1].C[O:2][H:3]", "CC[O:1][H:3].C[O-:2]"),  # a proton between two alcohols
        ("CC[O-].CO", "CCO.C[O-]"),  # the same with every H implicit
    ],
)
def test_the_sigma_key_separates_what_resonance_cannot_reach(a, b):
    """An O with no H (a zeroed alkoxide) is not an O-H: every H count is written out."""
    assert key(a) != key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("[H:5][O:1][CH3:2]", "CO"),  # an H drawn as an atom counts on its neighbour
        ("[H][H]", "[HH]"),  # H2 drawn either way
        ("C/C=C/C", "C/C=C\\C"),  # no stereo
        ("[CH3:1][C:2](=[O:3])[CH2-:4]", "[CH2-:1][C:2](=[O:3])[CH3:4]"),  # no atom maps
    ],
)
def test_the_sigma_key_is_a_property_of_the_molecule_not_of_the_drawing(a, b):
    assert key(a) == key(b) and ":" not in key(a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("[OH2:1].[OH3+:2]", "[OH3+:1].[OH2:2]"),  # a proton passed between two waters
        ("C[S-:10].C[S:21][H:20]", "C[S-:21].C[S:10][H:20]"),  # thiolate / thiol
        ("C[S-:10].C[S:20][S:21]C", "C[S-:21].C[S:10][S:20]C"),  # thiol-disulfide exchange
    ],
)
def test_a_degenerate_exchange_is_a_real_step(a, b):
    """The same molecules on both sides, so one map-free sigma key, but atoms change partners:
    a real elementary step, which stays in the training target (docs/decisions.md,
    2026-10-25)."""
    assert key(a) == key(b) and not sides(a, b)


def test_a_long_conjugated_anion_needs_no_enumeration():
    """The key enumerates nothing, so it is never truncated."""
    assert sides(
        "[CH2-:1][CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=[O:7]",
        "[CH2:1]=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6][O-:7]",
    )


def test_a_charge_separation_reached_from_one_side_only_is_a_resonance_step():
    """C=N-O <-> C(+)-N(-)-O: RDKit enumerates the neutral form from the charged one, not the
    reverse; the sigma key needs neither."""
    assert sides("C[CH:10]=[N:11]O", "C[CH+:10][N-:11]O")
    assert sides("C[CH+:10][N-:11]O", "C[CH:10]=[N:11]O")


@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1[O-]", "CO[C:20]([O-:10])=[CH:21][CH:22]=[CH2:23]", "CC(=O)C=C[O-]", "C=C[CH2+]"],
)
def test_the_sigma_key_does_not_depend_on_atom_order(smiles):
    mol = Chem.MolFromSmiles(smiles)
    keys = set()
    for seed in range(30):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        keys.add(sigma_key(Chem.RenumberAtoms(mol, order)))
    assert len(keys) == 1, keys


def step(i: int, pathway: str | None, index: int | None, resonance: bool = False):
    return ElementaryStep(
        step_id=f"s{i}",
        source=Source.PMECHDB,
        pathway_id=pathway,
        step_index=index,
        reactants_smiles="C",
        products_smiles="C",
        resonance_step=resonance,
    )


def test_training_steps_drops_and_renumbers_within_pathway():
    steps = [
        step(0, "p", 0),
        step(1, "p", 1, resonance=True),
        step(2, "p", 2),
        step(3, "q", 5),
        step(4, None, None, resonance=True),
        step(5, None, None),
    ]
    kept = training_steps(steps)
    assert [s.step_id for s in kept] == ["s0", "s2", "s3", "s5"]
    assert [s.step_index for s in kept] == [0, 1, 0, None]
    assert steps[2].step_index == 2  # the input is not mutated


def test_resonance_step_survives_the_dict_round_trip():
    s = step(0, "p", 0, resonance=True)
    assert ElementaryStep.from_dict(s.to_dict()).resonance_step is True
    assert ElementaryStep.from_dict({**s.to_dict(), "resonance_step": None}).resonance_step is False
    assert replace(s, resonance_step=False).resonance_step is False


def test_ingestion_flags_the_mini_fixture_resonance_step(mini_steps, mini_result):
    assert mini_steps["resonance"].resonance_step is True
    assert all(not s.resonance_step for name, s in mini_steps.items() if name != "resonance")
    report = mini_result.report()
    assert report["n_resonance_steps"] == report["n_resonance_pairs"] == 1
    stats = compute_stats(list(mini_steps.values()))
    assert stats["n_resonance_steps"] == 1 and stats["n_training_steps"] == 4


def test_the_resonance_pairs_file_holds_the_in_domain_resonance_steps(mini_steps, tmp_path):
    import pyarrow.parquet as pq

    from ksmech.data.io import write_resonance_pairs

    out_of_domain = replace(mini_steps["resonance"], step_id="x", domain_exclusion="RADICAL_DRAWN")
    path = write_resonance_pairs([*mini_steps.values(), out_of_domain], tmp_path / "pairs.parquet")
    rows = pq.read_table(path).to_pylist()
    s = mini_steps["resonance"]
    assert rows == [
        {
            "step_id": s.step_id,
            "reactants_smiles": s.reactants_smiles,
            "products_smiles": s.products_smiles,
        }
    ]


def electron_state(mol: Chem.Mol) -> tuple:
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    atoms = list(m.GetAtoms())
    return (
        [a.GetTotalNumHs() for a in atoms],
        sum(a.GetFormalCharge() for a in atoms),
        sum(a.GetNumRadicalElectrons() for a in atoms),
    )


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_resonance_forms_keep_every_hydrogen_the_charge_and_the_pairing(name):
    """With the H frozen, every form keeps the H count of every atom, the total charge and no
    unpaired electron; the molecule itself is one of its forms, and all share its sigma key."""
    mol = frozen_hydrogens(Chem.MolFromSmiles(MOLECULES[name]))
    forms, n = resonance_forms(mol, FORMS)
    assert 1 <= len(forms) <= n < FORMS
    assert all(electron_state(f) == electron_state(mol) for f in forms)
    assert Chem.MolToSmiles(mol) in {Chem.MolToSmiles(f) for f in forms}
    assert {sigma_key(f) for f in forms} == {sigma_key(mol)}


def test_nitro_forms_never_move_a_hydrogen_or_unpair_electrons():
    """RDKit's own enumeration of nitroethane re-derives the implicit H of an unbracketed O
    ([OH2+]) and returns the radical CCN([O+])[O-]. Neither survives: the two equivalent N-O
    forms are left."""
    mol = Chem.MolFromSmiles("CC[N+](=O)[O-]")
    with rdBase.BlockLogs():
        raw = rdchem.ResonanceMolSupplier(mol, int(_FLAGS), 64)
        raw_smiles = {Chem.MolToSmiles(raw[k]) for k in range(len(raw))}
    assert {"CCN([O-])[OH2+]", "CCN([O+])[O-]"} <= raw_smiles  # what the filter is for
    forms, _ = resonance_forms(frozen_hydrogens(mol), 64)
    assert len(forms) == 2
    assert all(electron_state(f) == electron_state(mol) for f in forms)
    assert {sigma_key(f) for f in forms} == {sigma_key(mol)}


def test_enolate_forms_are_the_alkoxide_and_the_carbanion():
    """From either drawing: two forms, one with the charge on O and one on the CH2."""
    for smiles in (ENOLATE, CARBANION):
        forms, _ = resonance_forms(frozen_hydrogens(chem.parse_mapped(smiles)), 64)
        charged = sorted(a.GetSymbol() for f in forms for a in f.GetAtoms() if a.GetFormalCharge())
        assert charged == ["C", "O"], smiles
        assert all(electron_state(f) == electron_state(chem.parse_mapped(smiles)) for f in forms)


def test_frozen_hydrogens_keep_a_kekule_structure():
    """The edit labels read bond orders, so their enumeration starts from the Kekule structure
    as drawn: re-perceived aromaticity would change which structures RDKit returns."""
    kekule = Chem.MolFromSmiles("C1=CC=CC=C1")
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    assert not any(b.GetIsAromatic() for b in frozen_hydrogens(kekule).GetBonds())
