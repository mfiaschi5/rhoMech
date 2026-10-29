"""The open-shell survey (`data.survey`): one hand-written fixture per subclass, the oxidation
states, and on real data the counts the open-shell plan's P0 oracle lists."""

import pytest
from rdkit import Chem

from ksmech.data import survey as sv
from ksmech.data.schema import Arrow, ElementaryStep, SinkKind, SourceKind


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


@pytest.mark.parametrize(
    ("smiles", "subclass"),
    [
        ("[C]N(C)C", "C3_METHYL"),
        ("O=C(n1[c]ncc1)n1[c]ncc1", "CDI_IMIDAZOLYL"),
        ("C[Sn]", "SN_PLACEHOLDER"),
        ("CC(C)O[Al]OC(C)C", "AL_DRAWING"),
        ("CC(C)(C)[O-].[K]", "BARE_ALKALI"),
        ("O=C1C[N]C1", "BETA_LACTAM_N"),
        ("[OH+][N+](C)=CC", "NITRONE_DRAWING"),
        ("C[P](C)(C)C", "P_FOUR_BONDS"),
        ("C[C](C)O[Na]", "AMBIGUOUS_KETYL"),
        ("[C]=O", "CO_DRAWN"),
        ("Cl[C]Cl", "CARBENE"),
        ("CC(C)O[O]", "RADICAL_SPECTATOR"),
    ],
)
def test_one_fixture_per_radical_subclass(smiles, subclass):
    assert sv.classify_radical([_mol(smiles)], set())[0] == subclass


def test_a_radical_an_arrow_touches_is_reactive():
    mol = Chem.MolFromSmiles("CC[B:20](CC)CC.[O][O:10]")
    sub, touched = sv.classify_radical([mol], {10, 20})
    assert (sub, touched) == ("RADICAL_REACTIVE", [10])


@pytest.mark.parametrize(
    ("smiles", "code", "subclass"),
    [
        ("Cl[Fe-](Cl)(Cl)Cl.[Na+]", "OPEN_SHELL_METAL", "OSM_FE3"),
        ("Cl[Cu]Cl", "OPEN_SHELL_METAL", "OSM_CU2"),
        ("I[Cu]I", "OPEN_SHELL_METAL", "CU_SUSPECT"),
        ("C[NH+](C)[CuH2-]", "OPEN_SHELL_METAL", "CUH2_LIGAND"),
        ("Br[Pd](c1ccccc1)[P+](C)(C)C", "OPEN_SHELL_METAL", "PD_DATIVE"),
        ("Br[Pd-](Br)(c1ccccc1)[P+](C)(C)C", "OPEN_SHELL_METAL", "PD_ALREADY_NEG"),
        ("[Pd][P+](C)(C)C", "METAL_PLACEHOLDER", "L1PD"),
        ("C=[O+][Ti]", "METAL_PLACEHOLDER", "PLACEHOLDER"),
    ],
)
def test_one_fixture_per_metal_subclass(smiles, code, subclass):
    assert sv.classify_metal([_mol(smiles)], code) == subclass


@pytest.mark.parametrize(
    ("smiles", "subclass"),
    [
        ("C=[O+][Zn-]", "PLACEHOLDER_EVEN"),
        ("Cl[Zn+]", "ZNCL_ION"),
        ("Cl[Ni-2](C)([P+](C)(C)C)[P+](C)(C)C", "ONIUM_PD_NI_EVEN"),
        ("O=[Cr](O)O", "CR_IV"),
        ("CC(C)=O", None),
    ],
)
def test_kept_steps_with_a_suspicious_metal(smiles, subclass):
    assert sv.classify_kept([_mol(smiles)]) == subclass


def test_oxidation_states_and_d_counts():
    fe = next(a for a in _mol("Cl[Fe-](Cl)(Cl)Cl").GetAtoms() if a.GetSymbol() == "Fe")
    cu = next(a for a in _mol("Cl[Cu]Cl").GetAtoms() if a.GetSymbol() == "Cu")
    pd = next(a for a in _mol("Br[Pd-](Br)(C)[P+](C)(C)C").GetAtoms() if a.GetSymbol() == "Pd")
    assert (sv.oxidation_state(fe), sv.d_count(fe)) == (3, 5)
    assert (sv.oxidation_state(cu), sv.d_count(cu)) == (2, 9)
    # -1 charge, two Br and a methyl, and the P+ bond: the onium's +1 and the Pd's compensating
    # -1 cancel (an L ligand), so the neutral, odd PdBr2Me(PMe3) reads Pd(III) (domain.py, P0a;
    # the P0 helper read 2 because it dropped the P+ bond, which made [Pd-][P+]R3 read Pd(-I))
    assert sv.oxidation_state(pd) == 3


def test_spin_parity_failures_are_named_by_what_they_hold():
    assert sv.spin_parity_kind("Cl[Cu]Cl") == "Cu"
    assert sv.spin_parity_kind("C=[O+][Ti]") == "placeholder"
    assert sv.spin_parity_kind("CP(C)C") == "P"


def test_a_kept_step_without_a_suspicious_metal_has_no_row():
    arrows = [Arrow(SourceKind.LONE_PAIR, (1,), SinkKind.ATOM, (2,))]
    step = ElementaryStep(
        "t", "pmechdb", None, None, "[OH-:1].[CH3:2][Cl:3]", "[OH:1][CH3:2].[Cl-:3]", arrows
    )
    assert sv.survey_row(step) is None


@pytest.mark.realdata
def test_the_survey_matches_the_p0_oracle():
    import pandas as pd

    from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, steps_path

    def counts(source):
        steps = read_steps(steps_path(source, DEFAULT_PROCESSED_DIR))
        manifest = pd.read_parquet(DEFAULT_PROCESSED_DIR / source / "qm_manifest.parquet")
        return sv.survey(steps, manifest)

    pm = counts("pmechdb")
    radical = pm["counts"]["RADICAL_DRAWN"]
    genuine = radical["RADICAL_SPECTATOR"] + radical["RADICAL_REACTIVE"]
    assert (sum(radical.values()), genuine, radical["CARBENE"], radical["C3_METHYL"]) == (
        52,
        12,
        10,
        9,
    )
    osm = pm["counts"]["OPEN_SHELL_METAL"]
    assert osm["OSM_FE3"] == 43 and osm["OSM_CU2"] + osm["CUH2_LIGAND"] + osm["CU_SUSPECT"] == 35
    assert pm["counts"]["KEPT"] == {"PLACEHOLDER_EVEN": 24, "ZNCL_ION": 3}
    assert pm["spin_parity_molecules"] == {"Cu": 41, "placeholder": 34, "Fe": 20, "Ti": 4, "P": 1}
    rm = counts("reactmech")
    assert rm["counts"]["RADICAL_DRAWN"] == {"CDI_IMIDAZOLYL": 62}
    assert rm["counts"]["OPEN_SHELL_METAL"] == {"PD_DATIVE": 96, "PD_ALREADY_NEG": 2}
    assert rm["counts"]["METAL_PLACEHOLDER"] == {"L1PD": 109}
    assert rm["counts"]["KEPT"] == {"ONIUM_PD_NI_EVEN": 49, "CR_IV": 10}
