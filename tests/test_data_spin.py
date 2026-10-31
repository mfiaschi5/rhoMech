"""data.spin: RDKit's count for u <= 1, spin_table_v1 for u >= 2 (open-shell plan P3 step 1)."""

import pytest
from rdkit import Chem

from ksmech.data import spin, spin_table
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL


def mol(smiles: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles)


def table_value(smiles: str):
    return spin_table.lookup(spin.species_table(), smiles)


def test_a_doublet_takes_rdkit_s_count():
    assert spin.species_u(mol("[CH3]")) == 1


def test_triplet_o2_takes_the_table_s_value():
    assert spin.species_u(mol("[O][O]")) == table_value("[O][O]") == 2


@pytest.mark.parametrize(
    "smiles",
    ["Cl[C]Cl", "Br[C]Br", "[C]=CC", "[C]=C(C)C", "[C]=C(c1ccccc1)c1ccccc1", "[C]=O"],
)
def test_carbenes_vinylidenes_and_co_take_the_table_s_value(smiles):
    assert table_value(smiles) is not None
    assert spin.species_u(mol(smiles)) == table_value(smiles)


V1_DISAGREES = (  # spin_table_v1 assigns these no multiplicity (docs/decisions.md, P2)
    "the scan's B reads the linear saddle the embedding draws (triplet) against A's bent singlet; "
    "resolved by spin_table_v2 (P4's production geometry), not in code"
)
SINGLET_ORACLE = [  # the textbook singlet clauses: dihalocarbenes, vinylidenes, CO
    pytest.param("Cl[C]Cl", marks=pytest.mark.xfail(strict=True, reason=V1_DISAGREES)),
    pytest.param("Br[C]Br", marks=pytest.mark.xfail(strict=True, reason=V1_DISAGREES)),
    "[C]=CC",
    "[C]=C(C)C",
    pytest.param(
        "[C]=C(c1ccccc1)c1ccccc1",
        marks=pytest.mark.xfail(strict=True, reason="spin_table_v1: A failed for a state"),
    ),
    "[C]=O",
]


@pytest.mark.parametrize("smiles", SINGLET_ORACLE)
def test_the_textbook_singlets_read_zero_in_the_table(smiles):
    assert spin.species_u(mol(smiles)) == 0


def test_an_untabulated_u2_species_is_counted_as_such():
    assert table_value("[CH2]CCC[CH2]") is None
    assert spin.species_u(mol("[CH2]CCC[CH2]")) == spin.UNTABULATED


def test_an_ambiguous_entry_is_returned_as_such():
    assert spin.species_u(mol("Cl[C]Cl")) == spin.AMBIGUOUS


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_the_fixed_molecules_are_closed_shells(name):
    assert spin.species_u(mol(MOLECULES[name])) == 0


@pytest.mark.parametrize("name", sorted(n for n, (_, u) in OPEN_SHELL.items() if u <= 1))
def test_open_shell_fixtures_with_one_unpaired_electron_keep_rdkit_s_count(name):
    smiles, u = OPEN_SHELL[name]
    assert spin.species_u(mol(smiles)) == u


def test_a_mapped_fragment_is_looked_up_without_its_maps():
    assert spin.species_u(Chem.MolFromSmiles("[O:1][O:2]")) == 2


def test_the_spin_config_names_the_rule_and_the_table():
    config = spin.spin_config()
    assert config["spin_rule_version"] == spin.SPIN_RULE_VERSION
    assert config["spin_table"] == spin_table.VERSION
    assert config["spin_table_digest"] == spin.species_table()["content_hash"]
