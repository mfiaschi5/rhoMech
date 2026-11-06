"""data.spin: RDKit for u <= 1, the species table (v2) for u >= 2 (P3), the metal table (P4)."""

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


# the textbook singlet clauses: dihalocarbenes, vinylidenes, CO. spin_table_v1 left Cl[C]Cl,
# Br[C]Br and Ph2C=C: ambiguous; v2 decides them in the table (docs/decisions.md, 2026-11-01)
SINGLET_ORACLE = ["Cl[C]Cl", "Br[C]Br", "[C]=CC", "[C]=C(C)C", "[C]=C(c1ccccc1)c1ccccc1", "[C]=O"]


@pytest.mark.parametrize("smiles", SINGLET_ORACLE)
def test_the_textbook_singlets_read_zero_in_the_table(smiles):
    assert spin.species_u(mol(smiles)) == 0


def test_an_untabulated_u2_species_is_counted_as_such():
    assert table_value("[CH2]CCC[CH2]") is None
    assert spin.species_u(mol("[CH2]CCC[CH2]")) == spin.UNTABULATED


def test_an_ambiguous_entry_is_returned_as_such():
    ambiguous = [
        s for s, e in spin.species_table()["entries"].items() if e["unpaired"] == spin.AMBIGUOUS
    ]
    smiles = next(s for s in ambiguous if spin.rdkit_unpaired(mol(s)) >= 2)
    assert spin.species_u(mol(smiles)) == spin.AMBIGUOUS


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_the_fixed_molecules_are_closed_shells(name):
    assert spin.species_u(mol(MOLECULES[name])) == 0


METAL_FIXTURES = {"copper_dichloride": 1, "tetrabromoferrate": 5}  # the metal table (P4)


@pytest.mark.parametrize(
    "name", sorted(n for n, (_, u) in OPEN_SHELL.items() if u <= 1 and n not in METAL_FIXTURES)
)
def test_open_shell_fixtures_with_one_unpaired_electron_keep_rdkit_s_count(name):
    smiles, u = OPEN_SHELL[name]
    assert spin.species_u(mol(smiles)) == u


@pytest.mark.parametrize("name", sorted(METAL_FIXTURES))
def test_the_odd_metal_fixtures_take_the_metal_table_s_count(name):
    smiles, drawn = OPEN_SHELL[name]
    assert spin.rdkit_unpaired(mol(smiles)) == drawn == 0
    assert spin.species_u(mol(smiles)) == METAL_FIXTURES[name]


def test_a_mapped_fragment_is_looked_up_without_its_maps():
    assert spin.species_u(Chem.MolFromSmiles("[O:1][O:2]")) == 2


def test_the_spin_config_names_the_rule_and_the_table():
    config = spin.spin_config()
    assert config["spin_rule_version"] == spin.SPIN_RULE_VERSION
    assert config["spin_table"] == spin_table.VERSION
    assert config["spin_table_digest"] == spin.species_table()["content_hash"]


# ---- P4: `assign`, its sources and the metal table (open-shell plan P4 step 1)


def assigned(smiles: str) -> spin.Assignment:
    return spin.assign(mol(smiles))


@pytest.mark.parametrize(
    ("smiles", "u"),
    [
        ("[Cl][Cu][Cl]", 1),
        ("C[NH+](C)[Cu-]([Cl])[Cl]", 1),
        ("[Br][Fe-]([Br])([Br])[Br]", 5),
        ("C[N+](C)(c1cccc(Br)c1)[Fe-]([Br])([Br])[Br]", 5),  # N-ligated, confirmed by rule C
        ("C[O][Fe-]([Cl])([Cl])[Cl]", 5),
    ],
)
def test_a_confirmed_metal_row_gives_its_count_on_the_metal_atom(smiles, u):
    a = assigned(smiles)
    assert (a.unpaired, a.source) == (u, spin.METAL)
    centre = next(x.GetIdx() for x in mol(smiles).GetAtoms() if x.GetSymbol() in ("Cu", "Fe"))
    assert a.per_atom[centre] == u and sum(a.per_atom) == u


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("C[NH+](C)[CuH2-]", spin.REDOX_INCOMPATIBLE),  # Cu(II) hydride
        ("[I][Cu][I]", spin.REDOX_INCOMPATIBLE),  # Cu(II) iodide
        ("[Cl][Fe]([Cl])[Cl]", spin.NOT_CONFIRMED),  # rule C: A inconclusive, B undetermined
        ("Br[Br+][Fe-]([Br])([Br])[Br]", spin.NOT_CONFIRMED),  # A failed
        ("[Cl][Pd]([Cl])[Cl]", spin.NO_ROW),  # an odd Pd fragment: no row
    ],
)
def test_an_unconfirmed_or_rowless_metal_fragment_is_ambiguous(smiles, reason):
    a = assigned(smiles)
    assert (a.unpaired, a.source, a.reason) == (spin.AMBIGUOUS, spin.METAL, reason)


def test_a_metal_fragment_the_species_table_lacks_is_untabulated():
    a = assigned("ClN(Cl)[Fe-](Cl)(Cl)Cl")
    assert (a.unpaired, a.reason, a.row) == (spin.UNTABULATED, spin.NOT_CONFIRMED, "FE3_HS")


def test_a_ligand_dropped_metal_is_not_assigned():
    a = assigned("[Fe][Cl]")  # odd, one neighbour: the placeholder law's shape
    assert (a.unpaired, a.reason) == (spin.AMBIGUOUS, spin.PLACEHOLDER)


def test_the_metal_table_validates_and_every_row_cites_its_basis():
    table = spin.load_metal_table()
    assert table["version"] == spin.METAL_VERSION
    for row in table["rows"]:
        assert row["basis"] and set(row["citations"]) <= set(table["citations"])
        assert row.get("redox_incompatible") or isinstance(row["unpaired"], int)


@pytest.mark.parametrize(
    ("smiles", "source"),
    [("C", spin.RDKIT), ("[CH3]", spin.RDKIT), ("[O][O]", spin.SPECIES), ("Cl[C]Cl", spin.SPECIES)],
)
def test_the_source_of_a_count(smiles, source):
    assert assigned(smiles).source == source


def test_a_paired_species_has_no_unpaired_atom_and_a_triplet_keeps_rdkit_s_atoms():
    assert assigned("Cl[C]Cl").per_atom == (0, 0, 0)
    assert assigned("[O][O]").per_atom == (1, 1)


def test_reassigned_is_a_count_other_than_the_drawing_s():
    assert spin.reassigned(assigned("Cl[C]Cl"), 2) and spin.reassigned(assigned("[Cl][Cu][Cl]"), 0)
    assert not spin.reassigned(assigned("[O][O]"), 2)
    assert not spin.reassigned(assigned("[I][Cu][I]"), 0)  # ambiguous: RDKit's count stays


def test_no_carbene_or_co_clause_is_code_in_the_spin_module():
    import inspect

    source = inspect.getsource(spin)
    assert "MolFromSmarts" not in source and "HasSubstructMatch" not in source


def test_the_spin_config_names_both_tables():
    config = spin.spin_config()
    assert config["metal_table"] == spin.METAL_VERSION
    assert config["metal_table_digest"] == spin.metal_table()["content_hash"]
