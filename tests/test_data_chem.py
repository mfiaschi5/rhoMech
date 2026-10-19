"""Mapped-SMILES helpers, run on the fixed molecule set."""

import pytest
from rdkit import Chem

from ksmech.data import chem
from tests.fixtures.molecules import MOLECULES


def _fully_mapped(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_canonicalize_is_idempotent_and_keeps_maps(name):
    mapped = _fully_mapped(MOLECULES[name])
    canon = chem.canonicalize_mapped(mapped)
    assert chem.canonicalize_mapped(canon) == canon
    mol = chem.parse_mapped(canon)
    assert sorted(chem.map_numbers(mol)) == list(range(1, mol.GetNumAtoms() + 1))


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_counts_match_rdkit(name, mols):
    ref = Chem.AddHs(mols[name])
    mol = chem.parse_mapped(MOLECULES[name])
    assert chem.hydrogen_count(mol) == sum(a.GetAtomicNum() == 1 for a in ref.GetAtoms())
    assert chem.heavy_atom_count(mol) == mols[name].GetNumHeavyAtoms()
    assert chem.formal_charge(mol) == 0


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_unmapped_canonical_matches_plain_rdkit(name):
    mol = chem.parse_mapped(_fully_mapped(MOLECULES[name]))
    assert chem.unmapped_canonical(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(MOLECULES[name]))


def test_aromatic_input_is_kekulized():
    mol = chem.parse_mapped(MOLECULES["benzene"])
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())
    assert "c" not in chem.canonical_mapped_smiles(mol)


def test_kekule_forms_stay_distinct():
    """A resonance step only moves pi bonds; aromaticity perception would erase it."""
    k1 = "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1"
    k2 = "[CH:1]1-[CH:2]=[CH:3]-[CH:4]=[CH:5]-[CH:6]=1"
    assert chem.canonicalize_mapped(k1) != chem.canonicalize_mapped(k2)


def test_mapped_hydrogens_are_kept():
    canon = chem.canonicalize_mapped("CC(=O)[O:21][H:20]")
    assert "[H:20]" in canon


def test_fragments_and_map_to_fragment():
    mol = chem.parse_mapped("[OH-:10].[CH3:20][Cl:21]")
    frag = chem.map_to_fragment(mol)
    assert frag[20] == frag[21] != frag[10]
    assert len(chem.fragments(mol)) == 2


def test_parse_errors():
    with pytest.raises(chem.SmilesParseError):
        chem.parse_mapped("C1CC")
    with pytest.raises(chem.SanitizeError):
        chem.parse_mapped("C(C)(C)(C)(C)C")
