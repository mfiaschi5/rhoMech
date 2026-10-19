from rdkit import Chem

from tests.fixtures.molecules import MOLECULES

EXPECTED_FORMULAS = {
    "ethane": "C2H6",
    "ethanol": "C2H6O",
    "acetone": "C3H6O",
    "acetaldehyde": "C2H4O",
    "methyl_acetate": "C3H6O2",
    "chloromethane": "CH3Cl",
    "methylamine": "CH5N",
    "acetonitrile": "C2H3N",
    "benzene": "C6H6",
    "acrolein": "C3H4O",
}


def test_ten_molecules():
    assert len(MOLECULES) == 10


def test_mols_parse(mols):
    assert set(mols) == set(MOLECULES)


def test_formulas(mols):
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    for name, mol in mols.items():
        assert CalcMolFormula(mol) == EXPECTED_FORMULAS[name], name


def test_smiles_are_canonical_round_trip(mols):
    for mol in mols.values():
        smi = Chem.MolToSmiles(mol)
        assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == smi
