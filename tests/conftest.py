import pytest
from rdkit import Chem

from tests.fixtures.molecules import MOLECULES


@pytest.fixture(scope="session")
def smiles() -> dict[str, str]:
    return dict(MOLECULES)


@pytest.fixture(scope="session")
def mols() -> dict[str, Chem.Mol]:
    """RDKit mols (no explicit Hs) keyed by molecule name."""
    out = {}
    for name, smi in MOLECULES.items():
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"fixture {name!r} has invalid SMILES {smi!r}"
        out[name] = mol
    return out
