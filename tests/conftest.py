from pathlib import Path

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


MINI_PMECHDB = Path(__file__).parent / "fixtures" / "mini_pmechdb.csv"
# Line order of tests/fixtures/mini_pmechdb.csv (the raw format has no class column).
MINI_LABELS = ["sn2", "proton_transfer", "carbonyl_addition", "e2", "resonance", "unbalanced"]


@pytest.fixture(scope="session")
def mini_result():
    from ksmech.data import pmechdb

    return pmechdb.load(MINI_PMECHDB)


@pytest.fixture(scope="session")
def mini_steps(mini_result) -> dict:
    """Accepted mini-fixture steps keyed by their label."""
    return dict(zip(MINI_LABELS[:5], mini_result.steps, strict=True))
