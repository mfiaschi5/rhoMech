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


@pytest.fixture(scope="session")
def qm_config(tmp_path_factory):
    """Default QM config writing to a throwaway cache, so tests never touch data/cache."""
    from ksmech.qm.xtb_calc import QMConfig

    return QMConfig(cache_dir=tmp_path_factory.mktemp("qm_cache"))


@pytest.fixture(scope="session")
def qm_record(qm_config):
    """qm_record(smiles, config=None) -> QMRecord, computed once per (smiles, config)."""
    from ksmech.qm.validation import compute_fixture_records

    memo: dict = {}

    def get(smi: str, config=None):
        cfg = config or qm_config
        key = (smi, cfg)
        if key not in memo:
            memo[key] = compute_fixture_records({"x": smi}, cfg)["x"]
        return memo[key]

    return get


@pytest.fixture(scope="session")
def fixture_records(qm_config):
    """QMRecords for the ten fixture molecules (slow: runs optimizations)."""
    from ksmech.qm.validation import compute_fixture_records

    return compute_fixture_records(MOLECULES, qm_config)


@pytest.fixture(scope="session")
def sp_record():
    """sp_record(smiles) -> QMRecord from one gas-phase GFN2 single point at the force-field
    geometry (no optimization): fast enough for the non-slow orbital tests."""
    import numpy as np

    from ksmech.data import chem
    from ksmech.qm.embed import embed, prepare_mol
    from ksmech.qm.geometry import BOHR_ANG, new_calculator
    from ksmech.qm.xtb_calc import GFN2Calculator, resolve_charge_spin

    memo: dict = {}

    def get(smi: str):
        canonical = chem.unmapped_canonical(Chem.MolFromSmiles(smi))
        if canonical not in memo:
            mol = prepare_mol(canonical)
            embed(mol, n_conformers=1, seed=0)
            z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
            xyz = mol.GetConformer(0).GetPositions()
            charge, unpaired = resolve_charge_spin(mol)
            calc = new_calculator(z, xyz / BOHR_ANG, charge, unpaired, None, save_integrals=True)
            res = calc.singlepoint()
            memo[canonical] = GFN2Calculator()._assemble(
                res, z, xyz, canonical, 0, charge, unpaired
            )
        return memo[canonical]

    return get
