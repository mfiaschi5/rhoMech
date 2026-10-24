import os
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


@pytest.fixture(scope="session")
def contact_tables(sp_record):
    """Contact tables from a few small molecules at force-field geometries: enough to exercise
    every backoff level. The real table is built from the token cache."""
    from ksmech.orbitals.tokens import build_molecule_tokens
    from ksmech.physics.contact_table import tables_from_tokens

    smiles = ["CCO", "CC(=O)C", "CCCl", "OCCCCCl", "CC(=O)OC", "C=CC=O", "CCN", "CC#N", "CCCC"]
    return tables_from_tokens([build_molecule_tokens(sp_record(s)) for s in smiles], min_count=3)


@pytest.fixture(scope="session")
def contact_kernel():
    """The default (oriented) contact kernel, without a disk cache."""
    from ksmech.physics.contact_kernel import ContactKernel

    return ContactKernel()


@pytest.fixture(scope="session")
def pair_kernels():
    """The oriented kernel at every pair-feature distance, without a disk cache."""
    from ksmech.physics.pairs import oriented_kernels

    return oriented_kernels()


@pytest.fixture(scope="session")
def table_kernel(contact_tables):
    """The vdW-contact table kernel over the fixture tables (an ablation baseline)."""
    from ksmech.physics.contact_kernel import ContactConfig, ContactKernel

    return ContactKernel(contact_tables, ContactConfig(kernel="table"))


@pytest.fixture(scope="session")
def token_loader(sp_record):
    """token_loader(key) -> MoleculeTokens from a gas-phase single point at the force-field
    geometry (`sp_record`), memoized: a fast stand-in for the token cache in arrow tests."""
    from ksmech.orbitals.tokens import build_molecule_tokens

    memo: dict = {}

    def get(key: str):
        if key not in memo:
            memo[key] = build_molecule_tokens(sp_record(key))
        return memo[key]

    return get


@pytest.fixture
def mini_token_cache(mini_steps, token_loader, sp_record, tmp_path):
    """A `TokenCache` over token and bond-index files written for the mini steps' reactants:
    unlike `token_loader`, it can be sent to spawned workers. Each record has its own file, so
    no sidecar's input changes after it is written."""
    from ksmech.arrows.reactants import TokenCache, _fragment_keys
    from ksmech.data import chem
    from ksmech.orbitals.build import tokens_path
    from ksmech.orbitals.tokens import save_tokens
    from ksmech.qm.bond_index import save_bond_indices
    from ksmech.qm.records import save_record

    cache = TokenCache(tmp_path / "tokens", bond_index_dir=tmp_path / "wbi")
    keys = {
        key
        for step in mini_steps.values()
        for _, _, key in _fragment_keys(chem.parse_mapped(step.reactants_smiles))
        if key
    }
    for n, key in enumerate(sorted(keys)):
        save_tokens(token_loader(key), tokens_path(cache.tokens_dir, key, 0, cache.method))
        rec = sp_record(key)
        save_bond_indices(rec, save_record(rec, tmp_path / f"rec_{n}.npz"), tmp_path / "wbi")
    return cache


@pytest.fixture(scope="session")
def realdata_workers() -> int:
    """Spawned processes of one real-data pass: `KSMECH_TEST_WORKERS` (default 14), or 1
    inside an xdist run, whose workers already fill the host (15 heavy processes at most)."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return 1
    return int(os.environ.get("KSMECH_TEST_WORKERS", "14"))


@pytest.fixture(autouse=True)
def clean_code_state(monkeypatch):
    """Artifacts written by tests are stamped as built from a clean tree, whatever the state of
    the working copy (the commit hook runs on a tree that differs from HEAD); the dirty path
    has its own tests in test_provenance.py."""
    from ksmech.data import provenance

    monkeypatch.setattr(provenance, "code_state", lambda: ("0" * 40, False, ""))
