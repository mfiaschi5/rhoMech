"""synthetic.py: charge-iterated EHT records on the GFN2 basis and the real tokens built from
them, with the no-xTB guard on (no tblite SCF anywhere)."""

import numpy as np
import pytest
from rdkit import Chem

from ksmech.serve.chem.synthetic import (
    NOT_CONVERGED,
    SYNTHETIC_METHOD,
    SyntheticError,
    eht_record,
    supported_elements,
)
from tests.fixtures.molecules import MOLECULES
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    synthetic_keys,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

EXTRA = ("C[O-]", "CCl", "[OH-]", "[Cl-]", "CO", "COC", "[H-]")
KEYS = sorted({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in (*MOLECULES.values(), *EXTRA)})


@pytest.fixture(scope="module")
def records():
    return {}


def _record(records, key):
    if key not in records:
        records[key] = eht_record(key)
    return records[key]


@pytest.mark.parametrize("key", KEYS)
def test_record_fields_and_electron_count(key, records):
    from ksmech.orbitals.localize import assert_valence_only, valence_electrons
    from ksmech.qm.fukui import HARTREE_TO_EV
    from ksmech.serve.chem.gto import overlap_matrix

    rec = _record(records, key)
    z = rec.atomic_numbers
    n_val = sum(valence_electrons(int(x)) for x in z) - rec.charge
    assert rec.occupations.sum() == n_val
    assert set(np.unique(rec.occupations)) <= {0.0, 2.0}
    assert_valence_only(rec)  # holds by construction
    assert rec.method == SYNTHETIC_METHOD and rec.smiles == rec.atom_order_smiles == key
    assert "SYNTHETIC" in rec.warnings.split(",") and NOT_CONVERGED not in rec.warnings
    assert rec.unpaired == 0 and rec.conformer_id == 0
    assert np.allclose(rec.S, overlap_matrix(z, rec.coords_ang))
    assert np.allclose(rec.C.T @ rec.S @ rec.C, np.eye(rec.C.shape[1]), atol=1e-8)
    eps_ha = rec.orbital_energies_ev / HARTREE_TO_EV
    assert np.allclose(rec.C.T @ rec.fock @ rec.C, np.diag(eps_ha), atol=1e-8)
    assert np.all(np.diff(rec.orbital_energies_ev) >= -1e-12)  # ascending
    assert np.isclose(rec.charges.sum(), rec.charge, atol=1e-6)
    assert np.isnan(rec.fukui_dual).all() and np.isnan(rec.fukui_plus).all()
    assert len(rec.buried_volume) == len(z) and len(rec.ao_atom) == rec.S.shape[0]


def test_deterministic():
    for key in ("C=CC=O", "CCl", "c1ccccc1"):
        a, b = eht_record(key), eht_record(key)
        for name in ("coords_ang", "orbital_energies_ev", "C", "S", "fock", "charges"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), (key, name)


def test_tokens_deterministic():
    from ksmech.orbitals.tokens import build_molecule_tokens

    a = build_molecule_tokens(eht_record("CC(=O)C"))
    b = build_molecule_tokens(eht_record("CC(=O)C"))
    assert np.array_equal(a.C_loc, b.C_loc)
    assert np.array_equal(a.orbital_features(), b.orbital_features())


@pytest.mark.parametrize("key", [k for k in KEYS if k != "[H-]"])
def test_tokens_in_the_synthetic_root(key, synthetic_loader):  # noqa: F811
    t = synthetic_loader(key)
    assert t is not None, key
    assert t.method == SYNTHETIC_METHOD and t.smiles == key
    n_occ = len(t.occupied())
    assert sum(o.descriptors.occupation for o in t.occupied()) == pytest.approx(2 * n_occ)
    assert t.fermi_smeared is False and t.pm_converged


def test_methoxide_one_centre_oxygen_lone_pairs(synthetic_loader):  # noqa: F811
    t = synthetic_loader("C[O-]")
    one_centre_o = [
        o
        for o in t.occupied()
        if o.descriptors.n_centers == 1 and t.symbol(int(o.descriptors.center_atoms[0])) == "O"
    ]
    assert len(one_centre_o) == 3  # three O lone pairs, each a donor candidate
    assert all(o in t.donor_candidates() for o in one_centre_o)


def test_chloromethane_sigma_star(synthetic_loader):  # noqa: F811
    t = synthetic_loader("CCl")
    hits = [
        o
        for o in t.virtual()
        if {t.symbol(int(a)) for a in o.descriptors.all_center_atoms} == {"C", "Cl"}
    ]
    assert hits, "no virtual token with centres {C, Cl}"
    assert all(o.descriptors.overlap_population < 0 for o in hits)  # antibonding
    assert any(o in t.acceptor_candidates() for o in hits)


def test_hydride_has_no_virtual_block(serve_settings):  # noqa: F811
    from ksmech.serve.chem.featurize import FeaturizeStatus, SyntheticFeaturizer

    feat = SyntheticFeaturizer(serve_settings(chemistry="synthetic"))
    res = feat.ensure("[H-]")
    assert res.status == FeaturizeStatus.BUILT, res
    t = feat.loader()("[H-]")
    assert len(t.occupied()) == 1 and t.virtual() == []
    assert "NO_FRONTIER" not in (t.qm_warnings or "")


def test_featurizer_cached_on_second_call(serve_settings):  # noqa: F811
    from ksmech.serve.chem.featurize import FeaturizeStatus, SyntheticFeaturizer, qm_npz_path

    s = serve_settings(chemistry="synthetic")
    feat = SyntheticFeaturizer(s)
    first = feat.ensure("CO")
    assert first.status == FeaturizeStatus.BUILT and first.code is None
    second = feat.ensure("CO")
    assert second.status == FeaturizeStatus.CACHED
    assert second.tokens_path == first.tokens_path
    qm = qm_npz_path(s, "CO", SYNTHETIC_METHOD)
    assert qm is not None and qm.is_relative_to(s.serve_qm)
    occ = feat.occupations("CO")
    assert occ is not None and occ.sum() == 14
    assert first.to_dict()["status"] == "built"


def test_featurizer_failure_codes(serve_settings):  # noqa: F811
    from ksmech.serve.chem.featurize import FeaturizeStatus, SyntheticFeaturizer

    s = serve_settings(chemistry="synthetic")
    feat = SyntheticFeaturizer(s)
    res = feat.ensure("[SeH2]")
    assert res.status == FeaturizeStatus.FAILED
    assert res.code == "SYNTHETIC_FAILED" and res.upstream == "UNSUPPORTED_ELEMENT"
    assert s.failures_log.is_file() and "SYNTHETIC_FAILED" in s.failures_log.read_text()
    with pytest.raises(SyntheticError) as e:
        eht_record("[CH3]")
    assert e.value.code == "RADICAL_NOT_SUPPORTED"
    assert 34 not in supported_elements() and 17 in supported_elements()


def test_synthetic_keys_cover_the_fixture_set():
    keys = synthetic_keys()
    assert len(keys) == 15  # chloromethane is in both sets
    for smi in MOLECULES.values():
        assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) in keys
