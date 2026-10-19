"""QMRecord persistence: cache key, .npz round trip, atomic writes."""

from dataclasses import fields

import numpy as np
import pytest

from ksmech.qm.records import (
    QMRecord,
    ReasonCode,
    cache_key,
    cache_path,
    load_record,
    save_record,
)


def make_record(n_atoms: int = 3, n_ao: int = 5, seed: int = 0) -> QMRecord:
    rng = np.random.default_rng(seed)
    sym = rng.normal(size=(n_ao, n_ao))
    return QMRecord(
        smiles="CCO",
        atom_order_smiles="CCO",
        conformer_id=2,
        method="GFN2-xTB/ALPB-water/tblite-lbfgs",
        charge=-1,
        unpaired=1,
        atomic_numbers=np.array([6, 8, 1], dtype=np.int32),
        coords_ang=rng.normal(size=(n_atoms, 3)),
        energy_ha=-12.5,
        orbital_energies_ev=rng.normal(size=n_ao),
        occupations=np.array([2.0, 2.0, 1.0, 0.0, 0.0]),
        C=rng.normal(size=(n_ao, n_ao)),
        S=sym + sym.T,
        H_core=sym - sym.T,
        fock=sym @ sym.T,
        charges=rng.normal(size=n_atoms),
        ao_atom=np.array([0, 0, 1, 1, 2], dtype=np.int32),
        ao_l=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        fukui_plus=rng.normal(size=n_atoms),
        fukui_minus=np.full(n_atoms, np.nan),
        fukui_zero=rng.normal(size=n_atoms),
        fukui_dual=rng.normal(size=n_atoms),
        fukui_fmo_plus=rng.normal(size=n_atoms),
        fukui_fmo_minus=rng.normal(size=n_atoms),
        homo_ev=-9.1,
        lumo_ev=float("nan"),
        chemical_potential_ev=-4.0,
        hardness_ev=5.0,
        electrophilicity_ev=1.6,
        buried_volume=rng.uniform(0, 60, size=n_atoms),
        warnings="NO_FRONTIER,VBUR_FAIL",
    )


def test_every_field_round_trips(tmp_path):
    rec = make_record()
    path = save_record(rec, tmp_path / "a" / "rec.npz")
    back = load_record(path)
    for f in fields(QMRecord):
        a, b = getattr(rec, f.name), getattr(back, f.name)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype, f.name
            assert np.array_equal(a, b, equal_nan=True), f.name
        elif isinstance(a, float) and np.isnan(a):
            assert np.isnan(b), f.name
        else:
            assert a == b and type(a) is type(b), f.name


def test_save_leaves_no_temp_files(tmp_path):
    save_record(make_record(), tmp_path / "rec.npz")
    assert [p.name for p in tmp_path.iterdir()] == ["rec.npz"]


def test_load_does_not_need_pickle(tmp_path):
    path = save_record(make_record(), tmp_path / "rec.npz")
    with np.load(path, allow_pickle=False) as npz:
        assert all(npz[k].dtype != object for k in npz.files)


def test_cache_key_depends_on_each_part():
    base = cache_key("CCO", 0, "GFN2-xTB/ALPB-water/tblite-lbfgs")
    assert base == cache_key("CCO", 0, "GFN2-xTB/ALPB-water/tblite-lbfgs")
    assert len(base) == 64
    assert base != cache_key("CCN", 0, "GFN2-xTB/ALPB-water/tblite-lbfgs")
    assert base != cache_key("CCO", 1, "GFN2-xTB/ALPB-water/tblite-lbfgs")
    assert base != cache_key("CCO", 0, "GFN2-xTB/gas/tblite-lbfgs")  # solvation is in the key


def test_cache_path_layout(tmp_path):
    path = cache_path(tmp_path, "CCO", 0, "m")
    assert path.parent == tmp_path and path.suffix == ".npz"
    assert path.stem == cache_key("CCO", 0, "m")


def test_warning_codes_deduplicate():
    rec = make_record()
    rec.warnings = ""
    rec.add_warning(ReasonCode.VBUR_FAIL)
    rec.add_warning(ReasonCode.VBUR_FAIL)
    rec.add_warning(ReasonCode.NO_FRONTIER)
    assert rec.warnings == "VBUR_FAIL,NO_FRONTIER"


@pytest.mark.parametrize("code", list(ReasonCode))
def test_reason_codes_are_their_names(code):
    assert str(code) == code.name
