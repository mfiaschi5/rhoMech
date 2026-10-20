"""Pipek-Mezey localization: rotation properties, convergence, valence-only guard, AO order."""

from dataclasses import replace

import numpy as np
import pytest
from tblite.interface import Calculator

from ksmech.orbitals import localize as L
from ksmech.qm.ao import ao_layout


def test_valence_electron_table_matches_gfn2():
    z = (1, 6, 8, 17, 26, 29, 30, 35, 48, 57, 80)
    assert [L.valence_electrons(x) for x in z] == [1, 4, 6, 7, 8, 11, 2, 7, 2, 3, 2]
    with pytest.raises(L.LocalizationError):
        L.valence_electrons(87)


def test_pm_is_an_orthogonal_rotation_of_the_block(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    C_loc, U = L.pipek_mezey(occ, rec.S, rec.ao_atom)
    assert np.allclose(U.T @ U, np.eye(U.shape[0]), atol=1e-10)
    assert np.allclose(C_loc, occ @ U)
    assert np.allclose(C_loc.T @ rec.S @ C_loc, np.eye(C_loc.shape[1]), atol=1e-8)
    assert np.allclose(occ @ occ.T, C_loc @ C_loc.T, atol=1e-10)  # same subspace


def test_pm_raises_the_functional_and_converges(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    C_loc, _, st = L.pipek_mezey_stats(occ, rec.S, rec.ao_atom)
    assert st.converged and 1 <= st.n_sweeps < 50 and st.n_orbitals == occ.shape[1]
    assert st.functional_end > st.functional_start + 0.5
    # a second pass from the localized result has nothing left to gain
    _, U2, st2 = L.pipek_mezey_stats(C_loc, rec.S, rec.ao_atom)
    assert st2.n_sweeps == 1 and abs(st2.functional_end - st.functional_end) < 1e-6
    assert np.allclose(np.abs(U2), np.eye(U2.shape[0]), atol=1e-3)


def test_pm_reports_non_convergence_when_sweeps_run_out(sp_record):
    rec = sp_record("CCO")
    occ = rec.C[:, rec.occupations >= 0.5]
    _, _, st = L.pipek_mezey_stats(occ, rec.S, rec.ao_atom, L.PMConfig(tol=0.0, max_sweeps=2))
    assert not st.converged and st.n_sweeps == 2


def test_pm_trivial_blocks(sp_record):
    rec = sp_record("O")
    one = rec.C[:, :1]
    C_loc, U = L.pipek_mezey(one, rec.S, rec.ao_atom)
    assert np.array_equal(C_loc, one) and U.shape == (1, 1)
    empty = rec.C[:, :0]
    C_loc, U = L.pipek_mezey(empty, rec.S, rec.ao_atom)
    assert C_loc.shape == (rec.S.shape[0], 0) and U.shape == (0, 0)
    with pytest.raises(L.LocalizationError):
        L.pipek_mezey(rec.C[:3], rec.S, rec.ao_atom)


def test_localize_record_splits_occupied_and_virtual(sp_record):
    rec = sp_record("O")
    loc = L.localize_record(rec)
    assert loc.C_occ.shape == (6, 4) and loc.C_virt.shape == (6, 2)
    assert loc.converged and list(loc.occ_index) == [0, 1, 2, 3]


def test_somos_are_never_mixed_with_doubly_occupied_mos(sp_record):
    rec = sp_record("[CH2]")  # triplet methylene: two C-H bonds, two SOMOs
    loc = L.localize_record(rec)
    single = rec.occupations[loc.occ_index] < L.DOUBLE_THRESHOLD
    assert single.sum() == 2 and loc.stats_occ.n_orbitals == 4
    assert not loc.U_occ[np.ix_(single, ~single)].any()
    assert not loc.U_occ[np.ix_(~single, single)].any()
    assert np.allclose(loc.U_occ.T @ loc.U_occ, np.eye(4))
    assert np.allclose(loc.C_occ, rec.C[:, loc.occ_index] @ loc.U_occ)


def test_valence_only_guard_rejects_core_electrons(sp_record):
    rec = sp_record("O")
    L.assert_valence_only(rec)
    with_core = replace(rec, occupations=rec.occupations + 0.5)
    with pytest.raises(L.LocalizationError, match="not valence-only"):
        L.assert_valence_only(with_core)
    bad_map = replace(rec, ao_atom=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32))
    with pytest.raises(L.LocalizationError, match="non-minimal AO count"):
        L.assert_valence_only(bad_map)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_p_axis_order_matches_tblite(axis):
    """HF along one Cartesian axis: the sigma MO's p coefficient on F must sit on that axis."""
    from ksmech.orbitals.classify import p_vector

    z = np.array([1, 9], dtype=np.int32)
    xyz = np.zeros((2, 3))
    xyz[1, axis] = 1.75
    calc = Calculator("GFN2-xTB", z, xyz)
    calc.set("verbosity", 0)
    calc.set("save-integrals", 1)
    res = calc.singlepoint()
    C = np.asarray(res.get("orbital-coefficients"))
    occ = np.asarray(res.get("orbital-occupations")) > 1
    sigma = int(np.argmax(np.abs(C[0, occ])))  # the occupied MO with the largest H 1s weight
    ao_atom, ao_l = ao_layout(z)
    vec = p_vector(C[:, occ][:, sigma], ao_atom, ao_l, atom=1)
    assert abs(vec[axis]) > 0.5
    assert all(abs(vec[k]) < 1e-6 for k in range(3) if k != axis)


def test_mulliken_populations_sum_to_one(sp_record):
    rec = sp_record("CCO")
    loc = L.localize_record(rec)
    Q = L.mulliken_populations(loc.C_occ, rec.S, rec.ao_atom, rec.n_atoms)
    assert Q.shape == (rec.n_atoms, loc.C_occ.shape[1])
    assert np.allclose(Q.sum(axis=0), 1.0, atol=1e-8)
