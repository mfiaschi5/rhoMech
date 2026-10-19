"""Fukui functions and global descriptors, on hand-made inputs with known answers."""

import numpy as np
import pytest

from ksmech.qm import fukui


def test_finite_difference_sign_and_sum_rules():
    # q positive = electron-poor. Adding an electron lowers q; removing one raises it.
    q_n = np.array([0.30, -0.30, 0.0])
    q_plus = np.array([0.10, -0.50, -0.60])  # N+1: sums to q_n.sum() - 1
    q_minus = np.array([0.50, 0.20, 0.30])  # N-1: sums to q_n.sum() + 1
    assert q_plus.sum() == pytest.approx(-1.0)
    assert q_minus.sum() == pytest.approx(1.0)
    fd = fukui.finite_difference(q_n, q_plus, q_minus)
    assert np.allclose(fd.plus, q_n - q_plus)
    assert np.allclose(fd.minus, q_minus - q_n)
    assert fd.plus.sum() == pytest.approx(1.0)
    assert fd.minus.sum() == pytest.approx(1.0)
    assert fd.zero.sum() == pytest.approx(1.0)
    assert np.allclose(fd.dual, fd.plus - fd.minus)
    assert np.allclose(fd.zero, 0.5 * (fd.plus + fd.minus))


def test_dual_descriptor_sign():
    # atom 0 takes the added electron (electrophilic), atom 1 gives it up (nucleophilic)
    fd = fukui.finite_difference(np.array([0.0, 0.0]), np.array([-0.9, -0.1]), np.array([0.1, 0.9]))
    assert fd.dual[0] > 0 > fd.dual[1]


def test_homo_index_uses_half_occupation_threshold():
    assert fukui.homo_index(np.array([2.0, 2.0, 1e-11, 0.0])) == 1
    assert fukui.homo_index(np.array([2.0, 1.0, 0.0])) == 1  # SOMO of a radical
    assert fukui.homo_index(np.array([0.0, 0.0])) is None  # [H+], [Na+]


def test_unpaired_rule_for_ionized_systems():
    assert fukui.unpaired_for_ionized(0) == 1
    assert fukui.unpaired_for_ionized(1) == 0
    assert fukui.unpaired_for_ionized(2) == 1


def test_frozen_orbital_fukui_is_mulliken_population():
    # two atoms, one s AO each, overlap s: HOMO is bonding, LUMO antibonding
    s = 0.4
    S = np.array([[1.0, s], [s, 1.0]])
    C = np.array([[1.0, 1.0], [1.0, -1.0]])
    C = C / np.sqrt(np.einsum("ik,ij,jk->k", C, S, C))
    ao_atom = np.array([0, 1])
    f_plus, f_minus = fukui.frozen_orbital_fukui(C, S, ao_atom, 2, homo=0)
    assert f_plus.sum() == pytest.approx(1.0) and f_minus.sum() == pytest.approx(1.0)
    assert np.allclose(f_plus, [0.5, 0.5]) and np.allclose(f_minus, [0.5, 0.5])
    # polarize the LUMO onto atom 1: atom 1 must carry the larger f+
    C2 = np.array([[1.0, 0.2], [1.0, -1.0]])
    C2 = C2 / np.sqrt(np.einsum("ik,ij,jk->k", C2, S, C2))
    f_plus, _ = fukui.frozen_orbital_fukui(C2, S, ao_atom, 2, homo=0)
    assert f_plus[1] > f_plus[0]


def test_frozen_orbital_fukui_nan_without_frontier():
    f_plus, f_minus = fukui.frozen_orbital_fukui(
        np.eye(2), np.eye(2), np.array([0, 1]), 2, homo=None
    )
    assert np.isnan(f_plus).all() and np.isnan(f_minus).all()


def test_global_descriptors():
    d = fukui.global_descriptors(np.array([-12.0, -10.0, -2.0, 1.0]), homo=1)
    assert d["homo_ev"] == -10.0 and d["lumo_ev"] == -2.0
    assert d["chemical_potential_ev"] == pytest.approx(-6.0)
    assert d["hardness_ev"] == pytest.approx(4.0)
    assert d["electrophilicity_ev"] == pytest.approx(36.0 / 8.0)


def test_global_descriptors_nan_without_frontier():
    d = fukui.global_descriptors(np.array([1.0, 2.0]), homo=None)
    assert all(np.isnan(v) for v in d.values())
    d = fukui.global_descriptors(np.array([-1.0, -0.5]), homo=1)  # no virtual orbital
    assert all(np.isnan(v) for v in d.values())


def test_fock_rebuild_satisfies_roothaan_equations():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    S = A @ A.T + 6 * np.eye(6)
    w, v = np.linalg.eigh(S)
    C = v @ np.diag(w**-0.5)  # S-orthonormal columns
    assert np.allclose(C.T @ S @ C, np.eye(6))
    eps = np.sort(rng.normal(size=6))
    F = fukui.fock_from_mo(S, C, eps)
    assert np.allclose(F, F.T)
    assert np.allclose(F @ C, S @ C @ np.diag(eps))
