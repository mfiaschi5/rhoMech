"""K, eps and the Wolfsberg-Helmholz resonance integral."""

import numpy as np

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.ks_terms import resonance_integral, wolfsberg_helmholz
from ksmech.physics.pairs import PAIR_FEATURES
from ksmech.physics.params import (
    REDUCED,
    SCALING,
    KSParams,
    shell_fock_ev,
    to_params,
)
from ksmech.qm.fukui import HARTREE_TO_EV


def test_defaults_are_the_textbook_values():
    assert KSParams() == KSParams(K=1.75, eps=4.0)
    assert KSParams(K=1.0, eps=1.0) == REDUCED
    p = KSParams(K=2.0, eps=8.0)
    assert set(SCALING) == set(PAIR_FEATURES)
    z = np.ones((2, 3, len(PAIR_FEATURES)))
    out = to_params(z, p, PAIR_FEATURES)
    k = {n: i for i, n in enumerate(PAIR_FEATURES)}
    assert np.allclose(out[..., k["log_beta2_d2.3"]], 1 + 2 * np.log(2.0))
    assert np.allclose(out[..., k["log_abs_ks3"]], 1 + 2 * np.log(2.0))
    assert np.allclose(out[..., k["log_abs_ks1"]], 1 + np.log(2.0))
    assert np.allclose(out[..., k["ks2"]], 1 / 8.0) and np.allclose(
        out[..., k["cross_coupling"]], 2
    )
    assert np.allclose(out[..., k["log_abs_g"]], 1) and np.allclose(out[..., k["clash"]], 1)
    assert np.array_equal(to_params(z, REDUCED, PAIR_FEATURES), z)


def test_resonance_integral_is_wolfsberg_helmholz():
    assert resonance_integral(0.2, -10.0, -14.0) == 1.75 * 0.2 * -12.0
    assert resonance_integral(0.2, -10.0, -14.0, K=1.0) == 0.2 * -12.0
    S = np.array([[0.1, 0.3]])
    assert np.allclose(resonance_integral(S, -10.0, np.array([-12.0, -8.0])), [[-1.925, -4.725]])


def test_wolfsberg_helmholz_is_zero_within_an_atom_and_linear_in_k(sp_record):
    t = build_molecule_tokens(sp_record("CO"))
    B = wolfsberg_helmholz(t.S, t.F, t.ao_atom, K=1.75)
    assert np.allclose(B, B.T)
    same_atom = t.ao_atom[:, None] == t.ao_atom[None, :]
    assert (B[same_atom] == 0).all() and np.abs(B[~same_atom]).max() > 1.0  # eV
    assert np.allclose(wolfsberg_helmholz(t.S, t.F, t.ao_atom, K=1.0) * 1.75, B)
    f = np.diag(t.F) * HARTREE_TO_EV
    i, j = np.argwhere(~same_atom)[0]
    assert abs(B[i, j] - 1.75 * t.S[i, j] * 0.5 * (f[i] + f[j])) < 1e-12


def test_converged_occupied_virtual_coupling_survives_only_in_wh_form(sp_record):
    t = build_molecule_tokens(sp_record("CO"))
    L_o = np.column_stack([o.coeff_vector for o in t.occupied()])
    L_v = np.column_stack([o.coeff_vector for o in t.virtual()])
    B = wolfsberg_helmholz(t.S, t.F, t.ao_atom, K=1.75)
    assert np.abs(L_o.T @ t.S @ L_v).max() < 1e-8 and np.abs(L_o.T @ t.F @ L_v).max() < 1e-8
    assert np.abs(L_o.T @ B @ L_v).max() > 0.1


def test_shell_fock(sp_record):
    t = build_molecule_tokens(sp_record("CO"))
    oxygen = next(a.atom_index for a in t.atoms if a.atomic_number == 8)
    hydrogen = next(a.atom_index for a in t.atoms if a.atomic_number == 1)
    assert (
        shell_fock_ev(t.F, t.ao_atom, t.ao_l, oxygen, 0)
        < shell_fock_ev(t.F, t.ao_atom, t.ao_l, oxygen, 1)
        < 0
    )
    assert np.isnan(shell_fock_ev(t.F, t.ao_atom, t.ao_l, hydrogen, 1))
