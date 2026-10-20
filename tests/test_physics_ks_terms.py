"""Klopman-Salem term 3 in spectral form: the gap kernel and the coupling."""

import numpy as np

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.ks_terms import (
    coupling,
    gap_kernel,
    gap_kernel_matrix,
    ks3,
    ks3_matrix,
    token_gap_kernel,
)
from tests.token_select import highest_lone_pair, sigma_star, sigma_stars


def test_gap_kernel_sign_and_ordering():
    occ = (np.array([-10.0]), np.array([1.0]))
    low = (np.array([2.0]), np.array([1.0]))
    high = (np.array([5.0]), np.array([1.0]))
    g_low, g_high = gap_kernel(*occ, *low), gap_kernel(*occ, *high)
    assert g_low == 2.0 / (-12.0) and g_high == 2.0 / (-15.0)
    assert g_low < g_high < 0  # the lower virtual interacts more strongly
    assert gap_kernel(*low, *occ) == -g_low  # antisymmetric in the two arguments


def test_gap_kernel_mixes_spectral_weights_and_skips_padding():
    e_r = np.array([-10.0, -12.0, 0.0, 0.0])
    p_r = np.array([0.75, 0.25, 0.0, 0.0])  # zero-weight padding entries must not count
    e_s, p_s = np.array([2.0]), np.array([1.0])
    expected = 2 * 0.75 / (-12.0) + 2 * 0.25 / (-14.0)
    assert abs(gap_kernel(e_r, p_r, e_s, p_s) - expected) < 1e-12


def test_gap_kernel_clamps_degenerate_pairs():
    same = (np.array([1.0]), np.array([1.0]))
    assert np.isfinite(gap_kernel(*same, *same)) and gap_kernel(*same, *same) == 2.0 / 1e-3


def test_ks3_matrix_matches_the_scalar_form(sp_record):
    t = build_molecule_tokens(sp_record("CO"))
    occ, virt = t.occupied(), t.virtual()
    rng = np.random.default_rng(0)
    B = rng.normal(size=t.S.shape)
    B = B + B.T  # any coupling matrix: the identity must not rest on a vanishing coupling
    M = ks3_matrix(occ, virt, B)
    G = gap_kernel_matrix(occ, virt)
    assert M.shape == G.shape == (len(occ), len(virt)) and (G < 0).all() and (M < 0).all()
    for i, r in enumerate(occ):
        for j, s in enumerate(virt):
            d_r, d_s = r.descriptors, s.descriptors
            scalar = ks3(
                r.coeff_vector,
                s.coeff_vector,
                B,
                d_r.full_energies,
                d_r.full_weights,
                d_s.full_energies,
                d_s.full_weights,
            )
            assert abs(M[i, j] - scalar) < 1e-12 * max(1.0, abs(scalar))
            assert abs(G[i, j] - token_gap_kernel(r, s)) < 1e-12
    # occupied and virtual LMOs of one converged molecule are S-orthogonal: with the molecule's
    # own S the converged coupling, and with it term 3, vanishes
    assert np.abs(ks3_matrix(occ, virt, t.S)).max() < 1e-12
    a = coupling(occ[0].coeff_vector, t.F, occ[1].coeff_vector)
    b = coupling(occ[1].coeff_vector, t.F, occ[0].coeff_vector)
    assert abs(a - b) < 1e-12 and abs(a) > 1e-3  # symmetric; non-zero within one block


def test_gap_kernel_across_molecules(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    lp = highest_lone_pair(methanol, "O")
    c_cl = sigma_star(chloromethane, {"C", "Cl"})
    c_h = sigma_stars(chloromethane, {"C", "H"})[0]
    g_cl, g_h = token_gap_kernel(lp, c_cl), token_gap_kernel(lp, c_h)
    assert g_cl < g_h < 0
