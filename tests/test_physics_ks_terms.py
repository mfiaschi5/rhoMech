"""The geometry-free Klopman-Salem quantities: the gap kernel, the AO energies and the
cross-centre coupling."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.ks_terms import (
    ao_energies,
    centre_restricted,
    cross_center_coupling,
    cross_center_coupling_matrix,
    gap_kernel,
    gap_kernel_matrix,
    token_gap_kernel,
    wh_matrix,
)
from ksmech.physics.params import KSParams
from ksmech.qm.fukui import HARTREE_TO_EV
from ksmech.qm.geometry import BOHR_ANG
from tests.token_select import highest_lone_pair, leading, sigma_star, sigma_stars


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


def test_gap_kernel_matrix_matches_the_scalar_form(sp_record):
    t = build_molecule_tokens(sp_record("CO"))
    occ, virt = t.occupied(), t.virtual()
    G = gap_kernel_matrix(occ, virt)
    assert G.shape == (len(occ), len(virt)) and (G < 0).all()
    for i, r in enumerate(occ):
        for j, s in enumerate(virt):
            assert abs(G[i, j] - token_gap_kernel(r, s)) < 1e-12


def test_gap_kernel_across_molecules(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    lp = highest_lone_pair(methanol, "O")
    c_cl = sigma_star(chloromethane, {"C", "Cl"})
    c_h = sigma_stars(chloromethane, {"C", "H"})[0]
    g_cl, g_h = token_gap_kernel(lp, c_cl), token_gap_kernel(lp, c_h)
    assert g_cl < g_h < 0


def test_ao_energies_are_rotation_invariant(sp_record):
    """The resonance integral's AO energies do not depend on the record's orientation: the
    lab-frame Fock diagonal of a p shell does (GFN2's anisotropic terms), its shell average
    does not."""
    from ksmech.qm.geometry import new_calculator
    from ksmech.qm.xtb_calc import GFN2Calculator

    rec = sp_record("CC(C)=O")
    M = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    xyz = rec.coords_ang @ M.T
    calc = new_calculator(
        rec.atomic_numbers, xyz / BOHR_ANG, rec.charge, rec.unpaired, None, save_integrals=True
    )
    turned = GFN2Calculator()._assemble(
        calc.singlepoint(), rec.atomic_numbers, xyz, rec.smiles, 0, rec.charge, rec.unpaired
    )
    t, u = build_molecule_tokens(rec), build_molecule_tokens(turned)
    raw_t, raw_u = np.diag(t.F) * HARTREE_TO_EV, np.diag(u.F) * HARTREE_TO_EV
    assert np.abs(raw_t - raw_u).max() > 1e-3  # the lab-frame diagonal moves
    assert np.allclose(ao_energies(t), ao_energies(u), atol=1e-6)


def test_ao_energies_average_each_shell(sp_record):
    """One value per shell, the mean of its Fock diagonal (no GFN2 element has two shells of one
    l, so a shell is an (atom, l) block); a one-AO s shell keeps its own value."""
    t = build_molecule_tokens(sp_record("CC(=O)Cl"))
    f, h = np.diag(t.F) * HARTREE_TO_EV, ao_energies(t)
    for atom in range(len(t.atoms)):
        for ell in set(t.ao_l[t.ao_atom == atom]):
            rows = np.flatnonzero((t.ao_atom == atom) & (t.ao_l == ell))
            assert len(rows) == 2 * ell + 1
            assert np.allclose(h[rows], f[rows].mean(), atol=1e-12)
    assert np.array_equal(h[t.ao_l == 0], f[t.ao_l == 0])
    assert (t.ao_l == 2).any()  # chlorine's d shell is averaged too


def test_4_chlorobutanol_cross_centre_coupling(sp_record):
    t = build_molecule_tokens(sp_record("OCCCCCl"))
    lone = highest_lone_pair(t, "O")
    c_cl = sigma_star(t, {"C", "Cl"})
    # the converged S coupling is 0 by orthogonality; the Wolfsberg-Helmholz coupling with B zero
    # within an atom is not, even in the extended conformer (about 0.016 eV at K = 1.75)
    assert abs(lone.coeff_vector @ t.S @ c_cl.coeff_vector) < 1e-8
    physical = cross_center_coupling(lone, c_cl, wh_matrix(t), t.ao_atom)
    one = cross_center_coupling(lone, c_cl, wh_matrix(t, KSParams(K=1.0)), t.ao_atom)
    assert physical != 0.0 and abs(one * 1.75 - physical) < 1e-12
    assert token_gap_kernel(lone, c_cl) < 0


def test_cross_centre_coupling_is_restricted_to_the_centre_blocks(sp_record):
    t = build_molecule_tokens(sp_record("OCCCCCl"))
    B = wh_matrix(t)
    occ, virt = t.occupied(), t.virtual()
    M = cross_center_coupling_matrix(occ, virt, t.ao_atom, B)
    lone = highest_lone_pair(t, "O")
    c_cl = sigma_star(t, {"C", "Cl"})
    i, j = occ.index(lone), virt.index(c_cl)
    on_r = np.isin(t.ao_atom, lone.descriptors.all_center_atoms)
    on_s = np.isin(t.ao_atom, c_cl.descriptors.all_center_atoms)
    manual = lone.coeff_vector[on_r] @ B[np.ix_(on_r, on_s)] @ c_cl.coeff_vector[on_s]
    assert M[i, j] == pytest.approx(manual, rel=1e-12)
    L = centre_restricted([lone], t.ao_atom)[:, 0]
    assert (L[~on_r] == 0).all() and np.allclose(L[on_r], lone.coeff_vector[on_r])
    # a bond and its own antibond share both centres: a large coupling; through space, small
    sigma_c_cl = next(
        o for o in occ if set(o.descriptors.all_center_atoms[:2]) == set(leading(c_cl))
    )
    on_bond = abs(M[occ.index(sigma_c_cl), j])
    assert on_bond > 0.1 and on_bond > 100 * abs(M[i, j])


def test_truncation_matters_on_pmechdb_pairs():
    """G from the full spectra and from the 8 model-input entries differ by > 5% somewhere."""
    from pathlib import Path

    from ksmech.data.provenance import StaleArtifact
    from ksmech.orbitals.tokens import load_tokens
    from ksmech.physics.ks_terms import token_gap_kernel_truncated

    files = sorted(Path("data/cache/tokens").glob("*.npz"))[:20]
    if not files:
        pytest.skip("no PMechDB token files in data/cache/tokens")
    worst, n_pairs = 0.0, 0
    for path in files:
        try:
            t = load_tokens(path)
        except (ValueError, StaleArtifact):
            continue
        occ, virt = t.occupied(), t.virtual()  # every pair, not just the frontier ones
        for r in occ:
            for s in virt:
                full, cut = token_gap_kernel(r, s), token_gap_kernel_truncated(r, s)
                worst = max(worst, abs(full - cut) / abs(full))
                n_pairs += 1
    if not n_pairs:
        pytest.skip("no token files of the current schema; run ksmech-orbitals build --force")
    print(f"largest |G_full - G_truncated| / |G_full| over {n_pairs} pairs: {worst:.1%}")
    assert worst > 0.05, worst
