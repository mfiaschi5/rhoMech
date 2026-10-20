"""The contact kernels (table and analytic probe), unified KS3, intramolecular extras."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics import contact
from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
from ksmech.physics.ks_terms import (
    cross_center_coupling,
    intramolecular_pair_features,
    ks3_contact,
    ks3_contact_matrix,
    wh_matrix,
)
from ksmech.physics.params import KSParams
from tests.token_select import highest_lone_pair, leading, sigma_star


def test_contact_overlap_probe_is_symmetric_and_handles_hydrogen():
    cc = contact.contact_overlap(6, 6)
    assert cc.shape == (2, 2) and (cc >= 0).all() and cc[0, 0] > 0.1 and cc[1, 1] > 0.1
    assert np.allclose(contact.contact_overlap(6, 8), contact.contact_overlap(8, 6).T, atol=1e-10)
    ch = contact.contact_overlap(6, 1)
    assert ch[0, 0] > 0 and ch[1, 0] > 0 and ch[0, 1] == 0 and ch[1, 1] == 0  # H has no p
    assert contact.contact_overlap(6, 6) is contact.contact_overlap(6, 6)  # cached


@pytest.fixture(params=["table", "analytic", "oriented"])
def kernel(request, contact_tables):
    return ContactKernel(contact_tables, ContactConfig(kernel=request.param))


def test_contact_term_is_non_zero_within_and_across_molecules(sp_record, kernel):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    lone = highest_lone_pair(methanol, "O")
    c_cl = sigma_star(chloromethane, {"C", "Cl"})
    assert 0.001 < kernel.amplitude(lone, methanol, c_cl, chloromethane) < 2.0
    assert kernel.beta(lone, methanol, c_cl, chloromethane) < 0  # Fock diagonals are negative
    assert ks3_contact(lone, methanol, c_cl, chloromethane, kernel) < 0
    co_sigma_star = sigma_star(methanol, {"C", "O"})
    assert kernel.amplitude(lone, methanol, co_sigma_star, methanol) > 0.001  # intramolecular
    M = ks3_contact_matrix(methanol.occupied(), methanol, methanol.virtual(), methanol, kernel)
    assert M.shape == (len(methanol.occupied()), len(methanol.virtual())) and (M < 0).all()


def test_ks3_scales_as_k_squared(sp_record, kernel):
    methanol = build_molecule_tokens(sp_record("CO"))
    r, s = highest_lone_pair(methanol, "O"), sigma_star(methanol, {"C", "O"})
    one = ks3_contact(r, methanol, s, methanol, kernel, KSParams(K=1.0))
    two = ks3_contact(r, methanol, s, methanol, kernel, KSParams(K=2.0))
    assert abs(two - 4.0 * one) < 1e-12 * abs(two)


def test_4_chlorobutanol_intramolecular_features(sp_record, contact_tables):
    t = build_molecule_tokens(sp_record("OCCCCCl"))
    lone = highest_lone_pair(t, "O")
    c_cl = sigma_star(t, {"C", "Cl"})
    feats = intramolecular_pair_features(
        lone, c_cl, t, ContactKernel(contact_tables, ContactConfig(kernel="table"))
    )
    # the converged S coupling is 0 by orthogonality; the Wolfsberg-Helmholz coupling with B zero
    # within an atom is not, even in the extended conformer (about 0.016 eV at K = 1.75)
    assert abs(lone.coeff_vector @ t.S @ c_cl.coeff_vector) < 1e-8
    print(f"4-chlorobutanol cross-centre coupling (WH): {feats['cross_center_coupling']:.2e} eV")
    assert feats["cross_center_coupling"] != 0.0
    assert feats["centroid_distance"] > 3.0 and feats["contact_beta"] < 0
    assert feats["gap_kernel"] < 0 and feats["ks3_contact"] < 0
    B = wh_matrix(t, KSParams(K=1.0))
    one = cross_center_coupling(lone, c_cl, B, t.ao_atom)
    assert abs(one * 1.75 - feats["cross_center_coupling"]) < 1e-12


def test_truncation_matters_on_pmechdb_pairs():
    """G from the full spectra and from the 8 model-input entries differ by > 5% somewhere."""
    from pathlib import Path

    from ksmech.orbitals.tokens import load_tokens
    from ksmech.physics.ks_terms import token_gap_kernel, token_gap_kernel_truncated

    files = sorted(Path("data/cache/tokens").glob("*.npz"))[:20]
    if not files:
        pytest.skip("no PMechDB token files in data/cache/tokens")
    worst, n_pairs = 0.0, 0
    for path in files:
        try:
            t = load_tokens(path)
        except ValueError:
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


@pytest.mark.parametrize("name", ["table", "analytic"])
def test_matrix_forms_match_the_definitions(sp_record, contact_tables, name):
    from ksmech.physics.contact import contact_coupling
    from ksmech.physics.contact_kernel import main_shell_fock
    from ksmech.physics.contact_table import contact_key

    kernel = ContactKernel(contact_tables, ContactConfig(kernel=name))
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    rows, cols = methanol.occupied(), chloromethane.orbitals  # both blocks as columns
    A = kernel.amplitude_matrix(rows, methanol, cols, chloromethane)
    B = kernel.beta_matrix(rows, methanol, cols, chloromethane, KSParams(K=2.0))
    z_m = np.array([a.atomic_number for a in methanol.atoms])
    z_c = np.array([a.atomic_number for a in chloromethane.atoms])
    for i, r in enumerate(rows):
        for j, s in enumerate(cols):
            if name == "analytic":
                expected = contact_coupling(r.descriptors, z_m, s.descriptors, z_c)
            else:
                kind = "occ_virt" if s.kind == "virtual" else "occ_occ"
                kappa = (*contact_key(r, z_m), *contact_key(s, z_c))
                w = r.descriptors.center_weights[0] * s.descriptors.center_weights[0]
                expected = contact_tables[kind].lookup(kappa)[0] * w
            assert A[i, j] == pytest.approx(expected, rel=1e-12)
            f = 0.5 * (main_shell_fock(r, methanol) + main_shell_fock(s, chloromethane))
            assert B[i, j] == pytest.approx(2.0 * A[i, j] * f, rel=1e-12)


def test_table_lookups_are_counted_per_pair(sp_record, contact_tables):
    table = contact_tables["occ_virt"]
    before = sum(table.lookups.values())
    t = build_molecule_tokens(sp_record("CO"))
    kernel = ContactKernel(contact_tables, ContactConfig(kernel="table"))
    kernel.amplitude_matrix(t.occupied(), t, t.virtual(), t)
    assert sum(table.lookups.values()) - before == len(t.occupied()) * len(t.virtual())


def test_cross_centre_coupling_is_restricted_to_the_centre_blocks(sp_record):
    from ksmech.physics.ks_terms import centre_restricted, cross_center_coupling_matrix

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
