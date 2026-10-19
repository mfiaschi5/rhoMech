"""Contact coupling, unified KS3, and the intramolecular pair features."""

import numpy as np
import pytest

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics import contact
from ksmech.physics.ks_terms import (
    cross_center_coupling,
    intramolecular_pair_features,
    ks3_contact,
    ks3_contact_matrix,
    token_contact_coupling,
)

T = OrbitalType


def test_contact_overlap_probe_is_symmetric_and_handles_hydrogen():
    cc = contact.contact_overlap(6, 6)
    assert cc.shape == (2, 2) and (cc >= 0).all() and cc[0, 0] > 0.1 and cc[1, 1] > 0.1
    assert np.allclose(contact.contact_overlap(6, 8), contact.contact_overlap(8, 6).T, atol=1e-10)
    ch = contact.contact_overlap(6, 1)
    assert ch[0, 0] > 0 and ch[1, 0] > 0 and ch[0, 1] == 0 and ch[1, 1] == 0  # H has no p
    assert contact.contact_overlap(6, 6) is contact.contact_overlap(6, 6)  # cached


def test_contact_coupling_is_non_zero_within_and_across_molecules(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    lone = max((o for o in methanol.occupied() if o.type == T.LONE_PAIR), key=lambda o: o.energy_ev)
    c_cl = next(
        o
        for o in chloromethane.virtual()
        if o.type == T.SIGMA_STAR
        and 17 in {chloromethane.atoms[a].atomic_number for a in o.parent_atoms}
    )
    kappa = token_contact_coupling(lone, methanol, c_cl, chloromethane)
    assert 0.05 < kappa < 2.0
    assert ks3_contact(lone, methanol, c_cl, chloromethane) < 0
    co_sigma_star = next(
        o
        for o in methanol.virtual()
        if o.type == T.SIGMA_STAR and 8 in {methanol.atoms[a].atomic_number for a in o.parent_atoms}
    )
    assert (
        token_contact_coupling(lone, methanol, co_sigma_star, methanol) > 0.05
    )  # intramolecular too
    M = ks3_contact_matrix(methanol.occupied(), methanol, methanol.virtual(), methanol)
    assert M.shape == (len(methanol.occupied()), len(methanol.virtual())) and (M < 0).all()


def test_4_chlorobutanol_intramolecular_features(sp_record):
    t = build_molecule_tokens(sp_record("OCCCCCl"))
    oxygen = next(a.atom_index for a in t.atoms if a.atomic_number == 8)
    chlorine = next(a.atom_index for a in t.atoms if a.atomic_number == 17)
    lone = max(
        (o for o in t.occupied() if o.type == T.LONE_PAIR and o.parent_atoms == (oxygen,)),
        key=lambda o: o.energy_ev,
    )
    c_cl = next(o for o in t.virtual() if o.type == T.SIGMA_STAR and chlorine in o.parent_atoms)
    feats = intramolecular_pair_features(lone, c_cl, t)
    # the converged full coupling is 0 by orthogonality; the centre-block part is not, even in
    # the extended MMFF conformer where O sits three bonds from the C-Cl bond (about 2e-5 with S)
    assert abs(feats["cross_center_coupling"]) > 1e-6
    print(f"4-chlorobutanol cross-centre coupling (S): {feats['cross_center_coupling']:.2e}")
    assert abs(lone.coeff_vector @ t.S @ c_cl.coeff_vector) < 1e-8
    assert feats["centroid_distance"] > 3.0 and feats["contact_coupling"] > 0.05
    assert feats["gap_kernel"] < 0 and feats["ks3_contact"] < 0
    assert (
        abs(cross_center_coupling(lone, c_cl, t.S, t.ao_atom) - feats["cross_center_coupling"])
        < 1e-12
    )


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
