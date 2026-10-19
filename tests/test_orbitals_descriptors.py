"""Continuous orbital descriptors: feature vector, no type index, spectral entries, geometry."""

from dataclasses import fields

import numpy as np

from ksmech.orbitals.classify import OrbitalType
from ksmech.orbitals.descriptors import (
    MODEL_INPUT,
    DescriptorConfig,
    OrbitalDescriptors,
    feature_names,
    spectral_entries,
)
from ksmech.orbitals.localize import localize_record
from ksmech.orbitals.tokens import OrbitalToken, build_molecule_tokens
from ksmech.qm.fukui import HARTREE_TO_EV

T = OrbitalType


def test_feature_vector_matches_the_names(sp_record):
    t = build_molecule_tokens(sp_record("CC(=O)C"))
    X = t.orbital_features()
    names = feature_names()
    assert X.shape == (len(t.orbitals), len(names)) and np.isfinite(X).all()
    assert len(names) == 1 + 1 + 4 + 12 + 2 + 1 + 2 + 1 + 1 + 8 + 8 + 3
    assert names[-3:] == ["spectral_mean", "spectral_variance", "retained_fraction"]
    assert names[:2] == ["occupation", "n_centers"] and names[-4] == "spectral_weight_7"
    assert not any("type" in n or "energy_ev" in n or "atom" in n for n in names)


def test_model_input_has_no_type_index(sp_record):
    t = build_molecule_tokens(sp_record("CC(=O)C"))
    X = t.orbital_features()
    for o in t.orbitals:
        o.type = T.UNKNOWN  # the diagnostic label must not touch the model input
    assert np.array_equal(t.orbital_features(), X)
    assert [f.name for f in fields(OrbitalToken) if f.metadata.get(MODEL_INPUT) is False] == [
        "type"
    ]
    bookkeeping = {
        "center_atoms",
        "center_s_coeff",
        "center_p_norm",
        "spectral_indices",
        "full_energies",
        "full_weights",
        "energy_ev",
        "centroid",
    }
    assert {
        f.name for f in fields(OrbitalDescriptors) if not f.metadata[MODEL_INPUT]
    } == bookkeeping


def test_occupation_centres_and_padding(sp_record):
    t = build_molecule_tokens(sp_record("CC(=O)C"))
    for o in t.occupied():
        assert abs(o.descriptors.occupation - 2.0) < 1e-8
    for o in t.virtual():
        assert abs(o.descriptors.occupation) < 1e-8
    lone = next(o for o in t.occupied() if o.type == T.LONE_PAIR)
    sigma = next(o for o in t.occupied() if o.type == T.SIGMA)
    d = lone.descriptors
    assert d.n_centers == 1 and d.cos_a == d.cos_b == 0.0 and d.overlap_sign == 0.0
    assert d.n_axis_centers == 0 and d.cos_character() == 0.0
    assert d.distance_ab == 0.0 and d.center_weights[1:].sum() == 0.0
    assert d.center_atoms[0] == lone.parent_atoms[0] and (d.center_atoms[1:] == -1).all()
    assert abs(d.center_spd[0].sum() - 1.0) < 1e-8 and d.center_spd[1:].sum() == 0.0
    s = sigma.descriptors
    assert s.n_centers == 2 and s.center_weights[0] >= s.center_weights[1] > 0
    c_h = next(
        o.descriptors
        for o in t.occupied()
        if o.type == T.SIGMA and 1 in {t.atoms[a].atomic_number for a in o.parent_atoms}
    )
    assert c_h.n_axis_centers == 1 and c_h.cos_character() > 0.9 and c_h.cos_b == 0.0
    assert 0.9 < s.distance_ab < 1.8 and s.overlap_sign == 1.0 and s.overlap_magnitude > 0


def test_radical_somo_occupation(sp_record):
    t = build_molecule_tokens(sp_record("[CH3]"))
    occ = [o.descriptors.occupation for o in t.occupied()]
    assert abs(sum(occ) - 7.0) < 1e-6 and min(occ) < 1.99  # 7 valence electrons, one SOMO


def test_spectral_entries_truncate_and_renormalize():
    u = np.array([0.9, 0.3, 0.3, 0.1, 0.05, 0.02])
    u = u / np.linalg.norm(u)
    e, idx = np.arange(6, dtype=float) * 2.0, np.arange(10, 16)
    i, en, w, kept = spectral_entries(
        u, e, idx, DescriptorConfig(spectral_min=0.01, spectral_max=3)
    )
    assert abs(kept - (u[:3] ** 2).sum()) < 1e-12
    assert list(i) == [10, 11, 12] and list(en) == [0.0, 2.0, 4.0]
    assert abs(w.sum() - 1.0) < 1e-12 and w[0] > w[1] >= w[2] > 0
    i, en, w, kept = spectral_entries(u, e, idx, DescriptorConfig(spectral_min=0.5, spectral_max=8))
    assert abs(kept - u[0] ** 2) < 1e-12
    assert list(i[:2]) == [10, -1] and w[0] == 1.0 and w[1:].sum() == 0.0 and len(w) == 8


def test_spectral_weights_and_energy_are_consistent(sp_record):
    rec = sp_record("CCO")
    t = build_molecule_tokens(rec)
    loc = localize_record(rec)
    assert np.allclose((loc.U_occ**2).sum(axis=1), 1.0)  # each canonical MO is fully spent
    for o in t.orbitals:
        d = o.descriptors
        assert 1 <= d.n_spectral() <= 8 and abs(d.spectral_weights.sum() - 1.0) < 1e-12
        assert (d.spectral_indices[: d.n_spectral()] >= 0).all()
        exact = o.coeff_vector @ rec.fock @ o.coeff_vector * HARTREE_TO_EV
        assert abs(d.energy_ev - exact) < 1e-8
        approx = d.spectral_weights @ d.spectral_energies
        assert abs(approx - exact) < 0.5  # the truncated set is a close summary


def test_two_centre_geometry_on_ethylene(sp_record):
    t = build_molecule_tokens(sp_record("C=C"))
    pi = next(o for o in t.occupied() if o.type == T.PI).descriptors
    cc = next(o for o in t.occupied() if o.type == T.SIGMA and set(o.parent_atoms) == {0, 1})
    pi_star = next(o for o in t.virtual() if o.type == T.PI_STAR).descriptors
    assert pi.cos_character() < 0.1 and pi.overlap_sign == 1.0
    assert cc.descriptors.cos_character() > 0.9 and 1.25 < cc.descriptors.distance_ab < 1.45
    assert pi_star.cos_character() < 0.1 and pi_star.overlap_sign == -1.0
    assert pi.radius_gyration > 0.5 and pi.center_spd[0, 1] > 0.95  # pure p on both carbons


def test_radius_of_gyration_orders_lone_pair_bond_ring(sp_record):
    acetone = build_molecule_tokens(sp_record("CC(=O)C"))
    benzene = build_molecule_tokens(sp_record("c1ccccc1"))
    lone = next(o for o in acetone.occupied() if o.type == T.LONE_PAIR).descriptors
    bond = next(o for o in acetone.occupied() if o.type == T.SIGMA).descriptors
    ring_pi = max((o.descriptors for o in benzene.occupied()), key=lambda d: d.n_centers)
    assert lone.radius_gyration < bond.radius_gyration < ring_pi.radius_gyration
    assert ring_pi.n_centers >= 3 and ring_pi.center_weights[2] > 0


def test_full_spectrum_and_scalars(sp_record):
    rec = sp_record("CCO")
    t = build_molecule_tokens(rec)
    for o in t.orbitals:
        d = o.descriptors
        n_block = (
            int((rec.occupations >= 0.5).sum())
            if o.kind == "occupied"
            else int((rec.occupations < 0.5).sum())
        )
        assert d.full_weights.shape == d.full_energies.shape == (n_block,)
        assert abs(d.full_weights.sum() - 1.0) < 1e-10
        assert abs(d.spectral_mean - d.energy_ev) < 1e-12
        assert d.spectral_variance >= 0 and 0 < d.retained_fraction <= 1.0 + 1e-12
        assert (
            abs(d.full_weights @ (d.full_energies - d.spectral_mean) ** 2 - d.spectral_variance)
            < 1e-9
        )
        assert (d.center_s_coeff[: d.n_centers] >= 0).all() and (d.center_p_norm >= 0).all()
