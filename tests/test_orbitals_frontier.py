"""Per-atom frontier-orbital weights: degenerate blocks and the f+/f- generalization."""

import numpy as np

from ksmech.orbitals.frontier import LEVELS, frontier_blocks, frontier_weights
from ksmech.orbitals.tokens import build_molecule_tokens


def test_benzene_frontier_blocks_are_degenerate_pairs(sp_record):
    rec = sp_record("c1ccccc1")
    blocks = frontier_blocks(rec.orbital_energies_ev, rec.occupations)
    assert set(blocks) == set(LEVELS)
    assert len(blocks["homo"]) == 2 and len(blocks["lumo"]) == 2  # e1g / e2u pairs
    w = frontier_weights(rec)
    sums = w.sum(axis=0)  # one population per MO of the block
    assert w.shape == (12, 4) and np.allclose(sums, [len(blocks[lv]) for lv in LEVELS], atol=1e-6)
    carbons = w[:6, 1]
    assert np.allclose(carbons, carbons.mean(), atol=0.02)  # the summed pair is symmetric


def test_frontier_weights_generalize_the_fukui_populations(sp_record):
    rec = sp_record("CC(=O)C")
    w = frontier_weights(rec)
    assert np.allclose(w[:, 1], rec.fukui_fmo_minus, atol=1e-10)  # non-degenerate HOMO = f-
    assert np.allclose(w[:, 2], rec.fukui_fmo_plus, atol=1e-10)  # non-degenerate LUMO = f+
    t = build_molecule_tokens(rec)
    for a in t.atoms:
        assert np.allclose(a.fmo_weights, w[a.atom_index])


def test_missing_levels_are_nan(sp_record):
    proton = frontier_weights(sp_record("[H+]"))
    assert (
        np.isnan(proton[0, :2]).all() and np.isclose(proton[0, 2], 1.0) and np.isnan(proton[0, 3])
    )
    hydride = frontier_weights(sp_record("[H-]"))
    assert np.isclose(hydride[0, 1], 1.0) and np.isnan(hydride[0, [0, 2, 3]]).all()


def test_frozen_orbital_fukui_is_symmetric_on_benzene(sp_record):
    """One MO of the degenerate e1g pair gave the six carbons 0.27 / 0.22 / 0.00 (review Q1);
    the block average gives each carbon the same value, and f sums to 1 over the atoms."""
    from ksmech.orbitals.frontier import frozen_orbital_fukui_blocks

    rec = sp_record("c1ccccc1")
    f_plus, f_minus = frozen_orbital_fukui_blocks(rec)
    carbons = rec.atomic_numbers == 6
    for f in (f_plus, f_minus):
        assert np.ptp(f[carbons]) < 1e-3, f[carbons]
        assert abs(f.sum() - 1.0) < 1e-6


def test_equivalent_tokens_get_one_spectral_set(sp_record):
    """Methane's four C-H sigma LMOs split their weight differently over the degenerate t2
    MOs; summed per level, their spectral sets are identical."""
    from ksmech.orbitals.tokens import build_molecule_tokens

    tokens = build_molecule_tokens(sp_record("C"))
    sets = [
        np.round(np.c_[o.descriptors.spectral_energies, o.descriptors.spectral_weights], 3)
        for o in tokens.occupied()
    ]
    assert len(sets) == 4 and all(np.array_equal(sets[0], s) for s in sets[1:]), sets


def test_degenerate_levels_use_the_anchor_rule():
    from ksmech.orbitals.descriptors import degenerate_levels

    e = np.array([-10.0, -9.97, -9.94, -5.0, -5.04])
    assert degenerate_levels(e, 0.05).tolist() == [0, 0, 1, 2, 2]


def test_the_exact_tolerance_splits_what_the_window_merges():
    """Two occupied levels 0.03 eV apart: one HOMO block in the 0.05 eV window, two levels
    (HOMO and HOMO-1) with the exact tolerance (open-shell plan P0a, item 3)."""
    from ksmech.orbitals.frontier import EXACT_EV

    e = np.array([-12.0, -10.03, -10.0, -1.0, 2.0])
    occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0])
    window, exact = frontier_blocks(e, occ), frontier_blocks(e, occ, EXACT_EV)
    assert window["homo"].tolist() == [1, 2] and window["homo_minus_1"].tolist() == [0]
    assert exact["homo"].tolist() == [2] and exact["homo_minus_1"].tolist() == [1]
    assert window["lumo"].tolist() == exact["lumo"].tolist() == [3]


def test_benzene_pairs_are_one_block_under_both_tolerances(sp_record):
    from ksmech.orbitals.frontier import DEGENERACY_EV, EXACT_EV

    rec = sp_record("c1ccccc1")
    window = frontier_blocks(rec.orbital_energies_ev, rec.occupations, DEGENERACY_EV)
    exact = frontier_blocks(rec.orbital_energies_ev, rec.occupations, EXACT_EV)
    assert {k: v.tolist() for k, v in window.items()} == {k: v.tolist() for k, v in exact.items()}


def test_atom_tokens_store_both_frontier_sets(sp_record, smiles):
    """On the ten fixed molecules the legacy columns are the 0.05 eV blocks (schema 9's
    content) and the stored extras are the exact blocks and the finite-difference f+ / f-."""
    from ksmech.orbitals.frontier import EXACT_EV, frozen_orbital_fukui_blocks
    from ksmech.orbitals.tokens import ATOM_EXTRA_FEATURES, atom_feature_names

    names = atom_feature_names()
    for name, smi in smiles.items():
        rec = sp_record(smi)
        t = build_molecule_tokens(rec)
        legacy = np.where(t.atom_missing(), np.nan, t.atom_features())
        w, w_exact = frontier_weights(rec), frontier_weights(rec, EXACT_EV)
        cols = [names.index(f"fmo_{lv}") for lv in LEVELS]
        np.testing.assert_array_equal(legacy[:, cols], w, err_msg=name)
        f_plus, f_minus = frozen_orbital_fukui_blocks(rec)
        np.testing.assert_array_equal(legacy[:, names.index("f_plus_fmo")], f_plus)
        np.testing.assert_array_equal(legacy[:, names.index("f_minus_fmo")], f_minus)
        extra = t.atom_extra()
        assert extra.shape == (len(t.atoms), len(ATOM_EXTRA_FEATURES))
        np.testing.assert_array_equal(extra[:, 4:], w_exact, err_msg=name)
        exact_plus, exact_minus = frozen_orbital_fukui_blocks(rec, EXACT_EV)
        np.testing.assert_array_equal(extra[:, 2], exact_plus)
        np.testing.assert_array_equal(extra[:, 3], exact_minus)
        np.testing.assert_array_equal(extra[:, 0], [a.f_plus for a in t.atoms])
        np.testing.assert_array_equal(extra[:, 1], [a.f_minus for a in t.atoms])
