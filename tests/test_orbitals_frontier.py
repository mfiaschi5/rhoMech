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
