"""The candidate pairs of the first arrow: their features, labels, untrained score and ranks."""

import numpy as np
import pytest

from ksmech.arrows.steps import build_all_samples, sample_for_step
from ksmech.physics import baseline as bl
from ksmech.physics.stage_a import STAGE_A_FEATURES
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401 (fixture)


@pytest.fixture(scope="module")
def sets(mini_steps, token_loader):
    names = ("sn2", "proton_transfer", "carbonyl_addition", "e2")
    samples = [sample_for_step(mini_steps[k], token_loader) for k in names]
    return samples, [bl.candidate_set(s, token_loader) for s in samples]


def test_label_rank_is_pessimistic_and_marginalized():
    valid = np.ones((2, 3), bool)
    scores = np.array([[5.0, 1.0, 3.0], [3.0, 0.0, 2.0]])
    pos = np.zeros((2, 3), bool)
    pos[0, 2] = pos[1, 1] = True  # two labeled pairs: the better one (3.0) counts
    assert bl.label_rank(scores, valid, pos) == 2  # 5.0 and the tie at 3.0 rank ahead of it
    valid[0, 0] = False  # an invalid candidate is not a competitor
    assert bl.label_rank(scores, valid, pos) == 1
    valid[0, 2] = valid[1, 1] = False
    assert bl.label_rank(scores, valid, pos) is None  # every labeled pair filtered out


def test_summary_counts_missing_labels_as_misses():
    out = bl.summarize_ranks([0, 2, 4, None])
    assert out["top1"] == 0.25 and out["top3"] == 0.5 and out["top5"] == 0.75
    assert out["n"] == 4 and out["n_no_valid_label"] == 1 and out["median_rank"] == 2


def test_label_pairs_follow_the_label(sets):
    for s in sets[0]:
        lab = bl.first_label(s)
        pos = bl.label_pairs(s, lab)
        occ_ids = s.n_atoms + np.flatnonzero(s.orbital_occupied)
        virt_ids = s.n_atoms + np.flatnonzero(~s.orbital_occupied)
        rows, cols = np.nonzero(pos)
        assert set(occ_ids[rows]) == set(lab.donor_ids)
        assert set(virt_ids[cols]) == set(lab.acceptor_ids)
        assert (pos & s.pair_valid).any()  # the mini steps are all scoreable


def test_candidate_features(sets):
    """The geometry-free pair features as stored, NaN where undefined, then the token columns;
    the untrained score is -G."""
    samples, cs_list = sets
    n_pair = len(STAGE_A_FEATURES)
    assert bl.FEATURE_NAMES[:n_pair] == list(STAGE_A_FEATURES) and len(bl.FEATURE_NAMES) == 61
    for s, cs in zip(samples, cs_list, strict=True):
        assert cs.X.shape == (int(s.pair_valid.sum()), len(bl.FEATURE_NAMES))
        undefined = ~s.pair_defined[cs.rows, cs.cols]
        assert np.array_equal(np.isnan(cs.X[:, :n_pair]), undefined)
        assert np.isfinite(cs.X[:, n_pair:]).all()
        assert np.array_equal(cs.positive, bl.label_pairs(s, bl.first_label(s))[cs.rows, cs.cols])
        assert cs.positive.any() and not (cs.positive & cs.ignore).any()
        assert np.allclose(bl.gap_score(cs), -s.pair_z[cs.rows, cs.cols, 0])
        assert cs.rank(np.where(cs.positive, 1.0, 0.0)) == 0  # a perfect score ranks first


def test_main_centres_are_the_largest_weight(sets):
    s = sets[0][0]
    main = bl.main_centres(s)
    for j in range(s.n_orbitals):
        rows = s.incidence[s.incidence[:, 0] == j]
        assert main[j] == int(rows[np.argmax(rows[:, 2]), 1])


def test_gbdt_reader_skips_and_counts_samples_without_an_orbital_label(
    mini_processed,  # noqa: F811
    token_loader,
    monkeypatch,
):
    """A sample whose arrows have no orbital pair (a lone heterolysis) has no first arrow to
    rank: the LightGBM reader (`_read_part`) skips it and counts it as NO_ORBITAL_READING."""
    import argparse
    import dataclasses

    import ksmech.arrows.steps as steps
    import ksmech.physics.cli as cli

    loader = missing_iodide(token_loader)
    build_all_samples("pmechdb", ["molecule"], mini_processed, loader)
    args = argparse.Namespace(source="pmechdb", processed_dir=mini_processed, split="molecule")

    def strip_first(samples):
        return [dataclasses.replace(s, labels=[]) if k == 0 else s for k, s in enumerate(samples)]

    real_iter = steps.iter_samples
    monkeypatch.setattr(
        steps, "iter_samples", lambda *a, **k: iter(strip_first(real_iter(*a, **k)))
    )
    n_view, n, no_reading = cli._read_part(args, "train", token_loader, lambda *a: 0)
    assert no_reading == 1 and n == n_view - 1
