"""The trained rankers: candidate features, logistic regression, the numpy GBDT."""

import numpy as np
import pytest

from ksmech.arrows.steps import sample_for_step
from ksmech.physics import baseline as bl
from ksmech.physics.pairs import F, frozen_head
from ksmech.physics.ranking import first_label, label_pairs, physical_pair_z
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401 (fixture)


@pytest.fixture(scope="module")
def sets(mini_steps, token_loader, pair_kernels):
    names = ("sn2", "proton_transfer", "carbonyl_addition", "e2")
    samples = [sample_for_step(mini_steps[k], token_loader, pair_kernels) for k in names]
    return samples, [bl.candidate_set(s, token_loader) for s in samples]


def test_candidate_features(sets):
    samples, cs_list = sets
    assert len(bl.FEATURE_NAMES) == 61
    for s, cs in zip(samples, cs_list, strict=True):
        assert cs.X.shape == (int(s.pair_valid.sum()), 61) and np.isfinite(cs.X).all()
        z = physical_pair_z(s)[cs.rows, cs.cols]
        assert np.allclose(cs.X[:, :11], z)
        assert np.array_equal(cs.positive, label_pairs(s, first_label(s))[cs.rows, cs.cols])
        assert cs.positive.any() and not (cs.positive & cs.ignore).any()
        ks3 = cs.gap_sign * np.exp(z[:, F["log_abs_ks3"]])
        scores = bl.untrained_scores(cs)
        assert np.allclose(scores["ks3+ks2"], -(ks3 + z[:, F["ks2"]]))
        head = frozen_head(cs.X[:, :11], cs.gap_sign, cs.defined)
        assert np.allclose(scores["frozen_head"], head)  # logits in eV, no statistics
        assert cs.rank(np.where(cs.positive, 1.0, 0.0)) == 0  # a perfect score ranks first


def test_main_centres_are_the_largest_weight(sets):
    s = sets[0][0]
    main = bl.main_centres(s)
    for j in range(s.n_orbitals):
        rows = s.incidence[s.incidence[:, 0] == j]
        assert main[j] == int(rows[np.argmax(rows[:, 2]), 1])


def test_logistic_regression_runs_and_scores(sets):
    _, cs_list = sets
    model, scan = bl.select_logistic(cs_list, cs_list, grid=(0.1, 1.0))
    assert set(scan) == {0.1, 1.0} and model.l2 in scan
    r = bl.evaluate_scores(cs_list, [model.score(cs.X) for cs in cs_list])
    assert r["n"] == len(cs_list) and all(isinstance(x, int) for x in r["ranks"])


def _toy_sets(rng, n_sets=40, n=60):
    """Candidate sets where the positive is the row with the largest feature 3."""
    out = []
    for k in range(n_sets):
        X = rng.normal(size=(n, 61))
        pos = np.zeros(n, bool)
        pos[np.argmax(X[:, 3])] = True
        valid = np.ones((1, n), bool)
        out.append(
            bl.CandidateSet(
                f"t{k}",
                X,
                np.zeros(n, int),
                np.arange(n),
                pos,
                np.zeros(n, bool),
                -np.ones(n),
                np.ones((n, 11), bool),
                valid,
                pos[None, :].copy(),
            )
        )
    return out


def test_tree_finds_the_informative_split():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    g = np.where(X[:, 2] > 0.3, -1.0, 1.0)
    edges = bl.quantile_edges(X, 32)
    tree = bl.fit_tree(bl.binned(X, edges), g, np.ones(len(X)), bl.TreeConfig(depth=1))
    assert tree.feature[0] == 2
    pred = tree.predict(bl.binned(X, edges))
    assert pred[X[:, 2] > 0.5].min() > 0 > pred[X[:, 2] < 0.1].max()


def test_gbdt_learns_a_ranking_and_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    train, val = _toy_sets(rng), _toy_sets(rng, 15)
    cfg = bl.TreeConfig(max_rounds=60, check_every=10, patience=30, subsample=1.0)
    model = bl.fit_gbdt(train, val, cfg)
    r = bl.evaluate_scores(val, [model.score(cs.X) for cs in val])
    assert r["top3"] >= 0.8 and model.history and len(model.trees) >= 10
    back = bl.load_gbdt(bl.save_gbdt(model, tmp_path / "gbdt.npz"))
    X = val[0].X
    assert np.allclose(back.score(X), model.score(X))


def test_baseline_cli(mini_processed, token_loader, pair_kernels, monkeypatch):  # noqa: F811
    import json

    import ksmech.physics.cli as cli
    from ksmech.arrows.steps import build_all_samples

    build_all_samples(
        "pmechdb", pair_kernels, ["molecule"], mini_processed, missing_iodide(token_loader)
    )
    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: token_loader)
    quick = bl.TreeConfig(max_rounds=20, check_every=5)
    monkeypatch.setattr(bl, "TreeConfig", lambda: quick)
    code = cli.main(["baseline", "--source", "pmechdb", "--processed-dir", str(mini_processed)])
    out = mini_processed / "pmechdb"
    report = json.loads((out / "baseline_report_molecule.json").read_text())
    assert code in (0, 1) and "GBDT ranker" in report["results"]
    assert "not applicable" in report["loco"] or "found" in report["loco"]
    assert (out / "baseline_gbdt_molecule.npz").exists() and (
        out / "baseline_lr_molecule.npz"
    ).exists()
    from ksmech.data.provenance import read_header

    held = read_header(out / "baseline_gbdt_molecule_textbook-held-out.npz")
    assert held.extra["textbook_pools_in_training"] == []
    assert "GBDT ranker (textbook-held-out)" in report["results"]
    seen = read_header(out / "baseline_gbdt_molecule.npz").extra["textbook_pools_in_training"]
    assert seen == sorted(report["textbook_seen"])
