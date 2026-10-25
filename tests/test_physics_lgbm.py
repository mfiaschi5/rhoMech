"""The fast LightGBM ranker: negative sampling, the early-stopping metric, fitting, persistence,
and `ksmech-physics gbdt` on the mini dataset."""

import json

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.physics import baseline as bl
from ksmech.physics import lgbm
from ksmech.physics.ranking import label_rank, summarize_ranks
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401 (fixture)
from tests.test_physics_baseline import _toy_sets

LGBMConfig = lgbm.LGBMConfig  # the real class, before a test patches the module's


def _set(step_id, n, positive, ignore=()):
    pos = np.zeros(n, bool)
    pos[list(positive)] = True
    ign = np.zeros(n, bool)
    ign[list(ignore)] = True
    return bl.CandidateSet(
        step_id,
        np.zeros((n, 3)),
        np.zeros(n, int),
        np.arange(n),
        pos,
        ign,
        np.ones(n),
        np.ones((n, 3), bool),
        np.ones((1, n), bool),
        pos[None, :].copy(),
    )


def test_sample_negatives_keeps_positives_hard_and_random():
    cs = _set("s1", 200, positive=[5, 7], ignore=[0, 1, 2])
    score = np.arange(200, dtype=float)  # the hardest negatives are the last rows
    cfg = lgbm.LGBMConfig()
    keep = lgbm.sample_negatives(cs, score, cfg)
    assert {5, 7} <= set(keep) and not {0, 1, 2} & set(keep)
    negatives = [k for k in keep if not cs.positive[k]]
    assert len(negatives) == 50 and set(range(180, 200)) <= set(keep)
    assert np.array_equal(keep, lgbm.sample_negatives(cs, score, cfg))  # seeded by step id
    other = lgbm.sample_negatives(_set("s2", 200, [5, 7], [0, 1, 2]), score, cfg)
    assert not np.array_equal(keep, other)
    small = _set("s3", 10, positive=[0])
    assert len(lgbm.sample_negatives(small, np.zeros(10), cfg)) == 10  # fewer than 50: all


def test_group_ranks_match_label_rank_with_ties():
    rng = np.random.default_rng(0)
    sizes = np.array([7, 0, 5, 9, 4])
    scores = rng.integers(0, 4, size=sizes.sum()).astype(float)  # many ties
    positive = rng.random(sizes.sum()) < 0.25
    positive[7:12] = False  # one group without a positive: a miss
    ranks = lgbm.group_ranks(scores, positive, sizes)
    starts = np.r_[0, np.cumsum(sizes)]
    expected = []
    for g in range(len(sizes)):
        sl = slice(starts[g], starts[g + 1])
        valid = np.ones((1, sizes[g]), bool)
        expected.append(label_rank(scores[None, sl], valid, positive[None, sl]))
    assert [None if r < 0 else int(r) for r in ranks] == expected
    for k in (1, 3, 5):
        assert lgbm.top_k(ranks, k) == pytest.approx(summarize_ranks(expected)[f"top{k}"])


def _rows(sets, cfg):
    rows = lgbm.TrainRows()
    for cs in sets:
        rows.add(cs, cs.X[:, 0], cfg)
    return rows


def test_fit_learns_a_ranking_and_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    train, val = _toy_sets(rng, 80), _toy_sets(rng, 20)
    cfg = lgbm.LGBMConfig(max_rounds=60, patience=20, min_data_in_leaf=5, n_jobs=1)
    names = [f"f{k}" for k in range(61)]
    model = lgbm.fit_lgbm(_rows(train, cfg), lgbm.EvalRows.of(val), names, cfg)
    r = bl.evaluate_scores(val, [model.score(cs.X) for cs in val])
    assert r["top3"] >= 0.8 and model.n_trees >= 1
    path = lgbm.save_lgbm(model, tmp_path / "m.npz", cfg)
    if prov.read_header(path).code_dirty:  # a dirty tree's model is refused, by design
        with pytest.raises(prov.StaleArtifact):
            lgbm.load_lgbm(path, names, cfg)
    back = lgbm.load_lgbm(path, names, cfg, allow_stale=True)
    assert np.allclose(back.score(val[0].X), model.score(val[0].X))
    with pytest.raises(prov.StaleArtifact):  # another configuration
        lgbm.load_lgbm(path, names, lgbm.LGBMConfig(n_hard=5))
    with pytest.raises(ValueError):
        lgbm.load_lgbm(path, names[:-1], allow_stale=True)


def test_untrainable_steps_are_counted():
    cfg = lgbm.LGBMConfig()
    rows = lgbm.TrainRows()
    rows.add(_set("none", 10, positive=[]), np.zeros(10), cfg)
    rows.add(_set("one", 10, positive=[3]), np.zeros(10), cfg)
    assert rows.n_untrainable == 1 and rows.groups == [10] and rows.n_rows == 10


@pytest.mark.parametrize("features", ["contact", "geomfree"])
def test_gbdt_cli(features, mini_processed, token_loader, pair_kernels, monkeypatch):  # noqa: F811
    import ksmech.physics.cli as cli
    from ksmech.arrows.contacts import build_all_contacts
    from ksmech.arrows.steps import build_all_samples

    loader = missing_iodide(token_loader)
    build_all_samples("pmechdb", ["molecule"], mini_processed, loader)
    build_all_contacts("pmechdb", pair_kernels, "molecule", mini_processed, loader)
    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: token_loader)
    quick = {"max_rounds": 20, "patience": 10, "min_data_in_leaf": 1}
    monkeypatch.setattr(lgbm, "LGBMConfig", lambda **k: LGBMConfig(**(quick | k)))
    argv = ["gbdt", "--source", "pmechdb", "--processed-dir", str(mini_processed)]
    code = cli.main([*argv, "--features", features, "--n-jobs", "1"])
    out = mini_processed / "pmechdb"
    report = json.loads((out / f"gbdt_lgbm_report_molecule_{features}.json").read_text())
    assert code in (0, 1) and "LightGBM ranker" in report["results"]
    assert "untrained gap_kernel" in report["results"]
    assert ("untrained ks3+ks2" in report["results"]) == (features == "contact")
    assert report["counts"]["train"]["read"] <= report["counts"]["train"]["stored"]
    assert report["peak_rss_mb"] > 0 and report["config"]["lgbm"]["n_hard"] == 20
    held = prov.read_header(out / f"gbdt_lgbm_molecule_{features}_textbook-held-out.npz")
    assert held.kind == "baseline_lgbm" and held.extra["textbook_pools_in_training"] == []


def test_the_thread_count_does_not_change_the_model():
    """`deterministic` and `force_row_wise`: one thread or four give the same trees and scores,
    so `--n-jobs` only sets the speed and stays out of the model's configuration."""
    rng = np.random.default_rng(1)
    train, val = _toy_sets(rng, 80), _toy_sets(rng, 20)
    names = [f"f{k}" for k in range(61)]
    models = []
    for n_jobs in (1, 4):
        cfg = lgbm.LGBMConfig(max_rounds=60, patience=20, min_data_in_leaf=5, n_jobs=n_jobs)
        models.append(lgbm.fit_lgbm(_rows(train, cfg), lgbm.EvalRows.of(val), names, cfg))
    one, four = models
    assert one.n_trees == four.n_trees >= 1
    assert all(np.array_equal(one.score(cs.X), four.score(cs.X)) for cs in val)
    assert "n_jobs" not in lgbm.model_config(names, cfg)["lgbm"]
    Xs = [cs.X for cs in val] + [np.zeros((0, 61))]  # an empty step too
    batched = one.score_sets(Xs, n_jobs=4, batch_rows=50)  # several batches
    assert len(batched) == len(Xs) and batched[-1].shape == (0,)
    assert all(np.allclose(b, one.score(X)) for b, X in zip(batched[:-1], Xs[:-1], strict=True))


@pytest.mark.parametrize("features", ["contact", "geomfree"])
def test_reading_in_worker_processes_gives_the_same_rows_and_sets(
    features,
    mini_processed,  # noqa: F811
    mini_token_cache,
    pair_kernels,
    monkeypatch,
):
    """`--workers 2` on chunks of one sample: the training rows (negatives are drawn per step
    id), the val / test candidate sets and their untrained scores are those of one process,
    in the same order (undefined geometry-free features are NaN in both)."""
    import argparse

    import ksmech.physics.cli as cli
    from ksmech.arrows.contacts import build_all_contacts
    from ksmech.arrows.steps import build_all_samples
    from ksmech.physics import parallel

    build_all_samples("pmechdb", ["molecule"], mini_processed, mini_token_cache)
    build_all_contacts("pmechdb", pair_kernels, "molecule", mini_processed, mini_token_cache)
    monkeypatch.setattr(parallel, "CHUNK", 1)
    cfg = lgbm.LGBMConfig()
    read = {}
    for workers in (1, 2):
        args = argparse.Namespace(
            source="pmechdb", split="molecule", processed_dir=mini_processed, workers=workers
        )
        train = cli._read_kind(args, features, "train", mini_token_cache, "train", cfg)
        evals = {
            p: cli._read_kind(args, features, p, mini_token_cache, "eval", cfg)
            for p in ("val", "test")
        }
        read[workers] = (train, evals)
    (t1, e1), (t2, e2) = read[1], read[2]
    assert t1["n"] >= 1 and (t1["n"], t1["n_view"], t1["seen"]) == (
        t2["n"],
        t2["n_view"],
        t2["seen"],
    )
    for key in ("all", "held"):
        a, b = t1[key], t2[key]
        assert a.groups == b.groups and a.n_untrainable == b.n_untrainable
        assert all(
            np.array_equal(x, y, equal_nan=True) for x, y in zip(a.X + a.y, b.X + b.y, strict=True)
        )
    for p in ("val", "test"):
        assert e1[p]["n"] == e2[p]["n"] and e1[p]["textbook"] == e2[p]["textbook"]
        assert [cs.step_id for cs in e1[p]["sets"]] == [cs.step_id for cs in e2[p]["sets"]]
        assert all(
            np.array_equal(x.X, y.X, equal_nan=True)
            for x, y in zip(e1[p]["sets"], e2[p]["sets"], strict=True)
        )
        for name, scores in e1[p]["untrained"].items():
            other = e2[p]["untrained"][name]
            assert all(
                np.array_equal(x, y, equal_nan=True) for x, y in zip(scores, other, strict=True)
            )
