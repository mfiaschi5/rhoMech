"""First-arrow ranking: candidates, multi-label top-k, the untrained KS scorers, the CLI."""

import json

import numpy as np
import pytest

from ksmech.arrows.steps import build_all_samples, sample_for_step, samples_path
from ksmech.physics.pairs import PAIR_FEATURES
from ksmech.physics.params import KSParams
from ksmech.physics.ranking import (
    ALL_SCORERS,
    SCORERS,
    class_table,
    evaluate,
    failure_classes,
    first_label,
    label_pairs,
    label_rank,
    pair_components,
    physical_pair_z,
    summarize_ranks,
)
from tests.test_arrows_steps import mini_processed, missing_iodide  # noqa: F401 (fixture)

F = {name: k for k, name in enumerate(PAIR_FEATURES)}


@pytest.fixture(scope="module")
def samples(mini_steps, token_loader, contact_kernel):
    names = ("sn2", "proton_transfer", "carbonyl_addition", "e2")
    return [sample_for_step(mini_steps[k], token_loader, contact_kernel) for k in names]


@pytest.fixture(scope="module")
def components(samples, token_loader, contact_kernel):
    return [pair_components(s, token_loader, contact_kernel) for s in samples]


def test_label_rank_is_pessimistic_and_marginalized():
    valid = np.ones((2, 3), bool)
    scores = np.array([[5.0, 1.0, 3.0], [3.0, 0.0, 2.0]])
    pos = np.zeros((2, 3), bool)
    pos[0, 2] = pos[1, 1] = True  # two labeled pairs: the better one (3.0) counts
    assert label_rank(scores, valid, pos) == 2  # 5.0 and the tie at 3.0 rank ahead of it
    valid[0, 0] = False  # an invalid candidate is not a competitor
    assert label_rank(scores, valid, pos) == 1
    valid[0, 2] = valid[1, 1] = False
    assert label_rank(scores, valid, pos) is None  # every labeled pair filtered out


def test_summary_counts_missing_labels_as_misses():
    out = summarize_ranks([0, 2, 4, None])
    assert out["top1"] == 0.25 and out["top3"] == 0.5 and out["top5"] == 0.75
    assert out["n"] == 4 and out["n_no_valid_label"] == 1 and out["median_rank"] == 2


def test_label_pairs_follow_the_label(samples):
    for s in samples:
        lab = first_label(s)
        pos = label_pairs(s, lab)
        occ_ids = s.n_atoms + np.flatnonzero(s.orbital_occupied)
        virt_ids = s.n_atoms + np.flatnonzero(~s.orbital_occupied)
        rows, cols = np.nonzero(pos)
        assert set(occ_ids[rows]) == set(lab.donor_ids)
        assert set(virt_ids[cols]) == set(lab.acceptor_ids)
        assert (pos & s.pair_valid).any()  # the mini steps are all scoreable


def test_physical_scaling(samples):
    s = samples[0]
    z = physical_pair_z(s, KSParams(K=2.0, eps=8.0))
    stored = s.pair_z.astype(float)
    assert np.allclose(z[..., F["ks3"]], 4.0 * stored[..., F["ks3"]])
    assert np.allclose(z[..., F["ks2"]], stored[..., F["ks2"]] / 8.0)
    assert np.allclose(z[..., F["gap_kernel"]], stored[..., F["gap_kernel"]])


def test_components_are_the_factors_of_ks3(samples, components):
    for s, c in zip(samples, components, strict=True):
        K = KSParams().K
        assert np.allclose(c["ks3"], (K * c["amp"] * c["fbar"]) ** 2 * c["gap_kernel"])
        assert np.allclose(c["ks3"], physical_pair_z(s)[..., F["ks3"]], rtol=1e-5, atol=1e-12)
        assert (c["amp"] >= 0).all() and c["fbar"].shape == s.pair_valid.shape


def test_evaluate_and_failure_classes(samples, components, token_loader):
    report = evaluate(samples, components, ALL_SCORERS)
    assert set(report) == set(ALL_SCORERS)
    assert {"ks3", "fbar2_g", "fbar2_g+ks2", "ks3+ks2"} <= set(SCORERS)
    for r in report.values():
        assert r["n"] == len(samples) and r["n_no_valid_label"] == 0
        assert all(isinstance(x, int) for x in r["ranks"])
        assert 0.0 <= r["top1"] <= r["top3"] <= r["top5"] <= 1.0
    ranks = report["ks3"]["ranks"]
    fc = failure_classes(samples, components, ranks, token_loader, SCORERS["ks3"])
    assert fc["n"] == len(samples) and fc["n_miss"] == sum(r >= 5 for r in ranks)
    assert all("->" in row["label"]["class"] for row in fc["rows"])
    table = class_table(fc["rows"], {n: report[n]["ranks"] for n in ("ks3", "fbar2_g")})
    assert sum(r["n"] for r in table.values()) == len(samples)


def test_components_with_an_ablation_kernel(samples, token_loader, table_kernel, components):
    other = pair_components(samples[0], token_loader, table_kernel)
    assert np.array_equal(other["gap_kernel"], components[0]["gap_kernel"])
    assert np.array_equal(other["fbar"], components[0]["fbar"])
    assert not np.allclose(other["ks3"], components[0]["ks3"])


def test_ks_rank_cli(mini_processed, token_loader, contact_kernel, monkeypatch):  # noqa: F811
    import ksmech.physics.cli as cli

    build_all_samples(
        "pmechdb", contact_kernel, ["random"], mini_processed, missing_iodide(token_loader)
    )
    assert samples_path("pmechdb", "random_test", mini_processed).exists()
    monkeypatch.setattr(cli, "TokenCache", lambda *a, **k: token_loader)
    base = ["ks-rank", "--source", "pmechdb", "--processed-dir", str(mini_processed)]
    cache = ["--oriented-cache", str(mini_processed / "oriented")]
    code = cli.main([*base, *cache, "--distance", "2.3", "2.6"])
    for d in ("2.3", "2.6"):
        path = mini_processed / "pmechdb" / f"ks_rank_report_oriented_d{d}.json"
        report = json.loads(path.read_text())
        assert set(report["scorers"]) == set(ALL_SCORERS) and report["distance"] == float(d)
        assert report["scorers"]["ks3"]["all"]["n"] == report["failure_classes"]["ks3"]["n"]
        assert set(report["scorers"]["ks3"]) == {"test", "trainval", "all"}
        assert {"fbar2_g+ks2"} <= set(report["failure_classes"]) and "clash" in report
        gate = report["gate"]
        assert gate["pass"] == (gate["ks3_trainval_top5"] > gate["fbar2_g_trainval_top5"])
    assert code in (0, 1)
    assert cli.main(["ks-rank", "--source", "pmechdb", "--processed-dir", "/nonexistent"]) == 2


def test_teacher_cell_is_a_valid_labeled_pair_with_most_site_weight(samples):
    from ksmech.physics.ranking import teacher_cell

    for s in samples:
        cell = teacher_cell(s)
        assert cell is not None and s.pair_valid[cell]
        assert label_pairs(s, first_label(s))[cell]


def test_refresh_validity_applies_the_current_filter(samples, token_loader):
    import dataclasses

    from ksmech.physics.pairs import valid_pairs
    from ksmech.physics.ranking import refresh_validity

    stale = dataclasses.replace(samples[0], pair_valid=np.ones_like(samples[0].pair_valid))
    fresh = refresh_validity(stale, token_loader)
    assert np.array_equal(fresh.pair_valid, valid_pairs(samples[0].tokens(token_loader)))
    assert fresh.pair_valid.sum() < stale.pair_valid.sum()


def test_components_carry_fragment_and_clash_flags(samples, components):
    for s, c in zip(samples, components, strict=True):
        assert c["inter"].dtype == bool and c["clash"].dtype == bool
        assert np.array_equal(c["clash"], s.pair_clash)  # same kernel as the stored sample
