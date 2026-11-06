"""The paired comparison of two arms of decode runs (`eval.paired`, open-shell plan P0b)."""

import json
import math

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.eval import paired
from ksmech.eval.cli import DECODE_VERSION
from ksmech.eval.cli import main as eval_main

CONFIG = {"source": "pmechdb", "split": "molecule", "part": "val", "cohort": "closed_base",
          "scope": "all", "beam": 5, "limit": None, "caps": "legacy", "max_bonds": 8,
          "max_atoms": 2, "max_candidates": 48}  # fmt: skip


def _hits_file(path, hits, ids=None, version=DECODE_VERSION, **config):
    ids = ids or [f"s{k}" for k in range(len(hits))]
    pairs = zip(ids, hits, strict=True)
    rows = [{"step_id": i, "rank": 0 if h else None, "n_found": 1} for i, h in pairs]
    header = prov.stamp("decode_hits", version, {}, CONFIG | config)
    return prov.write_json(path, {"per_sample": rows}, header)


def test_mcnemar_is_the_exact_binomial_tail():
    a = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    b = np.array([1, 1, 1, 0, 1, 1, 1, 1])
    got = paired.mcnemar_one_sided(a, b)
    assert (got["a_miss_b_hit"], got["a_hit_b_miss"]) == (3, 1)
    assert got["p"] == pytest.approx((math.comb(4, 3) + math.comb(4, 4)) / 2**4)
    assert paired.mcnemar_one_sided(b, a)["p"] == pytest.approx(15 / 16)
    assert paired.mcnemar_one_sided(a, a)["p"] == 1.0


def test_mcnemar_on_a_large_discordant_set_stays_finite():
    rng = np.random.default_rng(0)
    a, b = rng.random(1200) < 0.6, rng.random(1200) < 0.7
    p = paired.mcnemar_one_sided(a, b)["p"]
    assert 0.0 <= p < 1e-3


def test_the_tolerance_is_two_pooled_sds_with_a_one_point_floor():
    fails = paired.decide([60.0, 61.0, 62.0], [63.0, 63.0, 63.0])
    assert fails["pooled_sd"] == pytest.approx(math.sqrt(2 / 4))
    assert fails["tolerance"] == pytest.approx(2 * math.sqrt(0.5)) and fails["a_fails"]
    close = paired.decide([62.5, 62.5, 62.5], [63.0, 63.0, 63.0])
    assert close["tolerance"] == 1.0 and not close["a_fails"]
    better = paired.decide([65.0, 64.0, 66.0], [63.0, 62.0, 64.0])
    assert not better["a_fails"] and better["difference"] == pytest.approx(2.0)


def test_majority_of_seeds():
    hits = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
    assert paired.majority(hits).tolist() == [True, False, True, False]


def test_arms_decoded_another_way_or_on_other_samples_are_refused(tmp_path):
    a = _hits_file(tmp_path / "a.json", [1, 0, 1])
    other_cohort = _hits_file(tmp_path / "b.json", [1, 1, 1], cohort="all")
    with pytest.raises(ValueError, match="cohort"):
        paired.compare_arms([a], [other_cohort])
    other_version = _hits_file(tmp_path / "c.json", [1, 1, 1], version=DECODE_VERSION - 1)
    with pytest.raises(ValueError, match="version"):
        paired.compare_arms([a], [other_version])
    other_ids = _hits_file(tmp_path / "d.json", [1, 1, 1], ids=["s0", "s1", "x"])
    with pytest.raises(ValueError, match="other samples"):
        paired.compare_arms([a], [other_ids])
    other_caps = _hits_file(tmp_path / "e.json", [1, 1, 1], caps="labels", max_bonds=6)
    with pytest.raises(ValueError, match="caps"):
        paired.compare_arms([a], [other_caps])


def test_the_compare_command_writes_the_decision(tmp_path):
    rng = np.random.default_rng(1)
    general = [_hits_file(tmp_path / f"g{s}.json", rng.random(200) < 0.6) for s in range(3)]
    legacy = [_hits_file(tmp_path / f"l{s}.json", rng.random(200) < 0.62) for s in range(3)]
    out = tmp_path / "compare.json"
    argv = ["compare", "--a", *map(str, general), "--b", *map(str, legacy), "--out", str(out)]
    assert eval_main(argv) == 0
    report = json.loads(out.read_text())
    assert report["n"] == 200 and len(report["seed_pairs"]) == 9
    assert set(report["decision"]) >= {"mean_a", "mean_b", "pooled_sd", "tolerance", "a_fails"}
    assert 0.0 <= report["majority"]["p"] <= 1.0
    bad = _hits_file(tmp_path / "bad.json", [1] * 200, cohort="all")
    assert eval_main(["compare", "--a", str(general[0]), "--b", str(bad)]) == 2
