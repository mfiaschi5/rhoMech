"""Open-shell plan P9, steps 5 and 6: the per-source split map, the joint run's leak guard and
validation cohort, what run.json states, serve's listing of a joint checkpoint, the caps by the
samples' source, the open-shell metrics and the paired comparison of decode files."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ksmech.arrows import steps
from ksmech.eval import cli as eval_cli
from ksmech.eval.paired import compare_arms, paired_hits
from ksmech.serve.core.settings import load_settings
from ksmech.serve.ml.checkpoints import NAME_RE, list_checkpoints
from ksmech.train import loop
from ksmech.train.loop import (
    LEAK_GUARD,
    OPEN_METRICS,
    TrainConfig,
    evaluate_sources,
    load_part,
    load_source_part,
    run_notes,
    run_pair_stats,
    split_of,
    stop_weight,
)
from tests.test_eval_paired import _hits_file

JOINT = TrainConfig(source="pmechdb+rmechdb", split="pmechdb=molecule,rmechdb=rc_molecule")


def _sample(step_id: str) -> SimpleNamespace:
    return SimpleNamespace(step_id=step_id, n_atoms=3, n_orbitals=4, drawings=[], source=None)


def _patched(monkeypatch, ids=("a", "b", "c"), held=("b",)):
    """Stored views of `ids`; the cohorts hold `held`. Returns the view paths read."""
    read = []

    def load(path, allow_stale=False, ids_=None, **kw):
        read.append(path.name)
        want = kw.get("ids", ids_)
        return [_sample(i) for i in (ids if want is None else want)]

    monkeypatch.setattr(loop, "load_samples", load)
    monkeypatch.setattr(loop, "sample_ids", lambda path: list(ids))
    monkeypatch.setattr(loop, "load_cohort", lambda source, name, processed: set(held))
    return read


def test_a_plain_split_is_every_sources():
    cfg = TrainConfig(source="pmechdb+reactmech", split="molecule")
    assert [split_of(cfg, s) for s in ("pmechdb", "reactmech")] == ["molecule", "molecule"]


def test_a_map_names_each_source():
    assert split_of(JOINT, "rmechdb") == "rc_molecule"
    assert split_of(JOINT, "pmechdb") == "molecule"
    for text in ("pmechdb=molecule", "pmechdb=molecule,rmechdb=x,reactmech=y"):
        with pytest.raises(ValueError, match="must name each"):
            split_of(TrainConfig(source=JOINT.source, split=text), "pmechdb")


def test_a_joint_run_resolves_each_sources_views(monkeypatch, tmp_path):
    read = _patched(monkeypatch, held=())
    samples, info = load_part(JOINT, "val", tmp_path)
    assert sorted(samples) == ["pmechdb", "rmechdb"]
    assert read == ["samples_molecule_val.json", "samples_rc_molecule_val.json"]
    assert [p.rsplit("/", 2)[1] for p in info["path"]] == ["pmechdb", "rmechdb"]


def test_the_joint_run_leaves_the_leak_guard_cohort_out_of_training(monkeypatch, tmp_path):
    _patched(monkeypatch)
    keep, info = load_source_part(JOINT, "rmechdb", "train", tmp_path)
    assert [s.step_id for s in keep] == ["a", "c"]
    assert info["skipped"] == {LEAK_GUARD: 1} and info["n"] == 3


def test_the_guard_touches_neither_evaluation_nor_a_run_without_pmechdb(monkeypatch, tmp_path):
    _patched(monkeypatch)
    assert len(load_source_part(JOINT, "rmechdb", "val", tmp_path)[0]) == 3
    assert len(load_source_part(JOINT, "pmechdb", "train", tmp_path)[0]) == 3
    alone = TrainConfig(source="rmechdb", split="rc_molecule")
    assert len(load_source_part(alone, "rmechdb", "train", tmp_path)[0]) == 3


def test_the_validation_cohort_restricts_only_its_sources_val(monkeypatch, tmp_path):
    _patched(monkeypatch)
    cfg = TrainConfig(source=JOINT.source, split=JOINT.split, val_cohort="pmechdb=closed_base")
    assert [s.step_id for s in load_source_part(cfg, "pmechdb", "val", tmp_path)[0]] == ["b"]
    assert len(load_source_part(cfg, "rmechdb", "val", tmp_path)[0]) == 3
    assert len(load_source_part(cfg, "pmechdb", "test", tmp_path)[0]) == 3


def test_a_joint_run_has_no_stop_term():
    assert stop_weight(JOINT) == 0.0
    assert stop_weight(TrainConfig(source="rmechdb")) == 0.0
    assert stop_weight(TrainConfig(source="reactmech")) == 0.5


def test_run_json_states_the_selection_and_the_statistics():
    cfg = TrainConfig(source=JOINT.source, split=JOINT.split, val_cohort="pmechdb=closed_base")
    notes = run_notes(cfg, None)
    assert notes["selection"] == (
        "best.pt: the highest anchor top-1 pooled over pmechdb molecule val, cohort closed_base "
        "and rmechdb rc_molecule val"
    )
    assert notes["pair_statistics"] == "fitted on the training samples"
    assert "on pmechdb molecule val" in run_notes(TrainConfig(), None)["selection"]


def test_an_open_shell_run_fits_its_statistics_even_on_one_source(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(loop, "pair_stats", lambda train, cells, extra: calls.append(extra) or "s")
    monkeypatch.setattr(loop, "load_pair_stats", lambda path: "stored")
    assert run_pair_stats(TrainConfig(), [], tmp_path, "all", ("g2",)) == (None, "s")
    assert calls == [("g2",)]
    assert run_pair_stats(TrainConfig(), [], tmp_path, "all")[1] == "stored"
    assert run_pair_stats(JOINT, [], tmp_path, "all")[0] is None


def test_serve_lists_a_joint_checkpoint(tmp_path):
    assert NAME_RE.fullmatch("pmechdb+rmechdb")
    run = tmp_path / "data" / "models" / "pmechdb+rmechdb" / "joint_s0"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"not a real checkpoint")
    found = list_checkpoints(load_settings({}, data_root=tmp_path, chemistry="synthetic"))
    assert [(c.source, c.run) for c in found] == [("pmechdb+rmechdb", "joint_s0")]


def test_decode_reads_its_caps_from_the_samples_source(monkeypatch, tmp_path):
    seen = []

    def caps(mode, train, version, overrides):
        seen.append(train)
        return {"max_bonds": 15, "max_atoms": 4, "max_candidates": 48}

    monkeypatch.setattr("ksmech.eval.decode.decode_caps", caps)
    argv = ["decode", "--run", "joint_s0", "--run-source", JOINT.source, "--source", "rmechdb"]
    argv += ["--split", "rc_molecule", "--caps", "labels", "--processed-dir", str(tmp_path)]
    args = eval_cli.build_parser().parse_args(argv)
    monkeypatch.setattr(eval_cli.prov, "file_input", lambda path: {str(path): "h"})
    assert eval_cli._decode_caps(args)[0]["max_atoms"] == 4
    assert seen == [steps.samples_path("rmechdb", "rc_molecule_train", tmp_path)]


def test_the_decode_options_carry_the_filter():
    argv = ["decode", "--run", "x"]
    assert not eval_cli.build_parser().parse_args(argv).closed_minus1_filter
    flag = [*argv, "--closed-minus1-filter"]
    assert eval_cli.build_parser().parse_args(flag).closed_minus1_filter


def _rates(n, m, top1, opened):
    keys = {"loss": 0.0, "anchor_nll": 0.0, "top3": 0.0, "top5": 0.0, "r_exact": 0.0, "p_stop": 0.0}
    return keys | {"top1": top1, "n": n, "open_n": m, "open_top1": opened, "open_r_exact": 0.0}


def test_the_open_shell_subset_pools_by_its_own_count(monkeypatch):
    per = {"pmechdb": _rates(90, 1, 0.5, 1.0), "rmechdb": _rates(10, 9, 0.1, 0.0)}
    monkeypatch.setattr(loop, "evaluate", lambda model, loader, *a: per[loader])
    out = evaluate_sources(None, {"pmechdb": "pmechdb", "rmechdb": "rmechdb"})
    assert out["top1"] == pytest.approx(0.46)
    assert out["open_n"] == 10 and out["open_top1"] == pytest.approx(0.1)
    assert out["rmechdb_open_n"] == 9 and out["rmechdb_top1"] == 0.1
    assert set(OPEN_METRICS) <= set(out)


def test_evaluate_counts_the_pools_with_an_odd_n(monkeypatch):
    batch = {"step_id": ["a", "b"], "nonbonding": torch.tensor([[1, 2], [2, 2]], dtype=torch.int8)}

    class Model:
        def eval(self):
            return self

        decoder = None

        def encode(self, batch, warmup):
            return None, None, None, None

    parts = {"loss": torch.zeros(2), "anchor_nll": torch.zeros(2), "lp_anchor": None,
             "p_stop": torch.zeros(2)}  # fmt: skip
    monkeypatch.setattr(loop, "step_loss", lambda *a: (None, parts))
    monkeypatch.setattr(loop, "anchor_ranks", lambda lp, b: torch.tensor([0, 9]))
    monkeypatch.setattr(loop, "greedy_edit", lambda *a: (None, None))
    monkeypatch.setattr(loop, "edit_matches", lambda p, a, b: torch.tensor([False, True]))
    out = loop.evaluate(Model(), [batch], torch.device("cpu"), None)
    assert (out["open_n"], out["open_top1"], out["open_r_exact"]) == (1, 1.0, 0.0)
    assert (out["n"], out["top1"], out["r_exact"]) == (2, 0.5, 0.5)


def test_compare_pairs_a_joint_and_a_control_run_on_one_cohort(tmp_path):
    a = _hits_file(tmp_path / "a.json", [1, 1, 0], cohort="closed_base", run_source=JOINT.source)
    b = _hits_file(tmp_path / "b.json", [1, 0, 0], cohort="closed_base", run_source="pmechdb")
    assert compare_arms([a], [b])["n"] == 3


def test_compare_pairs_a_decode_with_the_filter_and_one_without(tmp_path):
    a = _hits_file(tmp_path / "a.json", [1, 1, 0], closed_minus1_filter=True)
    b = _hits_file(tmp_path / "b.json", [1, 0, 0], closed_minus1_filter=False)
    assert compare_arms([a], [b])["n"] == 3


def test_compare_refuses_another_cohort(tmp_path):
    a = _hits_file(tmp_path / "a.json", [1, 1, 0], cohort="closed_base")
    b = _hits_file(tmp_path / "b.json", [1, 0, 0], cohort="open_shell")
    with pytest.raises(ValueError, match="cohort"):
        paired_hits({"a": [a], "b": [b]})
