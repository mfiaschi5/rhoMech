"""The out-of-distribution switches (docs/decisions.md, 2026-11-15) on the fixture samples:
without the fragment embedding the model does not see the order of the reactants; the element
dropout masks E_Z in training only, and at 1 the element is never read; the balanced sampler
draws a frequent edit signature less; a probe part is scored every epoch and never selected
on. The defaults are the model and the training as they were."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.eval import ood
from ksmech.eval.embedding import embedding_parts
from ksmech.eval.importance import Group, permute_group
from ksmech.model.config import ModelConfig
from ksmech.train.batching import CellBudgetSampler, balance_weights
from ksmech.train.loop import TrainConfig, fit, load_probes, make_loader
from tests.test_model import ENCODERS, TINY, _two_source_stores, make, samples  # noqa: F401


def _encode(model, batch):
    with torch.no_grad():
        return model.encode(batch)[:3]


def test_the_defaults_are_the_model_as_it_was(samples):  # noqa: F811
    old = {k: v for k, v in TINY.to_dict().items() if k not in ("fragment_embedding",)}
    del old["element_dropout"]
    cfg = ModelConfig.from_dict(old)  # a checkpoint written before the switches
    assert cfg.fragment_embedding is True and cfg.element_dropout == 0.0
    keys = set(make(samples).state_dict())
    assert "encoder.embed.fragment.weight" in keys
    without = set(make(samples, fragment_embedding=False).state_dict())
    assert keys - without == {"encoder.embed.fragment.weight"} and without <= keys


@pytest.mark.parametrize("encoder", ENCODERS)
def test_without_the_fragment_embedding_the_reactant_order_changes_nothing(samples, encoder):  # noqa: F811
    batch = collate_samples(samples)
    moved = ood.permute_fragments(batch, torch.Generator().manual_seed(0))
    assert not torch.equal(moved["fragment"], batch["fragment"])
    model = make(samples, encoder, fragment_embedding=False)
    for a, b in zip(_encode(model, batch), _encode(model, moved), strict=True):
        assert torch.equal(a, b)
    h0, h1 = _encode(make(samples, encoder), batch)[0], _encode(make(samples, encoder), moved)[0]
    assert not torch.allclose(h0, h1, atol=1e-4)  # as trained until now: it does


def test_the_embedding_trace_follows_the_switches(samples):  # noqa: F811
    batch = collate_samples(samples)
    for kw in ({"fragment_embedding": False}, {"element_dropout": 1.0}):
        model = make(samples, **kw)
        inp = model.prepare(batch)[0]
        with torch.no_grad():
            embed = model.encoder.embed
            parts = embedding_parts(embed, batch, inp)
            torch.testing.assert_close(sum(parts.values()), embed(batch, inp), atol=1e-5, rtol=0)
        assert parts["fragment"].any() == ("fragment_embedding" not in kw)


def test_element_dropout_masks_in_training_only(samples):  # noqa: F811
    batch = collate_samples(samples)
    plain, dropped = make(samples), make(samples, element_dropout=0.5)
    dropped.load_state_dict(plain.state_dict())
    assert torch.equal(_encode(plain, batch)[0], _encode(dropped, batch)[0])  # evaluation
    z = batch["atom_categorical"][..., 0]
    embed = dropped.encoder.embed.train()
    torch.manual_seed(0)
    rows = embed.elements(z)
    real = batch["is_atom"][:, : z.shape[1]]
    masked = (rows == 0) & real
    assert 0.2 < masked.float().sum() / real.sum() < 0.8
    assert torch.equal(rows[~masked], z[~masked])
    assert torch.equal(embed.eval().elements(z), z)
    with pytest.raises(ValueError, match="element_dropout"):
        make(samples, element_dropout=1.5)


def test_at_one_the_element_is_never_read(samples):  # noqa: F811
    batch = collate_samples(samples)
    swapped = permute_group(
        batch, Group("element", "atom_categorical", (0,)), torch.Generator().manual_seed(1)
    )
    assert not torch.equal(swapped["atom_categorical"], batch["atom_categorical"])
    blind = make(samples, element_dropout=1.0)
    for mode in (blind.eval(), blind.train()):
        with torch.no_grad():
            assert torch.equal(mode.encode(batch)[0], mode.encode(swapped)[0])
    seeing = make(samples)
    assert not torch.equal(_encode(seeing, batch)[0], _encode(seeing, swapped)[0])


def test_balance_weights():
    keys = ["a"] * 8 + ["b"] * 2 + [None, None]
    assert balance_weights(keys, 0.0) is None
    w = balance_weights(keys, 1.0)
    assert w.sum() == pytest.approx(1.0)
    mass = Counter()
    for k, x in zip(keys[:10], w[:10], strict=True):
        mass[k] += x
    assert mass["a"] == pytest.approx(mass["b"]) == pytest.approx(w[10]) == pytest.approx(w[11])
    half = balance_weights(keys, 0.5)
    assert half[0] < half[8] < half[10] and half[0] / half[8] == pytest.approx(0.5)


def test_a_weighted_epoch_draws_with_replacement_by_the_weights():
    n = 400
    sizes = np.full(n, 50)
    w = balance_weights(["common"] * 360 + [f"rare{k}" for k in range(40)], 1.0)
    sampler = CellBudgetSampler(sizes, 200_000, max_batch=16, seed=3, weights=w)
    first = [i for b in sampler.batches() for i in b]
    assert len(first) == n and sampler.batches() == sampler.batches()
    rare = sum(i >= 360 for i in first) / n
    assert 0.9 < rare <= 1.0  # 40 keys of 41: nearly every draw is a rare one
    sampler.set_epoch(1)
    assert [i for b in sampler.batches() for i in b] != first
    plain = CellBudgetSampler(sizes, 200_000, max_batch=16, seed=3)
    assert sorted(i for b in plain.batches() for i in b) == list(range(n))  # as before
    fixed = CellBudgetSampler(sizes, 200_000, 16, shuffle=False, weights=w).batches()
    assert sorted(i for b in fixed for i in b) == list(range(n))  # evaluation: every sample
    with pytest.raises(ValueError, match="weights"):
        CellBudgetSampler(sizes, 200_000, weights=w[:3])


def test_the_training_loader_balances_by_edit_signature(samples):  # noqa: F811
    many = [samples[0]] * 30 + list(samples[1:3])
    cfg = TrainConfig(workers=0, cell_budget=10**7, balance_power=1.0)
    loader = make_loader(many, cfg, True, balance=True)
    w = loader.batch_sampler.weights
    assert w is not None and w[0] < w[-1]
    assert make_loader(many, cfg, True).batch_sampler.weights is None  # validation, probes
    off = replace(cfg, balance_power=0.0)
    assert make_loader(many, off, True, balance=True).batch_sampler.weights is None


def test_a_probe_part_is_scored_every_epoch_and_never_selected_on(samples, tmp_path):  # noqa: F811
    _two_source_stores(samples, tmp_path, sources=("pmechdb", "reactmech", "rmechdb"))
    cfg = TrainConfig(
        source="pmechdb+reactmech", split="x", epochs=2, workers=0, cell_budget=10**7, warmup=1,
        probe="rmechdb=x", balance_power=0.5,
    )  # fmt: skip
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_cfg = replace(TINY, fragment_embedding=False, element_dropout=0.3)
    best = fit(cfg, model_cfg, run_dir, torch.device("cpu"), tmp_path)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    n = len(samples)
    for m in lines:
        assert m["probe_rmechdb_n"] == n and m["val_n"] == 2 * n
        assert {"probe_rmechdb_top1", "probe_rmechdb_anchor_nll"} <= set(m)
        assert set(m["skipped"]) == {"train", "val", "probe_rmechdb"}
    assert best["val_top1"] == max(m["val_top1"] for m in lines)
    header = json.loads((run_dir / "run.json").read_text())["provenance"]
    assert any("rmechdb/samples_x_val" in k for k in header["inputs"])
    assert not any("rmechdb/samples_x_train" in k for k in header["inputs"])
    assert header["config"]["train"]["probe"] == "rmechdb=x"
    assert header["config"]["model"]["fragment_embedding"] is False
    with pytest.raises(ValueError, match="sources of the run"):
        load_probes(replace(cfg, probe="pmechdb=x"), tmp_path)
    assert load_probes(replace(cfg, probe=""), tmp_path) == ({}, {})
