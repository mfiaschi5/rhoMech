"""The phase-3 pretraining (`train.pretrain`) on a store of fixture molecules: the geometry
mask, the target, the validation sums, a CPU smoke run, and `fit --init-from` reading its
checkpoint (`train.loop.initial_model`)."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.model.config import ModelConfig
from ksmech.pretrain import samples as ps
from ksmech.train import pretrain as tp
from ksmech.train.loop import TrainConfig, initial_model, load_model
from tests.fixtures.molecules import MOLECULES

KEYS = [MOLECULES[n] for n in ("acetone", "methyl_acetate", "acrolein", "chloromethane")]
TINY = ModelConfig(d=16, d_pair=8, layers=1, heads=2, decoder_hidden=16, rbf=4,
                   pair_cells="all", fragment_embedding=False, checkpoint=False)  # fmt: skip


@pytest.fixture
def store(pretrain_cache, tmp_path, monkeypatch):
    """A pretraining store of the four molecules, acetone in the validation part."""
    monkeypatch.setattr(ps, "part_of", lambda key: "val" if key == KEYS[0] else "train")
    species = tmp_path / "species.json"
    species.write_text(json.dumps({"species": KEYS}))
    ps.build_store(species, tmp_path / "store", pretrain_cache)
    return tmp_path / "store"


def _batch(store):
    return tp.collate_pretrain([ps.store.dense(s) for s in ps.load_part(store, "train")])


def test_collate_carries_the_target(store):
    batch = _batch(store)
    y, on = tp.target_log(batch, 1e-4)
    assert batch["pair_target"].shape == batch["pair_valid"].shape
    rows = batch["pair_row_pos"] >= 0
    assert not on[~rows].any(), "padded rows are never targets"
    e2 = batch["pair_target"][on]
    torch.testing.assert_close(y[on], torch.log10(e2 + 1e-4))


def test_geometry_mask(store):
    batch = _batch(store)
    names = batch["pair_features"]
    geo = [k for k, n in enumerate(names) if n in tp.GEOMETRY_FEATURES]
    rest = [k for k in range(len(names)) if k not in geo]
    before = batch["pair_defined"].clone()
    assert before[..., geo].any(), "the store has pose channels to mask"
    chosen = tp.mask_geometry(batch, 1.0, torch.Generator().manual_seed(0))
    assert chosen.all() and not batch["pair_defined"][..., geo].any()
    assert (batch["pair_z"][..., geo] == 0).all()
    assert torch.equal(batch["pair_defined"][..., rest], before[..., rest])
    again = _batch(store)
    assert not tp.mask_geometry(again, 0.0, torch.Generator()).any()
    assert torch.equal(again["pair_defined"], before)


def test_eval_sums_rank_the_strongest_cell():
    y = torch.tensor([[[0.0, -2.0], [-3.0, -4.0]], [[-1.0, 0.5], [-2.0, -9.0]]])
    on = torch.ones_like(y, dtype=torch.bool)
    on[1, 1, 1] = False
    pred = torch.tensor([[[1.0, 0.0], [-1.0, -2.0]], [[2.0, 0.0], [-2.0, 5.0]]])
    m = tp.eval_metrics(tp.eval_sums(torch.tensor(0.5), pred, y, on), "val")
    assert m["val_molecules"] == 2
    assert m["val_top1"] == 0.5 and m["val_top5"] == 1.0  # molecule 2's best cell ranks 2nd
    assert m["val_loss"] == pytest.approx(0.5)


def _cfg(tmp_path) -> TrainConfig:
    return TrainConfig(epochs=2, warmup=1, workers=0, amp="off", log_every=1, seed=0)


def test_smoke_run_and_init_from(store, tmp_path):
    pcfg = tp.PretrainConfig(store=str(store))
    run = tmp_path / "run"
    run.mkdir()
    best = tp.fit_pretrain(_cfg(tmp_path), pcfg, TINY, run, torch.device("cpu"))
    lines = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and best["val_molecules"] == 1
    for key in ("train_loss", "val_loss", "val_r", "val_masked_loss", "val_top1"):
        assert np.isfinite(lines[-1][key]), key
    model, ckpt = load_model(run / "best.pt")
    assert "pretrain_head" in ckpt and ckpt["provenance"]["kind"] == "pretrain_run"
    cfg = replace(_cfg(tmp_path), init_from=str(run / "best.pt"))
    ft = initial_model(cfg, TINY, [], model.pair_stats)
    for (k, a), (_, b) in zip(ft.encoder.state_dict().items(), model.encoder.state_dict().items(),
                              strict=True):  # fmt: skip
        assert torch.equal(a, b), k
    assert ft.feature_stats.to_dict() == model.feature_stats.to_dict()


def test_init_from_refuses_another_checkpoint_kind(store, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    tp.fit_pretrain(replace(_cfg(tmp_path), epochs=1), tp.PretrainConfig(store=str(store)),
                    TINY, run, torch.device("cpu"))  # fmt: skip
    ckpt = torch.load(run / "best.pt", weights_only=False)
    ckpt["provenance"]["kind"] = "train_run"
    torch.save(ckpt, tmp_path / "other.pt")
    model, _ = load_model(run / "best.pt")
    with pytest.raises(ValueError, match="not pretrain_run"):
        initial_model(replace(_cfg(tmp_path), init_from=str(tmp_path / "other.pt")), TINY, [],
                      model.pair_stats)  # fmt: skip
