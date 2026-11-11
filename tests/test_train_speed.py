"""The training-speed changes (docs/model_impl.md): each one keeps the numbers it promises,
bit for bit where it is a rearrangement and to rounding where the summation order changes."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from ksmech.arrows import store
from ksmech.arrows.batches import collate_samples
from ksmech.arrows.steps import (
    SAMPLE_VERSION,
    MechanismSample,
    load_samples,
    pair_stats,
    sample_for_step,
    write_split_file,
)
from ksmech.model.config import ModelConfig
from ksmech.model.encoder import EncoderLayer
from ksmech.model.inputs import fit_feature_stats, select_columns
from ksmech.model.loss import admissible_rows, step_loss
from ksmech.model.net import KSMech
from ksmech.train.batching import CellBudgetSampler, DrawnSamples, cells
from ksmech.train.loop import NonFiniteError, TrainConfig, compile_encoder, train_epoch

TINY = ModelConfig(
    d=32, d_pair=16, layers=2, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)
HOST = ("pair_cells", "anchor_rows", "anchor_unique", "anchors_missing", "anchor_atom_values")


@pytest.fixture(scope="module")
def samples(mini_steps, token_loader) -> list[MechanismSample]:
    out = [sample_for_step(mini_steps[k], token_loader) for k in mini_steps]
    out = [s for s in out if isinstance(s, MechanismSample)]
    assert len(out) >= 3
    return out


def make(samples, **kw) -> KSMech:
    torch.manual_seed(0)
    cfg = replace(TINY, **kw)
    return KSMech(cfg, fit_feature_stats(samples), pair_stats(samples)).eval()


def _stored(tmp_path, samples: list, writer: bool = False):
    """The samples in a store under tmp_path (sorted, or in list order by `StoreWriter`) and
    one view over them."""
    parquet = tmp_path / "steps.parquet"
    parquet.write_bytes(b"stand-in parquet")
    ids = [s.step_id for s in samples]
    split = write_split_file(tmp_path / "splits_x.json", "x", 0, {"x": sorted(ids)}, parquet)
    if writer:
        w = store.StoreWriter(tmp_path, parquet, SAMPLE_VERSION)
        for s in samples:
            w.add(s)
        index = w.close({})
    else:
        index = store.write_store(
            tmp_path, {s.step_id: s for s in samples}, {}, parquet, SAMPLE_VERSION
        )
    config = {"split": "x", "sample_version": SAMPLE_VERSION}
    return store.write_view(tmp_path / "samples_x.json", sorted(ids), {}, index, split, config)


def _same_arrays(a: MechanismSample, b: MechanismSample) -> None:
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape, f.name
            assert np.array_equal(x, y, equal_nan=True), f.name
        else:
            assert x == y, f.name


def test_compact_samples_expand_bit_for_bit(samples, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SHARD_SIZE", 2)  # several shards, read by several workers
    view = _stored(tmp_path, samples)
    whole = load_samples(view)
    light = load_samples(view, compact=True, workers=2)
    assert [s.step_id for s in light] == [s.step_id for s in whole]
    for a, b in zip(whole, light, strict=True):
        assert store.is_light(b) and b.pair_z.size == 0
        _same_arrays(a, store.dense(b))
        assert b._cells.nbytes() < a.pair_z.nbytes + a.pair_defined.nbytes
    assert store.dense(whole[0]) is whole[0]
    stripped = store.light(whole[0])
    with pytest.raises(ValueError, match="without its cells"):
        store.dense(stripped)


def test_the_stream_writer_stores_the_same_samples(samples, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SHARD_SIZE", 2)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    sorted_view = _stored(tmp_path / "a", samples)
    stream_view = _stored(tmp_path / "b", samples[::-1], writer=True)
    for a, b in zip(load_samples(sorted_view), load_samples(stream_view), strict=True):
        _same_arrays(a, b)


def test_pair_stats_in_one_pass_over_light_samples(samples, tmp_path):
    view = _stored(tmp_path, samples)
    whole, light = load_samples(view), load_samples(view, compact=True)
    for cells_ in ("valid", "all"):
        a, b = pair_stats(whole, cells=cells_), pair_stats(iter(light), cells=cells_)
        assert np.array_equal(a.mean, b.mean) and np.array_equal(a.std, b.std)
        assert np.array_equal(a.count, b.count) and a.names == b.names


def test_drawn_samples_expand_light_ones(samples, tmp_path):
    view = _stored(tmp_path, samples)
    whole, light = load_samples(view), load_samples(view, compact=True)
    for i in range(len(whole)):
        _same_arrays(DrawnSamples(whole)[i], DrawnSamples(light)[i])


def test_host_indices_are_the_device_rows_and_leave_the_loss_unchanged(samples):
    model = make(samples, pair_cells="all")
    b = collate_samples(samples)
    device = {k: v for k, v in b.items() if k not in HOST}
    assert torch.equal(admissible_rows(b), admissible_rows(device))
    assert b["anchors_missing"] == 0
    with torch.no_grad():
        outs = []
        for batch in (b, device):
            h, p, y, _ = model.encode(batch)
            outs.append(step_loss(model.decoder, h, p, y, batch, 0.0)[1]["loss"])
    assert torch.equal(*outs)


def test_select_columns_by_a_cached_device_index(samples):
    b = collate_samples(samples)
    cols = tuple(b["orbital_columns"])
    names = cols[::-2]
    want = b["orbital_features"][..., [cols.index(n) for n in names]]
    assert torch.equal(select_columns(b["orbital_features"], cols, names), want)


def test_split_pair_init_is_the_same_map(samples):
    b = collate_samples(samples)
    base = make(samples, pair_cells="all")
    split = make(samples, pair_cells="all", split_pair_init=True)
    split.load_state_dict(base.state_dict())
    with torch.no_grad():
        (h0, p0, y0, _), (h1, p1, y1, _) = base.encode(b), split.encode(b)
    assert torch.allclose(p0, p1, rtol=1e-4, atol=1e-5) and torch.allclose(h0, h1, atol=1e-4)
    assert torch.equal(y0, y1)


def test_half_pair_norm_changes_nothing_outside_autocast(samples):
    b = collate_samples(samples)
    base, half = make(samples), make(samples, half_pair_norm=True)
    half.load_state_dict(base.state_dict())
    with torch.no_grad():
        assert torch.equal(base.encode(b)[1], half.encode(b)[1])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            p0, p1 = base.encode(b)[1].float(), half.encode(b)[1].float()
    assert torch.allclose(p0, p1, rtol=0.05, atol=0.05)


def test_pair_update_every_second_layer(samples):
    model = make(samples, pair_update_every=2, layers=4)
    updates = [layer.update for layer in model.encoder.layers]
    assert updates == [False, True, False, True]
    assert not hasattr(model.encoder.layers[0], "u_out")
    with torch.no_grad():
        p = model.encode(collate_samples(samples))[1]
    assert torch.isfinite(p).all()
    with pytest.raises(ValueError, match="at least 1"):
        make(samples, pair_update_every=0)
    plain = EncoderLayer(TINY)
    assert plain.update and hasattr(plain, "u_out")


def test_cell_budget_counts_atoms_and_orbitals_apart():
    rng = np.random.default_rng(0)
    sizes = np.stack([rng.integers(5, 80, 400), rng.integers(10, 220, 400)], 1)
    sampler = CellBudgetSampler(sizes, cell_budget=200_000, max_batch=16, seed=3)
    batches = sampler.batches()
    assert sorted(i for batch in batches for i in batch) == list(range(400))
    for batch in batches:
        padded = sizes[batch, 0].max() + sizes[batch, 1].max()
        assert len(batch) <= 16 and len(batch) * cells(int(padded)) <= 200_000
    fixed = CellBudgetSampler(sizes, 200_000, 16, shuffle=False).batches()
    order = [i for batch in fixed for i in batch]
    assert order == list(np.lexsort((sizes[:, 0], sizes[:, 1])))  # by orbitals, then atoms


def test_a_fused_step_skips_a_non_finite_batch_on_the_device(samples, monkeypatch):
    import ksmech.train.loop as loop

    model = make(samples).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    real = loop.step_loss

    def nan_loss(*args):
        loss, parts = real(*args)
        return loss * math.nan, parts

    monkeypatch.setattr(loop, "step_loss", nan_loss)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True)
    except RuntimeError as exc:  # a build without the fused CPU kernel
        pytest.skip(f"no fused AdamW here: {exc}")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    cfg = TrainConfig(workers=0, log_every=1)
    state = {"step": 0, "total": 9, "epoch": 0}
    batch = collate_samples(samples)
    out = train_epoch(model, [batch], opt, sched, scaler, torch.device("cpu"), None, cfg, state)
    assert out["nonfinite_batches"] == 1 and state["step"] == 1 and math.isnan(out["train_loss"])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    monkeypatch.setattr(loop, "MAX_NONFINITE_IN_A_ROW", 2)
    with pytest.raises(NonFiniteError, match="2 non-finite training batches in a row"):
        train_epoch(model, [batch] * 2, opt, sched, scaler, torch.device("cpu"), None, cfg, state)


def test_compile_keeps_the_state_dict_keys(samples):
    model = make(samples)
    keys = list(model.state_dict())
    compile_encoder(model)
    assert list(model.state_dict()) == keys


def test_the_streaming_build_gives_the_same_views_samples_and_statistics(
    mini_result, token_loader, tmp_path, monkeypatch
):
    import json

    from ksmech.arrows.steps import build_all_samples, pair_stats_path, samples_path
    from ksmech.data.io import steps_path, write_steps

    monkeypatch.setattr(store, "SHARD_SIZE", 2)  # several shards, written as they fill
    reports = {}
    for name, stream in (("whole", False), ("stream", True)):
        root = tmp_path / name
        write_steps(list(mini_result.steps), steps_path("pmechdb", root))
        reports[name] = build_all_samples(
            "pmechdb", ["molecule"], root, token_loader, stream=stream
        )
    assert reports["whole"] == reports["stream"]
    for split in reports["whole"]["splits"]:
        views = [samples_path("pmechdb", split, tmp_path / n) for n in ("whole", "stream")]
        a, b = (load_samples(v) for v in views)
        assert [s.step_id for s in a] == [s.step_id for s in b]
        for x, y in zip(a, b, strict=True):
            _same_arrays(x, y)
    for cells_ in ("valid", "all"):
        files = [pair_stats_path("pmechdb", "molecule", tmp_path / n, cells_) for n in reports]
        a, b = (json.loads(f.read_text())["features"] for f in files)
        assert a.keys() == b.keys()
        for k in a:
            assert np.allclose(a[k]["mean"], b[k]["mean"], rtol=1e-12, atol=1e-15), k
            assert np.allclose(a[k]["std"], b[k]["std"], rtol=1e-12, atol=1e-15), k
