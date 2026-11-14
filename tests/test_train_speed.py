"""The training-speed changes (docs/model_impl.md): each one keeps the numbers it promises,
bit for bit where it is a rearrangement and to rounding where the summation order changes."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from torch import nn

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
from ksmech.model.encoder import KINDS, EncoderLayer
from ksmech.model.inputs import fit_feature_stats, frozen, pair_init_parts, select_columns
from ksmech.model.loss import admissible_rows, step_loss
from ksmech.model.net import KSMech
from ksmech.model.pair_linear import PairLinear, weight_grad
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


def test_split_pair_init_kind_table_by_embedding(samples):
    b = collate_samples(samples)
    split = make(samples, pair_cells="all", split_pair_init=True)
    base = make(samples, pair_cells="all")
    split.load_state_dict(base.state_dict())
    split.train()
    base.train()
    with frozen("cpu"):
        inp, z_hat, defined, _ = split.prepare(b)
        parts = pair_init_parts(b, z_hat, defined, TINY.rbf, TINY.rbf_max)
    w = split.encoder.pair_init[0].weight
    c = TINY.rbf + parts.z.shape[-1] + 1
    kinds = torch.cat([w[:, c + 5 : c + 5 + KINDS].T, torch.zeros_like(w[:, :1].T)])
    table = (kinds[:, None, :] + torch.stack([0 * w[:, c], w[:, c]])[None]).reshape(-1, w.shape[0])
    assert torch.equal(table[parts.kind], nn.functional.embedding(parts.kind, table))
    grads = []
    for m in (split, base):
        h, p, y, _ = m.encode(b)
        step_loss(m.decoder, h, p, y, b, 0.0)[0].backward()
        grads.append(dict(m.encoder.pair_init.named_parameters()))
    for name, gs in grads[0].items():
        assert torch.allclose(gs.grad, grads[1][name].grad, rtol=1e-4, atol=1e-6), name


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


def _evaluate_reference(model, loader, device, dtype, dense_warmup=False, lambda_stop=0.0):
    """`train.loop.evaluate` as it was before the device accumulator: host sums, one read per
    quantity per batch."""
    import ksmech.train.loop as loop
    from ksmech.model.inputs import to_device
    from ksmech.model.loss import anchor_ranks, edit_matches, greedy_edit

    model.eval()
    sums = {"loss": 0.0, "anchor_nll": 0.0, "top1": 0, "top3": 0, "top5": 0, "r_exact": 0}
    sums["p_stop"], n = 0.0, 0
    opened = dict.fromkeys(loop.OPEN_METRICS, 0)
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(
                device.type, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                h, p, y, _ = model.encode(batch, dense_warmup)
                _, parts = step_loss(model.decoder, h, p, y, batch, lambda_stop)
                lp = parts["lp_anchor"]
                rank = anchor_ranks(lp, batch)
                exact = edit_matches(*greedy_edit(model.decoder, h, p, lp, batch), batch)
            sums["loss"] += float(parts["loss"].sum())
            sums["anchor_nll"] += float(parts["anchor_nll"].sum())
            sums["p_stop"] += float(parts["p_stop"].sum())
            for k in (1, 3, 5):
                sums[f"top{k}"] += int((rank < k).sum())
            sums["r_exact"] += int(exact.sum())
            n += len(rank)
            pools = (batch["nonbonding"] % 2 != 0).any(1)
            opened["open_top1"] += int((rank[pools] < 1).sum())
            opened["open_r_exact"] += int(exact[pools].sum())
            opened["open_n"] += int(pools.sum())
    m = opened.pop("open_n")
    rates = {k: v / max(m, 1) for k, v in opened.items()}
    return {k: v / max(n, 1) for k, v in sums.items()} | {"n": n, "open_n": m} | rates


def test_evaluate_on_the_device_gives_the_host_numbers(samples):
    import ksmech.train.loop as loop

    batches = [collate_samples(samples[:k]) for k in (2, len(samples))]
    model = make(samples, pair_cells="all")
    dev = torch.device("cpu")
    new = loop.evaluate(model, batches, dev, None)
    old = _evaluate_reference(model, batches, dev, None)
    assert list(new.items()) == list(old.items())  # values and key order, exactly
    assert loop.evaluate(model, [], dev, None)["n"] == 0  # an empty loader as before


def test_make_loader_pins_only_with_a_gpu(samples, monkeypatch):
    import ksmech.train.loop as loop

    on = TrainConfig(workers=0, pin_memory=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert loop.make_loader(samples, on, False).pin_memory is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert loop.make_loader(samples, on, False).pin_memory is True
    assert loop.make_loader(samples, TrainConfig(workers=0), False).pin_memory is False  # default


def test_eval_cell_budget_default_keeps_the_training_budget(samples):
    import ksmech.train.loop as loop

    def budget(**kw):
        cfg = TrainConfig(workers=0, cell_budget=123_456, **kw)
        eval_cfg = replace(cfg, cell_budget=cfg.eval_cell_budget or cfg.cell_budget)
        return loop.make_loader(samples, eval_cfg, False).batch_sampler.cell_budget

    assert budget() == 123_456
    assert budget(eval_cell_budget=999_999) == 999_999


def test_a_header_without_the_loop_fields_compares_equal_to_the_defaults():
    from dataclasses import asdict

    from ksmech.data import provenance as prov
    from ksmech.train.loop import comparable_config, train_config

    train = TrainConfig(seed=1)
    now = prov.normalized({"train": train_config(train), "model": TINY.to_dict()})
    gone = ("pin_memory", "eval_cell_budget")
    old_train = {k: v for k, v in asdict(train).items() if k not in gone}
    old = prov.normalized({"train": old_train, "model": TINY.to_dict()})
    assert old != now and comparable_config(old) == comparable_config(now)


def test_the_checkpoint_is_serialized_once_for_last_and_best(samples, tmp_path, monkeypatch):
    import ksmech.train.loop as loop
    from ksmech.data import provenance as prov

    model = make(samples)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    header = prov.stamp("train_run", loop.TRAIN_VERSION, {}, {"train": {}})
    metrics = {"epoch": 0, "val_top1": 0.5, "val_top3": 0.6, "val_top5": 0.7,
               "val_r_exact": 0.4, "val_loss": 1.0, "train_loss": 1.0}  # fmt: skip
    state = {"step": 1, "epochs": 1, "best": None}
    calls = []
    real = torch.save
    monkeypatch.setattr(loop.torch, "save", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    loop._end_epoch(tmp_path, model, header, opt, sched, scaler, state, metrics)
    assert len(calls) == 1
    assert (tmp_path / "last.pt").read_bytes() == (tmp_path / "best.pt").read_bytes()


def _pair(samples, **kw):
    """(batch, flag-off model, flag-on model with the same weights), both in train mode."""
    b = collate_samples(samples)
    base = make(samples, pair_cells="all", **kw)
    flagged = make(samples, pair_cells="all", last_update_atoms=True, **kw)
    flagged.load_state_dict(base.state_dict())
    return b, base.train(), flagged.train()


def _backward(m, b):
    h, p, y, _ = m.encode(b)
    step_loss(m.decoder, h, p, y, b, 0.0)[0].backward()
    return h, p, {k: v.grad for k, v in m.named_parameters()}


def _same_grads(g0, g1):
    assert g0.keys() == g1.keys()
    for k in g0:
        assert (g0[k] is None) == (g1[k] is None), k
        if g0[k] is not None:
            assert torch.allclose(g0[k], g1[k], rtol=1e-4, atol=1e-6), k


def test_last_update_atoms_is_the_same_map(samples):
    b, base, flagged = _pair(samples, layers=3)
    h0, p0, g0 = _backward(base, b)
    h1, p1, g1 = _backward(flagged, b)
    assert torch.equal(h0, h1)
    assert p0.shape == p1.shape and torch.allclose(p0, p1, rtol=1e-5, atol=1e-6)
    _same_grads(g0, g1)


def test_last_update_atoms_with_pair_update_every(samples):
    b = collate_samples(samples)
    for every, no_update in ((3, True), (2, False)):
        kw = {"layers": 4, "pair_update_every": every, "pair_cells": "all"}
        base, flagged = make(samples, **kw), make(samples, last_update_atoms=True, **kw)
        assert flagged.encoder.layers[-1].update is not no_update
        with torch.no_grad():
            p0, p1 = base.encode(b)[1], flagged.encode(b)[1]
        assert torch.equal(p0, p1) if no_update else torch.allclose(p0, p1, rtol=1e-5, atol=1e-6)


def test_last_update_atoms_through_recompute(samples):
    b, _, flagged = _pair(samples, layers=3, split_pair_init=True)
    _, p1, g1 = _backward(flagged, b)
    kw = {"pair_cells": "all", "layers": 3, "split_pair_init": True, "last_update_atoms": True}
    ck = make(samples, checkpoint=True, **kw)
    ck.load_state_dict(flagged.state_dict())
    ck.train()
    _, p2, g2 = _backward(ck, b)
    assert torch.allclose(p1, p2, rtol=1e-5, atol=1e-6)
    _same_grads(g1, g2)


def test_pair_linear_gradcheck():
    torch.manual_seed(0)
    kw = {"dtype": torch.float64, "requires_grad": True}
    x, w, b = torch.randn(3, 5, 7, 6, **kw), torch.randn(4, 6, **kw), torch.randn(4, **kw)
    assert torch.autograd.gradcheck(PairLinear.apply, (x, w, b))  # R = 105: bmm part + remainder
    assert torch.autograd.gradcheck(lambda x, w: PairLinear.apply(x, w, None), (x, w))
    for r, s in ((50, 7), (5, 7), (64, 32)):
        g2, x2 = torch.randn(r, 4, dtype=torch.float64), torch.randn(r, 6, dtype=torch.float64)
        assert torch.allclose(weight_grad(g2, x2, s), g2.t() @ x2)


@pytest.mark.parametrize("split", [False, True])
def test_pair_linear_splitk_same_forward_close_gradients(samples, split):
    b = collate_samples(samples)
    base = make(samples, pair_cells="all", split_pair_init=split).train()
    on = make(samples, pair_cells="all", split_pair_init=split, pair_linear_splitk=True).train()
    on.load_state_dict(base.state_dict())
    with torch.no_grad():
        (h0, p0, _, _), (h1, p1, _, _) = base.encode(b), on.encode(b)
        assert torch.equal(h0, h1) and torch.equal(p0, p1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ref = base.encode(b)[1]
            if not torch.equal(ref, base.encode(b)[1]):
                pytest.skip("bf16 CPU F.linear is not deterministic between calls")
            assert torch.equal(ref, on.encode(b)[1])
    grads = []
    for model in (base, on):
        model.zero_grad()
        h, p, y, _ = model.encode(b)
        step_loss(model.decoder, h, p, y, b, 0.0)[0].backward()
        grads.append({k: v.grad for k, v in model.named_parameters() if v.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for k in grads[0]:
        assert torch.allclose(grads[0][k], grads[1][k], rtol=1e-5, atol=1e-6), k


def _layer_run(layer, n, device, b=2, dp=16, d=32):
    """One forward + backward of a layer through Recompute (activation checkpointing)."""
    from ksmech.model.encoder import Recompute

    h = torch.randn(b, n, d, device=device, requires_grad=True)
    p = torch.randn(b, n, n, dp, device=device, requires_grad=True)
    keys = torch.ones(b, n, dtype=torch.bool, device=device)
    dt = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=dt):
        h2, p2 = Recompute(True)(layer, h, p, keys)
    (h2.float().square().sum() + p2.float().square().sum()).backward()
    return h.grad, p.grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compiled_checkpointed_pair_linear_survives_a_second_batch_shape(device):
    """compile (dynamic) + checkpointing + pair_linear_splitk: the recompute after the first
    recompile must see the saved tensors of the forward (a CheckpointError with an inlined
    autograd.Function, on the GPU)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")
    cfg = ModelConfig(d=32, d_pair=16, heads=4, pair_linear_splitk=True)
    torch.manual_seed(0)
    eager = EncoderLayer(cfg).to(device)
    layer = EncoderLayer(cfg).to(device)
    layer.load_state_dict(eager.state_dict())
    layer.compile(dynamic=True)
    if torch.cuda.is_available():
        torch.cuda.init()  # checkpoint refuses a device state first initialized inside it
    try:
        for n in (9, 13, 11, 9):
            torch.manual_seed(n)
            ref = _layer_run(eager, n, device)
            torch.manual_seed(n)
            got = _layer_run(layer, n, device)
            for a, c in zip(ref, got, strict=True):
                assert torch.allclose(a, c, rtol=5e-2, atol=5e-2)
    except torch._dynamo.exc.BackendCompilerFailed as e:  # no C++ compiler on this host
        pytest.skip(f"inductor unavailable: {str(e)[:80]}")
    finally:
        torch._dynamo.reset()
