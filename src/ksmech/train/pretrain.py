"""Phase-3 pretraining (docs/decisions.md, 2026-11-22): the dense encoder of `KSMech` and a
pair head that reads the final pair track on every (occupied, virtual) cell of a QMugs molecule
and predicts y = log10(E2 + e2_floor_ev), E2 being the cell's donor-acceptor energy
(`physics.donor_acceptor`). The loss is a Huber loss on the cells where E2 is defined, averaged
over the cells of the batch.

In a share `mask_geometry` of the molecules of each batch the reactant-pose channels
(`GEOMETRY_FEATURES`: centroid distance, cross coupling, the two alignment cosines, g2) are left
undefined, which is how an intermolecular cell of a mechanism pool has them. The pair track
therefore also learns the interaction from the orbital descriptors, the gap factor and the
graph alone.

A run lives in `data/models/qmugs/<run>/`: `run.json`, `train.log` (`step n / total`),
`metrics.jsonl` per epoch, `last.pt` and `best.pt` (lowest validation loss). The checkpoints
are the ones `train.loop.load_model` reads (the KSMech with its statistics), with the head's
state beside it. `ksmech-train fit --init-from` copies the encoder from one of them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ksmech.arrows.batches import _pad2, collate_samples
from ksmech.arrows.steps import SAMPLE_VERSION, pair_stats
from ksmech.data import provenance as prov
from ksmech.model.config import ModelConfig
from ksmech.model.encoder import mlp, orbital_pair_cells
from ksmech.model.inputs import fit_feature_stats, to_device, token_columns
from ksmech.model.net import KSMech
from ksmech.physics.stage_a import OPEN_SHELL_BLOCK
from ksmech.pretrain import samples as ps
from ksmech.train.batching import (
    CellBudgetSampler,
    DrawnSamples,
    PackedSamples,
    sample_sizes,
)
from ksmech.train.loop import (
    TrainConfig,
    _optimizer,
    amp_dtype,
    checkpoint_payload,
    freeze_heap,
    lr_factor,
    save_checkpoint,
)

log = logging.getLogger(__name__)

PRETRAIN_VERSION = 1
MODELS_SOURCE = "qmugs"  # data/models/qmugs/<run>/
GEOMETRY_FEATURES = (
    "centroid_distance",
    "cross_coupling",
    "align_cos",
    "dihedral_cos",
    "g2",
    "log_abs_g2",
)
TOO_LARGE = "TOO_LARGE"


@dataclass
class PretrainConfig:
    store: str = str(ps.OUT_DIR)  # the pretraining store (`ksmech-pretrain samples`)
    mask_geometry: float = 0.5  # share of the molecules of a batch without the pose channels
    e2_floor_ev: float = 1e-4  # y = log10(E2 + floor): cells below ~0.002 kcal/mol are alike
    huber_delta: float = 1.0  # in decades of E2


class PairTargetHead(nn.Module):
    """y of every (occupied, virtual) cell from its pair vector: an MLP d_p -> d_p -> 1."""

    def __init__(self, d_pair: int):
        super().__init__()
        self.mlp = mlp(d_pair, d_pair, 1)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return self.mlp(cells).squeeze(-1)


def collate_pretrain(batch: list) -> dict:
    """`collate_samples` and the target, NaN on the padding, in the pair block's layout."""
    out = collate_samples(batch)
    rows, cols = out["pair_row_pos"].shape[1], out["pair_col_pos"].shape[1]
    out["pair_target"] = _pad2([s.pair_target for s in batch], rows, cols, np.nan)
    return out


def mask_geometry(batch: dict, share: float, generator: torch.Generator) -> torch.Tensor:
    """Leave the pose channels undefined (and 0) in a random `share` of the batch's molecules,
    in place; returns the (B,) mask of the molecules chosen."""
    b = batch["pair_defined"].shape[0]
    chosen = torch.rand(b, generator=generator) < share
    cols = [k for k, n in enumerate(batch["pair_features"]) if n in GEOMETRY_FEATURES]
    if cols and chosen.any():
        rows = chosen.nonzero().squeeze(1)
        index = torch.tensor(cols)
        for name in ("pair_defined", "pair_z"):
            sub = batch[name][rows]
            sub[..., index] = 0
            batch[name][rows] = sub
    return chosen


def target_log(batch: dict, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(y, on): log10(E2 + floor) on the cells, and the cells where E2 is defined and real."""
    e2 = batch["pair_target"]
    on = torch.isfinite(e2)
    on &= (batch["pair_row_pos"] >= 0)[:, :, None] & (batch["pair_col_pos"] >= 0)[:, None, :]
    y = torch.log10(torch.where(on, e2, 0.0).clamp(min=0.0) + floor)
    return y, on


def predict(model: KSMech, head: PairTargetHead, batch: dict) -> torch.Tensor:
    """y predicted on every cell of the pair block (B, R, C), from the whole pair track."""
    inp, z_hat, defined, _ = model.prepare(batch)
    _, p = model.encoder.pair_track(batch, inp, z_hat, defined)
    return head(orbital_pair_cells(p, batch)).float()


def pretrain_loss(model, head, batch: dict, pcfg: PretrainConfig) -> tuple:
    """(the mean Huber loss over the defined cells, the prediction, y, on)."""
    pred = predict(model, head, batch)
    y, on = target_log(batch, pcfg.e2_floor_ev)
    loss = nn.functional.huber_loss(pred[on], y[on], delta=pcfg.huber_delta)
    return loss, pred, y, on


class Pretrainer(nn.Module):
    """The model and its pretraining head, one module for the optimizer and the checkpoint."""

    def __init__(self, model: KSMech, head: PairTargetHead):
        super().__init__()
        self.model, self.head = model, head


def load_parts(cfg: TrainConfig, pcfg: PretrainConfig) -> tuple[list, list, dict]:
    """(train, val, the parts' info for run.json): each part of the store, its samples over
    `max_tokens` left out (TOO_LARGE, counted), the first `limit` only for a smoke run."""
    store, out, info = Path(pcfg.store), {}, {}
    for part in ps.PARTS:
        samples = ps.load_part(store, part, cfg.compact_samples, cfg.load_workers,
                               cfg.deflate_samples)  # fmt: skip
        samples = samples[: cfg.limit] if cfg.limit else samples
        keep = [s for s in samples if s.n_atoms + s.n_orbitals <= cfg.max_tokens]
        info[part] = {
            "path": str(ps.view_path(store, part)),
            "used": len(keep),
            "skipped": {TOO_LARGE: len(samples) - len(keep)},
        }
        out[part] = keep
    return out["train"], out["val"], info


def make_loader(samples, cfg: TrainConfig, shuffle: bool, budget: int = 0):
    sampler = CellBudgetSampler(
        sample_sizes(samples), budget or cfg.cell_budget, cfg.max_batch, shuffle, cfg.seed
    )
    return torch.utils.data.DataLoader(
        DrawnSamples(samples),
        batch_sampler=sampler,
        collate_fn=collate_pretrain,
        num_workers=cfg.workers,
        multiprocessing_context="fork" if cfg.workers else None,
        persistent_workers=bool(cfg.workers),
        pin_memory=bool(cfg.pin_memory) and torch.cuda.is_available(),
    )


def build_net(train: list, model_cfg: ModelConfig) -> Pretrainer:
    """The model with the statistics of the training molecules, and a fresh head."""
    if model_cfg.encoder != "dense":
        raise ValueError(f"encoder {model_cfg.encoder!r}: pretraining reads the dense pair track")
    extra = OPEN_SHELL_BLOCK if model_cfg.open_shell else ()
    stats = pair_stats(train, cells=model_cfg.pair_cells, extra=extra)
    feature_stats = fit_feature_stats(train, *token_columns(model_cfg))
    return Pretrainer(KSMech(model_cfg, feature_stats, stats), PairTargetHead(model_cfg.d_pair))


EVAL_SUMS = ("cells", "loss", "p", "y", "pp", "yy", "py", "molecules", "top1", "top5",
             "strong", "strong_err")  # fmt: skip
STRONG_Y = -1.0  # log10(E2 / eV) above this (E2 > 0.1 eV, ~2.3 kcal/mol): a strong interaction


def eval_sums(loss, pred, y, on) -> torch.Tensor:
    """The sums `EVAL_SUMS` of one batch: the pooled moments of (pred, y) over the defined cells,
    and per molecule whether its strongest cell is the top-1 / within the top-5 predicted."""
    p, t = pred[on].double(), y[on].double()
    n = on.sum()
    flat_p = torch.where(on, pred, -torch.inf).flatten(1)
    best = torch.where(on, y, -torch.inf).flatten(1).argmax(1)
    rank = (flat_p > flat_p.gather(1, best[:, None])).sum(1) + 1
    real = on.flatten(1).any(1)
    strong = on & (y > STRONG_Y)
    err = (pred - y).abs()[strong].double()
    row = [n, loss.double() * n, p.sum(), t.sum(), (p * p).sum(), (t * t).sum(), (p * t).sum(),
           real.sum(), (real & (rank == 1)).sum(), (real & (rank <= 5)).sum(), strong.sum(),
           err.sum()]  # fmt: skip
    return torch.stack([torch.as_tensor(x, dtype=torch.float64, device=pred.device) for x in row])


def eval_metrics(acc: torch.Tensor, prefix: str) -> dict:
    s = dict(zip(EVAL_SUMS, acc.tolist(), strict=True))
    n, m = max(s["cells"], 1.0), max(s["molecules"], 1.0)
    var_p = s["pp"] / n - (s["p"] / n) ** 2
    var_y = s["yy"] / n - (s["y"] / n) ** 2
    cov = s["py"] / n - s["p"] * s["y"] / n**2
    r = cov / max(np.sqrt(max(var_p * var_y, 0.0)), 1e-12)
    return {
        f"{prefix}_loss": s["loss"] / n,
        f"{prefix}_r": r,
        f"{prefix}_top1": s["top1"] / m,
        f"{prefix}_top5": s["top5"] / m,
        f"{prefix}_strong_mae": s["strong_err"] / max(s["strong"], 1.0),
        f"{prefix}_molecules": int(s["molecules"]),
    }


@torch.no_grad()
def evaluate(net: Pretrainer, loader, device, dtype, pcfg: PretrainConfig, masked: bool) -> dict:
    """The validation metrics, every molecule with its pose channels (`masked=False`, prefix
    val) or without them (val_masked: how an intermolecular cell is read)."""
    net.eval()
    gen = torch.Generator().manual_seed(0)
    acc = torch.zeros(len(EVAL_SUMS), dtype=torch.float64, device=device)
    for batch in loader:
        if masked:
            mask_geometry(batch, 1.0, gen)
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            loss, pred, y, on = pretrain_loss(net.model, net.head, batch, pcfg)
        acc += eval_sums(loss.float(), pred, y, on)
    return eval_metrics(acc, "val_masked" if masked else "val")


def _step(net, opt, scaler, loss, cfg: TrainConfig):
    """Backward, unscale, clip, step: the gradient norm."""
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
    scaler.step(opt)
    scaler.update()
    return norm


def train_epoch(net, loader, opt, sched, scaler, device, dtype, cfg, pcfg, state: dict) -> dict:
    """One pass over the training batches, the pose channels masked in a share of each batch;
    a non-finite loss skips its step (counted)."""
    net.train()
    t0, seen, losses = time.time(), 0, []
    for batch in loader:
        mask_geometry(batch, pcfg.mask_geometry, state["generator"])
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            loss, *_ = pretrain_loss(net.model, net.head, batch, pcfg)
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        if not torch.isfinite(loss):
            state["nonfinite"] += 1
            sched.step()
            continue
        norm = _step(net, opt, scaler, loss, cfg)
        sched.step()
        losses.append(float(loss.detach()))
        seen += len(batch["step_id"])
        if state["step"] % cfg.log_every == 0:
            log.info("step %d / %d epoch %d loss %.4f grad %.2f lr %.2e %.1f samples/s",
                     state["step"], state["total"], state["epoch"],
                     np.mean(losses[-cfg.log_every :]), float(norm), sched.get_last_lr()[0],
                     seen / (time.time() - t0))  # fmt: skip
    return {"train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_seconds": time.time() - t0, "nonfinite": state["nonfinite"]}  # fmt: skip


def run_header(cfg: TrainConfig, pcfg: PretrainConfig, model_cfg: ModelConfig, info: dict):
    inputs = {}
    for part in info.values():
        inputs |= prov.file_input(part["path"])
    config = {
        "train": asdict(cfg),
        "model": model_cfg.to_dict(),
        "pretrain": asdict(pcfg),
        "sample_version": SAMPLE_VERSION,
        "pretrain_target": ps.TARGET,
        "target_version": ps.TARGET_VERSION,
    }
    return prov.stamp("pretrain_run", PRETRAIN_VERSION, inputs, config, extra={"parts": info})


def _end_epoch(run_dir: Path, net: Pretrainer, header, pcfg, state: dict, metrics: dict) -> None:
    with open(run_dir / "metrics.jsonl", "a") as fh:
        fh.write(json.dumps(metrics) + "\n")
    payload = checkpoint_payload(net.model, header, pretrain_head=net.head.state_dict(),
                                 pretrain_config=asdict(pcfg), epoch=state["epoch"],
                                 metrics=metrics)  # fmt: skip
    save_checkpoint(run_dir / "last.pt", payload)
    if state["best"] is None or metrics["val_loss"] < state["best"]["val_loss"]:
        state["best"] = metrics
        save_checkpoint(run_dir / "best.pt", payload)
    log.info("epoch %d: %s", state["epoch"], metrics)


def fit_pretrain(cfg: TrainConfig, pcfg: PretrainConfig, model_cfg: ModelConfig, run_dir: Path,
                 device: torch.device) -> dict:  # fmt: skip
    """Pretrain one run; returns the best validation metrics."""
    torch.manual_seed(cfg.seed)
    dtype = amp_dtype(device, cfg.amp)
    train, val, info = load_parts(cfg, pcfg)
    net = build_net(train, model_cfg).to(device)
    if cfg.compile:
        from ksmech.train.loop import compile_encoder

        compile_encoder(net.model, "layers")
    header = run_header(cfg, pcfg, model_cfg, info)
    prov.write_json(run_dir / "run.json", {"device": str(device), "amp": str(dtype)}, header)
    if cfg.pack_samples:
        train, val = PackedSamples.take(train), PackedSamples.take(val)
    freeze_heap()
    loader = make_loader(train, cfg, True)
    val_loader = make_loader(val, cfg, False, cfg.eval_cell_budget)
    total = cfg.epochs * len(loader.batch_sampler)
    opt = _optimizer(net, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg.warmup, total))
    scaler = torch.amp.GradScaler(device.type, init_scale=2.0**10, enabled=dtype == torch.float16)
    gen = torch.Generator().manual_seed(cfg.seed)
    state = {"epoch": 0, "step": 0, "total": total, "best": None, "nonfinite": 0, "generator": gen}
    log.info("%d parameters, %d train / %d val molecules, %d steps on %s (%s)",
             sum(p.numel() for p in net.parameters()), info["train"]["used"],
             info["val"]["used"], total, device, dtype)  # fmt: skip
    for epoch in range(cfg.epochs):
        state["epoch"] = epoch
        loader.batch_sampler.set_epoch(epoch)
        metrics = train_epoch(net, loader, opt, sched, scaler, device, dtype, cfg, pcfg, state)
        metrics |= evaluate(net, val_loader, device, dtype, pcfg, masked=False)
        metrics |= evaluate(net, val_loader, device, dtype, pcfg, masked=True)
        metrics |= {"epoch": epoch, "step": state["step"]}
        _end_epoch(run_dir, net, header, pcfg, state, metrics)
    return state["best"]
