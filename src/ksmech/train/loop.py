"""Training (spec v9, "Training objective"): AdamW with a linear warm-up and a cosine decay,
gradient clipping, mixed precision, batches by a cell budget, and the edit-level metrics on the
validation part after every epoch.

The encoder is `--encoder dense | flex | sparse`. The sparse encoder's indexer KL is added to
the loss (it reaches only the indexer), and it runs in its dense warm-up mode while the
optimizer step is within the learning-rate warm-up.

A run lives in `data/models/<source>/<run>/`: `run.json` (provenance header: the training and
validation views, the pair statistics, both configurations), `train.log` (one timestamped
`step n / total` line per logging interval, the format `ksmech-eta` reads), `metrics.jsonl`
(one line per epoch) and the checkpoints `last.pt` and `best.pt` (best validation anchor
top-1). Samples too large for the cell budget are skipped with the reason TOO_LARGE and
counted in `run.json` and in every metrics line.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.steps import SAMPLE_VERSION, load_pair_stats, load_samples, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.model.config import ModelConfig
from ksmech.model.inputs import FeatureStats, fit_feature_stats, to_device
from ksmech.model.loss import anchor_ranks, edit_matches, greedy_edit, step_loss
from ksmech.model.net import KSMech
from ksmech.physics.pairs import PairStats
from ksmech.train.batching import CellBudgetSampler

log = logging.getLogger(__name__)

TRAIN_VERSION = 1
DEFAULT_MODELS_DIR = Path("data/models")
TOO_LARGE = "TOO_LARGE"


@dataclass(frozen=True)
class TrainConfig:
    source: str = "pmechdb"
    split: str = "molecule"
    epochs: int = 30
    cell_budget: int = 1_100_000  # padded pair-track cells per batch
    max_batch: int = 64
    max_tokens: int = 1024  # larger samples are skipped (TOO_LARGE)
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 2000  # optimizer steps
    grad_clip: float = 1.0
    amp: str = "auto"  # auto | fp16 | bf16 | off
    seed: int = 0
    workers: int = 2  # forked DataLoader workers that collate
    limit: int = 0  # use only the first n training and validation samples (smoke runs)
    log_every: int = 20  # optimizer steps
    stop_weight: float = -1.0  # lambda_stop; < 0: the source's (0.5, 0 on PMechDB)


STOP_WEIGHT, SOURCE_STOP_WEIGHT = 0.5, {"pmechdb": 0.0}  # PMechDB: isolated steps, no stop labels


def stop_weight(cfg: TrainConfig) -> float:
    """lambda_stop of the run: the configured value, else the source's default."""
    if cfg.stop_weight >= 0:
        return cfg.stop_weight
    return SOURCE_STOP_WEIGHT.get(cfg.source, STOP_WEIGHT)


KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")
RDNA2_WITHOUT_KERNELS = range(100301, 100307)  # gfx1031..gfx1036: the build ships gfx1030


def rocm_override(nodes: Path = KFD_NODES) -> str | None:
    """Set HSA_OVERRIDE_GFX_VERSION=10.3.0 when the ROCm build meets an RDNA2 card it has no
    kernels for (the RX 6600 XT is gfx1032) and the variable is unset; the value set, or None.
    Without it the first kernel segfaults, which no exception handler can catch. Must run
    before the first CUDA call: the HSA runtime reads the variable when it starts."""
    if torch.version.hip is None or "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        return None
    for props in sorted(nodes.glob("*/properties")):
        fields = dict(line.split(maxsplit=1) for line in props.read_text().splitlines() if line)
        if int(fields.get("gfx_target_version", 0)) in RDNA2_WITHOUT_KERNELS:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
            return "10.3.0"
    return None


def select_device(name: str = "auto") -> torch.device:
    """`cuda` (an NVIDIA card, or an AMD card through ROCm) when one works, else `cpu`. A ROCm
    device is checked with one small product: a card whose kernels the build lacks fails here
    rather than mid-run."""
    if name == "cpu" or (name == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    device = torch.device("cuda" if name == "auto" else name)
    try:
        x = torch.ones(8, 8, device=device)
        float((x @ x).sum())
    except RuntimeError as exc:
        hint = " (an RX 6600 needs HSA_OVERRIDE_GFX_VERSION=10.3.0)" if torch.version.hip else ""
        raise RuntimeError(f"{device} does not run{hint}: {exc}") from exc
    return device


def amp_dtype(device: torch.device, amp: str) -> torch.dtype | None:
    """The autocast dtype: none on the CPU; on ROCm float16 (RDNA2 has no fast bfloat16);
    on CUDA bfloat16 where supported."""
    if amp == "off" or (amp == "auto" and device.type == "cpu"):
        return None
    if amp == "auto":
        bf16 = torch.version.hip is None and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16 else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _size(sample) -> int:
    return sample.n_atoms + sample.n_orbitals


def load_part(cfg: TrainConfig, part: str, processed_dir: Path) -> tuple[list, dict]:
    """The samples of one part that fit the token cap, and the skip counts."""
    path = samples_path(cfg.source, f"{cfg.split}_{part}", processed_dir)
    samples = load_samples(path)
    if cfg.limit:
        samples = samples[: cfg.limit]
    keep = [s for s in samples if _size(s) + 1 <= cfg.max_tokens]
    skipped = {TOO_LARGE: len(samples) - len(keep)} if len(keep) < len(samples) else {}
    for s in samples:
        if _size(s) + 1 > cfg.max_tokens:
            log.warning("%s %s skipped: %s (%d tokens)", part, s.step_id, TOO_LARGE, _size(s))
    return keep, {"n": len(samples), "used": len(keep), "skipped": skipped, "path": str(path)}


def make_loader(samples: list, cfg: TrainConfig, shuffle: bool) -> torch.utils.data.DataLoader:
    sampler = CellBudgetSampler(
        [_size(s) for s in samples], cfg.cell_budget, cfg.max_batch, shuffle, cfg.seed
    )
    return torch.utils.data.DataLoader(
        samples,
        batch_sampler=sampler,
        collate_fn=collate_samples,
        num_workers=cfg.workers,
        multiprocessing_context="fork" if cfg.workers else None,
        persistent_workers=bool(cfg.workers),
    )


@torch.no_grad()
def evaluate(
    model: KSMech, loader, device, dtype, dense_warmup: bool = False, lambda_stop: float = 0.0
) -> dict:
    """Mean loss and anchor NLL; anchor top-1/3/5 with set marginalization; the greedy R
    (top anchor, most probable entries) matching a variant of the label class; the mean
    P(END_MECH)."""
    model.eval()
    sums = {"loss": 0.0, "anchor_nll": 0.0, "top1": 0, "top3": 0, "top5": 0, "r_exact": 0}
    sums["p_stop"], n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
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
    return {k: v / max(n, 1) for k, v in sums.items()} | {"n": n}


def train_epoch(model, loader, opt, sched, scaler, device, dtype, cfg: TrainConfig, state: dict):
    """One pass over the training batches; `state` carries the optimizer step and its total."""
    model.train()
    seen, t0, losses = 0, time.time(), []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            h, p, y, aux = model.encode(batch, state["step"] < cfg.warmup)
            loss, parts = step_loss(model.decoder, h, p, y, batch, stop_weight(cfg))
            loss = loss + sum(aux.values())  # the sparse indexer's KL trains the indexer only
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        state["step"] += 1
        seen += len(batch["step_id"])
        losses.append(float(loss.detach()))
        if state["step"] % cfg.log_every == 0:
            log.info(
                "step %d / %d epoch %d loss %.4f anchor_nll %.4f grad %.2f lr %.2e %.1f samples/s",
                state["step"],
                state["total"],
                state["epoch"],
                float(np.mean(losses[-cfg.log_every :])),
                float(parts["anchor_nll"].mean()),
                float(norm),
                sched.get_last_lr()[0],
                seen / (time.time() - t0),
            )
    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "seconds": time.time() - t0,
    }


def checkpoint_payload(model: KSMech, header: prov.Provenance, **extra) -> dict:
    return {
        "provenance": asdict(header),
        "model_config": model.cfg.to_dict(),
        "feature_stats": model.feature_stats.to_dict(),
        "pair_stats": model.pair_stats.to_dict(),
        "state_dict": model.state_dict(),
        **extra,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    buf = io.BytesIO()
    torch.save(payload, buf)
    prov.atomic_write(path, buf.getvalue())


def load_model(path: Path, device: torch.device | str = "cpu") -> tuple[KSMech, dict]:
    """The model of a checkpoint, with the checkpoint dict (its provenance header and state).
    A checkpoint trained from a dirty tree is refused, as every evaluation input is."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prov.require_clean(prov.Provenance.from_dict(ckpt["provenance"]), path)
    names = tuple(ckpt["pair_stats"])
    model = KSMech(
        ModelConfig.from_dict(ckpt["model_config"]),
        FeatureStats.from_dict(ckpt["feature_stats"]),
        PairStats.from_dict(ckpt["pair_stats"], names),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


def _optimizer(model: KSMech, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW; weight decay on the matrices only (not on biases or norm scales)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": rest, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)


def _run_header(cfg: TrainConfig, model_cfg: ModelConfig, parts: dict, stats_path: Path):
    inputs = {}
    for info in parts.values():
        inputs |= prov.file_input(info["path"])
    inputs |= prov.file_input(stats_path)
    config = {"train": asdict(cfg), "model": model_cfg.to_dict(), "sample_version": SAMPLE_VERSION}
    return prov.stamp("train_run", TRAIN_VERSION, inputs, config, extra={"parts": parts})


def _resume(path: Path, header, model, opt, sched, scaler) -> dict:
    """Restore a run from its `last.pt`; refused when the run's inputs or configuration
    differ from the checkpoint's."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old = prov.Provenance.from_dict(ckpt["provenance"])
    if old.inputs != header.inputs or old.config != header.config:
        raise prov.StaleArtifact(path, ["the run's inputs or configuration changed"])
    model.load_state_dict(ckpt["state_dict"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    log.info("resumed %s at epoch %d, step %d", path, ckpt["epoch"], ckpt["step"])
    return {"epoch": ckpt["epoch"] + 1, "step": ckpt["step"], "best": ckpt["best"]}


def _end_epoch(run_dir: Path, model, header, opt, sched, scaler, state: dict, metrics: dict):
    """Append the metrics line, log it, and write `last.pt` (and `best.pt` when the validation
    anchor top-1 improves, ties broken by the loss)."""
    with open(run_dir / "metrics.jsonl", "a") as fh:
        fh.write(json.dumps(metrics) + "\n")
    log.info(
        "epoch %d of %d done: val top1 %.3f top5 %.3f r_exact %.3f loss %.4f",
        metrics["epoch"] + 1,
        state["epochs"],
        metrics["val_top1"],
        metrics["val_top5"],
        metrics["val_r_exact"],
        metrics["val_loss"],
    )
    best = state["best"]
    key = (metrics["val_top1"], -metrics["val_loss"])
    is_best = best is None or key > (best["val_top1"], -best["val_loss"])
    state["best"] = metrics if is_best else best
    payload = checkpoint_payload(
        model,
        header,
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        scaler=scaler.state_dict(),
        epoch=metrics["epoch"],
        step=state["step"],
        best=state["best"],
        metrics=metrics,
    )
    save_checkpoint(run_dir / "last.pt", payload)
    if is_best:
        save_checkpoint(run_dir / "best.pt", payload)


def fit(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    run_dir: Path,
    device: torch.device,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    resume: bool = False,
) -> dict:
    """Train one run; returns the best validation metrics."""
    torch.manual_seed(cfg.seed)
    dtype = amp_dtype(device, cfg.amp)
    train, train_info = load_part(cfg, "train", processed_dir)
    val, val_info = load_part(cfg, "val", processed_dir)
    stats_path = source_dir(cfg.source, processed_dir) / f"pair_feature_stats_{cfg.split}.json"
    model = KSMech(model_cfg, fit_feature_stats(train), load_pair_stats(stats_path)).to(device)
    header = _run_header(cfg, model_cfg, {"train": train_info, "val": val_info}, stats_path)
    prov.write_json(run_dir / "run.json", {"device": str(device), "amp": str(dtype)}, header)
    train_loader, val_loader = make_loader(train, cfg, True), make_loader(val, cfg, False)
    total = cfg.epochs * len(train_loader.batch_sampler)
    opt = _optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg.warmup, total))
    # float16 only; 2^10, not 2^16: the first losses (~700 nats, a sum over every entry)
    # overflowed the first ten scaled backward passes
    scaler = torch.amp.GradScaler(device.type, init_scale=2.0**10, enabled=dtype == torch.float16)
    state = {"epoch": 0, "step": 0, "best": None}
    if resume and (run_dir / "last.pt").exists():
        state = _resume(run_dir / "last.pt", header, model, opt, sched, scaler)
    state |= {"total": total, "epochs": cfg.epochs}
    n_params = sum(p.numel() for p in model.parameters())
    log.info("%d parameters, %d train / %d val samples, %d steps on %s (%s), lambda_stop %g",
             n_params, len(train), len(val), total, device, dtype, stop_weight(cfg))  # fmt: skip
    skipped = {"train": train_info["skipped"], "val": val_info["skipped"]}
    for epoch in range(state["epoch"], cfg.epochs):
        state["epoch"] = epoch
        train_loader.batch_sampler.set_epoch(epoch)
        metrics = train_epoch(model, train_loader, opt, sched, scaler, device, dtype, cfg, state)
        warm = state["step"] < cfg.warmup
        val = evaluate(model, val_loader, device, dtype, warm, stop_weight(cfg))
        metrics |= {f"val_{k}": v for k, v in val.items()}
        metrics |= {"epoch": epoch, "step": state["step"], "skipped": skipped}
        _end_epoch(run_dir, model, header, opt, sched, scaler, state, metrics)
    return state["best"]
