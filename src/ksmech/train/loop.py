"""Training (spec v9, "Training objective"): AdamW with a linear warm-up and a cosine decay,
gradient clipping, mixed precision, batches by a cell budget, and the edit-level metrics on the
validation part after every epoch. Each epoch every training sample is drawn in a random one of
its resonance drawings (`--resonance-draws`, on by default; spec v9, "Resonance augmentation"),
with its label recomputed for that drawing; validation always reads the dataset's drawing.

The encoder is `--encoder dense | flex | sparse`. The sparse encoder's indexer KL is added to
the loss (it reaches only the indexer), and it runs in its dense warm-up mode while the
optimizer step is within the learning-rate warm-up.

A run lives in `data/models/<source>/<run>/`: `run.json` (provenance header: the training and
validation views, the pair statistics, both configurations), `train.log` (one timestamped
`step n / total` line per logging interval, the format `ksmech-eta` reads), `metrics.jsonl`
(one line per epoch) and the checkpoints `last.pt` and `best.pt` (best validation anchor
top-1). Samples too large for the cell budget are skipped with the reason TOO_LARGE and
counted in `run.json` and in every metrics line.

`--source pmechdb+reactmech` trains on several sources at once: their training parts are
pooled, the pair statistics are fitted on the pool (each source's stored statistics were fitted
on its own part), and validation runs on each source's part, reported per source
(`val_<source>_top1`, ...) and pooled (`val_top1`, which picks `best.pt`). The run lives in
`data/models/pmechdb+reactmech/<run>/`. `--split pmechdb=molecule,rmechdb=rc_molecule` names
each source's split, `--val-cohort pmechdb=closed_base` restricts a source's validation part to
a cohort; a run with both pmechdb and rmechdb leaves the rmechdb training steps of the cohort
leak_guard_joint out (LEAK_GUARD, counted). run.json states how the run is selected and how its
pair statistics were fitted. An open-shell run reads the pair features of
`stage_a.OPEN_SHELL_BLOCK`, which no stored statistics file holds, so its statistics are fitted
here on its training samples, one source or several.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.steps import (
    SAMPLE_VERSION,
    load_pair_stats,
    load_samples,
    pair_stats,
    pair_stats_path,
    sample_ids,
    samples_path,
)
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, steps_path
from ksmech.eval.cohorts import load_cohort, restrict
from ksmech.model.config import ModelConfig
from ksmech.model.inputs import FeatureStats, fit_feature_stats, to_device, token_columns
from ksmech.model.loss import anchor_ranks, edit_matches, greedy_edit, step_loss
from ksmech.model.net import KSMech
from ksmech.physics.pairs import PairStats
from ksmech.physics.stage_a import OPEN_SHELL_BLOCK
from ksmech.train.batching import CellBudgetSampler, DrawnSamples

log = logging.getLogger(__name__)

TRAIN_VERSION = 1
DEFAULT_MODELS_DIR = Path("data/models")
TOO_LARGE = "TOO_LARGE"
OPEN_METRICS = ("open_n", "open_top1", "open_r_exact")  # `evaluate`: the open-shell pools


@dataclass(frozen=True)
class TrainConfig:
    source: str = "pmechdb"  # several joined by "+": pmechdb+reactmech
    split: str = "molecule"  # every source's, or per source: pmechdb=molecule,rmechdb=rc_molecule
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
    resonance_draws: bool = True  # each epoch, every training sample in a random drawing
    eval_test: bool = False  # also score the split's test part each epoch (test_*), never for best
    drop_domain_class: tuple[str, ...] = ()  # training samples of these classes left out (P4)
    val_cohort: str = ""  # validation restricted to a cohort, per source: pmechdb=closed_base


# PMechDB and rmechdb: isolated steps, no stop labels; the minimum over a joint run's sources is
# 0, so END_MECH stays untrained there
STOP_WEIGHT, SOURCE_STOP_WEIGHT = 0.5, {"pmechdb": 0.0, "rmechdb": 0.0}
# a non-finite run stops (docs/decisions.md, 2026-10-28): at a loss scale of 0 the unscaled
# gradients are NaN and the scaler's inf check misses them (0 * inf); so many skipped batches
# in a row is no glitch
MIN_LOSS_SCALE, MAX_NONFINITE_IN_A_ROW = 1.0, 50


class NonFiniteError(RuntimeError):
    """The run cannot go on: its weights, its loss scale or its losses are no longer finite.
    Raised before the epoch's checkpoints are written, so `last.pt` keeps the last good epoch."""


def sources(cfg: TrainConfig) -> tuple[str, ...]:
    """The sources of a run: `pmechdb+reactmech` names two."""
    return tuple(cfg.source.split("+"))


def _source_map(text: str) -> dict[str, str]:
    """{source: value} of `a=x,b=y`."""
    return dict(item.split("=", 1) for item in text.split(",") if item)


def split_of(cfg: TrainConfig, source: str) -> str:
    """The split strategy of one source: `cfg.split`, or its entry when that is a map. A map
    must name every source of the run, and only those."""
    if "=" not in cfg.split:
        return cfg.split
    named = _source_map(cfg.split)
    if set(named) != set(sources(cfg)):
        raise ValueError(f"--split {cfg.split!r} must name each of {sources(cfg)}")
    return named[source]


def stop_weight(cfg: TrainConfig) -> float:
    """lambda_stop of the run: the configured value, else the smallest default of its
    sources."""
    if cfg.stop_weight >= 0:
        return cfg.stop_weight
    return min(SOURCE_STOP_WEIGHT.get(s, STOP_WEIGHT) for s in sources(cfg))


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


def drop_undecodable_drawings(samples: list, part: str) -> int:
    """Drop, in place, the resonance drawings whose label class holds a step the decoder cannot
    write (`ReactionMatrix.is_elementary` under the sample's source rule,
    `domain.open_shell_labels`: a bond order changing by 2, or a one-electron atom entry from a
    polar source). A store built before the build checked this (2026-10-26, 5 of 7,864
    drawings) still has them, and one such drawing put the loss's gather out of range. Returns
    how many were dropped, each logged."""
    from ksmech.data.domain import open_shell_labels

    dropped = 0
    for s in samples:
        rule = open_shell_labels(getattr(s, "source", None))
        keep = [d for d in s.drawings if all(v.is_elementary(rule) for v in d.edit_label.variants)]
        if len(keep) < len(s.drawings):
            log.warning("%s %s: %d resonance drawing(s) dropped: NOT_ELEMENTARY",
                        part, s.step_id, len(s.drawings) - len(keep))  # fmt: skip
            dropped += len(s.drawings) - len(keep)
            s.drawings = keep
    return dropped


DROPPED_DOMAIN_CLASS = "DROPPED_DOMAIN_CLASS"
LEAK_GUARD = "LEAK_GUARD"


def drop_domain_classes(samples: list, source: str, classes, processed_dir: Path) -> list:
    """The samples whose step's `domain_class` holds none of `classes` (the training part of an
    ablation, open-shell plan P4: the open_shell_metal cohort left out); each dropped one is
    logged."""
    from ksmech.data.domain import domain_classes

    want = set(classes)
    held = {
        s.step_id
        for s in read_steps(steps_path(source, processed_dir))
        if domain_classes(s.domain_class) & want
    }
    for s in samples:
        if s.step_id in held:
            log.info("train %s left out: %s (%s)", s.step_id, DROPPED_DOMAIN_CLASS, sorted(want))
    return [s for s in samples if s.step_id not in held]


def leak_guarded(samples: list, cfg: TrainConfig, source: str, processed_dir: Path) -> list:
    """The samples of a training part without those of the cohort leak_guard_joint: the rmechdb
    steps of a run with PMechDB that share a species with closed_base val or test (open-shell
    plan P8 step 6); each one left out is logged."""
    if source != "rmechdb" or "pmechdb" not in sources(cfg):
        return samples
    held = load_cohort("rmechdb", "leak_guard_joint", processed_dir) or set()
    for s in samples:
        if s.step_id in held:
            log.info("train %s left out: %s", s.step_id, LEAK_GUARD)
    return [s for s in samples if s.step_id not in held]


def _part_ids(cfg: TrainConfig, source: str, part: str, path: Path, processed_dir: Path):
    """The ids to read of a part: all (None), or for a validation part with a cohort in
    `cfg.val_cohort`, the cohort's."""
    name = _source_map(cfg.val_cohort).get(source)
    if part != "val" or not name:
        return None
    return restrict(sample_ids(path), load_cohort(source, name, processed_dir))


def _train_filters(samples: list, cfg: TrainConfig, source: str, processed_dir: Path):
    """(the training samples kept, the skip counts): `drop_domain_classes`, `leak_guarded`."""
    skipped, total = {}, len(samples)
    if cfg.drop_domain_class:
        samples = drop_domain_classes(samples, source, cfg.drop_domain_class, processed_dir)
        skipped[DROPPED_DOMAIN_CLASS] = total - len(samples)
        log.info("%d training samples left out by --drop-domain-class", total - len(samples))
    guarded = leak_guarded(samples, cfg, source, processed_dir)
    if len(guarded) < len(samples):
        skipped[LEAK_GUARD] = len(samples) - len(guarded)
    return guarded, skipped


def load_source_part(
    cfg: TrainConfig, source: str, part: str, processed_dir: Path
) -> tuple[list, dict]:
    """The samples of one source's part that fit the token cap, and the skip counts (samples
    too large, drawings the decoder cannot write, and in the training part the samples of the
    domain classes `cfg.drop_domain_class` names and the leak guard's)."""
    path = samples_path(source, f"{split_of(cfg, source)}_{part}", processed_dir)
    samples = load_samples(path, ids=_part_ids(cfg, source, part, path, processed_dir))
    if cfg.limit:
        samples = samples[: cfg.limit]
    skipped, total = {}, len(samples)
    if part == "train":
        samples, skipped = _train_filters(samples, cfg, source, processed_dir)
    keep = [s for s in samples if _size(s) + 1 <= cfg.max_tokens]
    if len(keep) < len(samples):
        skipped[TOO_LARGE] = len(samples) - len(keep)
    for s in samples:
        if _size(s) + 1 > cfg.max_tokens:
            log.warning("%s %s skipped: %s (%d tokens)", part, s.step_id, TOO_LARGE, _size(s))
    if n := drop_undecodable_drawings(keep, part):
        skipped["drawings NOT_ELEMENTARY"] = n
    return keep, {"n": total, "used": len(keep), "skipped": skipped, "path": str(path)}


def load_part(cfg: TrainConfig, part: str, processed_dir: Path) -> tuple[dict[str, list], dict]:
    """Source -> the samples of its part (`load_source_part`), and the counts: one source's
    as they are; for several, summed, with the views in `path` as a list and each source's
    counts under `sources`."""
    loaded = {s: load_source_part(cfg, s, part, processed_dir) for s in sources(cfg)}
    samples = {s: x for s, (x, _) in loaded.items()}
    infos = {s: info for s, (_, info) in loaded.items()}
    if len(infos) == 1:
        return samples, next(iter(infos.values()))
    skipped = sum((Counter(i["skipped"]) for i in infos.values()), Counter())
    return samples, {
        "n": sum(i["n"] for i in infos.values()),
        "used": sum(i["used"] for i in infos.values()),
        "skipped": dict(skipped),
        "path": [i["path"] for i in infos.values()],
        "sources": infos,
    }


def make_loader(
    samples: list, cfg: TrainConfig, shuffle: bool, draw: bool = False
) -> torch.utils.data.DataLoader:
    """Batches by the cell budget; with `draw` (and `shuffle`) each sample in a random one of
    its resonance drawings every epoch."""
    draws = [s.n_drawings for s in samples] if draw else None
    sizes = [_size(s) for s in samples]
    sampler = CellBudgetSampler(
        sizes, cfg.cell_budget, cfg.max_batch, shuffle, cfg.seed, drawings=draws
    )
    return torch.utils.data.DataLoader(
        DrawnSamples(samples),
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
    P(END_MECH); and the anchor top-1 and greedy R over the pools with an unpaired electron
    (`open_n` of them, an odd N), the subset the pooled numbers hide."""
    model.eval()
    sums = {"loss": 0.0, "anchor_nll": 0.0, "top1": 0, "top3": 0, "top5": 0, "r_exact": 0}
    sums["p_stop"], n = 0.0, 0
    opened = dict.fromkeys(OPEN_METRICS, 0)
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
        pools = (batch["nonbonding"] % 2 != 0).any(1)
        opened["open_top1"] += int((rank[pools] < 1).sum())
        opened["open_r_exact"] += int(exact[pools].sum())
        opened["open_n"] += int(pools.sum())
    m = opened.pop("open_n")
    rates = {k: v / max(m, 1) for k, v in opened.items()}
    return {k: v / max(n, 1) for k, v in sums.items()} | {"n": n, "open_n": m} | rates


def evaluate_sources(model: KSMech, loaders: dict, *args) -> dict:
    """`evaluate` on each source's loader (`args` as there): the metrics pooled over every
    sample, and with several sources each source's under `<source>_`."""
    per = {s: evaluate(model, loader, *args) for s, loader in loaders.items()}
    n = sum(r["n"] for r in per.values())
    m = sum(r["open_n"] for r in per.values())
    keys = [k for k in next(iter(per.values())) if k not in ("n", *OPEN_METRICS)]
    out = {k: sum(r[k] * r["n"] for r in per.values()) / max(n, 1) for k in keys} | {"n": n}
    rates = OPEN_METRICS[1:]
    out |= {k: sum(r[k] * r["open_n"] for r in per.values()) / max(m, 1) for k in rates}
    out["open_n"] = m
    if len(per) > 1:
        out |= {f"{s}_{k}": v for s, r in per.items() for k, v in r.items()}
    return out


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
        if _skip_nonfinite(loss, parts | aux, batch, state):  # takes no optimizer step
            sched.step()
            state["step"] += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        state["step"] += 1
        seen += len(batch["step_id"])
        losses.append(float(loss.detach()))
        if scaler.is_enabled() and scaler.get_scale() < MIN_LOSS_SCALE:
            raise NonFiniteError(f"step {state['step']}: fp16 loss scale {scaler.get_scale():g}")
        if state["step"] % cfg.log_every == 0:
            check_finite_weights(model, state["step"])
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
    check_finite_weights(model, state["step"])
    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "seconds": time.time() - t0,
        "nonfinite_batches": state.pop("nonfinite_epoch", 0),
    }


def _skip_nonfinite(loss: torch.Tensor, terms: dict, batch: dict, state: dict) -> bool:
    """True when the training loss is not finite: the batch is skipped, logged with its steps
    and its non-finite loss terms, and counted (`nonfinite_epoch`); `MAX_NONFINITE_IN_A_ROW`
    in a row stop the run."""
    if bool(torch.isfinite(loss.detach())):
        state["nonfinite_row"] = 0
        return False
    state["nonfinite_epoch"] = state.get("nonfinite_epoch", 0) + 1
    state["nonfinite_row"] = state.get("nonfinite_row", 0) + 1
    bad = [k for k, v in terms.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
    log.warning("step %d: non-finite loss (%s), batch skipped: %s", state["step"] + 1,
                ", ".join(bad) or "the sum", " ".join(batch["step_id"]))  # fmt: skip
    if state["nonfinite_row"] >= MAX_NONFINITE_IN_A_ROW:
        raise NonFiniteError(f"step {state['step'] + 1}: {state['nonfinite_row']} non-finite "
                             "training losses in a row")  # fmt: skip
    return True


def check_finite_weights(model: torch.nn.Module, step: int) -> None:
    """Raise `NonFiniteError` when a parameter holds a NaN or an infinity (one sync)."""
    params = [p for p in model.parameters() if p.is_floating_point()]
    if bool(torch.stack([torch.isfinite(p).all() for p in params]).all()):
        return
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    raise NonFiniteError(f"step {step}: {len(bad)} non-finite parameters, e.g. {bad[0]}")


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


def run_pair_stats(
    cfg: TrainConfig,
    train: list,
    processed_dir: Path,
    cells: str = "valid",
    extra: tuple[str, ...] = (),
):
    """(the statistics file or None, the pair statistics of the cells `cells`): one source's
    stored `pair_feature_stats_<split>[_allcells].json`; for several, or with the `extra`
    features no stored file holds, fitted here on the training samples, as `ksmech-arrows
    samples` fits one source's on its own."""
    if len(sources(cfg)) == 1 and not extra:
        split = split_of(cfg, cfg.source)
        path = pair_stats_path(cfg.source, split, processed_dir, cells)
        return path, load_pair_stats(path)
    return None, pair_stats(train, cells=cells, extra=extra)


def run_notes(cfg: TrainConfig, stats_path: Path | None) -> dict:
    """How the run is selected and how its pair statistics were fitted (plan P9, step 5f)."""
    parts = []
    cohorts = _source_map(cfg.val_cohort)
    for s in sources(cfg):
        cohort = f", cohort {cohorts[s]}" if s in cohorts else ""
        parts.append(f"{s} {split_of(cfg, s)} val{cohort}")
    pool = "pooled over " if len(parts) > 1 else "on "
    return {
        "selection": f"best.pt: the highest anchor top-1 {pool}{' and '.join(parts)}",
        "pair_statistics": str(stats_path) if stats_path else "fitted on the training samples",
    }


def _run_header(cfg: TrainConfig, model_cfg: ModelConfig, parts: dict, stats_path: Path | None):
    inputs = {}
    for info in parts.values():
        for path in info["path"] if isinstance(info["path"], list) else [info["path"]]:
            inputs |= prov.file_input(path)
    if stats_path is not None:
        inputs |= prov.file_input(stats_path)
    config = {"train": asdict(cfg), "model": model_cfg.to_dict(), "sample_version": SAMPLE_VERSION}
    extra = {"parts": parts} | run_notes(cfg, stats_path)
    return prov.stamp("train_run", TRAIN_VERSION, inputs, config, extra=extra)


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


def better_epoch(metrics: dict, best: dict | None) -> bool:
    """Whether an epoch's metrics make the new `best.pt`: a higher validation anchor top-1,
    ties broken by the lower loss; never an epoch with a non-finite loss."""
    if not (math.isfinite(metrics["val_loss"]) and math.isfinite(metrics["train_loss"])):
        return False
    key = (metrics["val_top1"], -metrics["val_loss"])
    return best is None or key > (best["val_top1"], -best["val_loss"])


def _end_epoch(run_dir: Path, model, header, opt, sched, scaler, state: dict, metrics: dict):
    """Append the metrics line, log it, and write `last.pt` (and `best.pt` when the validation
    anchor top-1 improves, ties broken by the loss; never for an epoch with a non-finite loss)."""
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
    if "test_top1" in metrics:
        log.info(
            "epoch %d test: top1 %.3f top5 %.3f r_exact %.3f loss %.4f",
            metrics["epoch"] + 1,
            metrics["test_top1"],
            metrics["test_top5"],
            metrics["test_r_exact"],
            metrics["test_loss"],
        )
    is_best = better_epoch(metrics, state["best"])
    state["best"] = metrics if is_best else state["best"]
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
    by_source, train_info = load_part(cfg, "train", processed_dir)
    val, val_info = load_part(cfg, "val", processed_dir)
    held = load_part(cfg, "test", processed_dir) if cfg.eval_test else ({}, None)
    train = [s for part in by_source.values() for s in part]
    extra = OPEN_SHELL_BLOCK if model_cfg.open_shell else ()
    stats_path, stats = run_pair_stats(cfg, train, processed_dir, model_cfg.pair_cells, extra)
    feature_stats = fit_feature_stats(train, *token_columns(model_cfg))
    model = KSMech(model_cfg, feature_stats, stats).to(device)
    parts = {"train": train_info, "val": val_info} | ({"test": held[1]} if held[1] else {})
    header = _run_header(cfg, model_cfg, parts, stats_path)
    prov.write_json(run_dir / "run.json", {"device": str(device), "amp": str(dtype)}, header)
    train_loader = make_loader(train, cfg, True, cfg.resonance_draws)
    val_loaders = {s: make_loader(part, cfg, False) for s, part in val.items()}
    test_loaders = {s: make_loader(part, cfg, False) for s, part in held[0].items()}
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
    n_params, n_val = sum(p.numel() for p in model.parameters()), val_info["used"]
    log.info("%d parameters, %d train / %d val samples, %d steps on %s (%s), lambda_stop %g",
             n_params, len(train), n_val, total, device, dtype, stop_weight(cfg))  # fmt: skip
    if cfg.resonance_draws:
        other = sum(s.n_drawings > 1 for s in train)
        log.info("resonance draws: %d of %d training samples have other drawings, %d in all",
                 other, len(train), sum(s.n_drawings - 1 for s in train))  # fmt: skip
    skipped = {p: info["skipped"] for p, info in parts.items()}
    for epoch in range(state["epoch"], cfg.epochs):
        state["epoch"] = epoch
        train_loader.batch_sampler.set_epoch(epoch)
        metrics = train_epoch(model, train_loader, opt, sched, scaler, device, dtype, cfg, state)
        warm = state["step"] < cfg.warmup
        res = evaluate_sources(model, val_loaders, device, dtype, warm, stop_weight(cfg))
        metrics |= {f"val_{k}": v for k, v in res.items()}
        if test_loaders:  # reported only: best.pt is chosen on val
            res = evaluate_sources(model, test_loaders, device, dtype, warm, stop_weight(cfg))
            metrics |= {f"test_{k}": v for k, v in res.items()}
        metrics |= {"epoch": epoch, "step": state["step"], "skipped": skipped}
        _end_epoch(run_dir, model, header, opt, sched, scaler, state, metrics)
    return state["best"]
