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

`--direction backward` trains the previous-step model (reaction-network plan, item H) on the
backward store (`arrows.backward`: views and pair statistics under `<source>/backward/`, each
checked to be backward); run.json records the direction, which a forward run's header leaves
out, bit for bit as before.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from ksmech.arrows.backward import BACKWARD, FORWARD, check_direction, require_direction
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
from ksmech.data.human_benchmark import refuse_training
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
    direction: str = "forward"  # backward: the previous-step samples (`arrows.backward`)
    compile: bool = False  # torch.compile every encoder layer (dynamic shapes; docs/model_impl.md)
    compact_samples: bool = True  # samples kept with their cells as stored, expanded per batch
    load_workers: int = 1  # forked processes that read and verify the store's shards
    pin_memory: bool = False  # GPU runs: batches collated into page-locked memory, so the
    #   copies to the device are asynchronous (numbers unchanged; no gain measured on ROCm)
    eval_cell_budget: int = 0  # cells per validation/test batch; 0: cell_budget (no backward and
    #   no recompute there, so more fit; another batching changes rounding only)


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
    """The sources of a run: `pmechdb+reactmech` names two. An evaluation-only set (the human
    benchmark) is refused (`EvaluationOnly`): it is never trained on."""
    names = tuple(cfg.source.split("+"))
    refuse_training(names)
    return names


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
    split = f"{split_of(cfg, source)}_{part}"
    path = samples_path(source, split, processed_dir, cfg.direction)
    require_direction(path, cfg.direction)
    ids = _part_ids(cfg, source, part, path, processed_dir)
    samples = load_samples(path, ids=ids, compact=cfg.compact_samples, workers=cfg.load_workers)
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
    sizes = [(s.n_atoms, s.n_orbitals) for s in samples]  # the collate pads each apart
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
        pin_memory=bool(cfg.pin_memory) and torch.cuda.is_available(),
    )


def compile_encoder(model: KSMech) -> None:
    """torch.compile every encoder layer in place (`nn.Module.compile`: the state dict keeps its
    keys, so checkpoints are unchanged), with dynamic shapes, since every batch has its own N.
    Train and eval, with and without the sparse warm-up, each compile a variant: dynamo's limit
    is raised to 32 so none falls back to eager."""
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
    for layer in model.encoder.layers:
        layer.compile(dynamic=True)


@torch.no_grad()
def evaluate(
    model: KSMech, loader, device, dtype, dense_warmup: bool = False, lambda_stop: float = 0.0
) -> dict:
    """Mean loss and anchor NLL; anchor top-1/3/5 with set marginalization; the greedy R
    (top anchor, most probable entries) matching a variant of the label class; the mean
    P(END_MECH); and the anchor top-1 and greedy R over the pools with an unpaired electron
    (`open_n` of them, an odd N), the subset the pooled numbers hide."""
    model.eval()
    acc, n = None, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            h, p, y, _ = model.encode(batch, dense_warmup)
            _, parts = step_loss(model.decoder, h, p, y, batch, lambda_stop)
            lp = parts["lp_anchor"]
            rank = anchor_ranks(lp, batch)
            exact = edit_matches(*greedy_edit(model.decoder, h, p, lp, batch), batch)
        pools = (batch["nonbonding"] % 2 != 0).any(1)
        if acc is None:
            assert rank.shape == pools.shape, (rank.shape, pools.shape)
        row = _eval_row(parts, rank, exact, pools)
        acc = row if acc is None else acc + row
        n += len(rank)
    v = acc.tolist() if acc is not None else [0.0] * 10
    sums = {"loss": v[0], "anchor_nll": v[1]}
    sums |= {"top1": int(v[3]), "top3": int(v[4]), "top5": int(v[5]), "r_exact": int(v[6])}
    sums["p_stop"] = v[2]
    m = int(v[9])
    opened = {"open_top1": int(v[7]), "open_r_exact": int(v[8])}
    rates = {k: x / max(m, 1) for k, x in opened.items()}
    return {k: x / max(n, 1) for k, x in sums.items()} | {"n": n, "open_n": m} | rates


def _eval_row(parts: dict, rank, exact, pools) -> torch.Tensor:
    """One batch's ten sums as float64 on the device (loss, anchor NLL, P(stop), top-1/3/5,
    exact R, open top-1, open exact R, open pools): the per-batch sums the host used to read
    one by one, added in the same order in the same IEEE double, read once per `evaluate`."""
    top1 = rank < 1
    zero = torch.zeros_like(exact)
    terms = [
        parts["loss"].sum(),
        parts["anchor_nll"].sum(),
        parts["p_stop"].sum(),
        top1.sum(),
        (rank < 3).sum(),
        (rank < 5).sum(),
        exact.sum(),
        (top1 & pools).sum(),
        torch.where(pools, exact, zero).sum(),
        pools.sum(),
    ]
    return torch.stack([t.double() for t in terms])


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


class StepTrack:
    """The non-finite batches of a fused-optimizer run, counted on the device and read once
    per logging interval (`check`), not once per step: a batch whose loss or gradient norm is
    not finite takes no optimizer step (the fused AdamW skips it on the device, `found_inf`),
    `MAX_NONFINITE_IN_A_ROW` in a row stop the run, and the fp16 loss scale is checked against
    `MIN_LOSS_SCALE` at the same time. The losses are summed there too."""

    def __init__(self, device: torch.device):
        def zero():
            return torch.zeros((), device=device)

        self.loss, self.n, self.bad, self.row, self.max_row = (zero() for _ in range(5))
        self.window_loss, self.window_n = zero(), zero()
        self.recent: list = []  # (flag, step ids, step) since the last check

    def add(self, loss: torch.Tensor, norm, ids, step: int) -> torch.Tensor:
        """Count one batch; returns its found_inf flag (a 0-d float) for the optimizer. A batch is
        bad when its loss is not finite, or (`norm` given: no fp16 scaler, whose own overflow
        check skips those steps and backs off) its gradient norm."""
        loss = loss.detach().float()
        bad = ~torch.isfinite(loss)
        if norm is not None:
            bad = bad | ~torch.isfinite(norm)
        f = bad.float()
        good_loss = torch.where(bad, 0.0, loss)
        self.loss += good_loss
        self.window_loss += good_loss
        self.n += 1 - f
        self.window_n += 1 - f
        self.bad += f
        self.row = (self.row + 1) * f
        self.max_row = torch.maximum(self.max_row, self.row)
        self.recent.append((bad, ids, step))
        return f  # shape (), as the scaler's found_inf

    def check(self, scaler, step: int) -> float:
        """One host sync: log the skipped batches since the last check, raise `NonFiniteError`
        on a run of them or a vanished loss scale; returns the window's mean finite loss."""
        if self.recent:
            flags = torch.stack([b for b, _, _ in self.recent]).tolist()
            for flag, (_, ids, at) in zip(flags, self.recent, strict=True):
                if flag:
                    log.warning("step %d: non-finite loss or gradient, batch skipped: %s",
                                at, " ".join(ids))  # fmt: skip
        self.recent = []
        if self.max_row.item() >= MAX_NONFINITE_IN_A_ROW:
            raise NonFiniteError(f"step {step}: {MAX_NONFINITE_IN_A_ROW} non-finite training "
                                 "batches in a row")  # fmt: skip
        if scaler.is_enabled() and scaler.get_scale() < MIN_LOSS_SCALE:
            raise NonFiniteError(f"step {step}: fp16 loss scale {scaler.get_scale():g}")
        mean = (self.window_loss / self.window_n).item()  # nan when no batch was finite
        self.window_loss.zero_()
        self.window_n.zero_()
        return mean


def _optimize(model, opt, scaler, loss, cfg: TrainConfig, track: StepTrack | None, batch, step):
    """Backward, clip, step: (the gradient norm, the batch's found_inf flag or None). With a
    tracker the step is skipped on the device when the batch is not finite: the fp16 scaler's
    own check, or `found_inf` handed to the fused AdamW."""
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaled = scaler.is_enabled()
    flag = None
    if track is not None:
        flag = track.add(loss, None if scaled else norm, batch["step_id"], step + 1)
    if flag is not None and not scaled:
        opt.grad_scale, opt.found_inf = None, flag
    scaler.step(opt)
    scaler.update()
    return norm, flag


def train_epoch(model, loader, opt, sched, scaler, device, dtype, cfg: TrainConfig, state: dict):
    """One pass over the training batches; `state` carries the optimizer step and its total.
    With a fused optimizer the step has no host sync (`StepTrack`); otherwise every loss is
    checked on the host before its backward (`_skip_nonfinite`)."""
    model.train()
    track = StepTrack(device) if opt.defaults.get("fused") else None
    seen, t0, losses = 0, time.time(), []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            h, p, y, aux = model.encode(batch, state["step"] < cfg.warmup)
            loss, parts = step_loss(model.decoder, h, p, y, batch, stop_weight(cfg))
            loss = loss + sum(aux.values())  # the sparse indexer's KL trains the indexer only
        opt.zero_grad(set_to_none=True)
        if track is None and _skip_nonfinite(loss, parts | aux, batch, state):  # no step
            sched.step()
            state["step"] += 1
            continue
        norm, _ = _optimize(model, opt, scaler, loss, cfg, track, batch, state["step"])
        sched.step()
        state["step"] += 1
        seen += len(batch["step_id"])
        if track is None:
            losses.append(float(loss.detach()))
            _check_scale(scaler, state["step"])
        if state["step"] % cfg.log_every == 0:
            window = (
                track.check(scaler, state["step"]) if track else np.mean(losses[-cfg.log_every :])
            )
            _log_step(model, state, window, parts, norm, sched, seen / (time.time() - t0))
    return _end_train_epoch(model, track, scaler, losses, state, t0)


def _check_scale(scaler, step: int) -> None:
    if scaler.is_enabled() and scaler.get_scale() < MIN_LOSS_SCALE:
        raise NonFiniteError(f"step {step}: fp16 loss scale {scaler.get_scale():g}")


def _log_step(model, state: dict, window: float, parts: dict, norm, sched, rate: float) -> None:
    check_finite_weights(model, state["step"])
    log.info(
        "step %d / %d epoch %d loss %.4f anchor_nll %.4f grad %.2f lr %.2e %.1f samples/s",
        state["step"],
        state["total"],
        state["epoch"],
        float(window),
        float(parts["anchor_nll"].mean()),
        float(norm),
        sched.get_last_lr()[0],
        rate,
    )


def _end_train_epoch(model, track, scaler, losses: list, state: dict, t0: float) -> dict:
    """The epoch's training metrics, after the last check of the non-finite batches."""
    if track is not None:
        track.check(scaler, state["step"])
        n = track.n.item()
        state["nonfinite_epoch"] = int(track.bad.item())
        train_loss = track.loss.item() / n if n else float("nan")
    else:
        train_loss = float(np.mean(losses)) if losses else float("nan")
    check_finite_weights(model, state["step"])
    return {
        "train_loss": train_loss,
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


def checkpoint_bytes(payload: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def save_checkpoint(path: Path, payload: dict) -> None:
    prov.atomic_write(path, checkpoint_bytes(payload))


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
    """AdamW; weight decay on the matrices only (not on biases or norm scales). On a GPU the
    fused kernel: one launch for every parameter, and a step it skips on the device when told
    the gradients are not finite (`found_inf`, no host sync; `train_epoch`)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": rest, "weight_decay": 0.0},
    ]
    fused = all(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(groups, lr=cfg.lr, fused=fused or None)


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
        path = pair_stats_path(cfg.source, split, processed_dir, cells, cfg.direction)
        return path, load_pair_stats(require_direction(path, cfg.direction))
    return None, pair_stats(train, cells=cells, extra=extra)


def run_notes(cfg: TrainConfig, stats_path: Path | None) -> dict:
    """How the run is selected and how its pair statistics were fitted (plan P9, step 5f)."""
    parts = []
    cohorts = _source_map(cfg.val_cohort)
    for s in sources(cfg):
        cohort = f", cohort {cohorts[s]}" if s in cohorts else ""
        parts.append(f"{s} {split_of(cfg, s)} val{cohort}")
    pool = "pooled over " if len(parts) > 1 else "on "
    samples = " (backward samples)" if cfg.direction == BACKWARD else ""
    return {
        "selection": f"best.pt: the highest anchor top-1 {pool}{' and '.join(parts)}{samples}",
        "pair_statistics": str(stats_path) if stats_path else "fitted on the training samples",
    }


def train_config(cfg: TrainConfig) -> dict:
    """The run's training configuration as run.json records it: `direction` only for a backward
    run, so a forward run's header (and its --resume check) is bit for bit as before."""
    out = asdict(cfg)
    if out["direction"] == FORWARD:
        del out["direction"]
    return out


def _run_header(cfg: TrainConfig, model_cfg: ModelConfig, parts: dict, stats_path: Path | None):
    inputs = {}
    for info in parts.values():
        for path in info["path"] if isinstance(info["path"], list) else [info["path"]]:
            inputs |= prov.file_input(path)
    if stats_path is not None:
        inputs |= prov.file_input(stats_path)
    config = {"train": train_config(cfg), "model": model_cfg.to_dict()}
    config["sample_version"] = SAMPLE_VERSION
    extra = {"parts": parts} | run_notes(cfg, stats_path)
    return prov.stamp("train_run", TRAIN_VERSION, inputs, config, extra=extra)


def comparable_config(config: dict) -> dict:
    """A run header's config with every `ModelConfig` and `TrainConfig` field present, the
    fields a run predates at their defaults (the defaults keep a run's behaviour bit for bit),
    so `--resume` compares what the run is, not which fields existed when it started."""
    out = dict(config)
    out["model"] = ModelConfig.from_dict(dict(config.get("model", {}))).to_dict()
    names = {f.name for f in fields(TrainConfig)}
    train = dict(config.get("train", {}))
    known = {k: v for k, v in train.items() if k in names}
    removed = {k: v for k, v in train.items() if k not in names}  # still compared
    out["train"] = train_config(TrainConfig(**known)) | removed
    return prov.normalized(out)


def _resume(path: Path, header, model, opt, sched, scaler) -> dict:
    """Restore a run from its `last.pt`; refused when the run's inputs or configuration
    differ from the checkpoint's (`comparable_config`: a field added since is at its default)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old = prov.Provenance.from_dict(ckpt["provenance"])
    same = comparable_config(old.config) == comparable_config(header.config)
    if old.inputs != header.inputs or not same:
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
    data = checkpoint_bytes(payload)  # serialized once, written under both names
    prov.atomic_write(run_dir / "last.pt", data)
    if is_best:
        prov.atomic_write(run_dir / "best.pt", data)


def fit(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    run_dir: Path,
    device: torch.device,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    resume: bool = False,
) -> dict:
    """Train one run; returns the best validation metrics."""
    check_direction(cfg.direction)
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
    if cfg.compile:
        compile_encoder(model)
    parts = {"train": train_info, "val": val_info} | ({"test": held[1]} if held[1] else {})
    header = _run_header(cfg, model_cfg, parts, stats_path)
    prov.write_json(run_dir / "run.json", {"device": str(device), "amp": str(dtype)}, header)
    train_loader = make_loader(train, cfg, True, cfg.resonance_draws)
    eval_cfg = replace(cfg, cell_budget=cfg.eval_cell_budget or cfg.cell_budget)
    val_loaders = {s: make_loader(part, eval_cfg, False) for s, part in val.items()}
    test_loaders = {s: make_loader(part, eval_cfg, False) for s, part in held[0].items()}
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
