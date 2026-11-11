# Training speed: the changes and their options (2026-11-11)

Why training a 7.4M-parameter model was slow, what changed on branch `claude/train-speed`, and
the option that turns each change on. The profiles and the measurements are in
`docs/decisions.md` (2026-11-11, "training speed"). The numbering is the one the user approved
the changes under; deferred and rejected items are listed at the end.

## What limits the speed

- **The dense pair track, not the parameters.** Every layer reads and writes a
  B x (N + 1)^2 x 64 tensor several times (N ~ 160 tokens on PMechDB, ~240 on ReactMech). The
  step is bound by GPU memory bandwidth: the 5090 runs 7.6x faster than the RX 6600 XT, their
  bandwidth ratio, while the FLOPs are ~40x below the compute limit. Under autocast the
  LayerNorm of the pair track ran in float32, so the largest tensor was cast up and back down in
  every layer (layer_norm and the casts: ~45% of the GPU time).
- **The pod times were a third of a GPU.** `pod_orb/start.sh` ran three runs per GPU; one run
  alone is ~2.6x faster per step.
- **Ruled out**: per-epoch I/O (the store is held in memory), the DataLoader (670 samples/s with
  2 workers and the resonance draws, against ~175 a 5090 consumes on PMechDB), the batch size
  (flat from ~8 samples per batch up), mixed precision (on: bf16 on CUDA, fp16 on ROCm, +26%
  over fp32), heavy CPU work inside the step.
- **Host syncs and kernel launches**: 22 syncs and ~2,200 kernels per step left the RX card idle
  38-54 ms per step; on a 5090, with kernels 7.6x shorter, that share grows.

## The changes

| # | Change | Option | Default | Effect on the numbers |
|---|---|---|---|---|
| 2 | `torch.compile` of every encoder layer, dynamic shapes, in place (`nn.Module.compile`: the state dict keys stay) | `ksmech-train fit --compile` | off | rounding only |
| 3 | The pair track's LayerNorm in the autocast dtype (fp32 accumulation inside the kernel) | `--half-pair-norm` (ModelConfig `half_pair_norm`) | off | none outside autocast; rounding under it |
| 4 | No host sync in a training step: the data-dependent index sets (real pair cells, admissible anchor rows, their distinct rows, the samples without an anchor, the atom entries the loss reads) computed in the collate on the CPU by the same expressions; constant tensors (statistics, column indices, the atom alphabet) cached on the device; the non-finite check on the device | always on | - | none: the same rows in the same order, the loss bit for bit |
| 4b | With a fused optimizer (every GPU run) a batch whose loss (or, without the fp16 scaler, gradient norm) is not finite is skipped on the device (`found_inf` handed to the fused AdamW) and read back once per `--log-every` steps (`train.loop.StepTrack`); `MAX_NONFINITE_IN_A_ROW` and the fp16 loss-scale floor are checked there | always on with a fused optimizer | - | a NaN batch: no step, as before; the fp16 scale now also backs off on it |
| 6 | AdamW fused (one kernel for every parameter) on a GPU | always on on a GPU | - | rounding; a `--resume` of a run started before is not bit for bit |
| 8 | Batches by orbitals, then atoms, within each pool, and a batch's cost counted as the collate pays it: len x (max atoms + max orbitals + 1)^2 | always on (`CellBudgetSampler` with (atoms, orbitals) sizes) | - | other batches, hence another training order; no batch over the cell budget any more |
| 9 | MLP_P's first layer applied part by part (RBF block, kind/same-fragment table, atom block, centre weights, z on the real cells) instead of on the (B, N + 1, N + 1, 83) channel tensor; the RBF computed as before | `--split-pair-init` (ModelConfig `split_pair_init`) | off | the same linear map: 6e-7 relative in float32 |
| 12 | The store's shards read and verified in forked processes, the order kept | `--load-workers N` | 1 | none |
| 13 | Samples kept in memory with their cells as the shard stores them (float16 byte planes, bit-packed flags and defined masks: `store.CompactCells`), expanded one batch at a time in the DataLoader workers (`store.dense`); the sample build can write the store while it builds | `--compact-samples` (training), `ksmech-arrows samples --stream` (build) | compact on, stream off | none: the expansion is `unpack`'s own code, bit for bit (checked on all 1,241 PMechDB val samples); stream: the shards hold the samples in step order instead of by step id |
| 15 | One run per GPU on a pod | `PER_GPU` in `rhoMech-hf-staging/pod_orb/start.sh` | 1 (was 3) | - |
| 18 | The pair update in every k-th layer only, the bias from p in every layer (ablation) | `--pair-update-every K` (ModelConfig `pair_update_every`) | 1 | a different model; k = 1 is the model as before, bit for bit |

Item 13 replaces the "schema v20" of the proposal: the store already keeps `pair_z` as float16
and the 0/1 features and the defined masks as bits; only the trainer expanded them to float32 on
load (~102 B per cell, ~3x the stored ~32 B). Keeping the stored form in memory saves the same
memory with no schema change, no store rebuild, and a lossless result by construction. For
ReactMech (~8,300 cells per sample) the training part is ~25 GB instead of ~85 GB; a light
sample without its cells (`store.light`) is what the reports, the splits and the views read,
and `--stream` writes each shard as it fills, so the build holds one shard of whole samples.
The pair statistics then stream the training part from the store (`pair_stats` is one pass).

Item 8 differs from the proposal ("by atoms, then orbitals"): simulated on the training parts,
atoms-first is worse than today's order by the total, orbitals-first the best.

| real / padded cells | today (by total) | atoms, orbitals | orbitals, atoms |
|---|---|---|---|
| PMechDB, budget 1.1M | 0.871 | 0.793 | **0.894** |
| ReactMech pilot, budget 1.1M | 0.909 | 0.858 | **0.930** |
| PMechDB, budget 4M | 0.801 | 0.722 | **0.817** |
| ReactMech pilot, budget 4M | 0.865 | 0.794 | **0.886** |

Today's packing counted a batch by its largest total size while the collate pads the atoms and
the orbitals apart: 25-29 batches per epoch held more cells than the budget (an OOM risk on a
full card); counted as the collate pays, none does.

## Measured (RX 6600 XT, fp16, budget 550k cells, the same pre-collated batches in every arm)

`train_epoch` over 400 PMechDB / 300 ReactMech-pilot training samples, the second pass (the
first compiles); scripts in the session's scratchpad, not kept.

| arm | PMechDB samples/s | ReactMech samples/s | peak memory (PMechDB) |
|---|---|---|---|
| A main, as trained | 24.2 | 13.2 | 2.3 GB |
| B branch, every option off (items 4, 6) | 24.4 | 13.3 | 2.4 GB |
| C + `--half-pair-norm --split-pair-init` | 24.7 | 13.3 | 1.9 GB |
| D C + `--compile` | 32.7 (+35%) | 17.6 (+33%) | 2.0 GB |
| E D + `--no-checkpoint` (deferred, item 1) | 39.2 (+62%) | 20.8 (+58%) | 4.7 GB |

With activation checkpointing on, the half LayerNorm and the split MLP_P save memory more than
time here (measured without checkpointing, the half LayerNorm alone gave +16%). Items 4 and 6
change little on this card, which is busy 84% of a step; they are aimed at a 5090, whose shorter
kernels leave the launch gaps a larger share. All of this wants a short run on the pod before
the long one.

## Verified

- Old checkpoints decode identically: `orb_a_s0` (dense), `flex_first` and
  `pmechdb+reactmech/sparse_resonance`, `ksmech-eval decode --part val --limit 150` with main's
  code and with this branch's, every flag at its default: the per-sample hits are equal on all
  150 samples of each.
- The compact samples expand to `unpack`'s arrays bit for bit on all 1,241 PMechDB val samples
  (cells 323 MB -> 108 MB), and the one-pass pair statistics over them equal the dense ones.
- The step's index sets from the collate give the loss bit for bit; `--split-pair-init` differs
  by 6e-7 relative in float32; `--half-pair-norm` is bit for bit outside autocast.
- Short GPU runs (fp16 with the scaler, fp32 with `found_inf`) with `--compile
  --half-pair-norm --split-pair-init --load-workers 4` train, validate and checkpoint.
- `tests/test_train_speed.py` (13 tests): the compact and streaming stores, the one-pass
  statistics, the host index sets, the split MLP_P, the half LayerNorm, the pair-update
  ablation, the padded-cost batches, the on-device skip of a non-finite batch, compile keeping
  the state dict, and the streaming build giving the same reports, views, samples and
  statistics as the whole build.

## Item 14: the largest samples (a measurement, nothing changed)

| | tokens median / p95 / max | cell share of the top 1% / 5% / 10% | `--max-tokens 512` drops / saves | 768 |
|---|---|---|---|---|
| PMechDB train (10,014) | 132 / 374 / 1,488 | 10.1% / 28.5% / 43.2% | 86 samples (0.9%) / 9.1% of the cells | 8 / 2.3% |
| ReactMech pilot train (7,191) | 244 / 410 / 571 | 3.7% / 14.5% / 25.1% | 37 (0.5%) / 2.0% | 0 / 0% |

On ReactMech a lower cap buys at most 2%; on PMechDB 512 tokens would save 9% of the cells for
0.9% of the samples. Lowering it is a data choice, not made here.

## How to run

```bash
# the full ReactMech store, written while it builds (one shard in memory)
ksmech-arrows samples --source reactmech --workers 8 --stream
# a run with every approved option
ksmech-train fit --run rm_dense_s0 --source reactmech --preset general \
    --compile --half-pair-norm --split-pair-init --load-workers 8
```

`--compact-samples` is on by default; `--no-compact-samples` keeps whole samples in memory.

## Ablations approved, to run after the reference ReactMech run

- Item 16, `--d-pair 32`: half the pair-track traffic (an option already).
- Item 18, `--pair-update-every 2`.

## Deferred

1 `--no-checkpoint` on the pod; 5 `pin_memory`; 7 fixed N buckets and CUDA graphs; 10 a fused
Triton kernel of the pair update; 11 attention without the float32 logits; 17 the pair track
without the occupied-occupied and virtual-virtual cells; 19 an atom-pair-only pair track.

## Found on the way (not changed)

- **The stop head is never trained.** `step_loss` reads `batch["stop_label"]`, but neither the
  sample nor `collate_samples` has one, so the stop BCE is always absent and `--stop-weight`
  (0.5 on ReactMech) has no effect: every ReactMech run so far trained END_MECH not at all. The
  pathways' stop labels exist (`pathways.parquet`, `is_last`, network plan item G) but are not
  wired into the samples.
