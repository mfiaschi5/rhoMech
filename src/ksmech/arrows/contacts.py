"""The legacy contact sidecar: the fixed-2.3 cap-and-roll kernel's pair features (`physics.pairs.
PAIR_FEATURES`: log beta^2 at three distances, KS3, KS2, KS1, CLASH, ...) and contact sites, per
step, kept apart from the stage-A samples (docs/decisions.md 2026-10-22).

It is built per split strategy for the val and test parts and for a fixed training subsample
(`TRAIN_SUBSAMPLE` steps), which is all the physics floor (`ksmech-physics ks-rank`) and the GBDT
baseline need; `with_legacy(sample, entry)` gives a sample whose pair arrays are the legacy ones,
so the ranking and baseline code read it unchanged. Layout under
`data/processed/<source>/samples/contacts_<strategy>/`: `shard_*.npz` (valid cells only, float16
values, bit-packed flags, int16 sites), `index.json` and `train_subsample.json`; every file
provenance-stamped.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np

from ksmech.arrows.reactants import CorrespondenceError, MissingTokens, ReactantTokens, TokenCache
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_training_steps, source_dir, steps_path
from ksmech.physics.pairs import PAIR_FEATURES, PairKernels, pair_block, valid_pairs
from ksmech.physics.params import REDUCED
from ksmech.processes import chunked, in_processes

log = logging.getLogger(__name__)

CONTACTS_VERSION = 1
SHARD = 256
TRAIN_SUBSAMPLE = 3000
CHUNK = 16  # consecutive steps per task of a parallel build


def contacts_dir(view_dir: Path, strategy: str) -> Path:
    return Path(view_dir) / "samples" / f"contacts_{strategy}"


def train_subsample(train_ids: list[str], n: int = TRAIN_SUBSAMPLE, seed: int = 0) -> list[str]:
    """`n` training step ids drawn without replacement with `seed`, sorted."""
    ids = sorted(set(train_ids))
    if len(ids) <= n:
        return ids
    pick = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
    return sorted(ids[i] for i in pick)


def legacy_entry(step, loader, kernels: PairKernels) -> dict[str, np.ndarray] | str:
    """The legacy arrays of one step (valid cells only), or a skip reason."""
    try:
        rt = ReactantTokens.from_smiles(step.reactants_smiles, loader)
    except MissingTokens as exc:
        return exc.reason
    except CorrespondenceError:
        return "CORRESPONDENCE"
    tok = [f.tokens for f in rt.fragments]
    block = pair_block(tok, kernels, REDUCED)  # at K = eps = 1, as the samples
    valid = valid_pairs(tok, rt.candidate)
    return {  # pair_block's sites are already global atoms
        "valid": valid,
        "z": block.z[valid].astype(np.float16),
        "defined": np.packbits(block.defined[valid], axis=-1),
        "sites": block.sites[valid].astype(np.int16),
    }


def _write_shard(path: Path, entries: dict[str, dict], inputs: dict, config: dict) -> None:
    used = dict(inputs)
    arrays = {}
    for sid, e in entries.items():
        used |= e["_inputs"]
        arrays |= {f"{sid}/{k}": v for k, v in e.items() if k != "_inputs"}
    header = prov.stamp("contact_shard", CONTACTS_VERSION, used, config, prov.array_digest(arrays))
    prov.save_npz(path, arrays, header)


def _entries(steps: list, loader, kernels: PairKernels) -> list[tuple[str, dict | str, dict]]:
    """(step id, legacy arrays or skip reason, the inputs read for it) of each step."""
    out = []
    for step in steps:
        with prov.collecting() as used:
            entry = legacy_entry(step, loader, kernels)
        out.append((step.step_id, entry, dict(used)))
    return out


def _entries_task(task: tuple) -> list[tuple[str, dict | str, dict]]:
    return _entries(*task)


def _all_entries(ordered: list, loader, kernels: PairKernels, n_workers: int) -> Iterator:
    """`_entries` of `ordered`, in order. With `n_workers` > 1, chunks of `CHUNK` consecutive
    steps run in that many spawned processes (one BLAS thread each), each with its own copy of
    `loader` (a `TokenCache` copy starts empty) and of `kernels`; contact blocks are written
    atomically, so two processes computing one block cannot corrupt it."""
    tasks = [(chunk, loader, kernels) for chunk in chunked(ordered, CHUNK)]
    for part in in_processes(_entries_task, tasks, n_workers):
        yield from part


def build_contacts(
    steps: list,
    loader,
    kernels: PairKernels,
    root: Path,
    inputs: dict[str, str],
    config: dict,
    n_workers: int = 1,
) -> dict:
    """Write the sidecar of `steps` under `root`; returns counts. `inputs`: the parquet and
    split file the step ids came from; every token file and contact block used is added per
    shard. The shards are the same for any `n_workers` (spawned processes)."""
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("shard_*.npz"):
        old.unlink()
    shards: dict[str, list[str]] = {}
    skipped: Counter = Counter()
    batch: dict[str, dict] = {}
    ordered = sorted(steps, key=lambda s: s.step_id)
    for k, (step_id, entry, used) in enumerate(_all_entries(ordered, loader, kernels, n_workers)):
        if isinstance(entry, str):
            skipped[entry] += 1
            log.info("%s: no contact entry (%s)", step_id, entry)
        else:
            batch[step_id] = entry | {"_inputs": used}
        if batch and (len(batch) == SHARD or k == len(ordered) - 1):
            name = f"shard_{len(shards):05d}.npz"
            _write_shard(root / name, batch, inputs, config)
            shards[name] = sorted(batch)
            batch = {}
    shard_inputs = {}
    for name in shards:
        shard_inputs |= prov.file_input(root / name)
    body = {"shards": shards, "skipped": dict(skipped)}
    digest = prov.sha256_text(json.dumps(body, sort_keys=True))
    header = prov.stamp("contact_index", CONTACTS_VERSION, shard_inputs, config, digest)
    prov.write_json(root / "index.json", body, header)
    n_entries = sum(map(len, shards.values()))
    return {"n_steps": len(ordered), "n_entries": n_entries, "skipped": dict(skipped)}


def build_all_contacts(
    source: str,
    kernels: PairKernels,
    strategy: str = "molecule",
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    loader=None,
    n_train: int = TRAIN_SUBSAMPLE,
    seed: int = 0,
    n_workers: int = 1,
) -> dict:
    """The sidecar of one split strategy: every val and test step and `n_train` training steps
    (`train_subsample`, recorded in `train_subsample.json`), over `n_workers` spawned
    processes. Needs the split file of `ksmech-arrows samples`."""
    from ksmech.arrows.steps import load_split_file, splits_path

    parquet = steps_path(source, processed_dir)
    split_file = splits_path(source, strategy, processed_dir)
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} missing; run `ksmech-arrows samples` first")
    parts = load_split_file(split_file)
    train = [i for name, ids in parts.items() if name.endswith("_train") for i in ids]
    held = {i for name, ids in parts.items() if not name.endswith("_train") for i in ids}
    sub = train_subsample(train, n_train, seed)
    by_id = {s.step_id: s for s in read_training_steps(parquet)}
    wanted = [by_id[i] for i in sorted(held | set(sub)) if i in by_id]
    root = contacts_dir(source_dir(source, processed_dir), strategy)
    inputs = {**prov.file_input(parquet), **prov.file_input(split_file)}
    config = {"strategy": strategy, "n_train": n_train, "seed": seed, **kernels.describe()}
    report = build_contacts(
        wanted, loader or TokenCache(), kernels, root, inputs, config, n_workers
    )
    body = {"ids": sub, "n_train": len(set(train)), "n": n_train, "seed": seed}
    digest = prov.sha256_text(json.dumps(body, sort_keys=True))
    header = prov.stamp(
        "train_subsample", CONTACTS_VERSION, prov.file_input(split_file), config, digest
    )
    prov.write_json(root / "train_subsample.json", body, header)
    return report | {"strategy": strategy, "n_train_subsample": len(sub), "n_held_out": len(held)}


def load_contacts(
    view_dir: Path, strategy: str, allow_stale: bool = False, ids: Iterable[str] | None = None
) -> dict[str, dict]:
    """step id -> dense legacy arrays (z float32, defined bool, sites int64) over the
    sample's (n_occ, n_virt) pair block, only for `ids` when given (a shard holding none of
    them is not opened). Verifies the index and every shard read; refuses dirty builds unless
    `allow_stale` (diagnostics). FileNotFoundError when it was never built."""
    root = contacts_dir(view_dir, strategy)
    if not (root / "index.json").exists():
        raise FileNotFoundError(f"{root / 'index.json'}: no contact sidecar for {strategy!r}")
    prov.verify_clean(
        root / "index.json", "contact_index", CONTACTS_VERSION, allow_stale=allow_stale
    )
    shards = json.loads((root / "index.json").read_text())["shards"]
    keep = None if ids is None else set(ids)
    out = {}
    n_feat = len(PAIR_FEATURES)
    for name, members in sorted(shards.items()):
        wanted = [sid for sid in members if keep is None or sid in keep]
        if not wanted:
            continue
        prov.verify_clean(root / name, "contact_shard", CONTACTS_VERSION, allow_stale=allow_stale)
        with np.load(root / name, allow_pickle=False) as z:
            for sid in wanted:
                valid = z[f"{sid}/valid"].astype(bool)
                dense_z = np.zeros((*valid.shape, n_feat), dtype=np.float32)
                dense_d = np.zeros((*valid.shape, n_feat), dtype=bool)
                sites = np.full((*valid.shape, 2), -1, dtype=np.int64)
                dense_z[valid] = z[f"{sid}/z"].astype(np.float32)
                bits = np.unpackbits(z[f"{sid}/defined"], axis=-1, count=n_feat)
                dense_d[valid] = bits.astype(bool)
                sites[valid] = z[f"{sid}/sites"]
                out[sid] = {"valid": valid, "z": dense_z, "defined": dense_d, "sites": sites}
    return out


def with_legacy(sample, entry: dict):
    """The sample with its pair arrays replaced by the legacy contact ones (the layout of
    `physics.pairs.PAIR_FEATURES`), for the floor and the GBDT."""
    if entry["valid"].shape != sample.pair_valid.shape:
        raise ValueError(f"{sample.step_id}: contact block {entry['valid'].shape} does not match")
    return replace(
        sample,
        pair_z=entry["z"],
        pair_defined=entry["defined"],
        pair_contact_sites=entry["sites"],
        pair_valid=sample.pair_valid & entry["valid"],
        pair_features=PAIR_FEATURES,
    )
