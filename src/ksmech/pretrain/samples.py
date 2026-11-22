"""The pretraining samples: one QMugs molecule each, the model inputs of a one-molecule pool
(`arrows.steps.reactant_sample` and its Lewis structure, the very code of the mechanism
samples) and the donor-acceptor energy of every (occupied, virtual) cell as the target.

The molecules are a seeded random subset of the QMugs records (`select_species`), written as
a species file that `ksmech-orbitals build --species-from` reads for the tokens. Each molecule
falls into the training or the validation part by a hash of its key (`part_of`), so a larger
subset keeps the parts of a smaller one. The store has the layout and provenance of a
mechanism store: shards and an index under `<out_dir>/samples/`, and one view per part
(`samples_pretrain_<part>.json`). Every molecule without a sample is counted by its reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ksmech.arrows import store
from ksmech.arrows.lewis import MissingBondIndex, reactant_lewis
from ksmech.arrows.reactants import CorrespondenceError, MissingTokens, ReactantTokens, TokenCache
from ksmech.arrows.steps import SAMPLE_VERSION, MechanismSample, reactant_sample, with_lewis
from ksmech.data import provenance as prov
from ksmech.data import qmugs
from ksmech.physics.donor_acceptor import (
    DEGENERATE_BASIS,
    OPEN_SHELL,
    ClosedShellOnly,
    DegenerateBasis,
    donor_acceptor,
)
from ksmech.processes import chunked, in_processes
from ksmech.qm.xtb_calc import QMConfig

log = logging.getLogger(__name__)

SOURCE = "qmugs"
OUT_DIR = qmugs.PROCESSED_DIR / "pretrain"
TOKENS_DIR = Path("data/cache/qmugs/tokens")
BOND_INDEX_DIR = Path("data/cache/qmugs/bond_index")
METHOD = QMConfig(given_geometry=True).method  # GFN2-xTB/ALPB-water/given
TARGET = "donor_acceptor_e2"  # the target's name and version, in every provenance config
TARGET_VERSION = 1
VAL_PERCENT = 5  # of the molecules, by hash (`part_of`)
PARTS = ("train", "val")
CHUNK = 32
MULTI_FRAGMENT = "MULTI_FRAGMENT"  # reason codes beside the loader's and `donor_acceptor`'s
CORRESPONDENCE = "CORRESPONDENCE"
NO_BOND_INDEX = "NO_BOND_INDEX"


@dataclass
class PretrainSample(MechanismSample):
    """A mechanism sample of a one-molecule pool (no label) with its pretraining target."""

    # (n_occ, n_virt) float32 E2 in eV on the cells of pair_z, NaN where it is undefined
    pair_target: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


def qmugs_loader() -> TokenCache:
    """The token cache of the QMugs records: their own tokens, Wiberg sidecars and method."""
    return TokenCache(TOKENS_DIR, method=METHOD, bond_index_dir=BOND_INDEX_DIR)


def sample_id(key: str) -> str:
    return "qmugs-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def part_of(key: str) -> str:
    """'val' for VAL_PERCENT of the keys by their hash, else 'train' (no seed: fixed)."""
    h = int(hashlib.sha256(("part:" + key).encode()).hexdigest()[:8], 16)
    return "val" if h % 100 < VAL_PERCENT else "train"


def select_species(manifest: Path, n: int, seed: int) -> list[str]:
    """`n` keys drawn at random with `seed` from the distinct canonical SMILES of the QMugs
    records that are `ok` in `manifest`, sorted."""
    rows = pq.read_table(manifest, columns=["smiles", "status"]).to_pylist()
    keys = sorted({r["smiles"] for r in rows if r["status"] == "ok"})
    if n > len(keys):
        raise ValueError(f"{n} species asked, {len(keys)} QMugs records are ok")
    return sorted(random.Random(seed).sample(keys, n))


def write_species_file(path: Path, manifest: Path, n: int, seed: int) -> Path:
    """The species file of `select_species` (`{"species": [...]}`, read by `ksmech-orbitals
    build --species-from`), with the manifest it was drawn from as its provenance input."""
    body = {"species": select_species(manifest, n, seed), "n": n, "seed": seed}
    header = prov.stamp(
        "pretrain_species",
        TARGET_VERSION,
        prov.file_input(manifest),
        {"n": n, "seed": seed},
        prov.sha256_text(json.dumps(body, sort_keys=True)),
    )
    return prov.write_json(path, body, header)


def _target(rt: ReactantTokens) -> np.ndarray | str:
    """The E2 array of the pool's one molecule (NaN where undefined), or the reason it has none."""
    try:
        da = donor_acceptor(rt.fragments[0].tokens)
    except ClosedShellOnly:
        return OPEN_SHELL
    except DegenerateBasis:
        return DEGENERATE_BASIS
    return np.where(da.defined, da.e2, np.nan).astype(np.float32)


def pretrain_sample(key: str, loader) -> PretrainSample | str:
    """The pretraining sample of one molecule, or the reason code it has none."""
    with prov.collecting() as used:
        try:
            rt = ReactantTokens.from_smiles(key, loader)
        except MissingTokens as exc:
            return next(iter(exc.reasons.values()))
        except CorrespondenceError:
            return CORRESPONDENCE
        if len(rt.fragments) != 1:
            return MULTI_FRAGMENT
        target = _target(rt)
        if isinstance(target, str):
            return target
        try:
            lewis = reactant_lewis(rt)
        except MissingBondIndex:
            return NO_BOND_INDEX
        base = with_lewis(reactant_sample(sample_id(key), SOURCE, rt), lewis)
    if target.shape != base.pair_valid.shape:
        raise AssertionError(f"{key}: target {target.shape} against cells {base.pair_valid.shape}")
    sample = PretrainSample(**{f.name: getattr(base, f.name) for f in fields(base)})
    sample.pair_target, sample.inputs = target, dict(sorted(used.items()))
    return sample


def _chunk_worker(task: tuple) -> tuple[dict, dict]:
    keys, loader = task
    samples, skipped = {}, {}
    for key in keys:
        out = pretrain_sample(key, loader)
        if isinstance(out, str):
            skipped[sample_id(key)] = out
        else:
            samples[out.step_id] = out
    return samples, skipped


def store_config() -> dict:
    return {"pretrain_target": TARGET, "target_version": TARGET_VERSION, "source": SOURCE}


def build_store(species_file: Path, out_dir: Path, loader, n_workers: int = 1) -> dict:
    """Every listed molecule's sample into the store (streamed, one shard held at a time), then
    one view per part; returns the report (also `pretrain_report.json`)."""
    keys = json.loads(Path(species_file).read_text())["species"]
    writer = store.StoreWriter(out_dir, species_file, SAMPLE_VERSION, store_config())
    ids: dict[str, list[str]] = {p: [] for p in PARTS}
    skipped: dict[str, str] = {}
    part = {sample_id(k): part_of(k) for k in keys}
    tasks = [(chunk, loader) for chunk in chunked(keys, CHUNK)]
    for done, (samples, skips) in enumerate(in_processes(_chunk_worker, tasks, n_workers), 1):
        for sid, sample in samples.items():
            writer.add(sample)
            ids[part[sid]].append(sid)
        skipped |= skips
        if done % 50 == 0 or done == len(tasks):
            log.info("pretrain samples: %d / %d chunks, %d skipped", done, len(tasks), len(skipped))
    index = writer.close(skipped)
    return _write_views(out_dir, species_file, index, ids, keys, skipped)


def _write_views(out_dir: Path, species_file: Path, index: Path, ids, keys, skipped) -> dict:
    config = {"sample_version": SAMPLE_VERSION, "strategy": "pretrain"} | store_config()
    report = {
        "n_species": len(keys),
        "n_samples": sum(len(v) for v in ids.values()),
        "parts": {p: len(v) for p, v in ids.items()},
        "skipped": dict(Counter(skipped.values()).most_common()),
    }
    for p, members in ids.items():
        path = view_path(out_dir, p)
        store.write_view(path, sorted(members), report | {"part": p}, index, species_file, config)
    (Path(out_dir) / "pretrain_report.json").write_text(json.dumps(report, indent=2))
    return report


def view_path(out_dir: Path, part: str) -> Path:
    return Path(out_dir) / f"samples_pretrain_{part}.json"


def load_part(
    out_dir: Path, part: str, compact: bool = True, workers: int = 1, deflate: bool = False
) -> list:
    """The `PretrainSample`s of one part (light, with compact cells, unless `compact=False`)."""
    return store.load_view(
        view_path(out_dir, part),
        PretrainSample,
        SAMPLE_VERSION,
        compact=compact,
        workers=workers,
        deflate=deflate and compact,
    )
