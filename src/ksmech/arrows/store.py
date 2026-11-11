"""The sample store: every step's sample once, in compressed shards, plus one small view per split.

Layout under `data/processed/<source>/samples/`:
- `shard_00000.npz` ... : `SHARD_SIZE` samples each, members `<i>/<field>` and a JSON `meta`
  member (ids, labels, keys), with a provenance header whose inputs are the steps parquet and
  every token file its samples used;
- `index.json`: step id -> shard, with the shards as provenance inputs.
A split is `samples_<split>.json` next to the store: its ids and counts, with the index and the
split file as inputs. Nothing is copied per split.

On disk a sample keeps its pair arrays on every (occupied, virtual) cell (since v18; v17 kept
the `pair_valid` cells only, ~89%), flattened cell by cell: `pair_z` as float16, the 0/1
features (`BINARY_FEATURES` among the sample's `pair_features`, the exclusion flags of
`physics.pairs.pair_flags` among them) and `pair_defined` bit-packed; `pair_valid` is stored as
the legacy mask. Its resonance drawings are stacked (`drawing_charges`, `drawing_orders`,
`drawing_lone_pairs`, and since v19 `drawing_nonbonding` and `drawing_unpaired`, one row per
drawing) with their labels in the JSON. In memory every array is dense float32 again.
float16 keeps 3 significant digits: the largest error measured on the pair features was 5e-3
on log-features spanning ~30 units, below 1e-7 absolute on the tiny ones
(docs/decisions.md, fix item 8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ksmech.data import provenance as prov

if TYPE_CHECKING:
    from ksmech.arrows.steps import MechanismSample

SHARD_SIZE = 256
CELL_ARRAYS = ("pair_z", "pair_defined", "pair_gap_sign")  # stored flat, one row per cell
# Drawing field -> the stacked array of a sample's drawings
DRAWING_ARRAYS = {
    "charges": "drawing_charges",
    "orders": "drawing_orders",
    "lone_pairs": "drawing_lone_pairs",
    "nonbonding": "drawing_nonbonding",
    "unpaired": "drawing_unpaired",
}
# exact 0/1 pair features: bits, not float16 (`physics.pairs.PAIR_FLAGS` and, v19, somo_pair)
BINARY_FEATURES = (
    "inter",
    "clash",
    "bonded",
    "d_acceptor",
    "own_antibond",
    "site_self",
    "somo_pair",  # v19 (open-shell plan P7)
)


def _binary_columns(names) -> list[int]:
    return [k for k, f in enumerate(names) if f in BINARY_FEATURES]


def store_dir(view_dir: Path) -> Path:
    return Path(view_dir) / "samples"


def pack(sample: MechanismSample) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """(arrays as stored, the JSON-able rest) of one MechanismSample."""
    arrays, meta = {}, {}
    for f in fields(sample):
        value = getattr(sample, f.name)
        if isinstance(value, np.ndarray):
            arrays[f.name] = value
        else:
            if f.name == "drawings":
                arrays |= _pack_drawings(sample)
                value = [d.edit_label.to_dict() for d in value]
            elif f.name in ("labels", "labels_all"):
                value = [lab.to_dict() for lab in value]
            elif f.name == "edit_label" and value is not None:
                value = value.to_dict()
            meta[f.name] = value
    n_cells = sample.pair_valid.size
    for name in CELL_ARRAYS:
        arrays[name] = arrays[name].reshape(n_cells, *arrays[name].shape[2:])
    binary = _binary_columns(sample.pair_features)
    flags = arrays["pair_z"][:, binary]
    if not np.isin(flags, (0.0, 1.0)).all():
        raise ValueError(f"{sample.step_id}: a binary pair feature is not 0/1")
    arrays["pair_flags"] = np.packbits(flags.astype(bool), axis=-1)
    arrays["pair_z"] = _byte_planes(np.delete(arrays["pair_z"], binary, axis=-1))
    arrays["pair_defined"] = np.packbits(arrays["pair_defined"], axis=-1)
    meta["n_pair_features"] = int(sample.pair_z.shape[-1])
    return arrays, meta


PACKED_CELLS = (*CELL_ARRAYS, "pair_valid", "pair_flags")  # the packed arrays of the cells


class CompactCells:
    """A sample's cell arrays as the shard stores them (float16 byte planes, bit-packed flags
    and defined masks: ~32 B per cell instead of ~102 dense), expanded by `dense` with the very
    code `unpack` runs, so a sample materialized later is bit for bit the one `unpack` returns.
    The trainer keeps these in memory (`load_view(compact=True)`) and expands one batch at a
    time in its DataLoader workers."""

    __slots__ = ("arrays", "n_feat", "pair_features")

    def __init__(self, arrays: dict[str, np.ndarray], n_feat: int, pair_features):
        self.arrays = {k: arrays[k] for k in PACKED_CELLS}
        self.n_feat, self.pair_features = int(n_feat), tuple(pair_features)

    def dense(self) -> dict[str, np.ndarray]:
        """pair_z, pair_defined, pair_gap_sign and pair_valid, dense."""
        arrays, n_feat = self.arrays, self.n_feat
        shape = arrays["pair_valid"].shape
        binary = _binary_columns(self.pair_features)
        rest = [k for k in range(n_feat) if k not in binary]
        z = np.zeros((shape[0] * shape[1], n_feat), dtype=np.float32)
        z[:, rest] = _from_byte_planes(arrays["pair_z"]).astype(np.float32)
        z[:, binary] = np.unpackbits(arrays["pair_flags"], axis=-1, count=len(binary))
        bits = np.unpackbits(arrays["pair_defined"], axis=-1, count=n_feat)
        return {
            "pair_z": z.reshape(*shape, n_feat),
            "pair_defined": bits.astype(bool).reshape(*shape, n_feat),
            "pair_gap_sign": arrays["pair_gap_sign"].reshape(shape),
            "pair_valid": arrays["pair_valid"].astype(bool),
        }

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())


_ABSENT = object()


def _no_cells(n_feat: int) -> dict[str, np.ndarray]:
    """Zero-size cell arrays: a light sample's (a reader that needs the cells fails loudly on
    their shape instead of reading wrong values)."""
    return {
        "pair_z": np.zeros((0, 0, n_feat), dtype=np.float32),
        "pair_defined": np.zeros((0, 0, n_feat), dtype=bool),
        "pair_gap_sign": np.zeros((0, 0), dtype=np.int8),
        "pair_valid": np.zeros((0, 0), dtype=bool),
    }


def light(sample: MechanismSample, cells: CompactCells | None = None) -> MechanismSample:
    """The sample without its cell arrays (what the store reports and the splits read), with
    `cells` attached for `dense` when given."""
    from dataclasses import replace

    out = replace(sample, **_no_cells(sample.pair_z.shape[-1] if cells is None else cells.n_feat))
    out._cells = cells
    return out


def is_light(sample: MechanismSample) -> bool:
    """Whether the sample was stripped of its cells (`light`), compact cells or none."""
    return getattr(sample, "_cells", _ABSENT) is not _ABSENT


def dense(sample: MechanismSample) -> MechanismSample:
    """A light sample with compact cells made whole again (`CompactCells.dense`); any other
    sample as it is. A light sample without cells is refused."""
    from dataclasses import replace

    cells = getattr(sample, "_cells", _ABSENT)
    if cells is _ABSENT:
        return sample
    if cells is None:
        raise ValueError(f"{sample.step_id}: a light sample without its cells")
    return replace(sample, **cells.dense())


def unpack(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    cls: type[MechanismSample],
    compact: bool = False,
) -> MechanismSample:
    """The dense MechanismSample of `pack`'s output (cls: MechanismSample); with `compact` a
    light one carrying its `CompactCells` (`dense` expands it, bit for bit the same)."""
    from ksmech.arrows.align import ArrowLabel
    from ksmech.arrows.delta_be import EditLabel
    from ksmech.arrows.drawings import Drawing

    meta = dict(meta)
    n_feat = meta.pop("n_pair_features")
    for name in ("pair_features", "atom_columns", "orbital_columns"):
        if name in meta:
            meta[name] = tuple(meta[name])
    cells = CompactCells(arrays, n_feat, meta["pair_features"])
    dense_cells = _no_cells(n_feat) if compact else cells.dense()
    skip = (*PACKED_CELLS, *DRAWING_ARRAYS.values())
    out = {k: v for k, v in arrays.items() if k not in skip}
    n_drawn = len(arrays["drawing_orders"])
    for name in ("drawing_nonbonding", "drawing_unpaired"):  # a v18 shard's drawings: no N, u
        arrays.setdefault(name, np.zeros((n_drawn, len(arrays["atom_map"])), dtype=np.int8))
    rows = [
        {f: arrays[name][k] for f, name in DRAWING_ARRAYS.items()}
        for k in range(len(arrays["drawing_orders"]))
    ]
    meta["drawings"] = [
        Drawing(**row, edit_label=EditLabel.from_dict(lab))
        for row, lab in zip(rows, meta["drawings"], strict=True)
    ]
    meta["labels"] = [ArrowLabel.from_dict(x) for x in meta["labels"]]
    meta["labels_all"] = [ArrowLabel.from_dict(x) for x in meta.get("labels_all", [])]
    if meta.get("edit_label") is not None:
        meta["edit_label"] = EditLabel.from_dict(meta["edit_label"])
    sample = cls(**out, **dense_cells, **meta)
    if compact:
        sample._cells = cells
    return sample


def _pack_drawings(sample: MechanismSample) -> dict[str, np.ndarray]:
    """The drawings' arrays stacked, one row per drawing (zero rows without drawings)."""
    width = {f: sample.n_atoms for f in DRAWING_ARRAYS} | {"orders": len(sample.bonds)}
    out = {}
    for f, name in DRAWING_ARRAYS.items():
        rows = [getattr(d, f) for d in sample.drawings]
        # (rows, width), not (-1, width): a pool of single atoms has no bond row to stack
        out[name] = np.array(rows, dtype=np.int8).reshape(len(rows), width[f])
    return out


def _byte_planes(z: np.ndarray) -> np.ndarray:
    """(2, n_features, n_cells) uint8: the float16 values feature by feature, high and low
    bytes apart. Lossless; the exponent plane compresses far better (shards: -21%)."""
    zt = np.ascontiguousarray(z.astype(np.float16).T)
    return zt.view(np.uint8).reshape(*zt.shape, 2).transpose(2, 0, 1).copy()


def _from_byte_planes(planes: np.ndarray) -> np.ndarray:
    """(n_cells, n_features) float16 from `_byte_planes`."""
    zt = np.ascontiguousarray(planes.transpose(1, 2, 0)).view(np.float16)[..., 0]
    return zt.T


def _write_shard(path: Path, samples: list, inputs: dict, version: int, extra: dict) -> str:
    arrays, metas = {}, []
    used = dict(inputs)
    for i, s in enumerate(samples):
        a, m = pack(s)
        arrays |= {f"{i}/{k}": v for k, v in a.items()}
        metas.append(m)
        used |= s.inputs
    arrays["meta"] = np.array(json.dumps(metas))
    config = {"sample_version": version} | extra
    header = prov.stamp("sample_shard", version, used, config, prov.array_digest(arrays))
    prov.save_npz(path, arrays, header)
    return header.content_sha256


class StoreWriter:
    """The store written as samples arrive (the streaming build): a shard every `SHARD_SIZE`
    samples, in the order they are added, then the index at `close`. Only one shard's samples
    are held. `write_store` is this writer over the samples sorted by step id."""

    def __init__(self, view_dir: Path, parquet: Path, version: int, config: dict | None = None):
        self.root, self.version = store_dir(view_dir), version
        self.extra, self.inputs = dict(config or {}), prov.file_input(parquet)
        self.root.mkdir(parents=True, exist_ok=True)
        for old in self.root.glob("shard_*.npz"):
            old.unlink()  # a rebuild replaces the whole store; the index names the shards it wrote
        self.shards: dict[str, list[str]] = {}
        self.pending: list = []

    def add(self, sample: MechanismSample) -> None:
        self.pending.append(sample)
        if len(self.pending) == SHARD_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self.pending:
            return
        name = f"shard_{len(self.shards):05d}.npz"
        _write_shard(self.root / name, self.pending, self.inputs, self.version, self.extra)
        self.shards[name] = [s.step_id for s in self.pending]
        self.pending = []

    def close(self, skipped: dict) -> Path:
        """Write the last shard and the index; returns the index path."""
        self._flush()
        inputs = {}
        for name in self.shards:
            inputs |= prov.file_input(self.root / name)
        body = {"shards": self.shards, "skipped": {i: str(r) for i, r in sorted(skipped.items())}}
        header = prov.stamp(
            "sample_index",
            self.version,
            inputs,
            {"sample_version": self.version, "shard_size": SHARD_SIZE} | self.extra,
            prov.sha256_text(json.dumps(body, sort_keys=True)),
        )
        return prov.write_json(self.root / "index.json", body, header)


def write_store(
    view_dir: Path,
    samples: dict[str, MechanismSample],
    skipped: dict,
    parquet: Path,
    version: int,
    config: dict | None = None,
) -> Path:
    """Write every sample once (sorted by step id) into shards and the index. Returns the
    index path. `config` joins the provenance config of every shard and of the index (the
    backward samples' direction, `arrows.backward.direction_config`)."""
    writer = StoreWriter(view_dir, parquet, version, config)
    for i in sorted(samples):
        writer.add(samples[i])
    return writer.close(skipped)


def write_view(
    path: Path,
    ids: list[str],
    report: dict[str, Any],
    index: Path,
    split_file: Path,
    config: dict[str, Any],
) -> Path:
    """`samples_<split>.json`: the split's ids and counts over the store."""
    body = {"ids": ids, "report": report}
    header = prov.stamp(
        "samples",
        config["sample_version"],
        {**prov.file_input(index), **prov.file_input(split_file)},
        config,
        prov.sha256_text(json.dumps(body, sort_keys=True)),
    )
    return prov.write_json(path, body, header)


def _load_shard(
    path: Path, members: list[int], cls, version: int, allow_stale: bool, compact: bool = False
):
    prov.verify_clean(path, "sample_shard", version, allow_stale=allow_stale)
    with np.load(path, allow_pickle=False) as z:
        metas = json.loads(str(z["meta"]))
        names: dict[str, list[str]] = {}
        for n in z.files:
            if "/" in n:
                names.setdefault(n.split("/", 1)[0], []).append(n)
        out = []
        for i in members:
            arrays = {n.split("/", 1)[1]: z[n] for n in names[str(i)]}
            out.append(unpack(arrays, metas[i], cls, compact))
    return out


def _load_task(task):
    return _load_shard(*task)


def _view_members(
    path: Path, version: int, allow_stale: bool, ids: Iterable[str] | None
) -> tuple[list[str], dict[str, list[int]]]:
    """(the view's stored ids, kept to `ids` when given, in view order; shard -> members).
    Verifies the view, its split file and the index."""
    header = prov.verify_clean(path, "samples", version, allow_stale=allow_stale)
    for name in header.inputs if header else ():
        if name.startswith("file:") and prov.input_path(name).name.startswith("splits_"):
            split = prov.input_path(name)  # verified too: its parquet
            prov.verify_clean(split, "split", version, allow_stale=allow_stale)
    view = json.loads(Path(path).read_text())
    index_path = store_dir(Path(path).parent) / "index.json"
    prov.verify_clean(index_path, "sample_index", version, allow_stale=allow_stale)
    shards = json.loads(index_path.read_text())["shards"]
    where = {sid: (name, k) for name, ids_ in shards.items() for k, sid in enumerate(ids_)}
    keep = None if ids is None else set(ids)
    wanted = [sid for sid in view["ids"] if sid in where and (keep is None or sid in keep)]
    by_shard: dict[str, list[int]] = {}
    for sid in wanted:
        name, k = where[sid]
        by_shard.setdefault(name, []).append(k)
    return wanted, by_shard


def view_ids(path: Path, version: int, allow_stale: bool = False) -> list[str]:
    """The ids of one split view that have a stored sample, in view order (verified)."""
    return _view_members(path, version, allow_stale, None)[0]


def iter_view(
    path: Path,
    cls: type[MechanismSample],
    version: int,
    allow_stale: bool = False,
    ids: Iterable[str] | None = None,
    compact: bool = False,
    workers: int = 1,
) -> Iterator[MechanismSample]:
    """The samples of one split view (only `ids` when given), one shard at a time in shard
    order, so a caller that reduces each sample never holds the whole view. Verifies the view,
    the index and every shard read: StaleArtifact names what changed. `compact`: light samples
    with their `CompactCells` (`dense`); `workers` > 1 reads and verifies the shards in that
    many forked processes, the order kept."""
    _, by_shard = _view_members(path, version, allow_stale, ids)
    yield from _iter_shards(path, by_shard, cls, version, allow_stale, compact, workers)


def _iter_shards(
    path: Path, by_shard: dict, cls, version: int, allow_stale, compact=False, workers: int = 1
):
    root = store_dir(Path(path).parent)
    tasks = [
        (root / name, members, cls, version, allow_stale, compact)
        for name, members in sorted(by_shard.items())
    ]
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield from _load_task(task)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("fork")  # numpy only in the children; no CUDA call
    with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=ctx) as pool:
        for shard in pool.map(_load_task, tasks):
            yield from shard


def load_store(
    view_dir: Path, cls: type[MechanismSample], version: int, cells: str = "dense"
) -> tuple[dict[str, MechanismSample], dict[str, str], Path]:
    """(every stored sample by step id, the skipped steps' reason codes, the index path),
    verified like a view: the index and every shard. `cells`: 'dense' (whole samples),
    'compact' (light, with their `CompactCells`) or 'none' (light, no cells: the reports and the
    splits; `is_light`)."""
    if cells not in ("dense", "compact", "none"):
        raise ValueError(f"cells {cells!r}: dense, compact or none")
    index_path = store_dir(view_dir) / "index.json"
    prov.verify_clean(index_path, "sample_index", version)
    body = json.loads(index_path.read_text())
    root, samples = store_dir(view_dir), {}
    for name, ids in sorted(body["shards"].items()):
        for s in _load_shard(
            root / name, list(range(len(ids))), cls, version, False, cells != "dense"
        ):
            if cells == "none":
                s._cells = None
            samples[s.step_id] = s
    return samples, body["skipped"], index_path


def load_view(
    path: Path,
    cls: type[MechanismSample],
    version: int,
    allow_stale: bool = False,
    ids: Iterable[str] | None = None,
    compact: bool = False,
    workers: int = 1,
) -> list[MechanismSample]:
    """The samples of one split view (only `ids` when given), in its id order (skipped steps
    have none). Verifies as `iter_view`; `compact` and `workers` as there."""
    wanted, by_shard = _view_members(path, version, allow_stale, ids)
    shards = _iter_shards(path, by_shard, cls, version, allow_stale, compact, workers)
    loaded = {s.step_id: s for s in shards}
    return [loaded[sid] for sid in wanted]
