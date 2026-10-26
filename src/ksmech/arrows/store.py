"""The sample store: every step's sample once, in compressed shards, plus one small view per split.

Layout under `data/processed/<source>/samples/`:
- `shard_00000.npz` ... : `SHARD_SIZE` samples each, members `<i>/<field>` and a JSON `meta`
  member (ids, labels, keys), with a provenance header whose inputs are the steps parquet and
  every token file its samples used;
- `index.json`: step id -> shard, with the shards as provenance inputs.
A split is `samples_<split>.json` next to the store: its ids and counts, with the index and the
split file as inputs. Nothing is copied per split.

On disk a sample keeps its pair arrays for the valid cells only (`pair_valid`, ~89% of the
cells): `pair_z` as float16, the 0/1 features (`BINARY_FEATURES` among the sample's
`pair_features`) and `pair_defined` bit-packed. Its resonance drawings are stacked
(`drawing_charges`, `drawing_orders`, `drawing_lone_pairs`, one row per drawing) with their
labels in the JSON. In memory every array is dense float32 again, invalid cells zero /
undefined.
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
VALID_ONLY = ("pair_z", "pair_defined", "pair_gap_sign")
# Drawing field -> the stacked array of a sample's drawings
DRAWING_ARRAYS = {
    "charges": "drawing_charges",
    "orders": "drawing_orders",
    "lone_pairs": "drawing_lone_pairs",
}
BINARY_FEATURES = ("inter", "clash", "bonded")  # exact 0/1 pair features: bits, not float16


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
            elif f.name == "labels":
                value = [lab.to_dict() for lab in value]
            elif f.name == "edit_label" and value is not None:
                value = value.to_dict()
            meta[f.name] = value
    valid = sample.pair_valid
    for name in VALID_ONLY:
        arrays[name] = arrays[name][valid]
    binary = _binary_columns(sample.pair_features)
    flags = arrays["pair_z"][:, binary]
    if not np.isin(flags, (0.0, 1.0)).all():
        raise ValueError(f"{sample.step_id}: a binary pair feature is not 0/1")
    arrays["pair_flags"] = np.packbits(flags.astype(bool), axis=-1)
    arrays["pair_z"] = _byte_planes(np.delete(arrays["pair_z"], binary, axis=-1))
    arrays["pair_defined"] = np.packbits(arrays["pair_defined"], axis=-1)
    meta["n_pair_features"] = int(sample.pair_z.shape[-1])
    return arrays, meta


def unpack(
    arrays: dict[str, np.ndarray], meta: dict[str, Any], cls: type[MechanismSample]
) -> MechanismSample:
    """The dense MechanismSample of `pack`'s output (cls: MechanismSample)."""
    from ksmech.arrows.align import ArrowLabel
    from ksmech.arrows.delta_be import EditLabel
    from ksmech.arrows.drawings import Drawing

    meta = dict(meta)
    n_feat = meta.pop("n_pair_features")
    meta["pair_features"] = tuple(meta["pair_features"])
    valid = arrays["pair_valid"].astype(bool)
    shape = valid.shape
    dense = {
        "pair_z": np.zeros((*shape, n_feat), dtype=np.float32),
        "pair_defined": np.zeros((*shape, n_feat), dtype=bool),
        "pair_gap_sign": np.zeros(shape, dtype=np.int8),
    }
    binary = _binary_columns(meta["pair_features"])
    rest = [k for k in range(n_feat) if k not in binary]
    z = np.zeros((int(valid.sum()), n_feat), dtype=np.float32)
    z[:, rest] = _from_byte_planes(arrays["pair_z"]).astype(np.float32)
    z[:, binary] = np.unpackbits(arrays["pair_flags"], axis=-1, count=len(binary))
    dense["pair_z"][valid] = z
    bits = np.unpackbits(arrays["pair_defined"], axis=-1, count=n_feat)
    dense["pair_defined"][valid] = bits.astype(bool)
    dense["pair_gap_sign"][valid] = arrays["pair_gap_sign"]
    skip = (*VALID_ONLY, "pair_flags", *DRAWING_ARRAYS.values())
    out = {k: v for k, v in arrays.items() if k not in skip}
    rows = [
        {f: arrays[name][k] for f, name in DRAWING_ARRAYS.items()}
        for k in range(len(arrays["drawing_orders"]))
    ]
    meta["drawings"] = [
        Drawing(**row, edit_label=EditLabel.from_dict(lab))
        for row, lab in zip(rows, meta["drawings"], strict=True)
    ]
    meta["labels"] = [ArrowLabel.from_dict(x) for x in meta["labels"]]
    if meta.get("edit_label") is not None:
        meta["edit_label"] = EditLabel.from_dict(meta["edit_label"])
    return cls(**out, **dense, **meta)


def _pack_drawings(sample: MechanismSample) -> dict[str, np.ndarray]:
    """The drawings' arrays stacked, one row per drawing (zero rows without drawings)."""
    width = {"charges": sample.n_atoms, "orders": len(sample.bonds), "lone_pairs": sample.n_atoms}
    out = {}
    for f, name in DRAWING_ARRAYS.items():
        rows = [getattr(d, f) for d in sample.drawings]
        out[name] = np.array(rows, dtype=np.int8).reshape(-1, width[f])
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


def _write_shard(path: Path, samples: list, inputs: dict, version: int) -> str:
    arrays, metas = {}, []
    used = dict(inputs)
    for i, s in enumerate(samples):
        a, m = pack(s)
        arrays |= {f"{i}/{k}": v for k, v in a.items()}
        metas.append(m)
        used |= s.inputs
    arrays["meta"] = np.array(json.dumps(metas))
    header = prov.stamp(
        "sample_shard", version, used, {"sample_version": version}, prov.array_digest(arrays)
    )
    prov.save_npz(path, arrays, header)
    return header.content_sha256


def write_store(
    view_dir: Path, samples: dict[str, MechanismSample], skipped: dict, parquet: Path, version: int
) -> Path:
    """Write every sample once (sorted by step id) into shards and the index. Returns the
    index path."""
    root = store_dir(view_dir)
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("shard_*.npz"):
        old.unlink()  # a rebuild replaces the whole store; the index names the shards it wrote
    ids = sorted(samples)
    shards: dict[str, list[str]] = {}
    for k in range(0, len(ids), SHARD_SIZE):
        name = f"shard_{k // SHARD_SIZE:05d}.npz"
        chunk = ids[k : k + SHARD_SIZE]
        _write_shard(root / name, [samples[i] for i in chunk], prov.file_input(parquet), version)
        shards[name] = chunk
    inputs = {}
    for name in shards:
        inputs |= prov.file_input(root / name)
    body = {"shards": shards, "skipped": {i: str(r) for i, r in sorted(skipped.items())}}
    header = prov.stamp(
        "sample_index",
        version,
        inputs,
        {"sample_version": version, "shard_size": SHARD_SIZE},
        prov.sha256_text(json.dumps(body, sort_keys=True)),
    )
    return prov.write_json(root / "index.json", body, header)


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


def _load_shard(path: Path, members: list[int], cls, version: int, allow_stale: bool):
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
            out.append(unpack(arrays, metas[i], cls))
    return out


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
) -> Iterator[MechanismSample]:
    """The samples of one split view (only `ids` when given), one shard at a time in shard
    order, so a caller that reduces each sample never holds the whole view. Verifies the view,
    the index and every shard read: StaleArtifact names what changed."""
    _, by_shard = _view_members(path, version, allow_stale, ids)
    yield from _iter_shards(path, by_shard, cls, version, allow_stale)


def _iter_shards(path: Path, by_shard: dict[str, list[int]], cls, version: int, allow_stale):
    root = store_dir(Path(path).parent)
    for name, members in sorted(by_shard.items()):
        yield from _load_shard(root / name, members, cls, version, allow_stale)


def load_view(
    path: Path,
    cls: type[MechanismSample],
    version: int,
    allow_stale: bool = False,
    ids: Iterable[str] | None = None,
) -> list[MechanismSample]:
    """The samples of one split view (only `ids` when given), in its id order (skipped steps
    have none). Verifies as `iter_view`."""
    wanted, by_shard = _view_members(path, version, allow_stale, ids)
    loaded = {s.step_id: s for s in _iter_shards(path, by_shard, cls, version, allow_stale)}
    return [loaded[sid] for sid in wanted]
