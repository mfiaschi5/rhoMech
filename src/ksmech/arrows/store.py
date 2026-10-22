"""The sample store: every step's sample once, in compressed shards, plus one small view per split.

Layout under `data/processed/<source>/samples/`:
- `shard_00000.npz` ... : `SHARD_SIZE` samples each, members `<i>/<field>` and a JSON `meta`
  member (ids, labels, keys), with a provenance header whose inputs are the steps parquet and
  every token file and contact block its samples used;
- `index.json`: step id -> shard, with the shards as provenance inputs.
A split is `samples_<split>.json` next to the store: its ids and counts, with the index and the
split file as inputs. Nothing is copied per split.

On disk a sample keeps its pair arrays for the valid cells only (`pair_valid`, ~89% of the
cells): `pair_z` and `atom_coulomb` as float16, the 0/1 features `inter` and `clash` and
`pair_defined` bit-packed, contact sites as int16, the symmetric `atom_coulomb` as its upper
triangle. In memory every array is dense float32 again, invalid cells zero / undefined / -1.
float16 keeps 3 significant digits: the largest error measured on the pair features was 5e-3
on log-features spanning ~30 units, below 1e-7 absolute on the tiny ones
(docs/decisions.md, fix item 8).
"""

from __future__ import annotations

import io
import json
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ksmech.data import provenance as prov

if TYPE_CHECKING:
    from ksmech.arrows.steps import MechanismSample

SHARD_SIZE = 256
VALID_ONLY = ("pair_z", "pair_defined", "pair_gap_sign", "pair_contact_sites")
_HALF = ("pair_z", "atom_coulomb")
BINARY_FEATURES = ("inter", "clash")  # exact 0/1 pair features: stored as bits, not float16


def _binary_columns() -> list[int]:
    from ksmech.physics.pairs import PAIR_FEATURES

    return [PAIR_FEATURES.index(f) for f in BINARY_FEATURES]


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
            meta[f.name] = [lab.to_dict() for lab in value] if f.name == "labels" else value
    valid = sample.pair_valid
    for name in VALID_ONLY:
        arrays[name] = arrays[name][valid]
    binary = _binary_columns()
    flags = arrays["pair_z"][:, binary]
    if not np.isin(flags, (0.0, 1.0)).all():
        raise ValueError(f"{sample.step_id}: a binary pair feature is not 0/1")
    arrays["pair_flags"] = np.packbits(flags.astype(bool), axis=-1)
    arrays["pair_z"] = np.delete(arrays["pair_z"], binary, axis=-1)
    for name in _HALF:
        arrays[name] = arrays[name].astype(np.float16)
    arrays["pair_z"] = _byte_planes(arrays["pair_z"])
    coul = arrays["atom_coulomb"]  # symmetric: the upper triangle is enough
    arrays["atom_coulomb"] = coul[np.triu_indices(len(coul))]
    arrays["pair_contact_sites"] = arrays["pair_contact_sites"].astype(np.int16)
    arrays["pair_defined"] = np.packbits(arrays["pair_defined"], axis=-1)
    meta["n_pair_features"] = int(sample.pair_z.shape[-1])
    return arrays, meta


def unpack(
    arrays: dict[str, np.ndarray], meta: dict[str, Any], cls: type[MechanismSample]
) -> MechanismSample:
    """The dense MechanismSample of `pack`'s output (cls: MechanismSample)."""
    from ksmech.arrows.align import ArrowLabel

    meta = dict(meta)
    n_feat = meta.pop("n_pair_features")
    valid = arrays["pair_valid"].astype(bool)
    shape = valid.shape
    dense = {
        "pair_z": np.zeros((*shape, n_feat), dtype=np.float32),
        "pair_defined": np.zeros((*shape, n_feat), dtype=bool),
        "pair_gap_sign": np.zeros(shape, dtype=np.int8),
        "pair_contact_sites": np.full((*shape, 2), -1, dtype=np.int64),
    }
    binary = _binary_columns()
    rest = [k for k in range(n_feat) if k not in binary]
    z = np.zeros((int(valid.sum()), n_feat), dtype=np.float32)
    z[:, rest] = _from_byte_planes(arrays["pair_z"]).astype(np.float32)
    z[:, binary] = np.unpackbits(arrays["pair_flags"], axis=-1, count=len(binary))
    dense["pair_z"][valid] = z
    bits = np.unpackbits(arrays["pair_defined"], axis=-1, count=n_feat)
    dense["pair_defined"][valid] = bits.astype(bool)
    dense["pair_gap_sign"][valid] = arrays["pair_gap_sign"]
    dense["pair_contact_sites"][valid] = arrays["pair_contact_sites"]
    out = {k: v for k, v in arrays.items() if k not in (*VALID_ONLY, "pair_flags")}
    n_atoms = len(arrays["atom_categorical"])
    coul = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    coul[np.triu_indices(n_atoms)] = arrays["atom_coulomb"].astype(np.float32)
    out["atom_coulomb"] = np.triu(coul) + np.triu(coul, 1).T
    meta["labels"] = [ArrowLabel.from_dict(x) for x in meta["labels"]]
    return cls(**out, **dense, **meta)


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
    _verify(path, "sample_shard", version, allow_stale)
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


def _verify(path: Path, kind: str, version: int, allow_stale: bool) -> prov.Provenance | None:
    """The verified header; a sample artifact built from a dirty tree is refused too (it feeds
    training and evaluation). `allow_stale` logs instead (diagnostic CLIs)."""
    import logging

    try:
        header = prov.verify(path, kind, version)
        prov.require_clean(header, path)
        return header
    except prov.StaleArtifact as exc:
        if not allow_stale:
            raise
        logging.getLogger(__name__).warning("reading a stale %s (--allow-stale): %s", kind, exc)
        return prov.read_header(path)


def load_view(
    path: Path, cls: type[MechanismSample], version: int, allow_stale: bool = False
) -> list[MechanismSample]:
    """The samples of one split view, in its id order (skipped steps have none). Verifies the
    view, the index and every shard read: StaleArtifact names what changed."""
    header = _verify(path, "samples", version, allow_stale)
    for name in header.inputs if header else ():
        if name.startswith("file:") and prov.input_path(name).name.startswith("splits_"):
            _verify(prov.input_path(name), "split", version, allow_stale)  # its parquet
    view = json.loads(Path(path).read_text())
    index_path = store_dir(Path(path).parent) / "index.json"
    _verify(index_path, "sample_index", version, allow_stale)
    shards = json.loads(index_path.read_text())["shards"]
    where = {sid: (name, k) for name, ids in shards.items() for k, sid in enumerate(ids)}
    wanted = [sid for sid in view["ids"] if sid in where]
    by_shard: dict[str, list[int]] = {}
    for sid in wanted:
        name, k = where[sid]
        by_shard.setdefault(name, []).append(k)
    loaded = {}
    for name, members in sorted(by_shard.items()):
        root = store_dir(Path(path).parent)
        for s in _load_shard(root / name, members, cls, version, allow_stale):
            loaded[s.step_id] = s
    return [loaded[sid] for sid in wanted]


def stored_bytes(sample: MechanismSample) -> int:
    """Compressed size of one sample as stored (for the size report)."""
    arrays, meta = pack(sample)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays, meta=np.array(json.dumps(meta)))
    return buf.tell()
