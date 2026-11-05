"""A pinned cohort and the per-column digests of its samples (open-shell plan P0b, step 1).

`ksmech-eval pin --source pmechdb --name closed_base` pins the samples of one split's three
parts as a cohort (`eval.cohorts`, written once) and, beside it, one cohort per cell set of the
samples a first-arrow ranker scores (`<name>_gbdt_valid`, `<name>_gbdt_all`: the samples with
ArrowLabels, `physics.baseline.sample_labels`). Every sample's columns are digested by name into
`baseline_<name>.json`: the ids, atom_categorical[:, :2], each atom and orbital column by name
(`atom_columns`, `orbital_columns`; atom values masked where missing), the spectral sets, each
pair feature by name (masked where undefined, with its definition mask), pair_valid, the edit
label and both ArrowLabel lists. Like a cohort the baseline has no file inputs and a content
hash, so a rebuilt store never makes it stale; the store it was read from is recorded in its
note. A later store is compared with it column by column (`compare_digests`): a column only one
side holds is reported, never failed, so later phases add columns without breaking the gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir

BASELINE_SCHEMA = 1
DIGEST_CHARS = 16  # 64 bits per column and sample
CELL_SETS = ("valid", "all")
CHUNK = 256
# the columns v19 adds after a v18 baseline (open-shell plan P7): reported as only in the current
# store, never a difference
V19_COLUMNS = frozenset(
    ["atom.somo_population", "orbital.spin_role"]
    + [f"pair.{n}" for n in ("g2", "log_abs_g2", "own_g", "own_g2", "somo_pair", "somo_level_diff")]
)


def _digest(*parts) -> str:
    """sha256 of arrays (dtype, shape and bytes) and JSON-able values, truncated."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            a = np.ascontiguousarray(part)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
        else:
            h.update(json.dumps(part, sort_keys=True).encode())
    return h.hexdigest()[:DIGEST_CHARS]


def _masked(values: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    missing = np.asarray(missing, dtype=bool)
    return np.where(missing, 0.0, values).astype(np.float32), missing


def column_digests(s) -> dict[str, str]:
    """Column name -> digest of one MechanismSample's content in that column."""
    out = {
        "ids": _digest([s.step_id, list(s.fragment_keys)]),
        "atom_categorical": _digest(np.asarray(s.atom_categorical[:, :2], dtype=np.int64)),
    }
    for k, name in enumerate(s.atom_columns):
        out[f"atom.{name}"] = _digest(*_masked(s.atom_features[:, k], s.atom_missing[:, k]))
    for k, name in enumerate(s.orbital_columns):
        out[f"orbital.{name}"] = _digest(np.asarray(s.orbital_features[:, k], dtype=np.float32))
    out["orbital.spectra"] = _digest(np.asarray(s.orbital_spectra, dtype=np.float32))
    for k, name in enumerate(s.pair_features):
        out[f"pair.{name}"] = _digest(*_masked(s.pair_z[..., k], ~s.pair_defined[..., k]))
    out["pair_valid"] = _digest(np.asarray(s.pair_valid, dtype=bool))
    out["edit_label"] = _digest(s.edit_label.to_dict() if s.edit_label is not None else None)
    out["labels"] = _digest([lab.to_dict() for lab in s.labels])
    out["labels_all"] = _digest([lab.to_dict() for lab in s.labels_all])
    return out


def _read_chunk(task: tuple) -> list[tuple[str, dict[str, str], tuple[bool, ...]]]:
    """(step id, column digests, has labels per cell set) of one chunk of a view's samples."""
    from ksmech.arrows.steps import iter_samples
    from ksmech.physics.baseline import sample_labels

    path, ids = task
    return [
        (s.step_id, column_digests(s), tuple(bool(sample_labels(s, c)) for c in CELL_SETS))
        for s in iter_samples(path, ids=ids)
    ]


def read_view(path: Path, n_workers: int = 1) -> list[tuple[str, dict, tuple[bool, ...]]]:
    """`_read_chunk` over every sample of one view, in view order."""
    from ksmech import processes
    from ksmech.arrows.steps import sample_ids

    tasks = [(Path(path), chunk) for chunk in processes.chunked(sample_ids(path), CHUNK)]
    return [row for rows in processes.in_processes(_read_chunk, tasks, n_workers) for row in rows]


def baseline_path(source: str, name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return source_dir(source, processed_dir) / f"baseline_{name}.json"


def write_baseline(
    source: str, name: str, digests: dict[str, dict[str, str]], note: dict, processed_dir: Path
) -> Path:
    """`baseline_<name>.json`: the column names once and one digest list per sample; written
    once, like the cohort it describes."""
    out = baseline_path(source, name, processed_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists: a baseline is replaced only by hand")
    columns = sorted({c for d in digests.values() for c in d})
    body = {
        "columns": columns,
        "digests": {i: [d.get(c, "") for c in columns] for i, d in sorted(digests.items())},
    }
    content = prov.sha256_text(json.dumps(body, sort_keys=True))
    header = prov.stamp("cohort_baseline", BASELINE_SCHEMA, {}, {"name": name}, content, note)
    return prov.write_json(out, body, header)


def load_baseline(
    source: str, name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> dict[str, dict[str, str]]:
    """Step id -> column -> digest of a pinned baseline (kind, schema, clean build and content
    hash checked)."""
    path = baseline_path(source, name, processed_dir)
    header = prov.verify_clean(path, "cohort_baseline", BASELINE_SCHEMA)
    stored = json.loads(path.read_text())
    body = {k: stored[k] for k in ("columns", "digests")}
    if prov.sha256_text(json.dumps(body, sort_keys=True)) != header.content_sha256:
        raise prov.StaleArtifact(path, ["its digests differ from its content hash"])
    cols = body["columns"]
    return {
        i: {c: d for c, d in zip(cols, row, strict=True) if d} for i, row in body["digests"].items()
    }


def compare_digests(
    baseline: dict[str, dict[str, str]], current: dict[str, dict[str, str]]
) -> dict:
    """Column by column over the samples both hold: the samples whose shared columns differ
    (per column), the columns only one side has, and the samples only one side has."""
    shared = sorted(set(baseline) & set(current))
    differ: dict[str, list[str]] = {}
    only_base, only_new = set(), set()
    for i in shared:
        old, new = baseline[i], current[i]
        only_base |= set(old) - set(new)
        only_new |= set(new) - set(old)
        for col in set(old) & set(new):
            if old[col] != new[col]:
                differ.setdefault(col, []).append(i)
    return {
        "n_shared": len(shared),
        "differ": {c: sorted(ids) for c, ids in sorted(differ.items())},
        "columns_only_in_baseline": sorted(only_base),
        "columns_only_in_current": sorted(only_new),
        "samples_only_in_baseline": sorted(set(baseline) - set(current)),
        "samples_only_in_current": sorted(set(current) - set(baseline)),
    }


def pin_cohort(
    source: str,
    name: str,
    split: str = "molecule",
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    n_workers: int = 1,
) -> dict:
    """Pin `name` (every sample of the split's three parts), its two GBDT subsets and its
    baseline digests. Nothing is written when any of the files exists."""
    from ksmech.arrows.steps import PARTS, SAMPLE_VERSION, samples_path
    from ksmech.eval.cohorts import cohort_path, write_cohort

    targets = [cohort_path(source, n, processed_dir) for n in _cohort_names(name)]
    if existing := [
        p for p in [*targets, baseline_path(source, name, processed_dir)] if p.exists()
    ]:
        raise FileExistsError(f"{existing[0]} exists: a cohort is pinned once")
    views = {p: samples_path(source, f"{split}_{p}", processed_dir) for p in PARTS}
    rows = {p: read_view(path, n_workers) for p, path in views.items()}
    counts = _subset_counts(rows)
    note = {
        "split": split,
        "sample_version": SAMPLE_VERSION,
        "views": {p: prov.file_sha256(path) for p, path in views.items()},
        "counts": counts,
    }
    ids = {i for part in rows.values() for i, _, _ in part}
    written = [write_cohort(source, name, ids, note, processed_dir)]
    for k, cells in enumerate(CELL_SETS):
        sub = {i for part in rows.values() for i, _, has in part if has[k]}
        written.append(write_cohort(source, f"{name}_gbdt_{cells}", sub, note, processed_dir))
    digests = {i: d for part in rows.values() for i, d, _ in part}
    written.append(write_baseline(source, name, digests, note, processed_dir))
    return {"written": [str(p) for p in written], **counts}


def _cohort_names(name: str) -> list[str]:
    return [name, *(f"{name}_gbdt_{c}" for c in CELL_SETS)]


def _subset_counts(rows: dict[str, list]) -> dict:
    """Per part: the samples, and those with ArrowLabels per cell set."""
    return {
        part: {"samples": len(r)}
        | {f"gbdt_{c}": sum(has[k] for _, _, has in r) for k, c in enumerate(CELL_SETS)}
        for part, r in rows.items()
    }


def digests_of(samples: Iterable) -> dict[str, dict[str, str]]:
    """Step id -> column digests of in-memory samples (a rebuilt store, a test)."""
    return {s.step_id: column_digests(s) for s in samples}
