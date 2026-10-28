"""Cohorts, slices and the covered fraction of an evaluation (open-shell plan P0).

A cohort is a frozen set of step ids, `cohort_<name>.json` beside the source's samples (kind
`cohort`, COHORT_SCHEMA, no file inputs, a content hash: re-ingests and store bumps never make
it stale). `ksmech-eval floor|decode --cohort <name>` scores only its steps, so new data is
measured in its own cohort and never pooled into a headline number. A slice reports the same
numbers on the samples with and without one attribute (`--slice fermi_smeared`). The covered
fraction says which share of a part's steps became samples, with the reason codes of the rest.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir

COHORT_SCHEMA = 1
SLICES: dict[str, Callable] = {"fermi_smeared": lambda s: bool(s.fermi_smeared)}


def cohort_path(source: str, name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return source_dir(source, processed_dir) / f"cohort_{name}.json"


def write_cohort(
    source: str,
    name: str,
    ids: Iterable[str],
    note: dict | None = None,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """A new cohort file; an existing one is never overwritten (a cohort is pinned once)."""
    out = cohort_path(source, name, processed_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists: a cohort is replaced only by hand")
    ids = sorted(set(ids))
    content = prov.sha256_text(json.dumps(ids))
    header = prov.stamp("cohort", COHORT_SCHEMA, {}, {"name": name}, content, note or {})
    return prov.write_json(out, {"ids": ids}, header)


def load_cohort(
    source: str, name: str, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> set[str] | None:
    """The cohort's ids; None for `all` (no restriction). Kind, schema, clean build and content
    hash are checked."""
    if name == "all":
        return None
    path = cohort_path(source, name, processed_dir)
    header = prov.verify_clean(path, "cohort", COHORT_SCHEMA)
    ids = json.loads(path.read_text())["ids"]
    if prov.sha256_text(json.dumps(ids)) != header.content_sha256:
        raise prov.StaleArtifact(path, ["its ids differ from its content hash"])
    return set(ids)


def restrict(ids: list[str], cohort: set[str] | None) -> list[str]:
    return ids if cohort is None else [i for i in ids if i in cohort]


def top_k(ranks: list[int | None], ks=(1, 3, 5)) -> dict:
    """Top-k of a list of 0-based ranks (None a miss) and n."""
    n = len(ranks)
    out = {f"top{k}": sum(r is not None and r < k for r in ranks) / n if n else 0.0 for k in ks}
    return out | {"n": n}


def sliced(rows: list[dict], samples: dict, name: str) -> dict:
    """Top-k of the per-sample `rows` (step id, rank) on the samples with and without the
    slice's attribute; `samples` maps step id -> sample."""
    has = SLICES[name]
    with_, without = [], []
    for row in rows:
        (with_ if has(samples[row["step_id"]]) else without).append(row["rank"])
    return {"slice": name, "with": top_k(with_), "without": top_k(without)}


def covered_fraction(view: Path) -> dict:
    """Steps of the view's split, samples among them, their ratio and the skip reasons of the
    rest (`arrows.steps.split_report` counts stored in the view)."""
    counts = json.loads(Path(view).read_text()).get("report", {})
    steps, samples = counts.get("n_steps", 0), counts.get("n_samples", 0)
    covered = samples / steps if steps else 0.0
    return {"n_steps": steps, "n_samples": samples, "covered": covered, **counts.get("skipped", {})}
