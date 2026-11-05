"""Cohorts, slices and the covered fraction of an evaluation (open-shell plan P0).

A cohort is a frozen set of step ids, `cohort_<name>.json` beside the source's samples (kind
`cohort`, COHORT_SCHEMA, no file inputs, a content hash: re-ingests and store bumps never make
it stale). `ksmech-eval floor|decode --cohort <name>` scores only its steps, so new data is
measured in its own cohort and never pooled into a headline number. A slice reports the same
numbers on the samples with and without one attribute (`--slice fermi_smeared`, or
`bond_stretched`: an R entry on a Li/Na/Mg/K bond the record does not hold). The covered
fraction says which share of a part's steps became samples, with the reason codes of the rest.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir

COHORT_SCHEMA = 1


def _bond_stretched(sample) -> bool:
    from ksmech.arrows.steps import stretched_bonds

    return bool(stretched_bonds(sample))


SLICES: dict[str, Callable] = {
    "fermi_smeared": lambda s: bool(s.fermi_smeared),
    "bond_stretched": _bond_stretched,  # SKIPPED_BOND_STRETCHED (open-shell plan P3)
}


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


def class_cohort_ids(steps: Iterable, classes: Iterable[str]) -> list[str]:
    """The ids of the steps whose `domain_class` holds one of `classes` (pass the training
    target, `io.read_training_steps`): the cohorts of admitted new data, e.g. open_shell from
    RADICAL_SPECTATOR and RADICAL_REACTIVE (open-shell plan P3)."""
    from ksmech.data.domain import domain_classes

    want = set(classes)
    return sorted(s.step_id for s in steps if domain_classes(s.domain_class) & want)


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


R_SHAPES = ("homolysis", "recombination", "addition", "abstraction_sh2", "beta_scission", "other")


def _order(sample, i: int, j: int) -> int:
    i, j = min(i, j), max(i, j)
    hit = [int(o) for a, b, o in sample.bonds if a == i and b == j]
    return hit[0] if hit else 0


def _two_bond_shape(sample, bonds, odd: dict[int, int]) -> str:
    """One bond formed and one broken that share an atom, the single electrons on the two
    other ends: abstraction or SH2 (the broken bond goes, the formed one is new), addition
    (the formed bond is new, the broken one keeps an order), beta-scission (the broken bond
    goes, the formed one raises an order)."""
    (fi, fj, _), (ki, kj, _) = sorted(bonds, key=lambda b: -b[2])
    shared = {fi, fj} & {ki, kj}
    if len(shared) != 1:
        return "other"
    x = shared.pop()
    f_end, k_end = ({fi, fj} - {x}).pop(), ({ki, kj} - {x}).pop()
    if odd != {f_end: -1, k_end: 1}:
        return "other"
    before_f, before_k = _order(sample, fi, fj), _order(sample, ki, kj)
    if before_f == 0 and before_k == 1:
        return "abstraction_sh2"
    if before_f == 0 and before_k >= 2:
        return "addition"
    return "beta_scission" if before_f >= 1 and before_k == 1 else "other"


def r_shape(sample) -> str:
    """The shape of a sample's R (its first label variant; open-shell plan, evaluation (6)),
    read from R and the reactant's bond orders alone: one of `R_SHAPES`."""
    label = getattr(sample, "edit_label", None)
    if label is None or not label.variants:
        return "other"
    r = label.variants[0]
    bonds = [(i, j, d) for i, j, d in r.bonds if d]
    odd = {a: d for a, d in r.lone if d % 2}
    if len(odd) != len([d for _, d in r.lone if d]):
        return "other"  # an even atom entry beside: not one of the one-electron shapes
    if len(bonds) == 1:
        i, j, d = bonds[0]
        if d == -1 and odd == {i: 1, j: 1}:
            return "homolysis"
        if d == 1 and odd == {i: -1, j: -1}:
            return "recombination"
        return "other"
    if len(bonds) == 2 and {d for *_, d in bonds} == {1, -1}:
        return _two_bond_shape(sample, bonds, odd)
    return "other"


def by_r_shape(rows: list[dict], samples: dict) -> dict:
    """Top-k of the per-sample `rows` (step id, rank) by `r_shape`; `samples` maps step id ->
    sample. A group under 30 is reported as counts (its n), never read as a percentage."""
    groups: dict[str, list] = {shape: [] for shape in R_SHAPES}
    for row in rows:
        groups[r_shape(samples[row["step_id"]])].append(row["rank"])
    return {shape: top_k(ranks) for shape, ranks in groups.items() if ranks}


def covered_fraction(view: Path) -> dict:
    """Steps of the view's split, samples among them, their ratio and the skip reasons of the
    rest (`arrows.steps.split_report` counts stored in the view)."""
    counts = json.loads(Path(view).read_text()).get("report", {})
    steps, samples = counts.get("n_steps", 0), counts.get("n_samples", 0)
    covered = samples / steps if steps else 0.0
    return {"n_steps": steps, "n_samples": samples, "covered": covered, **counts.get("skipped", {})}
