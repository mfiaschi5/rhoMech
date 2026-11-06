"""Open-shell plan P5 step 4: the two ReactMech checkpoints re-scored by slice, from the
per-sample decode hits.

    uv run python scripts/p5_rescore.py [--pin]

Runs: reactmech/sparse_resonance_stop0 and pmechdb+reactmech/sparse_resonance (decoded with
`--run-source`). Hits on the frozen molecule val: P0's (`<run>/p0_reference/`, the v17 store),
P0a's re-score (copied to `<run>/p0a_reference/` before P5 decoded) and P5's (the run
directory). Slices of the molecule val: (a) unchanged since P0 (in P0's hits, neither
re-labelled nor re-charged by P0a); (b) P0a's Kekule-closest re-labels and (c) its 49 dative
re-charges (`p0a_changes.json`); (d) P5's returns in the val (samples P0a's re-score did not
have). Each is reported with and without the changed slices. Checks: on (a) every sample's rank
is P0's, and every pre-P5 sample's rank is P0a's.

The returns bridge groups the molecule snapshot froze in two parts, so nearly all are
SPLIT_CONFLICT there and (d) is empty in its val. Every return is new to both checkpoints (none
was a sample when they trained), so all of them (`cohort_p5_returns.json`) are decoded through
the scaffold views, which place every step, and scored as (d) by scaffold part; their parts per
view are reported. `--pin` writes the val slices (b) and (c) as cohorts (`cohort_p5_*.json`,
pinned once). The report goes to data/processed/reactmech/p5_rescore.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ksmech.arrows.steps import PARTS, sample_ids
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.eval.cohorts import load_cohort, top_k, write_cohort

MODELS = Path("data/models")
RM = DEFAULT_PROCESSED_DIR / "reactmech"
RUNS = {  # run directory, hits file stem suffix (the samples' source when it is not the run's)
    "stop0": ("reactmech/sparse_resonance_stop0", ""),
    "joint": ("pmechdb+reactmech/sparse_resonance", "_reactmech"),
}


def ranks(path: Path) -> dict[str, int | None]:
    return {r["step_id"]: r["rank"] for r in json.loads(path.read_text())["per_sample"]}


def slices(p0: dict, p0a: dict, p5: dict) -> dict[str, set[str]]:
    rules = json.loads((RM / "p0a_changes.json").read_text())["rules"]
    kekule = set(rules["KEKULE_CLOSEST"]["changed"])
    dative = set(rules["DATIVE"]["changed"])
    returns = set(p5) - set(p0a)
    unchanged = (set(p5) & set(p0)) - kekule - dative
    return {"a_unchanged": unchanged, "b_kekule": kekule & set(p5), "c_dative": dative & set(p5),
            "d_returns": returns}  # fmt: skip


def scored(hits: dict, ids: set[str]) -> dict:
    return top_k([hits[i] for i in sorted(ids) if i in hits])


def val_report(root: Path, suffix: str) -> dict:
    """The molecule val by slice, and the two reproduction checks."""
    file = f"decode_hits_molecule_val{suffix}.json"
    p0 = ranks(root / "p0_reference" / file)
    p0a = ranks(root / "p0a_reference" / file)
    p5 = ranks(root / file)
    cut = slices(p0, p0a, p5)
    changed = cut["b_kekule"] | cut["c_dative"] | cut["d_returns"]
    out = {"whole": scored(p5, set(p5)), "without_changed": scored(p5, set(p5) - changed)}
    for key, ids in cut.items():
        out[key] = {"with": scored(p5, ids), "without": scored(p5, set(p5) - ids)}
    out["p0_on_a"] = scored(p0, cut["a_unchanged"])
    out["p0a_whole"] = scored(p0a, set(p0a))
    out["check_a_equals_p0"] = all(p5[i] == p0[i] for i in cut["a_unchanged"])
    out["check_pre_p5_equals_p0a"] = all(p5[i] == p0a[i] for i in p0a if i in p5)
    out["pre_p5_missing"] = sorted(set(p0a) - set(p5))
    return out | {"_slices": {k: sorted(v) for k, v in cut.items()}}


def returns_report(root: Path, suffix: str) -> dict:
    """(d): every return, decoded through the scaffold views, by scaffold part and pooled."""
    hits: dict[str, int | None] = {}
    out = {}
    for part in PARTS:
        got = ranks(root / f"decode_hits_scaffold_{part}{suffix}_p5_returns.json")
        out[f"scaffold_{part}"] = scored(got, set(got))
        hits |= got
    return {"all": scored(hits, set(hits)), **out}


def placement(returns: set[str]) -> dict:
    """The parts the returns hold in every view (a step in no part of a strategy is
    SPLIT_CONFLICT there)."""
    out: dict[str, Counter] = {}
    for path in sorted(RM.glob("samples_*_*.json")):
        name = path.name.removeprefix("samples_").removesuffix(".json")
        if "report" in name:
            continue
        strategy, part = name.rsplit("_", 1)
        held = returns & set(sample_ids(path))
        out.setdefault(strategy, Counter())[part] += len(held)
    return {s: dict(c) | {"none": len(returns) - sum(c.values())} for s, c in out.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pin", action="store_true", help="pin the val slices (b), (c)")
    args = parser.parse_args(argv)
    returns = load_cohort("reactmech", "p5_returns")
    report = {}
    for name, (run, suffix) in RUNS.items():
        root = MODELS / run
        report[name] = val_report(root, suffix) | {"d_all_returns": returns_report(root, suffix)}
    report["returns_placement"] = placement(returns)
    if args.pin:
        cut = report["stop0"]["_slices"]
        for key in ("b_kekule", "c_dative"):
            note = {"what": f"P5 slice {key} of the ReactMech molecule val (scripts/p5_rescore.py)"}
            print("wrote", write_cohort("reactmech", f"p5_{key}", cut[key], note))
    for name in RUNS:
        report[name].pop("_slices")
        shown = {k: v for k, v in report[name].items() if k != "pre_p5_missing"}
        print(name, json.dumps(shown, indent=1))
    print("returns_placement", json.dumps(report["returns_placement"]))
    (RM / "p5_rescore.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
