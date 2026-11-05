"""Compare two sample stores column by column (open-shell plan P8 step 7): a rebuilt store
(v19) against an earlier one (the v18 stores of P0a/P0b), on the columns both hold.

    uv run python scripts/compare_stores.py --a OLD/samples --b NEW/samples [--workers 8]
        [--out FILE]

Each argument is a store directory (its `index.json` and shards; `arrows.store`), read at the
schema version its own index records, so a v18 store and a v19 store are read side by side.
Every sample is digested by name as the P0b pin does (`eval.pin.column_digests`: the ids, the
categorical atom columns, each atom, orbital and pair column, pair_valid, the edit label and
both ArrowLabel lists). The report gives the step ids only one store holds, the columns only
one store holds (a later version's new columns are listed, never a difference), and for every
column both hold the number of samples present in both whose digest differs, with up to
`--examples` ids. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLES = 10


def _index(store: Path) -> tuple[int, dict[str, list[str]]]:
    """(the schema version the index records, shard name -> step ids)."""
    body = json.loads((Path(store) / "index.json").read_text())
    return int(body["provenance"]["schema_version"]), body["shards"]


def _digest_shard(task: tuple) -> dict[str, dict[str, str]]:
    """Step id -> column digests of every sample of one shard."""
    from ksmech.arrows import store as st
    from ksmech.arrows.steps import MechanismSample
    from ksmech.eval.pin import column_digests

    path, n, version = task
    samples = st._load_shard(Path(path), list(range(n)), MechanismSample, version, False)
    return {s.step_id: column_digests(s) for s in samples}


def store_digests(store: Path, n_workers: int = 1) -> tuple[int, dict[str, dict[str, str]]]:
    """(schema version, step id -> column digests) of every sample of a store."""
    from ksmech.processes import in_processes

    version, shards = _index(store)
    tasks = [(str(Path(store) / name), len(ids), version) for name, ids in sorted(shards.items())]
    out: dict[str, dict[str, str]] = {}
    for part in in_processes(_digest_shard, tasks, n_workers):
        out.update(part)
    return version, out


def compare(a: dict[str, dict[str, str]], b: dict[str, dict[str, str]], examples: int = EXAMPLES):
    """The report of the module docstring over two digest maps."""
    both = sorted(a.keys() & b.keys())
    cols_a = set().union(*(a[i].keys() for i in both)) if both else set()
    cols_b = set().union(*(b[i].keys() for i in both)) if both else set()
    columns = {}
    for col in sorted(cols_a & cols_b):
        diff = [i for i in both if a[i].get(col) != b[i].get(col)]
        columns[col] = {"n_differ": len(diff), "examples": diff[:examples]}
    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_both": len(both),
        "only_a": {
            "n": len(a.keys() - b.keys()),
            "examples": sorted(a.keys() - b.keys())[:examples],
        },
        "only_b": {
            "n": len(b.keys() - a.keys()),
            "examples": sorted(b.keys() - a.keys())[:examples],
        },
        "columns_only_a": sorted(cols_a - cols_b),
        "columns_only_b": sorted(cols_b - cols_a),
        "columns": columns,
        "columns_that_differ": sorted(c for c, r in columns.items() if r["n_differ"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", type=Path, required=True, help="the earlier store directory")
    parser.add_argument("--b", type=Path, required=True, help="the rebuilt store directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--examples", type=int, default=EXAMPLES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    va, da = store_digests(args.a, args.workers)
    vb, db = store_digests(args.b, args.workers)
    report = {"a": str(args.a), "b": str(args.b), "version_a": va, "version_b": vb}
    report |= compare(da, db, args.examples)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    differ = {c: report["columns"][c]["n_differ"] for c in report["columns_that_differ"]}
    print(f"a v{va}: {report['n_a']}, b v{vb}: {report['n_b']}, both {report['n_both']}")
    print(f"only a {report['only_a']['n']}, only b {report['only_b']['n']}; differ {differ}")
    print(f"columns only in b: {report['columns_only_b']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
