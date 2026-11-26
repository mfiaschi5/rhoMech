"""Freeze the human benchmark's dev and test halves once (`network.benchmarks.freeze_halves`).

    uv run python scripts/benchmark_halves.py --inspected FILE

FILE is a JSON list of the pathway ids inspected by hand before the halves were drawn; they all
go to dev. Writes `<processed-dir>/human_benchmark/halves.json` and refuses to overwrite it.
"""

import argparse
import hashlib
import json
from pathlib import Path

from ksmech.data import human_benchmark as hb
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.network.benchmarks import HALVES_FILE, freeze_halves


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspected", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = p.parse_args()
    out = args.processed_dir / hb.NAME / HALVES_FILE
    if out.exists():
        print(f"{out} exists: the halves are frozen")
        return 1
    ids = [p.pathway_id for p in hb.load_pathways(args.processed_dir / hb.NAME)]
    halves = freeze_halves(ids, json.loads(args.inspected.read_text()))
    out.write_text(json.dumps(halves, indent=1))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(
        f"dev {len(halves['dev'])}, test {len(halves['test'])}, inspected "
        f"{len(halves['inspected'])} -> {out} (sha256 {digest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
