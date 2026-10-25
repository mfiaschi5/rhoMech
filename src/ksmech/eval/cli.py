"""`ksmech-eval`: evaluation of anchors and reaction matrices (spec v9, "Evaluation protocol").

    ksmech-eval floor --source pmechdb --split molecule

`floor` ranks each step's anchor by |G^atom| (no learned parameter) on the validation and test
parts and writes `anchor_floor_report_<split>.json`, the reference every trained model is
compared with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.arrows.steps import load_samples, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.data.schema import Source
from ksmech.eval.floor import floor_report

FLOOR_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("floor", help="the |G^atom| anchor ranking on val and test")
    f.add_argument("--source", required=True, choices=[str(x) for x in Source])
    f.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    f.add_argument("--split", default="molecule", help="split strategy of the sample files")
    return parser


def cmd_floor(args, console: Console) -> int:
    results, inputs = {}, {}
    for part in ("val", "test"):
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        inputs |= prov.file_input(path)
        results[part] = floor_report(load_samples(path))
    header = prov.stamp("anchor_floor", FLOOR_VERSION, inputs, {"split": args.split})
    out = source_dir(args.source, args.processed_dir) / f"anchor_floor_report_{args.split}.json"
    prov.write_json(out, {"scorer": "|G^atom| over valid pairs", **results}, header)
    table = Table(title=f"anchor floor, {args.source} {args.split} split")
    for col in ("part", "n", "top-1", "top-3", "top-5", "median rank", "unreachable"):
        table.add_column(col)
    for part, r in results.items():
        tops = (f"{r[f'top{k}']:.1%}" for k in (1, 3, 5))
        table.add_row(part, str(r["n"]), *tops, f"{r['median_rank']:.0f}", str(r["unreachable"]))
    console.print(table)
    console.print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"floor": cmd_floor}[args.command](args, Console())


if __name__ == "__main__":
    sys.exit(main())
