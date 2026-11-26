"""`ksmech-pretrain`: the data of the phase-3 pretraining (`pretrain.samples`).

    ksmech-pretrain species --n 50000 --seed 0     # the seeded QMugs subset, a species file
    ksmech-pretrain samples --species FILE --workers 14

Between the two: `ksmech-qm bond-index` over the QMugs records and `ksmech-orbitals build
--species-from FILE` with their cache, token directory and method (printed by `species`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console

from ksmech.data import qmugs
from ksmech.pretrain import samples as ps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ksmech-pretrain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("species", help="the seeded random subset of the QMugs records")
    sp.add_argument("--n", type=int, required=True, help="molecules drawn")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--manifest", type=Path, default=qmugs.PROCESSED_DIR / "qm_manifest.parquet")
    sp.add_argument("--out", type=Path, help="default <out-dir>/species_<n>.json")
    sp.add_argument("--out-dir", type=Path, default=ps.OUT_DIR)
    sa = sub.add_parser("samples", help="the pretraining store of a species file")
    sa.add_argument("--species", type=Path, required=True, help="a file of `species`")
    sa.add_argument("--out-dir", type=Path, default=ps.OUT_DIR)
    sa.add_argument("--workers", type=int, default=1)
    return parser


def cmd_species(args: argparse.Namespace, console: Console) -> int:
    out = args.out or args.out_dir / f"species_{args.n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ps.write_species_file(out, args.manifest, args.n, args.seed)
    console.print(f"wrote {out}; then:")
    wbi = f"--cache-dir {qmugs.CACHE_DIR} --out-dir {ps.BOND_INDEX_DIR}"
    console.print(f"  ksmech-qm bond-index {wbi}")
    console.print(
        f"  ksmech-orbitals build --species-from {out} --qm-cache-dir {qmugs.CACHE_DIR} "
        f"--out-dir {ps.TOKENS_DIR} --method {ps.METHOD} --workers 14"
    )
    console.print(f"  ksmech-pretrain samples --species {out} --workers 14")
    return 0


def cmd_samples(args: argparse.Namespace, console: Console) -> int:
    report = ps.build_store(args.species, args.out_dir, ps.qmugs_loader(), args.workers)
    console.print(json.dumps(report, indent=2))
    return 0 if report["n_samples"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    console = Console()
    if args.command == "species":
        return cmd_species(args, console)
    return cmd_samples(args, console)


if __name__ == "__main__":
    sys.exit(main())
