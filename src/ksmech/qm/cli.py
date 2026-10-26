"""`ksmech-qm`: GFN2-xTB featurization of the molecules in a processed dataset, the Wiberg bond
indices of the cached records, and single points at the QMugs geometries (`qm.given`)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from ksmech.data import qmugs
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.data.schema import Source
from ksmech.qm.batch import featurize_dataset, summarize
from ksmech.qm.bond_index import DEFAULT_BOND_INDEX_DIR, build_bond_indices
from ksmech.qm.given import featurize_qmugs
from ksmech.qm.records import DEFAULT_CACHE_DIR, ReasonCode
from ksmech.qm.xtb_calc import QMConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-qm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="embed, optimize and featurize every unique molecule")
    run.add_argument("--source", required=True, choices=[str(s) for s in Source])
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--n-conformers", type=int, default=1)
    run.add_argument("--optimizer", choices=["tblite", "xtb"], default="tblite")
    run.add_argument("--limit", type=int, default=None, help="random subset, for pilots")
    run.add_argument("--timeout", type=int, default=900, help="seconds per molecule")
    run.add_argument("--maxiter", type=int, default=500, help="geometry optimizer iterations")
    run.add_argument("--scf-maxiter", type=int, default=250, help="SCF cycles per energy")
    run.add_argument("--conn-cov-factor", type=float, default=1.3)
    run.add_argument("--min-success", type=float, default=0.97)
    run.add_argument("--no-retry-failed", action="store_true", help="skip known failures")
    run.add_argument(
        "--retry-reason",
        nargs="+",
        default=None,
        choices=[str(r) for r in ReasonCode],
        metavar="REASON",
        help="retry only the failures with these reason codes",
    )
    run.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    qm = sub.add_parser("qmugs", help="GFN2 single points at the QMugs geometries")
    qm.add_argument("--tar", type=Path, default=qmugs.DEFAULT_TAR)
    qm.add_argument("--conformers", type=int, nargs="+", default=[0], help="QMugs conformers")
    qm.add_argument("--workers", type=int, default=1)
    qm.add_argument("--limit", type=int, default=None, help="the first N conformers")
    qm.add_argument("--chunk", type=int, default=2000, help="conformers read per pass")
    qm.add_argument("--timeout", type=int, default=900, help="seconds per conformer")
    qm.add_argument("--retry-failed", action="store_true")
    qm.add_argument("--cache-dir", type=Path, default=qmugs.CACHE_DIR)
    qm.add_argument("--out-dir", type=Path, default=qmugs.PROCESSED_DIR)
    wbi = sub.add_parser("bond-index", help="Wiberg bond indices of every cached QM record")
    wbi.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    wbi.add_argument("--out-dir", type=Path, default=DEFAULT_BOND_INDEX_DIR)
    return parser


def cmd_bond_index(args: argparse.Namespace, console: Console) -> int:
    counts = build_bond_indices(args.cache_dir, args.out_dir)
    console.print(f"bond indices under {args.out_dir}: {dict(counts.most_common())}")
    return 0 if not any(k.startswith("failed") for k in counts) else 1


def cmd_qmugs(args: argparse.Namespace, console: Console) -> int:
    config = QMConfig(given_geometry=True, timeout_s=args.timeout, cache_dir=args.cache_dir)
    manifest = featurize_qmugs(
        args.tar,
        args.conformers,
        args.workers,
        args.limit,
        args.chunk,
        config,
        args.out_dir,
        args.retry_failed,
        console,
    )
    return 0 if len(manifest) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    if args.command == "bond-index":
        return cmd_bond_index(args, console)
    if args.command == "qmugs":
        return cmd_qmugs(args, console)
    config = QMConfig(
        optimizer=args.optimizer,
        timeout_s=args.timeout,
        maxiter=args.maxiter,
        scf_maxiter=args.scf_maxiter,
        conn_cov_factor=args.conn_cov_factor,
        cache_dir=args.cache_dir,
    )
    try:
        manifest = featurize_dataset(
            args.source,
            n_workers=args.workers,
            n_conformers=args.n_conformers,
            config=config,
            processed_dir=args.processed_dir,
            limit=args.limit,
            retry_failed=not args.no_retry_failed,
            console=console,
            retry_reasons=args.retry_reason,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]; run `ksmech-data ingest --source {args.source}` first")
        return 2
    success = summarize(manifest)["success_fraction"]
    if not success >= args.min_success:
        console.print(
            f"[red]success {success:.2%} is below --min-success {args.min_success:.0%}[/red]"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
