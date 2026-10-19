"""`ksmech-qm`: GFN2-xTB featurization of the molecules in a processed dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.data.schema import Source
from ksmech.qm.batch import featurize_dataset, summarize
from ksmech.qm.records import DEFAULT_CACHE_DIR
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
    run.add_argument("--conn-cov-factor", type=float, default=1.3)
    run.add_argument("--min-success", type=float, default=0.97)
    run.add_argument("--no-retry-failed", action="store_true", help="skip known failures")
    run.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    config = QMConfig(
        optimizer=args.optimizer,
        timeout_s=args.timeout,
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
