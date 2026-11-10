"""`ksmech-qm`: GFN2-xTB featurization of the molecules in a processed dataset, the Wiberg bond
indices of the cached records, single points at the QMugs geometries (`qm.given`), the
thermochemistry sidecars of the cached records (`qm.thermo_batch`), and the SQLite index of a
cache root (`qm.index`)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.data import qmugs
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.data.schema import Source
from ksmech.qm import index
from ksmech.qm.batch import collect_molecules, featurize_dataset, summarize
from ksmech.qm.bond_index import DEFAULT_BOND_INDEX_DIR, build_bond_indices
from ksmech.qm.given import featurize_qmugs
from ksmech.qm.records import DEFAULT_CACHE_DIR, ReasonCode
from ksmech.qm.thermo_batch import make_tasks, print_summary, run_thermo, write_report
from ksmech.qm.thermo_cache import DEFAULT_THERMO_DIR, ThermoConfig
from ksmech.qm.thermo_reopt import REOPT_FMAX, REOPT_MAXITER, REOPT_ROUNDS, REOPT_THRESHOLD_CM
from ksmech.qm.xtb_calc import ALPB_SOLVENTS, GAS, QMConfig, solvent_option

DEFAULT_ROOT = DEFAULT_CACHE_DIR.parent  # data/cache: qm/, tokens/, bond_index/, thermo/
SOLVENT_CHOICES = (*ALPB_SOLVENTS, GAS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-qm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
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
    _add_thermo_parser(sub)
    idx = sub.add_parser("index", help="the SQLite index of a cache root, read from file headers")
    idx.add_argument("action", choices=["refresh", "stats"])
    idx.add_argument(
        "--root",
        type=Path,
        action="append",
        help=f"a root holding qm/, tokens/, bond_index/, thermo/ (repeatable; default "
        f"{DEFAULT_ROOT})",
    )
    idx.add_argument("--rebuild", action="store_true", help="refresh: empty the index first")
    idx.add_argument("--no-refresh", action="store_true", help="stats: read the index as it is")
    return parser


def _add_run_parser(sub) -> None:
    run = sub.add_parser("run", help="embed, optimize and featurize every unique molecule")
    run.add_argument("--source", required=True, choices=[str(s) for s in Source])
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--n-conformers", type=int, default=1)
    run.add_argument("--optimizer", choices=["tblite", "tblite-anc", "xtb"], default="tblite")
    run.add_argument(
        "--solvent",
        choices=SOLVENT_CHOICES,
        default="water",
        help="ALPB solvent of every calculation, or gas; part of the method and the cache key",
    )
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
    run.add_argument(
        "--steps-from",
        type=Path,
        default=None,
        help="only the molecules of the steps a JSON file lists (its 'qm_steps', else 'ids')",
    )
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)


def _add_thermo_parser(sub) -> None:
    th = sub.add_parser("thermo", help="frequencies and qRRHO thermochemistry of cached records")
    pick = th.add_mutually_exclusive_group(required=True)
    pick.add_argument("--source", nargs="+", choices=[str(s) for s in Source])
    pick.add_argument("--smiles", nargs="+", help="species, canonicalized map-free")
    th.add_argument("--steps-from", type=Path, default=None, help="with one --source: its steps")
    th.add_argument("--workers", type=int, default=1)
    th.add_argument("--limit", type=int, default=None, help="random subset (seed 0), for pilots")
    th.add_argument("--optimizer", choices=["tblite", "tblite-anc", "xtb"], default="tblite")
    th.add_argument("--solvent", choices=SOLVENT_CHOICES, default="water")
    th.add_argument("--timeout", type=int, default=3600, help="seconds per molecule")
    th.add_argument("--scf-maxiter", type=int, default=250, help="SCF cycles per gradient")
    th.add_argument("--retry-failed", action="store_true", help="ignore the tombstones")
    th.add_argument(
        "--reopt-threshold",
        type=float,
        default=REOPT_THRESHOLD_CM,
        help="cm^-1: re-optimize along the imaginary modes below it (below -1)",
    )
    th.add_argument(
        "--reopt-rounds",
        type=int,
        default=REOPT_ROUNDS,
        help="re-optimization rounds at most; 0: xtb's rule on the record's modes",
    )
    th.add_argument("--reopt-fmax", type=float, default=REOPT_FMAX, help="Hartree/Bohr")
    th.add_argument(
        "--reopt-maxiter", type=int, default=REOPT_MAXITER, help="energies per re-optimization"
    )
    th.add_argument("--temperatures", type=float, nargs="+", default=[298.15], help="K, report")
    th.add_argument("--report", type=Path, default=None, help="every row as JSON")
    th.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    th.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="the QM records")
    th.add_argument("--out-dir", type=Path, default=DEFAULT_THERMO_DIR)


def steps_from(path: Path | None) -> list[str] | None:
    """The step ids of a JSON file: its `qm_steps` list (`p0a_changes.json`: the re-charged
    kept steps), else its `ids` (a view or a cohort); None without a file."""
    if path is None:
        return None
    body = json.loads(Path(path).read_text())
    if "qm_steps" not in body and "ids" not in body:
        raise ValueError(f"{path} has neither 'qm_steps' nor 'ids'")
    return list(body.get("qm_steps", body.get("ids")))


def thermo_species(args: argparse.Namespace) -> list[str]:
    """The map-free canonical SMILES `ksmech-qm thermo` runs on: `--smiles`, or every species of
    the `--source` datasets (of the `--steps-from` steps with one source), `--limit` of them at
    random (seed 0)."""
    from rdkit import Chem

    from ksmech.data import chem

    if args.smiles:
        species = sorted({chem.unmapped_canonical(Chem.MolFromSmiles(s)) for s in args.smiles})
    else:
        if args.steps_from is not None and len(args.source) != 1:
            raise ValueError("--steps-from needs exactly one --source")
        ids = steps_from(args.steps_from)
        found = set()
        for source in args.source:
            found |= set(collect_molecules(source, args.processed_dir, ids)[0])
        species = sorted(found)
    if args.limit is not None and args.limit < len(species):
        species = sorted(random.Random(0).sample(species, args.limit))
    return species


def thermo_config(args: argparse.Namespace) -> ThermoConfig:
    """The `ThermoConfig` of `ksmech-qm thermo`'s options; ValueError on a bad threshold."""
    return ThermoConfig(
        scf_maxiter=args.scf_maxiter,
        timeout_s=args.timeout,
        reopt_threshold_cm=args.reopt_threshold,
        reopt_rounds=args.reopt_rounds,
        reopt_fmax=args.reopt_fmax,
        reopt_maxiter=args.reopt_maxiter,
        qm_dir=args.cache_dir,
        out_dir=args.out_dir,
    )


def cmd_thermo(args: argparse.Namespace, console: Console) -> int:
    try:
        species = thermo_species(args)
        config = thermo_config(args)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    qm = QMConfig(optimizer=args.optimizer, solvent=solvent_option(args.solvent))
    tasks = make_tasks(species, qm.method, config, memo=not args.retry_failed)
    console.print(f"records {qm.method} under {args.cache_dir}, sidecars under {args.out_dir}")
    rows = run_thermo(tasks, args.workers, console)
    print_summary(rows, console)
    if args.report is not None:
        console.print(f"wrote {write_report(rows, args.report, args.temperatures)}")
    return 1 if any(r["status"] == "failed" for r in rows) else 0


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


def _print_refresh(root: Path, counts: Counter, console: Console) -> None:
    table = Table(title=f"index of {root}: refresh")
    for col in ("kind", "added", "updated", "removed", "unchanged", "problems"):
        table.add_column(col, justify="left" if col == "kind" else "right")
    for kind, _, _ in index.LAYOUT:
        problems = sum(n for k, n in counts.items() if k.startswith(f"{kind}:problem:"))
        cells = [str(counts[f"{kind}:{what}"]) for what in ("added", "updated", "removed")]
        table.add_row(kind, *cells, str(counts[f"{kind}:unchanged"]), str(problems))
    console.print(table)
    for key, n in sorted(counts.items()):
        if ":problem:" in key:
            console.print(f"[red]{key}: {n}[/red]")


def _print_stats(root: Path, s: dict, console: Console) -> None:
    table = Table(title=f"index of {root} (refreshed {s['refreshed_at']})")
    for col in ("kind", "files", "GB", "problems", "dirty builds"):
        table.add_column(col, justify="left" if col == "kind" else "right")
    for kind, n, size, problems, dirty in s["kinds"]:
        table.add_row(kind, str(n), f"{(size or 0) / 1e9:.2f}", str(problems), str(dirty or 0))
    console.print(table)
    for kind, method, n in s["methods"]:
        console.print(f"  {kind} {method}: {n}")
    for kind, source, version, unpaired, n in s.get("spin", []):
        console.print(f"  {kind} tagged {version}, spin_source {source}, u={unpaired}: {n}")
    if s["reasons"]:
        reasons = Table(title="remembered failures")
        for col in ("reason", "count", "example"):
            reasons.add_column(col)
        for reason, n, example in s["reasons"]:
            reasons.add_row(reason, str(n), str(example)[:60])
        console.print(reasons)
    if s["thermo_reasons"]:
        reasons = Table(title="remembered thermo failures")
        for col in ("reason", "count", "example"):
            reasons.add_column(col)
        for reason, n, example in s["thermo_reasons"]:
            reasons.add_row(reason, str(n), str(example)[:60])
        console.print(reasons)
    for warning, n in s["thermo_warnings"]:
        console.print(f"  thermo warning {warning}: {n}")
    for kind, problem, n in s["problems"]:
        console.print(f"[red]{kind} {problem}: {n}[/red]")
    console.print(
        f"records without tokens {s['records_without_tokens']}, tokens without a Wiberg sidecar "
        f"{s['tokens_without_bond_index']}, tombstones next to a record "
        f"{s['failures_with_a_record']}, records without a thermo sidecar "
        f"{s['records_without_thermo']}"
    )


def cmd_index(args: argparse.Namespace, console: Console) -> int:
    status = 0
    for root in args.root or [DEFAULT_ROOT]:
        if not root.is_dir():
            console.print(f"[red]{root}: no such directory[/red]")
            status = 2
            continue
        if args.action == "refresh" or not args.no_refresh:
            _print_refresh(root, index.refresh(root, rebuild=args.rebuild), console)
        if args.action == "stats":
            _print_stats(root, index.stats(root), console)
    return status


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    if args.command == "bond-index":
        return cmd_bond_index(args, console)
    if args.command == "index":
        return cmd_index(args, console)
    if args.command == "qmugs":
        return cmd_qmugs(args, console)
    if args.command == "thermo":
        return cmd_thermo(args, console)
    config = QMConfig(
        optimizer=args.optimizer,
        solvent=solvent_option(args.solvent),
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
            step_ids=steps_from(args.steps_from),
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
