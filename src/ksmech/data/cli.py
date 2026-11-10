"""`ksmech-data`: ingest raw mechanism data into parquet and report dataset statistics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.data import mech_uspto, pmechdb, provenance, reactmech, rmechdb
from ksmech.data.ingest import (
    IngestResult,
    freeze_sample,
    load_frozen_sample,
    write_outputs,
)
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, source_dir, steps_path
from ksmech.data.schema import Source
from ksmech.data.stats import compute_stats

LOADERS: dict[Source, Callable[[Path | None], IngestResult]] = {
    Source.PMECHDB: pmechdb.load,
    Source.REACTMECH: reactmech.load,
    Source.MECH_USPTO: mech_uspto.load,
    Source.RMECHDB: rmechdb.load,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("ingest", "parse + validate raw data"), ("stats", "dataset stats")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--source", required=True, choices=[str(s) for s in LOADERS])
        p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
        if name == "ingest":
            p.add_argument("--raw", type=Path, default=None, help="raw file or dir")
            p.add_argument(
                "--sample", type=int, default=0, help="keep N random accepted steps (pilots)"
            )
            p.add_argument("--seed", type=int, default=0, help="the --sample generator's seed")
    o = sub.add_parser("open-shell-survey", help="subclass every excluded step (read-only)")
    o.add_argument("--source", required=True, choices=[str(s) for s in LOADERS])
    o.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    f = sub.add_parser("freeze-sample", help="snapshot a pilot's step ids (sample_frozen.json)")
    f.add_argument("--source", required=True, choices=[str(s) for s in LOADERS])
    f.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    a = sub.add_parser("leakage-audit", help="rmechdb's leakage audit in both directions (P8)")
    a.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--write-cohorts", action="store_true", help="pin the four overlap cohorts")
    w = sub.add_parser("pathways", help="ReactMech's chains as a sidecar (pathways.parquet)")
    w.add_argument("--source", required=True, choices=[str(Source.REACTMECH)])
    w.add_argument("--raw", type=Path, default=None, help="raw dir (default data/raw/reactmech)")
    w.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    w.add_argument("--workers", type=int, default=4, help="spawned processes for RDKit")
    h = sub.add_parser("human-benchmark", help="the 350 human pathways, evaluation only")
    h.add_argument("--raw", type=Path, default=None, help="350_HumanBenchmark.csv")
    h.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    h.add_argument("--split", default="molecule", help="PMechDB split of the overlap check")
    h.add_argument("--workers", type=int, default=4, help="spawned processes for RDKit")
    h.add_argument("--qm-cache-dir", type=Path, default=Path("data/cache/qm"))
    h.add_argument("--tokens-dir", type=Path, default=Path("data/cache/tokens"))
    h.add_argument(
        "--species-file",
        type=Path,
        nargs="?",
        const=True,  # the flag alone: the default path
        default=None,
        metavar="PATH",
        help="only write the QM batch's species list from the existing qm_coverage.json "
        "(default <processed-dir>/human_benchmark/qm_species.json) and print the batch "
        "commands; nothing else is rebuilt",
    )
    v = sub.add_parser("provenance", help="print an artifact's provenance chain, or backfill")
    v.add_argument("path", type=Path, nargs="?", help="artifact to describe")
    v.add_argument("--backfill", choices=["qm"], help="add headers to existing QM records")
    v.add_argument("--qm-cache-dir", type=Path, default=Path("data/cache/qm"))
    v.add_argument("--depth", type=int, default=4, help="levels of inputs to follow")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    if args.command == "ingest":
        sample = (args.sample, args.seed) if args.sample else None
        return cmd_ingest(Source(args.source), args.raw, args.processed_dir, console, sample)
    if args.command == "provenance":
        return cmd_provenance(args, console)
    if args.command == "open-shell-survey":
        return cmd_survey(Source(args.source), args.processed_dir, console)
    if args.command == "leakage-audit":
        return cmd_leakage(args, console)
    if args.command == "pathways":
        return cmd_pathways(args, console)
    if args.command == "human-benchmark":
        return cmd_human_benchmark(args, console)
    if args.command == "freeze-sample":
        out = freeze_sample(Source(args.source), args.processed_dir)
        console.print(f"wrote {out}")
        return 0
    return cmd_stats(Source(args.source), args.processed_dir, console)


def cmd_leakage(args: argparse.Namespace, console: Console) -> int:
    """`leakage_audit.json` (`data.leakage`): every overlapping step id, in both directions."""
    from ksmech.data.leakage import write_audit

    out, report = write_audit(args.processed_dir, args.seed, args.write_cohorts)
    for key, block in report.items():
        if isinstance(block, dict) and "n" in block:
            console.print(f"{key}: {block['n']}")
        elif key.startswith("rc_heldout"):
            console.print(f"{key}: {block}")
    console.print(f"wrote {out}")
    return 0


def _print_counts(title: str, counts: dict, console: Console) -> None:
    console.print(f"{title}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def cmd_pathways(args: argparse.Namespace, console: Console) -> int:
    """`pathways.parquet` + `pathways_report.json` (`data.pathways`): the chains of the
    mechanism files joined to the ingest's step ids; the steps parquet is not touched."""
    from ksmech.data.pathways import write_pathways

    table, out, report = write_pathways(args.raw, args.processed_dir, args.workers)
    console.print(
        f"{report['chains']} chains, {report['rows']} rows, {report['stop_rows']} stop rows"
    )
    for name, f in report["chain_files"].items():
        _print_counts(name, f, console)
    _print_counts("codes", report["codes"], console)
    _print_counts("ingest status", report["ingest_status"], console)
    _print_counts("maps", report["maps"]["boundaries"], console)
    maps = report["maps"]
    console.print(
        f"whole chains with a boundary {maps['whole_chains_with_a_boundary']}: consistent "
        f"{maps['consistent_chains']}, by renumbering {maps['renumbering_chains']}, "
        f"same numbers {maps['same_number_chains']}"
    )
    console.print(f"wrote {table}\nwrote {out}")
    return 0


def cmd_human_benchmark(args: argparse.Namespace, console: Console) -> int:
    """The evaluation-only table of the PMechRP human benchmark, its overlap with PMechDB's
    split parts and its QM / token coverage (`data.human_benchmark`, `data.benchmark_checks`),
    under `<processed>/human_benchmark/`. Never trained on."""
    from ksmech.data import benchmark_checks as checks
    from ksmech.data import human_benchmark as hb

    out_dir = Path(args.processed_dir) / hb.NAME
    if args.species_file is not None:
        return cmd_benchmark_species(out_dir, args.species_file, console)
    table, report_path, report, pathways = hb.write_benchmark(args.raw or hb.RAW_FILE, out_dir)
    console.print(f"{report['pathways']} pathways, {report['steps']} steps (evaluation only)")
    _print_counts("codes", {k: v for k, v in report["codes"].items() if v}, console)
    ov_path, ov = checks.write_overlap(
        pathways, args.processed_dir, out_dir, args.split, args.workers
    )
    for view, block in ov["pathways"]["train"].items():
        _print_counts(f"pathways vs PMechDB {args.split} train ({view})", block, console)
    cov_path, cov = checks.write_coverage(pathways, out_dir, args.qm_cache_dir, args.tokens_dir)
    _print_counts("QM", cov["qm"], console)
    _print_counts("tokens", cov["tokens"], console)
    for path in (table, report_path, ov_path, cov_path):
        console.print(f"wrote {path}")
    return 0


def cmd_benchmark_species(out_dir: Path, path: Path | bool, console: Console) -> int:
    """`human-benchmark --species-file [PATH]`: the species list of the QM batch and the token
    build from the existing `qm_coverage.json` (`benchmark_checks.write_species_file`), then
    the commands that compute them, in order."""
    from ksmech.data import benchmark_checks as checks
    from ksmech.data.provenance import StaleArtifact

    try:
        out, body = checks.write_species_file(out_dir, None if path is True else path)
    except (FileNotFoundError, StaleArtifact) as exc:
        console.print(f"[red]{exc}[/red]; run `ksmech-data human-benchmark` first")
        return 2
    console.print(f"{body['n_species']} species without a QM record or current tokens")
    _print_counts("left out (no batch completes them)", body["left_out"], console)
    console.print(f"wrote {out}", soft_wrap=True)
    console.print("then run, in order:")
    for line in checks.batch_commands(out):
        console.print(f"  {line}", markup=False, highlight=False, soft_wrap=True)
    return 0


def cmd_survey(source: Source, processed: Path, console: Console) -> int:
    """`open_shell_survey.json`: every excluded step's subclass (`data.survey`)."""
    from ksmech.data.survey import write_survey

    out, result = write_survey(str(source), processed)
    for code, counts in result["counts"].items():
        console.print(f"{code}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if "spin_parity_molecules" in result:
        kinds = result["spin_parity_molecules"]
        console.print("SPIN_PARITY molecules: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    console.print(f"wrote {out}")
    return 0


def cmd_provenance(args: argparse.Namespace, console: Console) -> int:
    """`ksmech-data provenance <path>` prints the chain; `--backfill qm` stamps the QM records
    written before headers existed (run once, never while a QM batch is writing)."""
    if args.backfill == "qm":
        from ksmech.qm.records import backfill_headers

        if not args.qm_cache_dir.is_dir():
            console.print(f"[red]{args.qm_cache_dir} is not a directory[/red]")
            return 2
        counts = backfill_headers(args.qm_cache_dir)
        console.print(f"QM records in {args.qm_cache_dir}: {dict(counts)}")
        return 1 if counts.get("unreadable") else 0
    if args.path is None or not args.path.is_file():
        console.print(f"[red]no such artifact: {args.path}[/red]")
        return 2
    lines = provenance.describe_chain(args.path, max_depth=args.depth)
    for line in lines:
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    return 1 if any(" STALE" in line or "  ! " in line for line in lines) else 0


def cmd_ingest(
    source: Source,
    raw: Path | None,
    processed: Path,
    console: Console,
    sample: tuple[int, int] | None = None,
) -> int:
    """Load, validate and write a source; `sample` (n, seed) keeps n random accepted steps."""
    try:
        result = LOADERS[source](raw)
    except (NotImplementedError, FileNotFoundError) as exc:
        console.print(f"[red]cannot ingest {source}:[/red] {exc}")
        return 2
    if sample is not None:
        result.keep_sample(*sample, frozen=load_frozen_sample(source, processed))
    paths = write_outputs(result, processed)
    print_ingest_report(result, console)
    for kind, path in paths.items():
        console.print(f"wrote {kind}: {path}")
    return 0


def print_ingest_report(result: IngestResult, console: Console) -> None:
    n = result.n_records
    console.print(
        f"[bold]{result.source}[/bold]: {n} records, {len(result.steps)} accepted, "
        f"{len(result.rejections)} rejected ({result.rejected_fraction:.2%})"
    )
    table = Table(title="Rejection reasons")
    table.add_column("reason")
    table.add_column("count", justify="right")
    table.add_column("% of records", justify="right")
    for reason, count in result.reason_counts().most_common():
        table.add_row(reason, str(count), f"{count / n:.2%}")
    if not result.rejections:
        table.add_row("(none)", "0", "0.00%")
    console.print(table)
    if result.notes:
        notes = Table(title="Normalizations / notes (not rejections)")
        notes.add_column("note")
        notes.add_column("count", justify="right")
        for note, count in result.notes.most_common():
            notes.add_row(note, str(count))
        console.print(notes)


def cmd_stats(source: Source, processed: Path, console: Console) -> int:
    path = steps_path(source, processed)
    if not path.is_file():
        console.print(f"[red]{path} not found; run `ksmech-data ingest --source {source}`[/red]")
        return 2
    stats = compute_stats(read_steps(path))
    out = source_dir(source, processed) / "stats.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print_stats(stats, console)
    console.print(f"wrote {out}")
    return 0


def _kv_table(title: str, rows: list[tuple[str, object]]) -> Table:
    table = Table(title=title, show_header=False)
    table.add_column("key")
    table.add_column("value", justify="right")
    for k, v in rows:
        table.add_row(k, "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
    return table


def _hist_table(
    title: str, hist: dict[str, int], label: str, limit: int = 25, counted: str = "steps"
) -> Table:
    table = Table(title=title)
    table.add_column(label)
    table.add_column(counted, justify="right")
    table.add_column("", no_wrap=True)
    top = max(hist.values(), default=1)
    for i, (k, v) in enumerate(hist.items()):
        if i == limit:
            table.add_row(f"... {len(hist) - limit} more", "", "")
            break
        table.add_row(k, str(v), "█" * max(1, round(30 * v / top)))
    return table


def print_stats(stats: dict, console: Console) -> None:
    mols, charged = stats["reactant_molecules"], stats["charged"]
    console.print(
        _kv_table(
            f"Stats: {', '.join(stats['sources'])}",
            [
                ("steps", stats["n_steps"]),
                ("pathways", stats["n_pathways"]),
                ("steps without pathway", stats["n_steps_without_pathway"]),
                ("reactant fragments (total)", mols["total_fragments"]),
                ("reactant molecules (unique)", mols["unique"]),
                ("mean reactant fragments / step", mols["mean_fragments_per_step"]),
                ("intermolecular fraction", stats["intermolecular_fraction"]),
                ("steps with a charged reactant", charged["steps_with_charged_reactant"]),
                ("unique reactants that are charged", charged["unique_reactants_charged"]),
            ],
        )
    )
    if stats["steps_per_pathway"] is not None:
        hist = stats["steps_per_pathway"]["hist"]
        console.print(_hist_table("Steps per pathway", hist, "steps", counted="pathways"))
    console.print(_hist_table("Steps per mechanism class", stats["steps_per_class"], "class"))
    console.print(_hist_table("Arrows per step", stats["arrows_per_step"], "arrows"))
    heavy = stats["heavy_atoms_per_step"]
    quant = ", ".join(f"{k}={heavy[k]:g}" for k in ("min", "p25", "median", "p75", "max", "mean"))
    console.print(_hist_table(f"Heavy atoms per step ({quant})", heavy["hist"], "heavy atoms"))
    console.print(_hist_table("Removed from the target at ingest", stats["domain"], "reason"))
    if stats["charge_repairs_by_rule"]:
        rules = stats["charge_repairs_by_rule"]
        console.print(_hist_table("Repaired at ingest (steps per rule)", rules, "rule"))
    elems = stats["elements"]
    console.print("Elements (steps containing): " + ", ".join(f"{e}:{n}" for e, n in elems.items()))


if __name__ == "__main__":
    sys.exit(main())
