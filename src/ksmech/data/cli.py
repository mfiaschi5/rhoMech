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

from ksmech.data import mech_uspto, pmechdb, reactmech
from ksmech.data.ingest import IngestResult, write_outputs
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, source_dir, steps_path
from ksmech.data.schema import Source
from ksmech.data.stats import compute_stats

LOADERS: dict[Source, Callable[[Path | None], IngestResult]] = {
    Source.PMECHDB: pmechdb.load,
    Source.REACTMECH: reactmech.load,
    Source.MECH_USPTO: mech_uspto.load,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    if args.command == "ingest":
        return cmd_ingest(Source(args.source), args.raw, args.processed_dir, console)
    return cmd_stats(Source(args.source), args.processed_dir, console)


def cmd_ingest(source: Source, raw: Path | None, processed: Path, console: Console) -> int:
    try:
        result = LOADERS[source](raw)
    except (NotImplementedError, FileNotFoundError) as exc:
        console.print(f"[red]cannot ingest {source}:[/red] {exc}")
        return 2
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
    elems = stats["elements"]
    console.print("Elements (steps containing): " + ", ".join(f"{e}:{n}" for e, n in elems.items()))


if __name__ == "__main__":
    sys.exit(main())
