"""`ksmech-arrows`: arrow application, arrow-to-orbital alignment and training samples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.arrows.backward import BACKWARD, DIRECTIONS, FORWARD, direction_dir
from ksmech.arrows.delta_be import label_report, step_label
from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.roundtrip import evaluate_alignment, run_round_trip
from ksmech.arrows.steps import (
    DEFAULT_STRATEGIES,
    STRATEGIES,
    build_all_samples,
    default_strategies,
    write_views_from_store,
)
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, read_training_steps
from ksmech.data.io import steps_path as _steps_path
from ksmech.data.provenance import StaleArtifact
from ksmech.data.schema import Source
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-arrows", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("roundtrip", help="apply dataset arrows, compare with the products")
    a = sub.add_parser("align", help="align arrows to tokens; predicted-arrow round trip")
    s = sub.add_parser("samples", help="sample store (geometry-free pair features) + split views")
    e = sub.add_parser("edit-labels", help="reaction-matrix labels (spec v8) and their coverage")
    f = sub.add_parser("freeze-splits", help="snapshot the split files as frozen splits")
    f.add_argument("--strategy", nargs="+", required=True, choices=STRATEGIES)
    c = sub.add_parser("coverage", help="token and QM coverage of both sides' species (read-only)")
    for p in (r, a, s, e, f, c):
        p.add_argument("--source", required=True, choices=[str(x) for x in Source])
        p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    r.add_argument("--include-resonance", action="store_true", help="also resonance steps")
    for p in (a, s, e, c):
        p.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    for p in (s, e):
        p.add_argument("--direction", default=FORWARD, choices=DIRECTIONS,
                       help="backward: each step seen from its child, in backward/")  # fmt: skip
        p.add_argument("--limit", type=int, default=0, help="backward only: that many training "
                       "steps drawn at random, a pilot")  # fmt: skip
    a.add_argument("--no-member-check", action="store_true", help="skip the set-member swap")
    s.add_argument(
        "--strategy",
        nargs="+",
        default=None,
        choices=STRATEGIES,
        help=f"default {' '.join(DEFAULT_STRATEGIES)} (rmechdb: rc_molecule flower_given); "
        "random_leaky (the per-step split) only when named",
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--views-only",
        action="store_true",
        help="write the strategies' views over the store already built, without rebuilding it",
    )
    for p in (a, s, c):
        p.add_argument(
            "--workers", type=int, default=1, help="spawned processes (host total <= 15)"
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    path = _steps_path(args.source, args.processed_dir)
    if not path.exists():
        console.print(f"[red]{path} missing[/red]; run `ksmech-data ingest --source {args.source}`")
        return 2
    commands = {
        "roundtrip": cmd_roundtrip,
        "align": cmd_align,
        "samples": cmd_samples,
        "edit-labels": cmd_edit_labels,
        "freeze-splits": cmd_freeze_splits,
        "coverage": cmd_coverage,
    }
    if getattr(args, "limit", 0) and args.direction == FORWARD:
        console.print("[red]--limit builds a partial store or report: --direction backward only")
        return 2
    return commands[args.command](args, path, console)


def cmd_freeze_splits(args: argparse.Namespace, path: Path, console: Console) -> int:
    """`split_frozen_<strategy>.json` from each named strategy's current split file
    (`arrows.steps.freeze_split_file`): from then on `samples` keeps every frozen step in its
    part, places new groups by hash and counts SPLIT_CONFLICT."""
    from ksmech.arrows.steps import freeze_split_file

    for strategy in args.strategy:
        try:
            out = freeze_split_file(args.source, strategy, args.processed_dir)
        except (FileExistsError, FileNotFoundError, StaleArtifact) as exc:
            console.print(f"[red]{strategy}: {exc}[/red]")
            return 1
        console.print(f"wrote {out}")
    return 0


def _write(report: dict, args: argparse.Namespace, name: str, console: Console) -> Path:
    """`name` in the directory of the command's direction (`backward.direction_dir`)."""
    out = direction_dir(args.source, args.processed_dir, getattr(args, "direction", FORWARD))
    out.mkdir(parents=True, exist_ok=True)
    out = out / name
    out.write_text(json.dumps(report, indent=1) + "\n")
    console.print(f"wrote {out}")
    return out


def print_failures(summary: dict, title: str, console: Console) -> None:
    console.print(
        f"{title}: {summary['n_match']}/{summary['n_steps']} = {summary['match_rate']:.2%} "
        f"({', '.join(f'{k} {v}' for k, v in summary['status'].items())})"
    )
    table = Table(title=f"{title}: top failure patterns")
    for col in ("count", "pattern", "example step_ids"):
        table.add_column(col)
    for row in summary["top_failures"]:
        table.add_row(str(row["count"]), row["pattern"], "\n".join(row["examples"]))
    console.print(table)


def cmd_roundtrip(args: argparse.Namespace, path: Path, console: Console) -> int:
    steps = read_steps(path) if args.include_resonance else read_training_steps(path)
    _, summary = run_round_trip(steps)
    print_failures(summary, "dataset-arrow round trip", console)
    _write(summary, args, "arrows_roundtrip.json", console)
    return 0 if summary["match_rate"] >= 0.95 else 1


def print_alignment(report: dict, console: Console) -> None:
    console.print(
        f"aligned {report['n_steps_aligned_or_tried']} steps ({report['n_steps_complete']} "
        f"complete), skipped {report['steps_skipped']}; arrows {report['n_arrows']}: "
        f"{report['n_labels']} labels, {report['n_absorbed_heterolysis']} absorbed, "
        f"{report['n_unaligned']} unaligned -> {report['arrows_aligned_fraction']:.2%} aligned; "
        f"mean set sizes donor {report['mean_donor_set']:.2f}, "
        f"acceptor {report['mean_acceptor_set']:.2f}"
    )
    console.print("flags: " + ", ".join(f"{k} {v:.2%}" for k, v in report["flag_fraction"].items()))
    table = Table(title="unaligned arrows by (reason, source>sink kind, element pair)")
    for col in ("count", "reason", "group"):
        table.add_column(col)
    for row in report["unaligned_groups"]:
        table.add_row(str(row["count"]), row["reason"], row["group"])
    console.print(table)


def cmd_align(args: argparse.Namespace, path: Path, console: Console) -> int:
    report = evaluate_alignment(
        read_training_steps(path),
        TokenCache(args.tokens_dir),
        not args.no_member_check,
        n_workers=args.workers,
    )
    print_alignment(report, console)
    print_failures(report["predicted"], "predicted-arrow round trip", console)
    console.print(f"set-member swaps: {report['predicted']['member_agreement']}")
    _write(report, args, "alignment_report.json", console)
    ok = report["arrows_aligned_fraction"] >= 0.90 and report["predicted"]["match_rate"] >= 0.95
    return 0 if ok and report["flag_fraction"]["ALIGN_FALLBACK"] <= 0.10 else 1


def cmd_samples(args: argparse.Namespace, path: Path, console: Console) -> int:
    args.strategy = list(args.strategy or default_strategies(args.source))
    if args.views_only:
        return _views_only(args, console)
    report = build_all_samples(
        args.source,
        args.strategy,
        args.processed_dir,
        TokenCache(args.tokens_dir),
        args.seed,
        n_workers=args.workers,
        direction=args.direction,
        limit=args.limit,
    )
    built = f" (a pilot of {report['n_built']})" if args.limit else ""
    console.print(
        f"{args.direction} samples: {report['n_samples']} from {report['n_training_steps']} "
        f"training steps{built}; skipped {report['skipped']}"
    )
    console.print(
        f"edit-label classes {report['class_size']}, anchors {report['anchor_kind']}; "
        f"orbital reading {report['orbital_reading']}"
    )
    if report["label_filter"]:
        console.print("labeled pairs that pair_valid rejects, by group (LABEL_FILTERED):")
        for group, n in report["label_filter"].items():
            console.print(f"  {n:6d}  {group}")
    for split, r in report["splits"].items():
        console.print(f"  {split}: {r['n_samples']}/{r['n_steps']} samples, skipped {r['skipped']}")
    for strategy, why in report["strategies_not_applicable"].items():
        console.print(f"  [yellow]{strategy}: not applicable ({why})[/yellow]")
    _print_leakage(report, console)
    _write(report, args, "samples_report.json", console)
    return 0 if report["splits"] else 1


def _print_leakage(report: dict, console: Console) -> None:
    """The backward inputs of each held-out part that share a molecule with training."""
    for strategy, parts in report.get("leakage", {}).items():
        text = ", ".join(f"{p} {r['n_leaking']}/{r['n_steps']}" for p, r in parts.items())
        console.print(f"  leakage of the backward inputs, {strategy}: {text}")


def _views_only(args: argparse.Namespace, console: Console) -> int:
    report = write_views_from_store(
        args.source, args.strategy, args.processed_dir, args.seed, args.direction
    )
    _print_leakage(report, console)
    for split, r in report["splits"].items():
        console.print(f"  {split}: {r['n_samples']}/{r['n_steps']} samples, skipped {r['skipped']}")
    for strategy, why in report["strategies_not_applicable"].items():
        console.print(f"  [yellow]{strategy}: not applicable ({why})[/yellow]")
    _write(report, args, f"samples_report_{'+'.join(args.strategy)}.json", console)
    return 0 if report["splits"] else 1


EDIT_LABEL_GATE = 0.95  # labeled fraction of the training steps


def cmd_edit_labels(args: argparse.Namespace, path: Path, console: Console) -> int:
    """Label every training step with its reaction matrix (spec v8) on the token atoms; the
    data gate is the labeled fraction. Every class member reproduces its step by construction:
    one that does not is dropped and counted (`steps_with_rejected_variants`). Backward: the
    steps seen from their child (`backward.backward_step`), with how each class compares with
    the negated forward class (`backward.label_agreement`), written to backward/."""
    from ksmech.arrows.backward import backward_agreement, oriented, pilot_steps

    loader = TokenCache(args.tokens_dir)
    steps = pilot_steps(read_training_steps(path), args.limit)
    labels = {s.step_id: step_label(s, loader) for s in oriented(steps, args.direction)}
    report = label_report(labels)
    if args.direction == BACKWARD:
        report |= {"direction": BACKWARD, "limit": args.limit, **backward_agreement(steps)}
        console.print(f"backward class vs negated forward class: {report['agreement']}")
    console.print(
        f"{report['n_labeled']} / {report['n_steps']} training steps labeled "
        f"({report['labeled_fraction']:.2%}); skipped {report['skipped']}"
    )
    console.print(
        f"class size {report['class_size']}, drawn R parsimonious in "
        f"{report['data_parsimonious']}, resonance enumeration truncated in "
        f"{report['resonance_truncated']}, anchors {report['anchor_kind']}"
    )
    _write(report, args, "edit_labels_report.json", console)
    return 0 if report["labeled_fraction"] >= EDIT_LABEL_GATE else 1


def cmd_coverage(args: argparse.Namespace, path: Path, console: Console) -> int:
    """Token and QM coverage of the species of both sides of the training steps (the product
    species are the backward samples' inputs; reaction-network plan, item H), read through the
    token loader and the QM manifest; written to backward/species_coverage.json."""
    from ksmech.arrows.backward import species_coverage
    from ksmech.data.io import source_dir

    manifest = source_dir(args.source, args.processed_dir) / "qm_manifest.parquet"
    steps = read_training_steps(path)
    loader = TokenCache(args.tokens_dir)
    report = species_coverage(steps, loader, manifest, args.workers)
    for side in ("reactants", "products", "products_only"):
        r = report[side]
        served = f"; steps fully served {r['n_steps_served']}" if "n_steps_served" in r else ""
        console.print(
            f"{side}: {r['n_with_tokens']} / {r['n_species']} species with tokens (missing "
            f"{r['missing_tokens']}), {r['n_without_qm']} without a QM record "
            f"{r['without_qm']}{served}"
        )
    args.direction = BACKWARD
    _write(report, args, "species_coverage.json", console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
