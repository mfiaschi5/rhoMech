"""`ksmech-arrows`: arrow application, arrow-to-orbital alignment and training samples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.arrows.contacts import TRAIN_SUBSAMPLE, build_all_contacts
from ksmech.arrows.delta_be import label_report, step_label
from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.roundtrip import evaluate_alignment, run_round_trip
from ksmech.arrows.steps import DEFAULT_STRATEGIES, STRATEGIES, build_all_samples
from ksmech.data.io import DEFAULT_PROCESSED_DIR, read_steps, read_training_steps, source_dir
from ksmech.data.io import steps_path as _steps_path
from ksmech.data.schema import Source
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
from ksmech.physics.contact.legacy_fixed23 import DEFAULT_CACHE_DIR as ORIENTED_CACHE_DIR
from ksmech.physics.contact_kernel import KERNELS, ContactConfig, ContactKernel
from ksmech.physics.contact_table import DEFAULT_TABLE_PATH, load_tables
from ksmech.physics.pairs import PairKernels, oriented_kernels, single_kernel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-arrows", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("roundtrip", help="apply dataset arrows, compare with the products")
    a = sub.add_parser("align", help="align arrows to tokens; predicted-arrow round trip")
    s = sub.add_parser("samples", help="sample store (stage-A pair features) + split views")
    c = sub.add_parser("contacts", help="legacy contact sidecar: val, test, a train subsample")
    e = sub.add_parser("edit-labels", help="reaction-matrix labels (spec v8) and their coverage")
    for p in (r, a, s, c, e):
        p.add_argument("--source", required=True, choices=[str(x) for x in Source])
        p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    r.add_argument("--include-resonance", action="store_true", help="also resonance steps")
    for p in (a, s, c, e):
        p.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    c.add_argument("--kernel", choices=KERNELS, default="oriented", help="contact kernel")
    c.add_argument("--oriented-cache", type=Path, default=ORIENTED_CACHE_DIR)
    c.add_argument("--contact-table", type=Path, default=DEFAULT_TABLE_PATH)
    c.add_argument("--split", default="molecule", help="split strategy (its split file)")
    c.add_argument("--n-train", type=int, default=TRAIN_SUBSAMPLE, help="training subsample")
    c.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-member-check", action="store_true", help="skip the set-member swap")
    s.add_argument(
        "--strategy",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=STRATEGIES,
        help="random_leaky (the per-step split) only when named",
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=1, help="processes (keep the host total <= 4)")
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
        "contacts": cmd_contacts,
        "edit-labels": cmd_edit_labels,
    }
    return commands[args.command](args, path, console)


def _write(report: dict, args: argparse.Namespace, name: str, console: Console) -> Path:
    out = source_dir(args.source, args.processed_dir) / name
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
        read_training_steps(path), TokenCache(args.tokens_dir), not args.no_member_check
    )
    print_alignment(report, console)
    print_failures(report["predicted"], "predicted-arrow round trip", console)
    console.print(f"set-member swaps: {report['predicted']['member_agreement']}")
    _write(report, args, "alignment_report.json", console)
    ok = report["arrows_aligned_fraction"] >= 0.90 and report["predicted"]["match_rate"] >= 0.95
    return 0 if ok and report["flag_fraction"]["ALIGN_FALLBACK"] <= 0.10 else 1


def samples_kernels(args: argparse.Namespace) -> PairKernels:
    """The contact kernels of the legacy sidecar: the oriented kernel at every feature
    distance by default; an ablation kernel (table, analytic) for all."""
    if args.kernel == "oriented":
        return oriented_kernels(args.oriented_cache)
    tables = load_tables(args.contact_table) if args.kernel == "table" else None
    return single_kernel(ContactKernel(tables, ContactConfig(kernel=args.kernel)))


def cmd_samples(args: argparse.Namespace, path: Path, console: Console) -> int:
    report = build_all_samples(
        args.source,
        args.strategy,
        args.processed_dir,
        TokenCache(args.tokens_dir),
        args.seed,
        n_workers=args.workers,
    )
    console.print(
        f"{report['n_samples']} samples from {report['n_training_steps']} training steps; "
        f"skipped {report['skipped']}"
    )
    if report["label_filter"]:
        console.print("labeled pairs that pair_valid rejects, by group (LABEL_FILTERED):")
        for group, n in report["label_filter"].items():
            console.print(f"  {n:6d}  {group}")
    for split, r in report["splits"].items():
        console.print(f"  {split}: {r['n_samples']}/{r['n_steps']} samples, skipped {r['skipped']}")
    for strategy, why in report["strategies_not_applicable"].items():
        console.print(f"  [yellow]{strategy}: not applicable ({why})[/yellow]")
    _write(report, args, "samples_report.json", console)
    return 0 if report["splits"] else 1


def cmd_contacts(args: argparse.Namespace, path: Path, console: Console) -> int:
    try:
        kernels = samples_kernels(args)
        report = build_all_contacts(
            args.source,
            kernels,
            args.split,
            args.processed_dir,
            TokenCache(args.tokens_dir),
            args.n_train,
            args.seed,
        )
    except FileNotFoundError as exc:  # no split file (run `samples`) or no contact table
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(
        f"{args.split}: {report['n_entries']} / {report['n_steps']} steps with contacts "
        f"({report['n_held_out']} val + test, {report['n_train_subsample']} training), "
        f"skipped {report['skipped']}"
    )
    _write(report, args, f"contacts_report_{args.split}.json", console)
    return 0 if report["n_entries"] else 1


EDIT_LABEL_GATE = 0.95  # labeled fraction of the training steps


def cmd_edit_labels(args: argparse.Namespace, path: Path, console: Console) -> int:
    """Label every training step with its reaction matrix (spec v8) on the token atoms; the
    data gate is the labeled fraction, and no class member may fail to reproduce its step."""
    loader = TokenCache(args.tokens_dir)
    labels = {step.step_id: step_label(step, loader) for step in read_training_steps(path)}
    report = label_report(labels)
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
    ok = report["labeled_fraction"] >= EDIT_LABEL_GATE
    return 0 if ok and "VARIANT_MISMATCH" not in report["skipped"] else 1


if __name__ == "__main__":
    sys.exit(main())
