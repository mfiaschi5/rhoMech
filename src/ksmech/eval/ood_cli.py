"""`ksmech-eval ood`: the out-of-distribution report of a checkpoint on one part (`eval.ood`).

    ksmech-eval ood --run rm_sub30k_s0 --run-source reactmech --source pmechdb --split scaffold
    ksmech-eval ood --run pm_v19_s0 --source pmechdb --part val --repeats 3

The checkpoint of `--run` (under `<models-dir>/<run-source>/`) scores the stored samples of
`--source`'s `<split>_<part>` view: the baseline anchor metrics, the symmetry probe (fragment
order), the permutation importance of the pair-level inputs, and the metrics by how often a
sample's edit signature occurs in the run's training part (the views the checkpoint names, in
`--train-processed-dir`, default `--processed-dir`; `--no-coverage` skips it). It writes
`ood_report_<split>_<part>[_<source>][_<cohort>][_n<limit>].json` beside the checkpoint, with
the per-sample baseline rows (`per_sample`).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.arrows.steps import load_samples, sample_ids, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR
from ksmech.data.schema import Source
from ksmech.eval.cohorts import load_cohort, restrict

DEFAULT_MODELS_DIR = Path("data/models")
TOO_LARGE = "TOO_LARGE"


def add_parser(sub) -> None:
    o = sub.add_parser("ood", help="anchor metrics, symmetry and input probes away from training")
    o.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    o.add_argument("--checkpoint", default="best", choices=["best", "last"])
    o.add_argument("--source", default="pmechdb", choices=[str(x) for x in Source])
    o.add_argument("--run-source", help="the models directory of the run (default --source)")
    o.add_argument("--split", default="molecule", help="split strategy of the sample files")
    o.add_argument("--part", default="val", choices=["train", "val", "test"])
    o.add_argument("--limit", type=int, help="the first N samples of the part")
    o.add_argument("--cohort", default="all", help="score only cohort_<name>.json's steps")
    o.add_argument("--repeats", type=int, default=1, help="passes per probe")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--no-coverage", action="store_true", help="skip the signature coverage")
    o.add_argument("--train-processed-dir", type=Path,
                   help="where the run's training views are (default --processed-dir)")  # fmt: skip
    o.add_argument("--allow-stale", action="store_true",
                   help="read the training views for the coverage although stale")  # fmt: skip
    o.add_argument("--device", default="auto", help="auto | cpu | cuda")
    o.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"])
    o.add_argument("--cell-budget", type=int, default=1_100_000)
    o.add_argument("--max-tokens", type=int, default=1024, help="larger samples skipped")
    o.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    o.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)


def _samples(args, path: Path, console: Console) -> tuple[list, dict]:
    """The part's samples (cohort, limit) without those over --max-tokens, and the skips."""
    ids = restrict(sample_ids(path), load_cohort(args.source, args.cohort, args.processed_dir))
    ids = ids[: args.limit] if args.limit is not None else ids
    samples = load_samples(path, ids=ids)
    keep = [s for s in samples if s.n_atoms + s.n_orbitals + 1 <= args.max_tokens]
    keep = [s for s in keep if s.edit_label is not None]
    skipped = {TOO_LARGE: len(samples) - len(keep)} if len(keep) < len(samples) else {}
    for code, count in skipped.items():
        console.print(f"  skipped {code} or unlabelled: {count}")
    return keep, skipped


def training_views(payload: dict, processed_dir: Path) -> tuple[list[Path], int]:
    """(the training views the checkpoint's run names, each source with its split; the size
    of its random training subset, 0 for the whole part)."""
    from ksmech.train.loop import TrainConfig, sources, split_of

    train = payload.get("provenance", {}).get("config", {}).get("train", {})
    cfg = TrainConfig(source=train.get("source", "pmechdb"), split=train.get("split", "molecule"))
    direction = train.get("direction", "forward")
    views = [
        samples_path(s, f"{split_of(cfg, s)}_train", processed_dir, direction) for s in sources(cfg)
    ]
    return views, int(train.get("train_subset", 0) or 0)


def _coverage(args, payload: dict, scores: dict, samples: list, console: Console):
    """(the coverage section, its inputs): the buckets of `eval.ood.coverage` against the
    run's training views; a view that is missing is named and the section says so."""
    from ksmech.eval.ood import coverage, signature, training_signatures

    views, subset = training_views(payload, args.train_processed_dir or args.processed_dir)
    missing = [str(v) for v in views if not v.is_file()]
    if missing:
        console.print(f"[yellow]no coverage: training view missing: {missing}[/yellow]")
        return {"skipped": "TRAIN_VIEW_MISSING", "views": missing}, {}
    train: Counter = Counter()
    inputs: dict[str, str] = {}
    for view in views:
        train += training_signatures(view, args.allow_stale, subset)
        inputs |= prov.file_input(view)
    out = coverage(scores, {s.step_id: signature(s) for s in samples}, train)
    out |= {"train_signatures": len(train), "train_samples": sum(train.values())}
    return out | {"views": [str(v) for v in views], "train_subset": subset}, inputs


def cmd_ood(args, console: Console) -> int:
    from ksmech.eval.ood import ood_report, per_sample
    from ksmech.train.loop import TrainConfig, amp_dtype, load_model, make_loader, select_device

    run_source = args.run_source or args.source
    ckpt = args.models_dir / run_source / args.run / f"{args.checkpoint}.pt"
    device = select_device(args.device)
    model, payload = load_model(ckpt, device)
    path = samples_path(args.source, f"{args.split}_{args.part}", args.processed_dir)
    samples, skipped = _samples(args, path, console)
    cfg = TrainConfig(cell_budget=args.cell_budget, max_tokens=args.max_tokens, workers=0)
    loader = make_loader(samples, cfg, shuffle=False)
    console.print(f"{len(samples)} samples on {device}, {args.repeats} pass(es) per probe")

    def log(section, name, row):
        console.print(f"  {section} {name}: top-1 {row['top1']:.1%} ({row['d_top1']:+.1%}), "
                      f"top anchor changed {row['top_changed']:.1%}")  # fmt: skip

    report, scores = ood_report(model, loader, device, amp_dtype(device, args.amp),
                                args.repeats, args.seed, log)  # fmt: skip
    inputs = prov.file_input(path) | prov.file_input(ckpt)
    if not args.no_coverage:
        report["coverage"], extra = _coverage(args, payload, scores, samples, console)
        inputs |= extra
    report |= {"per_sample": per_sample(scores), "skipped": skipped}
    out = _write(args, ckpt, run_source, report, inputs)
    print_report(console, report, f"{args.run} on {args.source} {args.split} {args.part}")
    console.print(f"wrote {out}")
    return 0


def _write(args, ckpt: Path, run_source: str, report: dict, inputs: dict) -> Path:
    """The report beside the checkpoint, stamped with the view, the checkpoint and the
    training views of the coverage."""
    from ksmech.eval.ood import OOD_VERSION

    names = ("source", "split", "part", "limit", "cohort", "repeats", "seed")
    config = {k: getattr(args, k) for k in names} | {"run_source": run_source, "run": args.run}
    config |= {"checkpoint": args.checkpoint}
    suffix = f"_{args.source}" if args.source != run_source else ""
    suffix += f"_{args.cohort}" if args.cohort != "all" else ""
    suffix += f"_n{args.limit}" if args.limit is not None else ""
    out = ckpt.parent / f"ood_report_{args.split}_{args.part}{suffix}.json"
    return prov.write_json(out, report, prov.stamp("ood_report", OOD_VERSION, inputs, config))


def print_report(console: Console, report: dict, title: str) -> None:
    table = Table(title=f"anchor ranking, {title}")
    for col in ("", "n", "NLL", "top-1", "top-3", "top-5", "R exact", "top changed"):
        table.add_column(col)

    def row(name: str, r: dict, changed: str = "") -> None:
        if not r.get("n"):
            table.add_row(name, "0", *[""] * 6)
            return
        pct = (f"{r[k]:.1%}" for k in ("top1", "top3", "top5", "r_exact"))
        table.add_row(name, str(r["n"]), f"{r['anchor_nll']:.3f}", *pct, changed)

    row("baseline", report["baseline"])
    for section in ("symmetry", "reads"):
        for name, r in report[section].items():
            row(f"{section}: {name}", r, f"{r['top_changed']:.1%}")
    for name, r in report.get("terms", {}).items():
        pct = (f"{r[k]:.1%}" for k in ("top1", "top3", "top5"))
        table.add_row(f"terms: {name}", str(r["n"]), "", *pct, "", "")
    cov = report.get("coverage", {})
    for name in ("seen_often", "seen_rarely", "unseen"):
        if name in cov:
            row(f"coverage: {name} ({cov[name]['share']:.1%})", cov[name])
    console.print(table)
