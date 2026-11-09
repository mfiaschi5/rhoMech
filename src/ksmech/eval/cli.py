"""`ksmech-eval`: evaluation of anchors and reaction matrices (spec v9, "Evaluation protocol").

    ksmech-eval floor --source pmechdb --split molecule
    ksmech-eval embedding --run flex_first --step-id pmechdb-018a36c64ec74e5e
    ksmech-eval embedding --run flex_first --smiles 'CC(=O)C.[OH-]'
    ksmech-eval decode --run flex_first --part val --limit 200
    ksmech-eval importance --run pm_v19_s0 --part val
    ksmech-eval pin --source pmechdb --name closed_base --workers 4
    ksmech-eval cohort --source pmechdb --name open_shell \
        --domain-class RADICAL_SPECTATOR RADICAL_REACTIVE
    ksmech-eval compare --a <GENERAL hit files> --b <LEGACY hit files> --out FILE

`floor` ranks each step's anchor by |G^atom| (no learned parameter) on the validation and test
parts and writes `anchor_floor_report_<split>.json`, the reference every trained model is
compared with.

`decode` runs the beam decoder (`model.decode`) on the stored samples of one part and scores the
full reaction matrix: a hit at k when one of the k best completions is a variant of the label
class (`eval.decode`). It writes `decode_report_<split>_<part>[_n<limit>].json` next to the
checkpoint. `--caps legacy` (the default: at most 8 bond and 2 atom changes, 48 candidates)
or `--caps labels` (the label maxima of the training part of --source, never val or test, so
each cohort of a joint run decodes with its own source's caps), then `--max-bonds`,
`--max-atoms`, `--max-candidates`; the header records the caps, and caps other than the legacy
ones add `_caps<bonds>-<atoms>-<candidates>` to the file names. `--closed-minus1-filter`
refuses any -1 atom entry on a pool with no unpaired electron (`model.decode`, an option whose
default P9's ablation sets) and adds `_minus1filter` to the file names; the closed-shell screen
is always on. A run's selection note (`train.loop.run_notes`) is copied into the report.
`--direction backward` decodes the previous-step samples (`arrows.backward`): B_parent =
B_child + R on the child's Lewis structure, the same laws; the config records the direction and
the file names carry `_backward`.

`importance` permutes each token input column, and each block of related columns, across the
tokens of a part and scores the part again with the training metrics (`eval.importance`): how
much the anchor NLL rises and top-1 / top-3 / the greedy R fall without that input. It writes
`token_importance_<split>_<part>[_n<limit>].json` next to the checkpoint.

`pin` pins every sample of a split's three parts as a cohort, its two GBDT subsets (the
samples with ArrowLabels on the valid cells and on every cell) and `baseline_<name>.json`, the
per-sample digests of every named column (`eval.pin`); each file is written once.

`cohort` pins the training steps whose `domain_class` holds one of the given classes as a
cohort (`eval.cohorts.class_cohort_ids`), written once; an empty cohort is refused.

`compare` pairs two arms of decode runs sample by sample (`eval.paired`): arm A fails when its
mean top-k is lower than B's by more than max(2 x pooled SD, 1 point); beside it, the exact
one-sided McNemar test on the majority-of-seeds hits and on every seed pair. Hit files decoded
on other samples or another way are refused.

`embedding` traces one sample through a trained model (`eval.embedding`) and writes one
self-contained HTML page (`eval.embedding_page`): the tokens on the Lewis structure, the parts
of h^(0), every layer's attention and token updates, and the decoder's anchor terms. The sample
is a stored one (`--step-id`, searched in val, test, train; or `--index` of `--part`) or a pool
given as SMILES, whose missing tokens are computed first. The page goes to
`data/models/<source>/<run>/embedding/` unless `--out`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from ksmech.arrows.backward import BACKWARD, DIRECTIONS, FORWARD, require_direction
from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.steps import MechanismSample, load_samples, sample_ids, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir, steps_path
from ksmech.data.schema import Source
from ksmech.eval.cohorts import (
    SLICES,
    by_r_shape,
    covered_fraction,
    load_cohort,
    restrict,
    sliced,
)
from ksmech.eval.floor import floor_report
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
from ksmech.train.loop import DEFAULT_MODELS_DIR

FLOOR_VERSION = 4  # 3: --cells valid|all (pooling over the pair_valid cells or every cell);
#   4: --floor legacy|own_break, --kernel g|g2, --drop-intra-open (open-shell plan P7)
DECODE_VERSION = 5  # 3: decoder caps legacy | labels, recorded in the header; 4: the closed-shell
#   screen, --closed-minus1-filter, odd_top1 and the screened counts per sample (open-shell P9);
#   5: rank_any against every product the source records for the pool (network plan, item A)


def _floor_options(f) -> None:
    f.add_argument(
        "--floor",
        default="legacy",
        choices=["legacy", "own_break"],
        help="legacy (each pair once by |G^atom|) or own_break (break anchors by |B^own|, P7)",
    )
    f.add_argument("--kernel", default="g", choices=["g", "g2"], help="the pooled gap column")
    f.add_argument(
        "--drop-intra-open",
        action="store_true",
        help="leave the open-shell rows and hole columns within one fragment out of the pooling",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("floor", help="the |G^atom| anchor ranking on val and test")
    f.add_argument("--source", required=True, choices=[str(x) for x in Source])
    f.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    f.add_argument("--split", default="molecule", help="split strategy of the sample files")
    f.add_argument(
        "--cells",
        default="valid",
        choices=["valid", "all"],
        help="pool G^atom over the pair_valid cells (v17) or over every (occupied, virtual) cell",
    )
    _floor_options(f)
    _cohort_args(f)
    _decode_parser(sub)
    _importance_parser(sub)
    c = sub.add_parser("compare", help="two arms of decode runs, paired: tolerance and McNemar")
    c.add_argument("--a", nargs="+", type=Path, required=True, help="arm A's hit files (GENERAL)")
    c.add_argument("--b", nargs="+", type=Path, required=True, help="arm B's hit files (LEGACY)")
    c.add_argument("--k", type=int, default=1, help="top-k of a hit")
    c.add_argument("--metric", default="own", choices=["own", "any"], help="own: the sample's "
                   "label; any: any product the source records for the pool")  # fmt: skip
    c.add_argument("--out", type=Path, help="the report (JSON)")
    pin = sub.add_parser("pin", help="pin a split's samples as a cohort with per-column digests")
    pin.add_argument("--source", required=True, choices=[str(x) for x in Source])
    pin.add_argument("--name", required=True, help="cohort name, e.g. closed_base")
    pin.add_argument("--split", default="molecule", help="split strategy of the sample files")
    pin.add_argument("--workers", type=int, default=1, help="spawned processes reading shards")
    pin.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    k = sub.add_parser("cohort", help="pin the training steps of some domain_class values")
    k.add_argument("--source", required=True, choices=[str(x) for x in Source])
    k.add_argument("--name", required=True, help="cohort name, e.g. open_shell")
    k.add_argument("--domain-class", nargs="+", required=True, help="classes, any of which")
    k.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    e = sub.add_parser("embedding", help="one sample through a trained model, as an HTML page")
    e.add_argument("--run", required=True, help="training run under <models-dir>/<source>/")
    e.add_argument("--checkpoint", default="best", choices=["best", "last"])
    e.add_argument("--source", default="pmechdb", choices=[str(x) for x in Source])
    e.add_argument("--split", default="molecule", help="split strategy of the sample files")
    e.add_argument("--part", choices=["train", "val", "test"], help="where to find the sample")
    pick = e.add_mutually_exclusive_group()
    pick.add_argument("--step-id", help="a stored sample (default: the first of --part)")
    pick.add_argument("--index", type=int, help="the k-th sample of --part (default val)")
    pick.add_argument("--smiles", help="a reactant pool, '.'-separated, no label")
    e.add_argument("--drawing", type=int, default=0, help="resonance drawing (0: the dataset's)")
    e.add_argument("--top", type=int, default=10, help="anchors listed")
    e.add_argument("--device", default="cpu")
    e.add_argument("--out", type=Path, help="the HTML page")
    e.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    e.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    e.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    return parser


def _decode_parser(sub) -> None:
    from ksmech.eval.decode import CAP_FIELDS, CAPS

    d = sub.add_parser("decode", help="the beam decoder on stored samples: full-R top-k")
    d.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    d.add_argument("--checkpoint", default="best", choices=["best", "last"])
    d.add_argument("--source", default="pmechdb", choices=[str(x) for x in Source],
                   help="the samples decoded")  # fmt: skip
    d.add_argument("--run-source", help="the models directory of the run (default --source), "
                   "e.g. pmechdb+reactmech")  # fmt: skip
    d.add_argument("--split", default="molecule", help="split strategy of the sample files")
    d.add_argument("--part", default="val", choices=["train", "val", "test"])
    d.add_argument("--limit", type=int, help="the first N samples of the part")
    d.add_argument("--beam", type=int, default=5, help="anchors completed and products kept each")
    d.add_argument("--scope", default="all", choices=["all", "inter"], help="inter: anchors "
                   "between two molecules only")  # fmt: skip
    d.add_argument("--caps", default="legacy", choices=CAPS, help="legacy: 8 bond, 2 atom "
                   "changes; labels: the label maxima of --source's training part")  # fmt: skip
    for name in CAP_FIELDS:
        d.add_argument(f"--{name.replace('_', '-')}", type=int, help="overrides --caps")
    d.add_argument("--closed-minus1-filter", action="store_true",
                   help="refuse any -1 atom entry on a pool with no unpaired electron")  # fmt: skip
    d.add_argument("--direction", default=FORWARD, choices=DIRECTIONS,
                   help="backward: the previous-step samples (<source>/backward/)")  # fmt: skip
    d.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    d.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    _cohort_args(d)


def _importance_parser(sub) -> None:
    i = sub.add_parser("importance", help="permutation importance of the token inputs")
    i.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    i.add_argument("--checkpoint", default="best", choices=["best", "last"])
    i.add_argument("--source", default="pmechdb", choices=[str(x) for x in Source])
    i.add_argument("--run-source", help="the models directory of the run (default --source)")
    i.add_argument("--split", default="molecule", help="split strategy of the sample files")
    i.add_argument("--part", default="val", choices=["train", "val", "test"])
    i.add_argument("--limit", type=int, help="the first N samples of the part")
    i.add_argument("--scope", default="both", choices=["columns", "blocks", "both"],
                   help="each column alone, the blocks of related columns, or both")  # fmt: skip
    i.add_argument("--repeats", type=int, default=3, help="permutations per group")
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--device", default="auto", help="auto | cpu | cuda")
    i.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"])
    i.add_argument("--cell-budget", type=int, default=1_100_000)
    i.add_argument("--max-tokens", type=int, default=1024, help="larger samples skipped")
    i.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    i.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    i.add_argument("--cohort", default="all", help="score only cohort_<name>.json's steps")


def _cohort_args(p) -> None:
    p.add_argument("--cohort", default="all", help="score only cohort_<name>.json's steps")
    p.add_argument("--slice", choices=sorted(SLICES), help="also report with and without it")


def _cohort_ids(args, path: Path) -> list[str]:
    return restrict(sample_ids(path), load_cohort(args.source, args.cohort, args.processed_dir))


def _extras(args, rows: list[dict], samples: list, view: Path) -> dict:
    """The report's cohort, covered fraction and (with --slice) the with / without numbers."""
    out = {"cohort": args.cohort, "covered": covered_fraction(view)}
    out["by_r_shape"] = by_r_shape(rows, {s.step_id: s for s in samples})
    if args.slice:
        out["slice"] = sliced(rows, {s.step_id: s for s in samples}, args.slice)
    return out


def _run_notes(payload: dict) -> dict:
    """How the checkpoint's run was selected and standardized (`train.loop.run_notes`), for a
    report that quotes one model's numbers; nothing for a run older than the notes."""
    extra = payload.get("provenance", {}).get("extra", {})
    return {k: extra[k] for k in ("selection", "pair_statistics") if k in extra}


def _sample_versions(path: Path, payload: dict, console: Console) -> dict:
    """The sample version the checkpoint was trained on and the decoded view's. A mismatch
    warns and is recorded; it never blocks (flex_first is v15, the pod runs v16): decoding
    fails only when an input the checkpoint needs is missing from the store."""
    trained = payload.get("provenance", {}).get("config", {}).get("sample_version")
    header = prov.read_header(path)
    view = header.schema_version if header else None
    if trained != view:
        console.print(f"[yellow]checkpoint trained on samples v{trained}, view is v{view}[/yellow]")
    return {"checkpoint_sample_version": trained, "view_sample_version": view}


def _directions(direction: str, payload: dict, console: Console) -> dict:
    """The decoded samples' direction and the checkpoint's (`arrows.backward`), for the decode
    config: nothing when both are forward (a forward header as before), else both; a mismatch
    warns. With "direction" in `eval.paired.PAIRED_KEYS`, hits of two directions never pair."""
    train = payload.get("provenance", {}).get("config", {}).get("train", {})
    trained = train.get("direction", FORWARD)
    if trained != direction:
        console.print(f"[yellow]checkpoint trained on {trained} samples, decoding {direction}")
    if direction == trained == FORWARD:
        return {}
    return {"direction": direction, "checkpoint_direction": trained}


def _decode_caps(args) -> tuple[dict[str, int], dict[str, str]]:
    """The caps of a decode (`eval.decode.decode_caps`: --caps, then the overrides) and, with
    --caps labels, the training view of --source they are read from, an input of the report."""
    from ksmech.arrows.steps import SAMPLE_VERSION
    from ksmech.eval.decode import CAP_FIELDS, decode_caps

    train = samples_path(args.source, f"{args.split}_train", args.processed_dir, args.direction)
    overrides = {name: getattr(args, name) for name in CAP_FIELDS}
    caps = decode_caps(args.caps, train, SAMPLE_VERSION, overrides)
    return caps, prov.file_input(train) if args.caps == "labels" else {}


def _outcomes(args, console: Console) -> tuple[dict, dict, dict[str, str]]:
    """(step id -> the pool's recorded outcomes, the skip counts, the inputs): the source's
    training steps grouped by reactant key (product key for --direction backward,
    `eval.siblings`), parts from the split file."""
    from collections import Counter

    from ksmech.arrows.steps import load_split_file, splits_path
    from ksmech.data.io import read_training_steps
    from ksmech.eval.siblings import outcome_table, parts_of, side_keys

    steps_file = steps_path(args.source, args.processed_dir)
    split_file = splits_path(args.source, args.split, args.processed_dir)
    skips: Counter = Counter()
    keys = side_keys(read_training_steps(steps_file), skips)
    if args.direction == BACKWARD:  # a pool is the drawn products, its outcomes the reactants
        keys = {step: (product, pool) for step, (pool, product) in keys.items()}
    parts = parts_of(load_split_file(split_file)) if split_file.is_file() else {}
    if not parts:
        console.print(f"[yellow]no split file {split_file.name}: sibling parts unknown[/yellow]")
    inputs = prov.file_input(steps_file) | (prov.file_input(split_file) if parts else {})
    return outcome_table(keys, parts), dict(skips), inputs


def _print_decode(console: Console, report: dict, title: str) -> None:
    """The top-k table (own label and any recorded outcome) and the multi-outcome subset."""
    table = Table(title=f"full-R decoding, {title}")
    for col in ("n", "top-1", "top-3", "top-5", "not found", "s / sample"):
        table.add_column(col)
    per_sample = f"{report['mean_seconds']:.1f}"
    for name, suffix in (("own", ""), ("any", "_any")):
        tops = (f"{report[f'top{k}{suffix}']:.1%}" for k in (1, 3, 5))
        n = f"{report['n']} {name}"
        table.add_row(n, *tops, str(report["not_found"]) if not suffix else "", per_sample)
    console.print(table)
    m = report["multi_outcome"]
    if m["n"]:
        console.print(f"pools with several recorded products: {m['n']} samples, top-1 own "
                      f"{m['top1']:.1%}, any {m['top1_any']:.1%}")  # fmt: skip


def cmd_decode(args, console: Console) -> int:
    from ksmech.eval.decode import decode_report, sample_decoder
    from ksmech.eval.siblings import outcome_summary
    from ksmech.model.decode import DecodeConfig
    from ksmech.train.loop import load_model

    caps, caps_input = _decode_caps(args)
    run_source = args.run_source or args.source
    ckpt = args.models_dir / run_source / args.run / f"{args.checkpoint}.pt"
    model, payload = load_model(ckpt, "cpu")
    path = samples_path(
        args.source, f"{args.split}_{args.part}", args.processed_dir, args.direction
    )
    require_direction(path, args.direction)
    ids = _cohort_ids(args, path)
    ids = ids[: args.limit] if args.limit is not None else ids
    minus1 = args.closed_minus1_filter
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam, closed_minus1_filter=minus1, **caps)
    samples = load_samples(path, ids=ids)
    outcomes, sibling_skips, sibling_inputs = _outcomes(args, console)
    report = decode_report(samples, sample_decoder(model, cfg, args.scope), outcomes=outcomes)
    hits = report.pop("per_sample")
    report |= _extras(args, hits, samples, path) | _run_notes(payload)
    report["siblings"] = outcome_summary(outcomes, ids, args.part) | {"skips": sibling_skips}
    keys = ("source", "split", "part", "limit", "beam", "scope", "cohort", "slice")
    keys = (*keys, "closed_minus1_filter")
    config = {k: getattr(args, k) for k in keys}
    config |= {"run_source": run_source, "checkpoint": _checkpoint_text(args.checkpoint, payload)}
    config |= _sample_versions(path, payload, console) | {"caps": args.caps, **caps}
    config |= _directions(args.direction, payload, console)
    inputs = prov.file_input(path) | prov.file_input(ckpt) | caps_input | sibling_inputs
    header = prov.stamp("decode", DECODE_VERSION, inputs, config)
    suffix = "_backward" if args.direction == BACKWARD else ""
    suffix += f"_{args.source}" if args.source != run_source else ""
    suffix += f"_{args.cohort}" if args.cohort != "all" else ""
    suffix += f"_n{args.limit}" if args.limit is not None else ""
    suffix += "_minus1filter" if minus1 else ""
    if caps != {k: getattr(DecodeConfig(), k) for k in caps}:  # the legacy files stay as they are
        suffix += "_caps{max_bonds}-{max_atoms}-{max_candidates}".format(**caps)
    out = ckpt.parent / f"decode_report_{args.split}_{args.part}{suffix}.json"
    prov.write_json(out, report, header)
    hits_out = ckpt.parent / f"decode_hits_{args.split}_{args.part}{suffix}.json"
    prov.write_json(
        hits_out, {"per_sample": hits}, prov.stamp("decode_hits", DECODE_VERSION, inputs, config)
    )
    _print_decode(console, report, f"{args.run} {args.checkpoint}, {args.split} {args.part}")
    for code, count in report["skips"].items():
        console.print(f"  {code}: {count}")
    console.print(f"wrote {out} and {hits_out.name}")
    return 0


def _importance_samples(args, path: Path, console: Console) -> tuple[list, dict]:
    """The part's samples (cohort, limit), without those over --max-tokens (TOO_LARGE) and
    the drawings the decoder cannot write, with the skip counts."""
    from ksmech.train.loop import TOO_LARGE, drop_undecodable_drawings

    ids = _cohort_ids(args, path)
    ids = ids[: args.limit] if args.limit is not None else ids
    samples = load_samples(path, ids=ids)
    keep = [s for s in samples if s.n_atoms + s.n_orbitals + 1 <= args.max_tokens]
    skipped = {TOO_LARGE: len(samples) - len(keep)} if len(keep) < len(samples) else {}
    if n := drop_undecodable_drawings(keep, args.part):
        skipped["drawings NOT_ELEMENTARY"] = n
    for code, count in skipped.items():
        console.print(f"  skipped {code}: {count}")
    return keep, skipped


def cmd_importance(args, console: Console) -> int:
    from ksmech.eval.importance import IMPORTANCE_VERSION, groups, importance
    from ksmech.train.loop import TrainConfig, amp_dtype, load_model, make_loader, select_device

    run_source = args.run_source or args.source
    ckpt = args.models_dir / run_source / args.run / f"{args.checkpoint}.pt"
    device = select_device(args.device)
    model, payload = load_model(ckpt, device)
    path = samples_path(args.source, f"{args.split}_{args.part}", args.processed_dir)
    samples, skipped = _importance_samples(args, path, console)
    cfg = TrainConfig(cell_budget=args.cell_budget, max_tokens=args.max_tokens, workers=0)
    loader = make_loader(samples, cfg, shuffle=False)
    found = groups(*model.feature_stats.names(), args.scope)
    console.print(f"{len(samples)} samples, {len(found)} groups x {args.repeats} on {device}")

    def log(i, n, row):
        console.print(f"  {i}/{n} {row['group']}: dNLL {row['d_anchor_nll']:+.4f}, "
                      f"dtop1 {row['d_top1']:+.2%}")  # fmt: skip

    report = importance(model, loader, found, args.repeats, args.seed, device,
                        amp_dtype(device, args.amp), log)  # fmt: skip
    report |= {"skipped": skipped} | _run_notes(payload)
    keys = ("source", "split", "part", "limit", "scope", "repeats", "seed", "cohort")
    config = {k: getattr(args, k) for k in keys} | {"run_source": run_source}
    config |= {"checkpoint": _checkpoint_text(args.checkpoint, payload)}
    config |= _sample_versions(path, payload, console)
    header = prov.stamp("token_importance", IMPORTANCE_VERSION,
                        prov.file_input(path) | prov.file_input(ckpt), config)  # fmt: skip
    suffix = f"_{args.source}" if args.source != run_source else ""
    suffix += f"_{args.cohort}" if args.cohort != "all" else ""
    suffix += f"_n{args.limit}" if args.limit is not None else ""
    out = ckpt.parent / f"token_importance_{args.split}_{args.part}{suffix}.json"
    prov.write_json(out, report, header)
    _print_importance(console, report, f"{args.run} {args.checkpoint}, {args.split} {args.part}")
    console.print(f"wrote {out}")
    return 0


def _print_importance(console: Console, report: dict, title: str) -> None:
    base = report["baseline"]
    console.print(f"baseline: n {base['n']}, anchor NLL {base['anchor_nll']:.4f}, "
                  f"top-1 {base['top1']:.1%}, top-3 {base['top3']:.1%}, "
                  f"R exact {base['r_exact']:.1%}")  # fmt: skip
    table = Table(title=f"token input importance (permuted), {title}")
    for col in ("group", "kind", "dNLL", "sd", "dtop-1", "dtop-3", "dR exact"):
        table.add_column(col)
    for r in report["groups"]:
        pct = (f"{r[k]:+.1%}" for k in ("d_top1", "d_top3", "d_r_exact"))
        table.add_row(r["group"], r["kind"], f"{r['d_anchor_nll']:+.4f}",
                      f"{r['d_anchor_nll_sd']:.4f}", *pct)  # fmt: skip
    console.print(table)


def cmd_compare(args, console: Console) -> int:
    from ksmech.eval.paired import compare_arms

    try:
        report = compare_arms(args.a, args.b, args.k, args.metric)
    except ValueError as err:
        console.print(f"[red]{err}")
        return 2
    d, m = report["decision"], report["majority"]
    console.print(f"n {report['n']}, top-{args.k} ({args.metric}) per run: "
                  f"A {report['top_k']['a']}, "
                  f"B {report['top_k']['b']}")  # fmt: skip
    console.print(f"mean A {d['mean_a']:.2f}, mean B {d['mean_b']:.2f}, difference "
                  f"{d['difference']:+.2f}, pooled SD {d['pooled_sd']:.2f}, tolerance "
                  f"{d['tolerance']:.2f}: A {'fails' if d['a_fails'] else 'passes'}")  # fmt: skip
    console.print(f"McNemar (majority of seeds, H1 A worse): A miss / B hit {m['a_miss_b_hit']}, "
                  f"A hit / B miss {m['a_hit_b_miss']}, p {m['p']:.3g}")  # fmt: skip
    if args.out:
        inputs = {}
        for path in [*args.a, *args.b]:
            inputs |= prov.file_input(path)
        config = {"k": args.k, "metric": args.metric}
        header = prov.stamp("decode_compare", DECODE_VERSION, inputs, config)
        prov.write_json(args.out, report, header)
        console.print(f"wrote {args.out}")
    return 0


def cmd_pin(args, console: Console) -> int:
    from ksmech.eval.pin import pin_cohort

    out = pin_cohort(args.source, args.name, args.split, args.processed_dir, args.workers)
    table = Table(title=f"cohort {args.name}: {args.source} {args.split} split")
    for col in ("part", "samples", "GBDT valid cells", "GBDT all cells"):
        table.add_column(col)
    for part in ("train", "val", "test"):
        c = out[part]
        table.add_row(part, str(c["samples"]), str(c["gbdt_valid"]), str(c["gbdt_all"]))
    console.print(table)
    for path in out["written"]:
        console.print(f"wrote {path}")
    return 0


def cmd_cohort(args, console: Console) -> int:
    from ksmech.data.io import read_training_steps
    from ksmech.eval.cohorts import class_cohort_ids, write_cohort

    steps = read_training_steps(steps_path(args.source, args.processed_dir))
    ids = class_cohort_ids(steps, args.domain_class)
    if not ids:
        console.print(f"no training step has a domain_class in {args.domain_class}")
        return 1
    note = {"domain_class": sorted(args.domain_class), "n_training_steps": len(steps)}
    path = write_cohort(args.source, args.name, ids, note, args.processed_dir)
    console.print(f"wrote {path}: {len(ids)} steps")
    return 0


def cmd_floor(args, console: Console) -> int:
    results, inputs = {}, {}
    ranks = {}
    for part in ("val", "test"):
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        inputs |= prov.file_input(path)
        samples = load_samples(path, ids=_cohort_ids(args, path))
        options = {"floor": args.floor, "kernel": args.kernel}
        results[part] = floor_report(
            samples, cells=args.cells, drop_intra_open=args.drop_intra_open, **options
        )
        ranks[part] = results[part].pop("per_sample")
        results[part] |= _extras(args, ranks[part], samples, path)
    config = {"split": args.split, "cohort": args.cohort, "slice": args.slice, "cells": args.cells}
    config |= {"floor": args.floor, "kernel": args.kernel, "drop_intra_open": args.drop_intra_open}
    header = prov.stamp("anchor_floor", FLOOR_VERSION, inputs, config)
    out_dir = source_dir(args.source, args.processed_dir)
    tag = args.split + (f"_{args.cohort}" if args.cohort != "all" else "")
    tag += "_allcells" if args.cells == "all" else ""
    tag += "" if args.floor == "legacy" else f"_{args.floor}"
    tag += "" if args.kernel == "g" else f"_{args.kernel}"
    tag += "_nointraopen" if args.drop_intra_open else ""
    out = out_dir / f"anchor_floor_report_{tag}.json"
    scorer = f"|G^atom| over {'valid' if args.cells == 'valid' else 'all'} pairs"
    scorer += f", |B^own| on bonded pairs ({args.kernel})" if args.floor == "own_break" else ""
    prov.write_json(out, {"scorer": scorer, **results}, header)
    ranks_header = prov.stamp("anchor_floor_ranks", FLOOR_VERSION, inputs, config)
    prov.write_json(out_dir / f"anchor_floor_ranks_{tag}.json", ranks, ranks_header)
    table = Table(title=f"anchor floor, {args.source} {args.split} split")
    for col in ("part", "n", "top-1", "top-3", "top-5", "median rank", "unreachable"):
        table.add_column(col)
    for part, r in results.items():
        tops = (f"{r[f'top{k}']:.1%}" for k in (1, 3, 5))
        table.add_row(part, str(r["n"]), *tops, f"{r['median_rank']:.0f}", str(r["unreachable"]))
    console.print(table)
    console.print(f"wrote {out}")
    return 0


def _stored_sample(args) -> tuple[MechanismSample, str] | None:
    """(the sample of --step-id or --index, its part), None when there is none."""
    parts = [args.part] if args.part else (["val", "test", "train"] if args.step_id else ["val"])
    for part in parts:
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        ids = sample_ids(path)
        if args.step_id is None and 0 <= (args.index or 0) < len(ids):
            return load_samples(path, ids=[ids[args.index or 0]])[0], part
        if args.step_id in ids:
            return load_samples(path, ids=[args.step_id])[0], part
    return None


def _reaction(args, step_id: str) -> str:
    """The step's reaction SMILES from the steps parquet, "" when it has none."""
    path = steps_path(args.source, args.processed_dir)
    if not path.exists():
        return ""
    cols = ["reactants_smiles", "products_smiles"]
    df = pd.read_parquet(path, columns=cols, filters=[("step_id", "==", step_id)])
    return "" if df.empty else f"{df.iloc[0, 0]}>>{df.iloc[0, 1]}"


def _orbital_types(sample: MechanismSample, loader: TokenCache, console: Console):
    """The diagnostic orbital types of the sample's token files, None (reported) if a file is
    missing or does not match the sample."""
    from ksmech.eval.embedding_page import orbital_types

    tokens = sample.tokens(loader)
    types = None if any(t is None for t in tokens) else orbital_types(tokens)
    if types is None or len(types) != sample.n_orbitals:
        console.print("[yellow]orbital types unavailable: the page names orbitals by occupation")
        return None
    return types


def _embedding_input(args, loader: TokenCache, console: Console):
    """(sample, where, reaction) of the command, None when the sample is not found."""
    from ksmech.eval.embedding import ensure_pool, sample_from_smiles
    from ksmech.physics.textbook import fragment_keys

    if args.smiles:
        if built := ensure_pool(fragment_keys(args.smiles), loader):
            console.print(f"built tokens for {built}")
        return sample_from_smiles(args.smiles, loader), "SMILES", args.smiles
    found = _stored_sample(args)
    if found is None:
        what = args.step_id or f"index {args.index} of {args.part or 'val'}"
        console.print(f"[red]no stored sample: {what} ({args.split} split)")
        return None
    sample, part = found
    return sample, f"{args.split} {part}", _reaction(args, sample.step_id)


def cmd_embedding(args, console: Console) -> int:
    from ksmech.eval.embedding import trace
    from ksmech.eval.embedding_page import page_payload, write_page
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.source / args.run / f"{args.checkpoint}.pt"
    model, payload = load_model(ckpt, args.device)
    loader = TokenCache(args.tokens_dir)
    found = _embedding_input(args, loader, console)
    if found is None:
        return 2
    sample, where, reaction = found
    if not 0 <= args.drawing < sample.n_drawings:
        console.print(f"[red]the sample has {sample.n_drawings} drawing(s), no {args.drawing}")
        return 2
    sample = sample.drawn(args.drawing)
    tr = trace(model, sample, args.device, args.top)
    name = sample.step_id if not args.smiles else "smiles " + args.smiles
    meta = {
        "sample": f"{name} ({where}, drawing {args.drawing})",
        "reaction": reaction,
        "run": args.run,
        "checkpoint": _checkpoint_text(args.checkpoint, payload),
        "encoder": model.cfg.encoder,
        "heads": model.cfg.heads,
    }
    slug = (
        sample.step_id
        if not args.smiles
        else "smiles-" + hashlib.sha1(args.smiles.encode()).hexdigest()[:10]
    )
    out = args.out or ckpt.parent / "embedding" / f"{slug}_d{args.drawing}.html"
    types = _orbital_types(sample, loader, console)
    write_page(page_payload(tr, sample, types, meta), out)
    _print_trace(console, tr, sample)
    console.print(f"wrote {out}")
    return 0


def _checkpoint_text(which: str, payload: dict) -> str:
    """The checkpoint, its epoch and the validation top-1 its trainer logged."""
    text = f"{which}.pt, epoch {payload.get('epoch', '?')}"
    top1 = payload.get("metrics", {}).get("val_top1")
    return text if top1 is None else f"{text}, val top-1 {top1:.1%} when trained"


def _print_trace(console: Console, tr: dict, sample: MechanismSample) -> None:
    r = tr["readout"]
    worst = max([tr["errors"]["embedding"], *tr["errors"]["layers"]])
    console.print(f"{sample.n_atoms} atoms, {sample.n_orbitals} orbitals; rebuilt layers within "
                  f"{worst:.1e} of the model")  # fmt: skip
    top = r["top"][0]
    console.print(f"top anchor {top['i']}-{top['j']} {top['side']}, P = {top['p']:.1%}")
    if "label_rank" in r:
        p_label = r["label_probability"]
        console.print(f"label rank {r['label_rank']}, P(label anchors) = {p_label:.1%}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "floor": cmd_floor,
        "decode": cmd_decode,
        "importance": cmd_importance,
        "embedding": cmd_embedding,
        "pin": cmd_pin,
        "cohort": cmd_cohort,
        "compare": cmd_compare,
    }
    return commands[args.command](args, Console())


if __name__ == "__main__":
    sys.exit(main())
