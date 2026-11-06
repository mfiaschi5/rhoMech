"""`ksmech-physics`: the geometry-free LightGBM reference ranker and the textbook suite."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.steps import sample_ids, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.data.provenance import StaleArtifact
from ksmech.data.schema import Source
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR

FEATURES = "geomfree"  # the file tag of the ranker's reports and models
CHUNK = 64  # consecutive samples per task of a parallel read
HELD_OUT = "textbook-held-out"  # file tag of the ranker fitted without the textbook pools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-physics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gbdt", help="the fast LightGBM lambdarank ranker")
    g.add_argument("--source", required=True, choices=[str(x) for x in Source])
    g.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    g.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    g.add_argument("--split", default="molecule", help="split strategy of the sample files")
    g.add_argument("--hard", type=int, default=20, help="negatives per step: best untrained")
    g.add_argument("--random", type=int, default=30, help="negatives per step: uniform draw")
    g.add_argument(
        "--n-jobs", type=int, default=14, help="LightGBM threads (recorded: the trees depend on it)"
    )
    g.add_argument(
        "--workers",
        type=int,
        default=1,
        help="spawned processes loading the samples (host total <= 15)",
    )
    g.add_argument(
        "--cells",
        default="valid",
        choices=["valid", "all"],
        help="candidates: the pair_valid cells (v17) or every cell, exclusion flags as features",
    )
    g.add_argument(
        "--score-only",
        action="store_true",
        help="no fit: score the stored ranker of --split on val and test, per-sample ranks",
    )
    g.add_argument(
        "--model",
        type=Path,
        default=None,
        help="--score-only: this stored ranker instead of the split's (a reference copy)",
    )
    g.add_argument(
        "--allow-stale",
        action="store_true",
        help="--score-only: score a ranker whose training views changed (logged; a legacy gate)",
    )
    t = sub.add_parser("textbook", help="textbook selectivity checks (expected vs top-3)")
    t.add_argument("--source", default="pmechdb", help="where the trained ranker lives")
    t.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    t.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    t.add_argument("--gbdt", type=Path, default=None, help="ranker model (default: molecule split)")
    t.add_argument("--allow-stale", action="store_true", help="read stale artifacts (warns)")
    t.add_argument(
        "--radical",
        action="store_true",
        help="the radical selectivity suite instead (untrained kernel, P7 diagnostics)",
    )
    t.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="with --radical: a trained model's anchors instead of the untrained kernel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    commands = {"gbdt": cmd_gbdt, "textbook": cmd_textbook}
    return commands[args.command](args, console)


def _baseline_table(results: dict, console: Console, title: str) -> None:
    table = Table(title=title)
    for col in ("scorer", "test top-1", "test top-3", "test top-5", "test median", "val top-5"):
        table.add_column(col)
    for name, r in results.items():
        t, v = r["test"], r["val"]
        table.add_row(
            name,
            *(f"{t[f'top{k}']:.1%}" for k in (1, 3, 5)),
            f"{t['median_rank']:.0f}",
            f"{v['top5']:.1%}",
        )
    console.print(table)


def _cells(args) -> str:
    return getattr(args, "cells", "valid")


def _tag(args) -> str:
    """The file tag of a ranker: `<split>_geomfree`, `_allcells` appended for --cells all."""
    return f"{args.split}_{FEATURES}" + ("_allcells" if _cells(args) == "all" else "")


def _read_part(
    args, part: str, loader, visit, ids: list[str] | None = None
) -> tuple[int, int, int]:
    """`visit(sample, candidate_set)` for every sample of one part (only `ids` when given) that
    has ArrowLabels (the candidate sets rank the first arrow's orbital pairs), one shard at a
    time. Returns (stored samples of the view or of `ids`, samples visited, samples without an
    orbital label)."""
    from ksmech.arrows.steps import iter_samples
    from ksmech.physics import baseline as bl

    path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
    ids = sample_ids(path) if ids is None else ids
    n = no_reading = 0
    for s in iter_samples(path, ids=ids):
        if not bl.sample_labels(s, _cells(args)):
            no_reading += 1
            continue
        visit(s, bl.candidate_set(s, loader, _cells(args)))
        n += 1
    return len(ids), n, no_reading


def _new_train() -> dict:
    from ksmech.physics.lgbm import TrainRows

    return {"all": TrainRows(), "held": TrainRows(), "seen": {}}


def _train_visit(out: dict, s, cs, cfg) -> None:
    """The sampled training rows of every step, and of the steps without a textbook pool. The
    hard negatives are the pairs -G ranks highest."""
    from ksmech.physics import baseline as bl
    from ksmech.physics.textbook import pools_in

    hard = bl.gap_score(cs)
    out["all"].add(cs, hard, cfg)
    pools = pools_in(s.fragment_keys)
    for name in pools:
        out["seen"][name] = out["seen"].get(name, 0) + 1
    if not pools:
        out["held"].add(cs, hard, cfg)


def _merge_train(out: dict, part: dict) -> None:
    out["all"].extend(part["all"])
    out["held"].extend(part["held"])
    for name, n in part["seen"].items():
        out["seen"][name] = out["seen"].get(name, 0) + n


def _new_eval() -> dict:
    return {"sets": [], "untrained": {}, "textbook": []}


def _eval_visit(out: dict, s, cs, cfg) -> None:
    """Every candidate set of a val / test part (X kept as float32), its untrained score
    (computed at full precision first) and whether the sample holds a textbook pool."""
    from ksmech.physics import baseline as bl
    from ksmech.physics.textbook import pools_in

    out["untrained"].setdefault("gap_kernel", []).append(bl.gap_score(cs))
    cs.X = cs.X.astype(np.float32)
    out["sets"].append(cs)
    out["textbook"].append(bool(pools_in(s.fragment_keys)))


def _merge_eval(out: dict, part: dict) -> None:
    out["sets"] += part["sets"]
    out["textbook"] += part["textbook"]
    for name, scores in part["untrained"].items():
        out["untrained"].setdefault(name, []).extend(scores)


READERS = {
    "train": (_new_train, _train_visit, _merge_train),
    "eval": (_new_eval, _eval_visit, _merge_eval),
}


def _read_chunk(task: tuple) -> tuple[dict, tuple[int, int, int]]:
    """A worker's share of `_read_kind`: one chunk of step ids, read and reduced."""
    args, part, ids, loader, kind, cfg = task
    new, visit, _ = READERS[kind]
    out = new()
    read = _read_part(args, part, loader, lambda s, cs: visit(out, s, cs, cfg), ids)
    return out, read


def _read_kind(args, part: str, loader, kind: str, cfg) -> dict:
    """`_read_part` reduced by the `kind` reader ("train": sampled rows; "eval": candidate
    sets), in chunks of `CHUNK` over `args.workers` spawned processes when it is above 1: the
    chunks come back in step order, and negatives are drawn per step id, so the result is the
    same."""
    from ksmech import processes

    new, visit, merge = READERS[kind]
    out = new()
    if getattr(args, "workers", 1) <= 1:
        read = _read_part(args, part, loader, lambda s, cs: visit(out, s, cs, cfg))
    else:
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        ids = sample_ids(path)
        tasks = [(args, part, chunk, loader, kind, cfg) for chunk in processes.chunked(ids, CHUNK)]
        n = no_reading = 0
        for got, (_, k, nr) in processes.in_processes(_read_chunk, tasks, args.workers):
            merge(out, got)
            n, no_reading = n + k, no_reading + nr
        read = (len(ids), n, no_reading)
    out["n_view"], out["n"], out["n_no_orbital_reading"] = read
    return out


def _lgbm_results(parts: dict, models: dict, n_jobs: int = 1) -> dict:
    from ksmech.physics import baseline as bl

    def summary(sets, scores):
        return {k: v for k, v in bl.evaluate_scores(sets, scores).items() if k != "ranks"}

    results = {}
    names = {n for p in ("val", "test") for n in parts[p]["untrained"]}
    for name in sorted(names):
        results[f"untrained {name}"] = {
            p: summary(parts[p]["sets"], parts[p]["untrained"].get(name, []))
            for p in ("val", "test")
        }
    for name, model in models.items():
        results[name] = {
            p: summary(
                parts[p]["sets"], model.score_sets([cs.X for cs in parts[p]["sets"]], n_jobs)
            )
            for p in ("val", "test")
        }
    return results


def _fit_two(train: dict, val: dict, cfg, console: Console, cells: str = "valid") -> dict:
    """The ranker on every training step and the one without the textbook pools (validated
    without them too), with fit times."""
    from ksmech.physics import baseline as bl
    from ksmech.physics.lgbm import EvalRows, fit_lgbm

    val_held = [cs for cs, t in zip(val["sets"], val["textbook"], strict=True) if not t]
    models, seconds = {}, {}
    for tag, rows, vsets in (("all", train["all"], val["sets"]), ("held", train["held"], val_held)):
        start = time.perf_counter()
        models[tag] = fit_lgbm(rows, EvalRows.of(vsets), bl.feature_names(cells), cfg)
        seconds[tag] = round(time.perf_counter() - start, 1)
        console.print(
            f"{tag}: {rows.n_rows} rows in {len(rows.groups)} queries, "
            f"{models[tag].n_trees} trees, {seconds[tag]} s"
        )
    return {"models": models, "fit_seconds": seconds}


def _lgbm_inputs(args) -> dict[str, str]:
    inputs = {}
    for part in ("train", "val"):
        inputs |= prov.file_input(
            samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        )
    return inputs


def _save_two(args, fit: dict, cfg, seen: list[str]) -> None:
    from ksmech.physics import lgbm

    out_dir = source_dir(args.source, args.processed_dir)
    tag, inputs = _tag(args), _lgbm_inputs(args)
    notes = {
        "all": ("", {"textbook_pools_in_training": seen}),
        "held": (f"_{HELD_OUT}", {"textbook_pools_in_training": [], "held_out": seen}),
    }
    for key, (suffix, extra) in notes.items():
        path = out_dir / f"gbdt_lgbm_{tag}{suffix}.npz"
        lgbm.save_lgbm(fit["models"][key], path, cfg, inputs, extra)


def _lgbm_report(args, train: dict, parts: dict, fit: dict, results: dict, cfg) -> dict:
    from ksmech.physics import lgbm

    models = fit["models"]
    return {
        "split": args.split,
        "features": FEATURES,
        "cells": _cells(args),
        "results": results,
        "counts": {
            p: {
                "read": d["n"],
                "stored": d["n_view"],
                "no_orbital_reading": d["n_no_orbital_reading"],
            }
            for p, d in (("train", train), *parts.items())
        },
        "train_rows": {"all": train["all"].n_rows, "held": train["held"].n_rows},
        "n_untrainable": train["all"].n_untrainable,
        "n_trees": {k: m.n_trees for k, m in models.items()},
        "history": {k: m.history for k, m in models.items()},
        "config": lgbm.model_config(models["all"].features, cfg),
        "load_seconds": fit["load_seconds"],
        "fit_seconds": fit["fit_seconds"],
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "textbook_seen": train["seen"],
    }


def _read_all(args, loader, cfg, console: Console) -> tuple[dict, dict] | None:
    try:
        train = _read_kind(args, "train", loader, "train", cfg)
        parts = {p: _read_kind(args, p, loader, "eval", cfg) for p in ("val", "test")}
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]; run `ksmech-arrows samples`")
        return None
    except (StaleArtifact, ValueError) as exc:  # stale or old: rebuild, never reinterpret
        console.print(f"[red]{exc}[/red]; rebuild it (`ksmech-arrows samples`)")
        return None
    for p, d in (("train", train), *parts.items()):
        console.print(
            f"{_tag(args)} {p}: {d['n']} samples read of {d['n_view']}, "
            f"{d['n_no_orbital_reading']} without an orbital label (NO_ORBITAL_READING)"
        )
    console.print(f"textbook pools in {args.split}_train: {train['seen'] or 'none'}")
    return train, parts


def cmd_gbdt_score(args: argparse.Namespace, console: Console) -> int:
    """The stored ranker of --split (`gbdt_lgbm_<split>_geomfree.npz`) scored on val and test
    without a fit: the summary and one rank per step (`gbdt_lgbm_ranks_<split>_geomfree.json`),
    so a later model is compared with it sample by sample, not refitted (a refit samples 80% of
    the columns and does not reproduce)."""
    from ksmech.physics import baseline as bl
    from ksmech.physics import lgbm

    out_dir, tag = source_dir(args.source, args.processed_dir), _tag(args)
    path = getattr(args, "model", None) or out_dir / f"gbdt_lgbm_{tag}.npz"
    stale = getattr(args, "allow_stale", False)
    model = lgbm.load_lgbm(path, bl.feature_names(_cells(args)), allow_stale=stale)
    loader = TokenCache(args.tokens_dir, maxsize=128)
    result, inputs = {}, prov.file_input(path)
    for part in ("val", "test"):
        sets = _read_kind(args, part, loader, "eval", None)["sets"]
        ranks = bl.evaluate_scores(sets, model.score_sets([cs.X for cs in sets], args.n_jobs))
        ranks = ranks["ranks"]
        rows = [{"step_id": cs.step_id, "rank": r} for cs, r in zip(sets, ranks, strict=True)]
        result[part] = {"summary": bl.summarize_ranks(ranks), "per_sample": rows}
        inputs |= prov.file_input(
            samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        )
    config = {"model": path.name, "split": args.split, "cells": _cells(args)}
    header = prov.stamp("gbdt_lgbm_ranks", 1, inputs, config)
    suffix = f"_{Path(path).stem}" if getattr(args, "model", None) else ""
    out = out_dir / f"gbdt_lgbm_ranks_{tag}{suffix}.json"
    prov.write_json(out, result, header)
    for part, r in result.items():
        tops = " / ".join(f"{r['summary'][f'top{k}']:.1%}" for k in (1, 3, 5))
        console.print(f"{part}: n {r['summary']['n']}, top-1/3/5 {tops}")
    console.print(f"wrote {out}")
    return 0


def cmd_gbdt(args: argparse.Namespace, console: Console) -> int:
    """The fast LightGBM ranker (`physics.lgbm`) on the geometry-free features."""
    from ksmech.physics.lgbm import LGBMConfig

    if args.score_only:
        return cmd_gbdt_score(args, console)

    cfg = LGBMConfig(n_hard=args.hard, n_random=args.random, n_jobs=args.n_jobs)
    loader = TokenCache(args.tokens_dir, maxsize=128)  # tokens only for the LMO energies
    start = time.perf_counter()
    read = _read_all(args, loader, cfg, console)
    if read is None:
        return 2
    train, parts = read
    load_seconds = round(time.perf_counter() - start, 1)
    fit = _fit_two(train, parts["val"], cfg, console, _cells(args))
    fit |= {"load_seconds": load_seconds}
    models = {"LightGBM ranker": fit["models"]["all"]}
    models[f"LightGBM ranker ({HELD_OUT})"] = fit["models"]["held"]
    results = _lgbm_results(parts, models, args.n_jobs)
    n_test = len(parts["test"]["sets"])
    _baseline_table(results, console, f"first-arrow ranking on {args.split}_test (n {n_test})")
    _save_two(args, fit, cfg, sorted(train["seen"]))
    report = _lgbm_report(args, train, parts, fit, results, cfg)
    out = source_dir(args.source, args.processed_dir) / f"gbdt_lgbm_report_{_tag(args)}.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    console.print(
        f"load {load_seconds} s, fit {fit['fit_seconds']} s, peak RSS {report['peak_rss_mb']} MB"
    )
    console.print(f"wrote {out}")
    best = max(r["test"]["top5"] for n, r in results.items() if n.startswith("untrained"))
    return 0 if results["LightGBM ranker"]["test"]["top5"] > best else 1


def print_textbook(results, console: Console, title: str, seen: frozenset = frozenset()) -> int:
    """The suite's table; a case whose reactant pool was in the model's training data is
    marked `seen in training`. Returns the number of held-out cases in the top 3."""
    table = Table(title=title, show_lines=True)
    for col in ("case", "expected", "rank", "top-3 predicted (donor @ site -> acceptor @ site)"):
        table.add_column(col)
    for res in results:
        lines = [
            f"{p['donor']} @ {p['donor_site']} -> {p['acceptor']} @ {p['acceptor_site']}"
            f"  {p['score']:.3g}"
            for p in res.top
        ]
        expected = f"{res.case.expected.label}  {res.expected_score:.3g}"
        for alt, rank, score in res.alternatives:
            better = "below" if score < res.expected_score else "ABOVE"
            expected += f"\nvs {alt.label}: rank {rank}, {score:.3g} ({better} expected)"
        mark = "[green]top-3[/green]" if res.passed else "[red]miss[/red]"
        name = res.case.name + (
            "\n[yellow]seen in training[/yellow]" if res.case.name in seen else ""
        )
        table.add_row(name, expected, f"{res.expected_rank} {mark}", "\n".join(lines))
    console.print(table)
    held = [r for r in results if r.case.name not in seen]
    n = sum(r.passed for r in held)
    console.print(f"{title}: {n}/{len(held)} held-out expected pairs in the top 3")
    if seen:
        n_seen = sum(r.passed for r in results if r.case.name in seen)
        console.print(f"  and {n_seen}/{len(seen)} of the cases seen in training (not a test)")
    return n


def _untrained_scorer(sample, loader) -> np.ndarray:
    """-G on every (occupied, virtual) pair: the untrained gap-kernel score."""
    from ksmech.physics.stage_a import A

    return -sample.pair_z[..., A["g"]].astype(np.float64)


def _gbdt_scorer(path: Path, allow_stale: bool):
    """(scorer, the textbook cases whose pools the model was trained on) of a saved LightGBM
    ranker of `gbdt`."""
    from ksmech.physics import baseline as bl
    from ksmech.physics import lgbm

    gbdt = lgbm.load_lgbm(path, bl.FEATURE_NAMES, allow_stale=allow_stale)
    header = prov.read_header(path)
    seen = frozenset(header.extra.get("textbook_pools_in_training", [])) if header else frozenset()

    def trained(sample, loader):
        cs = bl.candidate_set(sample, loader)
        out = np.full(sample.pair_valid.shape, -np.inf)
        out[cs.rows, cs.cols] = gbdt.score(cs.X)
        return out

    return trained, seen


def cmd_radical(args: argparse.Namespace, console: Console) -> int:
    """The radical selectivity suite (open-shell plan P7 step 11) with the untrained kernel,
    g and g2, with and without the intramolecular open-shell cells, or with `--checkpoint` a
    trained model's anchors (P9 step 6). Diagnostics: exit 0."""
    from ksmech.physics.textbook import (
        RADICAL_CASES,
        ensure_tokens,
        fragment_keys,
        network_scorer,
        run_radical_suite,
    )

    keys = sorted({k for case in RADICAL_CASES for k in fragment_keys(case.smiles)})
    if built := ensure_tokens(keys, args.tokens_dir):
        console.print(f"built tokens for {built}")
    loader = TokenCache(args.tokens_dir)
    if args.checkpoint is not None:
        from ksmech.train.loop import load_model

        model, _ = load_model(args.checkpoint, "cpu")
        out = run_radical_suite(loader, scorer=network_scorer(model))
        print_textbook(out["results"], console, f"radical suite, {args.checkpoint}")
        console.print(f"Me3Sn. abstraction order I > Br > Cl: {out['snap_order_i_br_cl']}")
        return 0
    for kernel in ("g", "g2"):
        for drop in (False, True):
            out = run_radical_suite(loader, kernel, drop)
            title = f"radical suite, |{kernel}|" + (", no intra open-shell cells" if drop else "")
            print_textbook(out["results"], console, title)
            console.print(f"Me3Sn. abstraction order I > Br > Cl: {out['snap_order_i_br_cl']}")
    return 0


def cmd_textbook(args: argparse.Namespace, console: Console) -> int:
    """The selectivity suite with the untrained gap kernel and both rankers. Only results on
    cases the model never trained on count towards the exit code."""
    from ksmech.physics.textbook import CASES, ensure_tokens, fragment_keys, run_suite

    if args.radical:
        return cmd_radical(args, console)

    keys = sorted({k for case in CASES for k in fragment_keys(case.smiles)})
    built = ensure_tokens(keys, args.tokens_dir)
    if built:
        console.print(f"built tokens for {built}")
    loader = TokenCache(args.tokens_dir)
    runs = {"untrained gap_kernel": (_untrained_scorer, frozenset())}
    out_dir = source_dir(args.source, args.processed_dir)
    stem = f"gbdt_lgbm_molecule_{FEATURES}"
    gbdts = {"LightGBM ranker": args.gbdt or out_dir / f"{stem}.npz"}
    gbdts[f"LightGBM ranker ({HELD_OUT})"] = out_dir / f"{stem}_{HELD_OUT}.npz"
    for name, path in gbdts.items():
        if not path.exists():
            console.print(f"[yellow]{path} missing: run `ksmech-physics gbdt`[/yellow]")
            continue
        try:
            runs[name] = _gbdt_scorer(path, args.allow_stale)
        except StaleArtifact as exc:
            console.print(f"[red]{exc}[/red]; re-run `ksmech-physics gbdt`")
            return 2
    passes = {
        name: print_textbook(run_suite(fn, loader), console, name, seen)
        for name, (fn, seen) in runs.items()
    }
    return 0 if max(passes.values()) >= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
