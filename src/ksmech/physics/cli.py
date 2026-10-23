"""`ksmech-physics`: the untrained Klopman-Salem ranking and the descriptor contact kernel."""

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

from ksmech.arrows import contacts
from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.steps import PARTS, load_samples, sample_ids, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.data.provenance import StaleArtifact
from ksmech.data.schema import Source
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
from ksmech.physics.contact.legacy_fixed23 import DEFAULT_CACHE_DIR as ORIENTED_CACHE_DIR
from ksmech.physics.contact.legacy_fixed23 import OrientedConfig
from ksmech.physics.contact_kernel import KERNELS
from ksmech.physics.contact_table import DEFAULT_TABLE_PATH

# the untrained physics floor: KS3 + KS2 at d = 2.3 (user decision, docs/decisions.md)
UNTRAINED_SCORE = "ks3+ks2"
# `ksmech-physics gbdt --features`: the candidate pair set and the scorer of the hard negatives
GBDT_PAIRS = {"contact": "legacy", "geomfree": "stage_a"}
HARD_NEGATIVES = {"contact": "ks3+ks2", "geomfree": "gap_kernel"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-physics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("ks-rank", help="rank first arrows by the untrained KS terms")
    b = sub.add_parser("baseline", help="train the logistic-regression and GBDT rankers")
    b.add_argument("--split", default="molecule", help="split strategy of the sample files")
    g = sub.add_parser("gbdt", help="the fast LightGBM lambdarank ranker")
    g.add_argument("--split", default="molecule", help="split strategy of the sample files")
    g.add_argument(
        "--features",
        choices=sorted(GBDT_PAIRS),
        default="contact",
        help="contact: legacy contact features on the sidecar subsample; geomfree: stage-A "
        "pair features on the whole training part",
    )
    g.add_argument("--hard", type=int, default=20, help="negatives per step: best untrained")
    g.add_argument("--random", type=int, default=30, help="negatives per step: uniform draw")
    g.add_argument("--n-jobs", type=int, default=2, help="LightGBM threads")
    t = sub.add_parser("textbook", help="textbook selectivity checks (expected vs top-3)")
    t.add_argument("--source", default="pmechdb", help="where the trained GBDT model lives")
    t.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    t.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    t.add_argument("--untrained", default=UNTRAINED_SCORE, help="untrained scorer name")
    t.add_argument("--oriented-cache", type=Path, default=ORIENTED_CACHE_DIR)
    t.add_argument("--gbdt", type=Path, default=None, help="GBDT model (default: molecule split)")
    t.add_argument("--fast", action="store_true", help="the LightGBM models of `gbdt` (contact)")
    for p in (r, t):  # diagnostic commands only: read stale artifacts with a warning
        p.add_argument("--allow-stale", action="store_true", help="read stale artifacts (warns)")
    for p in (r, b, g):
        p.add_argument("--source", required=True, choices=[str(x) for x in Source])
        p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
        p.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    r.add_argument("--split", default="molecule", help="split strategy of the sample files")
    r.add_argument("--kernel", choices=KERNELS, default="oriented")
    r.add_argument(
        "--distance",
        type=float,
        nargs="+",
        default=[OrientedConfig().distance],
        help="oriented kernel: main-centre distance(s) in Angstrom, one report each",
    )
    r.add_argument("--contact-table", type=Path, default=DEFAULT_TABLE_PATH)
    r.add_argument("--oriented-cache", type=Path, default=ORIENTED_CACHE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    commands = {
        "ks-rank": cmd_ks_rank,
        "baseline": cmd_baseline,
        "gbdt": cmd_gbdt,
        "textbook": cmd_textbook,
    }
    return commands[args.command](args, console)


def make_kernel(args: argparse.Namespace, distance: float | None = None):
    from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
    from ksmech.physics.contact_table import load_tables

    tables = load_tables(args.contact_table) if args.kernel == "table" else None
    oriented = OrientedConfig(
        distance=distance or OrientedConfig().distance, cache_dir=args.oriented_cache
    )
    return ContactKernel(tables, ContactConfig(kernel=args.kernel, oriented=oriented))


def _load_parts(
    args: argparse.Namespace, console: Console, parts: tuple[str, ...] = PARTS
) -> dict[str, list] | None:
    """The samples of `parts` of `args.split` that have a legacy contact entry, joined with it
    (the training part is the sidecar's subsample). Only those samples and the sidecar shards
    holding them are read; the steps without an entry are counted."""
    allow_stale = getattr(args, "allow_stale", False)
    view_dir = source_dir(args.source, args.processed_dir)
    out = {}
    for part in parts:
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        if not path.exists():
            console.print(f"[red]{path} missing[/red]; run `ksmech-arrows samples`")
            return None
        try:
            ids = sample_ids(path, allow_stale)
            legacy = contacts.load_contacts(view_dir, args.split, allow_stale, ids)
            samples = load_samples(path, allow_stale, ids=legacy)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]; run `ksmech-arrows contacts --split {args.split}`")
            return None
        except StaleArtifact as exc:
            console.print(f"[red]{exc}[/red]; re-run `ksmech-arrows samples` / `contacts`")
            return None
        except ValueError as exc:  # an old sample store: rebuild, never reinterpret
            console.print(f"[red]{exc}[/red]; re-run `ksmech-arrows samples`")
            return None
        out[part] = [contacts.with_legacy(s, legacy.pop(s.step_id)) for s in samples]
        console.print(
            f"{args.split}_{part}: {len(out[part])} samples with contacts, "
            f"{len(ids) - len(out[part])} without"
        )
    return out


def print_rank_table(report: dict, console: Console, title: str) -> None:
    table = Table(title=title)
    for col in ("scorer", "subset", "n", "top-1", "top-3", "top-5", "median rank", "no label"):
        table.add_column(col)
    for name, by_subset in report["scorers"].items():
        for subset, r in by_subset.items():
            table.add_row(
                name,
                subset,
                str(r["n"]),
                *(f"{r[f'top{k}']:.1%}" for k in (1, 3, 5)),
                f"{r['median_rank']:.0f}",
                str(r["n_no_valid_label"]),
            )
    console.print(table)


def print_failures(fc: dict, console: Console, name: str) -> None:
    console.print(f"{name} misses (rank >= 5): {fc['n_miss']} of {fc['n']} samples")
    console.print("  facts: " + ", ".join(f"{k} {v:.1%}" for k, v in fc["miss_facts"].items()))
    for key, title in (
        ("miss_label_classes", f"{name}: labeled pair of the misses (diagnostic labels)"),
        ("miss_top_classes", f"{name}: pair ranked first instead"),
    ):
        table = Table(title=title)
        table.add_column("count")
        table.add_column("donor -> acceptor")
        for cls, n in fc[key][:10]:
            table.add_row(str(n), cls)
        console.print(table)


def print_class_table(classes: dict, console: Console, min_n: int = 5) -> None:
    names = [k for k in next(iter(classes.values())) if k != "n"] if classes else []
    table = Table(title=f"top-5 hit rate by labeled-pair class (classes with >= {min_n} samples)")
    for col in ("class", "n", *names):
        table.add_column(col)
    for cls, r in classes.items():
        if r["n"] >= min_n:
            table.add_row(cls, str(r["n"]), *(f"{r[n]:.0%}" for n in names))
    console.print(table)


FLOORS = ("ks3+ks2", "frozen_head")  # scorers whose misses are broken down by class


def _subsets(parts: dict, comps: dict) -> dict[str, tuple[list, list]]:
    """(samples, components) of the test split, train + val and everything."""

    def join(names):
        return [s for p in names for s in parts[p]], [c for p in names for c in comps[p]]

    return {"test": join(["test"]), "trainval": join(["train", "val"]), "all": join(PARTS)}


def clash_summary(samples, comps, ranks) -> dict:
    """CLASH shares: over the valid pairs (within / across fragments), over the pairs ranked
    first by the floor (KS3 + KS2), and over the first arrows' teacher pairs."""
    from ksmech.physics.ranking import SCORERS, teacher_cell

    intra, inter, top, teacher = [], [], [], []
    for s, c in zip(samples, comps, strict=True):
        v = s.pair_valid
        intra.append(c["clash"][v & ~c["inter"]])
        inter.append(c["clash"][v & c["inter"]])
        sc = np.where(v, SCORERS["ks3+ks2"](c), -np.inf)
        top.append(bool(c["clash"][np.unravel_index(np.argmax(sc), sc.shape)]))
        cell = teacher_cell(s)
        if cell is not None:
            teacher.append(bool(c["clash"][cell]))

    def share(x) -> float:
        x = np.concatenate(x) if x and isinstance(x[0], np.ndarray) else np.array(x)
        return float(x.mean()) if x.size else float("nan")

    return {
        "valid_intramolecular": share(intra),
        "valid_intermolecular": share(inter),
        "floor_top_ranked": share(top),
        "teacher_pair": share(teacher),
    }


def _rank_report(parts: dict, loader, kernel) -> dict:
    from ksmech.physics.ranking import (
        ALL_SCORERS,
        class_table,
        evaluate,
        failure_classes,
        pair_components,
        refresh_validity,
    )

    scorers = ALL_SCORERS
    parts = {p: [refresh_validity(s, loader) for s in xs] for p, xs in parts.items()}
    comps = {p: [pair_components(s, loader, kernel) for s in xs] for p, xs in parts.items()}
    subsets = _subsets(parts, comps)
    results = {sub: evaluate(*subsets[sub], scorers) for sub in subsets}
    report = {"scorers": {}, "failure_classes": {}}
    for name in scorers:
        report["scorers"][name] = {
            sub: {k: v for k, v in res[name].items() if k != "ranks"}
            for sub, res in results.items()
        }
    everything, all_comps = subsets["all"]
    for name in FLOORS:
        ranks = results["all"][name]["ranks"]
        fc = failure_classes(everything, all_comps, ranks, loader, scorers[name])
        report["failure_classes"][name] = fc
    rows = report["failure_classes"]["ks3+ks2"]["rows"]
    ranks_all = {n: results["all"][n]["ranks"] for n in scorers}
    report["class_table"] = class_table(rows, ranks_all)
    if kernel.config.kernel == "oriented":
        report["clash"] = clash_summary(everything, all_comps, ranks_all["ks3+ks2"])
    sc = report["scorers"]
    report["floor"] = {
        f"{n}_{sub}_top{k}": sc[n][sub][f"top{k}"]
        for n in FLOORS
        for sub in ("trainval", "test")
        for k in (1, 3, 5)
    }
    return report


def cmd_ks_rank(args: argparse.Namespace, console: Console) -> int:
    parts = _load_parts(args, console)
    if parts is None:
        return 2
    loader = TokenCache(args.tokens_dir)
    distances = args.distance if args.kernel == "oriented" else [None]
    for d in distances:
        kernel = make_kernel(args, d)
        tag = f"{args.split}_{args.kernel}" + ("" if d is None else f"_d{d:g}")
        report = {"split": args.split, "kernel": args.kernel, "distance": d}
        report |= _rank_report(parts, loader, kernel)
        print_rank_table(report, console, f"first-arrow ranking, untrained KS, {tag}")
        for name in FLOORS:
            print_failures(report["failure_classes"][name], console, name)
        print_class_table(report["class_table"], console)
        if "clash" in report:
            console.print(
                "CLASH shares: " + ", ".join(f"{k} {v:.1%}" for k, v in report["clash"].items())
            )
        floor = report["floor"]
        console.print(
            f"floor ({tag}): KS3 + KS2 top-1/3/5 train + val "
            + "/".join(f"{floor[f'ks3+ks2_trainval_top{k}']:.1%}" for k in (1, 3, 5))
            + ", test "
            + "/".join(f"{floor[f'ks3+ks2_test_top{k}']:.1%}" for k in (1, 3, 5))
        )
        out = source_dir(args.source, args.processed_dir) / f"ks_rank_report_{tag}.json"
        out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        console.print(f"wrote {out}")
    return 0


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


def _loco_note(args: argparse.Namespace) -> str:
    found = sorted(source_dir(args.source, args.processed_dir).glob("samples_loco-*_test.json"))
    if found:
        return f"{len(found)} leave-one-class-out splits found (evaluate with --split loco-<class>)"
    return "leave-one-class-out: not applicable (the source has no mechanism classes)"


UNTRAINED_ROWS = ("ks3+ks2", "frozen_head", "ks3", "fbar2_g", "fbar2_g+ks2", "gap_kernel")
HELD_OUT = "textbook-held-out"  # file tag of the GBDT fitted without the textbook pools


def _results(bl, sets: dict, scores: dict) -> dict:
    return {
        name: {
            part: {
                k: v
                for k, v in bl.evaluate_scores(sets[part], [fn(cs) for cs in sets[part]]).items()
                if k != "ranks"
            }
            for part in ("train", "val", "test")
        }
        for name, fn in scores.items()
    }


def _textbook_split(parts: dict, sets: dict) -> tuple[dict, dict[str, int]]:
    """The candidate sets without every sample whose reactants contain a textbook pool (train
    and val only; test is untouched), and how many train samples held each pool."""
    from ksmech.physics.textbook import pools_in

    seen: dict[str, int] = {}
    for s in parts["train"]:
        for name in pools_in(s.fragment_keys):
            seen[name] = seen.get(name, 0) + 1
    held = {
        p: [cs for s, cs in zip(parts[p], sets[p], strict=True) if not pools_in(s.fragment_keys)]
        for p in ("train", "val")
    }
    return held | {"test": sets["test"]}, seen


def _fit(bl, sets: dict, console: Console, tag: str):
    lr, scan = bl.select_logistic(sets["train"], sets["val"])
    gbdt = bl.fit_gbdt(sets["train"], sets["val"], bl.TreeConfig())
    console.print(f"{tag}: logistic L2 {lr.l2}, GBDT {len(gbdt.trees)} trees")
    return lr, scan, gbdt


def cmd_baseline(args: argparse.Namespace, console: Console) -> int:
    from ksmech.physics import baseline as bl

    parts = _load_parts(args, console)
    if parts is None:
        return 2
    loader = TokenCache(args.tokens_dir)
    sets = {p: [bl.candidate_set(s, loader) for s in xs] for p, xs in parts.items()}
    held, seen = _textbook_split(parts, sets)
    console.print(f"textbook pools in {args.split}_train: {seen or 'none'}")
    lr, scan, gbdt = _fit(bl, sets, console, "all training samples")
    _, _, gbdt_held = _fit(bl, held, console, "textbook pools held out")
    scores = {
        f"untrained {n}": (lambda cs, n=n: bl.untrained_scores(cs)[n]) for n in UNTRAINED_ROWS
    }
    scores |= {"logistic regression": lambda cs: lr.score(cs.X)}
    scores |= {"GBDT ranker": lambda cs: gbdt.score(cs.X)}
    scores |= {f"GBDT ranker ({HELD_OUT})": lambda cs: gbdt_held.score(cs.X)}
    results = _results(bl, sets, scores)
    title = f"first-arrow ranking on {args.split}_test (n {len(sets['test'])})"
    _baseline_table(results, console, title)
    note = _loco_note(args)
    console.print(note)
    out_dir = source_dir(args.source, args.processed_dir)
    inputs = {}
    for part in ("train", "val"):
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        inputs |= prov.file_input(path)
    side = contacts.contacts_dir(out_dir, args.split)
    inputs |= prov.file_input(side / "index.json") | prov.file_input(side / "train_subsample.json")
    seen_note = {"textbook_pools_in_training": sorted(seen)}
    bl.save_gbdt(gbdt, out_dir / f"baseline_gbdt_{args.split}.npz", inputs, seen_note)
    bl.save_gbdt(
        gbdt_held,
        out_dir / f"baseline_gbdt_{args.split}_{HELD_OUT}.npz",
        inputs,
        {"textbook_pools_in_training": [], "held_out": sorted(seen)},
    )
    bl.save_logistic(lr, out_dir / f"baseline_lr_{args.split}.npz", inputs)
    report = {"split": args.split, "results": results, "loco": note, "lr_scan": scan}
    report |= {"gbdt_history": gbdt.history, "features": bl.FEATURE_NAMES, "textbook_seen": seen}
    out = out_dir / f"baseline_report_{args.split}.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    best_untrained = max(results[f"untrained {n}"]["test"]["top5"] for n in FLOORS)
    return 0 if results["GBDT ranker"]["test"]["top5"] > best_untrained else 1


def _untrained(cs, features: str) -> dict[str, np.ndarray]:
    from ksmech.physics import baseline as bl

    if features == "contact":
        scores = bl.untrained_scores(cs)
        return {n: scores[n] for n in UNTRAINED_ROWS}
    return {"gap_kernel": bl.gap_score(cs, "stage_a")}


def _read_part(args, features: str, part: str, loader, visit) -> tuple[int, int]:
    """`visit(sample, candidate_set)` for every sample of one part, one shard at a time (with
    `contact`, only the samples that have a legacy contact entry, joined with it). Returns
    (stored samples of the view, samples visited)."""
    from ksmech.arrows.steps import iter_samples
    from ksmech.physics import baseline as bl

    path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
    ids = sample_ids(path)
    legacy = None
    if features == "contact":
        view_dir = source_dir(args.source, args.processed_dir)
        legacy = contacts.load_contacts(view_dir, args.split, ids=ids)
    n = 0
    for s in iter_samples(path, ids=ids if legacy is None else legacy):
        if legacy is not None:
            s = contacts.with_legacy(s, legacy.pop(s.step_id))
        visit(s, bl.candidate_set(s, loader, GBDT_PAIRS[features]))
        n += 1
    return len(ids), n


def _train_rows(args, features: str, loader, cfg) -> dict:
    """The sampled training rows of every step and of the steps without a textbook pool."""
    from ksmech.physics.lgbm import TrainRows
    from ksmech.physics.textbook import pools_in

    out = {"all": TrainRows(), "held": TrainRows(), "seen": {}}

    def visit(s, cs):
        hard = _untrained(cs, features)[HARD_NEGATIVES[features]]
        out["all"].add(cs, hard, cfg)
        pools = pools_in(s.fragment_keys)
        for name in pools:
            out["seen"][name] = out["seen"].get(name, 0) + 1
        if not pools:
            out["held"].add(cs, hard, cfg)

    out["n_view"], out["n"] = _read_part(args, features, "train", loader, visit)
    return out


def _eval_part(args, features: str, part: str, loader) -> dict:
    """Every candidate set of a val / test part (X kept as float32), its untrained scores
    (computed at full precision first) and which samples hold a textbook pool."""
    from ksmech.physics.textbook import pools_in

    out = {"sets": [], "untrained": {}, "textbook": []}

    def visit(s, cs):
        for name, score in _untrained(cs, features).items():
            out["untrained"].setdefault(name, []).append(score)
        cs.X = cs.X.astype(np.float32)
        out["sets"].append(cs)
        out["textbook"].append(bool(pools_in(s.fragment_keys)))

    out["n_view"], out["n"] = _read_part(args, features, part, loader, visit)
    return out


def _lgbm_results(parts: dict, models: dict) -> dict:
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
            p: summary(parts[p]["sets"], [model.score(cs.X) for cs in parts[p]["sets"]])
            for p in ("val", "test")
        }
    return results


def _fit_two(train: dict, val: dict, features: str, cfg, console: Console) -> dict:
    """The ranker on every training step and the one without the textbook pools (validated
    without them too), with fit times."""
    from ksmech.physics import baseline as bl
    from ksmech.physics.lgbm import EvalRows, fit_lgbm

    names = bl.PAIR_SETS[GBDT_PAIRS[features]]
    val_held = [cs for cs, t in zip(val["sets"], val["textbook"], strict=True) if not t]
    models, seconds = {}, {}
    for tag, rows, vsets in (("all", train["all"], val["sets"]), ("held", train["held"], val_held)):
        start = time.perf_counter()
        models[tag] = fit_lgbm(rows, EvalRows.of(vsets), names, cfg)
        seconds[tag] = round(time.perf_counter() - start, 1)
        console.print(
            f"{tag}: {rows.n_rows} rows in {len(rows.groups)} queries, "
            f"{models[tag].n_trees} trees, {seconds[tag]} s"
        )
    return {"models": models, "fit_seconds": seconds}


def _lgbm_inputs(args, features: str) -> dict[str, str]:
    inputs = {}
    for part in ("train", "val"):
        inputs |= prov.file_input(
            samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        )
    if features == "contact":
        side = contacts.contacts_dir(source_dir(args.source, args.processed_dir), args.split)
        inputs |= prov.file_input(side / "index.json")
        inputs |= prov.file_input(side / "train_subsample.json")
    return inputs


def _save_two(args, fit: dict, cfg, seen: list[str]) -> None:
    from ksmech.physics import lgbm

    out_dir = source_dir(args.source, args.processed_dir)
    tag, inputs = f"{args.split}_{args.features}", _lgbm_inputs(args, args.features)
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
        "features": args.features,
        "results": results,
        "counts": {
            p: {"read": d["n"], "stored": d["n_view"]}
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
        train = _train_rows(args, args.features, loader, cfg)
        parts = {p: _eval_part(args, args.features, p, loader) for p in ("val", "test")}
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]; run `ksmech-arrows samples` / `contacts`")
        return None
    except (StaleArtifact, ValueError) as exc:  # stale or old: rebuild, never reinterpret
        console.print(f"[red]{exc}[/red]; rebuild it (`ksmech-arrows samples` / `contacts`)")
        return None
    for p, d in (("train", train), *parts.items()):
        console.print(f"{args.split}_{args.features} {p}: {d['n']} samples read of {d['n_view']}")
    console.print(f"textbook pools in {args.split}_train: {train['seen'] or 'none'}")
    return train, parts


def cmd_gbdt(args: argparse.Namespace, console: Console) -> int:
    """The fast LightGBM ranker (`physics.lgbm`) on the contact or geometry-free features."""
    from ksmech.physics.lgbm import LGBMConfig

    cfg = LGBMConfig(n_hard=args.hard, n_random=args.random, n_jobs=args.n_jobs)
    loader = TokenCache(args.tokens_dir, maxsize=128)  # tokens only for the LMO energies
    start = time.perf_counter()
    read = _read_all(args, loader, cfg, console)
    if read is None:
        return 2
    train, parts = read
    load_seconds = round(time.perf_counter() - start, 1)
    fit = _fit_two(train, parts["val"], args.features, cfg, console) | {
        "load_seconds": load_seconds
    }
    models = {"LightGBM ranker": fit["models"]["all"]}
    models[f"LightGBM ranker ({HELD_OUT})"] = fit["models"]["held"]
    results = _lgbm_results(parts, models)
    n_test = len(parts["test"]["sets"])
    _baseline_table(results, console, f"first-arrow ranking on {args.split}_test (n {n_test})")
    _save_two(args, fit, cfg, sorted(train["seen"]))
    report = _lgbm_report(args, train, parts, fit, results, cfg)
    out = (
        source_dir(args.source, args.processed_dir)
        / f"gbdt_lgbm_report_{args.split}_{args.features}.json"
    )
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


def _gbdt_scorer(path: Path, allow_stale: bool):
    """(scorer, the textbook cases whose pools the model was trained on) of a saved GBDT: the
    numpy ranker of `baseline` or the LightGBM one of `gbdt --features contact`."""
    from ksmech.physics import baseline as bl
    from ksmech.physics import lgbm

    header = prov.read_header(path)
    if header is not None and header.kind == "baseline_lgbm":
        gbdt = lgbm.load_lgbm(path, bl.FEATURE_NAMES, allow_stale=allow_stale)
    else:
        gbdt = bl.load_gbdt(path, allow_stale=allow_stale)
    seen = frozenset(header.extra.get("textbook_pools_in_training", [])) if header else frozenset()

    def trained(sample, loader):
        cs = bl.candidate_set(sample, loader)
        out = np.full(sample.pair_valid.shape, -np.inf)
        out[cs.rows, cs.cols] = gbdt.score(cs.X)
        return out

    return trained, seen


def cmd_textbook(args: argparse.Namespace, console: Console) -> int:
    """The selectivity suite with the untrained scorer and both GBDTs. Only results on cases
    the model never trained on count towards the exit code."""
    from ksmech.physics.pairs import oriented_kernels
    from ksmech.physics.ranking import ALL_SCORERS, pair_components
    from ksmech.physics.textbook import CASES, ensure_tokens, fragment_keys, run_suite

    keys = sorted({k for case in CASES for k in fragment_keys(case.smiles)})
    built = ensure_tokens(keys, args.tokens_dir)
    if built:
        console.print(f"built tokens for {built}")
    loader = TokenCache(args.tokens_dir)
    kernels = oriented_kernels(args.oriented_cache)
    scorer = ALL_SCORERS[args.untrained]

    def untrained(sample, loader):
        return scorer(pair_components(sample, loader, kernels.main))

    runs = {f"untrained {args.untrained}": (untrained, frozenset())}
    out_dir = source_dir(args.source, args.processed_dir)
    stem = "gbdt_lgbm_molecule_contact" if args.fast else "baseline_gbdt_molecule"
    gbdts = {"GBDT ranker": args.gbdt or out_dir / f"{stem}.npz"}
    gbdts[f"GBDT ranker ({HELD_OUT})"] = out_dir / f"{stem}_{HELD_OUT}.npz"
    for name, path in gbdts.items():
        if not path.exists():
            console.print(f"[yellow]{path} missing: run `ksmech-physics baseline`[/yellow]")
            continue
        try:
            runs[name] = _gbdt_scorer(path, args.allow_stale)
        except StaleArtifact as exc:
            console.print(f"[red]{exc}[/red]; re-run `ksmech-physics baseline`")
            return 2
    passes = {
        name: print_textbook(run_suite(fn, loader, kernels), console, name, seen)
        for name, (fn, seen) in runs.items()
    }
    return 0 if max(passes.values()) >= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
