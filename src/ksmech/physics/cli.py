"""`ksmech-physics`: the untrained Klopman-Salem ranking and the descriptor contact kernel."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from ksmech.arrows.reactants import TokenCache
from ksmech.arrows.steps import PARTS, load_samples, samples_path
from ksmech.data import provenance as prov
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir
from ksmech.data.provenance import StaleArtifact
from ksmech.data.schema import Source
from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
from ksmech.physics.contact.legacy_fixed23 import DEFAULT_CACHE_DIR as ORIENTED_CACHE_DIR
from ksmech.physics.contact.legacy_fixed23 import OrientedConfig
from ksmech.physics.contact_kernel import KERNELS
from ksmech.physics.contact_mlp import DEFAULT_MLP_PATH
from ksmech.physics.contact_table import DEFAULT_TABLE_PATH

# the untrained physics floor: KS3 + KS2 at d = 2.3 (user decision, docs/decisions.md)
UNTRAINED_SCORE = "ks3+ks2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-physics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("ks-rank", help="rank first arrows by the untrained KS terms")
    m = sub.add_parser("contact-mlp", help="fit the descriptor contact kernel (kernel=mlp)")
    b = sub.add_parser("baseline", help="train the logistic-regression and GBDT rankers")
    b.add_argument("--split", default="molecule", help="split strategy of the sample files")
    t = sub.add_parser("textbook", help="textbook selectivity checks (expected vs top-3)")
    t.add_argument("--source", default="pmechdb", help="where the trained GBDT model lives")
    t.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    t.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    t.add_argument("--untrained", default=UNTRAINED_SCORE, help="untrained scorer name")
    t.add_argument("--oriented-cache", type=Path, default=ORIENTED_CACHE_DIR)
    t.add_argument("--gbdt", type=Path, default=None, help="GBDT model (default: molecule split)")
    for p in (r, t):  # diagnostic commands only: read stale artifacts with a warning
        p.add_argument("--allow-stale", action="store_true", help="read stale artifacts (warns)")
    for p in (r, m, b):
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
    r.add_argument("--contact-mlp", type=Path, default=DEFAULT_MLP_PATH)
    r.add_argument("--oriented-cache", type=Path, default=ORIENTED_CACHE_DIR)
    m.add_argument("--epochs", type=int, default=20)
    m.add_argument("--window", type=float, default=0.3)
    m.add_argument("--out", type=Path, default=DEFAULT_MLP_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    console = Console()
    commands = {
        "ks-rank": cmd_ks_rank,
        "contact-mlp": cmd_contact_mlp,
        "baseline": cmd_baseline,
        "textbook": cmd_textbook,
    }
    return commands[args.command](args, console)


def make_kernel(args: argparse.Namespace, distance: float | None = None):
    from ksmech.physics.contact_kernel import ContactConfig, ContactKernel
    from ksmech.physics.contact_mlp import load_contact_mlp
    from ksmech.physics.contact_table import load_tables

    tables = load_tables(args.contact_table) if args.kernel == "table" else None
    mlp = load_contact_mlp(args.contact_mlp) if args.kernel == "mlp" else None
    oriented = OrientedConfig(
        distance=distance or OrientedConfig().distance, cache_dir=args.oriented_cache
    )
    return ContactKernel(tables, ContactConfig(kernel=args.kernel, oriented=oriented), mlp)


def _load_parts(args: argparse.Namespace, console: Console) -> dict[str, list] | None:
    parts = {}
    for part in PARTS:
        path = samples_path(args.source, f"{args.split}_{part}", args.processed_dir)
        if not path.exists():
            console.print(f"[red]{path} missing[/red]; run `ksmech-arrows samples`")
            return None
        try:
            parts[part] = load_samples(path, allow_stale=getattr(args, "allow_stale", False))
        except (StaleArtifact, ValueError) as exc:  # stale or old: rebuild, never reinterpret
            console.print(f"[red]{exc}[/red]; re-run `ksmech-arrows samples`")
            return None
    return parts


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
    """(scorer, the textbook cases whose pools the model was trained on) of a saved GBDT."""
    from ksmech.physics import baseline as bl

    gbdt = bl.load_gbdt(path, allow_stale=allow_stale)
    header = prov.read_header(path)
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
    gbdts = {"GBDT ranker": args.gbdt or out_dir / "baseline_gbdt_molecule.npz"}
    gbdts[f"GBDT ranker ({HELD_OUT})"] = out_dir / f"baseline_gbdt_molecule_{HELD_OUT}.npz"
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


def cmd_contact_mlp(args: argparse.Namespace, console: Console) -> int:
    from ksmech.orbitals.build import tokens_path
    from ksmech.physics.contact_mlp import fit_contact_mlp, samples_from_paths, save_contact_mlp
    from ksmech.qm.batch import collect_molecules
    from ksmech.qm.xtb_calc import QMConfig

    method = QMConfig().method
    smiles_list, _ = collect_molecules(args.source, args.processed_dir)
    paths = [tokens_path(args.tokens_dir, s, 0, method) for s in smiles_list]
    samples, skipped = samples_from_paths(paths, method, args.window)
    n_used = len(paths) - sum(skipped.values())
    console.print(
        f"contact samples from {n_used}/{len(paths)} molecules (skipped: {skipped or 'none'})"
    )
    model = fit_contact_mlp(samples, epochs=args.epochs, method=method)
    for kind in ("occ_virt", "occ_occ"):
        meta = model.meta[kind]
        console.print(f"  {kind}: {meta['n_train']} training samples, {meta['n_molecules']} mols")
        for which in ("mlp_held_out", "table_held_out"):
            if which in meta:
                r = meta[which]
                console.print(
                    f"    {which}: R2(log) {r['r2_log']:.3f}, "
                    f"median |log10 err| {r['median_abs_log10_error']:.3f} (n {r['n']})"
                )
    console.print(f"wrote {save_contact_mlp(model, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
