"""`ksmech-network`: the reaction network from the command line (reaction-network plan, item D).

    ksmech-network search --run pm_control_s0 --start 'C[O-].CC(C)=O' --target 'CC(C)(O)OC' \
        [--k 5] [--max-expansions 50] [--max-depth 6] [--step-cost 0] [--levels exact skeleton] \
        [--beam 5] [--build-missing] [--out FILE]

`search` loads a trained checkpoint (CPU), expands pool states with it (`network.pools`) and
runs the search towards the target (`network.towards`): the k best pathways with their arrival
level, log P and cost, the mechanism classes up to the order of independent steps with
p(class | reaches) among the pathways found, the nodes expanded, the cuts and every skip code.
`--out` writes them as JSON with a provenance header. A molecule without tokens fails its node
(MISSING_TOKENS) unless `--build-missing` computes them with GFN2-xTB first.

    ksmech-network evaluate --set reactmech|human_benchmark --run NAME [--run-source reactmech] \
        [--part val] [--limit N] [the search options] [--out FILE]

`evaluate` runs the search on a pathway benchmark (`network.benchmarks`, `network.evaluate`):
per case the arrival, the recorded pathway's rank, the step targets passed and the expansions;
over the cases the rates. Its JSON goes to `--out`, or beside the checkpoint as
`network_eval_<set>[_<part>][_n<limit>].json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.data import provenance as prov

SEARCH_VERSION = 1
SEARCH_KEYS = ("k", "max_expansions", "max_depth", "step_cost", "levels", "beam", "build_missing")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ksmech-network", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="the k best pathways from a pool to a target product")
    _search_options(s)
    s.add_argument("--start", required=True, help="the starting pool (SMILES, mapped or not)")
    s.add_argument("--target", required=True, help="the target product(s), mapped on the start "
                   "pool's map numbers for the remaining-change heuristic")  # fmt: skip
    s.add_argument("--out", type=Path, help="the result (JSON)")
    _evaluate_parser(sub)
    return p


def _search_options(s) -> None:
    from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
    from ksmech.train.loop import DEFAULT_MODELS_DIR

    s.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    s.add_argument("--run-source", default="pmechdb", help="the run's models directory")
    s.add_argument("--checkpoint", default="best", choices=["best", "last"])
    s.add_argument("--k", type=int, default=5, help="pathways to report")
    s.add_argument("--max-expansions", type=int, default=50, help="nodes expanded (model calls)")
    s.add_argument("--max-depth", type=int, default=6, help="steps of a pathway")
    s.add_argument("--step-cost", type=float, default=0.0, help="added to -log p per step")
    s.add_argument("--levels", nargs="+", default=["EXACT", "SKELETON"], type=str.upper,
                   choices=["EXACT", "SKELETON"], help="arrival levels that count")  # fmt: skip
    s.add_argument("--beam", type=int, default=5, help="anchors completed and products kept each")
    s.add_argument("--build-missing", action="store_true", help="compute missing tokens (GFN2)")
    s.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    s.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)


def _evaluate_parser(sub) -> None:
    from ksmech.data.io import DEFAULT_PROCESSED_DIR

    e = sub.add_parser("evaluate", help="the search towards the product on a pathway benchmark")
    e.add_argument("--set", required=True, choices=["reactmech", "human_benchmark"])
    _search_options(e)
    e.add_argument("--split", default="molecule", help="the split whose parts label the chains")
    e.add_argument("--part", default="all", help="only the cases of this part (all: every one)")
    e.add_argument("--limit", type=int, help="the first N cases")
    e.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    e.add_argument("--out", type=Path, help="the result (JSON)")


def _print_result(console: Console, res, rows: list[dict]) -> None:
    table = Table(title=f"pathways ({res.search.reason}, {res.search.expanded} nodes expanded)")
    for col in ("rank", "steps", "log P", "arrival", "class", "pools"):
        table.add_column(col)
    for r in rows:
        pools = " >> ".join(r["pools"][1:])
        table.add_row(str(r["rank"]), str(len(r["steps"])), f"{r['logp']:.2f}", r["arrival"],
                      str(r["mechanism_class"]), pools)  # fmt: skip
    console.print(table)
    for k, c in enumerate(res.classes):
        console.print(f"class {k}: p(class | reaches) {c.probability:.3f}, "
                      f"{len(c.pathways)} pathway(s), {len(c.steps)} step(s)")  # fmt: skip
    for code, n in sorted({**res.search.cuts, **res.skips}.items()):
        console.print(f"  {code}: {n}")


def cmd_search(args, console: Console) -> int:
    from ksmech.network.pools import pool_state
    from ksmech.network.search import SearchParams
    from ksmech.network.target import target_from_smiles
    from ksmech.network.towards import pathway_rows, search_towards

    ckpt, expander, cfg = _expander(args)
    params = SearchParams(args.k, args.max_expansions, args.max_depth, step_cost=args.step_cost)
    caps = (cfg.max_bonds, cfg.max_atoms)
    target = target_from_smiles(args.target)
    res = search_towards(expander, pool_state(args.start), target, params, tuple(args.levels), caps)
    rows = pathway_rows(res)
    _print_result(console, res, rows)
    if args.out:
        keys = ("run", "run_source", "checkpoint", "start", "target", *SEARCH_KEYS)
        header = prov.stamp("network_search", SEARCH_VERSION, prov.file_input(ckpt),
                            {k: getattr(args, k) for k in keys})  # fmt: skip
        body = {"reason": res.search.reason, "expanded": res.search.expanded, "pathways": rows,
                "classes": [{"steps": list(c.steps), "probability": c.probability,
                             "n_pathways": len(c.pathways)} for c in res.classes],
                "cuts": dict(res.search.cuts), "skips": dict(res.skips)}  # fmt: skip
        prov.write_json(args.out, body, header)
        console.print(f"wrote {args.out}")
    return 0


def _expander(args):
    """(the checkpoint path, the model's expander, the decoder caps) of the search options."""
    from ksmech.arrows.reactants import TokenCache
    from ksmech.model.decode import DecodeConfig
    from ksmech.network.pools import ModelExpander
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.run_source / args.run / f"{args.checkpoint}.pt"
    model, _ = load_model(ckpt, "cpu")
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam)
    loader = TokenCache(args.tokens_dir)
    return ckpt, ModelExpander(model, loader, cfg, build_missing=args.build_missing), cfg


def _cases(args, counts):
    from ksmech.network.benchmarks import human_benchmark_cases, reactmech_cases

    if args.set == "reactmech":
        cases = reactmech_cases(args.processed_dir, args.split, counts)
    else:
        cases = human_benchmark_cases(None, counts)
    cases = [c for c in cases if args.part == "all" or c.part == args.part]
    return cases[: args.limit] if args.limit is not None else cases


def cmd_evaluate(args, console: Console) -> int:
    from collections import Counter

    from ksmech.network.evaluate import evaluate_case, summary
    from ksmech.network.search import SearchParams

    counts: Counter = Counter()
    cases = _cases(args, counts)
    ckpt, expander, cfg = _expander(args)
    params = SearchParams(args.k, args.max_expansions, args.max_depth, step_cost=args.step_cost)
    rows = []
    for i, case in enumerate(cases, start=1):
        rows.append(evaluate_case(expander, case, params, tuple(args.levels),
                                  (cfg.max_bonds, cfg.max_atoms)))  # fmt: skip
        r = rows[-1]
        console.print(f"case {i} / {len(cases)} {case.case_id}: {r.get('arrival')} "
                      f"rank {r.get('recorded_rank')}, {r.get('expanded')} expanded")  # fmt: skip
    report = summary(rows, args.k) | {"cases_skipped": dict(counts)}
    console.print(report)
    keys = ("set", "run", "run_source", "checkpoint", "split", "part", "limit", *SEARCH_KEYS)
    header = prov.stamp("network_evaluate", SEARCH_VERSION, prov.file_input(ckpt),
                        {k: getattr(args, k) for k in keys})  # fmt: skip
    tag = args.set + (f"_{args.part}" if args.part != "all" else "")
    tag += f"_n{args.limit}" if args.limit is not None else ""
    out = args.out or ckpt.parent / f"network_eval_{tag}.json"
    prov.write_json(out, {"summary": report, "cases": rows}, header)
    console.print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"search": cmd_search, "evaluate": cmd_evaluate}[args.command](args, Console())


if __name__ == "__main__":
    raise SystemExit(main())
