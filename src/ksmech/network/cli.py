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
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.data import provenance as prov

SEARCH_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
    from ksmech.train.loop import DEFAULT_MODELS_DIR

    p = argparse.ArgumentParser(prog="ksmech-network", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="the k best pathways from a pool to a target product")
    s.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    s.add_argument("--run-source", default="pmechdb", help="the run's models directory")
    s.add_argument("--checkpoint", default="best", choices=["best", "last"])
    s.add_argument("--start", required=True, help="the starting pool (SMILES, mapped or not)")
    s.add_argument("--target", required=True, help="the target product(s), mapped on the start "
                   "pool's map numbers for the remaining-change heuristic")  # fmt: skip
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
    s.add_argument("--out", type=Path, help="the result (JSON)")
    return p


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
    from ksmech.arrows.reactants import TokenCache
    from ksmech.model.decode import DecodeConfig
    from ksmech.network.pools import ModelExpander, pool_state
    from ksmech.network.search import SearchParams
    from ksmech.network.target import target_from_smiles
    from ksmech.network.towards import pathway_rows, search_towards
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.run_source / args.run / f"{args.checkpoint}.pt"
    model, _ = load_model(ckpt, "cpu")
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam)
    expander = ModelExpander(
        model, TokenCache(args.tokens_dir), cfg, build_missing=args.build_missing
    )
    params = SearchParams(args.k, args.max_expansions, args.max_depth, step_cost=args.step_cost)
    caps = (cfg.max_bonds, cfg.max_atoms)
    target = target_from_smiles(args.target)
    res = search_towards(expander, pool_state(args.start), target, params, tuple(args.levels), caps)
    rows = pathway_rows(res)
    _print_result(console, res, rows)
    if args.out:
        keys = ("run", "run_source", "checkpoint", "start", "target", "k", "max_expansions",
                "max_depth", "step_cost", "levels", "beam", "build_missing")  # fmt: skip
        header = prov.stamp("network_search", SEARCH_VERSION, prov.file_input(ckpt),
                            {k: getattr(args, k) for k in keys})  # fmt: skip
        body = {"reason": res.search.reason, "expanded": res.search.expanded, "pathways": rows,
                "classes": [{"steps": list(c.steps), "probability": c.probability,
                             "n_pathways": len(c.pathways)} for c in res.classes],
                "cuts": dict(res.search.cuts), "skips": dict(res.skips)}  # fmt: skip
        prov.write_json(args.out, body, header)
        console.print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"search": cmd_search}[args.command](args, Console())


if __name__ == "__main__":
    raise SystemExit(main())
