"""`ksmech-network retro`: the previous steps from a product alone (reaction-network plan, item K).

    ksmech-network retro --run BACKWARD_RUN --target SMILES [--run-source pmechdb] \
        [--reservoir-source pmechdb] [--n-byproducts 3] [--max-heavy 4] \
        [--building-blocks SMILES ... | --building-blocks-file FILE] [--max-expansions 100] \
        [--max-depth 7] [--beam 5] [--build-missing] \
        [--verify-run FORWARD_RUN [--verify-expansions 50]] [--out FILE]

The backward checkpoint (`ksmech-train fit --direction backward`) expands species going back
(`network.retro.ModelRetroExpander`), on the species alone and with each of the most frequent
by-products of `--reservoir-source`'s recorded steps; the AND-OR search (`retro_search`) finds
the cheapest route to building blocks (`--building-blocks`: no catalogue ships with the repo) or
reservoir species; `--verify-run` runs the forward search from the route's leaves to the target
with a forward checkpoint, and the route counts as verified only when it arrives.

    ksmech-network retro-eval --set human_benchmark|reactmech --run BACKWARD_RUN \
        [--verify-run FORWARD_RUN] [--part val] [--limit N] [the retro options] [--out FILE]

`retro-eval` runs it on a pathway benchmark (`network.evaluate.retro_case`): each case's final
product the target, its recorded reactants the catalogue (decided 2026-11-10); per case solved,
recovered (the route's leaves outside the reservoir are exactly the recorded reactants),
verified forward, the species expanded; the shares over the cases, as JSON beside the backward
checkpoint (`network_retro_eval_<set>[_<part>][_n<limit>].json`) or at `--out`.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ksmech.data import provenance as prov

RETRO_VERSION = 1


def _retro_options(r, models_dir, tokens_dir, processed_dir) -> None:
    r.add_argument("--run", required=True, help="a backward run under <models-dir>/<run-source>/")
    r.add_argument("--run-source", default="pmechdb", help="the run's models directory")
    r.add_argument("--checkpoint", default="best", choices=["best", "last"])
    r.add_argument("--reservoir-source", default="pmechdb", help="the steps the by-products "
                   "are counted from")  # fmt: skip
    r.add_argument("--n-byproducts", type=int, default=3, help="by-products tried per species")
    r.add_argument("--max-heavy", type=int, default=4, help="heavy atoms of a by-product")
    r.add_argument("--max-expansions", type=int, default=100, help="species expanded")
    r.add_argument("--max-depth", type=int, default=7, help="backward steps on a branch")
    r.add_argument("--beam", type=int, default=5, help="anchors completed and products kept each")
    r.add_argument("--build-missing", action="store_true", help="compute missing tokens (GFN2)")
    r.add_argument("--verify-run", help="a forward run for the forward verification")
    r.add_argument("--verify-expansions", type=int, default=50, help="its node budget")
    r.add_argument("--models-dir", type=Path, default=models_dir)
    r.add_argument("--tokens-dir", type=Path, default=tokens_dir)
    r.add_argument("--processed-dir", type=Path, default=processed_dir)
    r.add_argument("--out", type=Path, help="the result (JSON)")


def add_parser(sub) -> None:
    from ksmech.data.io import DEFAULT_PROCESSED_DIR
    from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
    from ksmech.train.loop import DEFAULT_MODELS_DIR

    r = sub.add_parser("retro", help="the previous steps from a product alone (AND-OR search)")
    _retro_options(r, DEFAULT_MODELS_DIR, DEFAULT_TOKENS_DIR, DEFAULT_PROCESSED_DIR)
    r.add_argument("--target", required=True, help="the product (SMILES)")
    r.add_argument("--building-blocks", nargs="*", default=[], help="catalogue species (SMILES)")
    r.add_argument("--building-blocks-file", type=Path, help="one SMILES per line")
    e = sub.add_parser("retro-eval", help="the search from a product alone on a benchmark")
    _retro_options(e, DEFAULT_MODELS_DIR, DEFAULT_TOKENS_DIR, DEFAULT_PROCESSED_DIR)
    e.add_argument("--set", required=True, choices=["reactmech", "human_benchmark"])
    e.add_argument("--split", default="molecule", help="the split whose parts label the chains")
    e.add_argument("--part", default="all", help="only the cases of this part (all: every one)")
    e.add_argument("--limit", type=int, help="the first N cases")


def _expander(args, run: str):
    from ksmech.arrows.reactants import TokenCache
    from ksmech.model.decode import DecodeConfig
    from ksmech.network.pools import ModelExpander
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.run_source / run / f"{args.checkpoint}.pt"
    model, _ = load_model(ckpt, "cpu")
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam)
    return ckpt, ModelExpander(model, TokenCache(args.tokens_dir), cfg,
                               build_missing=args.build_missing)  # fmt: skip


def _leaves(args, reservoir_keys):
    from ksmech.network.pools import pool_state
    from ksmech.network.retro import Leaves

    smiles = list(args.building_blocks)
    if args.building_blocks_file:
        smiles += [
            x.strip() for x in args.building_blocks_file.read_text().splitlines() if x.strip()
        ]
    keys = {k for smi in smiles for k in pool_state(smi).species}
    return Leaves(catalogue=frozenset(keys), reservoir=frozenset(reservoir_keys))


def cmd_retro(args, console: Console) -> int:
    from collections import Counter

    from ksmech.data.io import read_training_steps, steps_path
    from ksmech.network.retro import (
        ModelRetroExpander,
        RetroParams,
        byproduct_reservoir,
        leaves_of,
        retro_search,
        route_rows,
        verify_forward,
    )  # fmt: skip
    from ksmech.network.search import SearchParams

    skips: Counter = Counter()
    steps_file = steps_path(args.reservoir_source, args.processed_dir)
    byproducts = byproduct_reservoir(read_training_steps(steps_file), args.max_heavy, 20, skips)
    ckpt, backward = _expander(args, args.run)
    expander = ModelRetroExpander(backward, args.target, byproducts, args.n_byproducts)
    leaves = _leaves(args, [k for k, _, _ in byproducts])
    res = retro_search(expander, expander.target, leaves,
                       RetroParams(args.max_expansions, args.max_depth))  # fmt: skip
    verified, n_byproducts = None, args.n_byproducts
    if args.verify_run and res.route is not None:
        _, forward = _expander(args, args.verify_run)
        params = SearchParams(k=1, max_expansions=args.verify_expansions)
        verified, _ = verify_forward(forward, res.route, args.target, expander.smiles, params)
    body = {"reason": res.reason, "expanded": res.expanded, "verified": verified,
            "route": route_rows(res.route), "leaves": leaves_of(res.route),
            "leaf_smiles": {k: expander.smiles.get(k, "") for k in leaves_of(res.route)},
            "cost": None if res.route is None else res.route.cost,
            "byproducts": [{"smiles": s, "steps": n} for _, s, n in byproducts[:n_byproducts]],
            "cuts": dict(res.cuts), "skips": dict(skips + expander.skips)}  # fmt: skip
    console.print(body)
    if args.out:
        keys = ("run", "run_source", "checkpoint", "target", "reservoir_source", "n_byproducts",
                "max_heavy", "building_blocks", "max_expansions", "max_depth", "beam",
                "build_missing", "verify_run", "verify_expansions")  # fmt: skip
        inputs = prov.file_input(ckpt) | prov.file_input(steps_file)
        header = prov.stamp("network_retro", RETRO_VERSION, inputs,
                            {k: getattr(args, k) for k in keys})  # fmt: skip
        prov.write_json(args.out, body, header)
        console.print(f"wrote {args.out}")
    return 0


def _cases(args, counts):
    from ksmech.network.benchmarks import load_cases

    return load_cases(args.set, args.processed_dir, args.split, args.part, args.limit, counts)


def cmd_retro_eval(args, console: Console) -> int:
    from collections import Counter

    from ksmech.data.io import read_training_steps, steps_path
    from ksmech.network.evaluate import retro_case, retro_summary
    from ksmech.network.retro import RetroParams, byproduct_reservoir
    from ksmech.network.search import SearchParams

    counts: Counter = Counter()
    cases = _cases(args, counts)
    steps_file = steps_path(args.reservoir_source, args.processed_dir)
    byproducts = byproduct_reservoir(read_training_steps(steps_file), args.max_heavy, 20, counts)
    ckpt, backward = _expander(args, args.run)
    forward = _expander(args, args.verify_run)[1] if args.verify_run else None
    params = RetroParams(args.max_expansions, args.max_depth)
    fparams = SearchParams(k=1, max_expansions=args.verify_expansions)
    rows = []
    for i, case in enumerate(cases, start=1):
        rows.append(retro_case(backward, case, byproducts, params, args.n_byproducts, forward,
                               fparams))  # fmt: skip
        r = rows[-1]
        console.print(f"case {i} / {len(cases)} {case.case_id}: solved {r.get('solved')}, "
                      f"recovered {r.get('recovered')}, verified {r.get('verified')}")  # fmt: skip
    report = retro_summary(rows) | {"cases_skipped": dict(counts)}
    console.print(report)
    keys = ("set", "run", "run_source", "checkpoint", "split", "part", "limit", "n_byproducts",
            "max_heavy", "max_expansions", "max_depth", "beam", "build_missing", "verify_run",
            "verify_expansions", "reservoir_source")  # fmt: skip
    inputs = prov.file_input(ckpt) | prov.file_input(steps_file)
    header = prov.stamp("network_retro_eval", RETRO_VERSION, inputs,
                        {k: getattr(args, k) for k in keys})  # fmt: skip
    tag = args.set + (f"_{args.part}" if args.part != "all" else "")
    tag += f"_n{args.limit}" if args.limit is not None else ""
    out = args.out or ckpt.parent / f"network_retro_eval_{tag}.json"
    prov.write_json(out, {"summary": report, "cases": rows}, header)
    console.print(f"wrote {out}")
    return 0
