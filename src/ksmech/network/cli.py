"""`ksmech-network`: the reaction network from the command line (reaction-network plan, items D,
E, F, I3 and I4).

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

    ksmech-network predict --run pm_control_s0 --start 'C[O-].CC(C)=O' [--temperature-k 298.15] \
        [--max-expansions 50] [--mass-tol 0.01] [--epsilon 0.01] [--margin 5.0] \
        [--max-climb 20.0 | --no-climb-limit] [--rescan-factor 4] [--no-stop-at-turnover] \
        [--reservoir SMILES ...] [--reservoir-copies 1] [--free-energy none] [--beam 5] \
        [--build-missing] [--tpt-ends 3] [--pathways 5] [--out FILE]

`predict` (items E, F, I3, I4): what forms from the start pool without a product. The network
grows best-first by probability mass (`network.explore`) on the Markov chain of the explored
graph (reverse edges scored by the model at the children, detailed balance with the free
energies), with the thermodynamic stop at `--temperature-k` (kelvin, not the predictor's
softmax temperature): the end states with their probabilities, the sinks with the steps their
rescan examined, the unexplored mass (the uncertainty), NO_NET_REACTION, and the dominant
pathways to the most probable end states by transition path theory. `--reservoir` names the
species always available (solvent, reagents in excess), kept in every pool. Free energies come
from `--free-energy` (`network.free_energy.load_thermo_model`, the adapter point of item B;
'none' until item B's provider is wired there: the chain then runs on the model's own
probabilities, no sink can be declared, and every state counts as NO_FREE_ENERGY).

`search` and `evaluate` with `--backward-run BWD` search from both sides, the forward run from
the start and the backward run from the whole product pool, meeting on the pool key, the
pathways ranked by the I5 cost (`network.bidirectional`, `network.bidirectional_cli`).

`species` (item J) grows the species graph of the start pool and reports its hyperpaths and atom
flux (`network.species_cli`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.data import provenance as prov
from ksmech.network import bidirectional_cli, retro_cli, species_cli

SEARCH_VERSION = 1
SEARCH_KEYS = ("k", "max_expansions", "max_depth", "step_cost", "levels", "beam", "build_missing")
PREDICT_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    from ksmech.orbitals.build import DEFAULT_TOKENS_DIR
    from ksmech.train.loop import DEFAULT_MODELS_DIR

    p = argparse.ArgumentParser(prog="ksmech-network", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="the k best pathways from a pool to a target product")
    _search_options(s)
    s.add_argument("--start", required=True, help="the starting pool (SMILES, mapped or not)")
    s.add_argument("--target", required=True, help="the target product(s), mapped on the start "
                   "pool's map numbers for the remaining-change heuristic")  # fmt: skip
    s.add_argument("--out", type=Path, help="the result (JSON)")
    _evaluate_parser(sub)
    _predict_parser(sub, DEFAULT_MODELS_DIR, DEFAULT_TOKENS_DIR)
    species_cli.add_parser(sub)  # item J: `ksmech-network species`
    retro_cli.add_parser(sub)  # item K: `ksmech-network retro`
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
    bidirectional_cli.add_options(s)  # item I1: --backward-run searches from both sides


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


def _predict_parser(sub, models_dir: Path, tokens_dir: Path) -> None:
    s = sub.add_parser("predict", help="what forms from a pool, without a product")
    s.add_argument("--run", required=True, help="training run under <models-dir>/<run-source>/")
    s.add_argument("--run-source", default="pmechdb", help="the run's models directory")
    s.add_argument("--checkpoint", default="best", choices=["best", "last"])
    s.add_argument("--start", required=True, help="the starting pool (SMILES, mapped or not)")
    s.add_argument("--temperature-k", type=float, default=298.15, help="physical T (kelvin)")
    s.add_argument("--max-expansions", type=int, default=50, help="nodes expanded (model calls)")
    s.add_argument("--mass-tol", type=float, default=0.01, help="stop at this unexplored mass")
    s.add_argument("--epsilon", type=float, default=0.01, help="population share of the stop")
    s.add_argument("--margin", type=float, default=5.0, help="kcal/mol added to RT ln(1/eps)")
    s.add_argument("--max-climb", type=float, default=20.0,
                   help="kcal/mol above the start a pool may lie, still expanded")  # fmt: skip
    s.add_argument("--no-climb-limit", action="store_true", help="expand pools at any height")
    s.add_argument("--rescan-factor", type=int, default=4, help="the sink rescan's beam factor")
    s.add_argument("--no-stop-at-turnover", action="store_true",
                   help="expand pools that close a catalytic turnover")  # fmt: skip
    s.add_argument("--reservoir", nargs="*", default=[], help="species always available")
    s.add_argument("--reservoir-copies", type=int, default=1, help="copies kept in each pool")
    s.add_argument("--reservoir-molar", nargs="*", type=float, default=[],
                   help="the reservoir species' concentrations (mol/L, default 1)")  # fmt: skip
    s.add_argument("--free-energy", default="none", choices=["none", "qm"],
                   help="none, or qm: the thermochemistry sidecars (ksmech-qm thermo)")  # fmt: skip
    s.add_argument("--beam", type=int, default=5, help="anchors completed and products kept each")
    s.add_argument("--build-missing", action="store_true", help="compute missing tokens (GFN2)")
    s.add_argument("--tpt-ends", type=int, default=3, help="end states given pathways (TPT)")
    s.add_argument("--pathways", type=int, default=5, help="dominant pathways per end state")
    s.add_argument("--top", type=int, default=10, help="end states printed")
    s.add_argument("--models-dir", type=Path, default=models_dir)
    s.add_argument("--tokens-dir", type=Path, default=tokens_dir)
    s.add_argument("--out", type=Path, help="the result (JSON)")


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
    if args.backward_run:
        keys = ("run", "run_source", "checkpoint", "start", "target", *SEARCH_KEYS)
        return bidirectional_cli.cmd_search(args, console, ckpt, expander, cfg, keys)
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
    from ksmech.network.benchmarks import load_cases

    return load_cases(args.set, args.processed_dir, args.split, args.part, args.limit, counts)


def cmd_evaluate(args, console: Console) -> int:
    from collections import Counter

    from ksmech.network.evaluate import evaluate_case, summary
    from ksmech.network.search import SearchParams

    counts: Counter = Counter()
    cases = _cases(args, counts)
    ckpt, expander, cfg = _expander(args)
    keys = ("set", "run", "run_source", "checkpoint", "split", "part", "limit", *SEARCH_KEYS)
    if args.backward_run:
        return bidirectional_cli.cmd_evaluate(args, console, ckpt, expander, cfg, cases, counts,
                                              keys)  # fmt: skip
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
    header = prov.stamp("network_evaluate", SEARCH_VERSION, prov.file_input(ckpt),
                        {k: getattr(args, k) for k in keys})  # fmt: skip
    tag = args.set + (f"_{args.part}" if args.part != "all" else "")
    tag += f"_n{args.limit}" if args.limit is not None else ""
    out = args.out or ckpt.parent / f"network_eval_{tag}.json"
    prov.write_json(out, {"summary": report, "cases": rows}, header)
    console.print(f"wrote {out}")
    return 0


def _print_predict(console: Console, rows: dict, top: int) -> None:
    smiles = {s["id"]: s["smiles"] for s in rows["states"]}
    table = Table(title=f"end states ({rows['reason']}, {rows['expanded']} nodes expanded)")
    for col in ("rank", "reason", "p", "G - G(start)", "T*", "pool"):
        table.add_column(col)
    for n, e in enumerate(rows["ends"][:top]):
        t_star = "" if e["t_star"] is None else f"{e['favoured']} {e['t_star']:.0f} K"
        g = "" if e["g_rel"] is None else f"{e['g_rel']:.2f}"
        table.add_row(str(n + 1), e["reason"], f"{e['probability']:.4f}", g, t_star,
                      smiles[e["state"]])  # fmt: skip
    console.print(table)
    console.print(f"unexplored mass {rows['unexplored_mass']} (frontier {rows['frontier_mass']})"
                  f"; NO_NET_REACTION: {rows['no_net_reaction'] or 'no'}"
                  f"; reactant share {rows['reactant_share']}")  # fmt: skip
    for s in rows["sinks"]:
        console.print(f"sink {smiles[s['state']]}: lowest climb {s['lowest_climb']} > "
                      f"{s['threshold']} kcal/mol, {s['examined']} steps examined "
                      f"(rescan proposed {s['rescan_proposals']})")  # fmt: skip
    for p in rows["pathways"]:
        for path in p["paths"]:
            console.print(f"to {p['end']}: share {path['share']:.3f} via {path['states']}")
    for code, n in rows["counts"].items():
        console.print(f"  {code}: {n}")


def _predict_expander(args):
    from ksmech.arrows.reactants import TokenCache
    from ksmech.model.decode import DecodeConfig
    from ksmech.network.pools import ModelExpander, pool_state
    from ksmech.network.reservoir import ReservoirExpander, reservoir_from_smiles
    from ksmech.network.stop import ModelRescan
    from ksmech.train.loop import load_model

    ckpt = args.models_dir / args.run_source / args.run / f"{args.checkpoint}.pt"
    model, _ = load_model(ckpt, "cpu")
    cfg = DecodeConfig(anchors=args.beam, per_anchor=args.beam)
    inner = ModelExpander(model, TokenCache(args.tokens_dir), cfg, build_missing=args.build_missing)
    expander, start = ModelRescan(inner), pool_state(args.start)
    if args.reservoir:
        reservoir = reservoir_from_smiles(args.reservoir, args.reservoir_copies)
        expander = ReservoirExpander(expander, reservoir)
        start = expander.start(start)
    return ckpt, expander, start


def _thermo_model(args, expander):
    """The free-energy provider of --free-energy, the reservoir species at --reservoir-molar."""
    from ksmech.network.free_energy import load_thermo_model

    if args.free_energy != "qm":
        return load_thermo_model(args.free_energy)
    keys = list(getattr(getattr(expander, "reservoir", None), "smiles", {}))
    if len(args.reservoir_molar) not in (0, len(keys)):
        raise SystemExit(f"--reservoir-molar takes one value per reservoir species ({len(keys)})")
    smiles = getattr(getattr(expander, "reservoir", None), "smiles", {})
    return load_thermo_model("qm", reservoir_smiles=smiles,
                             molar=dict(zip(keys, args.reservoir_molar, strict=False)))  # fmt: skip


def cmd_predict(args, console: Console) -> int:
    from ksmech.network.explore import ExploreParams, explore, pathways_to_ends, predict_rows

    ckpt, expander, start = _predict_expander(args)
    climb = None if args.no_climb_limit else args.max_climb
    params = ExploreParams(temperature_k=args.temperature_k, max_expansions=args.max_expansions,
                           mass_tol=args.mass_tol, epsilon=args.epsilon, margin_kcal=args.margin,
                           max_climb_kcal=climb, rescan_factor=args.rescan_factor,
                           stop_at_turnover=not args.no_stop_at_turnover)  # fmt: skip
    res = explore(expander, start, _thermo_model(args, expander), params)
    rows = predict_rows(res, pathways_to_ends(res, args.tpt_ends, args.pathways))
    rows["threshold_kcal"] = params.stop.threshold
    _print_predict(console, rows, args.top)
    if args.out:
        keys = ("run", "run_source", "checkpoint", "start", "temperature_k", "max_expansions",
                "mass_tol", "epsilon", "margin", "max_climb", "no_climb_limit", "rescan_factor",
                "no_stop_at_turnover",
                "reservoir", "reservoir_copies", "reservoir_molar", "free_energy", "beam",
                "build_missing",
                "tpt_ends", "pathways")  # fmt: skip
        header = prov.stamp("network_predict", PREDICT_VERSION, prov.file_input(ckpt),
                            {k: getattr(args, k) for k in keys})  # fmt: skip
        prov.write_json(args.out, rows, header)
        console.print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "search": cmd_search,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "species": species_cli.cmd_species,
        "retro": retro_cli.cmd_retro,
        "retro-eval": retro_cli.cmd_retro_eval,
    }
    return commands[args.command](args, Console())


if __name__ == "__main__":
    raise SystemExit(main())
