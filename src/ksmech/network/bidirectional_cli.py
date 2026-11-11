"""The bidirectional search from the command line (reaction-network plan, item I1, with I5):
`ksmech-network search` and `ksmech-network evaluate` with `--backward-run BWD`.

    ksmech-network search --run orb_a_s0 --backward-run bwd_s0 \
        --start 'C[O-].CC(C)=O' --target 'CC(C)([O-])OC' [--roundtrip-lambda 1]

The forward run expands pools from the start, the backward run (`ksmech-train fit --direction
backward`) from the target, which must be the whole product pool (TARGET_NOT_FULL_POOL
otherwise); `--max-expansions` counts both sides; the pathways are ranked by the I5 cost with
`--roundtrip-lambda` (`network.bidirectional`). `--levels` and `--step-cost`'s heuristic do not
apply: the trees meet on the exact pool key, and no remaining-change bound is used. `evaluate`
writes `network_eval_<set>[_<part>][_n<limit>]_bidir_<backward run>.json`, its rows in the keys
of the forward search's, so the two are compared case by case on the same budgets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ksmech.data import provenance as prov

BIDIR_KEYS = ("backward_run", "backward_checkpoint", "roundtrip_lambda")


def add_options(s) -> None:
    s.add_argument("--backward-run", help="a backward run: search from both sides (item I1)")
    s.add_argument("--backward-checkpoint", default="best", choices=["best", "last"])
    s.add_argument("--roundtrip-lambda", type=float, default=1.0,
                   help="lambda of log P_b(-R | child) in the pathways' cost (I5)")  # fmt: skip


def _direction(payload: dict) -> str:
    train = payload.get("provenance", {}).get("config", {}).get("train", {})
    return train.get("direction", "forward")


def backward_expander(args, ckpt: Path, cfg):
    """(the backward checkpoint, its expander); ValueError when either run's direction is
    not the one its side needs."""
    from ksmech.arrows.reactants import TokenCache
    from ksmech.network.pools import ModelExpander
    from ksmech.train.loop import load_model

    if _direction(load_model(ckpt, "cpu")[1]) != "forward":
        raise ValueError(f"{ckpt}: not a forward run")
    bckpt = args.models_dir / args.run_source / args.backward_run / f"{args.backward_checkpoint}.pt"
    model, payload = load_model(bckpt, "cpu")
    if _direction(payload) != "backward":
        raise ValueError(f"{bckpt}: not a backward run (ksmech-train fit --direction backward)")
    loader = TokenCache(args.tokens_dir)
    return bckpt, ModelExpander(model, loader, cfg, build_missing=args.build_missing)


def pathway_rows(res) -> list[dict]:
    return [
        {
            "rank": r + 1,
            "pools": [p.steps[0].parent.smiles, *(s.child.smiles for s in p.steps)],
            "sides": [s.side for s in p.steps],
            "meeting": p.meeting,
            "search_cost": p.cost,
            "rescored": p.rescored,
            "logp_forward": p.logp_forward,
            "logp_backward": p.logp_backward,
        }
        for r, p in enumerate(res.pathways)
    ]


def _print(console: Console, res, rows: list[dict]) -> None:
    sides = f"{res.expanded['forward']} forward + {res.expanded['backward']} backward"
    title = f"pathways from both sides ({res.reason}, {sides} expanded, {res.meetings} meetings)"
    table = Table(title=title)
    for col in ("rank", "steps", "sides", "I5 cost", "log P_f", "log P_b", "pools"):
        table.add_column(col)
    for r in rows:
        f = lambda x: "" if x is None else f"{x:.2f}"  # noqa: E731
        table.add_row(str(r["rank"]), str(len(r["sides"])), "".join(s[0] for s in r["sides"]),
                      f(r["rescored"]), f(r["logp_forward"]), f(r["logp_backward"]),
                      " >> ".join(r["pools"][1:]))  # fmt: skip
    console.print(table)
    for code, n in sorted({**res.cuts, **res.skips}.items()):
        console.print(f"  {code}: {n}")


def cmd_search(args, console: Console, ckpt: Path, expander, cfg, keys) -> int:
    from ksmech.network.bidirectional import bidirectional_search
    from ksmech.network.pools import pool_state
    from ksmech.network.search import SearchParams

    bckpt, backward = backward_expander(args, ckpt, cfg)
    params = SearchParams(args.k, args.max_expansions, args.max_depth, step_cost=args.step_cost)
    res = bidirectional_search(expander, backward, pool_state(args.start),
                               pool_state(args.target), params, args.roundtrip_lambda)  # fmt: skip
    rows = pathway_rows(res)
    _print(console, res, rows)
    if args.out:
        header = prov.stamp("network_search_bidirectional", 1,
                            prov.file_input(ckpt) | prov.file_input(bckpt),
                            {k: getattr(args, k) for k in (*keys, *BIDIR_KEYS)})  # fmt: skip
        body = {"reason": res.reason, "expanded": res.expanded, "meetings": res.meetings,
                "pathways": rows, "cuts": dict(res.cuts), "skips": dict(res.skips)}  # fmt: skip
        prov.write_json(args.out, body, header)
        console.print(f"wrote {args.out}")
    return 0


def cmd_evaluate(args, console: Console, ckpt: Path, expander, cfg, cases, counts: Counter,
                 keys) -> int:  # fmt: skip
    from ksmech.network.evaluate import evaluate_case_bidirectional, summary
    from ksmech.network.search import SearchParams

    bckpt, backward = backward_expander(args, ckpt, cfg)
    params = SearchParams(args.k, args.max_expansions, args.max_depth, step_cost=args.step_cost)
    rows = []
    for i, case in enumerate(cases, start=1):
        rows.append(evaluate_case_bidirectional(expander, backward, case, params,
                                                args.roundtrip_lambda))  # fmt: skip
        r = rows[-1]
        console.print(f"case {i} / {len(cases)} {case.case_id}: "
                      f"{r.get('failure') or r.get('arrival')} rank {r.get('recorded_rank')}, "
                      f"{r.get('expanded')} expanded")  # fmt: skip
    report = summary(rows, args.k) | {"cases_skipped": dict(counts)}
    console.print(report)
    header = prov.stamp("network_evaluate_bidirectional", 1,
                        prov.file_input(ckpt) | prov.file_input(bckpt),
                        {k: getattr(args, k) for k in (*keys, *BIDIR_KEYS)})  # fmt: skip
    tag = args.set + (f"_{args.part}" if args.part != "all" else "")
    tag += f"_n{args.limit}" if args.limit is not None else ""
    out = args.out or ckpt.parent / f"network_eval_{tag}_bidir_{args.backward_run}.json"
    prov.write_json(out, {"summary": report, "cases": rows}, header)
    console.print(f"wrote {out}")
    return 0
