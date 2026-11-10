"""The search towards a product on pathway benchmarks (reaction-network plan, section 5 "with a
product"): per case the arrival (EXACT, SKELETON or none) within the node budget, the rank of
the recorded pathway among the pathways found, how many recorded step targets the best pathway
passes through, the nodes expanded and the cuts; over the cases the arrival rates, the share of
recorded pathways found at rank 1 and within k, and the mean expansions.

The recorded pathway is matched by its sequence of pool keys (`Case.recorded`), so a found
pathway that takes the same steps in another order of independent steps does not count as it:
the rank is a lower bound on the class-level match the plan asks for, which needs each recorded
step's matrix on the start pool's numbering. A step target counts as passed when its species
(identity keys) are all in some pool of the best pathway.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ksmech.network.benchmarks import Case
from ksmech.network.pools import pool_state
from ksmech.network.search import DEFAULT_PARAMS, SearchParams
from ksmech.network.target import EXACT, LEVELS, SKELETON, arrival, target_from_smiles
from ksmech.network.towards import search_towards


def recorded_rank(res, recorded: tuple[str, ...] | None) -> int | None:
    """The 1-based rank of the first pathway found whose pool keys are the recorded ones."""
    if recorded is None:
        return None
    for rank, p in enumerate(res.search.pathways, start=1):
        if tuple(p.nodes) == tuple(recorded):
            return rank
    return None


def targets_passed(res, step_targets, skips: Counter) -> int | None:
    """How many recorded step targets the best pathway passes through (None: no target, or no
    pathway); a target RDKit cannot read is counted as STEP_TARGET_UNPARSED."""
    wanted = [t for t in step_targets if t]
    if not wanted or not res.search.pathways:
        return None
    pools = [res.graph.states[n] for n in res.search.pathways[0].nodes]
    passed = 0
    for smiles in wanted:
        try:
            target = target_from_smiles(smiles)
        except Exception:  # noqa: BLE001 - counted
            skips["STEP_TARGET_UNPARSED"] += 1
            continue
        passed += any(arrival(target, s.species, s.skeletons, (EXACT,)) for s in pools)
    return passed


def evaluate_case(expander, case: Case, params: SearchParams = DEFAULT_PARAMS, levels=LEVELS,
                  caps=(8, 2)) -> dict:  # fmt: skip
    """One case's row (module docstring)."""
    skips: Counter = Counter()
    try:
        start, target = pool_state(case.start), target_from_smiles(case.target)
    except Exception as exc:  # noqa: BLE001 - the case is reported, not dropped
        return {
            "case": case.case_id,
            "part": case.part,
            "failure": f"UNPARSED:{type(exc).__name__}",
        }
    res = search_towards(expander, start, target, params, levels, caps)
    best = res.search.pathways[0] if res.search.pathways else None
    skips.update(res.skips)
    row = {
        "case": case.case_id,
        "part": case.part,
        "reason": res.search.reason,
        "expanded": res.search.expanded,
        "arrival": best.arrival if best else None,
        "n_pathways": len(res.search.pathways),
        "best_logp": best.logp if best else None,
        "best_steps": len(best.edges) if best else None,
        "recorded_steps": len(case.recorded) - 1 if case.recorded else None,
        "recorded_rank": recorded_rank(res, case.recorded),
        "step_targets": len([t for t in case.step_targets if t]),
        "step_targets_passed": targets_passed(res, case.step_targets, skips),
        "cuts": dict(res.search.cuts),
    }
    return row | {"skips": dict(skips)}


def summary(rows: list[dict], k: int) -> dict:
    """Arrival rates, recorded-pathway ranks and expansions over the evaluated cases."""
    done = [r for r in rows if "failure" not in r]
    n = len(done)
    rate = lambda f: sum(map(f, done)) / n if n else 0.0  # noqa: E731
    with_rec = [r for r in done if r["recorded_steps"] is not None]
    ranks = [r["recorded_rank"] for r in with_rec]
    skips: Counter = Counter()
    for r in done:
        skips.update(r["skips"])
        skips.update(r["cuts"])
    return {
        "n": n,
        "failed": len(rows) - n,
        "arrival_exact": rate(lambda r: r["arrival"] == EXACT),
        "arrival_skeleton": rate(lambda r: r["arrival"] == SKELETON),
        "arrival_any": rate(lambda r: r["arrival"] is not None),
        "n_with_recorded": len(with_rec),
        "recorded_top1": sum(x == 1 for x in ranks) / max(len(ranks), 1),
        f"recorded_top{k}": sum(x is not None and x <= k for x in ranks) / max(len(ranks), 1),
        "mean_expanded": float(np.mean([r["expanded"] for r in done])) if done else 0.0,
        "skips": dict(sorted(skips.items())),
    }


def retro_case(backward, case: Case, byproducts, params, n_byproducts: int = 3,
               forward=None, forward_params=None) -> dict:  # fmt: skip
    """One case of the search from the product alone (item K, plan section 5): the case's
    final product the target, its recorded reactants the catalogue (decided 2026-11-10) and the
    counted by-products the reservoir. Per case: solved, the route's cost and leaves, whether
    its leaves outside the reservoir are exactly the recorded reactants (`recovered`), the
    forward verification with `forward` when given, the species expanded."""
    from ksmech.network.retro import (
        Leaves,
        ModelRetroExpander,
        RetroParams,
        leaves_of,
        retro_search,
        verify_forward,
    )  # fmt: skip

    try:
        start = pool_state(case.start)
        expander = ModelRetroExpander(backward, case.target, byproducts, n_byproducts)
    except Exception as exc:  # noqa: BLE001 - the case is reported, not dropped
        return {
            "case": case.case_id,
            "part": case.part,
            "failure": f"UNPARSED:{type(exc).__name__}",
        }
    reservoir = frozenset(k for k, _, _ in byproducts)
    leaves = Leaves(catalogue=frozenset(start.species), reservoir=reservoir)
    res = retro_search(expander, expander.target, leaves, params or RetroParams())
    found = leaves_of(res.route)
    recorded = {k for k in start.species if k not in reservoir}
    recovered = res.route is not None and {k for k in found if k not in reservoir} == recorded
    row = {"case": case.case_id, "part": case.part, "reason": res.reason,
           "expanded": res.expanded, "solved": res.route is not None,
           "cost": None if res.route is None else res.route.cost,
           "leaves": {expander.smiles.get(k, k): n for k, n in found.items()},
           "recovered": recovered,
           "verified": None, "cuts": dict(res.cuts), "skips": dict(expander.skips)}  # fmt: skip
    if forward is not None and res.route is not None:
        row["verified"], _ = verify_forward(forward, res.route, case.target, expander.smiles,
                                            forward_params)  # fmt: skip
    return row


def retro_summary(rows: list[dict]) -> dict:
    """Solved, recovered and verified shares and the mean species expanded."""
    done = [r for r in rows if "failure" not in r]
    n = len(done)
    share = lambda f: sum(map(f, done)) / n if n else 0.0  # noqa: E731
    checked = [r for r in done if r["verified"] is not None]
    return {
        "n": n,
        "failed": len(rows) - n,
        "solved": share(lambda r: r["solved"]),
        "recovered": share(lambda r: r["recovered"]),
        "n_verified_checked": len(checked),
        "verified": sum(bool(r["verified"]) for r in checked) / max(len(checked), 1),
        "mean_expanded": float(np.mean([r["expanded"] for r in done])) if done else 0.0,
    }
