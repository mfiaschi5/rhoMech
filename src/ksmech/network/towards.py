"""Search towards a product (reaction-network plan, item D, with I1 and I2).

`search_towards(expander, start, target, params)` runs `search.k_best_pathways` from the start
pool with the arrival test of `target.arrival` as the goal and, when the target is mapped on the
start pool's map numbers and `step_cost` > 0, `step_cost` x the remaining-change bound of
`target.steps_lower_bound` as the heuristic (exact A*; without it the search is uniform-cost and
still exact). The result lists the pathways found, best first, and their mechanism classes up to
the order of independent steps with p(class | reaches). A target without a usable atom map is
counted as MAPPING_FAILED once per search.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ksmech.network.search import (
    DEFAULT_PARAMS,
    ImplicitGraph,
    MechanismClass,
    SearchParams,
    SearchResult,
    k_best_pathways,
    mechanism_classes,
)
from ksmech.network.target import LEVELS, MAPPING_FAILED, Target, arrival, steps_lower_bound


@dataclass
class TowardsResult:
    search: SearchResult
    classes: list[MechanismClass]
    graph: ImplicitGraph
    skips: Counter = field(default_factory=Counter)


def goal_of(target: Target, levels=LEVELS):
    """state -> the arrival level at `target`, None when not arrived."""
    return lambda state: arrival(target, state.species, state.skeletons, levels)


def heuristic_of(target: Target, params: SearchParams, caps: tuple[int, int], skips: Counter):
    """state -> step_cost x the remaining-change bound; None when the search runs without one
    (step_cost 0, or the target is not mapped on the start pool: MAPPING_FAILED)."""
    if params.step_cost <= 0:
        return None
    if target.orders is None:
        skips[MAPPING_FAILED] += 1
        return None
    max_bonds, max_atoms = caps
    memo: dict[str, float] = {}  # by pool state SMILES: each state bounded (and counted) once

    def h(state) -> float:
        if state.smiles not in memo:
            bound = steps_lower_bound(target, state.orders, state.nonbonding, max_bonds, max_atoms)
            if bound is None:
                skips[f"{MAPPING_FAILED}:STATE"] += 1
            memo[state.smiles] = 0.0 if bound is None else params.step_cost * bound
        return memo[state.smiles]

    return h


def search_towards(
    expander, start, target: Target, params: SearchParams = DEFAULT_PARAMS,
    levels=LEVELS, caps: tuple[int, int] = (8, 2),
) -> TowardsResult:  # fmt: skip
    """The k best pathways from `start` to `target` and their mechanism classes."""
    skips: Counter = Counter()
    graph = ImplicitGraph(expander, start)
    heuristic = heuristic_of(target, params, caps, skips)
    result = k_best_pathways(graph, goal_of(target, levels), heuristic, params)
    skips.update(graph.skips)
    return TowardsResult(result, mechanism_classes(result.pathways), graph, skips)


def pathway_rows(res: TowardsResult) -> list[dict]:
    """The pathways as JSON-able rows: the pool SMILES along the way, the steps' signatures,
    log P, cost, arrival level and the mechanism class index."""
    index = {id(p): k for k, c in enumerate(res.classes) for p in c.pathways}
    return [
        {
            "rank": r + 1,
            "pools": [res.graph.states[n].smiles for n in p.nodes],
            "steps": [e.proposal.signature for e in p.edges],
            "logp": p.logp,
            "cost": p.cost,
            "arrival": p.arrival,
            "mechanism_class": index[id(p)],
        }
        for r, p in enumerate(res.search.pathways)
    ]
