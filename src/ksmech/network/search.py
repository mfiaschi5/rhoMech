"""Search over the implicit network (reaction-network plan, items D, I1 and I2).

The models generate the graph on demand: `Expander.expand(state)` proposes elementary steps from
a pool state with their log P, and costs seconds, so `ImplicitGraph` expands each node at most
once and every algorithm is measured by `ImplicitGraph.expanded`.

`k_best_pathways` is best-first search over simple paths from the root, the priority of a path
being g + h(its last state): g sums -log p + `step_cost` over its steps, h is an admissible
lower bound on the cost still to pay (0 without a heuristic, which is uniform-cost search). With
costs >= 0 and h admissible, paths that end on a goal pop in non-decreasing cost: the first is
the A* optimum (item I1), the first k the k best simple pathways (item I2, the lazy K*-like
enumeration on the implicit graph). A path ends at its first goal state. Every cut is counted:
DEPTH_CUT (the path has `max_depth` steps), CYCLE (a step back onto the path), EXPANSION_BUDGET
(the last state is unexpanded and `max_expansions` nodes are), and the search stops at
`max_pops` popped paths.

Two pathways that differ only in the order of independent steps (two proton transfers on
different atoms) are one mechanism: `mechanism_class` gives the steps' canonical order under
the dependency DAG (a step precedes a later one that touches one of its atoms; spec v9,
"Inference"), and `mechanism_classes` groups the pathways found by it (partial-order
reduction), each class with p(class | reaches) = the summed probability of its pathways over
that of every pathway found. Those probabilities are relative to what the budgets found.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Proposal:
    """One step proposed from a state: the child state, its log P under the predictor, the
    step on map numbers as a canonical text (`signature`) and the map numbers of the atoms it
    changes (`touched`), both for the mechanism classes, and anything else in `info`."""

    child: Any
    logp: float
    signature: str = ""
    touched: frozenset[int] = frozenset()
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Expansion:
    """The steps proposed from one state, the skip counts, and a reason code when the state
    could not be expanded at all (then `proposals` is empty)."""

    proposals: list[Proposal]
    skips: Counter = field(default_factory=Counter)
    failure: str | None = None


class Expander(Protocol):
    def key(self, state: Any) -> str: ...

    def expand(self, state: Any) -> Expansion: ...


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    proposal: Proposal

    @property
    def logp(self) -> float:
        return self.proposal.logp


class ImplicitGraph:
    """The network as far as it is expanded: node key -> the first state reached with that
    key, each node expanded at most once, its edges kept; the expanders' skips summed."""

    def __init__(self, expander: Expander, root: Any) -> None:
        self.expander = expander
        self.root = expander.key(root)
        self.states: dict[str, Any] = {self.root: root}
        self.out: dict[str, list[Edge]] = {}
        self.failures: dict[str, str] = {}
        self.skips: Counter = Counter()

    @property
    def expanded(self) -> int:
        return len(self.out)

    def edges(self, key: str) -> list[Edge]:
        """The edges leaving `key`, expanding it on the first call."""
        if key in self.out:
            return self.out[key]
        ex = self.expander.expand(self.states[key])
        self.skips.update(ex.skips)
        if ex.failure is not None:
            self.failures[key] = ex.failure
            self.skips[f"EXPAND_FAILED:{ex.failure}"] += 1
        edges = []
        for p in ex.proposals:
            child = self.expander.key(p.child)
            self.states.setdefault(child, p.child)
            edges.append(Edge(key, child, p))
        self.out[key] = edges
        return edges


@dataclass(frozen=True)
class SearchParams:
    k: int = 5  # pathways to report
    max_expansions: int = 200  # nodes expanded (each one a model call)
    max_depth: int = 8  # steps of a pathway
    max_pops: int = 100_000  # partial paths popped
    step_cost: float = 0.0  # added to -log p per step (> 0: a remaining-change heuristic bites)


DEFAULT_PARAMS = SearchParams()


@dataclass(frozen=True)
class Pathway:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    cost: float  # sum of -log p + step_cost
    logp: float  # sum of log p
    arrival: str  # the goal level reached


@dataclass
class SearchResult:
    pathways: list[Pathway]
    expanded: int
    pops: int
    reason: str  # K_FOUND | EXHAUSTED | POP_BUDGET | START_IS_GOAL
    cuts: Counter = field(default_factory=Counter)


Goal = Callable[[Any], str | None]  # state -> the arrival level, None when not arrived
Heuristic = Callable[[Any], float]  # state -> admissible lower bound on the cost to a goal


def _extend(graph, item, params, h, heap, cuts, counter) -> None:
    """Push every simple one-step extension of a popped path (g, nodes, edges)."""
    g, nodes, edges = item
    if len(edges) >= params.max_depth:
        cuts["DEPTH_CUT"] += 1
        return
    if nodes[-1] not in graph.out and graph.expanded >= params.max_expansions:
        cuts["EXPANSION_BUDGET"] += 1
        return
    for e in graph.edges(nodes[-1]):
        if e.target in nodes:
            cuts["CYCLE"] += 1
            continue
        g2 = g - e.logp + params.step_cost
        f2 = g2 + h(graph.states[e.target])
        heapq.heappush(
            heap, (f2, len(edges) + 1, next(counter), g2, nodes + (e.target,), edges + (e,))
        )


def k_best_pathways(
    graph: ImplicitGraph, goal: Goal, heuristic: Heuristic | None = None,
    params: SearchParams = DEFAULT_PARAMS,
) -> SearchResult:  # fmt: skip
    """The k best simple pathways from the root to a goal state (module docstring)."""
    h = heuristic or (lambda state: 0.0)
    if goal(graph.states[graph.root]) is not None:
        return SearchResult([], graph.expanded, 0, "START_IS_GOAL")
    counter = iter(range(1 << 62))
    heap = [(h(graph.states[graph.root]), 0, next(counter), 0.0, (graph.root,), ())]
    found: list[Pathway] = []
    cuts: Counter = Counter()
    pops, reason = 0, "EXHAUSTED"
    while heap:
        if pops >= params.max_pops:
            reason = "POP_BUDGET"
            break
        _, _, _, g, nodes, edges = heapq.heappop(heap)
        pops += 1
        level = goal(graph.states[nodes[-1]]) if edges else None
        if level is not None:
            found.append(Pathway(nodes, edges, g, sum(e.logp for e in edges), level))
            if len(found) >= params.k:
                reason = "K_FOUND"
                break
            continue
        _extend(graph, (g, nodes, edges), params, h, heap, cuts, counter)
    return SearchResult(found, graph.expanded, pops, reason, cuts)


def mechanism_class(edges: Sequence[Edge]) -> tuple[str, ...]:
    """The steps' signatures in their canonical order up to the order of independent steps:
    step i precedes a later step j when they touch a common atom, and among the steps whose
    predecessors are placed the smallest signature goes first (Kahn's algorithm)."""
    n = len(edges)
    touched = [e.proposal.touched for e in edges]
    before = [{i for i in range(j) if touched[i] & touched[j]} for j in range(n)]
    placed: list[int] = []
    left = set(range(n))
    while left:
        ready = [j for j in left if before[j] <= set(placed)]
        j = min(ready, key=lambda x: (edges[x].proposal.signature, x))
        placed.append(j)
        left.remove(j)
    return tuple(edges[j].proposal.signature for j in placed)


@dataclass(frozen=True)
class MechanismClass:
    steps: tuple[str, ...]  # the canonical order (`mechanism_class`)
    pathways: tuple[Pathway, ...]  # the pathways found in the class, best first
    logp: float  # log of the summed probability of its pathways
    probability: float  # p(class | reaches), over every pathway found


def _logsumexp(xs: Sequence[float]) -> float:
    top = max(xs, default=float("-inf"))
    if not math.isfinite(top):
        return top
    return top + math.log(sum(math.exp(x - top) for x in xs))


def mechanism_classes(pathways: Sequence[Pathway]) -> list[MechanismClass]:
    """The pathways grouped by mechanism class, classes by decreasing probability, then by
    their best pathway's cost."""
    groups: dict[tuple[str, ...], list[Pathway]] = {}
    for p in pathways:
        groups.setdefault(mechanism_class(p.edges), []).append(p)
    total = _logsumexp([p.logp for p in pathways])
    out = []
    for steps, members in groups.items():
        members = sorted(members, key=lambda p: p.cost)
        logp = _logsumexp([p.logp for p in members])
        prob = math.exp(logp - total) if math.isfinite(total) else 0.0
        out.append(MechanismClass(steps, tuple(members), logp, prob))
    return sorted(out, key=lambda c: (-c.probability, c.pathways[0].cost, c.steps))
