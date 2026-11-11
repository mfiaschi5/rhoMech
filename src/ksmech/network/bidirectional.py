"""Bidirectional search between a start pool and a full product pool (reaction-network plan,
item I1, with the rescoring of I5).

The forward model grows a best-first tree from the start pool, the backward model (item H,
`ksmech-train fit --direction backward`) one from the target pool, each by its own cost
(-log p of its proposals + `step_cost`, `network.search.ImplicitGraph` on each side); the trees
meet on the pool key (identity keys v2, `network.keys`). The side whose frontier is cheaper is
expanded next, and the search stops when the two frontiers' cheapest costs sum to at least the
k-th best usable meeting (one whose two tree paths join a simple path within `max_depth`; the
bidirectional Dijkstra rule, exact for one meeting when both sides share one cost; here each
side has its model, so it is the stopping rule and the rescoring ranks), at
`max_expansions` nodes over both sides (EXPANSION_BUDGET), or at `max_pops`. A node is expanded
at most once per side, a side's paths keep at most `max_depth` steps, and a meeting whose two
halves sum past `max_depth` is cut (DEPTH_CUT); a joined path through a node twice is CYCLE.
Each meeting node gives one pathway, the best tree path on each side; a path met at several of
its nodes (the two sides split it at different places) counts once (DUPLICATE_PATH).

The target must be the whole product pool, by-products included, or the atoms do not balance
and the trees cannot meet (`full_pool`: the same element counts, hydrogens included, and the
same total charge; else TARGET_NOT_FULL_POOL). The two halves number the atoms differently (the
start's numbers forward, the target's backward), so every step keeps its parent in its own
half's numbering and R on those numbers.

**Rescoring (I5).** A pathway found from two sides is ranked by one criterion: the sum over its
steps of -(log P_f(R | parent) + lambda log P_b(-R | child)) + `step_cost`, the forward term the
decoder's (`MatrixScore.best`, the score its proposals carry), the backward term the marginal
over -R's anchors, as `eval.roundtrip` reranks the forward beam; each term the proposing side
does not carry is scored by the other model (`step_logp`). A step one model cannot score (its
tokens are missing, say) leaves the pathway unrescored, counted, ranked after the rescored ones.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, rdBase

from ksmech.network.search import DEFAULT_PARAMS, Edge, ImplicitGraph, SearchParams

INF = math.inf
TARGET_NOT_FULL_POOL = "TARGET_NOT_FULL_POOL"
FORWARD, BACKWARD = "forward", "backward"


@dataclass(frozen=True)
class JoinedStep:
    """One step of a joined pathway in the forward direction: the pools it leaves and reaches
    (in the numbering of the half that proposed it), R on the parent's map numbers, the side
    that proposed it and its score there."""

    parent: Any
    child: Any
    bonds: tuple
    lone: tuple
    side: str
    logp: float  # the proposing model's log P (forward: of R at the parent; backward: of -R)


@dataclass(frozen=True)
class JoinedPathway:
    nodes: tuple[str, ...]  # pool keys, start first, target last
    steps: tuple[JoinedStep, ...]
    meeting: str  # the pool key where the trees met
    cost: float  # the search cost: forward half by P_f, backward half by P_b
    rescored: float | None = None  # the I5 cost; None when a step could not be scored
    logp_forward: float | None = None  # sum of log P_f over the steps (rescored)
    logp_backward: float | None = None  # sum of log P_b over the steps (rescored)


@dataclass
class BidirectionalResult:
    pathways: list[JoinedPathway]  # rescored first, by rescored cost; then the others by cost
    expanded: dict[str, int]  # nodes expanded on each side
    pops: int
    reason: str  # K_FOUND | EXHAUSTED | POP_BUDGET | START_IS_GOAL | TARGET_NOT_FULL_POOL
    meetings: int
    cuts: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)


@dataclass
class _Side:
    graph: ImplicitGraph
    dist: dict[str, float]
    steps: dict[str, int]
    parent: dict[str, Edge]
    heap: list
    settled: set

    @classmethod
    def at(cls, expander, root) -> _Side:
        graph = ImplicitGraph(expander, root)
        return cls(graph, {graph.root: 0.0}, {graph.root: 0}, {}, [(0.0, 0, graph.root)], set())

    def top(self) -> float:
        return self.heap[0][0] if self.heap else INF


def _atoms(state) -> tuple[Counter, int]:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(state.smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    counts = Counter(a.GetAtomicNum() for a in mol.GetAtoms())
    counts[1] += sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    return counts, sum(a.GetFormalCharge() for a in mol.GetAtoms())


def full_pool(start, target) -> bool:
    """Whether the target holds the start's atoms (element counts with H) and total charge."""
    return _atoms(start) == _atoms(target)


def _relax(side: _Side, other: _Side, key: str, ctx: dict) -> None:
    """Expand `key` on `side` and push every cheaper way to its children; record a meeting
    whenever a child is reached by the other side too."""
    params, cuts, meetings = ctx["params"], ctx["cuts"], ctx["meetings"]
    if side.steps[key] >= params.max_depth:
        cuts["DEPTH_CUT"] += 1
        return
    if key not in side.graph.out and ctx["expanded"]() >= params.max_expansions:
        cuts["EXPANSION_BUDGET"] += 1
        return
    for e in side.graph.edges(key):
        d2 = side.dist[key] - e.logp + params.step_cost
        if d2 >= side.dist.get(e.target, INF):
            continue
        side.dist[e.target], side.parent[e.target] = d2, e
        side.steps[e.target] = side.steps[key] + 1
        heapq.heappush(side.heap, (d2, next(ctx["counter"]), e.target))
        if e.target in other.dist:
            meetings[e.target] = min(meetings.get(e.target, INF), d2 + other.dist[e.target])


def _path_keys(f: _Side, b: _Side, m: str) -> list[str]:
    """The pool keys from the start through `m` to the target on the two trees as they are."""
    head, key = [m], m
    while key != f.graph.root:
        key = f.parent[key].source
        head.append(key)
    tail, key = [], m
    while key != b.graph.root:
        key = b.parent[key].source
        tail.append(key)
    return head[::-1] + tail


def usable(f: _Side, b: _Side, meetings: dict[str, float], params: SearchParams,
           cuts: Counter | None = None) -> list[tuple[float, str]]:  # fmt: skip
    """(cost, meeting) of each distinct simple path within `max_depth` the meetings join, best
    first: one path met at several of its nodes (the split between the sides moved) counts once,
    at its cheapest meeting (else DUPLICATE_PATH); the others are DEPTH_CUT or CYCLE."""
    cuts = Counter() if cuts is None else cuts
    seen: set[tuple[str, ...]] = set()
    out = []
    for m in sorted(meetings, key=lambda x: (meetings[x], x)):
        if f.steps[m] + b.steps[m] > params.max_depth:
            cuts["DEPTH_CUT"] += 1
            continue
        keys = tuple(_path_keys(f, b, m))
        cut = "CYCLE" if len(set(keys)) < len(keys) else "DUPLICATE_PATH" if keys in seen else None
        if cut is not None:
            cuts[cut] += 1
            continue
        seen.add(keys)
        out.append((meetings[m], m))
    return out


def _kth(f: _Side, b: _Side, meetings: dict[str, float], params: SearchParams) -> float:
    """The k-th best cost among the distinct usable paths (INF when fewer)."""
    if len(meetings) < params.k:
        return INF
    costs = usable(f, b, meetings, params)
    return costs[params.k - 1][0] if len(costs) >= params.k else INF


def meet(forward, backward, start, target, params: SearchParams = DEFAULT_PARAMS):
    """(the forward side, the backward side, {meeting key: cost}, pops, reason, cuts)."""
    f, b = _Side.at(forward, start), _Side.at(backward, target)
    meetings: dict[str, float] = {}
    if f.graph.root == b.graph.root:
        return f, b, meetings, 0, "START_IS_GOAL", Counter()
    ctx = {"params": params, "cuts": Counter(), "meetings": meetings,
           "counter": iter(range(1, 1 << 62)),
           "expanded": lambda: f.graph.expanded + b.graph.expanded}  # fmt: skip
    pops, reason = 0, "EXHAUSTED"
    while f.heap or b.heap:
        if pops >= params.max_pops:
            reason = "POP_BUDGET"
            break
        kth = _kth(f, b, meetings, params)
        if math.isfinite(kth) and f.top() + b.top() >= kth:
            reason = "K_FOUND"
            break
        side, other = (f, b) if f.top() <= b.top() else (b, f)
        d, _, key = heapq.heappop(side.heap)
        pops += 1
        if key in side.settled or d > side.dist[key]:
            continue
        side.settled.add(key)
        if key in other.dist:
            meetings[key] = min(meetings.get(key, INF), d + other.dist[key])
        _relax(side, other, key, ctx)
    return f, b, meetings, pops, reason, ctx["cuts"]


def _negated(bonds, lone) -> tuple[tuple, tuple]:
    return tuple((m, n, -d) for m, n, d in bonds), tuple((m, -d) for m, d in lone)


def _halves(f: _Side, b: _Side, m: str) -> tuple[list[str], list[JoinedStep]]:
    """The keys and forward-direction steps from the start through `m` to the target."""
    nodes, steps, key = [m], [], m
    while key != f.graph.root:
        e = f.parent[key]
        info = e.proposal.info
        steps.append(JoinedStep(f.graph.states[e.source], e.proposal.child, tuple(info["bonds"]),
                                tuple(info["lone"]), FORWARD, e.logp))  # fmt: skip
        nodes.append(e.source)
        key = e.source
    nodes.reverse()
    steps.reverse()
    key = m
    while key != b.graph.root:
        # e: source (nearer the target) -> key, a step back from the source state that was
        # expanded; its child is drawn on that state's numbers, which the graph's first state
        # with this key need not share
        e = b.parent[key]
        bonds, lone = _negated(e.proposal.info["bonds"], e.proposal.info["lone"])
        steps.append(JoinedStep(e.proposal.child, b.graph.states[e.source], bonds, lone,
                                BACKWARD, e.logp))  # fmt: skip
        nodes.append(e.source)
        key = e.source
    return nodes, steps


def join(f: _Side, b: _Side, meetings: dict[str, float], params: SearchParams,
         cuts: Counter) -> list[JoinedPathway]:  # fmt: skip
    """The k best distinct usable pathways (`usable`), best first."""
    out = []
    for cost, m in usable(f, b, meetings, params, cuts)[: params.k]:
        nodes, steps = _halves(f, b, m)
        out.append(JoinedPathway(tuple(nodes), tuple(steps), m, cost))
    return out


Scorer = Callable[[Any, Any, Any], Any]  # (state, bonds, lone) -> MatrixScore | failure code


def _terms(step: JoinedStep, forward: Scorer, backward: Scorer) -> tuple[Any, Any]:
    """(log P_f of R at the parent, log P_b of -R at the child): the proposing side's own
    score where it is that quantity, the other model's `step_logp` otherwise."""
    rb, rl = _negated(step.bonds, step.lone)
    if step.side == FORWARD:
        score = backward(step.child, rb, rl)
        return step.logp, score if isinstance(score, str) else score.marginal
    lp_b = backward(step.child, rb, rl)  # the proposal carries the best anchor, not the marginal
    lp_f = forward(step.parent, step.bonds, step.lone)
    return (lp_f if isinstance(lp_f, str) else lp_f.best), (
        lp_b if isinstance(lp_b, str) else lp_b.marginal
    )


def rescore(p: JoinedPathway, forward: Scorer, backward: Scorer, lam: float, step_cost: float,
            skips: Counter) -> JoinedPathway:  # fmt: skip
    """The pathway with its I5 cost (module docstring), or unrescored with the reason counted."""
    lf = lb = 0.0
    for step in p.steps:
        a, c = _terms(step, forward, backward)
        for code in (x for x in (a, c) if isinstance(x, str)):
            skips[f"RESCORE_FAILED:{code}"] += 1
            return p
        lf, lb = lf + a, lb + c
    cost = -(lf + lam * lb) + step_cost * len(p.steps) if lam else -lf + step_cost * len(p.steps)
    return JoinedPathway(p.nodes, p.steps, p.meeting, p.cost, cost, lf, lb)


def bidirectional_search(forward, backward, start, target, params: SearchParams = DEFAULT_PARAMS,
                         lam: float = 1.0) -> BidirectionalResult:  # fmt: skip
    """The k best pathways from `start` to the full pool `target` found from both sides,
    ranked by the I5 cost (module docstring). `forward` and `backward` are expanders with
    `step_logp` (`network.pools.ModelExpander` with each checkpoint)."""
    if not full_pool(start, target):
        return BidirectionalResult([], {FORWARD: 0, BACKWARD: 0}, 0, TARGET_NOT_FULL_POOL, 0)
    f, b, meetings, pops, reason, cuts = meet(forward, backward, start, target, params)
    skips = f.graph.skips + b.graph.skips
    found = join(f, b, meetings, params, cuts)
    found = [rescore(p, forward.step_logp, backward.step_logp, lam, params.step_cost, skips)
             for p in found]  # fmt: skip
    found.sort(key=lambda p: (p.rescored is None, p.rescored if p.rescored is not None else p.cost))
    expanded = {FORWARD: f.graph.expanded, BACKWARD: b.graph.expanded}
    return BidirectionalResult(found, expanded, pops, reason, len(meetings), cuts, skips)
