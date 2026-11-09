"""`network.search` and `network.graph` on small synthetic graphs (reaction-network plan, items
C, D, I1, I2): the k best pathways equal exhaustive enumeration, an admissible heuristic changes
the expansions but not the result, every cut is counted, and the mechanism classes are invariant
under swapping independent steps."""

import heapq
import itertools
import math
import random
from collections import Counter

import pytest

from ksmech.network.graph import PathOps, best_simple_paths, turnover_closure
from ksmech.network.search import (
    Expansion,
    ImplicitGraph,
    Proposal,
    SearchParams,
    k_best_pathways,
    mechanism_class,
    mechanism_classes,
)


class GraphExpander:
    """An explicit digraph as an expander: node -> [(child, log p, touched atoms)]."""

    def __init__(self, edges):
        self.edges, self.calls = edges, 0

    def key(self, state):
        return state

    def expand(self, state):
        self.calls += 1
        out = [
            Proposal(t, lp, f"{state}>{t}", frozenset(touched))
            for t, lp, touched in self.edges.get(state, [])
        ]
        return Expansion(out, Counter({"SEEN": 1}), "DEAD_END" if state == "dead" else None)


def _random_graph(seed: int, n: int = 9, p: float = 0.35):
    rng = random.Random(seed)
    names = [f"n{i}" for i in range(n)]
    edges = {a: [] for a in names}
    for a, b in itertools.permutations(names, 2):
        if rng.random() < p:
            edges[a].append((b, -rng.uniform(0.05, 3.0), ()))
    return edges


def _all_simple_costs(edges, root, goals, max_depth):
    """Every simple path's cost (-sum log p) from root to its first goal, sorted."""
    out = []

    def walk(node, seen, cost, depth):
        if node in goals and depth:
            out.append(cost)
            return
        if depth == max_depth:
            return
        for t, lp, _ in edges.get(node, []):
            if t not in seen:
                walk(t, seen | {t}, cost - lp, depth + 1)

    walk(root, {root}, 0.0, 0)
    return sorted(out)


def _exact_remaining(edges, goals):
    """Dijkstra on the reversed graph: the cost to the nearest goal (a perfect heuristic)."""
    rev = {}
    for a, outs in edges.items():
        for b, lp, _ in outs:
            rev.setdefault(b, []).append((a, -lp))
    dist = {g: 0.0 for g in goals}
    heap = [(0.0, g) for g in goals]
    while heap:
        d, x = heapq.heappop(heap)
        if d > dist.get(x, math.inf):
            continue
        for y, c in rev.get(x, []):
            if d + c < dist.get(y, math.inf):
                dist[y] = d + c
                heapq.heappush(heap, (d + c, y))
    return dist


@pytest.mark.parametrize("seed", range(12))
def test_the_k_best_pathways_equal_exhaustive_enumeration(seed):
    edges = _random_graph(seed)
    goals = {"n7", "n8"}
    want = _all_simple_costs(edges, "n0", goals, 6)[:4]
    params = SearchParams(k=4, max_depth=6, max_expansions=10_000)
    got = k_best_pathways(ImplicitGraph(GraphExpander(edges), "n0"), lambda s: s in goals or None,
                          params=params)  # fmt: skip
    assert [p.cost for p in got.pathways] == pytest.approx(want)
    assert all(p.logp == pytest.approx(-p.cost) for p in got.pathways)


@pytest.mark.parametrize("seed", range(12))
def test_an_admissible_heuristic_keeps_the_result_and_expands_no_more(seed):
    edges = _random_graph(seed)
    goals = {"n7", "n8"}
    goal = lambda s: "EXACT" if s in goals else None  # noqa: E731
    params = SearchParams(k=1, max_depth=6, max_expansions=10_000)
    plain, guided = GraphExpander(edges), GraphExpander(edges)
    base = k_best_pathways(ImplicitGraph(plain, "n0"), goal, params=params)
    dist = _exact_remaining(edges, goals)
    astar = k_best_pathways(ImplicitGraph(guided, "n0"), goal, lambda s: dist.get(s, 0.0), params)
    assert [p.cost for p in astar.pathways] == pytest.approx([p.cost for p in base.pathways])
    assert guided.calls <= plain.calls


def test_a_step_cost_adds_to_every_step():
    edges = {"a": [("b", -0.1, ()), ("g", -1.0, ())], "b": [("g", -0.1, ())]}
    goal = lambda s: s == "g" or None  # noqa: E731
    cheap = k_best_pathways(
        ImplicitGraph(GraphExpander(edges), "a"), goal, params=SearchParams(k=1)
    )
    assert cheap.pathways[0].nodes == ("a", "b", "g")
    costly = SearchParams(k=1, step_cost=1.0)
    short = k_best_pathways(ImplicitGraph(GraphExpander(edges), "a"), goal, params=costly)
    assert short.pathways[0].nodes == ("a", "g") and short.pathways[0].cost == pytest.approx(2.0)


def test_cuts_budgets_and_failures_are_counted():
    edges = {"a": [("b", -0.1, ()), ("dead", -0.2, ()), ("a", -0.1, ())], "b": [("c", -0.1, ())],
             "c": [("g", -0.1, ())]}  # fmt: skip
    goal = lambda s: s == "g" or None  # noqa: E731
    graph = ImplicitGraph(GraphExpander(edges), "a")
    deep = k_best_pathways(graph, goal, params=SearchParams(k=1, max_depth=2))
    assert deep.pathways == [] and deep.cuts["DEPTH_CUT"] >= 1 and deep.cuts["CYCLE"] == 1
    assert deep.reason == "EXHAUSTED" and graph.failures == {"dead": "DEAD_END"}
    assert graph.skips["EXPAND_FAILED:DEAD_END"] == 1 and graph.skips["SEEN"] == graph.expanded
    tight = k_best_pathways(ImplicitGraph(GraphExpander(edges), "a"), goal,
                            params=SearchParams(k=1, max_expansions=1))  # fmt: skip
    assert tight.expanded == 1 and tight.cuts["EXPANSION_BUDGET"] >= 1
    assert k_best_pathways(ImplicitGraph(GraphExpander(edges), "g"), goal).reason == "START_IS_GOAL"
    pops = k_best_pathways(ImplicitGraph(GraphExpander(edges), "a"), goal,
                           params=SearchParams(max_pops=1))  # fmt: skip
    assert pops.reason == "POP_BUDGET"


def _two_orders():
    """a -> b -> d and a -> c -> d by the same two independent steps (atoms 1 and 2), and
    a -> e -> d by two dependent ones (both on atom 3)."""
    s1, s2, s3, s4 = "1:+2", "2:+2", "3:-2", "3:+2"
    edges = {
        "a": [("b", -0.2, s1, {1}), ("c", -0.4, s2, {2}), ("e", -0.3, s3, {3})],
        "b": [("d", -0.4, s2, {2})],
        "c": [("d", -0.2, s1, {1})],
        "e": [("d", -0.1, s4, {3})],
    }

    class Exp(GraphExpander):
        def expand(self, state):
            props = [Proposal(t, lp, sig, frozenset(touched))
                     for t, lp, sig, touched in self.edges.get(state, [])]  # fmt: skip
            return Expansion(props)

    return Exp(edges)


def test_independent_steps_in_either_order_are_one_mechanism():
    res = k_best_pathways(ImplicitGraph(_two_orders(), "a"), lambda s: s == "d" or None,
                          params=SearchParams(k=5))  # fmt: skip
    assert len(res.pathways) == 3
    classes = mechanism_classes(res.pathways)
    assert [len(c.pathways) for c in classes] == [2, 1]
    assert classes[0].steps == ("1:+2", "2:+2")
    assert sum(c.probability for c in classes) == pytest.approx(1.0)
    p = {x.nodes[1]: math.exp(x.logp) for x in res.pathways}
    assert classes[0].probability == pytest.approx((p["b"] + p["c"]) / sum(p.values()))
    dependent = next(x for x in res.pathways if x.nodes[1] == "e")
    assert mechanism_class(dependent.edges) == ("3:-2", "3:+2")  # never reordered


def test_best_simple_paths_ranks_and_stops_on_ends():
    edges = {"r": [("x", -0.1), ("y", -0.5)], "x": [("y", -0.1), ("r", -0.1)], "y": []}

    def admissible(node, on_path):
        return [(node, t, w) for t, w in edges[node] if t not in on_path]

    ops = PathOps(
        target=lambda e: e[1], weight=lambda e: e[2], edge_id=lambda e: f"{e[0]}{e[1]}",
        admissible=admissible, ends=lambda e, nodes: "END" if e[1] == "y" else None,
    )  # fmt: skip
    paths = best_simple_paths("r", ops)
    assert [p[0] for p in paths] == [("r", "x", "y"), ("r", "y")]
    assert paths[0][2] == pytest.approx(-0.2) and paths[0][3] == "END"
    assert best_simple_paths("r", ops, limit=1) == paths[:1]


def test_turnover_closure_finds_the_catalytic_return():
    root = Counter({"A": 1, "cat": 1})
    pools = [root, Counter({"A-cat": 1}), Counter({"B": 1, "cat": 1})]
    t = turnover_closure(pools, set(root))
    assert t is not None and (t.anchor, t.consumed, t.produced) == (0, {"A": 1}, {"B": 1})
    assert t.catalyst == {"cat": 1}
    assert turnover_closure(pools[:2], set(root)) is None
    assert (
        turnover_closure(
            [root, Counter({"A": 1, "cat": 1}), Counter({"B": 1, "cat": 1})], set(root)
        )
        is None
    )  # fmt: skip (the catalyst never dipped)
