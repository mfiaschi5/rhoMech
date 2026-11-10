"""Pure algorithms over an explored network: best-first simple paths and the turnover rule of a
cycle (design 4.4 and 4.5 of the web app's network, moved here so the engine and the web app
share them; `serve.predict.network` keeps its data model and JSON form and calls these).

No RDKit, no model, no I/O: nodes are ids, edges are whatever the caller stores, read through
the callables it passes.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

MAX_PATHWAY_POPS = 10_000


@dataclass(frozen=True)
class Turnover:
    """A turnover closed by the last pool of a path: it returns to the pool at `anchor` (an
    index on the path) having consumed some species of the root and produced others, while
    `catalyst` dipped below its count in between and came back."""

    anchor: int
    consumed: dict[str, int]
    produced: dict[str, int]
    catalyst: dict[str, int]


def _catalyst(pools: Sequence[Counter], j: int, inter: Counter) -> dict[str, int]:
    between = pools[j + 1 : -1]
    return {x: n for x, n in sorted(inter.items()) if min(p[x] for p in between) < n}


def turnover_closure(pools: Sequence[Counter], root_support: set[str]) -> Turnover | None:
    """The latest pool w on the path (at least two steps back) that the last pool v turns over:
    v consumed species of w drawn from the root and produced others, and some species shared by
    w and v dipped in between (the catalyst). None when there is none."""
    p_v = pools[-1]
    for j in range(len(pools) - 3, -1, -1):
        p_w = pools[j]
        consumed, produced = p_w - p_v, p_v - p_w
        if not consumed or not produced or not set(consumed) <= root_support:
            continue
        catalyst = _catalyst(pools, j, p_w & p_v)
        if catalyst:
            return Turnover(j, dict(sorted(consumed.items())), dict(sorted(produced.items())),
                            catalyst)  # fmt: skip
    return None


@dataclass(frozen=True)
class PathOps:
    """How `best_simple_paths` reads the caller's edges: `target(e)` the node an edge enters,
    `weight(e)` its log weight (added along a path), `edge_id(e)` a sortable id (ties),
    `admissible(x, on_path)` the edges leaving x a path through `on_path` may take, and
    `ends(e, nodes)` how a path whose last edge is e ends (None: it continues)."""

    target: Callable[[Any], str]
    weight: Callable[[Any], float]
    edge_id: Callable[[Any], str]
    admissible: Callable[[str, Sequence[str]], Sequence[Any]]
    ends: Callable[[Any, Sequence[str]], str | None]


def best_simple_paths(
    root: str, ops: PathOps, limit: int = 20, max_pops: int = MAX_PATHWAY_POPS
) -> list[tuple[tuple[str, ...], tuple[str, ...], float, str]]:
    """(nodes, edge ids, summed log weight, how it ends) of the best-first paths from `root`,
    by summed weight desc, then fewer edges, then edge ids; at most `limit`, within `max_pops`
    heap pops. A path is extended only along `ops.admissible` edges, so the caller decides
    which revisits close a path (a recurrence) and which are refused."""
    heap: list[tuple] = []

    def push(nodes: tuple[str, ...], eids: tuple[str, ...], logp: float, e) -> None:
        nodes2, eids2 = nodes + (ops.target(e),), eids + (ops.edge_id(e),)
        logp2 = logp + ops.weight(e)
        heapq.heappush(heap, (-logp2, len(eids2), eids2, nodes2, ops.ends(e, nodes2)))

    for e in ops.admissible(root, ()):  # every edge of the root, as the web app's ranking
        push((root,), (), 0.0, e)
    out: list[tuple[tuple[str, ...], tuple[str, ...], float, str]] = []
    pops = 0
    while heap and len(out) < limit and pops < max_pops:
        neg, _, eids, nodes, ends = heapq.heappop(heap)
        pops += 1
        if ends is not None:
            out.append((nodes, eids, -neg, ends))
            continue
        for e in ops.admissible(nodes[-1], nodes):
            push(nodes, eids, -neg, e)
    return out
