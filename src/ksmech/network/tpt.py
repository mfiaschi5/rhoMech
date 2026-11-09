"""Transition path theory on the network's Markov chain (reaction-network plan, item I3): all
pathways from a reactant set A to a product set B, each with its weight.

On the strongly connected component of the chain (`network.chain`, rates w) that holds A, with
pi its stationary distribution (Boltzmann when the component is balanced):

- the forward committor q+(x), the probability of reaching B before A, solves
  sum_j w_xj (q+_j - q+_x) = 0 off A and B with q+ = 0 on A and 1 on B (one sparse LU solve);
  the backward committor q- is the same on the time-reversed chain w~_ij = pi_j w_ji / pi_i
  with q- = 1 on A and 0 on B (for a balanced chain q- = 1 - q+);
- the reactive flux f_ij = pi_i q-_i w_ij q+_j and its net part f+_ij = max(f_ij - f_ji, 0),
  conserved at every state off A and B; the total reactive flux F leaves A and enters B;
- the dominant pathways: repeatedly the path from A to B whose smallest net flux (its
  bottleneck) is largest, the bottleneck subtracted along it, each pathway with its share of
  F. On a balanced chain the net flux runs up the committor, so it has no cycle and the shares
  add up to 1 once every pathway is extracted; `residual` is what the decomposition left.

The committor of each intermediate also says how productive it is. States of B outside the
component are dropped (TPT_SINK_UNREACHABLE, counted); with none left the result is None.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ksmech.network.chain import MarkovChain, equilibrium_weights

PI_FLOOR = 1e-300


@dataclass(frozen=True)
class FluxPathway:
    nodes: tuple[str, ...]
    flux: float  # its bottleneck, the net reactive flux it carries
    share: float  # flux / total reactive flux


@dataclass
class TransitionPaths:
    keys: list[str]  # the component's states
    q_plus: dict[str, float]
    q_minus: dict[str, float]
    pi: dict[str, float]
    net_flux: dict[tuple[str, str], float]  # f+ on each edge with a positive one
    total: float  # the total reactive flux A -> B
    pathways: list[FluxPathway]
    residual: float  # the share of the total flux left undecomposed
    counts: Counter = field(default_factory=Counter)


def committor(rates: sp.csr_matrix, a: Sequence[int], b: Sequence[int]) -> np.ndarray:
    """q = 0 on a, 1 on b, harmonic for the generator elsewhere (module docstring); every
    other state must reach a or b."""
    n = rates.shape[0]
    q = np.zeros(n)
    q[list(b)] = 1.0
    inner = [i for i in range(n) if i not in set(a) | set(b)]
    if inner:
        w = rates.tocsr()
        gen = w - sp.diags(np.asarray(w.sum(axis=1)).ravel())
        rhs = -np.asarray(w[inner][:, list(b)].sum(axis=1)).ravel()
        q[inner] = splu(gen[inner][:, inner].tocsc()).solve(rhs)
    return np.clip(q, 0.0, 1.0)


def _component(chain: MarkovChain, sources: Sequence[str]) -> list[str]:
    """The states of the strongly connected component holding the first source."""
    _, labels = connected_components(chain.rates, directed=True, connection="strong")
    label = labels[chain.index[sources[0]]]
    return [k for k, lab in zip(chain.keys, labels, strict=True) if lab == label]


def _net_flux(w: sp.csr_matrix, pi: np.ndarray, qp: np.ndarray, qm: np.ndarray):
    f = (sp.diags(pi * qm) @ w @ sp.diags(qp)).tocsr()
    net = (f - f.T).tocsr()
    net.data = np.maximum(net.data, 0.0)
    net.eliminate_zeros()
    return net


def _widest_path(adj: dict[int, dict[int, float]], a: set[int], b: set[int]):
    """(path, bottleneck) of the path from a to b with the largest smallest capacity, None
    when b is not reachable through positive capacities."""
    best = {s: math.inf for s in a}
    prev: dict[int, int] = {}
    heap = [(-math.inf, s) for s in sorted(a)]
    done: set[int] = set()
    while heap:
        neg, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u in b:
            path = [u]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1], -neg
        for v, cap in adj.get(u, {}).items():
            width = min(-neg, cap)
            if cap > 0 and v not in done and width > best.get(v, 0.0):
                best[v], prev[v] = width, u
                heapq.heappush(heap, (-width, v))
    return None


def dominant_pathways(net: sp.csr_matrix, a: set[int], b: set[int], max_paths: int, tol: float):
    """[(path, flux)] extracted by largest bottleneck until `max_paths`, no path, or a
    bottleneck below `tol`."""
    coo = net.tocoo()
    adj: dict[int, dict[int, float]] = {}
    for i, j, v in zip(coo.row, coo.col, coo.data, strict=True):
        if int(i) not in b and int(j) not in a:
            adj.setdefault(int(i), {})[int(j)] = float(v)
    out = []
    while len(out) < max_paths:
        found = _widest_path(adj, a, b)
        if found is None or found[1] <= tol:
            break
        path, flux = found
        for u, v in zip(path, path[1:], strict=False):
            adj[u][v] -= flux
        out.append((path, flux))
    return out


def transition_paths(
    chain: MarkovChain, sources: Sequence[str], sinks: Sequence[str], max_paths: int = 10,
    tol: float = 1e-12,
) -> TransitionPaths | None:  # fmt: skip
    """Transition path theory from `sources` (A) to `sinks` (B) on the chain (module
    docstring); None when no state of B lies in A's component."""
    counts: Counter = Counter()
    keys = _component(chain, sources)
    inside = set(keys)
    a_keys = [k for k in sources if k in inside]
    b_keys = [k for k in sinks if k in inside and k not in set(a_keys)]
    counts["TPT_SOURCE_OUTSIDE"] += len(sources) - len(a_keys)
    counts["TPT_SINK_UNREACHABLE"] += len([k for k in sinks if k not in inside])
    if not b_keys:
        return None
    idx = {k: n for n, k in enumerate(keys)}
    rows = [chain.index[k] for k in keys]
    w = chain.rates[rows][:, rows].tocsr()
    weights = equilibrium_weights(chain, keys)
    pi = np.maximum([weights[k] for k in keys], PI_FLOOR)  # an underflowed Boltzmann weight
    a, b = [idx[k] for k in a_keys], [idx[k] for k in b_keys]
    qp = committor(w, a, b)
    back = (sp.diags(1.0 / pi) @ w.T @ sp.diags(pi)).tocsr()
    qm = committor(back, b, a)
    net = _net_flux(w, pi, qp, qm)
    total = float(net[a].sum())
    paths = dominant_pathways(net, set(a), set(b), max_paths, tol * max(total, 1e-300))
    flows = [FluxPathway(tuple(keys[i] for i in p), f, f / total if total > 0 else 0.0)
             for p, f in paths]  # fmt: skip
    residual = 1.0 - sum(p.share for p in flows) if total > 0 else 0.0
    coo = net.tocoo()
    flux = {
        (keys[i], keys[j]): float(v) for i, j, v in zip(coo.row, coo.col, coo.data, strict=True)
    }
    return TransitionPaths(keys, dict(zip(keys, qp.tolist(), strict=True)),
                           dict(zip(keys, qm.tolist(), strict=True)), weights, flux, total,
                           flows, residual, counts)  # fmt: skip
