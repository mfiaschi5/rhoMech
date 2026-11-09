"""The explored network as a Markov chain (reaction-network plan, items E and F).

Reverse edges. Every explored edge R (parent i -> child j) implies the reverse step -R from j
to i (microscopic reversibility), scored by the forward model at the child
(`ReverseScorer.reverse_logp`; `ModelExpander.reverse_logp` returns a `MatrixScore`, whose
`best` anchor is read, the decoder's own ranking). A child that cannot be featurized, or a -R
without an admissible anchor, leaves the reverse missing, counted once per edge as
REVERSE_MISSING:<code>. The log p of a directed pair (i, j) is the best of what scores it: i's
proposal for j and the reverse of j's proposal for i, merged by max as the decoder merges the
completions of one product.

Potentials. phi(x) = G(x) - sum_r D_r(x) mu_r: G from the `ThermoModel`, D_r the copies of
reservoir species r drawn from the start to x (`network.reservoir`; summed along the first path
found, another path with other draws is RESERVOIR_DRAW_MISMATCH), mu_r its chemical potential.
A state without G, or with a draw of a species without mu, has none (NO_FREE_ENERGY).

Detailed balance. A pair with both potentials and both directions gets the rates
    w_ij = s_ij exp(-dG_ij / 2RT),  w_ji = s_ij exp(+dG_ij / 2RT),  s_ij = sqrt(p_ij p_ji),
dG_ij = phi_j - phi_i, so pi_i w_ij = pi_j w_ji with pi proportional to exp(-phi / RT): the
stationary distribution of the chain (continuous time, rates w) is Boltzmann over the pools.
The model shapes the pathways (s), the free energies fix where they end. A pair without both
potentials falls back to the model's own probabilities, w_ij = p_ij, w_ji = p_ji
(NO_FREE_ENERGY_LINK); a pair scored one way only has w_ij = p_ij exp(-dG_ij / 2RT) (or p_ij)
and w_ji = 0 (ONE_WAY). A step to its own pool key does not move the chain (SELF_LOOP). Such
pairs are `MarkovChain.unbalanced`. Every explored edge records dG(T), K(T) = exp(-dG/RT), dH,
dS and the inversion temperature T* = dH/dS (item F); no edge is cut for being uphill.

Absorption. Given the absorbing states, `absorb` gives the probability of ending in each from
every transient state by one sparse LU solve on the jump chain P = W / rowsum(W) (the jump
chain of the continuous-time chain has the same absorption probabilities). A closed class of
transient states that reaches no absorbing state (an explored region at equilibrium) is a
target of its own, EQUILIBRIUM:<n>, its states weighted by the chain's stationary distribution
within it (`stationary`; Boltzmann where the potentials are known and the pairs balanced).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ksmech.network.free_energy import (
    EXP_CLIP,
    NO_FREE_ENERGY,
    Inversion,
    Thermo,
    ThermoModel,
    as_thermo,
    combine,
    difference,
    equilibrium_constant,
    inversion,
    rt,
)
from ksmech.network.reservoir import RESERVOIR_DRAW
from ksmech.network.search import ImplicitGraph, Proposal

EQUILIBRIUM = "EQUILIBRIUM"
REVERSE_MISSING = "REVERSE_MISSING"


class ReverseScorer(Protocol):
    def reverse_logp(self, proposal: Proposal) -> Any: ...  # MatrixScore | float | failure code


def reverse_score(scorer, proposal: Proposal) -> tuple[float | None, str | None]:
    """(log p of -R at the proposal's child, None) or (None, the reason it is missing)."""
    score = getattr(scorer, "reverse_logp", None)
    if score is None:
        return None, "NO_SCORER"
    out = score(proposal)
    if isinstance(out, str):
        return None, out
    value = float(getattr(out, "best", out))
    return (value, None) if math.isfinite(value) else (None, "INADMISSIBLE")


class ReverseCache:
    """The reverse log p of every explored edge, (source, target) -> float | None, scored once
    (a model call at the child); `counts` holds REVERSE_MISSING:<code> once per edge."""

    def __init__(self, scorer) -> None:
        self.scorer = scorer
        self.scores: dict[tuple[str, str], float | None] = {}
        self.counts: Counter = Counter()

    def update(self, graph: ImplicitGraph) -> None:
        for src, edges in graph.out.items():
            for e in edges:
                if (src, e.target) in self.scores or e.target == src:
                    continue
                value, why = reverse_score(self.scorer, e.proposal)
                if why is not None:
                    self.counts[f"{REVERSE_MISSING}:{why}"] += 1
                self.scores[(src, e.target)] = value


class ThermoCache:
    """G of each state (by key) and mu of each reservoir species at one temperature, asked once."""

    def __init__(self, model: ThermoModel | None, temperature_k: float) -> None:
        self.model, self.temperature_k = model, float(temperature_k)
        self._g: dict[str, Thermo | None] = {}
        self._mu: dict[str, Thermo | None] = {}

    def state(self, key: str, state) -> Thermo | None:
        if key not in self._g:
            raw = None if self.model is None else self.model.free_energy(state, self.temperature_k)
            self._g[key] = as_thermo(raw)
        return self._g[key]

    def mu(self, species: str) -> Thermo | None:
        if species not in self._mu:
            f = None if self.model is None else self.model.chemical_potential
            self._mu[species] = None if f is None else as_thermo(f(species, self.temperature_k))
        return self._mu[species]


def _signed_add(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, n in b.items():
        out[k] = out.get(k, 0) + int(n)
    return {k: n for k, n in sorted(out.items()) if n}


def cumulative_draws(graph: ImplicitGraph, counts: Counter) -> dict[str, dict[str, int]]:
    """The reservoir copies drawn from the root to each state, along the first path found by
    breadth-first search over the explored edges (module docstring)."""
    draws: dict[str, dict[str, int]] = {graph.root: {}}
    queue = deque([graph.root])
    while queue:
        src = queue.popleft()
        for e in graph.out.get(src, []):
            d = _signed_add(draws[src], e.proposal.info.get(RESERVOIR_DRAW, {}))
            if e.target not in draws:
                draws[e.target] = d
                queue.append(e.target)
            elif draws[e.target] != d:
                counts["RESERVOIR_DRAW_MISMATCH"] += 1
    return draws


def potentials(graph: ImplicitGraph, thermo: ThermoCache, counts: Counter) -> dict:
    """phi of every state of the graph (a `Thermo`), None without one (NO_FREE_ENERGY)."""
    draws = cumulative_draws(graph, counts)
    out: dict[str, Thermo | None] = {}
    for key, state in graph.states.items():
        g, d = thermo.state(key, state), draws.get(key)
        terms = [(thermo.mu(x), -float(n)) for x, n in (d or {}).items()]
        if g is not None and d is not None and any(m is None for m, _ in terms):
            counts["NO_CHEMICAL_POTENTIAL"] += 1
            g = None
        out[key] = None if g is None or d is None else combine(g, terms)
        counts[NO_FREE_ENERGY] += out[key] is None
    return out


def directed_logps(graph: ImplicitGraph, reverse: Mapping, counts: Counter) -> dict:
    """(i, j) -> log p of the step from i to j: the best of i's proposal and the reverse score
    of j's proposal for i."""
    out: dict[tuple[str, str], float] = {}

    def put(i: str, j: str, lp: float) -> None:
        out[(i, j)] = max(out.get((i, j), -math.inf), lp)

    for src, edges in graph.out.items():
        for e in edges:
            if e.target == src:
                counts["SELF_LOOP"] += 1
                continue
            put(src, e.target, e.logp)
            back = reverse.get((src, e.target))
            if back is not None:
                put(e.target, src, back)
    return out


def log_rates(logps: Mapping, phi: Mapping, temperature_k: float, counts: Counter):
    """(i, j) -> log w_ij, and the unbalanced pairs (module docstring)."""
    out: dict[tuple[str, str], float] = {}
    unbalanced: set[frozenset[str]] = set()
    two_rt = 2.0 * rt(temperature_k)
    for (i, j), lp in logps.items():
        back, gi, gj = logps.get((j, i)), phi.get(i), phi.get(j)
        if gi is None or gj is None:
            out[(i, j)] = lp
            counts["NO_FREE_ENERGY_LINK"] += back is None or i < j
        elif back is None:
            out[(i, j)] = lp - (gj.g - gi.g) / two_rt
            counts["ONE_WAY"] += 1
        else:
            out[(i, j)] = 0.5 * (lp + back) - (gj.g - gi.g) / two_rt
            continue
        unbalanced.add(frozenset((i, j)))
    return out, unbalanced


@dataclass(frozen=True)
class EdgeRecord:
    """One explored edge with its thermodynamics at T (item F)."""

    source: str
    target: str
    signature: str
    logp: float  # the forward model's log p
    reverse_logp: float | None  # -R scored at the child; None when missing
    draw: Mapping[str, int]  # reservoir copies the step took
    dg: float | None = None  # kcal/mol, phi(target) - phi(source)
    k_eq: float | None = None  # exp(-dG / RT)
    dh: float | None = None
    ds: float | None = None  # kcal/(mol K)
    inversion: Inversion | None = None


def edge_records(graph: ImplicitGraph, reverse: Mapping, phi: Mapping, temperature_k: float):
    out = []
    for src, edges in graph.out.items():
        for e in edges:
            rec = dict(source=src, target=e.target, signature=e.proposal.signature,
                       logp=e.logp, reverse_logp=reverse.get((src, e.target)),
                       draw=dict(e.proposal.info.get(RESERVOIR_DRAW, {})))  # fmt: skip
            gi, gj = phi.get(src), phi.get(e.target)
            if gi is not None and gj is not None:
                d = difference(gj, gi)
                rec.update(dg=d.g, k_eq=equilibrium_constant(d.g, temperature_k), dh=d.h,
                           ds=d.s, inversion=inversion(d.h, d.s))  # fmt: skip
            out.append(EdgeRecord(**rec))
    return out


@dataclass
class MarkovChain:
    keys: list[str]  # the graph's states, the root first
    index: dict[str, int]
    rates: sp.csr_matrix  # w_ij, zero diagonal
    potentials: dict[str, Thermo | None]
    edges: list[EdgeRecord]
    temperature_k: float
    unbalanced: set[frozenset[str]] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)

    def out_links(self, key: str) -> list[str]:
        return [self.keys[j] for j in self.rates[self.index[key]].indices]

    def relative_g(self, key: str) -> float | None:
        """G of a state relative to the root (kcal/mol), None when either is unknown."""
        a, b = self.potentials.get(key), self.potentials.get(self.keys[0])
        return None if a is None or b is None else a.g - b.g


def build_chain(graph: ImplicitGraph, reverse: ReverseCache, thermo: ThermoCache) -> MarkovChain:
    """The chain of the graph as explored (module docstring); scores the missing reverses."""
    reverse.update(graph)
    counts: Counter = Counter()
    phi = potentials(graph, thermo, counts)
    logps = directed_logps(graph, reverse.scores, counts)
    logw, unbalanced = log_rates(logps, phi, thermo.temperature_k, counts)
    keys = list(graph.states)
    index = {k: n for n, k in enumerate(keys)}
    rows = [index[i] for i, _ in logw]
    cols = [index[j] for _, j in logw]
    vals = [math.exp(max(-EXP_CLIP, min(EXP_CLIP, v))) for v in logw.values()]
    rates = sp.csr_matrix((vals, (rows, cols)), shape=(len(keys), len(keys)))
    edges = edge_records(graph, reverse.scores, phi, thermo.temperature_k)
    counts.update(reverse.counts)
    return MarkovChain(keys, index, rates, phi, edges, thermo.temperature_k, unbalanced, counts)


# ---- equilibrium and absorption


def boltzmann(phi: Mapping[str, Thermo | None], keys: Iterable[str], temperature_k: float):
    """key -> exp(-phi / RT) / Z over the keys with a potential."""
    known = [(k, phi[k].g) for k in keys if phi.get(k) is not None]
    if not known:
        return {}
    low = min(g for _, g in known)
    w = {k: math.exp(-(g - low) / rt(temperature_k)) for k, g in known}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def stationary(chain: MarkovChain, keys: Sequence[str] | None = None) -> dict[str, float]:
    """The stationary distribution of the continuous-time chain restricted to `keys` (default
    every state; rates leaving the set ignored), which must be irreducible: pi Q = 0, sum 1."""
    keys = list(chain.keys if keys is None else keys)
    if len(keys) == 1:
        return {keys[0]: 1.0}
    idx = [chain.index[k] for k in keys]
    w = chain.rates[idx][:, idx].tocsr()
    q = (w - sp.diags(np.asarray(w.sum(axis=1)).ravel())).T.tolil()
    q[len(keys) - 1, :] = np.ones(len(keys))
    b = np.zeros(len(keys))
    b[-1] = 1.0
    pi = np.clip(splu(q.tocsc()).solve(b), 0.0, None)  # round-off below 0
    return {k: float(p) for k, p in zip(keys, pi / pi.sum(), strict=True)}


def equilibrium_weights(chain: MarkovChain, members: list[str]) -> dict[str, float]:
    """A closed class's states weighted by Boltzmann when it is balanced, else by `stationary`."""
    balanced = all(chain.potentials.get(k) is not None for k in members) and not any(
        pair <= set(members) for pair in chain.unbalanced
    )
    if balanced:
        return boltzmann(chain.potentials, members, chain.temperature_k)
    return stationary(chain, members)


def _reaching(chain: MarkovChain, transient: list[int], absorbing: set[int]) -> set[int]:
    """The transient states with a path to an absorbing state."""
    t = set(transient)
    pred: dict[int, list[int]] = {}
    coo = chain.rates.tocoo()
    for i, j in zip(coo.row, coo.col, strict=True):
        if i in t:
            pred.setdefault(int(j), []).append(int(i))
    seen, queue = set(), deque(absorbing)
    while queue:
        for i in pred.get(queue.popleft(), []):
            if i not in seen:
                seen.add(i)
                queue.append(i)
    return seen


def closed_classes(chain: MarkovChain, closed: list[int]) -> list[list[int]]:
    """The bottom strongly connected classes among `closed` (states reaching no absorbing one)."""
    if not closed:
        return []
    sub = chain.rates[closed][:, closed]
    n, labels = connected_components(sub, directed=True, connection="strong")
    out = []
    for c in range(n):
        members = [closed[i] for i in np.flatnonzero(labels == c)]
        outside = chain.rates[members].indices
        if set(int(j) for j in outside) <= set(members):
            out.append(sorted(members))
    return sorted(out)


@dataclass
class Absorption:
    """Absorption probabilities: `targets` are absorbing states' keys and EQUILIBRIUM:<n>
    classes; `members[t]` the states of a target with their weights; `rows[key]` the
    probability of ending in each target from the state `key`."""

    targets: list[str]
    reasons: dict[str, str]
    members: dict[str, dict[str, float]]
    rows: dict[str, dict[str, float]]

    def state_probabilities(self, key: str) -> dict[str, float]:
        """End state -> probability from `key`, a class's mass spread over its members."""
        out: Counter = Counter()
        for t, p in self.rows[key].items():
            for state, w in self.members[t].items():
                out[state] += p * w
        return dict(out)


def _targets(chain: MarkovChain, absorbing: Mapping[str, str]):
    """(targets, reasons, members, target index of each absorbing or class state, transient)."""
    absorbed = {chain.index[k] for k in absorbing}
    transient = [i for i in range(len(chain.keys)) if i not in absorbed]
    reach = _reaching(chain, transient, absorbed)
    classes = closed_classes(chain, [i for i in transient if i not in reach])
    targets = [k for k in chain.keys if k in absorbing]
    reasons = {k: absorbing[k] for k in targets}
    members = {k: {k: 1.0} for k in targets}
    where = {chain.index[k]: n for n, k in enumerate(targets)}
    for n, cls in enumerate(classes):
        name = f"{EQUILIBRIUM}:{n}"
        reasons[name] = EQUILIBRIUM
        members[name] = equilibrium_weights(chain, [chain.keys[i] for i in cls])
        where.update({i: len(targets) for i in cls})
        targets.append(name)
    rest = [i for i in transient if i not in where]
    return targets, reasons, members, where, rest


def absorb(
    chain: MarkovChain, absorbing: Mapping[str, str], sources: Sequence[str] | None = None
) -> Absorption:
    """The absorption probabilities from each source (default every state) into the absorbing
    states (key -> reason) and the closed classes (module docstring)."""
    targets, reasons, members, where, rest = _targets(chain, absorbing)
    sources = list(chain.keys if sources is None else sources)
    rows: dict[str, dict[str, float]] = {}
    for key in sources:
        if chain.index[key] in where:
            rows[key] = {targets[where[chain.index[key]]]: 1.0}
    todo = [k for k in sources if k not in rows]
    if todo:
        x = _solve_rows(chain, rest, where, len(targets), [chain.index[k] for k in todo])
        for key, row in zip(todo, x, strict=True):
            rows[key] = {targets[t]: float(p) for t, p in enumerate(row) if p > 0}
    return Absorption(targets, reasons, members, rows)


def _solve_rows(chain, rest: list[int], where: dict[int, int], n_targets: int, src: list[int]):
    """Rows of (I - P_TT)^-1 P_TA for the source states (all in `rest`)."""
    w = chain.rates.tocsr()
    total = np.asarray(w.sum(axis=1)).ravel()
    p = sp.diags(1.0 / np.where(total > 0, total, 1.0)) @ w
    p_tt = p[rest][:, rest]
    gather = sp.csr_matrix(
        (np.ones(len(where)), (list(where), list(where.values()))), shape=(w.shape[0], n_targets)
    )
    p_ta = (p[rest] @ gather).toarray()
    lu = splu((sp.identity(len(rest)) - p_tt).tocsc())
    pos = {i: n for n, i in enumerate(rest)}
    e = np.zeros((len(rest), len(src)))
    for n, i in enumerate(src):
        e[pos[i], n] = 1.0
    visits = lu.solve(e, trans="T")  # (I - P_TT)^-T e_s: expected visits from each source
    return np.clip(visits.T @ p_ta, 0.0, 1.0)
