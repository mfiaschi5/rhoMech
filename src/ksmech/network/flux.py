"""Masses, step fluxes and atom-flux analysis on the species graph (reaction-network plan, item J).

All of it is the fixed-concentration approximation: each step is a channel of its carrier (the
species whose expansion found it, `network.species`), whose partners are held at fixed
concentration, so the weights are the model's branching ratios and not a kinetic solution.
Hyperpath weights are not products of step probabilities once bimolecular steps compete for
partners; these stand until a microkinetic simulation with rates replaces them, and the end
state comes from the thermodynamic equilibrium (item F), not from here.

- `masses`: m(x) = amount(x) + sum over the steps s making x of (copies of x s makes) F(s), with
  F(s) = sum over carriers c of m(c) beta(s | c) (`step_fluxes`). Only core species carry, so the
  core solves (I - A) m = amount (A[p, c] = sum_s beta(s | c) copies of p in s) and every other
  species takes what the core sends it. A gain of at least 1 (chain branching at fixed partner
  concentration) has no finite solution: the series is cut after `DIVERGED_TERMS` terms and the
  solve reports it (MASS_DIVERGED). m is the expected number of times x forms per start, so a
  reversible pair counts each return.
- `balances`: for every species, produced (by steps), supplied (its start amount, and what it
  gives as a partner at fixed concentration or as a reservoir species), consumed (by steps, as
  carrier or partner) and terminal (what stays: a species not expanded, or the carrier's mass no
  channel took). produced + supplied = consumed + terminal at every species.
- Atom flux (`atom_edges`): every step carries its atom map, so each atom of a product is an atom
  of a reactant, and the flux of a target atom traces back through the network as an ordinary
  graph on (species, local atom). Going back from an atom of x, the share of the step s that made
  it is F(s) / (produced(x) + supplied(x)), and the share of x's supply ends the path (its origin:
  START, RESERVOIR, or PARTNER for an intermediate held at fixed concentration).
  `atom_pathways` extracts each target atom's most probable backward paths (best-first simple
  paths of `network.graph`), and `merged_pathway` merges the atoms' dominant paths into one set
  of steps. `atom_balance` is the conservation law at every (species, atom).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ksmech.network.graph import PathOps, best_simple_paths

DIVERGED_TERMS = 64
START, RESERVOIR, PARTNER = "START", "RESERVOIR", "PARTNER"
NO_ATOM_MAP = "NO_ATOM_MAP"


def _solve(a: np.ndarray, amount: np.ndarray) -> tuple[np.ndarray, bool]:
    """m with (I - a) m = amount, or the truncated series when the gain reaches 1."""
    if not len(amount):
        return amount, False
    radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    if radius < 1.0 - 1e-9:
        return np.linalg.solve(np.eye(len(amount)) - a, amount), False
    out, term = amount.copy(), amount.copy()
    for _ in range(DIVERGED_TERMS - 1):
        term = a @ term
        out += term
    return out, True


def masses(g) -> tuple[dict[str, float], bool]:
    """(m of every species of the graph, whether the solve diverged); module docstring."""
    core = list(g.core)
    at = {x: i for i, x in enumerate(core)}
    a = np.zeros((len(core), len(core)))
    feed: dict[str, np.ndarray] = {}
    for j, x in enumerate(core):
        for key, beta in g.branching.get(x, {}).items():
            for p in g.steps[key].products:
                if p in at:
                    a[at[p], j] += beta
                else:
                    feed.setdefault(p, np.zeros(len(core)))[j] += beta
    amount = np.array([float(g.sources.get(x, 0.0)) for x in core])
    m_core, diverged = _solve(a, amount)
    out = {x: float(m_core[i]) for i, x in enumerate(core)}
    for k in sorted((set(g.species) | set(g.sources) | set(feed)) - set(core)):
        sent = float(feed[k] @ m_core) if k in feed else 0.0
        out[k] = float(g.sources.get(k, 0.0)) + sent
    return out, diverged


def carrier_fluxes(g, mass: dict[str, float]) -> dict[tuple, dict[str, float]]:
    """step key -> {carrier: m(carrier) beta(step | carrier)}."""
    out: dict[tuple, dict[str, float]] = defaultdict(dict)
    for x, channels in g.branching.items():
        for key, beta in channels.items():
            out[key][x] = mass.get(x, 0.0) * beta
    return dict(out)


def step_fluxes(g, mass: dict[str, float]) -> dict[tuple, float]:
    """F(s) of every step of the graph (0 for a step no carrier takes)."""
    by = carrier_fluxes(g, mass)
    return {key: sum(by.get(key, {}).values()) for key in g.steps}


@dataclass(frozen=True)
class Balance:
    produced: float
    supplied: float
    consumed: float
    terminal: float

    @property
    def total(self) -> float:  # what forms or arrives: produced + supplied
        return self.produced + self.supplied


def balances(g, mass: dict[str, float]) -> dict[str, Balance]:
    """The balance of every species (module docstring)."""
    produced, supplied, consumed, carried = Counter(), Counter(), Counter(), Counter()
    for key, by in carrier_fluxes(g, mass).items():
        step, f = g.steps[key], sum(by.values())
        for u, n in Counter(step.reactants).items():
            consumed[u] += n * f
            supplied[u] += n * f - by.get(u, 0.0)
            carried[u] += by.get(u, 0.0)
        for p in step.products:
            produced[p] += f
    out = {}
    for k in sorted(set(mass) | set(produced) | set(consumed)):
        supply = float(g.sources.get(k, 0.0)) + supplied[k]
        terminal = mass.get(k, 0.0) - carried[k]
        out[k] = Balance(produced[k], supply, consumed[k], terminal)
    return out


@dataclass(frozen=True)
class AtomEdge:
    source: tuple[str, int]  # (reactant species, local atom)
    target: tuple[str, int]  # (product species, local atom)
    flux: float
    step: tuple  # the step key


def atom_edges(g, flux: dict[tuple, float], skips: Counter | None = None) -> list[AtomEdge]:
    """One edge per atom of every product slot of every step, with the step's flux; a step
    without an atom map is left out (NO_ATOM_MAP)."""
    out = []
    for key, step in g.steps.items():
        if len(step.atom_map) != len(step.products):
            if skips is not None:
                skips[NO_ATOM_MAP] += 1
            continue
        for j, p in enumerate(step.products):
            for b, (i, a) in enumerate(step.atom_map[j]):
                out.append(AtomEdge((step.reactants[i], a), (p, b), flux.get(key, 0.0), key))
    return out


def atom_balance(g, mass: dict[str, float]) -> dict[tuple[str, int], tuple[float, float]]:
    """(in, out) at every (species, local atom): in = the atom edges entering + the species'
    supply, out = the atom edges leaving + its terminal mass. Equal where atoms are conserved."""
    flux, bal = step_fluxes(g, mass), balances(g, mass)
    into, out = Counter(), Counter()
    for e in atom_edges(g, flux):
        into[e.target] += e.flux
        out[e.source] += e.flux
    res = {}
    for k, b in bal.items():
        for a in range(g.species[k].n_atoms if k in g.species else 0):
            res[(k, a)] = (into[(k, a)] + b.supplied, out[(k, a)] + b.terminal)
    return res


@dataclass(frozen=True)
class AtomPath:
    atom: int  # the target's local atom
    nodes: tuple[tuple[str, int], ...]  # (species, local atom) from the target back
    steps: tuple[tuple, ...]  # the step keys, from the target back
    origin: str  # START | RESERVOIR | PARTNER: the supply the path ends at
    share: float  # the probability that the atom came this way


def _origin(g, key: str) -> str:
    return START if key in g.sources else RESERVOIR if key in g.reservoir else PARTNER


class _Backward:
    """The backward atom graph as `graph.PathOps`: node ids index `nodes`; an edge is (from,
    to, share, edge id, step key), the supply edge's step None and its id SUPPLY_EDGE."""

    def __init__(self, g, mass: dict[str, float]) -> None:
        flux, bal = step_fluxes(g, mass), balances(g, mass)
        self.nodes: list[tuple[str, int]] = []
        self.index: dict[tuple[str, int], str] = {}
        self.back: dict[str, list] = defaultdict(list)
        self.step_of: dict[str, tuple] = {}
        for n, e in enumerate(atom_edges(g, flux)):
            total = bal[e.target[0]].total
            if e.flux > 0 and total > 0:
                eid = f"{n:09d}"
                self.step_of[eid] = e.step
                edge = (self.id(e.target), self.id(e.source), e.flux / total, eid, e.step)
                self.back[edge[0]].append(edge)
        for k, b in bal.items():
            for a in range(g.species[k].n_atoms if k in g.species and b.supplied > 0 else 0):
                x = self.id((k, a))
                self.back[x].append((x, f"S{k}", b.supplied / b.total, SUPPLY_EDGE, None))
        self.ops = PathOps(
            target=lambda e: e[1], weight=lambda e: math.log(e[2]), edge_id=lambda e: e[3],
            admissible=lambda x, on: [e for e in self.back.get(x, []) if e[1] not in on],
            ends=lambda e, _: _origin(g, self.nodes[int(e[0])][0]) if e[4] is None else None,
        )  # fmt: skip

    def id(self, node: tuple[str, int]) -> str:
        if node not in self.index:
            self.index[node] = f"{len(self.nodes):09d}"
            self.nodes.append(node)
        return self.index[node]


SUPPLY_EDGE = "SUPPLY"


def atom_pathways(
    g, mass: dict[str, float], target: str, per_atom: int = 1, heavy_only: bool = True
) -> list[AtomPath]:
    """The `per_atom` most probable backward paths of every atom of `target` (heavy atoms only
    when the species records its elements, has one and `heavy_only`), best first per atom."""
    bw = _Backward(g, mass)
    sp = g.species[target]
    heavy_only = heavy_only and any(z > 1 for z in sp.elements)
    out = []
    for a in range(sp.n_atoms):
        if heavy_only and sp.elements[a] == 1:
            continue
        for path, eids, logw, origin in best_simple_paths(bw.id((target, a)), bw.ops, per_atom):
            trail = tuple(bw.nodes[int(x)] for x in path[:-1])
            keys = tuple(bw.step_of[e] for e in eids if e != SUPPLY_EDGE)
            out.append(AtomPath(a, trail, keys, origin, math.exp(logw)))
    return out


@dataclass(frozen=True)
class MergedPathway:
    steps: tuple[tuple, ...]  # the union of the atoms' dominant paths' steps, sorted
    atom_shares: dict[int, float]  # each traced atom's share on its dominant path
    origins: dict[int, str]

    @property
    def share(self) -> float:
        """The smallest atom share: at most this fraction of the target follows every path."""
        return min(self.atom_shares.values(), default=0.0)


def merged_pathway(paths: Sequence[AtomPath]) -> MergedPathway:
    """The dominant path of each atom (its first in `paths`) merged into one set of steps."""
    first: dict[int, AtomPath] = {}
    for p in paths:
        first.setdefault(p.atom, p)
    steps = sorted({k for p in first.values() for k in p.steps})
    return MergedPathway(tuple(steps), {a: p.share for a, p in first.items()},
                         {a: p.origin for a, p in first.items()})  # fmt: skip
