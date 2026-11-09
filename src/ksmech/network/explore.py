"""What forms without a product: exploration by probability mass (reaction-network plan,
items E and I4).

`explore(expander, start, thermo, params)` grows the network best-first by the mass that reaches
each unexpanded pool. After every expansion the explored graph becomes a Markov chain
(`network.chain`: reverse edges scored at the children, detailed balance with the free
energies), each pool is classified, and the absorption probabilities from the start are solved:

- an expanded pool is EXPAND_FAILED when its expansion failed (no tokens, no Wiberg index: the
  model never saw it, so its mass is unexplained, reported with the unexplored mass and never as
  an end state), NO_OUTGOING_STEP when it proposed no step, SINK_AT_T when every step out
  climbs more than delta(T) after the rescan (`network.stop`), and transient otherwise;
- an unexpanded pool is TURNOVER when its first path from the start closes a catalytic
  turnover (`graph.turnover_closure`, never expanded; `stop_at_turnover` False expands it as any
  other pool), CLIMB_CUT when its G lies more than
  `max_climb_kcal` above the start (the cumulative climb allowed, a flag: never an edge cut,
  only a pool left unexpanded), and UNEXPLORED otherwise;
- an explored region that reaches no absorbing pool is an EQUILIBRIUM class (`chain.absorb`).

The probability of reaching each UNEXPLORED pool first is the mass it would explain; the most
probable one is expanded next, so an unlikely step that is the only way on (the enolate on the
way to the aldol) is followed once enough mass reaches it, instead of being pruned by a beam.
The search stops when the UNEXPLORED mass is at most `mass_tol` (MASS_TOL), when no pool is
left to expand (EXHAUSTED) or at `max_expansions` (EXPANSION_BUDGET). The result lists the end
states with their probabilities (sinks, turnovers, dead ends and equilibrium classes spread
over their pools), the unexplored mass (UNEXPLORED, CLIMB_CUT and EXPAND_FAILED: the
uncertainty), the sink checks, NO_NET_REACTION and the equilibrium populations over the
explored pools.

Monte Carlo tree search, the plan's option behind a flag, is not implemented.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ksmech.network.chain import (
    Absorption,
    MarkovChain,
    ReverseCache,
    ThermoCache,
    absorb,
    boltzmann,
    build_chain,
)
from ksmech.network.free_energy import Inversion, ThermoModel, difference, inversion
from ksmech.network.graph import turnover_closure
from ksmech.network.keys import split_pool_key
from ksmech.network.search import ImplicitGraph
from ksmech.network.stop import SINK_AT_T, SinkCheck, StopParams, confirm_sink, no_net_reaction
from ksmech.network.tpt import transition_paths

NO_OUTGOING_STEP, TURNOVER = "NO_OUTGOING_STEP", "TURNOVER"
UNEXPLORED, CLIMB_CUT, EXPAND_FAILED = "UNEXPLORED", "CLIMB_CUT", "EXPAND_FAILED"
NOT_ENDS = (UNEXPLORED, CLIMB_CUT, EXPAND_FAILED)


@dataclass(frozen=True)
class ExploreParams:
    temperature_k: float = 298.15  # K (not the predictor's softmax temperature)
    max_expansions: int = 50  # pools expanded (model calls)
    mass_tol: float = 0.01  # stop when the unexplored mass is at most this
    epsilon: float = 0.01  # provisional (docs/decisions.md)
    margin_kcal: float = 5.0  # provisional
    max_climb_kcal: float | None = 20.0  # provisional; None: no limit
    rescan_factor: int = 4
    stop_at_turnover: bool = True  # a pool closing a catalytic turnover is absorbing

    @property
    def stop(self) -> StopParams:
        return StopParams(self.temperature_k, self.epsilon, self.margin_kcal, self.rescan_factor)


DEFAULT_EXPLORE = ExploreParams()


@dataclass(frozen=True)
class EndState:
    key: str
    reason: str  # SINK_AT_T | TURNOVER | NO_OUTGOING_STEP | EQUILIBRIUM
    probability: float  # of ending there, from the start
    g_rel: float | None  # G - G(start), kcal/mol
    inversion: Inversion | None  # T* of the outcome start -> end


@dataclass
class ExploreResult:
    graph: ImplicitGraph
    chain: MarkovChain
    absorption: Absorption
    absorbing: dict[str, str]
    ends: list[EndState]
    unexplored_mass: float  # UNEXPLORED + CLIMB_CUT + EXPAND_FAILED
    frontier_mass: float  # UNEXPLORED only (the stop criterion)
    reason: str  # MASS_TOL | EXHAUSTED | EXPANSION_BUDGET
    sinks: dict[str, SinkCheck]
    no_net_reaction: str | None
    reactant_share: float | None  # the start's equilibrium share over the explored pools
    equilibrium: dict[str, float]  # Boltzmann over the pools with a free energy
    counts: Counter = field(default_factory=Counter)


class _Run:
    """The state of one exploration: the graph, the caches and the classification."""

    def __init__(self, expander, start, thermo: ThermoModel | None, params: ExploreParams):
        self.graph = ImplicitGraph(expander, start)
        self.params = params
        self.reverse = ReverseCache(expander)
        self.thermo = ThermoCache(thermo, params.temperature_k)
        self.parent: dict[str, str] = {}
        self.rescanned: dict[str, int] = {}
        self.turnover: dict[str, bool] = {}
        self.sinks: dict[str, SinkCheck] = {}
        self.counts: Counter = Counter()

    def chain(self) -> MarkovChain:
        return build_chain(self.graph, self.reverse, self.thermo)

    def expand(self, key: str) -> None:
        self.graph.edges(key)
        self.counts["EXPANDED"] += 1

    def _parents(self) -> None:
        for src, edges in self.graph.out.items():
            for e in edges:
                if e.target != self.graph.root:
                    self.parent.setdefault(e.target, src)

    def _closes_turnover(self, key: str) -> bool:
        if key not in self.turnover:
            path = [key]
            while path[-1] != self.graph.root:
                path.append(self.parent[path[-1]])
            pools = [split_pool_key(k) for k in reversed(path)]
            root = set(split_pool_key(self.graph.root))
            self.turnover[key] = turnover_closure(pools, root) is not None
        return self.turnover[key]

    def _unexpanded(self, chain: MarkovChain, key: str) -> str:
        if self.params.stop_at_turnover and self._closes_turnover(key):
            return TURNOVER
        g = chain.relative_g(key)
        limit = self.params.max_climb_kcal
        return CLIMB_CUT if limit is not None and g is not None and g > limit else UNEXPLORED

    def _pass(self) -> tuple[MarkovChain, dict[str, str], bool]:
        """One classification of every pool; True when a rescan changed the graph."""
        chain, absorbing, before = self.chain(), {}, len(self.rescanned)
        self._parents()
        for key in list(chain.keys):
            if key not in self.graph.out:
                absorbing[key] = self._unexpanded(chain, key)
            elif key in self.graph.failures:
                absorbing[key] = EXPAND_FAILED
            elif not any(e.target != key for e in self.graph.out[key]):
                absorbing[key] = NO_OUTGOING_STEP
            else:
                check, chain = confirm_sink(self.graph, chain, key, self.params.stop,
                                            self.chain, self.rescanned, self.counts)  # fmt: skip
                self.sinks[key] = check
                if check.sink:
                    absorbing[key] = SINK_AT_T
        return chain, absorbing, len(self.rescanned) > before

    def classify(self) -> tuple[MarkovChain, dict[str, str]]:
        """Classify until no rescan changes the graph (each pool is rescanned at most once)."""
        while True:
            chain, absorbing, changed = self._pass()
            if not changed:
                return chain, absorbing


def _ends(chain: MarkovChain, ab: Absorption, absorbing: dict[str, str]) -> list[EndState]:
    root = chain.keys[0]
    reason = {m: ab.reasons[t] for t in ab.targets for m in ab.members[t]}
    out = []
    for key, p in ab.state_probabilities(root).items():
        if reason[key] in NOT_ENDS:
            continue
        a, b = chain.potentials.get(key), chain.potentials.get(root)
        d = None if a is None or b is None else difference(a, b)
        inv = None if d is None else inversion(d.h, d.s)
        out.append(EndState(key, reason[key], p, chain.relative_g(key), inv))
    return sorted(out, key=lambda e: (-e.probability, e.key))


def _result(run: _Run, chain, absorbing, ab: Absorption, reason: str) -> ExploreResult:
    probs = ab.state_probabilities(run.graph.root)
    frontier = sum(p for k, p in probs.items() if absorbing.get(k) == UNEXPLORED)
    cut = sum(p for k, p in probs.items() if absorbing.get(k) in (CLIMB_CUT, EXPAND_FAILED))
    start_sink = absorbing.get(run.graph.root) == SINK_AT_T
    verdict, share = no_net_reaction(chain, start_sink, run.params.stop)
    counts = Counter(run.counts) + chain.counts + run.graph.skips
    counts.update(f"ABSORBING:{r}" for r in absorbing.values())
    eq = boltzmann(chain.potentials, chain.keys, run.params.temperature_k)
    return ExploreResult(run.graph, chain, ab, absorbing, _ends(chain, ab, absorbing),
                         frontier + cut, frontier, reason, dict(run.sinks), verdict, share, eq,
                         counts)  # fmt: skip


def explore(
    expander, start, thermo: ThermoModel | None = None, params: ExploreParams = DEFAULT_EXPLORE
) -> ExploreResult:
    """Best-first exploration by probability mass from `start` (module docstring)."""
    run = _Run(expander, start, thermo, params)
    root = run.graph.root
    run.expand(root)
    while True:
        chain, absorbing = run.classify()
        ab = absorb(chain, absorbing, [root])
        row = ab.state_probabilities(root)
        frontier = [k for k in chain.keys if absorbing.get(k) == UNEXPLORED]
        if not frontier:
            reason = "EXHAUSTED"
        elif sum(row.get(k, 0.0) for k in frontier) <= params.mass_tol:
            reason = "MASS_TOL"
        elif run.graph.expanded >= params.max_expansions:
            reason = "EXPANSION_BUDGET"
        else:
            run.expand(max(frontier, key=lambda k: row.get(k, 0.0)))  # ties: the first found
            continue
        return _result(run, chain, absorbing, ab, reason)


# ---- the pathways to the end states, and the result as JSON-able rows


def pathways_to_ends(res: ExploreResult, n_ends: int = 3, max_paths: int = 5) -> list:
    """[(end state, `tpt.TransitionPaths` from the start to it or None)] for the `n_ends` most
    probable end states other than the start (item I3 on the explored chain)."""
    root = res.graph.root
    ends = [e for e in res.ends if e.key != root][:n_ends]
    return [(e, transition_paths(res.chain, [root], [e.key], max_paths)) for e in ends]


def _opt(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def _state_rows(res: ExploreResult) -> list[dict]:
    out = []
    for n, key in enumerate(res.chain.keys):
        state = res.graph.states[key]
        out.append({"id": n, "key": key, "smiles": getattr(state, "smiles", key),
                    "class": res.absorbing.get(key, "TRANSIENT"),
                    "g_rel": _opt(res.chain.relative_g(key)),
                    "equilibrium": _opt(res.equilibrium.get(key), 6)})  # fmt: skip
    return out


def _edge_rows(res: ExploreResult, ids: dict[str, int]) -> list[dict]:
    out = []
    for e in res.chain.edges:
        inv = e.inversion
        out.append({"source": ids[e.source], "target": ids[e.target], "signature": e.signature,
                    "logp": _opt(e.logp), "reverse_logp": _opt(e.reverse_logp),
                    "dg": _opt(e.dg), "k_eq": e.k_eq, "dh": _opt(e.dh), "ds": _opt(e.ds, 6),
                    "t_star": _opt(inv.t_star, 1) if inv else None,
                    "favoured": inv.favoured if inv else None, "draw": dict(e.draw)})  # fmt: skip
    return out


def predict_rows(res: ExploreResult, tpts: list) -> dict:
    """The exploration as JSON: states, end states, sinks, edges with dG and K, pathways."""
    ids = {k: n for n, k in enumerate(res.chain.keys)}
    ends = [{"state": ids[e.key], "reason": e.reason, "probability": _opt(e.probability, 6),
             "g_rel": _opt(e.g_rel), "t_star": _opt(e.inversion.t_star, 1) if e.inversion
             else None, "favoured": e.inversion.favoured if e.inversion else None}
            for e in res.ends]  # fmt: skip
    sinks = [{"state": ids[c.key], "examined": c.examined, "rescanned": c.rescanned,
              "rescan_proposals": c.rescan_proposals, "lowest_climb": _opt(c.lowest_climb),
              "threshold": _opt(c.threshold)} for c in res.sinks.values() if c.sink]  # fmt: skip
    paths = [{"end": ids[e.key], "total_flux": None if tp is None else tp.total,
              "residual": None if tp is None else _opt(tp.residual, 6),
              "paths": [] if tp is None else [{"states": [ids[k] for k in p.nodes],
                                               "share": _opt(p.share, 6)} for p in tp.pathways]}
             for e, tp in tpts]  # fmt: skip
    return {"reason": res.reason, "expanded": res.graph.expanded,
            "unexplored_mass": _opt(res.unexplored_mass, 6),
            "frontier_mass": _opt(res.frontier_mass, 6), "no_net_reaction": res.no_net_reaction,
            "reactant_share": _opt(res.reactant_share, 6), "states": _state_rows(res),
            "ends": ends, "sinks": sinks, "edges": _edge_rows(res, ids), "pathways": paths,
            "counts": dict(sorted(res.counts.items()))}  # fmt: skip
