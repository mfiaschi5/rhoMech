"""The thermodynamic stop (reaction-network plan, item F; decision D2: it acts on states, never
by cutting an uphill step).

- `StopParams.temperature_k` is the physical temperature in kelvin, distinct from the
  predictor's softmax `temperature`; `epsilon` and `margin_kcal` set the climb threshold
  delta(T) = RT ln(1/epsilon) + margin (`free_energy.climb_threshold`). Their defaults are
  provisional (docs/decisions.md, 2026-11-09): they are calibrated so that no recorded reaction
  comes out a false sink.
- SINK_AT_T (`sink_test`): every step out of an expanded pool climbs more than delta(T). The
  steps out are the chain's links: the pool's proposals and the reverse steps of the edges
  that reach it (a pool above its parent is no sink: its way back descends). A step without a
  dG (NO_FREE_ENERGY) is not known to climb, so a pool with one is no sink. "Every proposed
  step climbs" is not "no step descends", so before a sink is declared `confirm_sink` rescans
  the pool with `rescan_factor` x the candidates (`Rescanner.rescan`, a hook on the expander;
  `ModelRescan` gives a `ModelExpander` one), adds the new steps to the graph (only to children
  it did not have) and tests again; `SinkCheck` reports how many steps were examined. An
  expander without the hook leaves the sink unconfirmed (SINK_NOT_RESCANNED, counted).
- NO_NET_REACTION at T (`no_net_reaction`): the start pool is a sink (START_IS_SINK), or the
  start holds more than 1 - epsilon of the equilibrium (Boltzmann) population over the explored
  pools (REACTANTS_DOMINATE), decided only when every explored pool has a free energy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ksmech.network.chain import MarkovChain, boltzmann
from ksmech.network.free_energy import climb_threshold
from ksmech.network.search import Edge, Expansion, ImplicitGraph

SINK_AT_T = "SINK_AT_T"
SINK_NOT_RESCANNED = "SINK_NOT_RESCANNED"
START_IS_SINK, REACTANTS_DOMINATE = "START_IS_SINK", "REACTANTS_DOMINATE"


@dataclass(frozen=True)
class StopParams:
    temperature_k: float = 298.15  # K (not the predictor's softmax temperature)
    epsilon: float = 0.01  # population share below which a state counts as absent
    margin_kcal: float = 5.0  # provisional: added to RT ln(1/epsilon) for the G error
    rescan_factor: int = 4  # the rescan's candidates, x the expansion's

    @property
    def threshold(self) -> float:
        return climb_threshold(self.temperature_k, self.epsilon, self.margin_kcal)


class Rescanner(Protocol):
    def rescan(self, state: Any, factor: int) -> Expansion: ...


@dataclass(frozen=True)
class SinkCheck:
    key: str
    sink: bool
    why_not: str | None  # NO_STEP | UNKNOWN_DG | WITHIN_DELTA; None for a sink
    examined: int  # steps out tested (proposals, the rescan's included, and reverse steps)
    rescanned: bool
    rescan_proposals: int  # steps the rescan proposed (0 without one)
    lowest_climb: float | None  # the smallest dG out, kcal/mol
    threshold: float  # delta(T), kcal/mol


def climbs(chain: MarkovChain, key: str) -> list[float | None]:
    """dG (kcal/mol) of every link out of `key`, None where a potential is unknown."""
    gi = chain.potentials.get(key)
    out = []
    for target in chain.out_links(key):
        gj = chain.potentials.get(target)
        out.append(None if gi is None or gj is None else gj.g - gi.g)
    return out


def sink_test(chain: MarkovChain, key: str, params: StopParams) -> tuple[bool, str | None, list]:
    """(is a sink, why not, the climbs) of one expanded pool on the chain as it stands."""
    dgs = climbs(chain, key)
    if not dgs:
        return False, "NO_STEP", dgs
    if any(d is None for d in dgs):
        return False, "UNKNOWN_DG", dgs
    if min(dgs) <= params.threshold:
        return False, "WITHIN_DELTA", dgs
    return True, None, dgs


def merge_rescan(graph: ImplicitGraph, key: str, ex: Expansion) -> int:
    """Add a rescan's proposals to the node's edges, one per child the node did not reach yet
    (edges already explored are kept as they are); returns the edges added."""
    graph.skips.update({f"RESCAN:{k}": n for k, n in ex.skips.items()})
    have = {e.target for e in graph.out.get(key, [])}
    added = []
    for p in ex.proposals:
        child = graph.expander.key(p.child)
        if child in have:
            continue
        have.add(child)
        graph.states.setdefault(child, p.child)
        added.append(Edge(key, child, p))
    graph.out[key] = sorted([*graph.out.get(key, []), *added], key=lambda e: -e.logp)
    return len(added)


def _check(chain, key, params, rescanned, n_rescan) -> SinkCheck:
    sink, why, dgs = sink_test(chain, key, params)
    low = min((d for d in dgs if d is not None), default=None)
    return SinkCheck(key, sink, why, len(dgs), rescanned, n_rescan, low, params.threshold)


def confirm_sink(
    graph: ImplicitGraph, chain: MarkovChain, key: str, params: StopParams,
    rebuild: Callable[[], MarkovChain], rescanned: dict[str, int], counts: Counter,
) -> tuple[SinkCheck, MarkovChain]:  # fmt: skip
    """The sink test of one expanded pool, rescanned once before a sink is declared (module
    docstring); `rescanned` (key -> proposals of its rescan) records the pools rescanned, and
    the chain is rebuilt after a rescan."""
    sink, _, _ = sink_test(chain, key, params)
    if not sink:
        return _check(chain, key, params, key in rescanned, rescanned.get(key, 0)), chain
    hook = getattr(graph.expander, "rescan", None)
    if key not in rescanned and hook is not None:
        ex = hook(graph.states[key], params.rescan_factor)
        rescanned[key] = len(ex.proposals)
        counts["RESCANS"] += 1
        if merge_rescan(graph, key, ex):
            chain = rebuild()
    check = _check(chain, key, params, key in rescanned, rescanned.get(key, 0))
    if check.sink and not check.rescanned:
        counts[SINK_NOT_RESCANNED] += 1
    return check, chain


def no_net_reaction(
    chain: MarkovChain, start_is_sink: bool, params: StopParams
) -> tuple[str | None, float | None]:
    """(NO_NET_REACTION's reason or None, the start's equilibrium share or None)."""
    known = all(chain.potentials.get(k) is not None for k in chain.keys)
    share = boltzmann(chain.potentials, chain.keys, params.temperature_k).get(chain.keys[0])
    if not known:
        share = None
    if start_is_sink:
        return START_IS_SINK, share
    if share is not None and share > 1.0 - params.epsilon:
        return REACTANTS_DOMINATE, share
    return None, share


class ModelRescan:
    """A `network.pools.ModelExpander` with the rescan hook: `rescan(state, factor)` decodes
    again with factor x the anchors, products per anchor, applications and search nodes of its
    `DecodeConfig` (the caps on bond and atom changes stay); the rest goes to the expander."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def key(self, state) -> str:
        return self.inner.key(state)

    def expand(self, state) -> Expansion:
        return self.inner.expand(state)

    def reverse_logp(self, proposal):
        return self.inner.reverse_logp(proposal)

    def rescan(self, state, factor: int) -> Expansion:
        cfg = self.inner.cfg
        self.inner.cfg = replace(cfg, anchors=cfg.anchors * factor,
                                 per_anchor=cfg.per_anchor * factor,
                                 max_applied=cfg.max_applied * factor,
                                 max_pops=cfg.max_pops * factor)  # fmt: skip
        try:
            return self.inner.expand(state)
        finally:
            self.inner.cfg = cfg
