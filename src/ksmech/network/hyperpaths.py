"""Hyperpaths on the species graph (reaction-network plan, item J): the shortest, the k shortest,
and their linearization into pool-graph pathways.

The species graph (`network.species`) is read as a B-hypergraph: a step with several products
is one B-arc per distinct product species (tail: the reactant multiset, head: that product,
weight: the step's -log p), and the target is a virtual GOAL node whose arc needs every target
species (each target slot reached EXACT by its identity key, or SKELETON by its heavy-atom
skeleton when that level is allowed). Starting and reservoir species are the sources, of cost 0.

A hyperpath to the target is a B-hyperpath: one arc into every non-source node it holds, the
arcs' tails in it, no cycle; a derivation DAG whose joins are arcs with several tail nodes. Its
cost is the additive weighting of Gallo, Longo, Pallottino and Nguyen (1993): W(v) = w(arc) +
sum over the tail of multiplicity x W(u), so a species needed twice is paid for twice (two
molecules), and a step whose two products are both needed is paid for once per product (one
B-arc each; one event makes both, which `unfold` fires once, so `Unfolding.logp` can be above
-cost: the additive weight is an upper bound there).

- `sbt`: the shortest-B-tree procedure SBT of Gallo et al., a Dijkstra for additive weightings
  with weights >= 0: an arc is examined once every tail node is final, and a node is final when
  it leaves the heap, so nodes become final in non-decreasing W. `shortest_hyperpath` reads the
  hyperpath to the target from its predecessor arcs.
- `k_shortest_hyperpaths`: the k shortest hyperpaths of Nielsen, Andersen and Pretolani (2005),
  Lawler's partition with their branching: for the hyperpath found, with its nodes in reverse
  valid order (the target first, each node before the nodes its arc needs), subproblem i
  forbids node i's arc and fixes the arcs of the nodes before it, which are then in every
  hyperpath of the subproblem; the subproblems partition the hyperpaths not yet found, so they
  come in non-decreasing cost, each once. A hyperpath is a set of steps, so the enumeration
  yields mechanisms up to the order of independent steps, never their permutations.
- `unfold` and `linearize`: a hyperpath as events. Each step fires as often as the needs below
  it require (the target once; a species needed n times by the events above it makes its arc
  fire ceil(n / copies made)), each event consuming instances made by its species' own arc (or
  drawn from the start). Atoms get map numbers from the start instances (`numbering`, e.g. a
  start pool's, `network.lift.start_numbering`, else 1, 2, ...) through each step's atom map,
  all fixed by the unfolding, so any order the event DAG allows is a pool-graph pathway (a list
  of `search.Edge`) with the same steps on the same map numbers, and `search.mechanism_class`
  gives every such order the same class. `species_class` is the order-free form: the multiset
  of steps, by (reactants, products).
"""

from __future__ import annotations

import heapq
import math
import random
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field

from ksmech.network.keys import pool_key_of
from ksmech.network.search import Edge, Proposal
from ksmech.network.species import SpeciesGraph, Step, StepKey
from ksmech.network.target import EXACT, LEVELS, SKELETON

GOAL = "@goal"
INF = math.inf


@dataclass(frozen=True)
class Arc:
    id: str
    tail: tuple[tuple[str, int], ...]  # (node, multiplicity), sorted by node
    head: str
    weight: float  # >= 0
    step: Step | None = None  # None: a virtual arc (a target slot, the goal)
    level: str | None = None  # a target slot's arc: EXACT or SKELETON


class Hypergraph:
    def __init__(self, arcs: Sequence[Arc], sources) -> None:
        self.arcs = {a.id: a for a in arcs}
        if len(self.arcs) != len(arcs):
            raise ValueError("arc ids are unique")
        if any(a.weight < 0 for a in arcs):
            raise ValueError("SBT needs arc weights >= 0")
        self.sources = frozenset(sources)
        self.into: dict[str, list[Arc]] = defaultdict(list)
        self.out: dict[str, list[Arc]] = defaultdict(list)
        for a in arcs:
            self.into[a.head].append(a)
            for u, _ in a.tail:
                self.out[u].append(a)


@dataclass
class Labels:
    cost: dict[str, float]  # W of every node reached
    pred: dict[str, Arc]  # the arc of every non-source node reached
    order: list[str]  # nodes in the order they became final


def sbt(hg: Hypergraph, removed=frozenset(), fixed: Mapping[str, str] | None = None) -> Labels:
    """SBT (module docstring); arcs in `removed` are left out and a node in `fixed` may enter
    only by its fixed arc."""
    fixed = fixed or {}
    usable = lambda a: a.id not in removed and fixed.get(a.head, a.id) == a.id  # noqa: E731
    cost = {s: 0.0 for s in hg.sources}
    heap = [(0.0, s) for s in sorted(hg.sources)]
    pred: dict[str, Arc] = {}
    for a in sorted((a for a in hg.arcs.values() if not a.tail), key=lambda a: a.id):
        if usable(a) and a.head not in hg.sources and a.weight < cost.get(a.head, INF):
            cost[a.head], pred[a.head] = a.weight, a
            heapq.heappush(heap, (a.weight, a.head))
    final: set[str] = set()
    order: list[str] = []
    seen: Counter = Counter()
    while heap:
        w, u = heapq.heappop(heap)
        if u in final or w > cost[u]:
            continue
        final.add(u)
        order.append(u)
        for a in hg.out[u]:
            seen[a.id] += 1
            if seen[a.id] < len(a.tail) or not usable(a) or a.head in final:
                continue
            if a.head in hg.sources:
                continue
            f = a.weight + sum(m * cost[x] for x, m in a.tail)
            if f < cost.get(a.head, INF):
                cost[a.head], pred[a.head] = f, a
                heapq.heappush(heap, (f, a.head))
    return Labels(cost, pred, order)


@dataclass
class Hyperpath:
    target: str
    pred: dict[str, Arc]  # every non-source node it holds -> the arc that makes it
    cost: float  # the additive weight
    order: tuple[str, ...]  # its non-source nodes in valid order (each after its arc's tail)
    sources: frozenset[str]  # the source nodes it needs

    @property
    def steps(self) -> list[Step]:
        """Its distinct steps, in valid order."""
        out: dict[StepKey, Step] = {}
        for v in self.order:
            s = self.pred[v].step
            if s is not None:
                out.setdefault(s.key, s)
        return list(out.values())

    @property
    def arrival(self) -> str | None:
        levels = {a.level for a in self.pred.values() if a.level is not None}
        return SKELETON if SKELETON in levels else EXACT if levels else None


def hyperpath_from(hg: Hypergraph, labels: Labels, target: str) -> Hyperpath | None:
    """The hyperpath to `target` that the labels' predecessor arcs hold; None if unreached."""
    if target not in labels.cost:
        return None
    nodes, sources, stack = set(), set(), [target]
    while stack:
        v = stack.pop()
        if v in hg.sources:
            sources.add(v)
        elif v not in nodes:
            nodes.add(v)
            stack.extend(x for x, _ in labels.pred[v].tail)
    order = tuple(v for v in labels.order if v in nodes)
    pred = {v: labels.pred[v] for v in order}
    return Hyperpath(target, pred, labels.cost[target], order, frozenset(sources))


def shortest_hyperpath(hg: Hypergraph, target: str = GOAL) -> Hyperpath | None:
    return hyperpath_from(hg, sbt(hg), target)


def k_shortest_hyperpaths(
    hg: Hypergraph, k: int, target: str = GOAL,
    distinct: Callable[[Hyperpath], Hashable] | None = None, max_subproblems: int = 100_000,
) -> tuple[list[Hyperpath], Counter]:  # fmt: skip
    """The k shortest hyperpaths to `target` in non-decreasing cost (module docstring), and the
    counts: SUBPROBLEMS solved, SAME_CLASS (a hyperpath whose `distinct` key an earlier one has,
    not reported) and SUBPROBLEM_BUDGET."""
    counts: Counter = Counter()
    first = shortest_hyperpath(hg, target)
    heap = [] if first is None else [(first.cost, 0, first, {}, frozenset())]
    tick, out, seen = 1, [], set()
    while heap and len(out) < k:
        _, _, hp, fixed, removed = heapq.heappop(heap)
        key = distinct(hp) if distinct else id(hp)
        if key in seen:
            counts["SAME_CLASS"] += 1
        else:
            seen.add(key)
            out.append(hp)
        branch = dict(fixed)
        for v in reversed(hp.order):  # consumers first: the target, then what it needs
            if v not in fixed:
                if counts["SUBPROBLEMS"] >= max_subproblems:
                    counts["SUBPROBLEM_BUDGET"] += 1
                    break
                counts["SUBPROBLEMS"] += 1
                cut = removed | {hp.pred[v].id}
                sub = hyperpath_from(hg, sbt(hg, cut, branch), target)
                if sub is not None:
                    heapq.heappush(heap, (sub.cost, tick, sub, dict(branch), cut))
                    tick += 1
            branch[v] = hp.pred[v].id
    return out, counts


def hypergraph_of(
    g: SpeciesGraph, target: Sequence[tuple[str, str]], levels=LEVELS,
    weight: Callable[[Step], float] | None = None,
) -> Hypergraph:  # fmt: skip
    """The species graph as a B-hypergraph towards `target`, its slots (identity key, skeleton
    key) with repeats (module docstring); `weight` defaults to the step's -log p."""
    weight = weight or (lambda s: s.cost)
    arcs = []
    for n, step in enumerate(g.steps.values()):
        tail = tuple(sorted(Counter(step.reactants).items()))
        for p in sorted(set(step.products) - set(step.reactants)):
            arcs.append(Arc(f"s{n}:{p}", tail, p, max(weight(step), 0.0), step))
    goal = []
    for i, (slot, mult) in enumerate(sorted(Counter(target).items())):
        node = f"@target{i}"
        goal.append((node, mult))
        for key, level in slot_matches(g, slot, levels):
            arcs.append(Arc(f"t{i}:{key}", ((key, 1),), node, 0.0, None, level))
    arcs.append(Arc(GOAL, tuple(goal), GOAL, 0.0))
    return Hypergraph(arcs, set(g.sources) | set(g.reservoir))


def slot_matches(g: SpeciesGraph, slot: tuple[str, str], levels=LEVELS) -> list[tuple[str, str]]:
    """(species key, level) of every species of the graph that is a target slot (identity
    key, skeleton key): EXACT by its key, else SKELETON by its skeleton when allowed."""
    key, skeleton = slot
    out = []
    for sp in sorted(g.species.values(), key=lambda s: s.key):
        if sp.key == key and EXACT in levels:
            out.append((sp.key, EXACT))
        elif SKELETON in levels and skeleton and sp.skeleton == skeleton:
            out.append((sp.key, SKELETON))
    return out


# ---- unfolding and linearization


@dataclass(frozen=True)
class Event:
    step: Step
    consumes: tuple[int, ...]  # an instance per reactant slot
    produces: tuple[int, ...]  # an instance per product slot


@dataclass
class Unfolding:
    events: list[Event]
    species: list[str]  # instance id -> species key
    start: list[int]  # the instances drawn from the start, by species then id
    fires: dict[StepKey, int] = field(default_factory=dict)

    @property
    def logp(self) -> float:
        return sum(e.step.logp for e in self.events)

    @property
    def species_class(self) -> tuple[tuple[StepKey, int], ...]:
        return species_class(e.step for e in self.events)


def species_class(steps) -> tuple[tuple[StepKey, int], ...]:
    """The multiset of steps by (reactants, products): a mechanism up to the order of
    independent steps, the choice among molecules of one species and symmetric atoms."""
    return tuple(sorted(Counter(s.key for s in steps).items()))


def _fire_counts(hp: Hyperpath) -> tuple[dict[StepKey, Step], dict[StepKey, int], Counter]:
    """(the steps, how often each fires, the start instances needed per source species)."""
    base: Counter = Counter({hp.target: 1})
    steps: dict[StepKey, Step] = {}
    fires: dict[StepKey, int] = {}

    def need(u: str) -> int:
        return base[u] + sum(n * steps[s].reactants.count(u) for s, n in fires.items())

    for v in reversed(hp.order):  # every consumer of v is done before v
        arc, n = hp.pred[v], need(v)
        if arc.step is None:
            for x, m in arc.tail:
                base[x] += m * n
            continue
        s = arc.step
        steps[s.key] = s
        fires[s.key] = max(fires.get(s.key, 0), -(-n // s.products.count(v)))
    return steps, fires, Counter({u: need(u) for u in sorted(hp.sources) if need(u)})


def unfold(hp: Hyperpath) -> Unfolding:
    """The hyperpath's events (module docstring), in valid order."""
    steps, fires, start = _fire_counts(hp)
    species: list[str] = []
    supply: dict[str, deque] = defaultdict(deque)
    for u, n in start.items():
        for _ in range(n):
            supply[u].append(len(species))
            species.append(u)
    first = list(range(len(species)))
    maker = {v: hp.pred[v].step.key for v in hp.order if hp.pred[v].step is not None}
    events = []
    for key in sorted(steps, key=lambda s: min(i for i, v in enumerate(hp.order)
                                               if maker.get(v) == s)):  # fmt: skip
        step = steps[key]
        for _ in range(fires[key]):
            consumes = tuple(supply[u].popleft() for u in step.reactants)
            produces = []
            for p in step.products:
                produces.append(len(species))
                if maker.get(p) == key:
                    supply[p].append(len(species))
                species.append(p)
            events.append(Event(step, consumes, tuple(produces)))
    return Unfolding(events, species, first, fires)


def _sizes(events: Sequence[Event], species: Sequence[str]) -> dict[int, int]:
    """Atoms of each start instance: the atom-map entries that name its reactant slot."""
    out: dict[int, int] = {}
    for e in events:
        for i, inst in enumerate(e.consumes):
            n = sum(1 for slot in e.step.atom_map for (r, _) in slot if r == i)
            out[inst] = max(out.get(inst, 0), n)
    return out


def _numbers(u: Unfolding, numbering: Mapping[str, Sequence[Sequence[int]]] | None):
    """Map numbers of every instance's atoms by local index: the start instances from
    `numbering` (species -> one tuple per molecule) in order, else and beyond it 1, 2, ..."""
    sizes, given = _sizes(u.events, u.species), Counter()
    nums: dict[int, tuple[int, ...]] = {}
    for inst in u.start:
        rows = (numbering or {}).get(u.species[inst], ())
        if given[u.species[inst]] < len(rows):
            nums[inst] = tuple(rows[given[u.species[inst]]])
            given[u.species[inst]] += 1
    top = max((m for t in nums.values() for m in t), default=0)
    for inst in u.start:
        if inst not in nums:
            nums[inst] = tuple(range(top + 1, top + 1 + sizes.get(inst, 0)))
            top += sizes.get(inst, 0)
    for e in u.events:
        for j, inst in enumerate(e.produces):
            slot = e.step.atom_map[j] if j < len(e.step.atom_map) else ()
            nums[inst] = tuple(nums[e.consumes[i]][a] for i, a in slot)
    return nums


def valid_order(u: Unfolding, rng: random.Random | None = None) -> list[int]:
    """An order of the events their DAG allows (an event after those whose products it takes):
    the earliest ready event first, or a random ready one with `rng`."""
    made = {inst: k for k, e in enumerate(u.events) for inst in e.produces}
    before = [{made[i] for i in e.consumes if i in made} for e in u.events]
    done: list[int] = []
    left = set(range(len(u.events)))
    while left:
        ready = sorted(k for k in left if before[k] <= set(done))
        k = rng.choice(ready) if rng else ready[0]
        done.append(k)
        left.remove(k)
    return done


def linearize(
    hp: Hyperpath | Unfolding, order: Sequence[int] | None = None,
    numbering: Mapping[str, Sequence[Sequence[int]]] | None = None,
) -> list[Edge]:  # fmt: skip
    """The hyperpath as a pool-graph pathway, its events in `order` (indices of `unfold`'s
    events; a valid order by default; ValueError when the DAG does not allow it)."""
    from ksmech.network.pools import signature

    u = hp if isinstance(hp, Unfolding) else unfold(hp)
    order = list(order) if order is not None else valid_order(u)
    made = {inst: k for k, e in enumerate(u.events) for inst in e.produces}
    if sorted(order) != list(range(len(u.events))):
        raise ValueError("an order holds every event once")
    nums, at = _numbers(u, numbering), {k: n for n, k in enumerate(order)}
    pool = Counter(u.species[i] for i in u.start)
    edges = []
    for k in order:
        e = u.events[k]
        if any(i in made and at[made[i]] > at[k] for i in e.consumes):
            raise ValueError(f"event {k} comes before an event it needs")
        bonds = [(nums[e.consumes[i]][a], nums[e.consumes[j]][b], d)
                 for i, a, j, b, d in e.step.bonds]  # fmt: skip
        lone = [(nums[e.consumes[i]][a], d) for i, a, d in e.step.lone]
        text, touched = signature(bonds, lone, _Identity())
        before = pool_key_of(pool.elements())
        pool = pool - Counter(e.step.reactants) + Counter(e.step.products)
        after = pool_key_of(pool.elements())
        info = {"step": e.step.key, "event": k, "bonds": tuple(sorted(bonds)), "lone": tuple(lone)}
        edges.append(Edge(before, after, Proposal(after, e.step.logp, text, touched, info)))
    return edges


class _Identity(dict):
    """map number -> itself, for `pools.signature` on steps already on map numbers."""

    def __missing__(self, key: int) -> int:
        return key
