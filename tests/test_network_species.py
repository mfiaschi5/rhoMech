"""The species graph, its hyperpaths and its atom flux on small synthetic graphs (reaction-network
plan, item J), brute force as the oracle: SBT's best cost and the k shortest hyperpaths equal
exhaustive enumeration, atom flux is conserved at every species, a hyperpath linearized in any
valid order has one mechanism class, a join is one hyperpath but several pool-graph paths, and
the species-by-species expansion stops at its tolerance and counts its cuts."""

import itertools
import math
import random
from collections import Counter

import pytest

from ksmech.network.flux import (
    START,
    atom_balance,
    atom_pathways,
    balances,
    masses,
    merged_pathway,
    step_fluxes,
)
from ksmech.network.hyperpaths import (
    GOAL,
    Arc,
    Hypergraph,
    k_shortest_hyperpaths,
    linearize,
    sbt,
    shortest_hyperpath,
    species_class,
    unfold,
    valid_order,
)
from ksmech.network.pools import signature
from ksmech.network.search import (
    Expansion,
    ImplicitGraph,
    Proposal,
    SearchParams,
    k_best_pathways,
    mechanism_class,
    mechanism_classes,
)
from ksmech.network.species import (
    BELOW_TOLERANCE,
    LEFT_ABOVE_TOLERANCE,
    PARTNER_ONLY,
    POOL_BUDGET,
    SELF_LOOP,
    SPECIES_BUDGET,
    TOLERANCE,
    ExpansionParams,
    PoolResult,
    Species,
    SpeciesGraph,
    Step,
    expand_species,
)

# ---- brute force over B-hyperpaths


def _cost(hg, pred, v, memo, stack=()):
    """The additive weight of v under `pred`, None on a cycle."""
    if v in hg.sources:
        return 0.0
    if v in stack:
        return None
    if v not in memo:
        arc = pred[v]
        parts = [_cost(hg, pred, x, memo, stack + (v,)) for x, _ in arc.tail]
        if any(p is None for p in parts):
            return None
        memo[v] = arc.weight + sum(m * p for (_, m), p in zip(arc.tail, parts, strict=True))
    return memo[v]


def all_hyperpaths(hg, target):
    """(cost, frozenset of (node, arc id)) of every B-hyperpath to `target`."""
    out = []

    def rec(pred, todo):
        todo = [v for v in todo if v not in pred and v not in hg.sources]
        if not todo:
            cost = _cost(hg, pred, target, {})
            if cost is not None:
                out.append((cost, frozenset((v, a.id) for v, a in pred.items())))
            return
        v = min(todo)
        for a in hg.into.get(v, []):
            rec({**pred, v: a}, todo + [x for x, _ in a.tail])

    rec({}, [target])
    return sorted(out, key=lambda x: x[0])


def _random_hypergraph(seed, n=7):
    rng = random.Random(seed)
    names = [f"n{i}" for i in range(n)]
    arcs = []
    for v in names[2:]:
        for _ in range(rng.randint(2, 4)):
            others = [x for x in names if x != v]
            tail = Counter(rng.sample(others, rng.randint(1, 2)))
            if rng.random() < 0.2:
                tail[next(iter(tail))] += 1  # a species needed twice
            arcs.append(Arc(f"a{len(arcs)}", tuple(sorted(tail.items())), v,
                            round(rng.uniform(0.1, 2.0), 3)))  # fmt: skip
    return Hypergraph(arcs, {"n0", "n1"})


@pytest.mark.parametrize("seed", range(30))
def test_sbt_equals_the_cheapest_enumerated_hyperpath(seed):
    hg = _random_hypergraph(seed)
    every = all_hyperpaths(hg, "n6")
    hp = shortest_hyperpath(hg, "n6")
    if not every:
        assert hp is None
        return
    assert hp.cost == pytest.approx(every[0][0])
    assert _cost(hg, hp.pred, "n6", {}) == pytest.approx(hp.cost)  # its own arcs give its cost
    labels = sbt(hg)
    assert all(labels.cost[u] <= labels.cost[v] + 1e-12
               for u, v in itertools.pairwise(labels.order))  # fmt: skip


@pytest.mark.parametrize("seed", range(30))
def test_the_k_shortest_hyperpaths_are_the_enumeration_in_order(seed):
    hg = _random_hypergraph(seed)
    every = all_hyperpaths(hg, "n6")
    got, counts = k_shortest_hyperpaths(hg, 8, "n6")
    assert [h.cost for h in got] == pytest.approx([c for c, _ in every[:8]])
    sets = [frozenset((v, a.id) for v, a in h.pred.items()) for h in got]
    assert len(set(sets)) == len(sets) and set(sets) <= {s for _, s in every}
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(h.cost for h in got))
    if len(every) > 8:
        assert counts["SUBPROBLEMS"] > 0


# ---- a small chemistry of abstract species


def _step(reac, prods, logp, bonds=(), lone=()):
    """A `Step` from reactants [(key, atoms)] and products [(key, [(reactant index, atom)])],
    sorted as a step is; bonds (ri, a, rj, b, d) and lone (ri, a, d) on the given order."""
    order = sorted(range(len(reac)), key=lambda i: reac[i][0])
    slot = {i: n for n, i in enumerate(order)}
    pord = sorted(range(len(prods)), key=lambda j: prods[j][0])
    amap = tuple(tuple((slot[i], a) for i, a in prods[j][1]) for j in pord)
    bonds = tuple(sorted((slot[i], a, slot[j], b, d) for i, a, j, b, d in bonds))
    lone = tuple(sorted((slot[i], a, d) for i, a, d in lone))
    return Step(tuple(reac[i][0] for i in order), tuple(prods[j][0] for j in pord), logp, amap,
                bonds, lone)  # fmt: skip


def _graph(steps, sizes, sources, reservoir=()):
    g = SpeciesGraph({k: Species(k, n) for k, n in sizes.items()}, dict(sources),
                     tuple(reservoir))  # fmt: skip
    for s in steps:
        g.add_step(s)
    return g


def _join():
    """A -> B, C -> D, B + D -> E; B + D bonds atom 1 of B to atom 0 of D, the atoms the two
    branches changed."""
    ab = _step([("A", 2)], [("B", [(0, 0), (0, 1)])], math.log(0.6), lone=[(0, 1, -2)])
    cd = _step([("C", 2)], [("D", [(0, 0), (0, 1)])], math.log(0.5), lone=[(0, 0, 2)])
    be = _step([("B", 2), ("D", 2)], [("E", [(0, 0), (0, 1), (1, 0), (1, 1)])], math.log(0.4),
               bonds=[(0, 1, 1, 0, 1)])  # fmt: skip
    sizes = {"A": 2, "B": 2, "C": 2, "D": 2, "E": 4}
    return _graph([ab, cd, be], sizes, {"A": 1, "C": 1})


def _target_hypergraph(g, target):
    from ksmech.network.hyperpaths import hypergraph_of

    return hypergraph_of(g, [(target, "")])


class PoolGraph:
    """The pool graph of a species graph: a state is a sorted tuple of molecules (species, map
    numbers by local atom); every step whose reactants are in the pool applies to the first
    molecules of each species, its products numbered through its atom map."""

    def __init__(self, g):
        self.g = g

    def key(self, state):
        return state

    def expand(self, state):
        out = []
        for step in self.g.steps.values():
            pool = list(state)
            taken = []
            for r in step.reactants:
                hit = next((m for m in pool if m[0] == r), None)
                if hit is None:
                    break
                pool.remove(hit)
                taken.append(hit)
            else:
                prods = [(p, tuple(taken[i][1][a] for i, a in step.atom_map[j]))
                         for j, p in enumerate(step.products)]  # fmt: skip
                bonds = [(taken[i][1][a], taken[j][1][b], d) for i, a, j, b, d in step.bonds]
                lone = [(taken[i][1][a], d) for i, a, d in step.lone]
                text, touched = signature(bonds, lone, {m: m for m in range(1, 99)})
                out.append(Proposal(tuple(sorted(pool + prods)), step.logp, text, touched,
                                    {"step": step.key}))  # fmt: skip
        return Expansion(out)


def test_a_join_is_one_hyperpath_and_several_interleavings():
    g = _join()
    hps, _ = k_shortest_hyperpaths(_target_hypergraph(g, "E"), 10)
    assert len(hps) == 1 and hps[0].arrival == "EXACT"
    assert hps[0].cost == pytest.approx(-math.log(0.6 * 0.5 * 0.4))
    start = (("A", (1, 2)), ("C", (3, 4)))
    res = k_best_pathways(ImplicitGraph(PoolGraph(g), start),
                          lambda s: "EXACT" if any(m[0] == "E" for m in s) else None,
                          params=SearchParams(k=10))  # fmt: skip
    assert len(res.pathways) == 2  # A -> B first, or C -> D first
    classes = mechanism_classes(res.pathways)
    assert len(classes) == 1 and classes[0].probability == pytest.approx(1.0)
    lin = linearize(hps[0], numbering={"A": [(1, 2)], "C": [(3, 4)]})
    assert mechanism_class(lin) == classes[0].steps == ("|2:-2", "|3:+2", "2-3:+1|")
    lifted = [g.steps[e.proposal.info["step"]] for e in res.pathways[0].edges]
    assert species_class(lifted) == unfold(hps[0]).species_class


def test_any_valid_order_of_a_hyperpath_has_one_mechanism_class():
    g = _join()
    ef = _step([("E", 4)], [("F", [(0, 2), (0, 3)]), ("G", [(0, 0), (0, 1)])], math.log(0.9),
               bonds=[(0, 1, 0, 2, -1)])  # fmt: skip
    gh = _step([("C", 2)], [("H", [(0, 1), (0, 0)])], math.log(0.3), lone=[(0, 0, 2)])
    fh = _step([("F", 2), ("H", 2)], [("T", [(0, 0), (0, 1), (1, 0), (1, 1)])], math.log(0.5),
               bonds=[(0, 0, 1, 1, 1)])  # fmt: skip
    for s in (ef, gh, fh):
        g.add_step(s)
    g.species.update({k: Species(k, n) for k, n in {"F": 2, "G": 2, "H": 2, "T": 4}.items()})
    g.sources["C"] = 2  # the second C becomes H
    hp = shortest_hyperpath(_target_hypergraph(g, "T"))
    u = unfold(hp)
    assert len(u.events) == 6 and Counter(u.species[i] for i in u.start) == {"A": 1, "C": 2}
    classes, orders = set(), set()
    for seed in range(20):
        order = valid_order(u, random.Random(seed))
        orders.add(tuple(order))
        classes.add(mechanism_class(linearize(u, order)))
    assert len(orders) >= 3 and len(classes) == 1
    with pytest.raises(ValueError):
        linearize(u, list(reversed(valid_order(u))))


def test_a_species_needed_twice_fires_its_step_twice():
    ab = _step([("A", 1)], [("B", [(0, 0)])], math.log(0.5), lone=[(0, 0, 2)])
    bb = _step([("B", 1), ("B", 1)], [("C", [(0, 0), (1, 0)])], math.log(0.5),
               bonds=[(0, 0, 1, 0, 1)])  # fmt: skip
    g = _graph([ab, bb], {"A": 1, "B": 1, "C": 2}, {"A": 2})
    hp = shortest_hyperpath(_target_hypergraph(g, "C"))
    assert hp.cost == pytest.approx(-3 * math.log(0.5))  # B twice, then C
    u = unfold(hp)
    assert u.fires == {ab.key: 2, bb.key: 1} and u.logp == pytest.approx(3 * math.log(0.5))
    edges = linearize(u)
    assert [e.proposal.signature for e in edges] == ["|1:+2", "|2:+2", "1-2:+1|"]


# ---- the expansion and the atom flux


class ChainReactor:
    """Species S0, S1, ... of one atom each: alone, Sn -> Sn+1 with probability `p`, and
    every pool proposes its partners' own steps too (PARTNER_ONLY); S0 + S0 -> S0 + S0 is a
    degenerate exchange (SELF_LOOP) on the S0 pair."""

    def __init__(self, p=0.5, swap=0.0):
        self.p, self.swap, self.calls = p, swap, []

    def react(self, pool):
        self.calls.append(tuple(s.key for s in pool))
        steps, made = [], {}
        for sp in dict.fromkeys(pool):  # one step per species, as the decoder merges by child
            n = int(sp.key[1:]) + 1
            steps.append(_step([(sp.key, 1)], [(f"S{n}", [(0, 0)])], math.log(self.p)))
            made[f"S{n}"] = Species(f"S{n}", 1)
        if self.swap and [s.key for s in pool] == ["S0", "S0"]:
            steps.append(_step([("S0", 1), ("S0", 1)], [("S0", [(1, 0)]), ("S0", [(0, 0)])],
                               math.log(self.swap)))  # fmt: skip
        return PoolResult(steps, made, Counter({"SEEN": 1}))


def _chain(params, reactor=None):
    reactor = reactor or ChainReactor()
    g = expand_species(reactor, {"S0": Species("S0", 1)}, {"S0": 1.0}, params=params)
    return g, reactor


def test_the_expansion_stops_at_its_tolerance_and_counts_its_cuts():
    g, reactor = _chain(ExpansionParams(tolerance=0.2, partner_weight=0.0, max_pools=50))
    assert g.reason == TOLERANCE and g.core == ["S0", "S1", "S2"]
    assert [g.mass[k] for k in ("S0", "S1", "S2", "S3")] == pytest.approx([1, 0.5, 0.25, 0.125])
    assert g.cuts[BELOW_TOLERANCE] == 1 and g.pools == 2 + 3 + 4 == len(reactor.calls)
    assert reactor.calls[2:5] == [("S1",), ("S1", "S1"), ("S1", "S0")]
    assert g.cuts[PARTNER_ONLY] == 1 + 2 and g.skips["SEEN"] == g.pools  # (S1, S0): S0's step
    assert g.leftover["S0"] == pytest.approx(0.5)
    looser, _ = _chain(ExpansionParams(tolerance=0.1, partner_weight=0.0))
    assert looser.core == ["S0", "S1", "S2", "S3"] and looser.cuts[BELOW_TOLERANCE] == 1


def test_the_expansion_budgets_are_cuts():
    g, _ = _chain(ExpansionParams(tolerance=0.2, partner_weight=0.0, max_pools=4))
    assert g.reason == POOL_BUDGET and g.cuts[POOL_BUDGET] == 1  # S1 + S0 not expanded
    assert g.cuts[LEFT_ABOVE_TOLERANCE] == 1  # S2 (0.25) left
    g2, _ = _chain(ExpansionParams(tolerance=0.01, partner_weight=0.0, max_species=2))
    assert g2.reason == SPECIES_BUDGET and g2.core == ["S0", "S1"]


def test_a_degenerate_exchange_returns_its_carrier():
    reactor = ChainReactor(p=0.4, swap=0.5)
    g, _ = _chain(ExpansionParams(tolerance=0.9, partner_weight=1.0), reactor)
    # S0 alone (w 1/2): S1 with 0.4; S0 + S0 (w 1/2): S1 with 0.4 and the swap with 0.5
    assert g.cuts[SELF_LOOP] == 1
    beta = (0.5 * 0.4 + 0.5 * 0.4) / (1 - 0.5 * 0.5)
    assert g.branching["S0"][(("S0",), ("S1",))] == pytest.approx(beta)
    assert g.mass["S1"] == pytest.approx(beta)


def _random_reactor_graph(seed, cycles=False):
    """An expansion over random splits, isomerizations and joins of abstract species; with
    `cycles`, an isomer may turn back into the species it came from."""
    rng = random.Random(seed)
    size: dict[str, int] = {"A": 3, "B": 2}

    def isomer(key):
        if cycles and "~" in key and rng.random() < 0.5:
            return key.rsplit("~", 1)[0]
        return f"{key}~{rng.randint(0, 2)}"

    def react(pool):
        x, partners = pool[0], pool[1:]
        out, made = [], {}
        n = size[x.key]
        y = isomer(x.key)
        made[y] = n
        out.append(_step([(x.key, n)], [(y, [(0, a) for a in rng.sample(range(n), n)])],
                         math.log(rng.uniform(0.05, 0.4))))  # fmt: skip
        if n >= 2:
            k = rng.randint(1, n - 1)
            atoms = rng.sample(range(n), n)
            a, b = f"{x.key}<{k}", f"{x.key}>{n - k}"
            made.update({a: k, b: n - k})
            out.append(_step([(x.key, n)], [(a, [(0, i) for i in atoms[:k]]),
                                            (b, [(0, i) for i in atoms[k:]])],
                             math.log(rng.uniform(0.05, 0.3))))  # fmt: skip
        for c in partners:
            m = size[c.key]
            j = f"({x.key}+{c.key})"
            made[j] = n + m
            out.append(_step([(x.key, n), (c.key, m)],
                             [(j, [(0, i) for i in range(n)] + [(1, i) for i in range(m)])],
                             math.log(rng.uniform(0.05, 0.3))))  # fmt: skip
        for k, v in made.items():
            size.setdefault(k, v)
        return PoolResult(out, {k: Species(k, size[k]) for k in made})

    class R:
        def react(self, pool):
            return react(pool)

    start = {"A": Species("A", 3), "B": Species("B", 2)}
    return expand_species(R(), start, {"A": 1.0, "B": 2.0},
                          params=ExpansionParams(tolerance=0.08, max_species=6))  # fmt: skip


@pytest.mark.parametrize("seed", range(8))
def test_atom_flux_is_conserved_at_every_species(seed):
    g = _random_reactor_graph(seed, cycles=seed % 2 == 1)
    mass, _ = masses(g)
    flux = step_fluxes(g, mass)
    assert len(g.core) >= 2 and any(len(s.reactants) == 2 for s in g.steps.values())
    for k, b in balances(g, mass).items():
        assert b.produced + b.supplied == pytest.approx(b.consumed + b.terminal, abs=1e-9)
        assert b.terminal >= -1e-9
        assert b.produced + float(g.sources.get(k, 0.0)) == pytest.approx(mass[k], abs=1e-9)
    for node, (inflow, outflow) in atom_balance(g, mass).items():
        assert inflow == pytest.approx(outflow, abs=1e-9), node
    assert all(f >= 0 for f in flux.values())


@pytest.mark.parametrize("seed", range(4))
def test_every_atom_of_a_product_traces_back_to_a_supply(seed):
    g = _random_reactor_graph(seed)
    mass, _ = masses(g)
    target = max((k for k in g.species if k not in g.sources), key=lambda k: (mass[k], k))
    paths = atom_pathways(g, mass, target, per_atom=50)
    for a in range(g.species[target].n_atoms):
        mine = [p for p in paths if p.atom == a]
        assert mine and all(x.share >= y.share for x, y in itertools.pairwise(mine))
        assert sum(p.share for p in mine) <= 1 + 1e-9
        if len(mine) < 50:  # every backward path listed (the graph is acyclic): they sum to 1
            assert sum(p.share for p in mine) == pytest.approx(1.0)
        assert all(p.nodes[0] == (target, a) and len(p.steps) == len(p.nodes) - 1 for p in mine)
    merged = merged_pathway(paths)
    assert set(merged.atom_shares) == set(range(g.species[target].n_atoms))
    assert 0 < merged.share <= 1 and set(merged.steps) <= set(g.steps)


def test_atom_shares_follow_the_step_fluxes():
    """Two steps make B, from A (0.3) and from C (0.1): B's atom came from A with 3/4."""
    ab = _step([("A", 1)], [("B", [(0, 0)])], math.log(0.3))
    cb = _step([("C", 1)], [("B", [(0, 0)])], math.log(0.1))
    g = _graph([ab, cb], {"A": 1, "B": 1, "C": 1}, {"A": 1.0, "C": 1.0})
    g.core, g.branching = ["A", "C"], {"A": {ab.key: 0.3}, "C": {cb.key: 0.1}}
    mass, _ = masses(g)
    paths = atom_pathways(g, mass, "B", per_atom=5)
    assert [(p.steps, p.origin) for p in paths] == [((ab.key,), START), ((cb.key,), START)]
    assert [p.share for p in paths] == pytest.approx([0.75, 0.25])
    assert merged_pathway(paths).steps == (ab.key,)


def test_the_target_arcs_take_a_skeleton_when_allowed():
    from ksmech.network.hyperpaths import hypergraph_of

    ab = _step([("A", 1)], [("B", [(0, 0)])], math.log(0.5))
    g = _graph([ab], {"A": 1, "B": 1}, {"A": 1.0})
    g.species["B"] = Species("B", 1, skeleton="sk")
    assert shortest_hyperpath(hypergraph_of(g, [("Bprime", "sk")], levels=("EXACT",))) is None
    hp = shortest_hyperpath(hypergraph_of(g, [("Bprime", "sk")]))
    assert hp.arrival == "SKELETON" and [s.key for s in hp.steps] == [ab.key]
    assert shortest_hyperpath(hypergraph_of(g, [("A", "")])).steps == []  # the start is the goal
    assert hp.target == GOAL
