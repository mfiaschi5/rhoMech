"""The network's Markov chain, thermodynamic stop, transition path theory and exploration by
probability mass (reaction-network plan, items E, F, I3, I4), as laws on small synthetic graphs
with a stub expander and synthetic free energies: the stationary distribution equals Boltzmann,
absorption probabilities sum to 1, the reactive flux is conserved, the committor is bounded, a
sink is declared only after the rescan, T* has the sign of the change, and the exploration stops
at its mass tolerance."""

import math
import random
from collections import Counter

import pytest

from ksmech.network.chain import (
    EQUILIBRIUM,
    ReverseCache,
    ThermoCache,
    absorb,
    boltzmann,
    build_chain,
    stationary,
)
from ksmech.network.explore import (
    CLIMB_CUT,
    NO_OUTGOING_STEP,
    TURNOVER,
    UNEXPLORED,
    ExploreParams,
    explore,
    pathways_to_ends,
    predict_rows,
)
from ksmech.network.free_energy import (
    R_KCAL,
    Thermo,
    ThermoModel,
    as_thermo,
    climb_threshold,
    inversion,
    load_thermo_model,
)
from ksmech.network.reservoir import RESERVOIR_DRAW
from ksmech.network.search import Expansion, ImplicitGraph, Proposal
from ksmech.network.stop import SINK_AT_T, START_IS_SINK, StopParams
from ksmech.network.tpt import transition_paths


class Stub:
    """An explicit digraph as an expander that also scores reverse steps: node -> [(child, log p,
    reverse log p or None, reservoir draw)]; `extra` holds what a rescan adds."""

    def __init__(self, edges, extra=None, rescan=True):
        self.edges, self.extra, self.calls = edges, extra or {}, Counter()
        if not rescan:
            self.rescan = None

    def key(self, state):
        return state

    def _proposals(self, state, rows):
        out = []
        for child, lp, back, *draw in rows:
            info = {"back": back, **({RESERVOIR_DRAW: draw[0]} if draw else {})}
            out.append(Proposal(child, lp, f"{state}>{child}", frozenset(), info))
        return out

    def expand(self, state):
        self.calls["expand"] += 1
        return Expansion(self._proposals(state, self.edges.get(state, [])))

    def rescan(self, state, factor):
        self.calls["rescan"] += 1
        rows = self.edges.get(state, []) + self.extra.get(state, [])
        return Expansion(self._proposals(state, rows), Counter({"WIDE": factor}))

    def reverse_logp(self, proposal):
        back = proposal.info["back"]
        return "MISSING_TOKENS" if back is None else back


def free_energies(g):
    return ThermoModel(lambda state, t: g.get(state))


def _random_graph(seed, n=7, extra=6):
    """A connected random graph, each edge with a forward and a reverse log p, and G per node."""
    rng = random.Random(seed)
    names = [f"n{i}" for i in range(n)]
    edges = {a: [] for a in names}
    pairs = [(names[rng.randrange(i)], names[i]) for i in range(1, n)]  # a spanning tree
    pairs += [tuple(rng.sample(names, 2)) for _ in range(extra)]
    for a, b in pairs:
        edges[a].append((b, -rng.uniform(0.1, 4.0), -rng.uniform(0.1, 4.0)))
    g = {x: rng.uniform(-6.0, 6.0) for x in names}
    return edges, g


def _explored(edges, start="n0"):
    graph = ImplicitGraph(Stub(edges), start)
    todo = [graph.root]
    while todo:
        for e in graph.edges(todo.pop()):
            if e.target not in graph.out and e.target not in todo:
                todo.append(e.target)
    return graph


def _chain(graph, thermo, t=298.15):
    return build_chain(graph, ReverseCache(graph.expander), ThermoCache(thermo, t))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("t", [250.0, 400.0])
def test_the_stationary_distribution_equals_boltzmann(seed, t):
    edges, g = _random_graph(seed)
    chain = _chain(_explored(edges), free_energies(g), t)
    assert not chain.unbalanced and chain.counts["NO_FREE_ENERGY"] == 0
    pi = stationary(chain)
    want = boltzmann(chain.potentials, chain.keys, t)
    assert pi == pytest.approx(want, rel=1e-6, abs=1e-12)
    w = chain.rates.toarray()
    for i in range(len(chain.keys)):  # detailed balance pair by pair
        for j in range(len(chain.keys)):
            a, b = pi[chain.keys[i]] * w[i, j], pi[chain.keys[j]] * w[j, i]
            assert a == pytest.approx(b, rel=1e-9, abs=1e-15)


def test_edges_record_dg_and_k_and_the_pair_weights_are_symmetric():
    edges = {"a": [("b", math.log(0.5), math.log(0.2))]}
    chain = _chain(_explored(edges, "a"), free_energies({"a": 0.0, "b": -2.0}))
    (rec,) = chain.edges
    assert rec.dg == pytest.approx(-2.0) and rec.reverse_logp == pytest.approx(math.log(0.2))
    rt = R_KCAL * 298.15
    assert rec.k_eq == pytest.approx(math.exp(2.0 / rt))
    w = chain.rates.toarray()
    s = math.sqrt(0.5 * 0.2)
    assert w[0, 1] == pytest.approx(s * math.exp(1.0 / rt))
    assert w[1, 0] == pytest.approx(s * math.exp(-1.0 / rt))


def test_without_free_energies_the_chain_runs_on_the_model():
    edges = {"a": [("b", math.log(0.5), math.log(0.2)), ("c", math.log(0.3), None)]}
    chain = _chain(_explored(edges, "a"), None)
    w = chain.rates.toarray()
    i = chain.index
    assert w[i["a"], i["b"]] == pytest.approx(0.5) and w[i["b"], i["a"]] == pytest.approx(0.2)
    assert w[i["a"], i["c"]] == pytest.approx(0.3) and w[i["c"], i["a"]] == 0.0
    assert chain.counts["NO_FREE_ENERGY"] == 3 and chain.counts["NO_FREE_ENERGY_LINK"] == 2
    assert chain.counts["REVERSE_MISSING:MISSING_TOKENS"] == 1  # c failed featurization
    assert len(chain.unbalanced) == 2
    one_way = _chain(_explored(edges, "a"), free_energies({"a": 0.0, "b": 1.0, "c": 1.0}))
    assert one_way.counts["ONE_WAY"] == 1 and one_way.rates[i["c"], i["a"]] == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_absorption_probabilities_sum_to_one_per_transient_state(seed):
    edges, g = _random_graph(seed, n=9, extra=8)
    chain = _chain(_explored(edges), free_energies(g))
    absorbing = {"n7": "SINK_AT_T", "n8": "NO_OUTGOING_STEP"}
    ab = absorb(chain, absorbing)
    for key, row in ab.rows.items():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= p <= 1.0 for p in row.values())
        if key in absorbing:
            assert row == {key: 1.0}
    probs = ab.state_probabilities("n0")
    assert sum(probs.values()) == pytest.approx(1.0) and set(probs) <= set(absorbing)


def test_a_closed_region_is_an_equilibrium_class():
    # x -> s (an absorbing sink) both ways; x -> p one way (p cannot be featurized); p <-> q.
    edges = {"x": [("s", -1.0, -3.0), ("p", -1.0, None)], "p": [("q", -0.5, -0.5)]}
    g = {"x": 0.0, "s": -5.0, "p": -1.0, "q": -2.0}
    chain = _chain(_explored(edges, "x"), free_energies(g))
    ab = absorb(chain, {"s": SINK_AT_T})
    (cls,) = [t for t in ab.targets if t.startswith(EQUILIBRIUM)]
    assert set(ab.members[cls]) == {"p", "q"}
    assert ab.members[cls] == pytest.approx(boltzmann(chain.potentials, ["p", "q"], 298.15))
    for row in ab.rows.values():
        assert sum(row.values()) == pytest.approx(1.0)
    assert 0.0 < ab.rows["x"][cls] < 1.0 and ab.rows["p"] == {cls: 1.0}


@pytest.mark.parametrize("seed", range(6))
def test_tpt_conserves_flux_bounds_the_committor_and_shares_add_up(seed):
    edges, g = _random_graph(seed, n=9, extra=10)
    chain = _chain(_explored(edges), free_energies(g))
    tp = transition_paths(chain, ["n0"], ["n8"], max_paths=1000)
    assert tp is not None and tp.total > 0
    assert tp.q_plus["n0"] == 0.0 and tp.q_plus["n8"] == 1.0
    assert all(0.0 <= q <= 1.0 for q in tp.q_plus.values())
    for k in tp.keys:  # balanced: q- = 1 - q+
        assert tp.q_minus[k] == pytest.approx(1.0 - tp.q_plus[k], abs=1e-9)
    div = Counter()
    for (i, j), f in tp.net_flux.items():
        div[i] += f
        div[j] -= f
    for k in tp.keys:
        if k not in ("n0", "n8"):
            assert div[k] == pytest.approx(0.0, abs=1e-9 * tp.total)
    assert div["n0"] == pytest.approx(tp.total) and -div["n8"] == pytest.approx(tp.total)
    assert sum(p.share for p in tp.pathways) == pytest.approx(1.0, abs=1e-9)
    assert tp.residual == pytest.approx(0.0, abs=1e-9)
    assert all(p.nodes[0] == "n0" and p.nodes[-1] == "n8" for p in tp.pathways)
    shares = [p.share for p in tp.pathways]
    assert shares[0] == max(shares)


def test_tpt_without_a_reachable_product_is_none():
    chain = _chain(_explored({"a": [("b", -1.0, -1.0)]}, "a"), None)
    assert transition_paths(chain, ["a"], ["zzz"]) is None


def _params(**kw):
    base = dict(margin_kcal=0.0, max_expansions=20, mass_tol=1e-6)
    return ExploreParams(**{**base, **kw})


def test_a_downhill_single_step_is_not_a_sink():
    edges = {"A": [("B", -0.2, -0.3)], "B": [("A", -0.3, -0.2)]}
    res = explore(Stub(edges), "A", free_energies({"A": 0.0, "B": -6.0}), _params())
    assert not res.sinks["A"].sink and res.sinks["A"].why_not == "WITHIN_DELTA"
    assert res.sinks["B"].sink and res.absorbing == {"B": SINK_AT_T}
    assert [(e.key, e.reason) for e in res.ends] == [("B", SINK_AT_T)]
    assert res.ends[0].probability == pytest.approx(1.0) and res.ends[0].g_rel == -6.0
    assert res.no_net_reaction is None and res.reactant_share < 0.01
    assert res.graph.expander.calls["rescan"] == 1  # B, before it was declared a sink


def test_an_all_uphill_pool_is_a_sink_only_after_the_rescan():
    uphill = {"A": [("B", -0.1, -0.1), ("C", -0.5, -0.5)]}
    g = {"A": 0.0, "B": 9.0, "C": 7.0, "D": -3.0}
    stub = Stub(uphill)
    res = explore(stub, "A", free_energies(g), _params())
    check = res.sinks["A"]
    assert stub.calls["rescan"] == 1 and check.rescanned and check.sink
    assert check.examined == 2 and check.rescan_proposals == 2
    assert check.lowest_climb == pytest.approx(7.0)
    assert check.threshold == pytest.approx(climb_threshold(298.15, 0.01, 0.0))
    assert res.no_net_reaction == START_IS_SINK and res.absorbing["A"] == SINK_AT_T
    assert res.graph.expanded == 1 and res.unexplored_mass == 0.0
    found = Stub(uphill, extra={"A": [("D", -6.0, -0.2)]})
    res2 = explore(found, "A", free_energies(g), _params())
    assert not res2.sinks["A"].sink and res2.sinks["A"].examined == 3
    assert res2.no_net_reaction is None and "D" in res2.graph.out
    blind = explore(Stub(uphill, rescan=False), "A", free_energies(g), _params())
    assert blind.sinks["A"].sink and not blind.sinks["A"].rescanned
    assert blind.counts["SINK_NOT_RESCANNED"] >= 1


def test_without_free_energies_no_pool_is_a_sink():
    res = explore(Stub({"A": [("B", -0.1, -0.1)]}), "A", load_thermo_model("none"), _params())
    assert res.sinks["A"].why_not == "UNKNOWN_DG" and not res.sinks["A"].sink
    assert res.no_net_reaction is None and res.reactant_share is None
    assert res.counts["NO_FREE_ENERGY"] == 2


def test_the_reactants_dominating_the_equilibrium_is_no_net_reaction():
    edges = {"A": [("B", -0.1, -0.1)], "B": [("A", -0.1, -0.1)]}
    g = free_energies({"A": 0.0, "B": 3.5})
    res = explore(Stub(edges), "A", g, _params(margin_kcal=5.0))
    assert not res.sinks["A"].sink  # +3.5 kcal/mol is within delta
    assert res.reason == "EXHAUSTED" and res.ends[0].reason == EQUILIBRIUM
    assert res.reactant_share == pytest.approx(1 / (1 + math.exp(-3.5 / (R_KCAL * 298.15))))
    assert res.no_net_reaction == "REACTANTS_DOMINATE"
    hot = explore(Stub(edges), "A", g, _params(margin_kcal=5.0, temperature_k=1000.0))
    assert hot.no_net_reaction is None and 0.5 < hot.reactant_share < 0.99


def _hs_model(h, s):
    """G(T) = H - T S of each state from its H and S."""
    return ThermoModel(lambda x, t: (h[x] - t * s[x], h[x], s[x]))


def test_t_star_an_association_loses_with_heat_and_a_release_gains():
    assoc = inversion(-10.0, -0.03)
    assert assoc.favoured == "below" and assoc.t_star == pytest.approx(333.33, abs=0.01)
    release = inversion(5.0, 0.04)
    assert release.favoured == "above" and release.t_star == pytest.approx(125.0)
    assert inversion(-1.0, 0.01).favoured == "always" and inversion(1.0, -0.01).favoured == "never"
    assert inversion(None, 0.01) is None
    edges = {"A.B": [("AB", -0.1, -0.1)], "C": [("D.H2O", -0.1, -0.1)]}
    h = {"A.B": 0.0, "AB": -10.0, "C": 0.0, "D.H2O": 5.0}
    s = {"A.B": 0.0, "AB": -0.03, "C": 0.0, "D.H2O": 0.04}
    for start, (t_cold, t_hot), sign in (("A.B", (200.0, 500.0), 1), ("C", (100.0, 200.0), -1)):
        cold = _chain(_explored(edges, start), _hs_model(h, s), t_cold).edges[0]
        hot = _chain(_explored(edges, start), _hs_model(h, s), t_hot).edges[0]
        assert sign * cold.dg < 0 < sign * hot.dg  # assoc: favoured cold; release: hot
        assert cold.inversion.favoured == ("below" if sign > 0 else "above")
        assert t_cold < cold.inversion.t_star < t_hot


def _tree():
    """R -> A (0.9) -> A1 (no step), R -> B (0.1) -> B1 -> B2; reverses nearly closed."""
    back = math.log(1e-9)
    return {"R": [("A", math.log(0.9), back), ("B", math.log(0.1), back)],
            "A": [("A1", 0.0, back)], "B": [("B1", 0.0, back)],
            "B1": [("B2", 0.0, back)]}  # fmt: skip


def test_the_mass_exploration_stops_at_the_tolerance_and_reports_the_unexplored_mass():
    stub = Stub(_tree())
    res = explore(stub, "R", None, _params(mass_tol=0.15))
    assert res.reason == "MASS_TOL" and list(res.graph.out) == ["R", "A", "A1"]
    assert res.frontier_mass == pytest.approx(0.1, abs=1e-6) and res.frontier_mass <= 0.15
    assert res.unexplored_mass == pytest.approx(res.frontier_mass)
    assert res.absorbing["B"] == UNEXPLORED and res.absorbing["A1"] == NO_OUTGOING_STEP
    assert [(e.key, round(e.probability, 6)) for e in res.ends] == [("A1", 0.9)]
    deeper = explore(Stub(_tree()), "R", None, _params(mass_tol=0.05))
    assert deeper.reason == "EXHAUSTED" and deeper.frontier_mass == 0.0
    assert {e.key for e in deeper.ends} == {"A1", "B2"}
    budget = explore(Stub(_tree()), "R", None, _params(max_expansions=2, mass_tol=0.0))
    assert budget.reason == "EXPANSION_BUDGET" and budget.graph.expanded == 2
    assert budget.unexplored_mass == pytest.approx(1.0)
    rows = predict_rows(budget, pathways_to_ends(budget))
    assert rows["expanded"] == 2 and rows["unexplored_mass"] == pytest.approx(1.0)


def test_the_climb_budget_leaves_high_pools_unexpanded_and_counts_their_mass():
    edges = {"R": [("hi", -0.1, -0.1), ("lo", -0.1, -0.1)]}
    g = {"R": 0.0, "hi": 25.0, "lo": 1.0}
    res = explore(Stub(edges), "R", free_energies(g), _params(max_climb_kcal=20.0))
    assert res.absorbing["hi"] == CLIMB_CUT and "hi" not in res.graph.out
    assert res.unexplored_mass >= res.frontier_mass and res.unexplored_mass > 0
    assert all(e.key != "hi" for e in res.ends)


def test_a_turnover_is_absorbing_and_never_expanded():
    edges = {"A.cat": [("A-cat", -0.1, -0.1)], "A-cat": [("B.cat", -0.1, -0.1)],
             "B.cat": [("C", -0.1, -0.1)]}  # fmt: skip
    res = explore(Stub(edges), "A.cat", None, _params())
    assert res.absorbing["B.cat"] == TURNOVER and "B.cat" not in res.graph.out
    assert [e.reason for e in res.ends] == [TURNOVER]
    on = explore(Stub(edges), "A.cat", None, _params(stop_at_turnover=False))
    assert "B.cat" in on.graph.out and TURNOVER not in on.absorbing.values()


def test_reservoir_draws_enter_the_balance_at_their_chemical_potential():
    edges = {"A": [("B", -0.1, -0.1, {"W": 1})], "B": [("C", -0.1, -0.1, {"W": -1})]}
    g = {"A": 0.0, "B": -2.0, "C": -1.0}
    mu = {"W": -1.5}
    model = ThermoModel(lambda x, t: g[x], lambda k, t: mu.get(k))
    chain = _chain(_explored(edges, "A"), model)
    rec = {(e.source, e.target): e for e in chain.edges}
    assert rec[("A", "B")].dg == pytest.approx(-2.0 + 1.5)  # a water taken from the reservoir
    assert rec[("B", "C")].dg == pytest.approx(1.0 - 1.5)  # given back
    assert chain.relative_g("C") == pytest.approx(-1.0)  # no net draw
    no_mu = _chain(_explored(edges, "A"), ThermoModel(lambda x, t: g[x]))
    assert no_mu.potentials["B"] is None and no_mu.counts["NO_CHEMICAL_POTENTIAL"] == 1


def test_free_energy_values_and_the_adapter_point():
    assert as_thermo(1.5) == Thermo(1.5) and as_thermo((1.0, 2.0, 0.003)) == Thermo(1.0, 2.0, 0.003)
    assert as_thermo(None) is None and as_thermo(float("nan")) is None
    with pytest.raises(TypeError):
        as_thermo((1.0, 2.0))
    assert load_thermo_model("none") is None
    with pytest.raises(ValueError):
        load_thermo_model("qrrho")
    assert climb_threshold(298.15, 0.01, 0.0) == pytest.approx(2.728, abs=1e-3)
    assert climb_threshold(383.15, 0.01, 0.0) == pytest.approx(3.506, abs=1e-3)
    assert StopParams().threshold == pytest.approx(2.728 + 5.0, abs=1e-3)


def test_the_qm_provider_sums_species_and_prices_the_reservoir_at_its_concentration():
    """Item B wired into the chain (`network.thermo_provider`): G(pool) = sum of the species'
    G, a reservoir species at RT ln(c / 1 M) above its standard G, None when one is missing."""
    import math

    from ksmech.network.free_energy import Thermo, load_thermo_model, rt
    from ksmech.network.pools import pool_state
    from ksmech.network.thermo_provider import THERMO_MISSING, QMThermo, at_concentration

    table = {"CCl": Thermo(-10.0, -8.0, 0.006), "[OH-]": Thermo(-5.0, -4.0, 0.004),
             "O": Thermo(-3.0, -2.0, 0.005)}  # fmt: skip

    class Fake(QMThermo):
        def _compute(self, record_key, t):
            if record_key not in table:
                self.counts[THERMO_MISSING] += 1
            return table.get(record_key)

    water = pool_state("O").species[0]
    model = load_thermo_model("qm", reservoir_smiles={water: "O"}, molar={water: 55.3},
                              species=Fake())  # fmt: skip
    t = 298.15
    g = model.free_energy(pool_state("CCl.[OH-]"), t)
    assert (g.g, g.h, g.s) == pytest.approx((-15.0, -12.0, 0.010))
    wet = model.free_energy(pool_state("CCl.[OH-].O"), t)
    assert wet.g == pytest.approx(-18.0 + rt(t) * math.log(55.3))
    assert wet.h == pytest.approx(-14.0)  # the shift is entropic: H unchanged, S lower
    assert wet.s == pytest.approx(0.015 - rt(t) * math.log(55.3) / t)
    mu = model.chemical_potential(water, t)
    assert mu.g == pytest.approx(at_concentration(table["O"], 55.3, t).g)
    assert model.chemical_potential("not-a-reservoir-key", t) is None
    assert model.free_energy(pool_state("CBr.[OH-]"), t) is None
    assert load_thermo_model("none") is None


def test_a_pool_that_fails_to_expand_is_unexplained_mass_not_an_end_state():
    """A pool the model cannot expand (no tokens) is EXPAND_FAILED: its mass joins the
    unexplored mass, never an end state; a pool that expands to nothing is NO_OUTGOING_STEP."""
    from ksmech.network.explore import EXPAND_FAILED

    class Failing(Stub):
        def expand(self, state):
            if state == "A":
                return Expansion([], Counter(), "MISSING_TOKENS")
            return super().expand(state)

    res = explore(Failing(_tree()), "R", None, _params(mass_tol=0.0))
    assert res.absorbing["A"] == EXPAND_FAILED and "A1" not in res.graph.states
    assert all(e.key != "A" for e in res.ends)
    assert res.unexplored_mass == pytest.approx(0.9, abs=1e-6) and res.frontier_mass == 0.0
    assert [e.key for e in res.ends] == ["B2"]
