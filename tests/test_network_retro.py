"""`network.retro` (reaction-network plan, item K): the AND-OR search from a product alone
returns the optimum of exhaustive search on small synthetic instances, stops on its leaves and
budgets, counts the by-product reservoir from recorded steps, and runs the backward expander's
by-product rule."""

import itertools
import math
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from ksmech.network.retro import (
    CYCLE,
    NOT_USING_BYPRODUCT,
    Leaves,
    ModelRetroExpander,
    RetroParams,
    RetroStep,
    byproduct_reservoir,
    retro_search,
    route_rows,
)
from ksmech.network.search import Expansion, Proposal


class Table:
    """species -> [(parents, log p)]."""

    def __init__(self, rows):
        self.rows, self.calls = rows, Counter()

    def expand(self, species):
        self.calls[species] += 1
        return [RetroStep(tuple(p), lp) for p, lp in self.rows.get(species, [])]


def _optimum(rows, species, leaves, depth, max_depth, on_path=frozenset()):
    """The cheapest route by exhaustive recursion (inf when none)."""
    if leaves.reason(species):
        return 0.0
    if depth >= max_depth:
        return math.inf
    best = math.inf
    for parents, lp in rows.get(species, []):
        if any(p in on_path | {species} for p in parents):
            continue
        cost = -lp + sum(_optimum(rows, p, leaves, depth + 1, max_depth, on_path | {species})
                         for p in parents)  # fmt: skip
        best = min(best, cost)
    return best


def _random_instance(seed, n=8):
    rng = random.Random(seed)
    names = [f"s{i}" for i in range(n)]
    rows = {}
    for s in names:
        rows[s] = []
        for _ in range(rng.randint(0, 3)):
            k = rng.choice([1, 1, 2, 2, 3])
            rows[s].append((tuple(rng.sample(names, k)), -rng.uniform(0.05, 3.0)))
    leaves = Leaves(catalogue=frozenset(rng.sample(names[1:], 3)))
    return rows, leaves


@pytest.mark.parametrize("seed", range(25))
def test_the_and_or_search_finds_the_optimum_of_exhaustive_search(seed):
    rows, leaves = _random_instance(seed)
    params = RetroParams(max_expansions=10_000, max_depth=5)
    want = _optimum(rows, "s0", leaves, 0, params.max_depth)
    res = retro_search(Table(rows), "s0", leaves, params)
    if math.isinf(want):
        assert res.route is None and res.reason in ("EXHAUSTED", "SOLVED")
    else:
        assert res.reason == "SOLVED" and res.route.cost == pytest.approx(want)
        assert set(res.route.leaves) <= leaves.catalogue


def test_an_admissible_value_keeps_the_optimum_and_expands_no_more():
    for seed in range(25):
        rows, leaves = _random_instance(seed)
        params = RetroParams(max_expansions=10_000, max_depth=5)
        plain, guided = Table(rows), Table(rows)
        base = retro_search(plain, "s0", leaves, params)

        def value(s, leaves=leaves):  # every step costs at least 0.05: 0.04 is a lower bound
            return 0.0 if leaves.reason(s) else 0.04

        best = retro_search(guided, "s0", leaves, params, value)
        assert (base.route is None) == (best.route is None)
        if base.route:
            assert best.route.cost == pytest.approx(base.route.cost)
        assert sum(guided.calls.values()) <= sum(plain.calls.values())


def test_a_split_is_two_subgoals_and_the_route_lists_its_steps():
    rows = {"T": [(("A", "B"), -0.5), (("C",), -0.1)], "C": [(("T",), -0.1)],
            "A": [(("a1",), -0.2)], "B": []}  # fmt: skip
    leaves = Leaves(catalogue=frozenset({"a1", "B"}))
    res = retro_search(Table(rows), "T", leaves)
    assert res.reason == "SOLVED" and res.route.cost == pytest.approx(0.7)
    assert res.route.leaves == Counter({"a1": 1, "B": 1}) and res.cuts[CYCLE] == 1
    assert [r["species"] for r in route_rows(res.route)] == ["T", "A"]
    root_leaf = retro_search(Table(rows), "a1", leaves)
    assert root_leaf.reason == "ROOT_IS_LEAF" and root_leaf.route.cost == 0.0


def test_budgets_are_counted():
    rows = {"T": [(("A",), -0.1)], "A": [(("B",), -0.1)], "B": [(("C",), -0.1)]}
    leaves = Leaves(catalogue=frozenset({"C"}))
    shallow = retro_search(Table(rows), "T", leaves, RetroParams(max_depth=2))
    assert shallow.route is None and shallow.cuts["DEPTH_CUT"] == 1
    tight = retro_search(Table(rows), "T", leaves, RetroParams(max_expansions=1))
    assert tight.reason == "EXPANSION_BUDGET" and tight.expanded == 1
    ok = retro_search(Table(rows), "T", Leaves(reservoir=frozenset({"C"})))
    assert ok.reason == "SOLVED" and ok.route.leaves == Counter({"C": 1})


def test_the_byproduct_reservoir_is_counted_from_the_recorded_steps():
    step = lambda r, p: SimpleNamespace(reactants_smiles=r, products_smiles=p)  # noqa: E731
    steps = [
        step("[CH3:1][Cl:2].[OH-:3]", "[CH3:1][OH:3].[Cl-:2]"),
        step("[CH3:1][Cl:2].[NH3:3]", "[CH3:1][NH3+:3].[Cl-:2]"),
        step(
            "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[OH-:6]", "[CH3:1][C:2](=[O:3])[OH:6].[O-:4][CH3:5]"
        ),  # fmt: skip
    ]
    skips = Counter()
    rows = byproduct_reservoir([*steps, step("C(", "C")], max_heavy=2, skips=skips)
    assert rows[0][1:] == ("[Cl-]", 2)
    assert {smi for _, smi, n in rows[1:] if n == 1} == {"CO", "C[NH3+]", "C[O-]"}
    assert all(smi != "CC(=O)O" for _, smi, _ in rows)  # four heavy atoms: not small
    assert skips["BYPRODUCT_UNREADABLE"] == 1


def test_a_step_on_a_pool_enlarged_by_a_byproduct_must_use_it():
    """Going back, a by-product joins: a step on species + x that leaves x untouched is not
    counted (NOT_USING_BYPRODUCT); the parents are the parent pool's species."""
    from ksmech.network.pools import pool_state

    class Backward:
        def expand(self, pool):
            parent = pool_state("CCl.[OH-]")
            uses_x = frozenset(range(1, 30))  # touches every atom, the by-product's included
            lazy = frozenset({1})  # only atom 1: of the species, never of the by-product
            return Expansion(
                [Proposal(parent, -0.3, "a", uses_x), Proposal(parent, -0.2, "b", lazy)]
            )

    chloride = pool_state("[Cl-]").species[0]
    ex = ModelRetroExpander(Backward(), "CO", [(chloride, "[Cl-]", 5)], n_byproducts=1)
    steps = ex.expand(ex.target)
    assert len(steps) == 3  # two on the species alone, one of the two on species + Cl-
    assert ex.skips[NOT_USING_BYPRODUCT] == 1
    assert all(set(s.parents) == set(pool_state("CCl.[OH-]").species) for s in steps)
    assert set(ex.smiles) >= set(pool_state("CCl.[OH-]").species)


def test_combinations_are_deterministic():
    rows, leaves = _random_instance(3)
    a = retro_search(Table(rows), "s0", leaves, RetroParams(max_depth=5))
    b = retro_search(Table(rows), "s0", leaves, RetroParams(max_depth=5))
    assert (a.reason, a.expanded) == (b.reason, b.expanded)
    assert list(itertools.islice(route_rows(a.route), 3)) == list(
        itertools.islice(route_rows(b.route), 3)
    )


def test_a_species_on_two_branches_is_expanded_once():
    """T comes from A + B or from A + C; A is met on both branches and costs one call."""
    rows = {"T": [(("A", "B"), -0.1), (("A", "C"), -0.2)], "A": [(("a",), -0.3)],
            "B": [(("b",), -5.0)], "C": [(("c",), -0.1)]}  # fmt: skip
    table = Table(rows)
    res = retro_search(table, "T", Leaves(catalogue=frozenset({"a", "b", "c"})))
    assert res.reason == "SOLVED" and res.route.cost == pytest.approx(0.6)
    assert table.calls["A"] == 1 and res.expanded == len(table.calls)
