"""`network.bidirectional`: the forward and backward trees meeting on the pool key, the joined
pathway in the forward direction and its I5 rescoring (reaction-network plan, items I1, I5), on
a hand-made network with stub models."""

from collections import Counter
from types import SimpleNamespace

from ksmech.network.bidirectional import (
    BACKWARD,
    FORWARD,
    TARGET_NOT_FULL_POOL,
    _halves,
    bidirectional_search,
    full_pool,
    join,
    meet,
)
from ksmech.network.pools import pool_state
from ksmech.network.search import Expansion, Proposal, SearchParams


def _state(key):
    return SimpleNamespace(key=key, smiles="[CH4:1]")  # one formula: every toy pool is full


S, A, B, C, T = (_state(k) for k in "SABCT")
R = {("S", "A"): ((1, 2, 1),), ("S", "B"): ((1, 3, 1),), ("A", "T"): ((2, 4, 1),),
     ("B", "C"): ((3, 5, 1),), ("C", "T"): ((5, 4, 1),)}  # fmt: skip
FWD = {("S", "A"): -1.0, ("S", "B"): -0.5, ("A", "T"): -1.0, ("B", "C"): -3.0, ("C", "T"): -1.0}
BWD = {("T", "A"): -0.2, ("T", "C"): -2.0, ("A", "S"): -0.3, ("C", "B"): -1.0, ("B", "S"): -1.0}
STATES = {s.key: s for s in (S, A, B, C, T)}


def _neg(bonds):
    return tuple((m, n, -d) for m, n, d in bonds)


class Toy:
    """A stub expander: forward proposals R along FWD, backward ones -R along BWD; its
    step_logp scores R at a state from the same tables (best = marginal + 0.1)."""

    def __init__(self, side):
        self.side, self.calls = side, Counter()

    def key(self, state):
        return state.key

    def expand(self, state):
        self.calls[state.key] += 1
        table = FWD if self.side == FORWARD else BWD
        out = []
        for (src, dst), lp in table.items():
            if src != state.key:
                continue
            bonds = R[(src, dst)] if self.side == FORWARD else _neg(R[(dst, src)])
            out.append(Proposal(STATES[dst], lp, info={"bonds": bonds, "lone": ()}))
        return Expansion(out)

    def step_logp(self, state, bonds, lone):
        for (src, dst), r in R.items():
            if self.side == FORWARD and state.key == src and tuple(bonds) == r:
                return SimpleNamespace(best=FWD[(src, dst)], marginal=FWD[(src, dst)] - 0.1)
            if self.side == BACKWARD and state.key == dst and tuple(bonds) == _neg(r):
                lp = BWD[(dst, src)]
                return SimpleNamespace(best=lp, marginal=lp - 0.1)
        return "STEP_OUTSIDE_POOL"


def test_the_trees_meet_where_the_two_costs_sum_least():
    f, b, meetings, _, reason, _ = meet(Toy(FORWARD), Toy(BACKWARD), S, T, SearchParams(k=1))
    assert reason == "K_FOUND"
    # the backward tree reaches the start itself: T-A-S by P_b, 0.2 + 0.3, under S-A by P_f
    # (1.0) + A-T by P_b (0.2)
    best = min(meetings, key=meetings.get)
    assert best == "S" and abs(meetings["S"] - 0.5) < 1e-9
    assert abs(meetings["A"] - 1.2) < 1e-9


def test_the_joined_pathway_runs_forward_with_r_on_each_half():
    params = SearchParams(k=3)
    f, b, meetings, _, _, cuts = meet(Toy(FORWARD), Toy(BACKWARD), S, T, params)
    paths = join(f, b, meetings, params, cuts)
    best = paths[0]
    assert best.nodes == ("S", "A", "T")
    assert best.meeting == "S" and [s.side for s in best.steps] == [BACKWARD, BACKWARD]
    keys = [(s.parent.key, s.child.key, s.bonds) for s in best.steps]
    assert keys == [("S", "A", R[("S", "A")]), ("A", "T", R[("A", "T")])]  # R, not -R
    nodes, at_a = _halves(f, b, "A")  # the same path split at A: S-A forward, A-T backward
    assert nodes == ["S", "A", "T"] and [s.side for s in at_a] == [FORWARD, BACKWARD]
    assert [(s.parent.key, s.child.key, s.bonds) for s in at_a] == keys
    assert all(len(set(p.nodes)) == len(p.nodes) for p in paths)


def test_rescoring_adds_the_other_model_on_every_step():
    """S-A-T: log P_f = -1 + -1, log P_b (marginals) = (-0.3 - 0.1) + (-0.2 - 0.1)."""
    res = bidirectional_search(Toy(FORWARD), Toy(BACKWARD), S, T, SearchParams(k=3), lam=0.5)
    best = res.pathways[0]
    assert best.nodes == ("S", "A", "T")
    assert abs(best.logp_forward - (-2.0)) < 1e-9
    assert abs(best.logp_backward - (-0.7)) < 1e-9
    assert abs(best.rescored - (2.0 + 0.5 * 0.7)) < 1e-9
    assert res.expanded[FORWARD] + res.expanded[BACKWARD] <= 50


def test_a_step_one_model_cannot_score_leaves_the_pathway_unrescored():
    class Blind(Toy):
        def step_logp(self, state, bonds, lone):
            return "MISSING_TOKENS"

    res = bidirectional_search(Toy(FORWARD), Blind(BACKWARD), S, T, SearchParams(k=2))
    assert all(p.rescored is None for p in res.pathways) and res.pathways
    assert res.skips["RESCORE_FAILED:MISSING_TOKENS"] == len(res.pathways)


def test_the_budget_counts_both_sides():
    res = bidirectional_search(Toy(FORWARD), Toy(BACKWARD), S, T, SearchParams(max_expansions=1))
    assert res.expanded[FORWARD] + res.expanded[BACKWARD] == 1
    assert res.cuts["EXPANSION_BUDGET"] >= 1


def test_each_node_is_expanded_once_per_side():
    fwd, bwd = Toy(FORWARD), Toy(BACKWARD)
    bidirectional_search(fwd, bwd, S, T, SearchParams(k=5))
    assert max(fwd.calls.values()) == 1 and max(bwd.calls.values()) == 1


def test_a_target_without_the_by_products_is_refused():
    start = pool_state("C[O-].CC(C)=O")
    assert full_pool(start, pool_state("CC(C)([O-])OC"))
    assert not full_pool(start, pool_state("CC(C)(O)OC"))  # a proton and the charge missing
    res = bidirectional_search(Toy(FORWARD), Toy(BACKWARD), start, pool_state("CC(C)(O)OC"))
    assert res.reason == TARGET_NOT_FULL_POOL and res.pathways == []


def test_a_path_met_at_several_nodes_counts_once_and_a_cycle_not_at_all(monkeypatch):
    """S-A-T is met at S (backward alone), A and T (forward alone): one pathway. Y, a parent
    of S that the forward model also reaches from S, joins S-Y-S-A-T: a cycle."""
    y = _state("Y")
    monkeypatch.setitem(STATES, "Y", y)
    monkeypatch.setitem(R, ("S", "Y"), ((1, 9, 1),))
    monkeypatch.setitem(R, ("Y", "S"), ((9, 1, -1),))
    monkeypatch.setitem(FWD, ("S", "Y"), -0.1)
    monkeypatch.setitem(BWD, ("S", "Y"), -0.1)
    res = bidirectional_search(Toy(FORWARD), Toy(BACKWARD), S, T, SearchParams(k=5))
    nodes = [p.nodes for p in res.pathways]
    assert len(nodes) == len(set(nodes)) and nodes.count(("S", "A", "T")) == 1
    assert res.cuts["DUPLICATE_PATH"] >= 2 and res.cuts["CYCLE"] >= 1
    assert all(len(set(n)) == len(n) for n in nodes)


def test_a_benchmark_case_row_has_the_forward_search_keys(monkeypatch):
    """`evaluate_case_bidirectional` on the toy: met, the recorded pool keys found at rank 1,
    in the keys `evaluate.summary` reads; a target without its by-products fails the case."""
    from ksmech.network import evaluate as ev
    from ksmech.network.benchmarks import Case

    monkeypatch.setattr(ev, "pool_state", lambda smiles: STATES[smiles])
    case = Case("c1", "S", "T", ("S", "A", "T"), (), "val")
    row = ev.evaluate_case_bidirectional(Toy(FORWARD), Toy(BACKWARD), case, SearchParams(k=3))
    assert row["arrival"] == "EXACT" and row["recorded_rank"] == 1 and row["best_steps"] == 2
    assert row["expanded"] == sum(row["expanded_by_side"].values())
    assert ev.summary([row], 3)["recorded_top1"] == 1.0
    real = {"S": pool_state("C[O-].CC(C)=O"), "T": pool_state("CC(C)(O)OC")}
    monkeypatch.setattr(ev, "pool_state", lambda smiles: real[smiles])
    bad = ev.evaluate_case_bidirectional(Toy(FORWARD), Toy(BACKWARD), case)
    assert bad["failure"] == TARGET_NOT_FULL_POOL


def test_backward_run_is_an_option_of_search_and_evaluate():
    from ksmech.network.cli import build_parser

    args = build_parser().parse_args(["search", "--run", "f", "--backward-run", "b", "--start",
                                      "C", "--target", "C"])  # fmt: skip
    assert (args.backward_run, args.backward_checkpoint) == ("b", "best")
    assert args.roundtrip_lambda == 1.0
    args = build_parser().parse_args(["evaluate", "--set", "reactmech", "--run", "f"])
    assert args.backward_run is None
