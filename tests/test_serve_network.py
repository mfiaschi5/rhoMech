"""network.py on StubChemistry only: beam expansion, merging by pool key, exact edge
weights, the cycle rule on the worked example (anchors P0-P3), ranked pathways (including the
RECURRENCE pathway [A, B, A]), user steps, continuation and byte-equal determinism of the JSON
form."""

import json
import math

import numpy as np
import pytest

from ksmech.serve.contracts import log_softmax_valid
from ksmech.serve.network import (
    NetHooks,
    NetParams,
    NodeStatus,
    add_user_edge,
    build_network,
    continue_from,
    network_from_dict,
    network_to_dict,
    rank_pathways,
)
from ksmech.serve.pools import SpeciesInput
from ksmech.serve.testing import StubChemistry, ToyPredictorRef, requires, toy_state
from tests.test_serve_fixtures_net import (  # noqa: F401 (fixtures)
    branching,
    job_ctx,
    keto_enol,
    no_xtb,
    rule,
    serve_settings,
    toy_cycle,
)

pytestmark = [pytest.mark.serve, requires("predict")]


def build(chem, roots, ctx, **params):
    root = chem.root([SpeciesInput(s, n) for s, n in roots])
    return build_network(chem, root, ToyPredictorRef, NetParams(**params), NetHooks(ctx))


def node_of(net, pool_key):
    return net.nodes[toy_state(_counter(pool_key)).pool_id]


def _counter(pool_key):
    from collections import Counter

    return Counter(k for k in pool_key.split(".") if k)


def edge_between(net, a, b, kind="predicted"):
    src, dst = node_of(net, a).id, node_of(net, b).id
    hits = [e for e in net.edges.values() if (e.source, e.target, e.kind) == (src, dst, kind)]
    assert len(hits) == 1, (a, b, kind, hits)
    return hits[0]


def softmax(scores):
    lp = log_softmax_valid(np.array(scores, dtype=float), np.ones(len(scores), dtype=bool), 1.0)
    return np.exp(lp)


CYCLE = [("Cat", 1), ("S", 2)]
P = ["Cat.S.S", "CS.S", "CP.S", "Cat.P.S", "CS.P", "CP.P", "Cat.P.P"]
BRANCH_P = softmax([1.0, 0.9, -3.0])  # A -> B, C, X


# ---- the worked example of design 4.4


def test_worked_example_anchors_p0_to_p3(toy_cycle, job_ctx):  # noqa: F811
    net = build(toy_cycle, CYCLE, job_ctx(), beam=2, max_steps=6, stop_at_turnover=False)
    assert [node_of(net, k).depth for k in P] == [0, 1, 2, 3, 4, 5, 6]
    turnovers = {c.closing_edge: c for c in net.cycles.values() if c.kind == "turnover"}
    anchors = []
    for i in range(1, len(P)):
        e = edge_between(net, P[i - 1], P[i])
        cyc = turnovers.get(e.id)
        assert e.closes_cycle == (cyc is not None) and not e.recurrence
        if cyc is not None:
            anchors.append(cyc.anchor)
            assert e.cycle_ids == [cyc.id]
    assert anchors == [node_of(net, k).id for k in P[:4]]  # P0, P1, P2, P3
    catalysts = [turnovers[edge_between(net, P[i - 1], P[i]).id].catalyst for i in range(3, 7)]
    assert catalysts == [{"Cat": 1}, {"CS": 1}, {"CP": 1}, {"Cat": 1}]
    assert node_of(net, "Cat.P.S").status == NodeStatus.EXPANDED  # not stopped
    assert node_of(net, "Cat.P.P").terminal_reason == "MAX_STEPS"


def test_p3_turnover_fields_and_stop(toy_cycle, job_ctx):  # noqa: F811
    net = build(toy_cycle, CYCLE, job_ctx(), beam=2, max_steps=8)  # stop_at_turnover=True
    assert sorted(n.pool_key for n in net.nodes.values()) == sorted(P[:4])
    p0, p3 = node_of(net, P[0]), node_of(net, P[3])
    assert (p3.status, p3.terminal_reason) == (NodeStatus.TERMINAL, "TURNOVER")
    e = edge_between(net, P[2], P[3])
    (cyc,) = net.cycles.values()
    assert (cyc.kind, cyc.anchor, cyc.closing_edge) == ("turnover", p0.id, e.id)
    assert (cyc.catalyst, cyc.consumed, cyc.produced) == ({"Cat": 1}, {"S": 1}, {"P": 1})
    assert cyc.node_ids == [node_of(net, k).id for k in P[:4]]
    assert cyc.edge_ids == [edge_between(net, P[i - 1], P[i]).id for i in (1, 2, 3)]
    assert cyc.virtual_edge == {"source": p3.id, "target": p0.id, "label": "−P +S"}
    assert cyc.id.startswith("y") and len(cyc.id) == 13
    (path,) = net.pathways
    assert (path.ends_with, path.closes_cycle, path.cycle_ids) == ("TURNOVER", True, [cyc.id])
    assert path.node_ids == cyc.node_ids and path.n_steps == 3


def test_spectator_is_never_a_catalyst(toy_cycle, job_ctx):  # noqa: F811
    net = build(toy_cycle, [*CYCLE, ("W", 1)], job_ctx(), beam=2, max_steps=8)
    (cyc,) = net.cycles.values()
    assert cyc.catalyst == {"Cat": 1} and "W" not in cyc.consumed


def test_one_step_change_is_never_a_turnover(job_ctx):  # noqa: F811
    chem = StubChemistry([rule("direct", "Cat.S", "Cat.P", 1.0)])
    net = build(chem, [("Cat", 1), ("S", 1)], job_ctx(), beam=2, max_steps=3)
    assert len(net.edges) == 1 and not net.cycles
    assert node_of(net, "Cat.P").terminal_reason == "NO_CANDIDATES"


def test_recurrence_cycle_and_pathway(keto_enol, job_ctx):  # noqa: F811
    net = build(keto_enol, [("K", 1)], job_ctx(), beam=2, max_steps=4)
    k, e = node_of(net, "K"), node_of(net, "E")
    assert keto_enol.calls == [("expand", "K"), ("expand", "E")]  # K never re-expanded
    back = edge_between(net, "E", "K")
    assert back.recurrence and not back.closes_cycle
    (cyc,) = net.cycles.values()
    assert (cyc.kind, cyc.anchor, cyc.node_ids) == ("recurrence", k.id, [k.id, e.id])
    assert cyc.edge_ids == [edge_between(net, "K", "E").id, back.id]
    (path,) = net.pathways
    assert path.node_ids == [k.id, e.id, k.id] and path.ends_with == "RECURRENCE"
    assert path.closes_cycle and path.cycle_ids == [cyc.id]


# ---- merging, probabilities, the scan


def test_merge_by_pool_key_and_alias(job_ctx):  # noqa: F811
    chem = branching()
    net = build(chem, [("A", 1)], job_ctx(), beam=2, max_steps=3)
    e_node = node_of(net, "E")
    into_e = [x for x in net.edges.values() if x.target == e_node.id]
    assert len(into_e) == 2 and chem.calls.count(("expand", "E")) == 1
    b, c = node_of(net, "B"), node_of(net, "C")
    assert b.best_logp == pytest.approx(math.log(BRANCH_P[0]), abs=1e-15)
    via_b = b.best_logp + math.log(softmax([1.0, 0.5])[0])
    via_c = c.best_logp + math.log(softmax([0.6, 0.2])[0])
    assert e_node.best_logp == pytest.approx(max(via_b, via_c), abs=1e-15) and via_b > via_c
    assert e_node.best_parent_edge == edge_between(net, "B", "E").id
    assert e_node.depth == 2
    alt = toy_state(_counter("E"), "alt").state_id
    assert e_node.state.state_id == toy_state(_counter("E")).state_id  # first reached (via B)
    assert e_node.alias_state_ids == [alt]
    assert edge_between(net, "C", "E").product_state_id == alt
    assert node_of(net, "G").status == NodeStatus.PRUNED  # third new pool of layer 2, beam 2
    assert "X" not in {n.pool_key for n in net.nodes.values()}


def test_marginal_edge_probability(job_ctx):  # noqa: F811
    chem = StubChemistry(
        [rule("r1", "A", "B", 1.0), rule("r2", "A", "B", 0.5), rule("r3", "A", "C", 0.0)]
    )
    net = build(chem, [("A", 1)], job_ctx(), beam=3, max_steps=1)
    p = softmax([1.0, 0.5, 0.0])
    ab, ac = edge_between(net, "A", "B"), edge_between(net, "A", "C")
    assert math.exp(ab.weight) == pytest.approx(p[0] + p[1], rel=1e-15)
    assert ab.logp == pytest.approx(math.log(p[0]), rel=1e-15)  # the representative's
    assert (ab.score, ab.rank, ab.n_alternatives) == (1.0, 1, 1)
    assert ab.alternatives[0]["rank"] == 2 and ab.alternatives[0]["score"] == 0.5
    assert math.exp(ac.weight) == pytest.approx(p[2], rel=1e-15) and ac.rank == 3
    root = net.nodes[net.root]
    assert root.scanned_mass == pytest.approx(1.0) and root.unscanned_mass == pytest.approx(0.0)
    assert root.n_valid == 3


# ---- stop conditions


def test_beam_pruning_and_node_budget(job_ctx):  # noqa: F811
    net = build(branching(), [("A", 1)], job_ctx(), beam=2, max_steps=2)
    statuses = [node_of(net, k).status for k in "EFG"]
    assert statuses == ["terminal", "terminal", "pruned"]
    assert node_of(net, "E").terminal_reason == "MAX_STEPS"
    assert len(net.pathways) == 3  # pruned nodes are excluded by default
    assert len(rank_pathways(net, include_pruned=True)) == 4
    net = build(branching(), [("A", 1)], job_ctx(), beam=2, max_steps=2, max_nodes=4)
    e, f, g = (node_of(net, k) for k in "EFG")
    assert e.terminal_reason == "MAX_STEPS"
    assert (f.status, f.terminal_reason) == (NodeStatus.TERMINAL, "NODE_BUDGET")
    assert g.status == NodeStatus.PRUNED  # outside the beam anyway
    assert sorted(p.ends_with for p in net.pathways) == ["MAX_STEPS", "MAX_STEPS", "NODE_BUDGET"]


def test_max_steps(toy_cycle, job_ctx):  # noqa: F811
    net = build(toy_cycle, CYCLE, job_ctx(), beam=2, max_steps=2)
    assert [node_of(net, k).status for k in P[:2]] == ["expanded", "expanded"]
    assert node_of(net, P[2]).terminal_reason == "MAX_STEPS"
    assert net.complete


def test_no_candidates_and_no_applicable_proposal(job_ctx):  # noqa: F811
    ctx = job_ctx()
    chem = StubChemistry([rule("noop", "A", "A", 1.0), rule("other", "B", "C", 1.0)])
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=2)
    root = net.nodes[net.root]
    assert (root.status, root.terminal_reason) == ("terminal", "NO_APPLICABLE_PROPOSAL")
    assert root.apply_failures == {"RESONANCE_ONLY_STEP": 1}
    assert net.skipped["RESONANCE_ONLY_STEP"] == 1 and ctx.skipped["RESONANCE_ONLY_STEP"] == 1
    assert net.failed_proposals[0]["node"] == root.id
    assert net.failed_proposals[0]["code"] == "RESONANCE_ONLY_STEP"
    net = build(chem, [("Q", 1)], ctx, beam=2, max_steps=2)
    assert net.nodes[net.root].terminal_reason == "NO_CANDIDATES" and not net.pathways


def test_featurization_failure_fails_the_node(job_ctx):  # noqa: F811
    ctx = job_ctx()
    chem = branching(failures={"B": "FEATURIZATION_FAILED"})
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=3)
    b = node_of(net, "B")
    assert b.status == NodeStatus.FAILED and b.failure["code"] == "FEATURIZATION_FAILED"
    assert net.skipped["FEATURIZATION_FAILED"] == 1
    assert ctx.skip_log[0]["code"] == "FEATURIZATION_FAILED"
    assert ctx.skip_log[0]["node"] == b.id
    ends = {tuple(p.node_ids): p.ends_with for p in net.pathways}
    assert ends[(net.root, b.id)] == "FAILED:FEATURIZATION_FAILED"


def test_cancellation_keeps_the_partial_network(job_ctx):  # noqa: F811
    ctx = job_ctx()

    class Cancelling(StubChemistry):
        def expand(self, state, req, c):
            c.cancel_flag = True  # cancelled while the first node expands
            return super().expand(state, req, c)

    net = build(Cancelling(branching().rules), [("A", 1)], ctx, beam=2, max_steps=3)
    assert not net.complete
    assert [node_of(net, k).terminal_reason for k in "BC"] == ["CANCELLED", "CANCELLED"]
    assert node_of(net, "A").status == NodeStatus.EXPANDED


def test_progress_and_snapshots(toy_cycle, job_ctx):  # noqa: F811
    ctx, seen = job_ctx(), []
    root = toy_cycle.root([SpeciesInput("Cat", 1), SpeciesInput("S", 2)])
    hooks = NetHooks(ctx, snapshot=lambda net: seen.append(net.revision))
    net = build_network(toy_cycle, root, ToyPredictorRef, NetParams(beam=2, max_steps=8), hooks)
    assert seen == sorted(seen) and len(seen) == 2 * 3 and net.revision > seen[-1]
    phases = [p["phase"] for p in ctx.progress_log]
    assert phases.count("expand") == 3 and phases[-1] == "pathways"
    assert ctx.progress_log[0]["total"] == 1 + 2 * 7 and hooks.owner == ctx.job_id


# ---- pathways


def test_pathway_ranking_order(job_ctx):  # noqa: F811
    net = build(branching(), [("A", 1)], job_ctx(), beam=3, max_steps=2)
    paths = net.pathways
    assert [p.rank for p in paths] == list(range(1, len(paths) + 1))
    assert [p.logp for p in paths] == sorted((p.logp for p in paths), reverse=True)
    assert all(p.id.startswith("w") and len(p.id) == 13 for p in paths)
    best = paths[0]
    assert best.node_ids == [node_of(net, k).id for k in "ABE"] and best.n_corrected == 0
    assert best.probability == pytest.approx(math.exp(best.logp))
    assert best.ends_with == "MAX_STEPS" and not best.closes_cycle
    x_path = [p for p in paths if p.node_ids[-1] == node_of(net, "X").id]
    assert x_path[0].ends_with == "NO_CANDIDATES"
    assert all(len(set(p.node_ids)) == len(p.node_ids) for p in paths)


# ---- user steps and continuation


def user_step(net, node_id, pool_key, n=1):
    """A user step from `node_id` to the toy pool `pool_key` (the matrix encodes `n`)."""
    parent = net.nodes[node_id]
    child = toy_state(_counter(pool_key))
    step = {"matrix": {"bonds": [[0, 1, n]], "lone": []}, "anchor": None, "score": None}
    return add_user_edge(net, parent, parent.state, child, step)


def test_user_step_adds_a_corrected_edge_and_a_frontier_node(job_ctx):  # noqa: F811
    net = build(branching(), [("A", 1)], job_ctx(), beam=2, max_steps=1)
    e = user_step(net, net.root, "Q")
    assert e.kind == "corrected" and e.id.startswith("c") and e.predictor == "user"
    assert (e.weight, e.rank, e.logp) == (0.0, 0, None) and e.path_logp == 0.0
    q = node_of(net, "Q")
    assert q.status == NodeStatus.FRONTIER and q.best_logp == 0.0 and q.best_parent_edge == e.id
    assert user_step(net, net.root, "Q").id == e.id  # idempotent
    assert len([x for x in net.edges.values() if x.target == q.id]) == 1
    path = next(p for p in rank_pathways(net) if p.node_ids[-1] == q.id)
    assert path.logp == 0.0 and path.n_corrected == 1


def test_continuation_from_a_user_node(job_ctx):  # noqa: F811
    chem = branching()
    net = build(chem, [("A", 1)], job_ctx(), beam=2, max_steps=1)
    user_step(net, net.root, "B.extra")
    target = node_of(net, "B.extra")
    other = ToyPredictorRef.__class__("other", "toy/2", None, 1.0)
    continue_from(chem, net, target.id, 1, net.params, NetHooks(job_ctx()), other)
    assert net.predictor == ToyPredictorRef  # restored after the run
    assert target.status in (NodeStatus.EXPANDED, NodeStatus.TERMINAL)
    assert ("expand", "B.extra") in chem.calls


def test_user_step_closing_a_cycle_is_marked(keto_enol, job_ctx):  # noqa: F811
    net = build(keto_enol, [("K", 1)], job_ctx(), beam=1, max_steps=1)
    e_node = node_of(net, "E")
    e = user_step(net, e_node.id, "K")
    assert e.recurrence and e.cycle_ids


def test_byte_equal_determinism_and_round_trip(toy_cycle, job_ctx):  # noqa: F811
    def dump(net):
        return json.dumps(network_to_dict(net), sort_keys=True)

    kw = dict(beam=2, max_steps=6, stop_at_turnover=False)
    a = build(toy_cycle, CYCLE, job_ctx(), **kw)
    b = build(toy_cycle, CYCLE, job_ctx(), **kw)
    assert dump(a) == dump(b)
    back = network_from_dict(json.loads(dump(a)))
    assert dump(back) == dump(a)
    assert back.params == a.params and back.predictor == ToyPredictorRef
    chem = branching()
    net = build(chem, [("A", 1)], job_ctx(), beam=2, max_steps=2)
    user_step(net, node_of(net, "C").id, "Q")
    assert dump(network_from_dict(json.loads(dump(net)))) == dump(net)
    assert "time" not in dump(net)  # no timestamps in the network
