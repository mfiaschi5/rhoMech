"""network.py on StubChemistry only: beam expansion, merging by pool key, exact edge
probabilities, the scan stopping rule, the cycle rule on the worked example (anchors P0-P3),
ranked pathways (including the RECURRENCE pathway [A, B, A]), overrides and byte-equal
determinism of the JSON form."""

import json
import math

import numpy as np
import pytest

from ksmech.serve.contracts import ArrowSpec, log_softmax_valid
from ksmech.serve.errors import ServeError
from ksmech.serve.network import (
    NetHooks,
    NetParams,
    NodeStatus,
    apply_override,
    build_network,
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
    assert ab.probability == pytest.approx(p[0] + p[1], rel=1e-15)
    assert ab.logp == pytest.approx(math.log(p[0] + p[1]), rel=1e-15)
    assert (ab.score, ab.rank, ab.n_alternatives) == (1.0, 1, 1)
    assert ab.alternatives[0]["rank"] == 2 and ab.alternatives[0]["score"] == 0.5
    assert ac.probability == pytest.approx(p[2], rel=1e-15) and ac.rank == 3
    root = net.nodes[net.root]
    assert root.scanned_mass == pytest.approx(1.0) and root.unscanned_mass == pytest.approx(0.0)
    assert root.n_valid == 3


def test_scan_stops_on_the_strict_bound_only(job_ctx):  # noqa: F811
    # p = (0.5, 0.25, 0.25) exactly. Beam 1. After r1: U = 0.5 == p_B - p_{B+1} = 0.5, so
    # the scan goes on; r2 joins the same child: U = 0.25 < 0.75, a strict "bound" stop.
    chem = StubChemistry(
        [rule("r1", "A", "X", math.log(2)), rule("r2", "A", "X", 0.0), rule("r3", "A", "Y", 0.0)]
    )
    net = build(chem, [("A", 1)], job_ctx(), beam=1, max_steps=1)
    root = net.nodes[net.root]
    assert softmax([math.log(2), 0.0, 0.0]).tolist() == [0.5, 0.25, 0.25]
    assert (root.scanned_mass, root.unscanned_mass) == (0.75, 0.25)
    assert [n.pool_key for n in net.nodes.values()] == ["A", "X"]  # Y never scanned
    assert edge_between(net, "A", "X").probability == 0.75


def test_scan_keeps_going_at_equality(job_ctx):  # noqa: F811
    # p = (0.5, 0.25, 0.25) on three different children: U equals p_B - p_{B+1} after each
    # of the first two proposals (never strictly below), so all three are scanned.
    chem = StubChemistry(
        [rule("r1", "A", "X", math.log(2)), rule("r2", "A", "Y", 0.0), rule("r3", "A", "Z", 0.0)]
    )
    net = build(chem, [("A", 1)], job_ctx(), beam=1, max_steps=1)
    root = net.nodes[net.root]
    assert (root.scanned_mass, root.unscanned_mass) == (1.0, 0.0)
    # every proposal was scanned, and the beam keeps the best child only
    assert [n.pool_key for n in net.nodes.values()] == ["A", "X"]
    assert edge_between(net, "A", "X").probability == 0.5


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


# ---- overrides (4.6)


def override(chem, net, ctx, **req):
    req.setdefault("allow_invalid", False)
    return apply_override(chem, net, req, net.params, NetHooks(ctx))


def arrow(i):
    return ArrowSpec(i, 0, 0, 1).to_dict()


def test_override_records_a_corrected_edge(job_ctx):  # noqa: F811
    ctx, chem = job_ctx(), branching()
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=2)  # G pruned
    old = edge_between(net, "C", "E")
    rev = net.revision
    out = override(chem, net, ctx, edge_id=old.id, arrows=[arrow(5)], continue_steps=1)
    ce = net.edges[out["corrected_edge_id"]]
    assert ce.kind == "corrected" and ce.id.startswith("c") and ce.logp == 0.0
    assert (ce.score, ce.probability, ce.rank) == (None, None, None)
    assert ce.original["edge_id"] == old.id and ce.original["target"] == old.target
    assert ce.predictor_view[0]["pair_valid"] and ce.arrows[0]["site_source"] == "user"
    assert (old.status, old.superseded_by) == ("superseded", ce.id)
    g = node_of(net, "G")  # merged into the existing PRUNED node, then continued one layer
    assert ce.target == g.id and out["target_node"] == g.id and out["continued_from"] == g.id
    assert g.status == NodeStatus.EXPANDED and node_of(net, "H").terminal_reason == "MAX_STEPS"
    assert out["network_revision"] == net.revision > rev and out["issues"] == []
    assert all(old.id not in p.edge_ids for p in net.pathways)
    (corrected,) = [p for p in net.pathways if ce.id in p.edge_ids]
    assert corrected.n_corrected == 1 and corrected.node_ids[-1] == node_of(net, "H").id
    assert corrected.logp == pytest.approx(node_of(net, "C").best_logp, abs=1e-15)


def test_corrected_edge_contributes_zero_logp(job_ctx):  # noqa: F811
    ctx, chem = job_ctx(), branching()
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=1)
    out = override(chem, net, ctx, node_id=net.root, arrows=[arrow(7)], continue_steps=1)
    ce = net.edges[out["corrected_edge_id"]]
    x = node_of(net, "X")  # a new node (never scanned by the beam)
    assert ce.original is None and ce.target == x.id and x.depth == 1
    assert x.best_logp == 0.0 and x.best_parent_edge == ce.id  # P.best_logp + 0
    assert x.terminal_reason == "NO_CANDIDATES" and out["continued_from"] == x.id
    (path,) = [p for p in net.pathways if ce.id in p.edge_ids]
    assert path.logp == 0.0 and path.probability == 1.0 and path.n_corrected == 1


def test_override_continuation_targets(job_ctx):  # noqa: F811
    ctx, chem = job_ctx(), branching()
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=2)  # E, F: MAX_STEPS; G: PRUNED
    out = override(chem, net, ctx, edge_id=edge_between(net, "B", "F").id, arrows=[arrow(2)])
    e = node_of(net, "E")  # TERMINAL(MAX_STEPS): continued for max(1, 2 - 2) = 1 layer
    assert out["continued_from"] == e.id and e.terminal_reason == "NO_CANDIDATES"
    out = override(chem, net, ctx, node_id=net.root, arrows=[arrow(7)], continue_steps=0)
    assert out["continued_from"] is None and node_of(net, "X").status == NodeStatus.FRONTIER
    out = override(chem, net, ctx, node_id=net.root, arrows=[arrow(0)], continue_steps=2)
    assert out["continued_from"] is None  # B was already expanded: never re-expanded
    old = edge_between(net, "C", "E")
    out = override(chem, net, ctx, edge_id=old.id, arrows=[arrow(5)], continue_prediction=False)
    assert out["continued_from"] is None and node_of(net, "G").status == NodeStatus.PRUNED
    assert chem.calls.count(("expand", "B")) == 1


def test_correction_of_a_correction(job_ctx):  # noqa: F811
    ctx, chem = job_ctx(), branching()
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=1)
    predicted = edge_between(net, "A", "B")
    out1 = override(chem, net, ctx, edge_id=predicted.id, arrows=[arrow(1)], continue_steps=0)
    c1 = net.edges[out1["corrected_edge_id"]]
    out2 = override(chem, net, ctx, edge_id=c1.id, arrows=[arrow(7)], continue_steps=0)
    c2 = net.edges[out2["corrected_edge_id"]]
    assert (c1.status, c1.superseded_by) == ("superseded", c2.id)  # the audit trail
    assert (predicted.status, predicted.superseded_by) == ("superseded", c1.id)
    # the comparison baseline stays the predictor's step, not the user's first correction
    assert c2.original["edge_id"] == predicted.id
    assert c2.original["score"] == predicted.score and c2.original["rank"] == predicted.rank
    assert c2.original["probability"] == predicted.probability
    for paths in (net.pathways, rank_pathways(net), rank_pathways(net, include_pruned=True)):
        assert all(c1.id not in p.edge_ids and predicted.id not in p.edge_ids for p in paths)
    assert any(c2.id in p.edge_ids for p in net.pathways)


def test_override_closing_a_turnover_does_not_continue(toy_cycle, job_ctx):  # noqa: F811
    ctx = job_ctx()
    net = build(toy_cycle, CYCLE, ctx, beam=2, max_steps=2)  # CP.S: TERMINAL(MAX_STEPS)
    cp_s = node_of(net, "CP.S")
    assert cp_s.terminal_reason == "MAX_STEPS"
    n_nodes = len(net.nodes)
    out = override(toy_cycle, net, ctx, node_id=cp_s.id, arrows=[arrow(2)])  # release
    ce = net.edges[out["corrected_edge_id"]]
    target = node_of(net, "Cat.P.S")
    assert ce.closes_cycle and out["target_node"] == target.id
    assert (target.status, target.terminal_reason) == (NodeStatus.TERMINAL, "TURNOVER")
    assert out["continued_from"] is None and len(net.nodes) == n_nodes + 1
    assert "CS.P" not in {n.pool_key for n in net.nodes.values()}
    assert ("expand", "Cat.P.S") not in toy_cycle.calls


def test_override_errors(job_ctx):  # noqa: F811
    ctx, chem = job_ctx(), branching()
    net = build(chem, [("A", 1)], ctx, beam=2, max_steps=1)
    old = edge_between(net, "A", "B")
    with pytest.raises(ServeError) as e:
        override(chem, net, ctx, edge_id=old.id, arrows=[arrow(0)])
    assert e.value.code == "OVERRIDE_NOOP" and e.value.status == 422
    with pytest.raises(ServeError) as e:
        override(chem, net, ctx, edge_id="e" + "0" * 16, arrows=[arrow(0)])
    assert (e.value.code, e.value.status) == ("EDGE_NOT_FOUND", 404)
    with pytest.raises(ServeError) as e:
        override(chem, net, ctx, node_id="p" + "0" * 20, arrows=[arrow(0)])
    assert (e.value.code, e.value.status) == ("NODE_NOT_FOUND", 404)
    with pytest.raises(ServeError) as e:
        override(chem, net, ctx, edge_id=old.id, node_id=net.root, arrows=[arrow(0)])
    assert e.value.code == "REQUEST_INVALID"
    n_edges, rev = len(net.edges), net.revision
    with pytest.raises(ServeError) as e:
        override(chem, net, ctx, edge_id=old.id, arrows=[arrow(2)])  # B -> E: not at A
    assert e.value.code == "APPLY_FAILED"
    assert old.status == "active" and (len(net.edges), net.revision) == (n_edges, rev)
    override(chem, net, ctx, node_id=net.root, arrows=[arrow(1)], continue_prediction=False)
    with pytest.raises(ServeError) as e:  # the same correction again
        override(chem, net, ctx, node_id=net.root, arrows=[arrow(1)], continue_prediction=False)
    assert e.value.code == "OVERRIDE_NOOP"


# ---- determinism and the JSON form


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
    old = edge_between(net, "C", "E")
    override(chem, net, job_ctx(), edge_id=old.id, arrows=[arrow(5)], continue_steps=1)
    assert dump(network_from_dict(json.loads(dump(net)))) == dump(net)
    assert "time" not in dump(net)  # no timestamps in the network
