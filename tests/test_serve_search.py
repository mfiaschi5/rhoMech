"""`serve.predict.search`: the web app's search towards a product (reaction-network plan,
item D) on StubChemistry: the engine's best-first search with the network's chemistry as the
expander, the pathways ending on the target, the budget, the start as the goal, the JSON form,
the arrivals after a hand expansion, the SKELETON level on real pool states, and the predict
job through JobService."""

import json

import pytest

from ksmech.serve.chem.pools import SpeciesInput, initial_state
from ksmech.serve.core.contracts import Limits
from ksmech.serve.core.errors import ServeError
from ksmech.serve.core.schemas import PredictIn
from ksmech.serve.core.testing import ToyPredictorRef, requires, toy_state
from ksmech.serve.predict.network import (
    NetHooks,
    NetParams,
    NodeStatus,
    Search,
    build_network,
    continue_from,
    network_from_dict,
    network_to_dict,
)
from ksmech.serve.predict.search import (
    arrival_level,
    new_search,
    refresh_pathways,
    search_network,
    skeletons_of,
)
from tests.test_serve_fixtures_net import (  # noqa: F401 (fixtures)
    branching,
    job_ctx,
    no_xtb,
    serve_settings,
)
from tests.test_serve_job_service import service  # noqa: F401 (fixture)

pytestmark = [pytest.mark.serve, requires("predict")]

LIMITS = Limits(6, 4, 200, 10, 12, 500)


def run(chem, ctx, target, max_expansions=50, **params):
    root = chem.root([SpeciesInput("A", 1)])
    search = new_search(chem, {"target_smiles": target, "max_expansions": max_expansions})
    p = NetParams(**({"beam": 5, "max_steps": 6, "n_pathways": 3} | params))
    return search_network(chem, root, ToyPredictorRef, p, search, NetHooks(ctx))


def keys(net, pathway):
    return [net.nodes[n].pool_key for n in pathway.node_ids]


def test_the_search_reaches_a_target_the_beam_prunes(job_ctx):  # noqa: F811
    """G -> H: the beam of 2 prunes G (fixture), the search goes through it."""
    beam = build_network(branching(), toy_state({"A": 1}), ToyPredictorRef,
                         NetParams(beam=2, max_steps=3), NetHooks(job_ctx()))  # fmt: skip
    assert "H" not in {n.pool_key for n in beam.nodes.values()}
    net = run(branching(), job_ctx(), "H", n_pathways=1)
    (best,) = net.pathways
    assert keys(net, best) == ["A", "C", "G", "H"] and best.ends_with == "TARGET_EXACT"
    assert not best.closes_cycle and best.n_steps == 3
    h = next(n for n in net.nodes.values() if n.pool_key == "H")
    assert (h.status, h.terminal_reason) == (NodeStatus.TERMINAL, "TARGET_EXACT")
    s = net.search
    assert s.reason == "K_FOUND" and s.arrivals[h.id] == "EXACT" and 0 < s.expanded <= 7
    assert net.complete
    left = [n for n in net.nodes.values() if n.status == NodeStatus.FRONTIER]
    assert not left  # every node the search did not expand is TERMINAL NODE_BUDGET
    assert any(n.terminal_reason == "NODE_BUDGET" for n in net.nodes.values())


def test_k_pathways_by_summed_weight(job_ctx):  # noqa: F811
    """E is reached by A-B-E and A-C-E: two pathways, the better first, ranks 1 and 2."""
    net = run(branching(), job_ctx(), "E", n_pathways=2)
    assert [keys(net, p) for p in net.pathways] == [["A", "B", "E"], ["A", "C", "E"]]
    assert [p.rank for p in net.pathways] == [1, 2]
    assert net.pathways[0].logp >= net.pathways[1].logp
    assert {p.ends_with for p in net.pathways} == {"TARGET_EXACT"}


def test_the_budget_and_the_depth_cut_the_search(job_ctx):  # noqa: F811
    net = run(branching(), job_ctx(), "H", max_expansions=1)
    assert net.pathways == [] and net.search.reason == "EXHAUSTED"
    assert net.search.expanded == 1 and net.search.cuts["EXPANSION_BUDGET"] >= 1
    net = run(branching(), job_ctx(), "H", max_steps=2)
    assert net.pathways == [] and net.search.cuts["DEPTH_CUT"] >= 1


def test_the_start_as_the_target(job_ctx):  # noqa: F811
    net = run(branching(), job_ctx(), "A")
    assert net.search.reason == "START_IS_GOAL" and net.pathways == []
    root = net.nodes[net.root]
    assert root.terminal_reason == "TARGET_EXACT" and net.search.expanded == 0


def test_the_json_form_keeps_the_search_and_only_then(job_ctx):  # noqa: F811
    net = run(branching(), job_ctx(), "H", n_pathways=1)
    d = network_to_dict(net)
    back = network_from_dict(json.loads(json.dumps(d)))
    assert back.search == net.search and back.pathways == net.pathways
    assert json.dumps(network_to_dict(back), sort_keys=True) == json.dumps(d, sort_keys=True)
    plain = build_network(branching(), toy_state({"A": 1}), ToyPredictorRef, NetParams(),
                          NetHooks(job_ctx()))  # fmt: skip
    assert "search" not in network_to_dict(plain)


def test_a_hand_expansion_marks_the_new_arrivals(job_ctx):  # noqa: F811
    """A search cut before H; expanding G by hand reaches H, and the pathway appears."""
    chem, ctx = branching(), job_ctx()
    net = run(chem, ctx, "H", max_expansions=3)
    assert net.pathways == []
    g = next(n for n in net.nodes.values() if n.pool_key == "G")
    continue_from(chem, net, g.id, 1, net.params, NetHooks(ctx))
    net.pathways = refresh_pathways(net)
    (best,) = [p for p in net.pathways if keys(net, p)[-1] == "H"]
    assert best.ends_with == "TARGET_EXACT"
    h = next(n for n in net.nodes.values() if n.pool_key == "H")
    assert h.terminal_reason == "TARGET_EXACT"


def _search_of(smiles: str, levels=("EXACT", "SKELETON")) -> Search:
    state = initial_state([SpeciesInput(smiles, 1)], LIMITS)
    sk = skeletons_of(state.mapped_smiles)
    return Search(smiles, {s.species_key: s.count for s in state.species}, dict(sk), list(levels),
                  50)  # fmt: skip


def test_the_skeleton_level_reaches_a_product_drawn_after_work_up():
    """The hemiketal drawn neutral is reached by its alkoxide at SKELETON, not EXACT; the
    ketone is not reached at either (its skeleton has C=O)."""
    alkoxide = initial_state([SpeciesInput("CC(C)([O-])OC", 1), SpeciesInput("O", 1)], LIMITS)
    sk = skeletons_of(alkoxide.mapped_smiles)
    assert arrival_level(_search_of("CC(C)(O)OC"), alkoxide, sk) == "SKELETON"
    assert arrival_level(_search_of("CC(C)(O)OC", ["EXACT"]), alkoxide, sk) is None
    assert arrival_level(_search_of("CC(C)([O-])OC"), alkoxide, sk) == "EXACT"
    assert arrival_level(_search_of("CC(C)=O"), alkoxide, sk) is None
    assert skeletons_of("not a smiles") is None


def test_a_predict_job_with_a_target(service):  # noqa: F811
    svc = service()
    body = PredictIn(reactants_smiles=["A"], predictor="floor", beam=5, max_steps=6,
                     n_pathways=1, target_smiles="H")  # fmt: skip
    created = svc.predict(body)
    job = svc.get(created.job_id)
    net = job.result
    assert job.status == "done" and net.search is not None
    assert net.search.reason == "K_FOUND" and net.request.target_smiles == "H"
    assert list(net.search.arrivals.values()) == ["EXACT"]  # arrived nodes only
    assert net.pathways[0].ends_with == "TARGET_EXACT"
    assert svc.pathways(created.job_id)[0].ends_with == "TARGET_EXACT"
    assert "→ H" in svc.list(limit=1)[0].label
    plain = svc.get(svc.predict(PredictIn(reactants_smiles=["A"], predictor="floor")).job_id)
    assert plain.result.search is None
    with pytest.raises(ServeError) as err:
        svc.predict(PredictIn(reactants_smiles=["A"], predictor="user", target_smiles="H"))
    assert err.value.status == 422
    with pytest.raises(ServeError) as err:
        svc.predict(PredictIn(reactants_smiles=["A"], predictor="floor", target_smiles="."))
    assert err.value.detail["field"] == "target_smiles"
