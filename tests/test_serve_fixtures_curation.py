"""Curation fixtures: a tmp curation database, typed `NetworkOut`s of the SN2 steps HO- + MeCl
and MeO- + MeCl (and a variant with a user step, and a two-step chain), an inline job submitter
and a `NetworkSource` stand-in.

Every network is built from serve code and upstream only (spec v9): `pools.initial_state` and
the synthetic runtime's tokens; each step is a reaction matrix on the pool atoms (O attacks the
C of MeCl, the C-Cl pair leaves on Cl; or O takes an H of MeCl), recorded with
`edgestep.step_record` (product state, arrow reading, orbital readings) and saved; the network
edges are `network.step_edge`s, the provenance is `synthetic_context.provenance("floor")`.

Import with the repo's pattern (and `# noqa: F811` on test defs that take them):

from tests.test_serve_fixtures_curation import curation_engine, oh_sn2_network  # noqa: F401
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest

from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    oh_sn2_state,
    serve_mini_processed,
    serve_settings,
    sn2_state,
    synthetic_context,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


# ---- tokens and arrows


def _fragment_atoms(state, key: str, nth: int = 0) -> list[int]:
    from ksmech.data import chem
    from ksmech.serve.core.keys import fragment_keys

    frags = [f for f in fragment_keys(chem.parse_mapped(state.mapped_smiles)) if f.token_key == key]
    return list(frags[nth].atoms)


def _first(mol, atoms, symbol: str) -> int:
    return next(i for i in atoms if mol.GetAtomWithIdx(i).GetSymbol() == symbol)


@dataclass
class Step:
    state: Any  # the source state
    product: Any  # the saved product state
    matrix: dict  # MatrixIO
    record: dict  # edgestep.step_record


def record_step(settings, runtime, state, bonds, lone) -> Step:
    from ksmech.serve.chem.pools import save_state
    from ksmech.serve.science.analysis import pool_work
    from ksmech.serve.science.edgestep import step_record
    from ksmech.serve.science.stepapply import matrix_of

    work = pool_work(runtime, state)
    R = matrix_of({"bonds": bonds, "lone": lone})
    rec = step_record(state, R, pg=work.pg, rt=work.rt, occupations=runtime.occupations)
    assert rec["ok"], rec["issues"]
    save_state(rec["product_state"], settings)
    return Step(state, rec["product_state"], {"bonds": bonds, "lone": lone}, rec)


def _sn2_atoms(state, donor_key: str, nth: int):
    from ksmech.data import chem

    mol = chem.parse_mapped(state.mapped_smiles)
    o = _first(mol, _fragment_atoms(state, donor_key, nth), "O")
    ccl = _fragment_atoms(state, "CCl", nth)
    return mol, o, _first(mol, ccl, "C"), _first(mol, ccl, "Cl")


def sn2_step(settings, runtime, state, donor_key: str, nth: int = 0) -> Step:
    """O of the donor (nth fragment of `donor_key`) attacks the C of the nth MeCl."""
    _, o, c, cl = _sn2_atoms(state, donor_key, nth)
    bonds = [[*sorted((o, c)), 1], [*sorted((c, cl)), -1]]
    return record_step(settings, runtime, state, bonds, [[o, -2], [cl, 2]])


def proton_transfer_step(settings, runtime, state, donor_key: str) -> Step:
    """O of the donor takes an H of MeCl (the C keeps the pair)."""
    mol, o, c, _ = _sn2_atoms(state, donor_key, 0)
    h = next(n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == "H")
    bonds = [[*sorted((o, h)), 1], [*sorted((c, h)), -1]]
    return record_step(settings, runtime, state, bonds, [[o, -2], [c, 2]])


def _hex(text: str, n: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def node_out(state, depth: int, *, root: bool, parent: str | None = None, logp: float = 0.0):
    from ksmech.serve.core.schemas import NodeOut
    from ksmech.serve.predict.network import display_smiles

    species = [
        {
            "species_key": sp.species_key,
            "token_keys": list(sp.token_keys),
            "display_smiles": sp.token_keys[0],
            "count": sp.count,
            "charge": sp.charge,
            "resonance_flag": sp.resonance_flag,
        }
        for sp in state.species
    ]
    return NodeOut(
        id=state.pool_id,
        pool_key=state.pool_key,
        state_id=state.state_id,
        alias_state_ids=[],
        species=species,
        display_smiles=display_smiles(state),
        depth=depth,
        best_logp=logp,
        best_parent_edge=parent,
        status="expanded" if root else "terminal",
        terminal_reason=None if root else "MAX_STEPS",
        failure=None,
        is_root=root,
        n_valid=None,
        scanned_mass=None,
        unscanned_mass=None,
        apply_failures={},
        fermi_smeared=False,
        warnings=[],
    )


def edge_out(step: Step, *, kind: str = "predicted", **kw):
    """The `EdgeOut` of a step (a predicted floor edge of rank 1 unless `kw` says otherwise)."""
    from ksmech.serve.core.schemas import EdgeOut
    from ksmech.serve.predict.network import Network, Node, NodeStatus, step_edge
    from ksmech.serve.predict.service import _edge_out

    parent = Node(step.state.pool_id, step.state.pool_key, step.state, [], 0, 0.0, None,
                  NodeStatus.EXPANDED)  # fmt: skip
    child = Node(step.product.pool_id, step.product.pool_key, step.product, [], 1, 0.0, None,
                 NodeStatus.TERMINAL)  # fmt: skip
    fields = {"score": 0.5, "logp": None, "weight": -0.1, "rank": 1, "predictor": "floor",
              "predictor_version": "floor/g-atom/test"} | kw  # fmt: skip
    if kind == "corrected":
        fields |= {"weight": 0.0, "rank": 0, "predictor": "user", "predictor_version": "user/1"}
    net = Network("n", 0, parent.id, {parent.id: parent, child.id: child}, {}, {}, [], [],
                  None, None, None, "synthetic", True)  # fmt: skip
    e = step_edge(net, parent, step.product, step.record, kind=kind, **fields)
    return EdgeOut.model_validate(_edge_out(e, net))


def network_out(nodes, edges, pathway_edges, provenance, request, *, n_corrected: int = 0):
    from ksmech.serve.core.schemas import NetworkOut

    ids = [e.id for e in pathway_edges]
    pathway = {
        "id": "w" + _hex("|".join(ids), 12),
        "rank": 1,
        "node_ids": [pathway_edges[0].source] + [e.target for e in pathway_edges],
        "edge_ids": ids,
        "logp": 0.0,
        "probability": None,
        "n_steps": len(ids),
        "n_corrected": n_corrected,
        "ends_with": "MAX_STEPS",
        "closes_cycle": False,
        "cycle_ids": [],
    }
    return NetworkOut.model_validate(
        {
            "network_id": "j" + _hex("|".join(e.id for e in edges), 16),
            "revision": 1,
            "complete": True,
            "root": nodes[0].id,
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
            "cycles": [],
            "pathways": [pathway],
            "failed_proposals": [],
            "skipped": {},
            "params": {
                "beam": 2,
                "max_steps": 1,
                "temperature": 1.0,
                "stop_at_turnover": True,
                "candidate_scope": "all",
                "max_scan": 64,
                "max_nodes": 200,
                "n_pathways": 20,
            },
            "request": request.model_dump(),
            "provenance": provenance.model_dump(),
            "scores_are_probabilities": False,
        }
    )


def predict_request(reactants: list[str], conditions: str | None = "water, 25 C"):
    from ksmech.serve.core.schemas import PredictIn

    return PredictIn(
        reactants_smiles=reactants,
        stoichiometry=[1] * len(reactants),
        conditions=conditions,
        predictor="floor",
        beam=2,
        max_steps=1,
    )


def floor_provenance(context):
    from ksmech.serve.core.schemas import Provenance

    return Provenance.model_validate(context.provenance("floor"))


def sn2_network(settings, runtime, context, state, donor_key: str, reactants: list[str]):
    """root -> SN2 product, one predicted edge, one pathway."""
    step = sn2_step(settings, runtime, state, donor_key)
    prov = floor_provenance(context)
    edge = edge_out(step, predictor_version=prov.predictor_version)
    nodes = [node_out(state, 0, root=True), node_out(step.product, 1, root=False, parent=edge.id)]
    return network_out(nodes, [edge], [edge], prov, predict_request(reactants))


def corrected_network(settings, runtime, context, state, donor_key: str, reactants: list[str]):
    """root -> SN2 product through a user step, next to the predicted proton transfer onto
    another node (both active: a user step never supersedes a prediction)."""
    prov = floor_provenance(context)
    wrong = proton_transfer_step(settings, runtime, state, donor_key)
    right = sn2_step(settings, runtime, state, donor_key)
    predicted = edge_out(wrong, predictor_version=prov.predictor_version, score=2.0)
    edge = edge_out(right, kind="corrected", score=None)
    nodes = [
        node_out(state, 0, root=True),
        node_out(right.product, 1, root=False, parent=edge.id),
        node_out(wrong.product, 1, root=False, parent=predicted.id),
    ]
    edges = [predicted, edge]
    return network_out(nodes, edges, [edge], prov, predict_request(reactants), n_corrected=1)


def double_sn2_network(settings, runtime, context):
    """2 MeO- + 2 MeCl: root -> mid -> final, two independent SN2 steps (disjoint maps,
    continuous states)."""
    from ksmech.serve.chem.pools import SpeciesInput, initial_state, save_state

    state = initial_state([SpeciesInput("C[O-]", 2), SpeciesInput("CCl", 2)], settings.limits)
    save_state(state, settings)
    prov = floor_provenance(context)
    first = sn2_step(settings, runtime, state, "C[O-]")
    second = sn2_step(settings, runtime, first.product, "C[O-]")
    e1 = edge_out(first, predictor_version=prov.predictor_version)
    e2 = edge_out(second, predictor_version=prov.predictor_version)
    nodes = [
        node_out(state, 0, root=True),
        node_out(first.product, 1, root=False, parent=e1.id),
        node_out(second.product, 2, root=False, parent=e2.id),
    ]
    request = predict_request(["C[O-]", "CCl"])
    request.stoichiometry = [2, 2]
    return network_out(nodes, [e1, e2], [e1, e2], prov, request)


# ---- the database, the network source and the job submitter


@pytest.fixture
def curation_engine(tmp_path):
    """A fresh curation database on tmp (make_engine + init_db)."""
    from ksmech.serve.core import db

    engine = db.make_engine(tmp_path / "curation" / "curation.sqlite")
    db.init_db(engine)
    yield engine
    engine.dispose()


class NetworkStore:
    """A `NetworkSource` stand-in: job id -> (NetworkOut, the job's PredictIn as dict)."""

    def __init__(self) -> None:
        self.networks: dict[str, tuple[Any, dict]] = {}

    def add(self, network) -> str:
        self.networks[network.network_id] = (network, network.request.model_dump(mode="json"))
        return network.network_id

    def __call__(self, job_id: str):
        from ksmech.serve.core.errors import Code, ServeError

        if job_id not in self.networks:
            raise ServeError(Code.JOB_NOT_FOUND, f"no job {job_id}", status=404)
        return self.networks[job_id]


class InlineJobs:
    """A `JobSubmitter` that runs the job handler at once in a `LocalJobContext`. A kind in
    `settings_for` runs on its own settings (e.g. an export into a tmp data root)."""

    def __init__(self, settings, scheduler, handlers: dict, settings_for: dict | None = None):
        self.settings, self.scheduler, self.handlers = settings, scheduler, dict(handlers)
        self.settings_for = dict(settings_for or {})
        self.results: dict[str, dict] = {}
        self.contexts: dict[str, Any] = {}

    def __call__(self, kind: str, request: dict) -> dict:
        from ksmech.serve.core.contracts import LocalJobContext

        job_id = "j" + _hex(f"{kind}|{len(self.results)}|{request}", 16)
        settings = self.settings_for.get(kind, self.settings)
        ctx = LocalJobContext(job_id, settings, self.scheduler)
        self.contexts[job_id] = ctx
        self.results[job_id] = self.handlers[kind](ctx, request)
        return {
            "job_id": job_id,
            "kind": kind,
            "status": "done",
            "status_url": f"/api/jobs/{job_id}",
            "parent_job_id": None,
        }

    @property
    def last(self) -> dict:
        return list(self.results.values())[-1]


def mini_pairs(processed) -> dict[tuple[str, str], str]:
    """(reactant pool key, product pool key) -> step id of the PMechDB steps under
    `processed` (keys.side_keys of both sides): the export's PairsProvider stand-in."""
    from ksmech.data.io import read_steps
    from ksmech.serve.core.keys import side_keys

    out = {}
    for s in read_steps(processed / "pmechdb" / "steps.parquet"):
        key = (side_keys(s.reactants_smiles).pool_key, side_keys(s.products_smiles).pool_key)
        out.setdefault(key, s.step_id)
    return out


@pytest.fixture(scope="session")
def oh_sn2_network(synthetic_settings, synthetic_runtime, synthetic_context, oh_sn2_state):  # noqa: F811
    return sn2_network(
        synthetic_settings,
        synthetic_runtime,
        synthetic_context,
        oh_sn2_state,
        "[OH-]",
        ["[OH-]", "CCl"],
    )


@pytest.fixture(scope="session")
def meo_sn2_network(synthetic_settings, synthetic_runtime, synthetic_context, sn2_state):  # noqa: F811
    return sn2_network(
        synthetic_settings,
        synthetic_runtime,
        synthetic_context,
        sn2_state,
        "C[O-]",
        ["C[O-]", "CCl"],
    )


@pytest.fixture(scope="session")
def meo_corrected_network(synthetic_settings, synthetic_runtime, synthetic_context, sn2_state):  # noqa: F811
    return corrected_network(
        synthetic_settings,
        synthetic_runtime,
        synthetic_context,
        sn2_state,
        "C[O-]",
        ["C[O-]", "CCl"],
    )


@pytest.fixture(scope="session")
def double_network(synthetic_settings, synthetic_runtime, synthetic_context):  # noqa: F811
    return double_sn2_network(synthetic_settings, synthetic_runtime, synthetic_context)


@pytest.fixture
def network_store(oh_sn2_network, meo_sn2_network, meo_corrected_network, double_network):
    store = NetworkStore()
    for net in (oh_sn2_network, meo_sn2_network, meo_corrected_network, double_network):
        store.add(net)
    return store


# ---- sanity tests of the fixtures


def test_engine_has_the_tables(curation_engine):
    from sqlalchemy import inspect

    names = set(inspect(curation_engine).get_table_names())
    assert {"mechanism", "curatedstep", "recheck", "exportrecord", "exportedstep"} <= names


def test_sn2_networks_are_typed_and_consistent(oh_sn2_network, meo_sn2_network):
    for net, root in (
        (oh_sn2_network, "[CH3][Cl]|0.[OH]|-1"),
        (meo_sn2_network, "[CH3][Cl]|0.[CH3][O]|-1"),
    ):  # noqa: E501
        (edge,) = net.edges
        assert net.nodes[0].pool_key == root and net.root == net.nodes[0].id
        assert {a.kind for a in edge.arrows} == {"lone_pair_to_bond", "bond_to_lone_pair"}
        assert edge.target_pool_key.endswith("[Cl]|-1") and edge.readings
        assert edge.pmechdb_code and edge.anchor.sign == "form"
        assert net.pathways[0].edge_ids == [edge.id]
        assert net.provenance.predictor == "floor" and edge.predictor_version
    assert oh_sn2_network.edges[0].target_pool_key == "[CH3][OH]|0.[Cl]|-1"
    assert meo_sn2_network.edges[0].target_pool_key == "[CH3][O][CH3]|0.[Cl]|-1"


def test_product_states_are_saved(meo_sn2_network, synthetic_settings):  # noqa: F811
    from ksmech.serve.chem.pools import load_state

    edge = meo_sn2_network.edges[0]
    assert load_state(edge.product_state_id, synthetic_settings).pool_key == edge.target_pool_key


def test_corrected_variant(meo_corrected_network):
    predicted, corrected = meo_corrected_network.edges
    assert predicted.status == "active" and predicted.kind == "predicted"
    assert corrected.kind == "corrected" and corrected.predictor == "user"
    assert corrected.target_pool_key == "[CH3][O][CH3]|0.[Cl]|-1"
    assert predicted.target_pool_key != corrected.target_pool_key
    assert meo_corrected_network.pathways[0].edge_ids == [corrected.id]


def test_double_network_is_a_chain(double_network):
    e1, e2 = double_network.edges
    assert e1.target == e2.source and e1.product_state_id == e2.source_state_id
    assert not set(e1.touched_maps) & set(e2.touched_maps)
    assert e2.target_pool_key.count("[Cl]|-1") == 2


def test_mini_pairs(serve_mini_processed):  # noqa: F811
    pairs = mini_pairs(serve_mini_processed)
    assert ("[CH3][Cl]|0.[OH]|-1", "[CH3][OH]|0.[Cl]|-1") in pairs  # sigma pool keys
