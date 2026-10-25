"""Curation fixtures (w2-curation-data): a tmp curation database, typed `NetworkOut`s of the
SN2 steps HO- + MeCl and MeO- + MeCl (and a corrected variant, and a two-step chain), an
inline job submitter and a `NetworkSource` stand-in.

Every network is built from wave-1 code and upstream only: `pools.initial_state` and the
synthetic runtime's tokens; the one-centre O lone pair and the {C, Cl} virtual token are found
from the descriptors; the product comes from upstream `apply_predicted_arrows` at the sites
(O, C) and is saved with `pools.state_from_mol` + `save_state`; the dataset arrows
[LONE_PAIR O -> ATOM C; BOND C-Cl -> ATOM Cl] are written by hand and verified with
`build_step`; the provenance is `synthetic_context.provenance("stub")`.

Import with the repo's pattern (and `# noqa: F811` on test defs that take them):

from tests.test_serve_fixtures_curation import curation_engine, oh_sn2_network  # noqa: F401
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from ksmech.serve.testing import requires
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


def element_of(rt, g: int) -> str:
    from rdkit import Chem

    z = [a.atomic_number for f in rt.fragments for a in f.tokens.atoms]
    return Chem.GetPeriodicTable().GetElementSymbol(int(z[g]))


def map_of(rt, g: int) -> int:
    return rt.mol.GetAtomWithIdx(rt.mol_atom[g]).GetAtomMapNum()


def _fragment_of(rt, key: str, nth: int = 0) -> int:
    hits = [f for f, frag in enumerate(rt.fragments) if frag.key == key]
    assert len(hits) > nth, (key, [f.key for f in rt.fragments])
    return hits[nth]


def _tokens_in(rt, frag: int, occupied: bool) -> list[int]:
    return [
        rt.n_atoms + j
        for j in range(rt.n_orbitals)
        if int(rt.orbital_fragment[j]) == frag and bool(rt.occupied[j]) == occupied
    ]


def lone_pair_token(rt, frag: int, element: str = "O") -> tuple[int, int]:
    """(token, site) of the first one-centre occupied donor token on `element` in `frag`."""
    for t in _tokens_in(rt, frag, occupied=True):
        centres = list(rt.centres(t))
        if len(centres) != 1 or not rt.donor_ok[t - rt.n_atoms]:
            continue
        if element_of(rt, centres[0]) == element:
            return t, centres[0]
    raise AssertionError(f"no one-centre {element} donor in fragment {frag}")


def virtual_token(
    rt, frag: int, elements: set[str], site: str, partner: str, *, exact: bool = True
) -> tuple[int, int, int]:
    """(token, site, partner) of the first acceptor token in `frag` whose centre elements
    are `elements` (or include them, `exact=False`): its heaviest centres of element `site`
    and `partner`."""
    for t in _tokens_in(rt, frag, occupied=False):
        centres = rt.centres(t)
        found = {element_of(rt, g) for g in centres}
        if not rt.acceptor_ok[t - rt.n_atoms] or not (
            found == elements if exact else elements <= found
        ):
            continue
        ranked = sorted(centres, key=lambda g: -centres[g])
        b = next(g for g in ranked if element_of(rt, g) == site)
        d = next(g for g in ranked if g != b and element_of(rt, g) == partner)
        return t, b, d
    raise AssertionError(f"no acceptor with centres {elements} in fragment {frag}")


def _row_col(rt, token: int) -> int:
    j = token - rt.n_atoms
    same = rt.occupied == rt.occupied[j]
    return int(same[:j].sum())


def arrow_out(rt, state, arrow, *, site_source: str, sink: int | None, source_kind: str):
    """An `ArrowOut` of one (donor, site, acceptor, site) arrow."""
    from ksmech.serve.keys import species_key_of
    from ksmech.serve.schemas import ArrowOut

    def species(g: int) -> str:
        return species_key_of(rt.fragments[int(rt.atom_fragment[g])].key)[0]

    def centres(t: int) -> dict[str, float]:
        return {str(map_of(rt, g)): float(w) for g, w in sorted(rt.centres(t).items())}

    return ArrowOut(
        donor=arrow.donor,
        donor_site=arrow.donor_site,
        acceptor=arrow.acceptor,
        acceptor_site=arrow.acceptor_site,
        donor_site_map=map_of(rt, arrow.donor_site),
        acceptor_site_map=map_of(rt, arrow.acceptor_site),
        donor_element=element_of(rt, arrow.donor_site),
        acceptor_element=element_of(rt, arrow.acceptor_site),
        donor_species=species(arrow.donor_site),
        acceptor_species=species(arrow.acceptor_site),
        source_kind=source_kind,
        source_partner=None,
        source_partner_map=None,
        sink_partner=sink,
        sink_partner_map=None if sink is None else map_of(rt, sink),
        site_source=site_source,
        row=_row_col(rt, arrow.donor),
        col=_row_col(rt, arrow.acceptor),
        donor_centres_map=centres(arrow.donor),
        acceptor_centres_map=centres(arrow.acceptor),
        diagnostic={},
        physics=None,
    )


@dataclass
class Step:
    """One applied step: its source and product states, arrow, dataset arrows and edge."""

    state: Any
    product: Any
    arrow: Any
    sink: int
    datasets: list[dict]
    code: str
    edge: Any = None
    extra: dict = field(default_factory=dict)


def apply_arrow(settings, rt, state, arrow, sink: int, datasets: list[dict]) -> Step:
    """Upstream `apply_predicted_arrows`, the product saved, the dataset arrows verified."""
    from ksmech.arrows.apply_predicted import PredictedArrow, apply_predicted_arrows
    from ksmech.data.schema import Arrow, Source
    from ksmech.data.validation import build_step
    from ksmech.serve.pmech_format import format_arrow_code
    from ksmech.serve.pools import save_state, state_from_mol

    res = apply_predicted_arrows(rt.mol, [PredictedArrow(*arrow.as_list())], rt)
    assert res.ok, (res.reason, res.detail)
    product = state_from_mol(res.mol)
    save_state(product, settings)
    arrows = [Arrow.from_dict(a) for a in datasets]
    rxn = f"{state.mapped_smiles}>>{product.mapped_smiles}"
    build_step(step_id="fixture-check", source=Source.USER_CURATED, rxn_smiles=rxn, arrows=arrows)
    return Step(
        state, product, arrow, sink, [a.to_dict() for a in arrows], format_arrow_code(arrows)
    )


def sn2_step(settings, runtime, state, donor_key: str, nth: int = 0) -> Step:
    """The SN2 of the `nth` fragment `donor_key` on the `nth` CCl: n(O) -> sigma*(C-Cl) at
    the sites (O, C), Cl leaving."""
    from ksmech.serve.contracts import ArrowSpec

    rt = runtime.reactant_tokens(state)
    donor, o = lone_pair_token(rt, _fragment_of(rt, donor_key, nth))
    acceptor, c, cl = virtual_token(rt, _fragment_of(rt, "CCl", nth), {"C", "Cl"}, "C", "Cl")
    mo, mc, mcl = map_of(rt, o), map_of(rt, c), map_of(rt, cl)
    datasets = [
        {"source_kind": "lone_pair", "source_atoms": [mo], "sink_kind": "atom", "sink_atoms": [mc]},
        {
            "source_kind": "bond",
            "source_atoms": [mc, mcl],
            "sink_kind": "atom",
            "sink_atoms": [mcl],
        },
    ]
    return apply_arrow(settings, rt, state, ArrowSpec(donor, o, acceptor, c), cl, datasets)


def proton_transfer_step(settings, runtime, state, donor_key: str) -> Step:
    """n(O) -> sigma*(C-H) of CCl at the sites (O, H): a deprotonation (the 'wrong' prediction
    of the corrected variant)."""
    from ksmech.serve.contracts import ArrowSpec

    rt = runtime.reactant_tokens(state)
    donor, o = lone_pair_token(rt, _fragment_of(rt, donor_key))
    frag = _fragment_of(rt, "CCl")
    acceptor, h, c = virtual_token(rt, frag, {"C", "H"}, "H", "C", exact=False)
    mo, mh, mc = map_of(rt, o), map_of(rt, h), map_of(rt, c)
    datasets = [
        {"source_kind": "lone_pair", "source_atoms": [mo], "sink_kind": "atom", "sink_atoms": [mh]},
        {"source_kind": "bond", "source_atoms": [mc, mh], "sink_kind": "atom", "sink_atoms": [mc]},
    ]
    return apply_arrow(settings, rt, state, ArrowSpec(donor, o, acceptor, h), c, datasets)


# ---- typed network models


def _hex(text: str, n: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def node_out(state, depth: int, *, root: bool, parent: str | None = None, logp: float = 0.0):
    from ksmech.serve.schemas import NodeOut, SpeciesCountOut

    species = [
        SpeciesCountOut(
            species_key=s.species_key,
            token_keys=list(s.token_keys),
            display_smiles=s.token_keys[0],
            count=s.count,
            charge=s.charge,
            resonance_flag=s.resonance_flag,
        )
        for s in state.species
    ]
    return NodeOut(
        id=state.pool_id,
        pool_key=state.pool_key,
        state_id=state.state_id,
        alias_state_ids=[],
        species=species,
        display_smiles=state.pool_key,
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


def edge_out(rt, step: Step, *, kind: str = "predicted", site_source: str = "rule", **kw):
    """The EdgeOut of an applied step (score 1.0, probability 0.5, rank 1 by default)."""
    from ksmech.serve.keys import edge_id
    from ksmech.serve.schemas import DatasetArrowOut, EdgeOut

    arrow = arrow_out(
        rt, step.state, step.arrow, site_source=site_source, sink=step.sink, source_kind="lone_pair"
    )
    maps = sorted({m for a in step.datasets for m in (*a["source_atoms"], *a["sink_atoms"])})
    base = dict(
        id=edge_id(kind, step.state.state_id, [step.arrow]),
        source=step.state.pool_id,
        target=step.product.pool_id,
        target_pool_key=step.product.pool_key,
        kind=kind,
        status="active",
        source_state_id=step.state.state_id,
        product_state_id=step.product.state_id,
        product_smiles=step.product.pool_key,
        source_mapped_smiles=step.state.mapped_smiles,
        product_mapped_smiles=step.product.mapped_smiles,
        arrows=[arrow],
        dataset_arrows=[DatasetArrowOut(**a) for a in step.datasets],
        pmechdb_code=step.code,
        encoding_base_smiles=step.state.mapped_smiles,
        encode_reason=None,
        touched_maps=maps,
        score=1.0,
        probability=0.5,
        logp=math.log(0.5),
        rank=1,
        n_alternatives=0,
        alternatives=[],
        predictor="stub",
        predictor_version=None,
        original=None,
        superseded_by=None,
        predictor_view=None,
        recurrence=False,
        closes_cycle=False,
        cycle_ids=[],
        revision=0,
    )
    return EdgeOut(**(base | kw))


def network_out(nodes, edges, pathway_edges, provenance, request, *, n_corrected: int = 0):
    """A finished NetworkOut with one pathway through `pathway_edges`."""
    from ksmech.serve.schemas import NetParamsOut, NetworkOut, PathwayOut

    root = nodes[0].id
    job_id = "j" + _hex("|".join(e.id for e in edges), 16)
    node_ids = [root] + [e.target for e in pathway_edges]
    pathway = PathwayOut(
        id="w" + _hex(job_id + "|path", 12),
        rank=1,
        node_ids=node_ids,
        edge_ids=[e.id for e in pathway_edges],
        logp=sum(e.logp for e in pathway_edges),
        probability=math.exp(sum(e.logp for e in pathway_edges)),
        n_steps=len(pathway_edges),
        n_corrected=n_corrected,
        ends_with="MAX_STEPS",
        closes_cycle=False,
        cycle_ids=[],
    )
    params = NetParamsOut(
        beam=2,
        max_steps=len(pathway_edges),
        temperature=1.0,
        stop_at_turnover=True,
        candidate_scope="all",
        max_scan=50,
        max_nodes=200,
        n_pathways=20,
    )
    return NetworkOut(
        network_id=job_id,
        revision=0,
        complete=True,
        root=root,
        nodes=nodes,
        edges=edges,
        cycles=[],
        pathways=[pathway],
        failed_proposals=[],
        skipped={},
        params=params,
        request=request,
        provenance=provenance,
    )


def predict_request(reactants: list[str], conditions: str | None = "water, 25 C"):
    from ksmech.serve.schemas import PredictIn

    return PredictIn(
        reactants_smiles=reactants,
        stoichiometry=[1] * len(reactants),
        conditions=conditions,
        predictor="stub",
        beam=2,
        max_steps=1,
    )


def stub_provenance(context):
    from ksmech.serve.schemas import Provenance

    return Provenance.model_validate(context.provenance("stub"))


def sn2_network(settings, runtime, context, state, donor_key: str, reactants: list[str]):
    """root -> SN2 product, one predicted edge, one pathway."""
    step = sn2_step(settings, runtime, state, donor_key)
    prov = stub_provenance(context)
    rt = runtime.reactant_tokens(state)
    edge = edge_out(rt, step, predictor_version=prov.predictor_version)
    nodes = [node_out(state, 0, root=True), node_out(step.product, 1, root=False, parent=edge.id)]
    return network_out(nodes, [edge], [edge], prov, predict_request(reactants))


def corrected_network(settings, runtime, context, state, donor_key: str, reactants: list[str]):
    """root -> SN2 product through a corrected edge; the original prediction (a proton
    transfer onto the other node) is kept with status superseded."""
    from ksmech.serve.keys import edge_id
    from ksmech.serve.schemas import EdgeOriginalOut, PredictorViewOut

    prov = stub_provenance(context)
    rt = runtime.reactant_tokens(state)
    wrong = proton_transfer_step(settings, runtime, state, donor_key)
    right = sn2_step(settings, runtime, state, donor_key)
    corrected_id = edge_id("corrected", state.state_id, [right.arrow])
    original = edge_out(
        rt,
        wrong,
        predictor_version=prov.predictor_version,
        status="superseded",
        superseded_by=corrected_id,
        score=2.0,
        probability=0.7,
        logp=math.log(0.7),
    )
    keep = ("edge_id", "target", "target_pool_key", "arrows", "score", "probability", "rank")
    orig = EdgeOriginalOut(
        **{k: getattr(original, "id" if k == "edge_id" else k) for k in keep},
        predictor=original.predictor,
        predictor_version=original.predictor_version,
    )
    view = PredictorViewOut(score=1.0, probability=0.2, rank=2, pair_valid=True, reason=None)
    edge = edge_out(
        rt,
        right,
        kind="corrected",
        site_source="user",
        score=None,
        probability=None,
        logp=0.0,
        rank=None,
        predictor=None,
        predictor_version=None,
        original=orig,
        predictor_view=[view],
    )
    nodes = [
        node_out(state, 0, root=True),
        node_out(right.product, 1, root=False, parent=edge.id),
        node_out(wrong.product, 1, root=False, parent=original.id),
    ]
    edges = [original, edge]
    return network_out(nodes, edges, [edge], prov, predict_request(reactants), n_corrected=1)


def double_sn2_network(settings, runtime, context):
    """2 MeO- + 2 MeCl: root -> mid -> final, two independent SN2 steps (disjoint maps,
    continuous states)."""
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    state = initial_state([SpeciesInput("C[O-]", 2), SpeciesInput("CCl", 2)], settings.limits)
    save_state(state, settings)
    prov = stub_provenance(context)
    first = sn2_step(settings, runtime, state, "C[O-]")
    second = sn2_step(settings, runtime, first.product, "C[O-]")
    e1 = edge_out(runtime.reactant_tokens(state), first, predictor_version=prov.predictor_version)
    e2 = edge_out(
        runtime.reactant_tokens(first.product), second, predictor_version=prov.predictor_version
    )
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
    from ksmech.serve import db

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
        from ksmech.serve.errors import Code, ServeError

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
        from ksmech.serve.contracts import LocalJobContext

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
    from ksmech.serve.keys import side_keys

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


@requires("correct")  # the fixtures apply arrows with apply_predicted_arrows
def test_sn2_networks_are_typed_and_consistent(oh_sn2_network, meo_sn2_network):
    for net, root in ((oh_sn2_network, "CCl.[OH-]"), (meo_sn2_network, "CCl.C[O-]")):
        (edge,) = net.edges
        (arrow,) = edge.arrows
        assert net.nodes[0].pool_key == root and net.root == net.nodes[0].id
        assert (arrow.donor_element, arrow.acceptor_element) == ("O", "C")
        assert edge.target_pool_key.endswith("[Cl-]")
        c, cl = arrow.acceptor_site_map, arrow.sink_partner_map
        bond = ",".join(str(m) for m in sorted((c, cl)))
        assert edge.pmechdb_code == f"{arrow.donor_site_map}={c};{bond}={cl}"
        assert net.pathways[0].edge_ids == [edge.id]
        assert net.provenance.predictor == "stub" and edge.predictor_version
    assert oh_sn2_network.edges[0].target_pool_key == "CO.[Cl-]"
    assert meo_sn2_network.edges[0].target_pool_key == "COC.[Cl-]"


@requires("correct")  # the fixtures apply arrows with apply_predicted_arrows
def test_product_states_are_saved(meo_sn2_network, synthetic_settings):  # noqa: F811
    from ksmech.serve.pools import load_state

    edge = meo_sn2_network.edges[0]
    assert load_state(edge.product_state_id, synthetic_settings).pool_key == "COC.[Cl-]"


@requires("correct")  # the fixtures apply arrows with apply_predicted_arrows
def test_corrected_variant(meo_corrected_network):
    original, corrected = meo_corrected_network.edges
    assert original.status == "superseded" and original.superseded_by == corrected.id
    assert corrected.kind == "corrected" and corrected.original.edge_id == original.id
    assert corrected.target_pool_key == "COC.[Cl-]"
    assert original.target_pool_key != corrected.target_pool_key
    assert original.arrows[0].acceptor_element == "H"
    assert meo_corrected_network.pathways[0].edge_ids == [corrected.id]


@requires("correct")  # the fixtures apply arrows with apply_predicted_arrows
def test_double_network_is_a_chain(double_network):
    e1, e2 = double_network.edges
    assert e1.target == e2.source and e1.product_state_id == e2.source_state_id
    assert not set(e1.touched_maps) & set(e2.touched_maps)
    assert e2.target_pool_key == "COC.COC.[Cl-].[Cl-]"


def test_mini_pairs(serve_mini_processed):  # noqa: F811
    pairs = mini_pairs(serve_mini_processed)
    assert ("[CH3][Cl]|0.[OH]|-1", "[CH3][OH]|0.[Cl]|-1") in pairs  # sigma pool keys
