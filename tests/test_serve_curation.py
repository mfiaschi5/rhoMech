"""Curation: CurateService actions and errors, StepDoc fields, the status transition table,
order groups, list filters, staleness, and the re-check job (featurize first, then the worker
task; Recheck rows; the status never changes)."""

import dataclasses
import json
import shutil
from functools import partial

import pytest
from sqlmodel import select

from ksmech.serve import db
from ksmech.serve.contracts import LocalJobContext, PredictorInfo
from ksmech.serve.curate_service import CurateService
from ksmech.serve.curation import (
    STATUSES,
    check_order_groups,
    independence_issues,
    original_doc,
    staleness,
    step_doc,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.export import run_export
from ksmech.serve.recheck import UNCHECKED, run_recheck
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
from tests.test_serve_fixtures_curation import (  # noqa: F401 (fixtures)
    InlineJobs,
    curation_engine,
    double_network,
    meo_corrected_network,
    meo_sn2_network,
    network_store,
    oh_sn2_network,
)

pytestmark = pytest.mark.serve


@pytest.fixture
def jobs(synthetic_settings, synthetic_context, curation_engine, tmp_path):  # noqa: F811
    from ksmech.serve.settings import load_settings

    export_settings = load_settings({}, data_root=tmp_path / "export-root")
    handlers = {
        "recheck": partial(
            run_recheck, engine=curation_engine, registry=synthetic_context.registry
        ),
        "export": partial(run_export, engine=curation_engine, pairs=lambda ctx: None),
    }
    return InlineJobs(
        synthetic_settings,
        synthetic_context.scheduler,
        handlers,
        settings_for={"export": export_settings},
    )


@pytest.fixture
def service(synthetic_context, curation_engine, network_store, jobs):  # noqa: F811
    return CurateService(synthetic_context, curation_engine, networks=network_store, submit=jobs)


def _create(service, network, action="validate", **kw):
    body = {"job_id": network.network_id, "pathway_id": network.pathways[0].id, "action": action}
    return service.create(body | kw)


def _raises(code, status, fn, *args, **kw) -> ServeError:
    with pytest.raises(ServeError) as info:
        fn(*args, **kw)
    assert (str(info.value.code), info.value.status) == (str(code), status), info.value.message
    return info.value


# ---- create


@pytest.mark.parametrize(
    "action, status", [("validate", "validated"), ("save", "predicted"), ("reject", "rejected")]
)
def test_create_actions(service, oh_sn2_network, action, status):  # noqa: F811
    out = _create(service, oh_sn2_network, action, notes="looks right")
    assert out.status == status and out.n_steps == 1 and out.notes == "looks right"
    assert out.reactants == ["[OH-]", "CCl"] and out.stoichiometry == [1, 1]
    assert out.conditions == "water, 25 C"
    assert out.predictor == "stub" and out.checkpoint_hash.startswith("stub/")
    assert out.spec_version == oh_sn2_network.provenance.spec_version
    (step,) = out.steps
    assert (step.reactant_pool_key, step.product_pool_key) == ("CCl.[OH-]", "CO.[Cl-]")
    assert step.order_group == 0 and not step.was_corrected and step.original is None
    assert not out.stale and out.stale_reasons == [] and out.warnings == []
    assert service.get(out.id).model_dump() == out.model_dump()


def test_validate_with_a_corrected_edge(service, meo_corrected_network):  # noqa: F811
    out = _create(service, meo_corrected_network, "validate", conditions="DMSO")
    assert out.status == "corrected" and out.conditions == "DMSO"
    (step,) = out.steps
    original, corrected = meo_corrected_network.edges
    assert step.was_corrected and step.step["edge_kind"] == "corrected"
    assert step.original["edge_id"] == original.id and step.original["dataset_arrows"] is None
    assert step.original["score"] == 2.0 and step.original["probability"] == 0.7
    assert step.original["products_mapped"] == original.product_mapped_smiles
    assert step.original["arrows"][0]["acceptor_element"] == "H"


def test_step_doc_fields(oh_sn2_network):  # noqa: F811
    edge, prov = oh_sn2_network.edges[0], oh_sn2_network.provenance
    doc = step_doc(edge, job_id="j0123456789abcdef", provenance=prov)
    assert doc["format"] == "ksmech-serve/step-v1" and doc["job_id"] == "j0123456789abcdef"
    assert (doc["edge_id"], doc["edge_kind"]) == (edge.id, "predicted")
    assert doc["reactant_state_id"] == edge.source_state_id
    assert doc["product_state_id"] == edge.product_state_id
    assert doc["reactants_mapped"] == edge.encoding_base_smiles
    assert doc["products_mapped"] == edge.product_mapped_smiles
    assert doc["dataset_arrows"] == [a.model_dump() for a in edge.dataset_arrows]
    assert doc["pmechdb_code"] == edge.pmechdb_code and doc["encode_reason"] is None
    assert (doc["score"], doc["probability"], doc["rank"]) == (1.0, 0.5, 1)
    assert (doc["predictor"], doc["predictor_version"]) == ("stub", edge.predictor_version)
    assert doc["spec_version"] == prov.spec_version
    assert doc["physics_fingerprint"] == prov.physics_fingerprint
    assert doc["touched_maps"] == edge.touched_maps
    (arrow,) = doc["arrows"]
    a = edge.arrows[0]
    assert arrow["donor_centres_map"] == a.donor_centres_map
    assert arrow["acceptor_centres_map"] == a.acceptor_centres_map
    assert (arrow["donor_site_map"], arrow["sink_partner_map"]) == (
        a.donor_site_map,
        a.sink_partner_map,
    )
    keys = "donor donor_site acceptor acceptor_site donor_site_map acceptor_site_map"
    keys += " donor_element acceptor_element source_kind source_partner_map sink_partner_map"
    assert set(arrow) == set((keys + " donor_centres_map acceptor_centres_map").split())
    json.dumps(doc, allow_nan=False)  # JSON-able


def test_step_doc_falls_back_to_the_source_state(oh_sn2_network):  # noqa: F811
    edge = oh_sn2_network.edges[0].model_copy(update={"encoding_base_smiles": None})
    doc = step_doc(edge, job_id="j", provenance=oh_sn2_network.provenance)
    assert doc["reactants_mapped"] == edge.source_mapped_smiles


def test_original_doc_without_the_superseded_edge(meo_corrected_network):  # noqa: F811
    corrected = meo_corrected_network.edges[1]
    doc = original_doc(corrected, job_id="j", provenance=meo_corrected_network.provenance)
    assert doc["edge_id"] == corrected.original.edge_id and doc["products_mapped"] is None
    assert doc["target_pool_key"] == corrected.original.target_pool_key
    assert doc["reactants_mapped"] == corrected.source_mapped_smiles
    assert original_doc(meo_corrected_network.edges[0], job_id="j", provenance={}) is None


def test_create_errors(service, oh_sn2_network, meo_corrected_network):  # noqa: F811
    net = oh_sn2_network
    edge = net.edges[0].id
    base = {"job_id": net.network_id, "action": "validate"}
    _raises(Code.REQUEST_INVALID, 422, service.create, base)  # neither
    both = base | {"pathway_id": net.pathways[0].id, "edge_ids": [edge]}
    _raises(Code.REQUEST_INVALID, 422, service.create, both)
    _raises(Code.REQUEST_INVALID, 422, service.create, base | {"edge_ids": []})
    _raises(Code.PATHWAY_NOT_FOUND, 404, service.create, base | {"pathway_id": "w" + "0" * 12})
    _raises(Code.EDGE_NOT_FOUND, 404, service.create, base | {"edge_ids": ["e" + "0" * 16]})
    missing = base | {"job_id": "j" + "0" * 16, "edge_ids": [edge]}
    _raises(Code.JOB_NOT_FOUND, 404, service.create, missing)
    old = meo_corrected_network.edges[0].id
    stale = {"job_id": meo_corrected_network.network_id, "action": "save", "edge_ids": [old]}
    _raises(Code.REQUEST_INVALID, 422, service.create, stale)
    assert service.list() == []  # nothing was written


def test_create_by_edge_ids_and_chain_check(service, double_network):  # noqa: F811
    e1, e2 = double_network.edges
    base = {"job_id": double_network.network_id, "action": "save"}
    out = service.create(base | {"edge_ids": [e1.id, e2.id]})
    assert [s.step_index for s in out.steps] == [0, 1]
    assert [s.order_group for s in out.steps] == [0, 1]
    assert out.steps[1].reactant_pool_key == "CCl.COC.C[O-].[Cl-]"
    err = _raises(Code.REQUEST_INVALID, 422, service.create, base | {"edge_ids": [e2.id, e1.id]})
    assert "chain" in err.message


# ---- order groups


def test_order_groups_on_create(service, double_network, oh_sn2_network):  # noqa: F811
    e1, e2 = double_network.edges
    base = {"job_id": double_network.network_id, "action": "save", "edge_ids": [e1.id, e2.id]}
    out = service.create(base | {"order_groups": [0, 0]})  # independent, continuous maps
    assert [s.order_group for s in out.steps] == [0, 0] and out.warnings == []
    for bad in ([1, 1], [0, 2], [1, 0], [0], [0, 1, 2]):
        _raises(Code.ORDER_GROUPS_INVALID, 422, service.create, base | {"order_groups": bad})
    one = {
        "job_id": oh_sn2_network.network_id,
        "action": "save",
        "edge_ids": [oh_sn2_network.edges[0].id],
    }
    assert service.create(one | {"order_groups": [0]}).steps[0].order_group == 0


def test_not_independent_and_unchecked(service, network_store, double_network):  # noqa: F811
    e1, e2 = double_network.edges
    overlapping = e2.model_copy(update={"touched_maps": [*e2.touched_maps, e1.touched_maps[0]]})
    net = double_network.model_copy(update={"edges": [e1, overlapping]})
    net = net.model_copy(update={"network_id": "j" + "1" * 16})
    network_store.add(net)
    base = {"job_id": net.network_id, "action": "save", "edge_ids": [e1.id, e2.id]}
    err = _raises(Code.NOT_INDEPENDENT, 422, service.create, base | {"order_groups": [0, 0]})
    assert err.detail == {"group": 0, "steps": [0, 1], "shared_maps": [e1.touched_maps[0]]}
    assert service.create(base | {"order_groups": [0, 1]}).warnings == []  # ordered: fine
    merged = overlapping.model_copy(update={"source_state_id": "s" + "2" * 20})
    net2 = net.model_copy(update={"edges": [e1, merged], "network_id": "j" + "2" * 16})
    network_store.add(net2)
    out = service.create(base | {"job_id": net2.network_id, "order_groups": [0, 0]})
    (warning,) = out.warnings
    assert warning.startswith("INDEPENDENCE_UNCHECKED")


def test_independence_issues_runs():
    def doc(src, dst, maps):
        return {"reactant_state_id": src, "product_state_id": dst, "touched_maps": maps}

    docs = [doc("a", "b", [1, 2]), doc("b", "c", [3]), doc("x", "y", [1]), doc("y", "z", [4])]
    warnings, conflicts = independence_issues(docs, [0, 0, 0, 0])
    assert len(warnings) == 1 and "steps 1 and 2" in warnings[0] and not conflicts
    _, conflicts = independence_issues(docs[:2] + [doc("c", "d", [2])], [0, 0, 0])
    assert [c.detail["steps"] for c in conflicts] == [[0, 2]]  # every pair of a run
    assert independence_issues(docs, [0, 1, 2, 3]) == ([], [])
    assert check_order_groups(None, 3) == [0, 1, 2]


def test_patch_order_groups(service, double_network):  # noqa: F811
    e1, e2 = double_network.edges
    base = {"job_id": double_network.network_id, "action": "save", "edge_ids": [e1.id, e2.id]}
    out = service.create(base)
    assert [s.order_group for s in service.patch(out.id, {"order_groups": [0, 0]}).steps] == [0, 0]
    _raises(Code.ORDER_GROUPS_INVALID, 422, service.patch, out.id, {"order_groups": [0, 2]})
    service.patch(out.id, {"status": "rejected"})
    _raises(Code.INVALID_TRANSITION, 409, service.patch, out.id, {"order_groups": [0, 1]})
    reopened = service.patch(out.id, {"status": "predicted", "order_groups": [0, 1]})
    assert [s.order_group for s in reopened.steps] == [0, 1]


# ---- statuses


def _insert(engine, status: str, corrected: bool, predictor="stub", version="stub/x/y", **kw):
    with db.session(engine) as s:
        m = db.Mechanism(
            reactants=json.dumps(["[OH-]", "CCl"]),
            stoichiometry=json.dumps([1, 1]),
            predictor=predictor,
            checkpoint_hash=version,
            spec_version=kw.pop("spec_version", "v3+test"),
            status=status,
            **kw,
        )
        s.add(m)
        s.flush()
        doc = {"format": "ksmech-serve/step-v1", "physics_fingerprint": "fp", "touched_maps": [1]}
        s.add(
            db.CuratedStep(
                mechanism_id=m.id,
                step_index=0,
                order_group=0,
                reactant_pool_key="CCl.[OH-]",
                product_pool_key="CO.[Cl-]",
                arrows_json=json.dumps(doc),
                was_corrected=corrected,
            )
        )
    return m.id


# the design table (9.3), written out independently of curation.TRANSITIONS
ALLOWED = {
    "predicted": {"validated", "corrected", "rejected"},
    "validated": {"rejected"},
    "corrected": {"rejected"},
    "rejected": {"predicted"},
}


@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("old", STATUSES)
@pytest.mark.parametrize("new", STATUSES)
def test_transition_table(service, curation_engine, old, new, corrected):  # noqa: F811
    mid = _insert(curation_engine, old, corrected)
    if new == old:
        out = service.patch(mid, {"status": new})
        assert out.status == old and "status" not in out.notes  # a no-op
    elif new not in ALLOWED[old]:
        err = _raises(Code.INVALID_TRANSITION, 409, service.patch, mid, {"status": new})
        assert err.detail["from"] == old and err.detail["to"] == new
    elif (new == "validated" and corrected) or (new == "corrected" and not corrected):
        _raises(Code.STATUS_INCONSISTENT, 422, service.patch, mid, {"status": new})
    else:
        out = service.patch(mid, {"status": new})
        assert out.status == new
        assert out.notes.startswith("\n[") and out.notes.endswith(f"] status {old} -> {new}")
        return
    assert service.get(mid).status == old  # unchanged after a refusal or a no-op


def test_notes_edit_and_audit_lines(service, oh_sn2_network):  # noqa: F811
    mid = _create(service, oh_sn2_network, "save").id
    out = service.patch(mid, {"notes": "checked by hand", "status": "rejected"})
    assert out.notes.startswith("checked by hand\n[") and out.notes.endswith(
        "predicted -> rejected"
    )
    out = service.patch(mid, {"status": "predicted"})
    assert out.notes.count("\n[") == 2 and out.notes.endswith("rejected -> predicted")
    _raises(Code.MECHANISM_NOT_FOUND, 404, service.get, 999)
    _raises(Code.MECHANISM_NOT_FOUND, 404, service.patch, 999, {"notes": "x"})


# ---- list and staleness


def test_list_filters(service, curation_engine, oh_sn2_network, meo_sn2_network):  # noqa: F811
    a = _create(service, oh_sn2_network, "validate").id
    b = _create(service, meo_sn2_network, "save").id
    c = _insert(curation_engine, "rejected", False, predictor="gbdt", version="gbdt/aaaa/fp")
    assert [m.id for m in service.list()] == [c, b, a]  # newest first
    assert [m.id for m in service.list(status="validated")] == [a]
    assert [m.id for m in service.list(predictor="gbdt")] == [c]
    assert [m.id for m in service.list(stale=True)] == [c]
    assert [m.id for m in service.list(stale=False)] == [b, a]
    assert [m.id for m in service.list(limit=1, offset=1)] == [b]
    assert service.list(status="corrected") == []
    summary = service.list(status="validated")[0]
    assert summary.n_steps == 1 and summary.last_recheck is None and not summary.stale


def test_staleness_reasons(service, curation_engine, oh_sn2_network, monkeypatch):  # noqa: F811
    mid = _create(service, oh_sn2_network).id
    ctx = service.ctx
    with db.session(curation_engine) as s:
        m = s.get(db.Mechanism, mid)
        steps = list(s.exec(select(db.CuratedStep).where(db.CuratedStep.mechanism_id == mid)))
    assert staleness(m, steps, ctx.registry, ctx.spec_version, ctx.physics_fp) == []
    fp = "0" * 12
    reasons = staleness(m, steps, ctx.registry, ctx.spec_version, fp)
    assert reasons == [f"physics_fingerprint {ctx.physics_fp} != {fp}"]
    reasons = staleness(m, steps, ctx.registry, "v3+other", ctx.physics_fp)
    assert reasons == [f"spec_version {ctx.spec_version} != v3+other"]
    other = dataclasses.replace(ctx, physics_fp=fp)
    out = CurateService(other, curation_engine, networks=None, submit=None).get(mid)
    assert out.stale and out.stale_reasons == [f"physics_fingerprint {ctx.physics_fp} != {fp}"]


def test_gbdt_version_staleness(service, curation_engine, monkeypatch):  # noqa: F811
    spec = service.ctx.spec_version
    mid = _insert(curation_engine, "validated", False, "gbdt", "gbdt/aaaa/fp", spec_version=spec)
    out = service.get(mid)
    assert out.stale and out.stale_reasons[0] == "predictor_unavailable"  # no gbdt file
    registry = service.ctx.registry
    real = registry.info_of("floor")

    def retrained(name):
        if name != "gbdt":
            return registry.__class__.info_of(registry, name)
        fields = real.to_dict() | {"name": "gbdt", "version": "gbdt/bbbb/fp", "warnings": []}
        return PredictorInfo.from_dict(fields)

    monkeypatch.setattr(registry, "info_of", retrained)
    out = service.get(mid)
    assert out.stale_reasons[0] == "predictor_version gbdt/aaaa/fp != gbdt/bbbb/fp"
    assert [m.id for m in service.list(stale=True)] == [mid]


# ---- export and re-check through the service


def test_export_job_and_latest(service, jobs, oh_sn2_network):  # noqa: F811
    _raises(Code.EXPORT_NOT_FOUND, 404, service.latest_export)
    _create(service, oh_sn2_network, "validate")
    created = service.export({})
    assert created.kind == "export" and created.status == "done"
    report = jobs.results[created.job_id]
    assert report["n_steps_written"] == 1 and report["warnings"][0].startswith(
        "PMECHDB_UNAVAILABLE"
    )
    latest = service.latest_export()
    assert latest.export_id == report["export_id"] and latest.sha256 == report["sha256"]


def test_recheck_service_errors(service, oh_sn2_network):  # noqa: F811
    _raises(Code.PREDICTOR_UNKNOWN, 422, service.recheck, {"predictor": "nope"})
    _raises(Code.PREDICTOR_UNAVAILABLE, 409, service.recheck, {"predictor": "transformer"})
    _raises(Code.MECHANISM_NOT_FOUND, 404, service.recheck_one, 42, {})
    mid = _create(service, oh_sn2_network).id
    _raises(Code.PREDICTOR_UNAVAILABLE, 409, service.recheck_one, mid, {"predictor": "gbdt"})


def _rows(engine, mid):
    with db.session(engine) as s:
        q = select(db.Recheck).where(db.Recheck.mechanism_id == mid).order_by(db.Recheck.id)
        return list(s.exec(q).all())


def test_recheck_job_fills_rows(service, jobs, curation_engine, oh_sn2_network):  # noqa: F811
    mid = _create(service, oh_sn2_network, "validate").id
    created = service.recheck_one(mid, {})
    report = jobs.results[created.job_id]
    assert report["n_steps"] == 1 and report["n_checked"] == 1 and report["skipped"] == {}
    (row,) = _rows(curation_engine, mid)
    assert row.aligned and row.reason is None and row.predictor == "stub"
    assert row.first_arrow_rank == 1 and row.top1 and row.top5 and row.n_valid > 0
    assert row.product_match in ("exact", "resonance")
    assert row.top1_reproduces_product is True  # the stub's rule sites (O, C)
    assert row.physics_fingerprint == service.ctx.physics_fp
    assert row.spec_version == service.ctx.spec_version
    assert report["top1"] == 1.0 and report["provenance"]["predictor"] == "stub"
    out = service.get(mid)
    assert out.status == "validated"  # never changed by a re-check
    assert out.steps[0].recheck.first_arrow_rank == 1 and out.last_recheck is not None
    assert len(out.rechecks) == 1
    # stale_only (the default): an up-to-date mechanism is not re-checked
    created = service.recheck({})
    assert jobs.results[created.job_id]["n_steps"] == 0


def _make_stale(engine, mid: int) -> None:
    """An older spec version on the mechanism row: stale, but its steps re-check cleanly."""
    with db.session(engine) as s:
        s.get(db.Mechanism, mid).spec_version = "v3+older"


def test_stale_only_after_recheck(service, jobs, curation_engine, oh_sn2_network, meo_sn2_network):  # noqa: F811
    """stale_only leaves out the stale mechanisms whose every step was re-checked at the
    current versions, counted RECHECKED_AT_CURRENT_VERSIONS; the stored versions stay."""
    mids = [_create(service, net, "validate").id for net in (oh_sn2_network, meo_sn2_network)]
    for mid in mids:
        _make_stale(curation_engine, mid)
    first = jobs.results[service.recheck({}).job_id]
    assert first["n_steps"] == 2 and first["n_checked"] == 2 and first["skipped"] == {}
    second_id = service.recheck({}).job_id
    second = jobs.results[second_id]
    assert second["n_steps"] == 0 and second["rows"] == []
    assert second["skipped"] == {"RECHECKED_AT_CURRENT_VERSIONS": len(mids)}
    skips = jobs.contexts[second_id].skip_log
    assert sorted(r["mechanism_id"] for r in skips) == mids
    assert all(r["code"] == "RECHECKED_AT_CURRENT_VERSIONS" for r in skips)
    assert all(len(_rows(curation_engine, mid)) == 1 for mid in mids)
    out = service.get(mids[0])
    assert out.stale and out.stale_reasons[0].startswith("spec_version v3+older")  # not rewritten
    # another predictor is not covered by the stub's rows; explicit ids without stale_only re-check
    floor = jobs.results[service.recheck({"predictor": "floor"}).job_id]
    assert floor["n_steps"] == 2 and "RECHECKED_AT_CURRENT_VERSIONS" not in floor["skipped"]
    again = jobs.results[service.recheck_one(mids[0], {}).job_id]
    assert again["n_steps"] == 1 and again["skipped"] == {}


def test_stale_only_repeats_a_failed_recheck(service, jobs, curation_engine):  # noqa: F811
    """A step whose latest re-check failed (reason set) is re-checked again by stale_only."""
    mid = _insert(curation_engine, "validated", False)  # stale: older versions throughout
    for _ in range(2):  # the StepDoc has no dataset arrows: ARROW_ENCODING_FAILED each time
        report = jobs.results[service.recheck({}).job_id]
        assert report["n_steps"] == 1 and report["skipped"] == {"ARROW_ENCODING_FAILED": 1}
    assert [r.reason for r in _rows(curation_engine, mid)] == ["ARROW_ENCODING_FAILED"] * 2


def test_recheck_with_the_floor(service, jobs, curation_engine, oh_sn2_network):  # noqa: F811
    mid = _create(service, oh_sn2_network).id
    report = jobs.results[service.recheck_one(mid, {"predictor": "floor"}).job_id]
    (row,) = _rows(curation_engine, mid)
    assert row.predictor == "floor" and row.predictor_version.startswith("floor/")
    assert row.aligned and row.first_arrow_rank is not None and row.first_arrow_rank >= 1
    assert row.top1 == (row.first_arrow_rank == 1)
    assert report["provenance"]["predictor"] == "floor"


def _doc(network) -> dict:
    return step_doc(network.edges[0], job_id="j", provenance=network.provenance)


def test_recheck_reasons(curation_engine, synthetic_context, synthetic_settings, oh_sn2_network):  # noqa: F811
    doc = _doc(oh_sn2_network)
    no_arrows = doc | {"dataset_arrows": None, "encode_reason": "SOURCE_UNDERIVABLE"}
    mid = _insert(curation_engine, "validated", False)
    with db.session(curation_engine) as s:
        s.get(db.CuratedStep, (mid, 0)).arrows_json = json.dumps(no_arrows)
    gbdt = _insert(curation_engine, "validated", False, predictor="gbdt", version="gbdt/a/b")
    ctx = LocalJobContext("j" + "3" * 16, synthetic_settings, synthetic_context.scheduler)
    request = {"mechanism_ids": [mid, gbdt, 777], "stale_only": False}
    report = run_recheck(ctx, request, engine=curation_engine, registry=synthetic_context.registry)
    want = {"ARROW_ENCODING_FAILED": 1, "MECHANISM_NOT_FOUND": 1, "PREDICTOR_UNAVAILABLE": 1}
    assert report["skipped"] == want
    assert report["n_steps"] == 2 and report["n_checked"] == 0 and report["top1"] is None
    assert report["product_match"] == {UNCHECKED: 2}
    assert dict(ctx.skipped) == report["skipped"]
    (row,) = _rows(curation_engine, mid)
    assert row.reason == "ARROW_ENCODING_FAILED" and not row.aligned


@pytest.fixture
def fresh_settings(tmp_path, synthetic_settings, synthetic_loader):  # noqa: F811
    """A synthetic data root holding every synthetic token file except [OH-]'s."""
    from ksmech.serve.featurize import qm_npz_path
    from ksmech.serve.settings import load_settings
    from ksmech.serve.synthetic import SYNTHETIC_METHOD

    fresh = load_settings(
        {}, data_root=tmp_path / "fresh", chemistry="synthetic", enable_stub_predictor=True
    )
    for sub in ("tokens", "qm", "oriented"):
        if (synthetic_settings.serve / sub).is_dir():
            shutil.copytree(synthetic_settings.serve / sub, fresh.serve / sub)
    (fresh.serve_tokens / synthetic_loader.path_of("[OH-]").name).unlink()
    qm = qm_npz_path(fresh, "[OH-]", SYNTHETIC_METHOD)
    if qm is not None:
        qm.unlink()
    return fresh


def test_recheck_featurizes_first(fresh_settings, curation_engine, oh_sn2_network):  # noqa: F811
    """A reactant key missing from the loader is featurized (JOB priority) before the
    recheck_step task, which itself never runs QM."""
    from ksmech.serve.predictors import PredictorRegistry
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.versions import physics_fingerprint
    from ksmech.serve.worker import InlineScheduler

    fresh = fresh_settings
    mid = _insert(curation_engine, "validated", False)
    with db.session(curation_engine) as s:
        s.get(db.CuratedStep, (mid, 0)).arrows_json = json.dumps(_doc(oh_sn2_network))
    scheduler = InlineScheduler(TaskEnv(fresh))
    ctx = LocalJobContext("j" + "4" * 16, fresh, scheduler)
    registry = PredictorRegistry(fresh, physics_fingerprint())
    report = run_recheck(ctx, {"stale_only": False}, engine=curation_engine, registry=registry)
    names = [name for name, _ in scheduler.history]
    assert names == ["featurize", "recheck_step"]
    assert "[OH-]" in scheduler.history[0][1]
    assert report["n_checked"] == 1 and report["rows"][0]["aligned"]
    assert any(p["phase"] == "featurize" for p in ctx.progress_log)


class _TimeoutScheduler:
    """Delegates to a real scheduler but times out every recheck_step task."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def run(self, name, payload, **kw):
        from ksmech.serve.worker import TaskResult

        self.calls.append((name, kw["priority"], kw["timeout_s"]))
        if name == "recheck_step":
            return TaskResult.failure("WORKER_TIMEOUT", "recheck_step timed out")
        return self.inner.run(name, payload, **kw)


def test_recheck_task_fails(curation_engine, synthetic_context, synthetic_settings, oh_sn2_network):  # noqa: F811
    from ksmech.serve.worker import Priority

    mid = _insert(curation_engine, "validated", False)
    with db.session(curation_engine) as s:
        s.get(db.CuratedStep, (mid, 0)).arrows_json = json.dumps(_doc(oh_sn2_network))
    scheduler = _TimeoutScheduler(synthetic_context.scheduler)
    ctx = LocalJobContext("j" + "5" * 16, synthetic_settings, scheduler)
    registry = synthetic_context.registry
    report = run_recheck(ctx, {"stale_only": False}, engine=curation_engine, registry=registry)
    assert scheduler.calls == [("recheck_step", Priority.JOB, 900)]  # tokens present: no featurize
    assert report["skipped"] == {"WORKER_TIMEOUT": 1}
    (row,) = _rows(curation_engine, mid)
    assert row.reason == "WORKER_TIMEOUT" and row.product_match == UNCHECKED
    assert ctx.skip_log[0]["detail"] == "recheck_step timed out"
