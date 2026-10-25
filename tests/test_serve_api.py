"""The HTTP layer of ksmech-serve over the synthetic app (design 6, 13.2): meta, predictors,
normalize errors, a stub predict whose network arrows carry the full physics brief, provenance
on every prediction response, every state id of the network resolvable, the /pool and /contact
endpoints, an override with the original kept, the 202 Pending shape, id validation, the error
envelope, and the browse index endpoints."""

from __future__ import annotations

import logging
import math
import re

import pytest

from ksmech.serve import schemas as S
from ksmech.serve.capabilities import CAPABILITIES
from ksmech.serve.testing import requires
from tests.test_serve_fixtures import serve_mini_processed  # noqa: F401 (fixture)
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    CH_TUPLE,
    SN2_REACTANTS,
    SN2_REQUEST,
    SN2_TUPLE,
    SPEC_RE,
    NeverScheduler,
    api_client,
    arrow_of,
    candidates,
    edge_to,
    fresh_synthetic_settings,
    no_xtb,
    predict,
    root_node,
    serve_settings,
    services,
    synthetic_app,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve

ERROR_KEYS = {"code", "message", "upstream", "detail", "request_id"}


def error_of(r, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == ERROR_KEYS
    assert body["error"]["code"] == code, body
    assert body["error"]["request_id"] == r.headers["x-request-id"]
    return body["error"]


@pytest.fixture(scope="module")
def sn2_job(synthetic_app):  # noqa: F811
    """One stub predict of MeO- + MeCl (beam 2, max steps 1), done."""
    return predict(synthetic_app)


@pytest.fixture(scope="module")
def sn2_root(synthetic_app):  # noqa: F811
    """The root state of MeO- + MeCl from POST /api/pools (tokens only: no analysis)."""
    r = synthetic_app.post("/api/pools", json={"reactants_smiles": SN2_REACTANTS})
    assert r.status_code == 200, r.text
    return r.json()["state_id"]


@pytest.fixture(scope="module")
def sn2_donor(synthetic_app, sn2_root):  # noqa: F811
    """A doubly occupied orbital token of the SN2 root (global id)."""
    tokens = synthetic_app.get(f"/api/pool/{sn2_root}/tokens").json()
    return next(o["token_id"] for o in tokens["orbitals"] if o["model_input"]["occupation"] > 1.5)


@pytest.fixture(scope="module")
def sn2_tuple(synthetic_app, sn2_root):  # noqa: F811
    rows = candidates(synthetic_app, sn2_root, **SN2_TUPLE)
    assert rows, "the O -> sigma*(C-Cl) tuple at sites (O, C) is not a valid candidate"
    return rows[0]


# ---- meta, health, predictors


def test_meta(synthetic_app):  # noqa: F811
    r = synthetic_app.get("/api/meta")
    assert r.status_code == 200
    m = S.MetaOut.model_validate(r.json())
    assert re.match(SPEC_RE, m.spec_version), m.spec_version
    assert re.fullmatch(r"[0-9a-f]{12}", m.physics_fingerprint)
    assert m.chemistry == "synthetic"
    assert m.qm_method == m.token_method == "SYNTHETIC-EHT/v1"
    assert m.primary_distance_ang == 2.3
    assert m.dev is False
    assert isinstance(m.frontend_built, bool)
    assert m.worker.alive


@requires("predict")
def test_predictors(synthetic_app):  # noqa: F811
    r = synthetic_app.get("/api/predictors")
    assert r.status_code == 200
    infos = {p["name"]: S.PredictorInfoOut.model_validate(p) for p in r.json()}
    assert {"floor", "gbdt", "transformer", "stub"} <= set(infos)
    assert infos["floor"].available and infos["stub"].available
    assert not infos["transformer"].available
    assert infos["transformer"].code == "TRANSFORMER_NOT_AVAILABLE"
    assert not infos["gbdt"].available
    assert infos["gbdt"].code == "GBDT_MISSING"


@requires("predict")
def test_predictors_and_job_created_carry_the_spec_version(synthetic_app):  # noqa: F811
    spec = synthetic_app.get("/api/meta").json()["spec_version"]
    infos = synthetic_app.get("/api/predictors").json()
    assert infos and all(S.PredictorInfoOut.model_validate(p).spec_version == spec for p in infos)
    r = synthetic_app.post("/api/predict", json=SN2_REQUEST)
    assert r.status_code == 202, r.text
    created = S.JobCreatedOut.model_validate(r.json())
    prov = created.provenance
    assert prov is not None and prov.spec_version == spec and prov.predictor == "stub"
    stub = next(p for p in infos if p["name"] == "stub")
    assert prov.predictor_version == stub["version"]
    assert synthetic_app.get(created.status_url).json()["provenance"] == r.json()["provenance"]


@requires("predict")
def test_predict_unavailable_and_unknown_predictor(synthetic_app):  # noqa: F811
    body = {"reactants_smiles": SN2_REACTANTS, "predictor": "transformer"}
    err = error_of(synthetic_app.post("/api/predict", json=body), 409, "PREDICTOR_UNAVAILABLE")
    assert err["detail"]["code"] == "TRANSFORMER_NOT_AVAILABLE"
    body["predictor"] = "no-such-predictor"
    error_of(synthetic_app.post("/api/predict", json=body), 422, "PREDICTOR_UNKNOWN")


# ---- normalize


def test_normalize_ok(synthetic_app):  # noqa: F811
    r = synthetic_app.post("/api/normalize", json={"reactants_smiles": SN2_REACTANTS})
    assert r.status_code == 200, r.text
    out = S.NormalizeOut.model_validate(r.json())
    assert out.ok and out.pool_key == "[CH3][Cl]|0.[CH3][O]|-1"  # sigma keys
    assert re.fullmatch(r"s[0-9a-f]{20}", out.state_id)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"reactants_smiles": ["C1CC"]}, "INVALID_SMILES"),
        ({"reactants_smiles": ["[CH3]"]}, "RADICAL_NOT_SUPPORTED"),
        ({"reactants_smiles": SN2_REACTANTS, "stoichiometry": [1]}, "STOICHIOMETRY_MISMATCH"),
    ],
)
def test_normalize_errors(synthetic_app, body, code):  # noqa: F811
    err = error_of(synthetic_app.post("/api/normalize", json=body), 422, code)
    if code != "STOICHIOMETRY_MISMATCH":
        assert err["detail"]["normalize"]["species"][0]["errors"][0]["code"] == code


def test_request_validation_errors(synthetic_app):  # noqa: F811
    if CAPABILITIES["predict"][0]:
        body = {"reactants_smiles": SN2_REACTANTS, "beam": 0}
        err = error_of(synthetic_app.post("/api/predict", json=body), 422, "REQUEST_INVALID")
        assert any("beam" in e["loc"] for e in err["detail"]["errors"])
    body = {"reactants_smiles": SN2_REACTANTS, "no_such_field": 1}
    error_of(synthetic_app.post("/api/normalize", json=body), 422, "REQUEST_INVALID")
    r = synthetic_app.post(
        "/api/normalize", content=b"{not json", headers={"content-type": "application/json"}
    )
    error_of(r, 422, "REQUEST_INVALID")
    error_of(synthetic_app.get("/api/curate/-1"), 422, "REQUEST_INVALID")
    error_of(synthetic_app.get("/api/curate/abc"), 422, "REQUEST_INVALID")


# ---- predict: the network


@requires("predict")
def test_predict_network(sn2_job):
    net = S.NetworkOut.model_validate(sn2_job["result"])
    assert sn2_job["kind"] == "predict" and sn2_job["result_kind"] == "network"
    sn2 = next(e for e in net.edges if e.target_pool_key == "COC.[Cl-]")
    assert sn2.source == net.root
    arrow = sn2.arrows[0]
    assert (arrow.donor_element, arrow.acceptor_element) == ("O", "C")
    assert arrow.site_source == "rule"
    assert sorted(s.species_key for s in _node(net, sn2.target).species) == ["COC", "[Cl-]"]
    assert sn2.pmechdb_code == "2=6;6,7=7"
    assert net.pathways and net.pathways[0].rank == 1


def _node(net: S.NetworkOut, node_id: str) -> S.NodeOut:
    return next(n for n in net.nodes if n.id == node_id)


@requires("predict")
def test_network_arrows_carry_the_full_brief(sn2_job):
    fields = set(S.ArrowPhysicsBrief.model_fields)
    arrows = [a for e in sn2_job["result"]["edges"] for a in e["arrows"]]
    assert arrows
    for a in arrows:
        brief = a["physics"]
        assert set(brief) == fields
        assert brief["contact_distance_ang"] == 2.3
        assert brief["ks2_form"] in ("difference", "contact", "undefined")
        assert brief["tau"] == 1.0
        assert brief["frozen_source"] == "untrained_kernel"
        dom = brief["dominant"]
        assert dom is not None, brief["flags"]
        assert math.isfinite(dom["E_k"]) and math.isfinite(dom["E_m"])
        assert dom["k_label"] and dom["m_label"]
        assert math.isfinite(brief["ks3"]) and math.isfinite(brief["G"])
    assert "NaN" not in str(sn2_job)


def _check_provenance(prov: dict, predictor: str) -> None:
    p = S.Provenance.model_validate(prov)
    assert p.predictor == predictor
    assert re.match(SPEC_RE, p.spec_version)
    assert p.chemistry in ("synthetic", "stub")


@requires("predict")
def test_provenance_on_every_prediction_response(synthetic_app, sn2_job, sn2_root, sn2_tuple):  # noqa: F811
    _check_provenance(sn2_job["provenance"], "stub")
    _check_provenance(sn2_job["result"]["provenance"], "stub")
    assert sn2_job["result"]["provenance"]["predictor_version"].startswith("stub/")
    r = synthetic_app.get(f"/api/pool/{sn2_root}/candidates", params={"predictor": "stub"})
    _check_provenance(r.json()["provenance"], "stub")
    params = arrow_of(sn2_tuple)
    _check_provenance(
        synthetic_app.get(f"/api/pool/{sn2_root}/physics", params=params).json()["provenance"],
        "floor",
    )
    _check_provenance(
        synthetic_app.get(f"/api/pool/{sn2_root}/tokens").json()["provenance"], "none"
    )
    pair = {"donor": params["donor"], "acceptor": params["acceptor"]}
    r = synthetic_app.get(f"/api/pool/{sn2_root}/site_pairs", params=pair)
    _check_provenance(r.json()["provenance"], "none")


@requires("predict")
def test_every_state_id_resolves(synthetic_app, sn2_job):  # noqa: F811
    _check_states(synthetic_app, sn2_job["result"])


def _check_states(client, net: dict) -> None:
    ids = {n["state_id"] for n in net["nodes"]}
    ids |= {a for n in net["nodes"] for a in n["alias_state_ids"]}
    ids |= {e["product_state_id"] for e in net["edges"]}
    ids |= {e["source_state_id"] for e in net["edges"]}
    assert len(ids) >= 3
    for sid in sorted(ids):
        r = client.get(f"/api/pool/{sid}")
        assert r.status_code == 200, (sid, r.text)
        assert r.json()["state_id"] == sid
        r = client.get(f"/api/pool/{sid}/depiction", params={"server_svg": "1"})
        assert r.status_code == 200, (sid, r.text)
        dep = S.DepictionOut.model_validate(r.json())
        assert dep.server_svg and len(dep.server_coords) == len(dep.atoms)


@requires("predict")
def test_jobs_endpoints(synthetic_app, sn2_job):  # noqa: F811
    job_id = sn2_job["job_id"]
    listed = synthetic_app.get("/api/jobs", params={"kind": "predict", "status": ""}).json()
    assert job_id in {j["job_id"] for j in listed}
    rev = sn2_job["result"]["revision"]
    r = synthetic_app.get(f"/api/jobs/{job_id}", params={"since_revision": rev}).json()
    assert r["unchanged"] is True and r["result"] is None and r["result_kind"] == "network"
    paths = synthetic_app.get(f"/api/jobs/{job_id}/pathways", params={"limit": 5}).json()
    assert [p["rank"] for p in paths][:1] == [1]
    assert synthetic_app.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "done"
    error_of(synthetic_app.get("/api/jobs/j0000000000000000"), 404, "JOB_NOT_FOUND")


# ---- pools, tokens, candidates, physics, 3D


def test_pool_and_tokens(synthetic_app, sn2_root):  # noqa: F811
    pool = S.PoolOut.model_validate(synthetic_app.get(f"/api/pool/{sn2_root}").json())
    want = "ready" if CAPABILITIES["physics"][0] else "disabled"
    assert pool.analysis == want and pool.chemistry == "synthetic"
    r = synthetic_app.get(f"/api/pool/{sn2_root}/tokens")
    assert r.status_code == 200 and "NaN" not in r.text
    tokens = S.TokensOut.model_validate(r.json())
    assert tokens.n_orbitals == len(tokens.orbitals) == pool.n_orbitals
    assert all(o.diagnostic.diagnostic is True for o in tokens.orbitals)
    assert {f.token_key for f in tokens.fragments} == set(SN2_REACTANTS)


@requires("candidates")
def test_candidates(synthetic_app, sn2_root, sn2_tuple):  # noqa: F811
    r = synthetic_app.get(f"/api/pool/{sn2_root}/candidates", params={"limit": 5, "offset": 0})
    out = S.CandidatesOut.model_validate(r.json())
    assert out.predictor == "floor" and len(out.candidates) == 5 and out.total >= out.n_tuples
    assert [c.rank for c in out.candidates] == [1, 2, 3, 4, 5]
    assert sn2_tuple["rank"] >= 1 and sn2_tuple["acceptor_centre_elements"] == ["C", "Cl"]
    excluded = candidates(synthetic_app, sn2_root, valid=False)
    assert excluded and all(c["reason"] for c in excluded)
    error_of(
        synthetic_app.get(f"/api/pool/{sn2_root}/candidates", params={"predictor": "gbdt"}),
        409,
        "PREDICTOR_UNAVAILABLE",
    )
    params = {"donor": sn2_tuple["acceptor"]}
    error_of(
        synthetic_app.get(f"/api/pool/{sn2_root}/candidates", params=params),
        422,
        "NOT_AN_ORBITAL_TOKEN",
    )


@requires("physics")
def test_physics_and_site_pairs(synthetic_app, sn2_root, sn2_tuple):  # noqa: F811
    r = synthetic_app.get(f"/api/pool/{sn2_root}/physics", params=arrow_of(sn2_tuple))
    assert r.status_code == 200, r.text
    p = S.PhysicsOut.model_validate(r.json())
    assert p.pair_valid and math.isfinite(p.ks3) and p.floor_rank is not None
    assert p.requested_sites_are_contact == sn2_tuple["is_contact_site_pair"]
    assert (p.site_pair is None) == sn2_tuple["is_contact_site_pair"]
    assert p.contact_distance_ang == 2.3
    params = {"donor": sn2_tuple["donor"], "acceptor": sn2_tuple["acceptor"]}
    sp = S.SitePairsOut.model_validate(
        synthetic_app.get(f"/api/pool/{sn2_root}/site_pairs", params=params).json()
    )
    assert sp.site_pairs and sum(x.is_contact for x in sp.site_pairs) == 1
    bad = arrow_of(sn2_tuple) | {"acceptor_site": sn2_tuple["donor_site"]}
    error_of(
        synthetic_app.get(f"/api/pool/{sn2_root}/physics", params=bad), 422, "SITE_NOT_A_CENTRE"
    )


def test_geometry_cube_depiction(synthetic_app, sn2_root, sn2_donor):  # noqa: F811
    geo = S.GeometryOut.model_validate(synthetic_app.get(f"/api/pool/{sn2_root}/geometry").json())
    assert len(geo.species) == 2
    token = sn2_donor
    r = synthetic_app.get(f"/api/pool/{sn2_root}/orbital/{token}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["content-disposition"] == f"inline; filename={sn2_root}-{token}.cube"
    assert r.headers.get("content-encoding") == "gzip"
    lines = r.text.splitlines()
    assert len(lines) > 6 and int(lines[2].split()[0]) != 0
    params = {"keep_maps": "2,6", "server_svg": "1"}
    dep = S.DepictionOut.model_validate(
        synthetic_app.get(f"/api/pool/{sn2_root}/depiction", params=params).json()
    )
    assert dep.server_svg and dep.server_coords
    bad = {"keep_maps": "2,x"}
    error_of(
        synthetic_app.get(f"/api/pool/{sn2_root}/depiction", params=bad), 422, "REQUEST_INVALID"
    )
    error_of(synthetic_app.get(f"/api/pool/{sn2_root}/orbital/0"), 422, "NOT_AN_ORBITAL_TOKEN")


@requires("contact")
def test_contact(synthetic_app, sn2_root, sn2_tuple):  # noqa: F811
    ids = "/".join(str(v) for v in arrow_of(sn2_tuple).values())
    path = f"/api/contact/{sn2_root}/{ids}"
    r = synthetic_app.get(path)
    assert r.status_code == 200, r.text
    c = S.ContactOut.model_validate(r.json())
    assert c.consistent is True
    assert set(c.cube_urls) == {"donor", "acceptor"}
    assert c.provenance.predictor == "none"
    for which, url in c.cube_urls.items():
        assert url == f"{path}/cube/{which}"
        cube = synthetic_app.get(url)
        assert cube.status_code == 200 and cube.headers["content-type"].startswith("text/plain")
    error_of(synthetic_app.get(f"{path}/cube/sideways"), 404, "POOL_NOT_FOUND")


# ---- override


@requires("correct")
def test_override_keeps_the_original(synthetic_app):  # noqa: F811
    job = predict(synthetic_app)
    net = job["result"]
    sn2 = edge_to(net, "COC.[Cl-]")
    rows = candidates(synthetic_app, sn2["source_state_id"], **CH_TUPLE)
    assert rows, "no O -> sigma*(C-H) tuple at site H"
    body = {"edge_id": sn2["id"], "arrows": [arrow_of(rows[0])], "continue_steps": 1}
    r = synthetic_app.post(f"/api/jobs/{job['job_id']}/override", json=body)
    assert r.status_code == 202, r.text
    created = S.JobCreatedOut.model_validate(r.json())
    assert created.kind == "override" and created.parent_job_id == job["job_id"]
    assert created.status_url == f"/api/jobs/{created.job_id}"
    ov = synthetic_app.get(created.status_url).json()
    assert ov["status"] == "done" and ov["result_kind"] == "override", ov.get("error")
    result = S.OverrideResultOut.model_validate(ov["result"])
    after = synthetic_app.get(f"/api/jobs/{job['job_id']}").json()["result"]
    edges = {e["id"]: e for e in after["edges"]}
    corrected = edges[result.corrected_edge_id]
    assert corrected["kind"] == "corrected" and corrected["status"] == "active"
    assert corrected["original"]["edge_id"] == sn2["id"]
    assert corrected["arrows"][0]["site_source"] == "user"
    assert edges[sn2["id"]]["status"] == "superseded"
    assert edges[sn2["id"]]["superseded_by"] == result.corrected_edge_id
    assert after["revision"] > net["revision"]
    _check_states(synthetic_app, after)


@requires("correct")
def test_override_at_non_contact_sites_carries_the_site_pair(synthetic_app):  # noqa: F811
    job = predict(synthetic_app)
    sn2 = edge_to(job["result"], "COC.[Cl-]")
    match = SN2_TUPLE | {"acceptor_site_element": "Cl", "is_contact_site_pair": False}
    rows = candidates(synthetic_app, sn2["source_state_id"], **match)
    assert rows, "no O -> sigma*(C-Cl) tuple at site Cl"
    arrow = arrow_of(rows[0])
    body = {"edge_id": sn2["id"], "arrows": [arrow], "continue_prediction": False}
    r = synthetic_app.post(f"/api/jobs/{job['job_id']}/override", json=body)
    assert r.status_code == 202, r.text
    ov = synthetic_app.get(r.json()["status_url"]).json()
    assert ov["status"] == "done", ov.get("error")
    net = S.NetworkOut.model_validate(
        synthetic_app.get(f"/api/jobs/{job['job_id']}").json()["result"]
    )
    ce = next(e for e in net.edges if e.id == ov["result"]["corrected_edge_id"])
    sp = ce.arrows[0].site_pair
    assert sp is not None and not sp.is_contact
    assert (sp.donor_site, sp.acceptor_site) == (arrow["donor_site"], arrow["acceptor_site"])
    assert "user_sites_not_contact" in ce.arrows[0].physics.flags
    assert all(a.site_pair is None for e in net.edges if e.kind == "predicted" for a in e.arrows)


@requires("correct")
def test_override_errors(synthetic_app, sn2_job):  # noqa: F811
    job_id = sn2_job["job_id"]
    arrow = {"donor": 0, "donor_site": 0, "acceptor": 1, "acceptor_site": 1}
    url = f"/api/jobs/{job_id}/override"
    error_of(
        synthetic_app.post(url, json={"edge_id": "x", "arrows": [arrow]}), 404, "EDGE_NOT_FOUND"
    )
    body = {"edge_id": "e0000000000000000", "arrows": [arrow]}
    error_of(synthetic_app.post(url, json=body), 404, "EDGE_NOT_FOUND")
    error_of(
        synthetic_app.post(url, json={"node_id": "p12", "arrows": [arrow]}), 404, "NODE_NOT_FOUND"
    )
    error_of(synthetic_app.post(url, json={"arrows": [arrow]}), 422, "REQUEST_INVALID")


# ---- Pending, id validation, error envelope


def test_pending_shape(serve_settings):  # noqa: F811
    s = serve_settings(chemistry="synthetic", interactive_wait_s=0.05)
    sched = NeverScheduler()
    with api_client(s, scheduler=sched) as client:
        r = client.post("/api/pools", json={"reactants_smiles": ["CCO"]})
        assert r.status_code == 202, r.text
        body = S.PendingOut.model_validate(r.json())
        assert set(r.json()) == set(S.PendingOut.model_fields)
        assert body.status == "pending" and body.retry_after_s > 0
        assert "CCO" in body.label and r.headers["retry-after"]
        again = client.post("/api/pools", json={"reactants_smiles": ["CCO"]}).json()
        assert again["task_id"] == body.task_id
        state_id = client.post("/api/normalize", json={"reactants_smiles": ["CCO"]}).json()[
            "state_id"
        ]
        assert client.get(f"/api/pool/{state_id}").json()["analysis"] == "running"
        assert client.get(f"/api/pool/{state_id}/tokens").status_code == 202
    assert [name for name, _ in sched.calls] == ["featurize"] * len(sched.calls)


BAD_IDS = [
    ("/api/pool/%2E%2E", "POOL_NOT_FOUND"),
    ("/api/pool/p0123456789abcdef0123", "POOL_NOT_FOUND"),
    ("/api/pool/s0123456789ABCDEF0123/tokens", "POOL_NOT_FOUND"),
    ("/api/pool/s0123/depiction", "POOL_NOT_FOUND"),
    ("/api/jobs/%2E%2E", "JOB_NOT_FOUND"),
    ("/api/jobs/s0123456789abcdef", "JOB_NOT_FOUND"),
    ("/api/jobs/j0123456789abcdef0/pathways", "JOB_NOT_FOUND"),
    ("/api/browse/step/pmechdb:%2E%2E", "STEP_NOT_FOUND"),
    ("/api/browse/step/curated:1:x", "STEP_NOT_FOUND"),
    ("/api/browse/step/pmechdb:pmechdb-0123", "STEP_NOT_FOUND"),
]


@pytest.mark.parametrize(("path", "code"), BAD_IDS)
def test_id_validation_never_reaches_a_service(synthetic_app, monkeypatch, path, code):  # noqa: F811
    svc = services(synthetic_app)

    def reached(*args, **kwargs):
        raise AssertionError(f"a service was reached for {path}")

    for owner, name in ((svc.ctx, "state"), (svc.runner, "get"), (svc.browse, "step")):
        monkeypatch.setattr(owner, name, reached)
    monkeypatch.setattr(svc.runner.store, "dir", reached)
    error_of(synthetic_app.get(path), 404, code)


@requires("predict")
def test_contact_cube_which_and_curate_ids(synthetic_app, sn2_job):  # noqa: F811
    sid = root_node(sn2_job["result"])["state_id"]
    error_of(synthetic_app.get(f"/api/contact/{sid}/1/1/2/2/cube/..%2E"), 404, "POOL_NOT_FOUND")
    job_id = sn2_job["job_id"]
    body = {"job_id": job_id, "pathway_id": "../x", "action": "save"}
    error_of(synthetic_app.post("/api/curate", json=body), 404, "PATHWAY_NOT_FOUND")
    body = {"job_id": "j..", "pathway_id": "w0123456789ab", "action": "save"}
    error_of(synthetic_app.post("/api/curate", json=body), 404, "JOB_NOT_FOUND")
    body = {"job_id": job_id, "edge_ids": ["e0000000000000000", "bad"], "action": "save"}
    error_of(synthetic_app.post("/api/curate", json=body), 404, "EDGE_NOT_FOUND")
    error_of(synthetic_app.get("/api/curate/999999"), 404, "MECHANISM_NOT_FOUND")
    error_of(
        synthetic_app.patch("/api/curate/999999", json={"notes": "x"}), 404, "MECHANISM_NOT_FOUND"
    )
    body = {"predictor": "transformer"}
    error_of(synthetic_app.post("/api/curate/recheck", json=body), 409, "PREDICTOR_UNAVAILABLE")


def test_unexpected_errors_are_500_with_a_request_id(synthetic_app, monkeypatch, caplog):  # noqa: F811
    svc = services(synthetic_app)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(svc.browse, "status", boom)
    with caplog.at_level(logging.ERROR, logger="ksmech.serve.api"):
        r = synthetic_app.get("/api/browse/status")
    err = error_of(r, 500, "UNEXPECTED")
    assert "RuntimeError: boom" in err["message"]
    assert any(err["request_id"] in rec.getMessage() for rec in caplog.records)


def test_unknown_api_route(synthetic_app):  # noqa: F811
    error_of(synthetic_app.get("/api/no/such/route"), 404, "REQUEST_INVALID")
    error_of(synthetic_app.delete("/api/health"), 405, "REQUEST_INVALID")


# ---- browse index and search


def test_browse_without_pmechdb(synthetic_app):  # noqa: F811
    status = S.IndexStatusOut.model_validate(synthetic_app.get("/api/browse/status").json())
    assert status.pmechdb.state == "unavailable"
    err = error_of(synthetic_app.post("/api/browse/index"), 409, "INDEX_NOT_READY")
    assert err["detail"]["state"] == "unavailable"
    r = synthetic_app.get("/api/browse/search", params={"q": "COC", "source": "pmechdb"})
    err = error_of(r, 409, "INDEX_NOT_READY")
    assert err["detail"]["build"] == "POST /api/browse/index"
    r = synthetic_app.get("/api/browse/search", params={"q": "COC", "source": "user_curated"})
    assert r.status_code == 200, r.text
    error_of(
        synthetic_app.get("/api/browse/search", params={"mode": "fuzzy", "source": "user_curated"}),
        422,
        "QUERY_INVALID",
    )


def test_browse_index_job_and_auto_index(serve_settings, serve_mini_processed):  # noqa: F811
    s = serve_settings(chemistry="synthetic", auto_index=True)
    assert s.processed == serve_mini_processed
    with api_client(s) as client:
        jobs = services(client).runner.list(kind="index")
        assert len(jobs) == 1 and jobs[0].status == "done", jobs
        status = client.get("/api/browse/status").json()
        assert status["pmechdb"]["state"] == "ready" and status["pmechdb"]["n_steps"] > 0
        r = client.post("/api/browse/index")
        assert r.status_code == 202, r.text
        assert S.JobCreatedOut.model_validate(r.json()).kind == "index"
        job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        assert job["status"] == "done" and job["result_kind"] == "index"
        found = client.get("/api/browse/search", params={"q": "", "source": "pmechdb"}).json()
        assert found["total"] > 0
        ref = found["results"][0]["ref"]
        view = S.StepViewOut.model_validate(client.get(f"/api/browse/step/{ref}").json())
        assert view.reactants_depiction.server_svg and view.dataset_arrows


# ---- Host allow-list and the same-origin guard


def test_foreign_host_is_refused(synthetic_app):  # noqa: F811
    for host in ("evil.example", "evil.example:8000", "localhost.evil.example"):
        r = synthetic_app.get("/api/meta", headers={"Host": host})
        assert r.status_code == 400, (host, r.text)
    for host in ("localhost:8000", "127.0.0.1:8000", "localhost", "127.0.0.1:5173"):
        assert synthetic_app.get("/api/meta", headers={"Host": host}).status_code == 200, host


def test_lan_hosts_are_accepted_only_when_configured(fresh_synthetic_settings):  # noqa: F811
    lan = {"Host": "192.168.1.50:8000"}
    origin = {**lan, "Origin": "http://192.168.1.50:8000"}
    body = {"reactants_smiles": SN2_REACTANTS}
    with api_client(fresh_synthetic_settings) as client:
        assert client.get("/api/meta", headers=lan).status_code == 400
    with api_client(fresh_synthetic_settings, lan_hosts=["192.168.1.50", "labhost"]) as client:
        assert client.get("/api/meta", headers=lan).status_code == 200
        assert client.get("/api/meta", headers={"Host": "labhost:8000"}).status_code == 200
        assert client.get("/api/meta", headers={"Host": "evil.example"}).status_code == 400
        assert client.post("/api/normalize", json=body, headers=origin).status_code == 200
        r = client.post("/api/normalize", json=body, headers={**lan, "Origin": "http://evil.x"})
        error_of(r, 403, "REQUEST_FORBIDDEN")


def _n_jobs(settings) -> int:
    return len(list(settings.jobs.iterdir())) if settings.jobs.is_dir() else 0


def test_cross_origin_mutations_are_refused(fresh_synthetic_settings):  # noqa: F811
    s = fresh_synthetic_settings
    with api_client(s) as client:
        before = _n_jobs(s)
        for origin in ("https://evil.example", "http://localhost.evil.example:8000", "null"):
            r = client.post("/api/curate/export", headers={"Origin": origin})
            err = error_of(r, 403, "REQUEST_FORBIDDEN")
            assert err["detail"] == {"origin": origin}
        plain = {"Origin": "https://evil.example", "Content-Type": "text/plain"}
        error_of(
            client.post("/api/curate/recheck", headers=plain, content=b""), 403, "REQUEST_FORBIDDEN"
        )
        error_of(client.post("/api/browse/index", headers=plain), 403, "REQUEST_FORBIDDEN")
        r = client.patch("/api/curate/1", json={"notes": "x"}, headers={"Origin": "null"})
        error_of(r, 403, "REQUEST_FORBIDDEN")
        assert _n_jobs(s) == before
        assert (
            client.get("/api/meta", headers={"Origin": "https://evil.example"}).status_code == 200
        )
        r = client.post("/api/curate/export", headers={"Origin": "http://localhost:8000"})
        if CAPABILITIES["export"][0]:
            assert r.status_code == 202, r.text
            created = S.JobCreatedOut.model_validate(r.json())
            assert created.kind == "export" and created.provenance is None
            assert _n_jobs(s) == before + 1
            job = client.get(created.status_url).json()
            assert job["status"] == "done", job.get("error")
            latest = client.get("/api/curate/export/latest").json()
            for report in (job["result"], latest):  # no PMechDB under this root: reported
                assert [w.split(":")[0] for w in report["warnings"]] == ["PMECHDB_UNAVAILABLE"]
        else:  # same origin: past the guard, refused by the capability
            error_of(r, 503, "FEATURE_DISABLED")
            assert _n_jobs(s) == before
        r = client.post(
            "/api/normalize",
            json={"reactants_smiles": SN2_REACTANTS},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert r.status_code == 200, r.text


def test_review_codes_and_their_statuses():
    from ksmech.serve.errors import STATUS, Code, ServeError

    assert STATUS[Code.REQUEST_FORBIDDEN] == 403 and STATUS[Code.TOKENS_CHANGED] == 503
    assert Code.PMECHDB_UNAVAILABLE not in STATUS  # a report reason, never an HTTP status
    assert ServeError("TOKENS_CHANGED", "tokens rebuilt").status == 503  # raised as a string
    assert ServeError(Code.REQUEST_FORBIDDEN, "x").code == "REQUEST_FORBIDDEN"


def test_export_report_keeps_its_warnings():
    report = {
        "export_id": "x0123456789abcdef",
        "created_at": "2026-10-21T00:00:00+00:00",
        "spec_version": "v8+0123456789ab;tok1;smp1;ori1;qm1",
        "paths": {},
        "n_mechanisms": 0,
        "n_steps_considered": 0,
        "n_steps_written": 0,
        "skipped": {},
        "ingest_rejections": {},
        "duplicates_pmechdb": 0,
        "duplicates_in_export": 0,
        "withdrawn": 0,
        "changed_ids": 0,
        "steps": [],
        "sha256": "0" * 64,
    }
    assert S.ExportReportOut.model_validate(report).warnings == []
    warnings = ["PMECHDB_UNAVAILABLE: no PMechDB steps", "COMPACT_FALLBACK: 2 CSV rows"]
    out = S.ExportReportOut.model_validate(report | {"warnings": warnings})
    assert out.warnings == warnings and out.model_dump(mode="json")["warnings"] == warnings


def test_export_latest_on_an_empty_database(fresh_synthetic_settings):  # noqa: F811
    with api_client(fresh_synthetic_settings) as client:
        error_of(client.get("/api/curate/export/latest"), 404, "EXPORT_NOT_FOUND")
        assert client.get("/api/curate").json() == []


# ---- capabilities (spec v8 port)

GATED = [
    ("get", "/api/predictors", None, "predict"),
    ("post", "/api/predict", {"reactants_smiles": ["C[O-]", "CCl"]}, "predict"),
    ("post", "/api/jobs/j0123456789abcdef0/override", {}, "correct"),
    ("get", "/api/pool/s0123456789abcdef0123/candidates", None, "candidates"),
    ("get", "/api/pool/s0123456789abcdef0123/physics?donor=1&acceptor=2", None, "physics"),
    ("get", "/api/pool/s0123456789abcdef0123/site_pairs?donor=1&acceptor=2", None, "site_pairs"),
    ("get", "/api/contact/s0123456789abcdef0123/1/1/2/2", None, "contact"),
    ("get", "/api/contact/s0123456789abcdef0123/1/1/2/2/cube/donor", None, "contact"),
    ("post", "/api/curate", {"job_id": "j0123456789abcdef0", "action": "save"}, "curate_edit"),
    ("patch", "/api/curate/1", {"notes": "x"}, "curate_edit"),
    ("post", "/api/curate/recheck", {}, "curate_recheck"),
    ("post", "/api/curate/1/recheck", {}, "curate_recheck"),
    ("post", "/api/curate/export", {}, "export"),
]


@pytest.mark.parametrize(("method", "path", "body", "cap"), GATED)
def test_disabled_capabilities_are_503(synthetic_app, monkeypatch, method, path, body, cap):  # noqa: F811
    """A disabled feature answers 503 FEATURE_DISABLED with its reason before any service."""
    if CAPABILITIES[cap][0]:
        pytest.skip(f"capability {cap} is on")
    svc = services(synthetic_app)

    def reached(*args, **kwargs):
        raise AssertionError(f"a service was reached for {path}")

    for owner, name in ((svc.ctx, "state"), (svc.runner, "get"), (svc.runner, "submit")):
        monkeypatch.setattr(owner, name, reached)
    kwargs = {} if body is None else {"json": body}
    err = error_of(getattr(synthetic_app, method)(path, **kwargs), 503, "FEATURE_DISABLED")
    assert err["detail"] == {"capability": cap, "reason": CAPABILITIES[cap][1]}
    assert cap in err["message"]


def test_meta_lists_the_capabilities(synthetic_app):  # noqa: F811
    m = S.MetaOut.model_validate(synthetic_app.get("/api/meta").json())
    assert set(m.capabilities) == set(CAPABILITIES)
    for name, (on, reason) in CAPABILITIES.items():
        assert m.capabilities[name].enabled is on and m.capabilities[name].reason == reason
    assert m.spec_label == "v8"
