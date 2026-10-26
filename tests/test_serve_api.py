"""The HTTP layer of ksmech-serve over the synthetic app (design 6, 13.2): meta, predictors,
normalize errors, a stub predict whose network arrows carry the full physics brief, provenance
on every prediction response, every state id of the network resolvable, the /pool and /contact
endpoints, an override with the original kept, the 202 Pending shape, id validation, the error
envelope, and the browse index endpoints."""

from __future__ import annotations

import logging
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
    """The root state of MeO- + MeCl from POST /api/pools (tokens and analysis)."""
    r = synthetic_app.post("/api/pools", json={"reactants_smiles": SN2_REACTANTS})
    assert r.status_code == 200, r.text
    return r.json()["state_id"]


@pytest.fixture(scope="module")
def sn2_donor(synthetic_app, sn2_root):  # noqa: F811
    """A doubly occupied orbital token of the SN2 root (global id)."""
    tokens = synthetic_app.get(f"/api/pool/{sn2_root}/tokens").json()
    return next(o["token_id"] for o in tokens["orbitals"] if o["model_input"]["occupation"] > 1.5)


# ---- meta, health, predictors


def test_meta(synthetic_app):  # noqa: F811
    r = synthetic_app.get("/api/meta")
    assert r.status_code == 200
    m = S.MetaOut.model_validate(r.json())
    assert re.match(SPEC_RE, m.spec_version), m.spec_version
    assert re.fullmatch(r"[0-9a-f]{12}", m.physics_fingerprint)
    assert m.chemistry == "synthetic"
    assert m.qm_method == m.token_method == "SYNTHETIC-EHT/v1"
    assert m.dev is False
    assert isinstance(m.frontend_built, bool)
    assert m.worker.alive


@requires("predict")
def test_predictors(synthetic_app):  # noqa: F811
    r = synthetic_app.get("/api/predictors")
    assert r.status_code == 200
    infos = {p["name"]: S.PredictorInfoOut.model_validate(p) for p in r.json()}
    assert list(infos) == ["floor", "gbdt", "network"]
    assert infos["floor"].available and not infos["floor"].scores_are_probabilities
    assert not infos["network"].available and infos["network"].scores_are_probabilities
    assert infos["network"].code == "TRANSFORMER_NOT_AVAILABLE"
    assert not infos["gbdt"].available and infos["gbdt"].code == "GBDT_MISSING"


@requires("predict")
def test_predictors_and_job_created_carry_the_spec_version(synthetic_app):  # noqa: F811
    spec = synthetic_app.get("/api/meta").json()["spec_version"]
    infos = synthetic_app.get("/api/predictors").json()
    assert infos and all(S.PredictorInfoOut.model_validate(p).spec_version == spec for p in infos)
    r = synthetic_app.post("/api/predict", json=SN2_REQUEST)
    assert r.status_code == 202, r.text
    created = S.JobCreatedOut.model_validate(r.json())
    prov = created.provenance
    assert prov is not None and prov.spec_version == spec and prov.predictor == "floor"
    floor = next(p for p in infos if p["name"] == "floor")
    assert prov.predictor_version == floor["version"]
    assert synthetic_app.get(created.status_url).json()["provenance"] == r.json()["provenance"]


@requires("predict")
def test_predict_unavailable_and_unknown_predictor(synthetic_app):  # noqa: F811
    body = {"reactants_smiles": SN2_REACTANTS, "predictor": "network"}
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
    sn2 = next(e for e in net.edges if e.target_pool_key == "[CH3][O][CH3]|0.[Cl]|-1")
    assert sn2.source == net.root and sn2.predictor == "floor" and sn2.rank == 1
    assert {a.kind for a in sn2.arrows} == {"lone_pair_to_bond", "bond_to_lone_pair"}
    assert sorted(s.token_keys[0] for s in _node(net, sn2.target).species) == ["COC", "[Cl-]"]
    assert sorted(sn2.product_smiles) == ["COC", "[Cl-]"] and sn2.pmechdb_code
    assert net.pathways and net.pathways[0].rank == 1


def _node(net: S.NetworkOut, node_id: str) -> S.NodeOut:
    return next(n for n in net.nodes if n.id == node_id)


def _check_provenance(prov: dict, predictor: str) -> None:
    p = S.Provenance.model_validate(prov)
    assert p.predictor == predictor
    assert re.match(SPEC_RE, p.spec_version)
    assert p.chemistry in ("synthetic", "stub")


@requires("predict")
def test_provenance_on_every_prediction_response(synthetic_app, sn2_job, sn2_root):  # noqa: F811
    _check_provenance(sn2_job["provenance"], "floor")
    _check_provenance(sn2_job["result"]["provenance"], "floor")
    assert sn2_job["result"]["provenance"]["predictor_version"].startswith("floor/")
    _check_provenance(
        synthetic_app.get(f"/api/pool/{sn2_root}/tokens").json()["provenance"], "none"
    )


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
    want = "ready" if CAPABILITIES["predict"][0] else "disabled"
    assert pool.analysis == want and pool.chemistry == "synthetic"
    r = synthetic_app.get(f"/api/pool/{sn2_root}/tokens")
    assert r.status_code == 200 and "NaN" not in r.text
    tokens = S.TokensOut.model_validate(r.json())
    assert tokens.n_orbitals == len(tokens.orbitals) == pool.n_orbitals
    assert all(o.diagnostic.diagnostic is True for o in tokens.orbitals)
    assert {f.token_key for f in tokens.fragments} == set(SN2_REACTANTS)


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


# ---- override


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
def test_curate_ids(synthetic_app, sn2_job):  # noqa: F811
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
    body = {"predictor": "network"}
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
            r = client.post("/api/curate/recheck", headers={"Origin": origin})
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
        r = client.post("/api/curate/recheck", json={}, headers={"Origin": "http://localhost:8000"})
        if CAPABILITIES["curate_recheck"][0]:  # same origin: past the guard
            assert r.status_code == 202, r.text
        else:  # past the guard, refused by the capability
            error_of(r, 503, "FEATURE_DISABLED")
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


# ---- capabilities

GATED = [
    ("get", "/api/predictors", None, "predict"),
    ("post", "/api/predict", {"reactants_smiles": ["C[O-]", "CCl"]}, "predict"),
    ("post", "/api/jobs/j0123456789abcdef0/override", {}, "correct"),
    ("post", "/api/jobs/j0123456789abcdef0/expand", {"node_id": "p" + "0" * 20}, "predict"),
    (
        "post",
        "/api/step/check",
        {"state_id": "s" + "0" * 20, "matrix": {"bonds": [], "lone": []}},
        "correct",
    ),  # noqa: E501
    ("post", "/api/curate", {"job_id": "j0123456789abcdef0", "action": "save"}, "curate_edit"),
    ("patch", "/api/curate/1", {"notes": "x"}, "curate_edit"),
    ("post", "/api/curate/recheck", {}, "curate_recheck"),
    ("post", "/api/curate/1/recheck", {}, "curate_recheck"),
    ("get", "/api/curate/export/pmechdb", None, "export"),
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
    from ksmech.serve.capabilities import status

    m = S.MetaOut.model_validate(synthetic_app.get("/api/meta").json())
    assert set(m.capabilities) == set(CAPABILITIES)
    settings = services(synthetic_app).settings
    for name in CAPABILITIES:
        on, reason = status(name, settings)  # embedding also depends on the local checkpoints
        assert m.capabilities[name].enabled is on and m.capabilities[name].reason == reason
    assert m.spec_label == "v9"
