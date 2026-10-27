"""`KsmechClient` against the real app in process (synthetic chemistry, jobs inline), and its
transport rules on a mock transport: headers, the error envelope, Pending retries, timeouts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from ksmech.client import KsmechAPIError, KsmechClient, KsmechPending, KsmechTimeout
from ksmech.client import summary as sm
from ksmech.serve.core import schemas as S
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    SN2_REACTANTS,
    api_client,
    copy_synthetic_root,
    no_xtb,
    synthetic_app,
    synthetic_root,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


@pytest.fixture(scope="module")
def client(synthetic_app):  # noqa: F811
    return KsmechClient(http=synthetic_app, via="cli")


@pytest.fixture
def public_app(tmp_path, synthetic_root):  # noqa: F811
    """The synthetic app in the public profile, on a copy of the synthetic tokens."""
    from ksmech.serve.core.settings import load_settings

    root = copy_synthetic_root(synthetic_root, tmp_path / "root")
    settings = load_settings({}, data_root=root, chemistry="synthetic", profile="public")
    with api_client(settings) as tc:
        yield tc


@pytest.fixture(scope="module")
def sn2_job(client) -> S.JobOut:
    return client.predict(SN2_REACTANTS, predictor="floor", beam=2, max_steps=1, timeout=60)


# ---- against the app


def test_health_meta_capabilities(client):
    assert client.health().ok
    meta = client.meta()
    assert meta.chemistry == "synthetic"
    caps = client.capabilities()
    assert caps["workbench"] is True and set(meta.capabilities) <= set(caps)


def test_predictors(client):
    names = [p.name for p in client.predictors()]
    assert names[:3] == ["floor", "gbdt", "network"]


def test_normalize_and_its_error_envelope(client):
    out = client.normalize(SN2_REACTANTS)
    assert out.ok and out.state_id and out.pool_key == "[CH3][Cl]|0.[CH3][O]|-1"
    with pytest.raises(KsmechAPIError) as err:
        client.normalize(["C1CC"])
    assert err.value.code == "INVALID_SMILES" and err.value.status == 422
    assert err.value.request_id and err.value.detail["normalize"]["ok"] is False


def test_request_body_is_checked_before_sending(client):
    with pytest.raises(KsmechAPIError) as err:
        client.submit_predict({"reactants_smiles": SN2_REACTANTS, "beem": 2})
    assert err.value.code == "REQUEST_INVALID" and err.value.status == 0


def test_predict_waits_for_the_job(client, sn2_job):
    assert sn2_job.status == "done" and isinstance(sn2_job.result, S.NetworkOut)
    assert sn2_job.job_id in {j.job_id for j in client.jobs(kind="predict")}
    assert client.job(sn2_job.job_id).status == "done"
    pathways = client.pathways(sn2_job.job_id, limit=5)
    assert pathways and pathways[0].rank == 1
    summary = sm.job_summary(sn2_job, n_pathways=2)
    steps = summary["network"]["pathways"][0]["steps"]
    assert steps and steps[0]["from"] and steps[0]["arrows"]
    assert "p" not in steps[0]  # floor scores are not probabilities


def test_cancel_a_finished_job(client, sn2_job):
    assert client.cancel_job(sn2_job.job_id).status == "done"
    with pytest.raises(KsmechAPIError) as err:
        client.job("j" + "0" * 16)
    assert err.value.code == "JOB_NOT_FOUND" and err.value.status == 404


def test_step_check_of_a_predicted_step(client, sn2_job):
    edge = sn2_job.result.edges[0]
    out = client.step_check(edge.source_state_id, edge.matrix)
    assert out.ok and out.product_smiles == edge.product_smiles
    summary = sm.step_check_summary(out)
    assert summary["arrows"] == [sm.arrow_text(a) for a in edge.arrows]


def test_pools_and_depiction(client):
    pool = client.create_pool(SN2_REACTANTS)
    assert client.pool(pool.state_id).pool_key == pool.pool_key
    assert client.pool_tokens(pool.state_id).n_atoms == pool.n_atoms
    dep = client.pool_depiction(pool.state_id, keep_maps=[2], server_svg=True)
    assert dep.server_svg and "<svg" in dep.server_svg


def test_molecule_workbench(client):
    mol = client.molecule_ready(smiles="CO")
    assert mol.status == "ready" and client.molecule(mol.molecule_id).key == mol.key
    props = client.molecule_properties(mol.molecule_id)
    rows = sm.properties_summary(props)["atoms"]
    assert [r["el"] for r in rows].count("O") == 1
    pg = client.molecule_symmetry(mol.molecule_id)
    assert pg.point_group == "Cs"
    orbs = sm.orbitals_summary(client.molecule_orbitals(mol.molecule_id), window=2)
    assert len(orbs["orbitals"]) == 4


def test_models_curate_browse(client):
    models = client.models()
    assert isinstance(models.checkpoints, list)
    assert client.curate_list() == []
    with pytest.raises(KsmechAPIError) as err:
        client.curate_get(999)
    assert err.value.code == "MECHANISM_NOT_FOUND"
    assert client.browse_status().pmechdb.state == "unavailable"
    found = client.browse_search("COC", source=["user_curated"])
    assert found.total == 0 and found.results == []


def test_ml_routes_and_whoami(client):
    runs = client.ml_runs()
    assert runs.runs == [] and [sm.report_row(r) for r in client.ml_reports().reports] == []
    with pytest.raises(KsmechAPIError) as err:
        client.ml_run("pmechdb", "no_such_run")
    assert (err.value.code, err.value.status) == ("RUN_NOT_FOUND", 404)
    who = client.whoami()
    assert who.via == "cli" and who.role == "admin"  # lab without tokens: the local admin


# ---- transport rules (mock transport)


def mock_client(handler, **kwargs) -> KsmechClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ksmech.test")
    return KsmechClient("http://ksmech.test", http=http, sleep=lambda s: None, **kwargs)


def test_headers_carry_the_token_and_who_acts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json=[])

    mock_client(handler, token="t0k", via="mcp").jobs()
    assert seen["authorization"] == "Bearer t0k" and seen["x-ksmech-via"] == "mcp"
    seen.clear()
    mock_client(handler).jobs()
    assert "authorization" not in seen and seen["x-ksmech-via"] == "cli"


PENDING = {
    "status": "pending",
    "task_id": "t1",
    "label": "featurize CO",
    "retry_after_s": 1.0,
    "worker": {
        "alive": True, "n_workers": 1, "pids": [], "busy": True, "current_task": None,
        "current_label": None, "elapsed_s": None, "queue_depth": 0, "restarts": 0,
        "unstable": False,
    },
}  # fmt: skip


def test_pending_is_retried_then_gives_up():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(202, json=PENDING)
        return httpx.Response(200, json=[])

    assert mock_client(handler).jobs() == [] and len(calls) == 3
    never = mock_client(lambda r: httpx.Response(202, json=PENDING), pending_timeout=0)
    with pytest.raises(KsmechPending) as err:
        never.jobs()
    assert err.value.code == "CLIENT_TIMEOUT" and err.value.pending.label == "featurize CO"


def test_errors_without_the_envelope():
    text = mock_client(lambda r: httpx.Response(400, text="Invalid host header"))
    with pytest.raises(KsmechAPIError) as err:
        text.health()
    assert err.value.code == "HTTP_400" and "Invalid host" in err.value.message
    bad = mock_client(lambda r: httpx.Response(200, json={"ok": True}))
    with pytest.raises(KsmechAPIError) as err:
        bad.health()
    assert err.value.code == "RESPONSE_INVALID"


def test_connection_failure():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(KsmechAPIError) as err:
        mock_client(refuse).health()
    assert err.value.code == "CONNECTION_FAILED" and err.value.status == 0


def running_job() -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "job_id": "j" + "1" * 16, "kind": "predict", "status": "running", "created_at": now,
        "started_at": now, "finished_at": None, "progress": {"phase": "expand", "updated_at": now},
        "error": None, "parent_job_id": None, "network_id": None, "partial": False,
        "provenance": None, "result_kind": None, "result": None,
    }  # fmt: skip


def test_wait_job_times_out_with_the_last_job():
    c = mock_client(lambda r: httpx.Response(200, content=json.dumps(running_job())))
    with pytest.raises(KsmechTimeout) as err:
        c.wait_job("j" + "1" * 16, timeout=0)
    assert err.value.last.status == "running"
    assert sm.job_summary(err.value.last)["progress"]["phase"] == "expand"


def test_from_env(monkeypatch):
    monkeypatch.setenv("KSMECH_URL", "http://lab:9000/")
    monkeypatch.setenv("KSMECH_TOKEN", "abc")
    c = KsmechClient.from_env(via="mcp")
    assert (c.url, c.token, c.via) == ("http://lab:9000", "abc", "mcp")
    c.close()


def test_public_profile_refuses_an_unknown_token(public_app):
    bad = KsmechClient(http=public_app, token="no-such-token")
    with pytest.raises(KsmechAPIError) as err:
        bad.whoami()
    assert (err.value.code, err.value.status) == ("TOKEN_INVALID", 401)
    anonymous = KsmechClient(http=public_app)
    assert anonymous.whoami().role == "reader" and anonymous.meta().profile == "public"
    with pytest.raises(KsmechAPIError) as err:
        anonymous.ml_runs()
    assert err.value.code == "FEATURE_DISABLED"
