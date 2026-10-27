"""Profiles, the auth seam and the audit log of ksmech-serve (core.capabilities.PROFILES,
core.auth, core.audit): the settings and the CLI option; every route of the public app either on
the explicit public allow-list or refused with FEATURE_DISABLED "not in the public profile";
bearer tokens (JSON and TOML files, hashes only, 401 / 403, the X-Ksmech-Via header); the role
table; job kinds hidden in public; one audit line per non-GET request."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from ksmech.serve.core.auth import (
    CAPABILITY_ROLES,
    Authenticator,
    Principal,
    Role,
    load_tokens,
    token_sha256,
)
from ksmech.serve.core.capabilities import CAPABILITIES, PROFILES, status
from ksmech.serve.core.errors import STATUS, Code, ServeError
from ksmech.serve.core.settings import load_settings
from tests.test_serve_fixtures_api import api_client, no_xtb  # noqa: F401 (fixtures)

pytestmark = pytest.mark.serve

PUBLIC_REASON = "not in the public profile"
# Every route the public profile serves. A new route is refused in public until it is added
# here on purpose (test_every_public_route_is_allowed_or_refused).
PUBLIC_ROUTES = {
    ("GET", "/api/health"),
    ("GET", "/api/meta"),
    ("GET", "/api/whoami"),
    ("GET", "/api/models"),
    ("GET", "/api/predictors"),
    ("POST", "/api/predict"),
    ("GET", "/api/jobs"),
    ("GET", "/api/jobs/{job_id}"),
    ("POST", "/api/jobs/{job_id}/cancel"),
    ("POST", "/api/jobs/{job_id}/override"),
    ("POST", "/api/jobs/{job_id}/expand"),
    ("GET", "/api/jobs/{job_id}/pathways"),
    ("POST", "/api/step/check"),
    ("POST", "/api/normalize"),
    ("POST", "/api/pools"),
    ("GET", "/api/pool/{state_id}"),
    ("GET", "/api/pool/{state_id}/tokens"),
    ("GET", "/api/pool/{state_id}/geometry"),
    ("GET", "/api/pool/{state_id}/orbital/{token_id}"),
    ("GET", "/api/pool/{state_id}/depiction"),
    ("POST", "/api/molecule"),
    ("GET", "/api/molecule/{molecule_id}"),
    ("GET", "/api/molecule/{molecule_id}/geometry"),
    ("GET", "/api/molecule/{molecule_id}/properties"),
    ("GET", "/api/molecule/{molecule_id}/symmetry"),
    ("GET", "/api/molecule/{molecule_id}/orbitals"),
    ("GET", "/api/molecule/{molecule_id}/cube"),
    ("POST", "/api/depict2d"),
}
# The dataset, curation and ML routes: refused in public whatever comes later.
NEVER_PUBLIC = re.compile(r"^/api/(browse|curate|embedding|ml)(/|$)")
INT_PARAMS = {"mid", "token_id"}
JOB = "j0123456789abcdef"

READER, USER, CURATOR = "reader-token", "user-token", "curator-token"


def _tokens(role_of: dict[str, str]) -> dict:
    return {token_sha256(t): {"name": f"{r}-1", "role": r} for t, r in role_of.items()}


TOKENS = _tokens({READER: "reader", USER: "user", CURATOR: "curator"})


def _bearer(token: str, via: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    return headers | ({"X-Ksmech-Via": via} if via else {})


def _settings(tmp_path, profile: str = "lab", tokens: dict | None = None, **kw):
    path = None
    if tokens is not None:
        path = tmp_path / "tokens.json"
        path.write_text(json.dumps(tokens), encoding="utf-8")
    root = tmp_path / "root"
    return load_settings(
        {}, data_root=root, chemistry="synthetic", profile=profile, tokens_file=path, **kw
    )


def _error(r, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()["error"]
    assert body["code"] == code, body
    assert body["request_id"] == r.headers["x-request-id"]
    return body


def _audit(settings) -> list[dict]:
    if not settings.audit_log.exists():
        return []
    return [json.loads(x) for x in settings.audit_log.read_text().splitlines()]


@pytest.fixture
def public_app(tmp_path):
    with api_client(_settings(tmp_path, "public", TOKENS)) as client:
        yield client


@pytest.fixture
def lab_app(tmp_path):
    with api_client(_settings(tmp_path, "lab", TOKENS)) as client:
        yield client


# ---- settings, CLI, capabilities


def test_profile_settings_and_cli(tmp_path):
    from ksmech.serve.app import cli

    assert load_settings({}, data_root=tmp_path).profile == "lab"
    env = {"KSMECH_SERVE_PROFILE": "public", "KSMECH_SERVE_TOKENS": str(tmp_path / "t.toml")}
    s = load_settings(env, data_root=tmp_path)
    assert s.profile == "public" and s.tokens_file == (tmp_path / "t.toml").resolve()
    assert s.audit_log == s.logs / "audit.jsonl"
    with pytest.raises(ValueError, match="profile must be one of"):
        load_settings({"KSMECH_SERVE_PROFILE": "online"}, data_root=tmp_path)
    parser = cli.build_parser()
    assert parser.parse_args(["run"]).profile is None
    assert parser.parse_args(["run", "--profile", "public"]).profile == "public"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--profile", "online"])


def test_profiles_and_roles_cover_every_capability(tmp_path):
    assert PROFILES["lab"] == frozenset(CAPABILITIES)
    off = set(CAPABILITIES) - PROFILES["public"]
    assert off == {"browse", "curate", "curate_edit", "curate_recheck", "export", "embedding", "ml"}
    assert set(CAPABILITY_ROLES) == set(CAPABILITIES)
    public = _settings(tmp_path, "public")
    for name in off:
        assert status(name, public) == (False, PUBLIC_REASON)
    assert status("workbench", public) == (True, None)
    assert status("browse", _settings(tmp_path)) == (True, None)
    assert STATUS[Code.TOKEN_INVALID] == STATUS[Code.AUTH_REQUIRED] == 401
    assert STATUS[Code.ROLE_INSUFFICIENT] == 403 and STATUS[Code.RUN_NOT_FOUND] == 404


# ---- the public profile: every route allowed or refused


def _url(template: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: "1" if m.group(1) in INT_PARAMS else "x", template)


def test_every_public_route_is_allowed_or_refused(public_app, monkeypatch):
    """Walk the OpenAPI of the public app: a route off the allow-list answers 503
    FEATURE_DISABLED (profile) before any service runs; the dataset / ML routes are never on it."""
    svc = public_app.app.state.services

    def reached(*args, **kwargs):
        raise AssertionError("a service was reached")

    for owner, name in ((svc.browse, "search"), (svc.curate, "list"), (svc.runner, "submit")):
        monkeypatch.setattr(owner, name, reached)
    routes = {
        (method.upper(), path)
        for path, ops in public_app.app.openapi()["paths"].items()
        for method in ops
    }
    assert routes >= PUBLIC_ROUTES, sorted(PUBLIC_ROUTES - routes)  # the list stays accurate
    assert not [r for r in PUBLIC_ROUTES if NEVER_PUBLIC.match(r[1])]
    refused = sorted(routes - PUBLIC_ROUTES)
    assert {p for _, p in refused} >= {"/api/browse/search", "/api/curate", "/api/ml/runs"}
    for method, template in refused:
        body = {} if method in ("POST", "PATCH", "PUT") else None
        r = public_app.request(method, _url(template), json=body, headers=_bearer(CURATOR))
        err = _error(r, 503, "FEATURE_DISABLED")
        assert err["detail"]["reason"] == PUBLIC_REASON, (method, template)
    meta = public_app.get("/api/meta").json()
    assert meta["profile"] == "public"
    for name in ("browse", "curate", "export", "embedding", "ml"):
        assert meta["capabilities"][name] == {"enabled": False, "reason": PUBLIC_REASON}


def test_public_roles(public_app):
    anon = public_app.get("/api/whoami").json()
    assert anon == {"name": "anonymous", "role": "reader", "via": None}
    assert public_app.get("/api/predictors").status_code == 200  # a reader reads
    assert public_app.get("/api/models").status_code == 200
    err = _error(public_app.post(f"/api/jobs/{JOB}/cancel"), 401, "AUTH_REQUIRED")
    assert err["detail"] == {
        "required": "user",
        "role": "reader",
        "for": f"POST /api/jobs/{JOB}/cancel",
    }
    _error(public_app.post("/api/step/check", json={}), 401, "AUTH_REQUIRED")
    r = public_app.post(f"/api/jobs/{JOB}/cancel", headers=_bearer(READER))
    _error(r, 403, "ROLE_INSUFFICIENT")
    r = public_app.post(f"/api/jobs/{JOB}/cancel", headers=_bearer(USER, "mcp"))
    _error(r, 404, "JOB_NOT_FOUND")  # past the gate
    me = public_app.get("/api/whoami", headers=_bearer(USER, "MCP")).json()
    assert me == {"name": "user-1", "role": "user", "via": "mcp"}


def test_wrong_tokens_are_401_everywhere(public_app, lab_app):
    for client in (public_app, lab_app):
        r = client.get("/api/health", headers=_bearer("not-a-token"))
        err = _error(r, 401, "TOKEN_INVALID")
        assert r.headers["www-authenticate"].startswith("Bearer")
        assert "not-a-token" not in json.dumps(err)
        _error(
            client.get("/api/meta", headers={"Authorization": "Basic abc"}), 401, "TOKEN_INVALID"
        )
        _error(
            client.get("/api/meta", headers={"X-Ksmech-Via": "no spaces"}), 422, "REQUEST_INVALID"
        )


def test_lab_roles_with_a_tokens_file(lab_app):
    assert lab_app.get("/api/whoami").json() == {"name": "local", "role": "admin", "via": None}
    patch = {"notes": "x"}
    r = lab_app.patch("/api/curate/1", json=patch, headers=_bearer(USER))
    _error(r, 403, "ROLE_INSUFFICIENT")
    _error(
        lab_app.patch("/api/curate/1", json=patch, headers=_bearer(CURATOR)),
        404,
        "MECHANISM_NOT_FOUND",
    )
    assert lab_app.get("/api/curate", headers=_bearer(READER)).status_code == 200
    _error(lab_app.get("/api/ml/runs", headers=_bearer(USER)), 403, "ROLE_INSUFFICIENT")
    assert lab_app.get("/api/ml/runs", headers=_bearer(CURATOR)).status_code == 200
    _error(lab_app.post("/api/browse/index", headers=_bearer(USER)), 403, "ROLE_INSUFFICIENT")


def test_lab_without_tokens_is_the_local_admin(tmp_path):
    with api_client(_settings(tmp_path)) as client:
        me = client.get("/api/whoami", headers=_bearer("anything", "cli")).json()
        assert me == {"name": "local", "role": "admin", "via": "cli"}
        assert client.get("/api/ml/runs").status_code == 200


# ---- tokens


def test_token_files(tmp_path):
    sha = hashlib.sha256(b"secret").hexdigest()
    toml = tmp_path / "t.toml"
    toml.write_text(f'[tokens.{sha}]\nname = "jaco"\nrole = "curator"\n', encoding="utf-8")
    assert load_tokens(toml) == {sha: ("jaco", Role.CURATOR)}
    flat = tmp_path / "t.json"
    flat.write_text(json.dumps({sha: {"name": "bot", "role": "user"}}), encoding="utf-8")
    assert load_tokens(flat) == {sha: ("bot", Role.USER)}
    bad = {
        "not-a-sha": {"name": "x", "role": "user"},
        sha: {"name": "x", "role": "root"},
        sha.upper(): {"name": "x", "role": "user"},
    }
    for key, entry in bad.items():
        flat.write_text(json.dumps({key: entry}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_tokens(flat)
    with pytest.raises(ValueError, match="cannot read"):
        load_tokens(tmp_path / "missing.json")
    s = _settings(tmp_path, tokens={"x": {}})
    with pytest.raises(ValueError):
        Authenticator.of(s)


def test_authenticator_resolve():
    auth = Authenticator("public", {token_sha256("t"): ("bot", Role.USER)})
    p = auth.resolve("Bearer t", "cli")
    assert p == Principal("bot", Role.USER, "cli", authenticated=True)
    assert p.at_least(Role.READER) and p.at_least("user") and not p.at_least(Role.CURATOR)
    assert auth.resolve(None, None) == Principal("anonymous", Role.READER)
    for header in ("Bearer x", "Bearer", "Token t", ""):
        with pytest.raises(ServeError) as exc:
            auth.resolve(header, None)
        assert exc.value.code == Code.TOKEN_INVALID and exc.value.status == 401
    assert Authenticator("lab").resolve("Bearer t", None).role == Role.ADMIN  # nothing to check
    with pytest.raises(ServeError):
        Authenticator("public").resolve("Bearer t", None)  # public: a token must be known


# ---- jobs of the lab are hidden in public


def test_public_hides_lab_job_kinds(tmp_path):
    from ksmech.serve.predict.jobs import JobKind

    settings = _settings(tmp_path, "public")
    with api_client(settings) as client:
        job = client.app.state.services.runner.submit(JobKind.INDEX, {})
        _error(client.get(f"/api/jobs/{job.id}"), 404, "JOB_NOT_FOUND")
        assert job.id not in {j["job_id"] for j in client.get("/api/jobs").json()}
    with api_client(_settings(tmp_path)) as client:  # the same data root in the lab
        assert client.get(f"/api/jobs/{job.id}").status_code == 200
        assert job.id in {j["job_id"] for j in client.get("/api/jobs").json()}


# ---- the audit log


def test_audit_lines(lab_app):
    settings = lab_app.app.state.services.settings
    lab_app.get("/api/curate")
    lab_app.patch("/api/curate/7", json={"notes": "x"}, headers=_bearer(CURATOR, "cli"))
    lab_app.post("/api/curate/recheck", json={}, headers=_bearer("wrong"))
    lab_app.post("/api/nowhere")
    lines = _audit(settings)
    assert [(x["method"], x["route"], x["status"]) for x in lines[:2]] == [
        ("PATCH", "/api/curate/{mid}", 404),
        ("POST", None, 401),
    ]
    assert lines[2]["method"] == "POST" and lines[2]["path"] == "/api/nowhere"
    assert lines[2]["status"] in (404, 405) and len(lines) == 3  # no route of the API
    assert lines[0]["principal"] == {"name": "curator-1", "role": "curator", "via": "cli"}
    assert lines[0]["path"] == "/api/curate/7" and lines[0]["request_id"]
    assert lines[1]["principal"] is None  # refused before a principal was named
    assert set(lines[0]) == {"time", "request_id", "principal", "method", "route", "path", "status"}


def test_audit_failure_is_reported_not_raised(lab_app, monkeypatch):
    from ksmech.serve.core import audit

    def refuse(path, settings):
        raise ValueError("read-only (test)")

    monkeypatch.setattr(audit, "assert_writable", refuse)
    r = lab_app.post(f"/api/jobs/{JOB}/cancel")
    _error(r, 404, "JOB_NOT_FOUND")  # the request itself is unaffected
    settings = lab_app.app.state.services.settings
    failures = settings.failures_log.read_text().splitlines()
    assert any(json.loads(x)["code"] == audit.AUDIT_FAILED for x in failures)
    assert not settings.audit_log.exists()
