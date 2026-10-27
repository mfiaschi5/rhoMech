"""Accounts of ksmech-serve (serve.accounts): password hashes, the auth file, users and the last
admin, password sign-in with the session cookie, the sign-in throttle, job quotas (admins never
limited), the admin routes, the `ksmech-serve users` CLI and an OIDC sign-in against a mock
provider (PKCE, state cookie, ID token claims)."""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from ksmech.serve.accounts import passwords, users
from ksmech.serve.accounts.config import AuthConfig, Provider, load_config, parse_config
from ksmech.serve.accounts.oidc import OidcClient, check_claims
from ksmech.serve.accounts.service import safe_return_to
from ksmech.serve.accounts.store import open_store
from ksmech.serve.core.auth import SESSION_COOKIE, Principal, Role
from ksmech.serve.core.errors import STATUS, Code, ServeError
from ksmech.serve.core.settings import load_settings
from tests.test_serve_fixtures_api import api_client, no_xtb  # noqa: F401 (fixtures)

pytestmark = pytest.mark.serve

PASSWORD = "correct horse battery"
ISSUER = "https://id.example.org"


def _settings(tmp_path, profile: str = "public", auth: str | None = None):
    path = None
    if auth is not None:
        path = tmp_path / "auth.toml"
        path.write_text(auth, encoding="utf-8")
    return load_settings(
        {}, data_root=tmp_path / "root", chemistry="synthetic", profile=profile, auth_file=path
    )


def _error(r, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()["error"]
    assert body["code"] == code, body
    return body


def _add(client, login: str, role: str, **kw):
    engine = client.app.state.services.accounts.engine
    return users.create_user(engine, login, login.title(), role, password=PASSWORD, **kw)


def _login(client, login: str, password: str = PASSWORD):
    return client.post("/api/auth/login", json={"login": login, "password": password})


# ---- units


def test_password_hashes():
    stored = passwords.hash_password(PASSWORD)
    assert stored.startswith("scrypt$") and PASSWORD not in stored
    assert passwords.verify_password(PASSWORD, stored)
    assert not passwords.verify_password(PASSWORD + "!", stored)
    assert stored != passwords.hash_password(PASSWORD)  # a fresh salt each time
    for bad in (None, "", "bcrypt$x", "scrypt$1$2"):
        assert not passwords.verify_password(PASSWORD, bad)
    with pytest.raises(ValueError, match="at least"):
        passwords.hash_password("short")
    assert len(passwords.generate_password()) >= 16


def test_auth_file(tmp_path):
    assert load_config(None) == AuthConfig()
    assert AuthConfig().quota(Role.ADMIN).unlimited
    assert AuthConfig().quota(Role.USER).per_day == 100
    doc = {
        "public_url": "https://ksmech.example.org/",
        "admin_emails": ["Boss@Example.org"],
        "quota": {"user": {"per_day": 5}},
        "providers": {
            "orcid": {"client_id": "APP-1", "client_secret_env": "ORCID_SECRET"},
            "email": {"issuer": ISSUER, "client_id": "c", "scopes": ["email"]},
        },
    }
    config = parse_config(doc, env={"ORCID_SECRET": "s3cret"})
    assert config.public_url == "https://ksmech.example.org" and config.cookie_secure
    assert config.admin_emails == {"boss@example.org"}
    assert config.quota(Role.USER).per_day == 5 and config.quota(Role.USER).per_minute == 10
    orcid = config.providers["orcid"]
    assert (orcid.issuer, orcid.label, orcid.client_secret) == (
        "https://orcid.org",
        "ORCID",
        "s3cret",
    )
    assert config.providers["email"].scopes == ("openid", "email")
    for bad, match in (
        ({"providers": {"orcid": {"client_id": "x", "client_secret_env": "NOPE"}}}, "NOPE"),
        ({"providers": {"x": {"issuer": "http://plain", "client_id": "c"}}}, "https"),
        ({"quota": {"admin": {"per_day": 1}}}, "not a limited role"),
        ({"default_role": "god"}, "default_role"),
        ({"session_days": 0}, "session_days"),
    ):
        with pytest.raises(ValueError, match=match):
            parse_config(bad, env={})
    (tmp_path / "broken.toml").write_text("public_url = ", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read the auth file"):
        load_config(tmp_path / "broken.toml")


def test_users_and_the_last_admin(tmp_path):
    engine = open_store(tmp_path / "a.sqlite")
    boss = users.create_user(engine, "Boss", "Boss", "admin", email="B@x.org", password=PASSWORD)
    assert (boss.login, boss.email) == ("boss", "b@x.org")
    with pytest.raises(ServeError) as err:
        users.create_user(engine, "other", "O", "user", email="b@X.org")
    assert err.value.code == Code.USER_EXISTS
    assert users.check_password(engine, "B@X.ORG", PASSWORD).id == boss.id
    for login, pw in (("boss", "wrong password"), ("nobody", PASSWORD)):
        with pytest.raises(ServeError) as err:
            users.check_password(engine, login, pw)
        assert (
            err.value.code == Code.LOGIN_FAILED and err.value.message == "wrong login or password"
        )
    for patch in ({"role": "user"}, {"disabled": True}):
        with pytest.raises(ServeError) as err:
            users.update_user(engine, boss.id, patch)
        assert err.value.code == Code.LAST_ADMIN
    assert users.get_user(engine, boss.id).role == "admin"  # rolled back
    second = users.create_user(engine, "second", "S", "admin")
    assert users.update_user(engine, boss.id, {"role": "curator"}).role == "curator"
    assert users.to_out(engine, second).has_password is False
    assert STATUS[Code.QUOTA_EXCEEDED] == STATUS[Code.LOGIN_THROTTLED] == 429
    assert STATUS[Code.LAST_ADMIN] == STATUS[Code.USER_EXISTS] == 409


def test_safe_return_to():
    assert safe_return_to("#/predict") == "#/predict"
    assert safe_return_to("#/browse?q=CC") == "#/browse?q=CC"
    for bad in (None, "", "https://evil.example", "//evil.example", "/api/x", "#/a b", "#/<x>"):
        assert safe_return_to(bad) == "#/"


# ---- password sign-in, sessions, roles


def test_password_sign_in_session_and_sign_out(tmp_path):
    with api_client(_settings(tmp_path)) as client:
        _add(client, "ada", "user", email="ada@example.org")
        anon = client.get("/api/auth/me").json()
        assert anon["principal"]["name"] == "anonymous" and anon["user"] is None
        assert anon["password_login"] and anon["providers"] == []
        _error(client.post("/api/jobs/j0123456789abcdef/cancel"), 401, "AUTH_REQUIRED")
        _error(_login(client, "ada", "wrong password"), 401, "LOGIN_FAILED")
        r = _login(client, "ada@example.org")
        assert r.status_code == 200, r.text
        cookie = r.headers["set-cookie"]
        assert (
            SESSION_COOKIE in cookie and "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
        )
        me = r.json()
        assert me["principal"] == {
            "name": "ada",
            "role": "user",
            "via": None,
            "authenticated": True,
            "user_id": me["user"]["id"],
        }
        assert me["user"]["last_login_at"] is not None and me["quota"]["per_day"] == 100
        assert client.get("/api/whoami").json()["name"] == "ada"
        _error(client.get("/api/admin/users"), 403, "ROLE_INSUFFICIENT")
        assert client.post("/api/auth/logout").json()["principal"]["name"] == "anonymous"
        assert client.get("/api/whoami").json()["name"] == "anonymous"


def test_a_session_follows_role_changes_and_ends_with_a_new_password(tmp_path):
    with api_client(_settings(tmp_path)) as client:
        user = _add(client, "bob", "user")
        _add(client, "root", "admin")
        assert _login(client, "bob").status_code == 200
        engine = client.app.state.services.accounts.engine
        users.update_user(engine, user.id, {"role": "curator"})
        assert client.get("/api/whoami").json()["role"] == "curator"
        r = client.post(
            "/api/auth/password", json={"current": "wrong password", "new": "another long one"}
        )
        _error(r, 401, "LOGIN_FAILED")
        r = client.post("/api/auth/password", json={"current": PASSWORD, "new": "another long one"})
        assert r.status_code == 200, r.text
        assert client.get("/api/whoami").json()["name"] == "anonymous"  # every session ended
        assert _login(client, "bob", "another long one").status_code == 200
        users.update_user(engine, user.id, {"disabled": True})
        assert client.get("/api/whoami").json()["name"] == "anonymous"
        _error(_login(client, "bob", "another long one"), 401, "LOGIN_FAILED")


def test_a_session_in_the_lab_profile(tmp_path):
    with api_client(_settings(tmp_path, "lab")) as client:
        assert client.get("/api/whoami").json()["name"] == "local"  # no session: local admin
        _add(client, "carol", "user")
        assert _login(client, "carol").status_code == 200
        assert client.get("/api/whoami").json() == {
            "name": "carol",
            "role": "user",
            "via": None,
            "authenticated": True,
            "user_id": 1,
        }
        _error(client.get("/api/ml/runs"), 403, "ROLE_INSUFFICIENT")


def test_failed_sign_ins_are_throttled(tmp_path):
    from ksmech.serve.accounts.sessions import MAX_FAILURES

    with api_client(_settings(tmp_path)) as client:
        _add(client, "dan", "user")
        for _ in range(MAX_FAILURES):
            _error(_login(client, "dan", "wrong password"), 401, "LOGIN_FAILED")
        err = _error(_login(client, "dan"), 429, "LOGIN_THROTTLED")
        assert err["detail"]["retry_after_s"] > 0


def test_password_sign_in_can_be_off(tmp_path):
    with api_client(_settings(tmp_path, auth="password_login = false\n")) as client:
        _add(client, "eve", "user")
        assert client.get("/api/auth/me").json()["password_login"] is False
        _error(_login(client, "eve"), 503, "FEATURE_DISABLED")


# ---- quotas


def test_quotas_limit_users_and_never_admins(tmp_path):
    """A POST through the predict capability is charged before its body is read: an invalid
    request counts too. The third job of a minute is refused; an admin is never limited."""
    auth = "[quota.user]\nper_day = 50\nper_minute = 2\n"
    with api_client(_settings(tmp_path, auth=auth)) as client:
        _add(client, "fay", "user")
        _add(client, "gil", "admin")
        assert _login(client, "fay").status_code == 200
        for _ in range(2):
            assert client.post("/api/predict", json={}).status_code == 422
        err = _error(client.post("/api/predict", json={}), 429, "QUOTA_EXCEEDED")
        assert err["detail"]["window"] == "minute" and err["detail"]["limit"] == 2
        assert client.post("/api/step/check", json={}).status_code == 422  # not a job: free
        quota = client.get("/api/auth/me").json()["quota"]
        assert quota == {"per_day": 50, "per_minute": 2, "used_today": 2, "unlimited": False}
        client.post("/api/auth/logout")
        assert _login(client, "gil").status_code == 200
        for _ in range(5):
            assert client.post("/api/predict", json={}).status_code == 422
        quota = client.get("/api/auth/me").json()["quota"]
        assert quota == {"per_day": None, "per_minute": None, "used_today": 0, "unlimited": True}


def test_the_lab_admin_has_no_quota(tmp_path):
    with api_client(_settings(tmp_path, "lab", auth="[quota.user]\nper_minute = 0\n")) as client:
        for _ in range(3):
            assert client.post("/api/predict", json={}).status_code == 422
        assert client.get("/api/auth/me").json()["quota"]["unlimited"] is True


# ---- admin routes


def test_admin_routes(tmp_path):
    with api_client(_settings(tmp_path)) as client:
        _add(client, "root", "admin")
        _error(client.get("/api/admin/users"), 401, "AUTH_REQUIRED")
        assert _login(client, "root").status_code == 200
        body = {"login": "hal", "name": "Hal", "role": "curator", "password": "a long password"}
        r = client.post("/api/admin/users", json=body)
        assert r.status_code == 201, r.text
        hal = r.json()
        assert (hal["login"], hal["role"], hal["has_password"]) == ("hal", "curator", True)
        _error(client.post("/api/admin/users", json=body), 409, "USER_EXISTS")
        bad = body | {"login": "Not Valid"}
        assert client.post("/api/admin/users", json=bad).status_code == 422
        r = client.patch(f"/api/admin/users/{hal['id']}", json={"role": "admin"})
        assert r.json()["role"] == "admin"
        logins = [u["login"] for u in client.get("/api/admin/users").json()["users"]]
        assert logins == ["root", "hal"]
        _error(client.patch("/api/admin/users/999", json={"name": "x"}), 404, "USER_NOT_FOUND")


def test_users_cli(tmp_path, capsys, monkeypatch):
    from ksmech.serve.app import cli

    monkeypatch.setenv("KSMECH_DATA_ROOT", str(tmp_path / "root"))
    args = ["users", "add", "ivy", "--name", "Ivy", "--role", "admin", "--generate"]
    assert cli.main(args) == 0
    password = capsys.readouterr().out.strip().splitlines()[-1].split()[-1]
    assert cli.main(["users", "list"]) == 0
    assert "ivy" in capsys.readouterr().out
    assert cli.main(["users", "set", "ivy", "--role", "user"]) == 2  # the last admin
    assert "LAST_ADMIN" in capsys.readouterr().err
    assert cli.main(["users", "set", "nobody", "--role", "user"]) == 2
    engine = open_store(load_settings(data_root=tmp_path / "root").accounts_db)
    assert users.check_password(engine, "ivy", password).role == "admin"
    assert cli.main(["users", "add", "ivy", "--name", "Ivy", "--generate"]) == 2


# ---- OIDC


def _jwt(claims: dict) -> str:
    def part(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256'})}.{part(claims)}.signature"


class MockProvider:
    """An OIDC provider behind httpx.MockTransport: discovery, token and userinfo."""

    def __init__(self, sub: str = "0000-0002-1825-0097", **claims) -> None:
        self.sub, self.claims, self.nonce, self.forms = sub, claims, None, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={
                "issuer": ISSUER,
                "authorization_endpoint": f"{ISSUER}/authorize",
                "token_endpoint": f"{ISSUER}/token",
                "userinfo_endpoint": f"{ISSUER}/userinfo",
                "token_endpoint_auth_methods_supported": ["client_secret_post"],
            })  # fmt: skip
        if request.url.path == "/token":
            form = parse_qs(request.content.decode())
            self.forms.append(form)
            if form["code"] != ["good-code"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            claims = {"iss": ISSUER, "aud": "client-1", "exp": time.time() + 300}
            claims |= {"sub": self.sub, "nonce": self.nonce, **self.claims}
            return httpx.Response(200, json={"id_token": _jwt(claims), "access_token": "at"})
        if request.url.path == "/userinfo":
            return httpx.Response(200, json={"sub": self.sub, "name": "Josiah Carberry"})
        return httpx.Response(404)


OIDC_AUTH = f"""
admin_subjects = ["test:0000-0002-1825-0097"]
[providers.test]
label = "Test ID"
issuer = "{ISSUER}"
client_id = "client-1"
client_secret = "secret-1"
"""


def _oidc_client(client, provider: MockProvider):
    accounts = client.app.state.services.accounts
    accounts.oidc = OidcClient(accounts.config, transport=httpx.MockTransport(provider.handler))
    return accounts.oidc


def _start(client, provider: MockProvider, return_to: str = "#/predict"):
    r = client.get(
        "/api/auth/oidc/test/start", params={"return_to": return_to}, follow_redirects=False
    )
    assert r.status_code == 302, r.text
    query = parse_qs(urlsplit(r.headers["location"]).query)
    provider.nonce = query["nonce"][0]
    return query


def test_oidc_sign_in(tmp_path):
    with api_client(_settings(tmp_path, auth=OIDC_AUTH)) as client:
        provider = MockProvider()
        _oidc_client(client, provider)
        assert client.get("/api/auth/me").json()["providers"] == [
            {"id": "test", "label": "Test ID", "start_url": "/api/auth/oidc/test/start"}
        ]
        query = _start(client, provider)
        assert query["code_challenge_method"] == ["S256"] and query["client_id"] == ["client-1"]
        assert query["redirect_uri"] == ["http://testserver/api/auth/oidc/test/callback"]
        r = client.get(
            "/api/auth/oidc/test/callback",
            params={"code": "good-code", "state": query["state"][0]},
            follow_redirects=False,
        )
        assert r.status_code == 302 and r.headers["location"] == "/#/predict"
        form = provider.forms[-1]
        assert form["client_secret"] == ["secret-1"] and len(form["code_verifier"][0]) > 30
        me = client.get("/api/auth/me").json()
        assert me["user"]["name"] == "Josiah Carberry"  # from userinfo
        assert me["user"]["providers"] == ["test"] and me["principal"]["role"] == "admin"
        assert me["quota"]["unlimited"]
        # a second sign-in of the same subject is the same user
        client.post("/api/auth/logout")
        query = _start(client, provider)
        client.get(
            "/api/auth/oidc/test/callback",
            params={"code": "good-code", "state": query["state"][0]},
            follow_redirects=False,
        )
        assert client.get("/api/auth/me").json()["user"]["id"] == me["user"]["id"]


def test_oidc_failures_land_on_the_login_page(tmp_path):
    with api_client(_settings(tmp_path, auth=OIDC_AUTH)) as client:
        provider = MockProvider()
        _oidc_client(client, provider)
        state = _start(client, provider)["state"][0]

        def callback(**params):
            r = client.get("/api/auth/oidc/test/callback", params=params, follow_redirects=False)
            assert r.status_code == 302
            return r.headers["location"]

        assert callback(code="bad-code", state=state).startswith("/#/login?error=")
        # the failed callback cleared the state cookie, and the state itself is spent
        assert "did%20not%20start" in callback(code="good-code", state=state)
        with pytest.raises(ServeError, match="already used"):
            client.app.state.services.accounts.oidc.finish("test", state, "good-code")
        assert callback(error="access_denied").startswith("/#/login?error=")
        state = _start(client, provider)["state"][0]
        client.cookies.delete("ksmech_oidc", path="/api/auth/oidc")
        assert "did%20not%20start" in callback(code="good-code", state=state)
        assert client.get("/api/whoami").json()["name"] == "anonymous"
        r = client.get("/api/auth/oidc/nope/start", follow_redirects=False)
        _error(r, 404, "PROVIDER_NOT_FOUND")


def test_id_token_claims():
    provider = Provider("p", "P", ISSUER, "client-1", None, ("openid",))
    now = time.time()
    good = {"iss": ISSUER, "aud": ["client-1", "x"], "exp": now + 60, "nonce": "n", "sub": "s"}
    check_claims(good, provider, "n", now)
    google = Provider("g", "G", "https://accounts.google.com", "client-1", None, ("openid",))
    check_claims(good | {"iss": "accounts.google.com"}, google, "n", now)
    for bad in (
        {"iss": "https://other.example"},
        {"aud": "client-2"},
        {"exp": now - 3600},
        {"nonce": "other"},
        {"sub": ""},
    ):
        with pytest.raises(ServeError) as err:
            check_claims(good | bad, provider, "n", now)
        assert err.value.code == Code.OIDC_FAILED


def test_a_verified_email_links_an_existing_user(tmp_path):
    engine = open_store(tmp_path / "a.sqlite")
    config = AuthConfig(admin_emails=frozenset({"kim@example.org"}))
    provider = Provider("test", "T", ISSUER, "c", None, ("openid",))
    kim = users.create_user(engine, "kim", "Kim", "curator", email="kim@example.org")
    claims = {"sub": "1", "email": "KIM@example.org", "email_verified": True}
    assert users.user_of_identity(engine, config, provider, claims).id == kim.id
    unverified = {"sub": "2", "email": "kim@example.org", "email_verified": False}
    other = users.user_of_identity(engine, config, provider, unverified)
    assert other.id != kim.id and other.email is None and other.role == "user"
    fresh = {"sub": "3", "email": "new@example.org", "email_verified": True, "name": "New One"}
    new = users.user_of_identity(engine, config, provider, fresh)
    assert (new.login, new.name, new.role) == ("new", "New One", "user")
    p = Principal(new.login, Role(new.role), authenticated=True, user_id=new.id)
    assert p.at_least(Role.USER)
