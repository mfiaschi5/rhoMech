"""`AccountService`: the one object the app holds for accounts (configuration, database, OIDC
client, quotas, sign-in throttle), and what the routes call."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import httpx
from sqlalchemy.engine import Engine

from ksmech.serve.accounts import sessions, users
from ksmech.serve.accounts.config import AuthConfig, load_config
from ksmech.serve.accounts.oidc import OidcClient
from ksmech.serve.accounts.quotas import Quotas
from ksmech.serve.accounts.store import open_store
from ksmech.serve.core import schemas as S
from ksmech.serve.core.auth import Principal, Role
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.settings import ServeSettings, assert_writable

RETURN_TO = re.compile(r"#/[A-Za-z0-9/_?=&.,:%+-]{0,300}")


def safe_return_to(value: str | None) -> str:
    """A hash route of this app to land on after sign-in (`#/predict`); `#/` for anything
    else, so a sign-in never redirects to another site."""
    return value if value and RETURN_TO.fullmatch(value) else "#/"


class AccountService:
    def __init__(
        self, config: AuthConfig, engine: Engine, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.config = config
        self.engine = engine
        self.oidc = OidcClient(config, transport=transport)
        self.quotas = Quotas(engine, config)
        self.throttle = sessions.LoginThrottle()

    @classmethod
    def of(
        cls, settings: ServeSettings, transport: httpx.BaseTransport | None = None
    ) -> AccountService:
        """ValueError for a malformed auth file (the server does not start)."""
        config = load_config(settings.auth_file)
        assert_writable(settings.accounts_db, settings)
        service = cls(config, open_store(settings.accounts_db), transport)
        sessions.purge_expired(service.engine)
        return service

    # ---- sessions and status
    def session_principal(self, token: str) -> Principal | None:
        return sessions.resolve_session(self.engine, token)

    def charge(self, principal: Principal, kind: str) -> None:
        self.quotas.charge(principal, kind)

    def providers_out(self) -> list[S.ProviderOut]:
        return [
            S.ProviderOut(id=p.id, label=p.label, start_url=f"/api/auth/oidc/{p.id}/start")
            for p in self.config.providers.values()
        ]

    def status(self, principal: Principal) -> S.AuthStatusOut:
        user = None
        if principal.user_id is not None:
            user = users.to_out(self.engine, users.get_user(self.engine, principal.user_id))
        return S.AuthStatusOut(
            principal=S.PrincipalOut(**principal.to_out()),
            user=user,
            providers=self.providers_out(),
            password_login=self.config.password_login,
            quota=self.quotas.quota_out(principal),
        )

    def _session_of(self, user_id: int, method: str) -> str:
        users.touch_login(self.engine, user_id)
        return sessions.create_session(self.engine, user_id, method, self.config.session_days)

    # ---- sign-in
    def login(self, login: str, password: str, client: str) -> tuple[Principal, str]:
        """(the user's principal, the session cookie value); LOGIN_FAILED or LOGIN_THROTTLED."""
        if not self.config.password_login:
            detail = {"capability": "password_login", "reason": "off in the auth file"}
            raise ServeError(Code.FEATURE_DISABLED, "password sign-in is off", detail=detail)
        keys = (f"ip:{client}", f"login:{login.strip().lower()}")
        self.throttle.check(*keys)
        try:
            user = users.check_password(self.engine, login, password)
        except ServeError:
            self.throttle.fail(*keys)
            raise
        self.throttle.clear(keys[1])
        token = self._session_of(int(user.id or 0), "password")
        return Principal(user.login, Role(user.role), authenticated=True, user_id=user.id), token

    def logout(self, token: str | None) -> None:
        sessions.end_session(self.engine, token)

    def redirect_uri(self, provider_id: str, base_url: str) -> str:
        base = (self.config.public_url or base_url).rstrip("/")
        return f"{base}/api/auth/oidc/{provider_id}/callback"

    def oidc_start(self, provider_id: str, base_url: str, return_to: str | None) -> tuple[str, str]:
        uri = self.redirect_uri(provider_id, base_url)
        return self.oidc.start(provider_id, uri, safe_return_to(return_to))

    def oidc_finish(self, provider_id: str, state: str, code: str) -> tuple[str, str]:
        """(the session cookie value, the return route) of a provider's callback."""
        claims, return_to = self.oidc.finish(provider_id, state, code)
        provider = self.oidc.provider(provider_id)
        user = users.user_of_identity(self.engine, self.config, provider, claims)
        return self._session_of(int(user.id or 0), provider_id), return_to

    def change_password(self, principal: Principal, current: str, new: str) -> S.UserOut:
        if principal.user_id is None:
            msg = "only a signed-in user has a password to change"
            raise ServeError(Code.AUTH_REQUIRED, msg, status=401)
        user = users.get_user(self.engine, principal.user_id)
        if user.password_hash is not None:
            users.check_password(self.engine, user.login, current)
        updated = users.update_user(self.engine, principal.user_id, {"password": new})
        return users.to_out(self.engine, updated)

    # ---- admin
    def list_users(self) -> S.UsersOut:
        return S.UsersOut(
            users=[users.to_out(self.engine, u) for u in users.list_users(self.engine)]
        )

    def create_user(self, body: Mapping[str, Any]) -> S.UserOut:
        user = users.create_user(
            self.engine,
            body["login"],
            body["name"],
            body.get("role", "user"),
            email=body.get("email"),
            password=body.get("password"),
        )
        return users.to_out(self.engine, user)

    def update_user(self, user_id: int, patch: Mapping[str, Any]) -> S.UserOut:
        return users.to_out(self.engine, users.update_user(self.engine, user_id, patch))

    def close(self) -> None:
        self.oidc.close()
        self.engine.dispose()
