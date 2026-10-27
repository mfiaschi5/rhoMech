"""Who sends a request and what it may do: the auth seam of ksmech-serve.

A `Principal` is a name, a role and the client it came through (`via`, from the `X-Ksmech-Via`
header: `mcp`, `cli`, ...; `None` without one). Roles are ordered reader < user < curator <
admin (`RANK`); a route needs at least one role (`CAPABILITY_ROLES` for a gated feature,
`require_role` for the rest), and every other route is open to any principal.

Tokens live in the optional file `KSMECH_SERVE_TOKENS` (`.toml` read as TOML, anything else as
JSON): a mapping, at the top level or under `tokens`, from the sha256 (64 lowercase hex) of a
token to `{name, role}`. The server never sees nor stores a token in clear: a request's
`Authorization: Bearer <token>` is hashed and compared with every stored hash in constant time
(`hmac.compare_digest`, no early exit). A token that matches none, or a malformed
`Authorization` header, is 401 TOKEN_INVALID on any route.

Without a token, a session cookie (`SESSION_COOKIE`, set by a sign-in of `serve.accounts`) names
the signed-in user: the app injects the lookup (`Authenticator.sessions`, cookie value ->
`Principal` or None), so this module never reads the accounts database. An unknown or expired
session is no error: the request is anonymous. A bearer token wins over the cookie.

Without either a request is the profile's anonymous principal: `local` admin in `lab` (the
private lab behind an SSH tunnel; today's behaviour, so the UI needs no token) and `anonymous`
reader in `public`, where a token raises the role. In `lab` without a tokens file a presented
token is ignored (there is nothing to check it against); in `public` it is 401. Below the
route's role an anonymous principal gets 401 AUTH_REQUIRED and a token's principal 403
ROLE_INSUFFICIENT.

`AuthGate` (pure ASGI) resolves the principal of every HTTP request into the scope state before
any route runs; `principal_of(request)` reads it back.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.datastructures import Headers
from starlette.requests import cookie_parser

from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.http import error_response

if TYPE_CHECKING:
    from fastapi import Request

    from ksmech.serve.core.settings import ServeSettings


class Role(StrEnum):
    READER = "reader"
    USER = "user"
    CURATOR = "curator"
    ADMIN = "admin"


RANK = {Role.READER: 0, Role.USER: 1, Role.CURATOR: 2, Role.ADMIN: 3}
CAPABILITY_ROLES: dict[str, Role] = {
    "workbench": Role.READER,  # display only: molecules, orbitals, densities
    "browse": Role.READER,
    "curate": Role.READER,  # reading curated mechanisms
    "predict": Role.USER,  # prediction and expansion jobs (GET /predictors: reader)
    "correct": Role.USER,  # user steps and the live step check
    "candidates": Role.USER,
    "physics": Role.USER,
    "embedding": Role.USER,  # submitting a trace (reading a stored one: reader)
    "curate_edit": Role.CURATOR,
    "curate_recheck": Role.CURATOR,
    "export": Role.CURATOR,
    "ml": Role.CURATOR,
}
"""The least role each capability needs (`capabilities.require` checks it)."""

ANONYMOUS = {"lab": ("local", Role.ADMIN), "public": ("anonymous", Role.READER)}
VIA_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,31}")
SHA_RE = re.compile(r"[0-9a-f]{64}")
VIA_HEADER = "x-ksmech-via"
SESSION_COOKIE = "ksmech_session"


@dataclass(frozen=True)
class Principal:
    name: str
    role: Role
    via: str | None = None
    authenticated: bool = False  # true when a bearer token or a session named it
    user_id: int | None = None  # the accounts user of a session (None: a token or anonymous)

    def at_least(self, role: Role) -> bool:
        return RANK[self.role] >= RANK[Role(role)]

    def to_out(self) -> dict:
        """`PrincipalOut` fields."""
        return {
            "name": self.name,
            "role": str(self.role),
            "via": self.via,
            "authenticated": self.authenticated,
            "user_id": self.user_id,
        }


def token_sha256(token: str) -> str:
    """The hex sha256 of a token: what the tokens file stores."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_entry(sha: str, entry: Any, path: Path) -> tuple[str, Role]:
    if not SHA_RE.fullmatch(str(sha)):
        raise ValueError(f"{path}: key {str(sha)[:16]!r}... is not a sha256 (64 lowercase hex)")
    if not isinstance(entry, Mapping):
        raise ValueError(f"{path}: entry {sha[:12]} must be a table {{name, role}}")
    name, role = entry.get("name"), entry.get("role")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: entry {sha[:12]} has no name")
    if role not in {r.value for r in Role}:
        raise ValueError(f"{path}: entry {sha[:12]} role {role!r} is not one of {list(RANK)}")
    return name.strip(), Role(role)


def load_tokens(path: Path) -> dict[str, tuple[str, Role]]:
    """{sha256: (name, role)} from the tokens file; ValueError on any malformed entry (the
    server refuses to start rather than run with a partial table)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        doc = tomllib.loads(text) if Path(path).suffix == ".toml" else json.loads(text)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the tokens file {path}: {exc}") from exc
    if isinstance(doc, Mapping) and isinstance(doc.get("tokens"), Mapping):
        doc = doc["tokens"]
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path}: expected a mapping sha256 -> {{name, role}}")
    return {str(sha): _token_entry(str(sha), entry, Path(path)) for sha, entry in doc.items()}


@dataclass(frozen=True)
class Authenticator:
    """The principal of a request's headers under one profile and token table."""

    profile: str
    tokens: Mapping[str, tuple[str, Role]] | None = None  # None: no tokens file
    sessions: Callable[[str], Principal | None] | None = None  # session cookie -> principal

    @classmethod
    def of(
        cls, settings: ServeSettings, sessions: Callable[[str], Principal | None] | None = None
    ) -> Authenticator:
        path = settings.tokens_file
        return cls(settings.profile, None if path is None else load_tokens(path), sessions)

    def anonymous(self, via: str | None = None) -> Principal:
        name, role = ANONYMOUS[self.profile]
        return Principal(name, role, via)

    def _match(self, token: str) -> tuple[str, Role] | None:
        presented = token_sha256(token)
        found = None
        for sha, entry in (self.tokens or {}).items():  # every entry: no early exit
            if hmac.compare_digest(presented, sha):
                found = entry
        return found

    def _session(self, session: str | None, via: str | None) -> Principal:
        found = self.sessions(session) if session and self.sessions is not None else None
        return self.anonymous(via) if found is None else replace(found, via=via)

    def resolve(
        self, authorization: str | None, via: str | None, session: str | None = None
    ) -> Principal:
        """The principal of the `Authorization` and `X-Ksmech-Via` header values and of the
        session cookie's value."""
        via = _via(via)
        if authorization is None or (self.tokens is None and self.profile == "lab"):
            return self._session(session, via)
        scheme, _, token = authorization.strip().partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise ServeError(Code.TOKEN_INVALID, "expected Authorization: Bearer <token>")
        entry = self._match(token)
        if entry is None:
            raise ServeError(Code.TOKEN_INVALID, "the bearer token matches no known token")
        return Principal(entry[0], entry[1], via, authenticated=True)


def _via(raw: str | None) -> str | None:
    if raw is None or not raw.strip():
        return None
    via = raw.strip().lower()
    if not VIA_RE.fullmatch(via):
        msg = f"X-Ksmech-Via must match {VIA_RE.pattern}, got {raw[:40]!r}"
        raise ServeError(Code.REQUEST_INVALID, msg, status=422)
    return via


class AuthGate:
    """Pure ASGI middleware: the principal of every HTTP request in `scope["state"]
    ["principal"]`, or the 401 / 422 error envelope before any route runs."""

    def __init__(self, app, authenticator: Authenticator) -> None:
        self.app = app
        self.authenticator = authenticator

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        state = scope.setdefault("state", {})
        try:
            cookies = cookie_parser(headers.get("cookie", ""))
            principal = self.authenticator.resolve(
                headers.get("authorization"), headers.get(VIA_HEADER), cookies.get(SESSION_COOKIE)
            )
        except ServeError as err:
            await error_response(err, str(state.get("request_id", "")))(scope, receive, send)
            return
        state["principal"] = principal
        await self.app(scope, receive, send)


def principal_of(request: Request) -> Principal:
    """The request's principal (`AuthGate`); without the gate, the anonymous principal of the
    app's settings (lab when the app holds none)."""
    found = request.scope.get("state", {}).get("principal")
    if isinstance(found, Principal):
        return found
    services = getattr(request.app.state, "services", None)
    profile = getattr(getattr(services, "settings", None), "profile", "lab")
    return Authenticator(profile).anonymous()


def authorize(principal: Principal, role: Role | str, what: str) -> None:
    """Raise AUTH_REQUIRED (anonymous) or ROLE_INSUFFICIENT unless `principal` has `role`."""
    if principal.at_least(Role(role)):
        return
    detail = {"required": str(Role(role)), "role": str(principal.role), "for": what}
    if not principal.authenticated:
        msg = f"{what} needs the {Role(role)} role: sign in or send Authorization: Bearer <token>"
        raise ServeError(Code.AUTH_REQUIRED, msg, detail=detail)
    msg = f"{what} needs the {Role(role)} role; {principal.name} is {principal.role}"
    raise ServeError(Code.ROLE_INSUFFICIENT, msg, detail=detail)


def require_role(role: Role | str) -> Any:
    """`Depends` for `dependencies=[require_role(role)]` on a route outside the capabilities."""
    from fastapi import Depends, Request

    wanted = Role(role)

    def _require_role(request) -> None:
        what = f"{request.method} {request.url.path}"
        authorize(principal_of(request), wanted, what)

    _require_role.__annotations__ = {"request": Request, "return": None}
    return Depends(_require_role)
