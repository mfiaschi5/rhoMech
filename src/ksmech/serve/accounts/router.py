"""Routes of the accounts: sign-in and sign-out (`/auth/*`, open to everyone) and the user
management of the admins (`/admin/users*`, admin only, in every profile).

The session travels in an HttpOnly, SameSite=Lax cookie (`core.auth.SESSION_COOKIE`), Secure
when the auth file says so. An OIDC sign-in is two browser navigations: `start` redirects to
the provider and binds the sign-in to this browser with a short cookie holding its state, and
`callback` checks that cookie, finishes the sign-in and redirects to the app, with
`#/login?error=...` when it failed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import RedirectResponse

from ksmech.serve.accounts.service import AccountService
from ksmech.serve.core import schemas as S
from ksmech.serve.core.auth import SESSION_COOKIE, Authenticator, Role, principal_of, require_role
from ksmech.serve.core.errors import Code, ServeError

STATE_COOKIE = "ksmech_oidc"
STATE_PATH = "/api/auth/oidc"

router = APIRouter(prefix="/api/auth")
admin = APIRouter(prefix="/api/admin", dependencies=[require_role(Role.ADMIN)])


def _accounts(request: Request) -> AccountService:
    found = getattr(request.app.state.services, "accounts", None)
    if found is None:
        raise ServeError(Code.FEATURE_DISABLED, "accounts are not set up in this app")
    return found


Accounts = Annotated[AccountService, Depends(_accounts)]


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_session(response: Response, accounts: AccountService, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=accounts.config.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=accounts.config.cookie_secure,
        path="/",
    )


@router.get("/me", response_model=S.AuthStatusOut)
def me(request: Request, accounts: Accounts) -> S.AuthStatusOut:
    """Who this request is, the providers and the job quota of its role."""
    return accounts.status(principal_of(request))


@router.post("/login", response_model=S.AuthStatusOut)
def login(
    request: Request, response: Response, body: S.LoginIn, accounts: Accounts
) -> S.AuthStatusOut:
    """Sign in with a local password: sets the session cookie."""
    via = principal_of(request).via
    principal, token = accounts.login(body.login, body.password, _client(request))
    _set_session(response, accounts, token)
    return accounts.status(replace(principal, via=via))


@router.post("/logout", response_model=S.AuthStatusOut)
def logout(request: Request, response: Response, accounts: Accounts) -> S.AuthStatusOut:
    """End this browser's session (the cookie is cleared even without one)."""
    accounts.logout(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    profile = request.app.state.services.settings.profile
    return accounts.status(Authenticator(profile).anonymous(principal_of(request).via))


@router.post("/password", response_model=S.UserOut)
def change_password(request: Request, body: S.PasswordChangeIn, accounts: Accounts) -> S.UserOut:
    """Change the signed-in user's password (every session of the user ends, this one too)."""
    return accounts.change_password(principal_of(request), body.current, body.new)


@router.get("/oidc/{provider}/start", response_class=RedirectResponse, status_code=302)
def oidc_start(
    request: Request,
    provider: str,
    accounts: Accounts,
    return_to: Annotated[str | None, Query(max_length=300)] = None,
) -> RedirectResponse:
    """Redirect to the provider's sign-in page."""
    url, state = accounts.oidc_start(provider, str(request.base_url), return_to)
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=accounts.config.cookie_secure,
        path=STATE_PATH,
    )
    return response


def _back(route: str) -> RedirectResponse:
    response = RedirectResponse(f"/{route}", status_code=302)
    response.delete_cookie(STATE_COOKIE, path=STATE_PATH)
    return response


@router.get("/oidc/{provider}/callback", response_class=RedirectResponse, status_code=302)
def oidc_callback(
    request: Request,
    provider: str,
    accounts: Accounts,
    code: Annotated[str | None, Query(max_length=4096)] = None,
    state: Annotated[str | None, Query(max_length=256)] = None,
    error: Annotated[str | None, Query(max_length=256)] = None,
) -> RedirectResponse:
    """The provider's redirect back: a session and the app, or `#/login?error=`."""
    try:
        if error is not None:
            raise ServeError(Code.OIDC_FAILED, f"the provider answered: {error}")
        if not code or not state or request.cookies.get(STATE_COOKIE) != state:
            raise ServeError(Code.OIDC_FAILED, "this sign-in did not start in this browser")
        token, return_to = accounts.oidc_finish(provider, state, code)
    except ServeError as exc:
        return _back(f"#/login?error={quote(exc.message[:200])}")
    response = _back(return_to)
    _set_session(response, accounts, token)
    return response


@admin.get("/users", response_model=S.UsersOut)
def list_users(accounts: Accounts) -> S.UsersOut:
    return accounts.list_users()


@admin.post("/users", response_model=S.UserOut, status_code=201)
def create_user(body: S.UserCreateIn, accounts: Accounts) -> S.UserOut:
    return accounts.create_user(body.model_dump())


@admin.patch("/users/{user_id}", response_model=S.UserOut)
def update_user(user_id: int, body: S.UserPatchIn, accounts: Accounts) -> S.UserOut:
    return accounts.update_user(user_id, body.model_dump(exclude_unset=True))
