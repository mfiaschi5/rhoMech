"""Users: create, edit, list, sign in by password, and link an OIDC identity.

Logins and emails are stored lowercase and are unique. The last active admin cannot be
demoted or disabled (LAST_ADMIN), so the platform always keeps someone who can manage it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import select

from ksmech.serve.accounts.config import AuthConfig, Provider
from ksmech.serve.accounts.passwords import hash_password, verify_password
from ksmech.serve.accounts.store import Identity, User, UserSession
from ksmech.serve.core import schemas as S
from ksmech.serve.core.auth import Role
from ksmech.serve.core.db import session, utcnow
from ksmech.serve.core.errors import Code, ServeError

_DUMMY_HASH = hash_password("x" * 16)  # verified against when the login is unknown


def _norm(value: str | None) -> str | None:
    value = None if value is None else value.strip().lower()
    return value or None


def to_out(engine: Engine, user: User) -> S.UserOut:
    with session(engine) as s:
        rows = s.exec(select(Identity.provider).where(Identity.user_id == user.id)).all()
    return S.UserOut(
        id=int(user.id or 0),
        login=user.login,
        name=user.name,
        email=user.email,
        role=user.role,
        disabled=user.disabled,
        has_password=user.password_hash is not None,
        providers=sorted(set(rows)),
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


def get_user(engine: Engine, user_id: int) -> User:
    with session(engine) as s:
        user = s.get(User, user_id)
    if user is None:
        raise ServeError(Code.USER_NOT_FOUND, f"no user {user_id}", status=404)
    return user


def list_users(engine: Engine) -> list[User]:
    with session(engine) as s:
        return list(s.exec(select(User).order_by(User.id)).all())


def _taken(s, login: str, email: str | None) -> None:
    clash = s.exec(select(User).where(User.login == login)).first()
    if clash is None and email is not None:
        clash = s.exec(select(User).where(User.email == email)).first()
    if clash is not None:
        detail = {"login": login, "email": email}
        raise ServeError(Code.USER_EXISTS, "that login or email is taken", detail=detail)


def create_user(
    engine: Engine,
    login: str,
    name: str,
    role: Role | str,
    *,
    email: str | None = None,
    password: str | None = None,
) -> User:
    """A new user; USER_EXISTS when the login or the email is taken."""
    login, email = str(_norm(login)), _norm(email)
    hashed = None if password is None else hash_password(password)
    with session(engine) as s:
        _taken(s, login, email)
        user = User(login=login, name=name.strip(), email=email, role=str(Role(role)))
        user.password_hash = hashed
        s.add(user)
        s.flush()
        s.refresh(user)
    return user


def _active_admins(s) -> int:
    query = select(User).where(User.role == str(Role.ADMIN), User.disabled == False)  # noqa: E712
    return len(s.exec(query).all())


def _revoke_all(s, user_id: int) -> None:
    for row in s.exec(select(UserSession).where(UserSession.user_id == user_id)).all():
        s.delete(row)


def update_user(engine: Engine, user_id: int, patch: Mapping[str, Any]) -> User:
    """Apply `patch` (`UserPatchIn` fields, None kept); LAST_ADMIN when it would leave no
    active admin. A new password or `disabled` ends the user's sessions."""
    with session(engine) as s:
        user = s.get(User, user_id)
        if user is None:
            raise ServeError(Code.USER_NOT_FOUND, f"no user {user_id}", status=404)
        was_admin = user.role == Role.ADMIN and not user.disabled
        if patch.get("name") is not None:
            user.name = str(patch["name"]).strip()
        if patch.get("role") is not None:
            user.role = str(Role(patch["role"]))
        if patch.get("disabled") is not None:
            user.disabled = bool(patch["disabled"])
        if patch.get("password") is not None:
            user.password_hash = hash_password(str(patch["password"]))
        s.add(user)
        s.flush()
        if was_admin and _active_admins(s) == 0:
            raise ServeError(Code.LAST_ADMIN, "the platform must keep one active admin")
        if user.disabled or patch.get("password") is not None:
            _revoke_all(s, user_id)
        s.refresh(user)
    return user


def check_password(engine: Engine, login: str, password: str) -> User:
    """The user of `login` (a login name or an email) if `password` is right; LOGIN_FAILED
    otherwise, with the same message and about the same time for every cause."""
    key = _norm(login) or ""
    with session(engine) as s:
        user = s.exec(select(User).where((User.login == key) | (User.email == key))).first()
    ok = verify_password(password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok or user.disabled:
        raise ServeError(Code.LOGIN_FAILED, "wrong login or password", status=401)
    return user


def touch_login(engine: Engine, user_id: int) -> None:
    with session(engine) as s:
        user = s.get(User, user_id)
        if user is not None:
            user.last_login_at = utcnow()
            s.add(user)


def _free_login(s, base: str) -> str:
    stem = "".join(c if c.isalnum() or c in "_.-" else "-" for c in base.lower())[:32]
    stem = stem.strip("-._") or "user"
    stem = stem if len(stem) >= 2 else f"{stem}-user"
    login, n = stem, 1
    while s.exec(select(User).where(User.login == login)).first() is not None:
        n += 1
        login = f"{stem}-{n}"
    return login


def _first_role(config: AuthConfig, provider: Provider, claims: Mapping[str, Any]) -> Role:
    email = _norm(claims.get("email")) if claims.get("email_verified") else None
    if f"{provider.id}:{claims['sub']}" in config.admin_subjects:
        return Role.ADMIN
    if email is not None and email in config.admin_emails:
        return Role.ADMIN
    return config.default_role


def _display_name(claims: Mapping[str, Any]) -> str:
    name = claims.get("name") or " ".join(
        str(claims[k]) for k in ("given_name", "family_name") if claims.get(k)
    )
    return str(name or claims.get("email") or claims["sub"]).strip()[:120]


def user_of_identity(
    engine: Engine, config: AuthConfig, provider: Provider, claims: Mapping[str, Any]
) -> User:
    """The user an OIDC sign-in names: the linked one; else the user with the same
    provider-verified email (linked now); else a new user with the configured first role."""
    subject = str(claims["sub"])
    email = _norm(claims.get("email")) if claims.get("email_verified") else None
    with session(engine) as s:
        query = select(Identity).where(
            Identity.provider == provider.id, Identity.subject == subject
        )
        linked = s.exec(query).first()
        if linked is not None:
            user = s.get(User, linked.user_id)
        else:
            user = (
                None if email is None else s.exec(select(User).where(User.email == email)).first()
            )
            if user is None:
                base = (email or "").split("@")[0] or _display_name(claims)
                login = _free_login(s, base)
                role = str(_first_role(config, provider, claims))
                user = User(login=login, name=_display_name(claims), email=email, role=role)
                s.add(user)
                s.flush()
            s.add(Identity(provider=provider.id, subject=subject, user_id=user.id, email=email))
        s.flush()
        s.refresh(user)
    if user.disabled:
        raise ServeError(Code.LOGIN_FAILED, "this account is disabled", status=401)
    return user
