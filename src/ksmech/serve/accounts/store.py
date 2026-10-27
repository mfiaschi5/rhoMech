"""The accounts database (`data/serve/accounts/accounts.sqlite`): users, their OIDC identities,
browser sessions and the job usage the quotas count.

A session row holds only the sha256 of its cookie value, as the tokens file of `core.auth` holds
only the sha256 of a token. Datetimes are aware UTC (`core.db.UTCDateTime`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import UniqueConstraint
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel

from ksmech.serve.core.db import UTCDateTime, make_engine, utcnow


class User(SQLModel, table=True):
    __tablename__ = "account_user"

    id: int | None = Field(default=None, primary_key=True)
    login: str = Field(index=True, unique=True)  # lowercase (schemas.LOGIN_PATTERN)
    name: str
    email: str | None = Field(default=None, index=True, unique=True)  # lowercase
    role: str  # core.auth.Role
    password_hash: str | None = None  # passwords.hash_password; None: OIDC sign-in only
    disabled: bool = False
    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime)
    last_login_at: datetime | None = Field(default=None, sa_type=UTCDateTime)


class Identity(SQLModel, table=True):
    """An OIDC subject (`sub` of the provider's ID token) linked to one user."""

    __tablename__ = "account_identity"
    __table_args__ = (UniqueConstraint("provider", "subject"),)

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)
    subject: str
    user_id: int = Field(foreign_key="account_user.id", index=True)
    email: str | None = None
    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime)


class UserSession(SQLModel, table=True):
    __tablename__ = "account_session"

    token_sha: str = Field(primary_key=True)  # sha256 of the cookie value
    user_id: int = Field(foreign_key="account_user.id", index=True)
    method: str  # password | <provider id>
    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime)
    expires_at: datetime = Field(sa_type=UTCDateTime, index=True)


class UsageEvent(SQLModel, table=True):
    """One submitted job of a principal (`quotas`): `key` is `user:<id>` or `name:<name>`."""

    __tablename__ = "account_usage"

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(index=True)
    kind: str  # the capability: predict | embedding
    at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime, index=True)


TABLES = (User, Identity, UserSession, UsageEvent)


def open_store(path: Path) -> Engine:
    """The engine of the accounts database at `path`, its tables created (idempotent)."""
    engine = make_engine(path)
    SQLModel.metadata.create_all(engine, tables=[t.__table__ for t in TABLES])
    return engine
