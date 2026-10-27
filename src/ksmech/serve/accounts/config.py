"""The accounts configuration: the optional TOML file `KSMECH_SERVE_AUTH`.

    public_url = "https://ksmech.example.org"  # base of the OIDC redirect URIs; default: the
                                               # request's own base URL
    session_days = 14
    cookie_secure = "auto"      # true | false | auto: true when public_url is https
    password_login = true       # local accounts (ksmech-serve users add)
    default_role = "user"       # the role of a user a first OIDC sign-in creates
    admin_emails = ["a@b.org"]  # a provider-verified email at first sign-in: admin
    admin_subjects = ["orcid:0000-0002-1825-0097"]   # <provider>:<sub>: admin

    [quota.user]                # per role; omitted: DEFAULT_QUOTAS. Admins: never limited
    per_day = 100
    per_minute = 10

    [providers.orcid]           # presets: orcid, google (issuer, label, scopes)
    client_id = "APP-XXXXXXXXXXXXXXXX"
    client_secret_env = "KSMECH_ORCID_SECRET"   # or client_secret = "..." (keep it out of git)
    # issuer = "https://sandbox.orcid.org"       # the ORCID sandbox

    [providers.email]           # any other OIDC issuer (Auth0, Keycloak, Clerk, ...)
    label = "Email"
    issuer = "https://tenant.eu.auth0.com"
    client_id = "..."
    client_secret_env = "KSMECH_AUTH0_SECRET"

Without the file: password sign-in on, no provider, the default quotas. A malformed file is a
ValueError: the server refuses to start rather than run with half a configuration.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ksmech.serve.core.auth import RANK, Role

PROVIDER_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}")
PRESETS: dict[str, dict[str, Any]] = {
    "orcid": {"label": "ORCID", "issuer": "https://orcid.org", "scopes": ["openid"]},
    "google": {
        "label": "Google",
        "issuer": "https://accounts.google.com",
        "scopes": ["openid", "email", "profile"],
    },
}
DEFAULT_QUOTAS: dict[Role, tuple[int | None, int | None]] = {
    Role.READER: (0, 0),  # a reader submits no job (its role is below predict's anyway)
    Role.USER: (100, 10),
    Role.CURATOR: (1000, 30),
}
"""(per_day, per_minute) of each role; an admin has no quota (`Quota.unlimited`)."""


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    issuer: str
    client_id: str
    client_secret: str | None
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class Quota:
    per_day: int | None
    per_minute: int | None
    unlimited: bool = False


@dataclass(frozen=True)
class AuthConfig:
    public_url: str | None = None
    session_days: int = 14
    cookie_secure: bool = False
    password_login: bool = True
    default_role: Role = Role.USER
    admin_emails: frozenset[str] = frozenset()
    admin_subjects: frozenset[str] = frozenset()
    quotas: Mapping[Role, Quota] = field(default_factory=dict)
    providers: Mapping[str, Provider] = field(default_factory=dict)

    def quota(self, role: Role) -> Quota:
        if role == Role.ADMIN:
            return Quota(None, None, unlimited=True)
        if role in self.quotas:
            return self.quotas[role]
        return Quota(*DEFAULT_QUOTAS[role])


def _limit(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: expected a non-negative integer, got {value!r}")
    return value


def _quotas(doc: Mapping[str, Any]) -> dict[Role, Quota]:
    out = {}
    for name, entry in doc.items():
        if name not in RANK or name == Role.ADMIN:
            raise ValueError(f"quota.{name}: not a limited role ({', '.join(DEFAULT_QUOTAS)})")
        default = DEFAULT_QUOTAS[Role(name)]
        day = _limit(entry.get("per_day", default[0]), f"quota.{name}.per_day")
        minute = _limit(entry.get("per_minute", default[1]), f"quota.{name}.per_minute")
        out[Role(name)] = Quota(day, minute)
    return out


def _secret(pid: str, entry: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    if "client_secret" in entry:
        return str(entry["client_secret"])
    var = entry.get("client_secret_env")
    if var is None:
        return None
    if not env.get(str(var)):
        raise ValueError(f"providers.{pid}: the environment variable {var} is not set")
    return env[str(var)]


def _provider(pid: str, entry: Mapping[str, Any], env: Mapping[str, str]) -> Provider:
    if not PROVIDER_ID.fullmatch(pid):
        raise ValueError(f"providers.{pid}: an id is lowercase letters, digits, _ and -")
    merged = {**PRESETS.get(pid, {}), **entry}
    for key in ("issuer", "client_id"):
        if not merged.get(key):
            raise ValueError(f"providers.{pid}: {key} is required")
    issuer = str(merged["issuer"]).rstrip("/")
    if not issuer.startswith("https://"):
        raise ValueError(f"providers.{pid}: the issuer must be an https URL")
    scopes = tuple(str(s) for s in merged.get("scopes", ["openid"]))
    if "openid" not in scopes:
        scopes = ("openid", *scopes)
    label = str(merged.get("label", pid))
    secret = _secret(pid, entry, env)
    return Provider(pid, label, issuer, str(merged["client_id"]), secret, scopes)


def _secure(value: Any, public_url: str | None) -> bool:
    if value in (None, "auto"):
        return bool(public_url and public_url.startswith("https://"))
    if isinstance(value, bool):
        return value
    raise ValueError(f"cookie_secure: expected true, false or 'auto', got {value!r}")


def parse_config(doc: Mapping[str, Any], env: Mapping[str, str] = os.environ) -> AuthConfig:
    """The `AuthConfig` of a parsed TOML document."""
    public_url = doc.get("public_url")
    public_url = None if public_url is None else str(public_url).rstrip("/")
    role = str(doc.get("default_role", "user"))
    if role not in RANK:
        raise ValueError(f"default_role: {role!r} is not one of {list(RANK)}")
    days = doc.get("session_days", 14)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError(f"session_days: expected 1..365, got {days!r}")
    providers = {pid: _provider(pid, e, env) for pid, e in doc.get("providers", {}).items()}
    return AuthConfig(
        public_url=public_url,
        session_days=days,
        cookie_secure=_secure(doc.get("cookie_secure"), public_url),
        password_login=bool(doc.get("password_login", True)),
        default_role=Role(role),
        admin_emails=frozenset(str(e).strip().lower() for e in doc.get("admin_emails", [])),
        admin_subjects=frozenset(str(s).strip() for s in doc.get("admin_subjects", [])),
        quotas=_quotas(doc.get("quota", {})),
        providers=providers,
    )


def load_config(path: Path | None, env: Mapping[str, str] = os.environ) -> AuthConfig:
    """The configuration of the auth file at `path`; the defaults when `path` is None."""
    if path is None:
        return AuthConfig()
    try:
        doc = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the auth file {path}: {exc}") from exc
    try:
        return parse_config(doc, env)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
