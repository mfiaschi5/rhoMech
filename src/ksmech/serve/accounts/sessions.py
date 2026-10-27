"""Browser sessions and the throttle of failed sign-ins.

A session is a random 32-byte cookie value; the database keeps only its sha256 and the expiry.
Resolving one reads the user each time, so a role change, a disabled account or a new password
takes effect on the next request.

`LoginThrottle` counts failed sign-ins per client address and per login in memory (one server
process): `MAX_FAILURES` within `WINDOW_S` blocks further attempts with LOGIN_THROTTLED until
the oldest failure leaves the window.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta

from sqlalchemy.engine import Engine

from ksmech.serve.accounts.store import User, UserSession
from ksmech.serve.core.auth import Principal, Role
from ksmech.serve.core.db import session, utcnow
from ksmech.serve.core.errors import Code, ServeError

MAX_FAILURES = 10
WINDOW_S = 15 * 60


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_session(engine: Engine, user_id: int, method: str, days: int) -> str:
    """A new session of `user_id`; returns the cookie value (never stored)."""
    token = secrets.token_urlsafe(32)
    expires = utcnow() + timedelta(days=days)
    with session(engine) as s:
        s.add(
            UserSession(token_sha=_sha(token), user_id=user_id, method=method, expires_at=expires)
        )
    return token


def resolve_session(engine: Engine, token: str) -> Principal | None:
    """The principal of a session cookie value; None when unknown, expired or disabled."""
    if not token or len(token) > 128:
        return None
    with session(engine) as s:
        row = s.get(UserSession, _sha(token))
        if row is None:
            return None
        if row.expires_at <= utcnow():
            s.delete(row)
            return None
        user = s.get(User, row.user_id)
    if user is None or user.disabled:
        return None
    return Principal(user.login, Role(user.role), authenticated=True, user_id=user.id)


def end_session(engine: Engine, token: str | None) -> None:
    if not token:
        return
    with session(engine) as s:
        row = s.get(UserSession, _sha(token))
        if row is not None:
            s.delete(row)


def purge_expired(engine: Engine) -> int:
    """Delete the expired sessions; returns how many."""
    from sqlmodel import select

    with session(engine) as s:
        rows = s.exec(select(UserSession).where(UserSession.expires_at <= utcnow())).all()
        for row in rows:
            s.delete(row)
    return len(rows)


class LoginThrottle:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque[float]:
        q = self._failures[key]
        while q and q[0] <= now - WINDOW_S:
            q.popleft()
        return q

    def check(self, *keys: str) -> None:
        """LOGIN_THROTTLED when any key has `MAX_FAILURES` recent failures."""
        now = self._clock()
        with self._lock:
            for key in keys:
                q = self._recent(key, now)
                if len(q) >= MAX_FAILURES:
                    wait = int(q[0] + WINDOW_S - now) + 1
                    msg = f"too many failed sign-ins: try again in {wait} s"
                    raise ServeError(Code.LOGIN_THROTTLED, msg, detail={"retry_after_s": wait})

    def fail(self, *keys: str) -> None:
        now = self._clock()
        with self._lock:
            for key in keys:
                self._recent(key, now).append(now)

    def clear(self, *keys: str) -> None:
        with self._lock:
            for key in keys:
                self._failures.pop(key, None)
