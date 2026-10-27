"""Job quotas: how many predict, expand and embedding jobs a principal may submit.

The quota is the role's (`AuthConfig.quota`): at most `per_day` jobs in the last 24 hours and
`per_minute` in the last minute, None meaning no limit of that kind. An admin is never limited
and its jobs are not even counted. A principal is counted by its user (`user:<id>`) or, for a
bearer token or the anonymous principal, by its name (`name:<name>`).

`charge` runs in the route's dependency, before the job exists (`core.capabilities`): a refused
job is QUOTA_EXCEEDED (429) and is not counted.
"""

from __future__ import annotations

import threading
from datetime import timedelta

from sqlalchemy import delete, func
from sqlalchemy.engine import Engine
from sqlmodel import select

from ksmech.serve.accounts.config import AuthConfig, Quota
from ksmech.serve.accounts.store import UsageEvent
from ksmech.serve.core import schemas as S
from ksmech.serve.core.auth import Principal
from ksmech.serve.core.db import session, utcnow
from ksmech.serve.core.errors import Code, ServeError

COUNTED = frozenset({"predict", "embedding"})  # the capabilities whose POST submits a job
KEEP = timedelta(days=2)


def usage_key(principal: Principal) -> str:
    if principal.user_id is not None:
        return f"user:{principal.user_id}"
    return f"name:{principal.name}"


class Quotas:
    def __init__(self, engine: Engine, config: AuthConfig) -> None:
        self.engine = engine
        self.config = config
        self._lock = threading.Lock()  # check-then-insert of one server process

    def _count(self, s, key: str, since) -> int:
        query = select(func.count()).select_from(UsageEvent)
        query = query.where(UsageEvent.key == key, UsageEvent.at > since)
        return int(s.exec(query).one())

    def used_today(self, principal: Principal) -> int:
        with session(self.engine) as s:
            return self._count(s, usage_key(principal), utcnow() - timedelta(days=1))

    def quota_out(self, principal: Principal) -> S.QuotaOut:
        quota = self.config.quota(principal.role)
        return S.QuotaOut(
            per_day=quota.per_day,
            per_minute=quota.per_minute,
            used_today=0 if quota.unlimited else self.used_today(principal),
            unlimited=quota.unlimited,
        )

    def _refuse(self, quota: Quota, which: str, limit: int) -> None:
        detail = {"limit": limit, "window": which, "per_day": quota.per_day}
        detail |= {"per_minute": quota.per_minute}
        msg = f"job quota reached: {limit} per {which} for this role"
        raise ServeError(Code.QUOTA_EXCEEDED, msg, detail=detail)

    def charge(self, principal: Principal, kind: str) -> None:
        """Count one job of `kind`, or QUOTA_EXCEEDED; admins and uncounted kinds pass."""
        quota = self.config.quota(principal.role)
        if kind not in COUNTED or quota.unlimited:
            return
        key, now = usage_key(principal), utcnow()
        with self._lock, session(self.engine) as s:
            windows = (("day", quota.per_day, now - timedelta(days=1)),)
            windows += (("minute", quota.per_minute, now - timedelta(minutes=1)),)
            for which, limit, since in windows:
                if limit is not None and self._count(s, key, since) >= limit:
                    self._refuse(quota, which, limit)
            s.add(UsageEvent(key=key, kind=kind, at=now))
            s.execute(delete(UsageEvent).where(UsageEvent.at < now - KEEP))
