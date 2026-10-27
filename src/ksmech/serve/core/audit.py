"""The audit log: one JSON line per request that may change something.

Every HTTP request whose method is not GET or HEAD (read-only by HTTP's definition) appends
`{time, request_id, principal: {name, role, via} | null, method, route, path, status}` to
`<data/serve/logs>/audit.jsonl` (`ServeSettings.audit_log`) once its response has started, or
with status 500 when it raised. `route` is the matched route template (`/api/curate/{mid}`),
None when no route matched (`path` then says what was asked, cut to 200 characters); the
principal is None when the request was refused before `auth.AuthGate` named one (a bad token,
a foreign origin). The line is written under `assert_writable`; a log that cannot be written is
reported through logging and `failures.jsonl` (no silent failure) and never fails the request.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ksmech.serve.core.settings import append_failure, assert_writable

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

log = logging.getLogger(__name__)

READ_METHODS = frozenset({"GET", "HEAD"})
AUDIT_FAILED = "AUDIT_WRITE_FAILED"
_LOCK = threading.Lock()


def append_audit(settings: ServeSettings, record: dict[str, Any]) -> None:
    """Append one record to the audit log. Never raises."""
    path = settings.audit_log
    try:
        assert_writable(path, settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with _LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError) as exc:
        log.error("cannot append to %s [%s]: %s", path, AUDIT_FAILED, exc)
        append_failure(settings, AUDIT_FAILED, str(exc), where="audit")


def audit_record(scope: dict, status: int) -> dict[str, Any]:
    """The audit line of a finished request (`scope` after routing)."""
    state = scope.get("state", {})
    principal = state.get("principal")
    route = scope.get("route")
    return {
        "time": datetime.now(UTC).isoformat(),
        "request_id": str(state.get("request_id", "")),
        "principal": None if principal is None else principal.to_out(),
        "method": scope.get("method"),
        "route": getattr(route, "path", None),
        "path": str(scope.get("path", ""))[:200],
        "status": int(status),
    }


class AuditLog:
    """Pure ASGI middleware: the audit line of every non-GET HTTP request."""

    def __init__(self, app, settings: ServeSettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") in READ_METHODS:
            await self.app(scope, receive, send)
            return
        status = 500

        async def send_status(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            append_audit(self.settings, audit_record(scope, status))
