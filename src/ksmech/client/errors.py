"""The errors of the ksmech HTTP client.

The API answers every error with the envelope `{"error": {"code", "message", "upstream",
"detail", "request_id"}}` (`ksmech.serve.core.schemas.ErrorOut`); `KsmechAPIError` carries those
fields and the HTTP status. Failures that never reach the API get a client code and status 0:
`CONNECTION_FAILED` (no answer), `CLIENT_TIMEOUT` (a job or a pending task outlived the wait),
`RESPONSE_INVALID` (an answer that is not the documented JSON).
"""

from __future__ import annotations

from typing import Any

CONNECTION_FAILED = "CONNECTION_FAILED"
CLIENT_TIMEOUT = "CLIENT_TIMEOUT"
RESPONSE_INVALID = "RESPONSE_INVALID"


class KsmechAPIError(Exception):
    """An error answer of the API (or a client-side failure, status 0)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 0,
        detail: dict | None = None,
        request_id: str | None = None,
        upstream: str | None = None,
    ) -> None:
        super().__init__(f"[{code}] {message}" + (f" (HTTP {status})" if status else ""))
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail
        self.request_id = request_id
        self.upstream = upstream

    def to_dict(self) -> dict[str, Any]:
        """The error as the API's error body (plus the status), for machine output."""
        return {
            "code": self.code,
            "message": self.message,
            "status": self.status,
            "detail": self.detail,
            "request_id": self.request_id,
            "upstream": self.upstream,
        }


class KsmechTimeout(KsmechAPIError):
    """A wait ran out: `last` is the last answer seen (a `JobOut` or a `PendingOut`)."""

    def __init__(self, message: str, last: Any = None) -> None:
        super().__init__(CLIENT_TIMEOUT, message)
        self.last = last


class KsmechPending(KsmechTimeout):
    """The API kept answering 202 `PendingOut` (a worker task still runs) past the wait."""

    def __init__(self, pending: Any) -> None:
        label = getattr(pending, "label", "") or "a worker task"
        super().__init__(f"still pending: {label} (retry later)", last=pending)
        self.pending = pending
