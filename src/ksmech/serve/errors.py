"""Reason codes, their HTTP statuses, and the two exceptions every service raises.

`ServeError` carries a code (`Code`), a message, the HTTP status (from `STATUS`, default 422),
an upstream reason (a qm `ReasonCode`, an `ApplyReason`, a `RejectReason`) and a JSON-able
detail dict. `Pending` means "compute is running, retry the same URL": the API turns it into
202 `PendingOut`. The FastAPI handlers live in `api.py`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ksmech.serve.contracts import to_jsonable

if TYPE_CHECKING:
    from ksmech.serve.worker import TaskResult


class Code(StrEnum):
    # 422: the request
    REQUEST_INVALID = "REQUEST_INVALID"
    EMPTY_INPUT = "EMPTY_INPUT"
    INVALID_SMILES = "INVALID_SMILES"
    RADICAL_NOT_SUPPORTED = "RADICAL_NOT_SUPPORTED"
    SPIN_PARITY = "SPIN_PARITY"
    UNSUPPORTED_ELEMENT = "UNSUPPORTED_ELEMENT"
    STOICHIOMETRY_MISMATCH = "STOICHIOMETRY_MISMATCH"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    PREDICTOR_UNKNOWN = "PREDICTOR_UNKNOWN"
    NOT_AN_ORBITAL_TOKEN = "NOT_AN_ORBITAL_TOKEN"
    TOKEN_OUT_OF_RANGE = "TOKEN_OUT_OF_RANGE"
    SITE_OUT_OF_RANGE = "SITE_OUT_OF_RANGE"
    SITE_NOT_DRAWN = "SITE_NOT_DRAWN"
    SITE_NOT_A_CENTRE = "SITE_NOT_A_CENTRE"
    SAME_SITE = "SAME_SITE"
    DONOR_NOT_DONOR = "DONOR_NOT_DONOR"
    ACCEPTOR_NOT_ACCEPTOR = "ACCEPTOR_NOT_ACCEPTOR"
    PAIR_EXCLUDED = "PAIR_EXCLUDED"
    APPLY_FAILED = "APPLY_FAILED"
    RESONANCE_ONLY_STEP = "RESONANCE_ONLY_STEP"
    OVERRIDE_NOOP = "OVERRIDE_NOOP"
    STATUS_INCONSISTENT = "STATUS_INCONSISTENT"
    ORDER_GROUPS_INVALID = "ORDER_GROUPS_INVALID"
    NOT_INDEPENDENT = "NOT_INDEPENDENT"
    QUERY_INVALID = "QUERY_INVALID"
    CUBE_TOO_LARGE = "CUBE_TOO_LARGE"
    # 404
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    NODE_NOT_FOUND = "NODE_NOT_FOUND"
    EDGE_NOT_FOUND = "EDGE_NOT_FOUND"
    PATHWAY_NOT_FOUND = "PATHWAY_NOT_FOUND"
    POOL_NOT_FOUND = "POOL_NOT_FOUND"
    MECHANISM_NOT_FOUND = "MECHANISM_NOT_FOUND"
    STEP_NOT_FOUND = "STEP_NOT_FOUND"
    EXPORT_NOT_FOUND = "EXPORT_NOT_FOUND"
    # 403: a cross-origin mutating request (api.SameOriginGuard)
    REQUEST_FORBIDDEN = "REQUEST_FORBIDDEN"
    # 409
    PREDICTOR_UNAVAILABLE = "PREDICTOR_UNAVAILABLE"
    NETWORK_BUSY = "NETWORK_BUSY"
    JOB_NOT_DONE = "JOB_NOT_DONE"
    POOL_NOT_ANALYZED = "POOL_NOT_ANALYZED"
    PHYSICS_VERSION_MISMATCH = "PHYSICS_VERSION_MISMATCH"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    # 500 / 503
    PLACEMENT_MISMATCH = "PLACEMENT_MISMATCH"
    CONSISTENCY_FAILED = "CONSISTENCY_FAILED"
    UNEXPECTED = "UNEXPECTED"
    WORKER_UNSTABLE = "WORKER_UNSTABLE"
    TOKENS_CHANGED = "TOKENS_CHANGED"  # transient: a token file was rebuilt mid-computation
    # predictor availability (PREDICTOR_UNAVAILABLE detail.code)
    TRANSFORMER_NOT_AVAILABLE = "TRANSFORMER_NOT_AVAILABLE"
    GBDT_MISSING = "GBDT_MISSING"
    GBDT_FEATURES_MISMATCH = "GBDT_FEATURES_MISMATCH"
    GBDT_LOAD_FAILED = "GBDT_LOAD_FAILED"
    # job, node and report reasons (never an HTTP status of their own)
    FEATURIZATION_FAILED = "FEATURIZATION_FAILED"
    FEATURIZATION_DISABLED = "FEATURIZATION_DISABLED"
    TOKENS_FAILED = "TOKENS_FAILED"
    SYNTHETIC_FAILED = "SYNTHETIC_FAILED"
    TOKENS_STALE = "TOKENS_STALE"
    MISSING_TOKENS = "MISSING_TOKENS"
    CORRESPONDENCE = "CORRESPONDENCE"
    IMPLICIT_H_IN_POOL = "IMPLICIT_H_IN_POOL"
    SPECIES_KEY_FAILED = "SPECIES_KEY_FAILED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    WORKER_TIMEOUT = "WORKER_TIMEOUT"
    WORKER_CRASH = "WORKER_CRASH"
    CANCELLED = "CANCELLED"
    SERVER_RESTART = "SERVER_RESTART"
    NODE_BUDGET = "NODE_BUDGET"
    NO_CANDIDATES = "NO_CANDIDATES"
    NO_APPLICABLE_PROPOSAL = "NO_APPLICABLE_PROPOSAL"
    MAX_STEPS = "MAX_STEPS"
    TURNOVER = "TURNOVER"
    ARROW_ENCODING_FAILED = "ARROW_ENCODING_FAILED"
    DUPLICATE_OF_PMECHDB = "DUPLICATE_OF_PMECHDB"
    DUPLICATE_IN_EXPORT = "DUPLICATE_IN_EXPORT"
    COMPACT_FALLBACK = "COMPACT_FALLBACK"
    UNALIGNED = "UNALIGNED"
    INDEPENDENCE_UNCHECKED = "INDEPENDENCE_UNCHECKED"
    ANALYTIC_OVERLAP_DRIFT = "ANALYTIC_OVERLAP_DRIFT"
    SITE_PAIR_EXCLUDED = "SITE_PAIR_EXCLUDED"
    ORIENTED_CACHE_WIPED = "ORIENTED_CACHE_WIPED"
    PMECHDB_UNAVAILABLE = "PMECHDB_UNAVAILABLE"  # export report: no PMechDB de-duplication


_BY_STATUS = {
    422: (
        "REQUEST_INVALID EMPTY_INPUT INVALID_SMILES RADICAL_NOT_SUPPORTED SPIN_PARITY "
        "UNSUPPORTED_ELEMENT STOICHIOMETRY_MISMATCH LIMIT_EXCEEDED PREDICTOR_UNKNOWN "
        "NOT_AN_ORBITAL_TOKEN TOKEN_OUT_OF_RANGE SITE_OUT_OF_RANGE SITE_NOT_DRAWN "
        "SITE_NOT_A_CENTRE SAME_SITE DONOR_NOT_DONOR ACCEPTOR_NOT_ACCEPTOR PAIR_EXCLUDED "
        "APPLY_FAILED RESONANCE_ONLY_STEP OVERRIDE_NOOP STATUS_INCONSISTENT "
        "ORDER_GROUPS_INVALID NOT_INDEPENDENT QUERY_INVALID CUBE_TOO_LARGE"
    ),
    404: (
        "JOB_NOT_FOUND NODE_NOT_FOUND EDGE_NOT_FOUND PATHWAY_NOT_FOUND POOL_NOT_FOUND "
        "MECHANISM_NOT_FOUND STEP_NOT_FOUND EXPORT_NOT_FOUND"
    ),
    403: "REQUEST_FORBIDDEN",
    409: (
        "PREDICTOR_UNAVAILABLE NETWORK_BUSY JOB_NOT_DONE POOL_NOT_ANALYZED "
        "PHYSICS_VERSION_MISMATCH INVALID_TRANSITION INDEX_NOT_READY"
    ),
    500: "PLACEMENT_MISMATCH CONSISTENCY_FAILED UNEXPECTED WORKER_CRASH",
    503: "WORKER_UNSTABLE TOKENS_CHANGED",
    504: "WORKER_TIMEOUT",
}
STATUS: dict[Code, int] = {Code(c): s for s, codes in _BY_STATUS.items() for c in codes.split()}
"""HTTP status per code. A code not listed (a job or report reason) defaults to 422."""


def _code(code: Code | str) -> Code | str:
    try:
        return Code(str(code))
    except ValueError:
        return str(code)  # an upstream or package-local code: kept verbatim


class ServeError(Exception):
    """A reason-coded failure with its HTTP status."""

    def __init__(
        self,
        code: Code | str,
        message: str,
        *,
        status: int | None = None,
        upstream: str | None = None,
        detail: dict | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = _code(code)
        self.message = message
        self.status = int(status) if status is not None else STATUS.get(self.code, 422)
        self.upstream = None if upstream is None else str(upstream)
        self.detail = detail

    @classmethod
    def from_task(cls, r: TaskResult) -> ServeError:
        """The error of a failed task. A `ServeError` raised inside the task keeps its status,
        message and detail (run_task ships them in `result["error"]`)."""
        shipped = (r.result or {}).get("error") if isinstance(r.result, dict) else None
        code = r.code or Code.UNEXPECTED
        message = r.detail or f"task failed with {code}"
        if shipped:
            return cls(
                code,
                str(shipped.get("message") or message),
                status=shipped.get("status"),
                upstream=r.upstream,
                detail=shipped.get("detail"),
            )
        detail = {"pid": r.pid, "seconds": r.seconds}
        return cls(code, message, upstream=r.upstream, detail=detail)

    def body(self, request_id: str = "") -> dict:
        return error_body(
            self.code,
            self.message,
            upstream=self.upstream,
            detail=self.detail,
            request_id=request_id,
        )


class Pending(Exception):  # noqa: N818 - not an error: "retry the same URL"
    """Compute for this request is queued or running (202 PendingOut)."""

    def __init__(self, task_id: str, label: str, retry_after_s: float, worker: dict) -> None:
        super().__init__(f"pending: {label} ({task_id})")
        self.task_id = task_id
        self.label = label
        self.retry_after_s = float(retry_after_s)
        self.worker = worker

    def to_dict(self) -> dict:
        """`PendingOut` fields."""
        return {
            "status": "pending",
            "task_id": self.task_id,
            "label": self.label,
            "retry_after_s": self.retry_after_s,
            "worker": jsonable(self.worker),
        }


def error_body(
    code: Code | str,
    message: str,
    *,
    upstream: str | None = None,
    detail: dict | None = None,
    request_id: str = "",
) -> dict:
    """`ErrorBody` fields."""
    return {
        "code": str(code),
        "message": message,
        "upstream": upstream,
        "detail": jsonable(detail),
        "request_id": request_id,
    }


def jsonable(obj: Any) -> Any:
    """`obj` as JSON-able Python: numpy to Python, NaN/inf to None, tuples to lists, Paths and
    other unknown leaves to str."""
    try:
        return to_jsonable(obj)
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    return str(obj)
