"""The FastAPI app of ksmech-serve (design 6): every route under `/api`, the error envelope, and
the wiring of every service (design 6.4).

Routes are thin. Each one checks its ids against the id table (a mismatch is a 404 with the
endpoint's not-found code, before any service runs, so an id never reaches a file path), calls
exactly one service function, and returns its value: a pydantic model goes out as JSON, a str
as `text/plain` (cubes). `errors.Pending` becomes 202 `PendingOut`, a `ServeError` becomes
`ErrorOut` with its status, FastAPI's validation errors become 422 `REQUEST_INVALID` with the
pydantic errors in `detail`, and anything else is a logged 500 `UNEXPECTED`. Every error body
carries the request id (also sent as the `X-Request-ID` header). JSON never carries NaN.

`create_app` wires, in this order: pin_threads, settings, the scheduler (a one-worker
`ProcessScheduler` unless injected), the `ApiContext`, the chemistry (`MolecularChemistry`
unless injected), the curation database, the job store and runner, `JobService`,
`BrowseService`, `CurateService`, the export / recheck / index job handlers, the routers, the
same-origin guard, the handlers, gzip, the Host allow-list, and the frontend mount last.
Shutdown closes the runner and the scheduler.

There is no authentication: by default the server binds 127.0.0.1 and is reached through an
SSH tunnel; `ksmech-serve --host 0.0.0.0` opens it to the local network (`lan_hosts` then
names this machine's addresses and host names). Two checks keep other web pages out. The Host
allow-list (`LOCAL_HOSTS` plus `lan_hosts`, port ignored) answers 400 to any other Host, which
defeats DNS rebinding, and `SameOriginGuard` answers 403 REQUEST_FORBIDDEN to a POST, PATCH,
PUT or DELETE whose Origin header names another host (or is "null"), which defeats cross-site
form and no-body POSTs. A request without an Origin header (curl, the test client) passes.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

import ksmech
from ksmech.serve.app import static
from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.errors import Code, Pending, ServeError, jsonable
from ksmech.serve.core.settings import (
    ServeSettings,
    TaskEnv,
    assert_writable,
    load_settings,
    pin_threads,
)
from ksmech.serve.pool import service as pool_service
from ksmech.serve.pool import view3d
from ksmech.serve.workbench import molecule
from ksmech.serve.workbench import service as workbench_service

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ksmech.serve.browse.service import BrowseService
    from ksmech.serve.chem.chemistry import Chemistry
    from ksmech.serve.compute.worker import Scheduler
    from ksmech.serve.curate.service import CurateService
    from ksmech.serve.predict.jobs import JobRunner
    from ksmech.serve.predict.predictors import PredictorRegistry
    from ksmech.serve.predict.service import JobService
    from ksmech.serve.runtime.context import ApiContext

log = logging.getLogger(__name__)

GZIP_MIN_BYTES = 1024
LOCAL_HOSTS = ("localhost", "127.0.0.1")
TEST_HOST = "testserver"  # the Host of Starlette's TestClient
MUTATING_METHODS = frozenset({"POST", "PATCH", "PUT", "DELETE"})

# ---- id validation (design 6.1): pattern and the 404 code of a mismatch

ID_RULES: dict[str, tuple[re.Pattern[str], Code]] = {
    "state_id": (re.compile(r"s[0-9a-f]{20}"), Code.POOL_NOT_FOUND),
    "node_id": (re.compile(r"p[0-9a-f]{20}"), Code.NODE_NOT_FOUND),
    "edge_id": (re.compile(r"[ec][0-9a-f]{16}"), Code.EDGE_NOT_FOUND),
    "job_id": (re.compile(r"j[0-9a-f]{16}"), Code.JOB_NOT_FOUND),
    "pathway_id": (re.compile(r"w[0-9a-f]{12}"), Code.PATHWAY_NOT_FOUND),
    "ref": (
        re.compile(r"pmechdb:pmechdb-[0-9a-f]{16}|curated:[0-9]+:[0-9]+"),
        Code.STEP_NOT_FOUND,
    ),
    "which": (re.compile(r"donor|acceptor"), Code.POOL_NOT_FOUND),
    "molecule_id": (re.compile(r"m[0-9a-f]{20}"), Code.MOLECULE_NOT_FOUND),
    "embedding_id": (re.compile(r"b[0-9a-f]{24}"), Code.EMBEDDING_NOT_FOUND),
}


def check_id(kind: str, value: str) -> str:
    """`value` if it matches the `kind` pattern (full match); else ServeError 404 with that
    kind's not-found code. No service or file is touched before this check."""
    pattern, code = ID_RULES[kind]
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        shown = value if isinstance(value, str) and len(value) <= 64 else f"{str(value)[:64]}..."
        raise ServeError(code, f"no such {kind.replace('_', ' ')}: {shown!r}", status=404)
    return value


# ---- the services one app holds


@dataclass
class Services:
    settings: ServeSettings
    ctx: ApiContext
    scheduler: Scheduler
    chemistry: Chemistry
    engine: Engine
    runner: JobRunner
    jobs: JobService
    browse: BrowseService
    curate: CurateService
    frontend_built: bool = False


def _services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(_services)]
StateId = Annotated[str, Path(description="state id (s + 20 hex)")]
JobId = Annotated[str, Path(description="job id (j + 16 hex)")]
TokenId = Annotated[int, Path(ge=0)]
Mid = Annotated[int, Path(ge=0)]
MoleculeId = Annotated[str, Path(description="molecule id (m + 20 hex)")]
EmbeddingId = Annotated[str, Path(description="embedding trace id (b + 24 hex)")]
TRACE_CACHE = "private, max-age=86400"  # a trace id names immutable content


def _blank(value: str | None) -> str | None:
    """Query strings: an empty value means "not given"."""
    return None if value is None or not value.strip() else value


def _cube(text: str, filename: str) -> PlainTextResponse:
    headers = {"Content-Disposition": f"inline; filename={filename}"}
    return PlainTextResponse(text, headers=headers)


# ---- error envelope (design 6.3)


def request_id_of(request: Request) -> str:
    return str(request.scope.get("state", {}).get("request_id", ""))


def error_response(err: ServeError, request_id: str) -> JSONResponse:
    body = err.body(request_id)
    if body["detail"] is not None and not isinstance(body["detail"], dict):
        body["detail"] = {"value": body["detail"]}
    out = S.ErrorOut(error=S.ErrorBody(**body))
    return JSONResponse(out.model_dump(mode="json"), status_code=err.status)


def _validation_message(errors: list[dict]) -> str:
    if not errors:
        return "invalid request"
    first = errors[0]
    where = ".".join(str(p) for p in first.get("loc", ()))
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"{where}: {first.get('msg', 'invalid')}{more}"


async def _on_serve_error(request: Request, exc: ServeError) -> JSONResponse:
    rid = request_id_of(request)
    level = logging.ERROR if exc.status >= 500 else logging.INFO
    log.log(
        level,
        "%s %s -> %d %s: %s [request_id=%s]",
        request.method,
        request.url.path,
        exc.status,
        exc.code,
        exc.message,
        rid,
    )
    return error_response(exc, rid)


async def _on_pending(request: Request, exc: Pending) -> JSONResponse:
    out = S.PendingOut(**exc.to_dict())
    headers = {"Retry-After": str(max(1, round(exc.retry_after_s)))}
    return JSONResponse(out.model_dump(mode="json"), status_code=202, headers=headers)


async def _on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable(list(exc.errors()))
    err = ServeError(
        Code.REQUEST_INVALID, _validation_message(errors), status=422, detail={"errors": errors}
    )
    return error_response(err, request_id_of(request))


async def _on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = str(exc.detail) if exc.detail else "request failed"
    if exc.status_code == 404:
        message = f"no route or file {request.url.path}"
    err = ServeError(Code.REQUEST_INVALID, message, status=exc.status_code)
    response = error_response(err, request_id_of(request))
    for key, value in (getattr(exc, "headers", None) or {}).items():
        response.headers[key] = value
    return response


class RequestContext:
    """Pure ASGI middleware: gives every HTTP request an id (scope state and the X-Request-ID
    header) and turns any exception the handlers did not map into a logged 500 UNEXPECTED."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = rid
        started = False

        async def send_with_id(message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message).append("X-Request-ID", rid)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception as exc:
            log.exception(
                "%s %s failed [UNEXPECTED] [request_id=%s]",
                scope.get("method"),
                scope.get("path"),
                rid,
            )
            if started:
                raise
            err = ServeError(Code.UNEXPECTED, f"{type(exc).__name__}: {exc}", status=500)
            await error_response(err, rid)(scope, receive, send_with_id)


def foreign_origin(origin: str, hosts: Sequence[str] = LOCAL_HOSTS) -> bool:
    """Whether an Origin header names a page outside `hosts` (this machine's loopback names,
    plus its network names when it serves the LAN): any other hostname, an opaque "null"
    origin, or a value that does not parse."""
    if origin.strip().lower() == "null":
        return True
    try:
        host = urlsplit(origin.strip()).hostname
    except ValueError:
        return True
    return host not in hosts


class SameOriginGuard:
    """Pure ASGI middleware: a mutating request (POST, PATCH, PUT, DELETE) whose Origin header
    is foreign gets 403 REQUEST_FORBIDDEN before any route runs. Installed inside
    `RequestContext`, so the refusal carries the request id."""

    def __init__(self, app, hosts: Sequence[str] = LOCAL_HOSTS) -> None:
        self.app = app
        self.hosts = tuple(hosts)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope.get("method") in MUTATING_METHODS:
            origin = Headers(scope=scope).get("origin")
            if origin is not None and foreign_origin(origin, self.hosts):
                rid = str(scope.get("state", {}).get("request_id", ""))
                log.warning(
                    "%s %s from origin %r -> 403 %s [request_id=%s]",
                    scope.get("method"),
                    scope.get("path"),
                    origin[:200],
                    Code.REQUEST_FORBIDDEN,
                    rid,
                )
                detail = {"origin": origin[:200]}
                msg = "cross-origin request refused"
                err = ServeError(Code.REQUEST_FORBIDDEN, msg, status=403, detail=detail)
                await error_response(err, rid)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def install_handlers(app: FastAPI) -> None:
    """The error envelope: ServeError, Pending (202), validation errors, HTTP errors, and the
    catch-all 500 with a request id."""
    app.add_exception_handler(ServeError, _on_serve_error)
    app.add_exception_handler(Pending, _on_pending)
    app.add_exception_handler(RequestValidationError, _on_validation)
    app.add_exception_handler(StarletteHTTPException, _on_http)
    app.add_middleware(RequestContext)


# ---- routes: health, meta, predictors, input

router = APIRouter(prefix="/api")


@router.get("/health", response_model=S.HealthOut)
async def health(svc: Svc) -> S.HealthOut:
    worker = svc.scheduler.status().to_dict()
    return S.HealthOut(
        ok=bool(worker["alive"]) and not bool(worker["unstable"]),
        chemistry=svc.ctx.chemistry,
        worker=worker,
        jobs_running=svc.jobs.running(),
    )


@router.get("/meta", response_model=S.MetaOut)
def meta(svc: Svc) -> S.MetaOut:
    from ksmech.serve.runtime.context import meta as context_meta

    body = context_meta(svc.ctx)
    return S.MetaOut(**body, frontend_built=svc.frontend_built, dev=svc.settings.dev)


@router.get(
    "/predictors",
    response_model=list[S.PredictorInfoOut],
    dependencies=[require("predict")],
)
def predictors(svc: Svc) -> list[S.PredictorInfoOut]:
    spec = svc.ctx.spec_version
    return [
        S.PredictorInfoOut(**info.to_dict(), spec_version=spec) for info in svc.ctx.registry.info()
    ]


@router.post("/normalize", response_model=S.NormalizeOut)
def normalize(svc: Svc, body: S.NormalizeIn) -> S.NormalizeOut:
    return pool_service.normalize(svc.ctx, body)


@router.post(
    "/predict",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("predict")],
)
def predict(svc: Svc, body: S.PredictIn) -> S.JobCreatedOut:
    return svc.jobs.predict(body)


# ---- jobs


@router.get("/jobs", response_model=list[S.JobSummaryOut])
def list_jobs(
    svc: Svc,
    kind: str | None = None,
    status: str | None = None,
    limit: Annotated[int, Query(ge=0, le=1000)] = 50,
) -> list[S.JobSummaryOut]:
    return svc.jobs.list(kind=_blank(kind), status=_blank(status), limit=limit)


@router.get("/jobs/{job_id}", response_model=S.JobOut)
def get_job(
    svc: Svc, job_id: JobId, since_revision: Annotated[int | None, Query(ge=0)] = None
) -> S.JobOut:
    return svc.jobs.get(check_id("job_id", job_id), since_revision=since_revision)


@router.post("/jobs/{job_id}/cancel", response_model=S.JobOut)
def cancel_job(svc: Svc, job_id: JobId, force: bool = False) -> S.JobOut:
    return svc.jobs.cancel(check_id("job_id", job_id), force=force)


@router.post(
    "/jobs/{job_id}/override",
    response_model=S.OverrideResultOut,
    dependencies=[require("correct")],
)
def override_job(svc: Svc, job_id: JobId, body: S.OverrideIn) -> S.OverrideResultOut:
    """A user step R on one state of the job's network (synchronous); 422 STEP_INVALID with
    the issues, 202 Pending while the state's analysis is computed."""
    check_id("job_id", job_id)
    check_id("state_id", body.parent_state_id)
    return svc.jobs.override(job_id, body)


@router.post(
    "/jobs/{job_id}/expand",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("predict")],
)
def expand_job(svc: Svc, job_id: JobId, body: S.ExpandIn) -> S.JobCreatedOut:
    check_id("job_id", job_id)
    check_id("node_id", body.node_id)
    return svc.jobs.expand(job_id, body)


@router.post("/step/check", response_model=S.StepCheckOut, dependencies=[require("correct")])
def step_check(svc: Svc, body: S.StepCheckIn) -> S.StepCheckOut:
    """Live validation of a reaction matrix on a pool state (no job needed)."""
    from ksmech.serve.science.steps import step_check as check

    return S.StepCheckOut(**check(svc.ctx, check_id("state_id", body.state_id), body.matrix))


@router.get("/jobs/{job_id}/pathways", response_model=list[S.PathwayOut])
def job_pathways(
    svc: Svc,
    job_id: JobId,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    include_pruned: bool = False,
) -> list[S.PathwayOut]:
    check_id("job_id", job_id)
    return svc.jobs.pathways(job_id, limit=limit, include_pruned=include_pruned)


# ---- pools and tokens


@router.post("/pools", response_model=S.PoolOut)
def create_pool(svc: Svc, body: S.NormalizeIn) -> S.PoolOut:
    return pool_service.create_pool(svc.ctx, body)


@router.get("/pool/{state_id}", response_model=S.PoolOut)
def get_pool(svc: Svc, state_id: StateId) -> S.PoolOut:
    return pool_service.pool_out(svc.ctx, check_id("state_id", state_id))


@router.get("/pool/{state_id}/tokens", response_model=S.TokensOut)
def pool_tokens(svc: Svc, state_id: StateId) -> S.TokensOut:
    return pool_service.tokens_out(svc.ctx, check_id("state_id", state_id))


# ---- geometry, cubes, depiction


def _keep_maps(raw: str | None) -> set[int]:
    """`keep_maps=1,7` -> {1, 7}; anything but non-negative integers is REQUEST_INVALID."""
    if raw is None or not raw.strip():
        return set()
    try:
        maps = {int(x) for x in raw.split(",") if x.strip()}
    except ValueError as exc:
        msg = f"keep_maps must be comma-separated integers, got {raw!r}"
        raise ServeError(Code.REQUEST_INVALID, msg, status=422) from exc
    if any(m < 0 for m in maps):
        raise ServeError(Code.REQUEST_INVALID, f"keep_maps must be >= 0, got {raw!r}", status=422)
    return maps


@router.get("/pool/{state_id}/geometry", response_model=S.GeometryOut)
def pool_geometry(svc: Svc, state_id: StateId) -> S.GeometryOut:
    return view3d.geometry_out(svc.ctx, check_id("state_id", state_id))


@router.get("/pool/{state_id}/orbital/{token_id}", response_class=PlainTextResponse)
def pool_orbital_cube(
    svc: Svc, state_id: StateId, token_id: TokenId, spacing: float = 0.2, padding: float = 4.0
) -> PlainTextResponse:
    sid = check_id("state_id", state_id)
    text = view3d.orbital_cube_text(svc.ctx, sid, token_id, spacing=spacing, padding=padding)
    return _cube(text, f"{sid}-{token_id}.cube")


@router.get("/pool/{state_id}/depiction", response_model=S.DepictionOut)
def pool_depiction(
    svc: Svc, state_id: StateId, keep_maps: str | None = None, server_svg: bool = False
) -> S.DepictionOut:
    sid = check_id("state_id", state_id)
    maps = _keep_maps(keep_maps)
    return view3d.depiction_out(svc.ctx, sid, keep_maps=maps, server_svg=server_svg)


# ---- molecule workbench (display only)

_WB = [require("workbench")]


@router.post("/molecule", response_model=S.MoleculeOut, dependencies=_WB)
def create_molecule(svc: Svc, body: S.MoleculeIn) -> S.MoleculeOut:
    return molecule.create_molecule(svc.ctx, body)


@router.get("/molecule/{molecule_id}", response_model=S.MoleculeOut, dependencies=_WB)
def get_molecule(svc: Svc, molecule_id: MoleculeId) -> S.MoleculeOut:
    return molecule.molecule_out(svc.ctx, check_id("molecule_id", molecule_id))


@router.get(
    "/molecule/{molecule_id}/geometry", response_model=S.MoleculeGeometryOut, dependencies=_WB
)
def molecule_geometry(svc: Svc, molecule_id: MoleculeId) -> S.MoleculeGeometryOut:
    return workbench_service.geometry_out(svc.ctx, check_id("molecule_id", molecule_id))


@router.get(
    "/molecule/{molecule_id}/properties",
    response_model=S.MoleculePropertiesOut,
    dependencies=_WB,
)
def molecule_properties(svc: Svc, molecule_id: MoleculeId) -> S.MoleculePropertiesOut:
    return workbench_service.properties_out(svc.ctx, check_id("molecule_id", molecule_id))


@router.get("/molecule/{molecule_id}/symmetry", response_model=S.PointGroupOut, dependencies=_WB)
def molecule_symmetry(
    svc: Svc,
    molecule_id: MoleculeId,
    tolerance: Annotated[float | None, Query(ge=0.01, le=0.5)] = None,
) -> S.PointGroupOut:
    mid = check_id("molecule_id", molecule_id)
    return workbench_service.symmetry_out(svc.ctx, mid, tolerance)


@router.get("/molecule/{molecule_id}/orbitals", response_model=S.OrbitalSetOut, dependencies=_WB)
def molecule_orbitals(
    svc: Svc,
    molecule_id: MoleculeId,
    set_: Annotated[str | None, Query(alias="set")] = None,
    start: str | None = None,
) -> S.OrbitalSetOut:
    mid = check_id("molecule_id", molecule_id)
    return workbench_service.orbitals_out(svc.ctx, mid, _blank(set_), _blank(start))


@router.get("/molecule/{molecule_id}/cube", response_class=PlainTextResponse, dependencies=_WB)
def molecule_cube(
    svc: Svc,
    molecule_id: MoleculeId,
    kind: str,
    set_: Annotated[str | None, Query(alias="set")] = None,
    start: str | None = None,
    index: Annotated[int | None, Query(ge=0)] = None,
    quality: str | None = None,
    spacing: float | None = None,
    padding: float | None = None,
    download: bool = False,
) -> PlainTextResponse:
    mid = check_id("molecule_id", molecule_id)
    cube = workbench_service.cube_out(
        svc.ctx,
        mid,
        kind,
        set_=_blank(set_),
        start=_blank(start),
        index=index,
        quality=_blank(quality),
        spacing=spacing,
        padding=padding,
    )
    headers = workbench_service.cube_headers(cube, download)
    return PlainTextResponse(cube.text, headers=headers)


@router.post("/depict2d", response_model=S.Depict2DOut, dependencies=_WB)
def depict2d(svc: Svc, body: S.Depict2DIn) -> S.Depict2DOut:
    if body.molecule_id is not None:
        check_id("molecule_id", body.molecule_id)
    return workbench_service.depict2d_out(svc.ctx, body)


# ---- local checkpoints and embedding traces


@router.get("/models", response_model=S.ModelsOut)
def list_models(svc: Svc) -> S.ModelsOut:
    """The local checkpoints and the embedding capability they give (never gated)."""
    from ksmech.serve.core import capabilities
    from ksmech.serve.predict.models import list_checkpoints, model_roots

    on, reason = capabilities.status("embedding", svc.settings)
    return S.ModelsOut(
        roots=[str(r) for r in model_roots(svc.settings)],
        checkpoints=[S.CheckpointOut(**c.to_out()) for c in list_checkpoints(svc.settings)],
        embedding=S.CapabilityOut(enabled=on, reason=reason),
    )


@router.post(
    "/embedding",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("embedding")],
)
def create_embedding(svc: Svc, body: S.EmbeddingIn) -> S.JobCreatedOut:
    """Trace a pool state (or a SMILES) through a local checkpoint: an `embedding` job whose
    result is `EmbeddingResultOut`."""
    from ksmech.serve.embedding.service import submit

    if body.state_id is not None:
        check_id("state_id", body.state_id)
    return submit(svc.ctx, svc.runner, body)


@router.get("/embedding/{embedding_id}", response_class=Response)
def get_embedding(svc: Svc, embedding_id: EmbeddingId) -> Response:
    """The stored payload of a trace (what the page draws), as JSON."""
    from ksmech.serve.embedding.service import payload_bytes

    data = payload_bytes(svc.settings, check_id("embedding_id", embedding_id))
    headers = {"Cache-Control": TRACE_CACHE}
    return Response(data, media_type="application/json", headers=headers)


@router.get("/embedding/{embedding_id}/page", response_class=HTMLResponse)
def get_embedding_page(
    svc: Svc, embedding_id: EmbeddingId, theme: str | None = None
) -> HTMLResponse:
    """The self-contained trace page (no external script), for an iframe or a tab; `theme`
    light | dark sets its colour scheme, else it follows the browser's."""
    from ksmech.serve.embedding.service import page_html

    html = page_html(svc.settings, check_id("embedding_id", embedding_id), _blank(theme))
    return HTMLResponse(html, headers={"Cache-Control": TRACE_CACHE})


# ---- curation (static routes before /curate/{mid})


@router.post(
    "/curate",
    response_model=S.MechanismOut,
    status_code=201,
    dependencies=[require("curate_edit")],
)
def curate_create(svc: Svc, body: S.CurateIn) -> S.MechanismOut:
    check_id("job_id", body.job_id)
    if body.pathway_id is not None:
        check_id("pathway_id", body.pathway_id)
    for edge in body.edge_ids or ():
        check_id("edge_id", edge)
    return svc.curate.create(body)


@router.get("/curate", response_model=list[S.MechanismSummaryOut])
def curate_list(
    svc: Svc,
    status: str | None = None,
    predictor: str | None = None,
    stale: bool | None = None,
    limit: Annotated[int, Query(ge=0, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[S.MechanismSummaryOut]:
    return svc.curate.list(
        status=_blank(status), predictor=_blank(predictor), stale=stale, limit=limit, offset=offset
    )


@router.get(
    "/curate/export/pmechdb", response_class=PlainTextResponse, dependencies=[require("export")]
)
def curate_export_pmechdb(
    svc: Svc, status: Annotated[list[str] | None, Query()] = None
) -> PlainTextResponse:
    """The curated steps as PMechDB raw records (a file download, never training data); the
    report (counts, skipped steps by reason) is in the X-Ksmech-Export-Report header."""
    import json

    text, report = svc.curate.export_pmechdb(status)
    headers = {
        "Content-Disposition": 'attachment; filename="ksmech_curated_pmechdb.csv"',
        "X-Ksmech-Export-Report": json.dumps(report, sort_keys=True),
    }
    return PlainTextResponse(text, media_type="text/csv", headers=headers)


@router.post(
    "/curate/recheck",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("curate_recheck")],
)
def curate_recheck(svc: Svc, body: Annotated[S.RecheckIn | None, Body()] = None) -> S.JobCreatedOut:
    return svc.curate.recheck(body or S.RecheckIn())


@router.get("/curate/{mid}", response_model=S.MechanismOut)
def curate_get(svc: Svc, mid: Mid) -> S.MechanismOut:
    return svc.curate.get(mid)


@router.patch("/curate/{mid}", response_model=S.MechanismOut, dependencies=[require("curate_edit")])
def curate_patch(svc: Svc, mid: Mid, body: S.MechanismPatchIn) -> S.MechanismOut:
    return svc.curate.patch(mid, body)


@router.post(
    "/curate/{mid}/recheck",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("curate_recheck")],
)
def curate_recheck_one(
    svc: Svc, mid: Mid, body: Annotated[S.RecheckOneIn | None, Body()] = None
) -> S.JobCreatedOut:
    return svc.curate.recheck_one(mid, body or S.RecheckOneIn())


# ---- browse


@router.get("/browse/status", response_model=S.IndexStatusOut)
def browse_status(svc: Svc) -> S.IndexStatusOut:
    return svc.browse.status()


@router.post("/browse/index", response_model=S.JobCreatedOut, status_code=202)
def browse_index(svc: Svc) -> S.JobCreatedOut:
    return svc.browse.build_index()


@router.get("/browse/search", response_model=S.SearchOut)
def browse_search(
    svc: Svc,
    q: str = "",
    mode: str = "exact",
    side: str = "either",
    source: Annotated[list[str] | None, Query()] = None,
    status: str | None = None,
    resonance: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> S.SearchOut:
    sources = tuple(s for s in (source or ()) if s.strip())
    kwargs: dict[str, Any] = {"sources": sources} if sources else {}
    return svc.browse.search(
        q=q,
        mode=mode,
        side=side,
        status=_blank(status),
        resonance=resonance,
        limit=limit,
        offset=offset,
        **kwargs,
    )


@router.get("/browse/step/{ref}", response_model=S.StepViewOut)
def browse_step(svc: Svc, ref: str) -> S.StepViewOut:
    return svc.browse.step(check_id("ref", ref))


# ---- wiring (design 6.4)


def _job_status(runner: JobRunner) -> Callable[[str], str | None]:
    def status(job_id: str) -> str | None:
        try:
            return str(runner.get(job_id).status)
        except ServeError:
            return None

    return status


def _submitter(runner: JobRunner) -> Callable[[str, dict], dict]:
    def submit(kind: str, request: dict) -> dict:
        return runner.submit(kind, request).created_out()

    return submit


def _register_handlers(runner: JobRunner, engine: Engine, browse, ctx: ApiContext) -> None:
    """recheck on the batch thread, index on the background thread, embedding traces on the
    trace thread."""
    from ksmech.serve.browse.pmindex import run_index
    from ksmech.serve.curate.recheck import run_recheck
    from ksmech.serve.predict.jobs import JobKind

    runner.register(JobKind.RECHECK, partial(run_recheck, engine=engine, registry=ctx.registry))
    from ksmech.serve.embedding.service import embedding_handler

    runner.register(JobKind.INDEX, partial(run_index, engine=engine), thread="background")
    runner.register(JobKind.EMBEDDING, embedding_handler(ctx), thread="trace")


def _curation_engine(settings: ServeSettings) -> Engine:
    from ksmech.serve.core import db

    assert_writable(settings.curation_db, settings)
    engine = db.make_engine(settings.curation_db)
    db.init_db(engine)
    return engine


def _services_of(
    settings: ServeSettings,
    scheduler: Scheduler | None,
    chemistry: Chemistry | None,
    registry: PredictorRegistry | None,
    jobs_inline: bool,
) -> Services:
    from ksmech.serve.browse.service import BrowseService
    from ksmech.serve.chem.chemistry import MolecularChemistry
    from ksmech.serve.curate.service import CurateService
    from ksmech.serve.predict.jobs import JobRunner, JobStore
    from ksmech.serve.predict.service import JobService
    from ksmech.serve.runtime.context import build_context

    if scheduler is None:
        from ksmech.serve.compute.procsched import ProcessScheduler

        scheduler = ProcessScheduler(TaskEnv(settings), settings.workers)
    ctx = build_context(settings, scheduler=scheduler, registry=registry)
    chem = chemistry or MolecularChemistry(settings, ctx.scheduler, ctx.featurizer)
    if getattr(chem, "mode", None) == "stub":
        ctx.chemistry = "stub"
    engine = _curation_engine(settings)
    store = JobStore(settings.jobs, settings)
    runner = JobRunner(store, settings, ctx.scheduler, inline=jobs_inline)
    jobs = JobService(ctx, runner, chem)
    submit = _submitter(runner)
    browse = BrowseService(ctx, engine, submit=submit, job_status=_job_status(runner))
    curate = CurateService(ctx, engine, networks=jobs.network_of, submit=submit)
    _register_handlers(runner, engine, browse, ctx)
    return Services(settings, ctx, ctx.scheduler, chem, engine, runner, jobs, browse, curate)


def _start_index(svc: Services) -> None:
    """auto_index: submit one index job at startup (never by default)."""
    try:
        created = svc.browse.build_index()
        log.info("auto_index: index job %s (%s)", created.job_id, created.status)
    except ServeError as exc:
        log.warning("auto_index: no index job [%s] %s", exc.code, exc.message)
        from ksmech.serve.core.settings import append_failure

        append_failure(svc.settings, str(exc.code), exc.message, where="auto_index")


def _shutdown(svc: Services) -> None:
    for name, close in (("runner", svc.runner.close), ("scheduler", svc.scheduler.close)):
        try:
            close()
        except Exception:  # noqa: BLE001 - shutdown must close both
            log.exception("closing the %s failed", name)
    svc.engine.dispose()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    svc: Services = app.state.services
    if svc.settings.auto_index:
        _start_index(svc)
    try:
        yield
    finally:
        _shutdown(svc)


def _mount_ui(app: FastAPI, settings: ServeSettings, serve_ui: bool) -> bool:
    if settings.dev:
        static.mount_dev_banner(app)
        return False
    if not serve_ui:
        static.mount_placeholder(app, static.NO_UI, "no-ui")
        return False
    return static.mount_frontend(app, settings.dist)


def create_app(
    settings: ServeSettings | None = None,
    *,
    scheduler: Scheduler | None = None,
    chemistry: Chemistry | None = None,
    registry: PredictorRegistry | None = None,
    jobs_inline: bool = False,
    serve_ui: bool = True,
    lan_hosts: Sequence[str] = (),
) -> FastAPI:
    """The app (module docstring). `serve_ui=False` (`--no-ui`) serves the API only;
    `lan_hosts` are the extra Host / Origin names accepted when serving the local network."""
    hosts = (*LOCAL_HOSTS, *lan_hosts)
    pin_threads()
    settings = settings or load_settings()
    svc = _services_of(settings, scheduler, chemistry, registry, jobs_inline)
    app = FastAPI(
        title="ksmech-serve",
        version=str(ksmech.__version__),
        lifespan=_lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.services = svc
    app.include_router(router)
    app.add_middleware(SameOriginGuard, hosts=hosts)  # added before RequestContext: runs inside
    install_handlers(app)
    app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_BYTES)
    allowed = [*hosts, TEST_HOST]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed)  # outermost
    svc.frontend_built = _mount_ui(app, settings, serve_ui)
    log.info(
        "ksmech-serve app: chemistry %s, data root %s, frontend %s",
        svc.ctx.chemistry,
        settings.data_root,
        "built" if svc.frontend_built else "not served",
    )
    return app
