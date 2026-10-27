"""The FastAPI app of ksmech-serve (design 6): the error envelope, the wiring of every service
(design 6.4) and the routers of the sub-packages under `/api` (`ROUTERS`: predict, pool,
workbench, embedding, curate, browse, ml; health, meta and whoami are here; the shared plumbing
is `core.http`).

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
auth gate, the same-origin guard, the audit log, the handlers, gzip, the Host allow-list, and
the frontend mount last. Shutdown closes the runner and the scheduler.

The profile (`ServeSettings.profile`) chooses what is served: `lab` everything, `public` no
dataset, curation or ML route (`core.capabilities.PROFILES`). `core.auth.AuthGate` names the
principal of every request (a bearer token from `KSMECH_SERVE_TOKENS`, else the profile's
anonymous principal: a local admin in `lab`, a reader in `public`) and the gated routes check
its role; `core.audit.AuditLog` writes one line per non-GET request to
`data/serve/logs/audit.jsonl`. In `lab` without a tokens file every request is the local
admin, as before: by default the server binds 127.0.0.1 and is reached through an SSH tunnel;
`ksmech-serve --host 0.0.0.0` opens it to the local network (`lan_hosts` then names this
machine's addresses and host names). Two checks keep other web pages out. The Host
allow-list (`LOCAL_HOSTS` plus `lan_hosts`, port ignored) answers 400 to any other Host, which
defeats DNS rebinding, and `SameOriginGuard` answers 403 REQUEST_FORBIDDEN to a POST, PATCH,
PUT or DELETE whose Origin header names another host (or is "null"), which defeats cross-site
form and no-body POSTs. A request without an Origin header (curl, the test client) passes.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

import ksmech
from ksmech.serve.app import static
from ksmech.serve.browse import router as browse_router
from ksmech.serve.core import schemas as S
from ksmech.serve.core.audit import AuditLog
from ksmech.serve.core.auth import Authenticator, AuthGate, principal_of
from ksmech.serve.core.errors import Code, Pending, ServeError, jsonable
from ksmech.serve.core.http import Services, Svc, error_response
from ksmech.serve.core.settings import (
    ServeSettings,
    TaskEnv,
    assert_writable,
    load_settings,
    pin_threads,
)
from ksmech.serve.curate import router as curate_router
from ksmech.serve.embedding import router as embedding_router
from ksmech.serve.ml import router as ml_router
from ksmech.serve.pool import router as pool_router
from ksmech.serve.predict import router as predict_router
from ksmech.serve.workbench import router as workbench_router

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ksmech.serve.chem.chemistry import Chemistry
    from ksmech.serve.compute.worker import Scheduler
    from ksmech.serve.predict.jobs import JobRunner
    from ksmech.serve.predict.predictors import PredictorRegistry
    from ksmech.serve.runtime.context import ApiContext

log = logging.getLogger(__name__)

GZIP_MIN_BYTES = 1024
LOCAL_HOSTS = ("localhost", "127.0.0.1")
TEST_HOST = "testserver"  # the Host of Starlette's TestClient
MUTATING_METHODS = frozenset({"POST", "PATCH", "PUT", "DELETE"})

# ---- error envelope (design 6.3)


def request_id_of(request: Request) -> str:
    return str(request.scope.get("state", {}).get("request_id", ""))


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


def _install_guards(
    app: FastAPI, settings: ServeSettings, authenticator: Authenticator, hosts: Sequence[str]
) -> None:
    """Added before `RequestContext`, so they run inside it (the request id exists), from the
    outside in: the audit log (it sees every refusal below), the same-origin guard, the auth
    gate."""
    app.add_middleware(AuthGate, authenticator=authenticator)
    app.add_middleware(SameOriginGuard, hosts=hosts)
    app.add_middleware(AuditLog, settings=settings)


def install_handlers(app: FastAPI) -> None:
    """The error envelope: ServeError, Pending (202), validation errors, HTTP errors, and the
    catch-all 500 with a request id."""
    app.add_exception_handler(ServeError, _on_serve_error)
    app.add_exception_handler(Pending, _on_pending)
    app.add_exception_handler(RequestValidationError, _on_validation)
    app.add_exception_handler(StarletteHTTPException, _on_http)
    app.add_middleware(RequestContext)


# ---- routes of the app itself: health and meta

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
    return S.MetaOut(
        **body,
        frontend_built=svc.frontend_built,
        dev=svc.settings.dev,
        profile=svc.settings.profile,
    )


@router.get("/whoami", response_model=S.PrincipalOut)
def whoami(request: Request) -> S.PrincipalOut:
    """The principal of this request (core.auth): what a token or its absence gives."""
    return S.PrincipalOut(**principal_of(request).to_out())


# ---- wiring (design 6.4)

ROUTERS = (
    router,
    predict_router.router,
    pool_router.router,
    workbench_router.router,
    embedding_router.router,
    curate_router.router,
    browse_router.router,
    ml_router.router,
    ml_router.lab,
)


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
    `lan_hosts` are the extra Host / Origin names accepted when serving the local network.
    ValueError for an unreadable or malformed tokens file."""
    hosts = (*LOCAL_HOSTS, *lan_hosts)
    pin_threads()
    settings = settings or load_settings()
    authenticator = Authenticator.of(settings)
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
    for r in ROUTERS:
        app.include_router(r)
    _install_guards(app, settings, authenticator, hosts)
    install_handlers(app)
    app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_BYTES)
    allowed = [*hosts, TEST_HOST]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed)  # outermost
    svc.frontend_built = _mount_ui(app, settings, serve_ui)
    log.info(
        "ksmech-serve app: profile %s, chemistry %s, data root %s, frontend %s, tokens %s",
        settings.profile,
        svc.ctx.chemistry,
        settings.data_root,
        "built" if svc.frontend_built else "not served",
        settings.tokens_file or "none",
    )
    return app
