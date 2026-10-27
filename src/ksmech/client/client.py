"""`KsmechClient`: a typed HTTP client of the ksmech-serve API (`docs/serve_v9_contract.md`).

The HTTP API is the single contract: the web UI, this client, the `ksmech` CLI and the MCP
server (`ksmech-mcp`) all call the same routes, so permissions, quotas and the audit log apply
equally. Every request carries `X-Ksmech-Via` (who acts: `cli`, `mcp`, ...) and, with a token,
`Authorization: Bearer <token>`.

One method per API operation; answers are parsed with the pydantic models of
`ksmech.serve.core.schemas` where the route has one, else returned as JSON (`dict` / `list`).
A 202 `PendingOut` (a worker task still runs) is retried after its `retry_after_s` until
`pending_timeout` runs out (then `KsmechPending`); `wait_job` polls a job until it finishes.

This module imports only httpx, pydantic and `ksmech.serve.core.schemas` (no FastAPI, no
RDKit, no torch): `tests/test_client_imports.py` checks it in a fresh interpreter.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

import ksmech
from ksmech.client.errors import (
    CONNECTION_FAILED,
    RESPONSE_INVALID,
    KsmechAPIError,
    KsmechPending,
    KsmechTimeout,
)
from ksmech.serve.core import schemas as S

DEFAULT_URL = "http://127.0.0.1:8000"
VIA_HEADER = "X-Ksmech-Via"
FINISHED = frozenset({"done", "failed", "cancelled"})
M = TypeVar("M", bound=BaseModel)


def _params(**kwargs: Any) -> dict[str, Any]:
    """Query parameters without the ones not given (None); booleans as true / false."""
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        out[key] = ("true" if value else "false") if isinstance(value, bool) else value
    return out


def _body(**kwargs: Any) -> dict[str, Any]:
    """A JSON body without the optional fields not given (the server's defaults apply)."""
    return {k: v for k, v in kwargs.items() if v is not None}


def parse(model: type[M], data: Any) -> M:
    """`model.model_validate(data)`; a mismatch is `KsmechAPIError(RESPONSE_INVALID)`."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        msg = f"the answer is not a {model.__name__}: {exc.errors()[:1]}"
        raise KsmechAPIError(RESPONSE_INVALID, msg) from exc


def parse_list(model: type[M], data: Any) -> list[M]:
    if not isinstance(data, list):
        raise KsmechAPIError(RESPONSE_INVALID, f"expected a list of {model.__name__}")
    return [parse(model, item) for item in data]


def error_of(response: httpx.Response) -> KsmechAPIError:
    """The `KsmechAPIError` of an error answer: the API's envelope, else the raw text."""
    status = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = None
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and "code" in err:
        return KsmechAPIError(
            str(err["code"]),
            str(err.get("message", "")),
            status=status,
            detail=err.get("detail"),
            request_id=err.get("request_id") or response.headers.get("x-request-id"),
            upstream=err.get("upstream"),
        )
    text = response.text.strip()[:300] or response.reason_phrase
    return KsmechAPIError(f"HTTP_{status}", text, status=status)


def pending_of(response: httpx.Response) -> S.PendingOut | None:
    """The `PendingOut` of a 202 answer (a worker task still runs), else None (a 202
    `JobCreatedOut` is not pending: the job exists)."""
    if response.status_code != 202:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and body.get("status") == "pending" and "task_id" in body:
        return parse(S.PendingOut, body)
    return None


class KsmechClient:
    """The ksmech-serve API over HTTP.

    `url` is the server root (default `http://127.0.0.1:8000`), `token` a bearer token, `via`
    the `X-Ksmech-Via` value (`cli`, `mcp`). `http` is an `httpx.Client` to send through
    instead of a new one (a FastAPI `TestClient` in tests; its own base URL is used).
    `pending_timeout` bounds the retries of a 202 Pending answer, `timeout` one request.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        via: str = "cli",
        *,
        timeout: float = 60.0,
        pending_timeout: float = 120.0,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.token = token or None
        self.via = via
        self.pending_timeout = float(pending_timeout)
        self._sleep = sleep
        self._owned = http is None
        self._http = http or httpx.Client(base_url=self.url, timeout=timeout)

    @classmethod
    def from_env(cls, via: str = "cli", **kwargs: Any) -> KsmechClient:
        """A client on `KSMECH_URL` (default http://127.0.0.1:8000) with `KSMECH_TOKEN`."""
        kwargs.setdefault("url", os.environ.get("KSMECH_URL") or DEFAULT_URL)
        kwargs.setdefault("token", os.environ.get("KSMECH_TOKEN") or None)
        return cls(via=via, **kwargs)

    def close(self) -> None:
        if self._owned:
            self._http.close()

    def __enter__(self) -> KsmechClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- transport

    def headers(self) -> dict[str, str]:
        out = {
            VIA_HEADER: self.via,
            "User-Agent": f"ksmech-client/{ksmech.__version__} ({self.via})",
            "Accept": "application/json",
        }
        if self.token:
            out["Authorization"] = f"Bearer {self.token}"
        return out

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, headers=self.headers(), **kwargs)
        except httpx.HTTPError as exc:
            msg = f"{method} {self.url}{path}: {type(exc).__name__}: {exc}"
            raise KsmechAPIError(CONNECTION_FAILED, msg) from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        wait_pending: float | None = None,
    ) -> httpx.Response:
        """One API call: a 202 Pending answer is retried after its `retry_after_s` until
        `wait_pending` (default `pending_timeout`) seconds have passed, then `KsmechPending`;
        an error answer raises `KsmechAPIError`."""
        budget = self.pending_timeout if wait_pending is None else float(wait_pending)
        deadline = time.monotonic() + budget
        while True:
            response = self._send(method, path, params=params, json=json)
            pending = pending_of(response)
            if pending is None:
                if response.is_error:
                    raise error_of(response)
                return response
            delay = max(0.2, min(float(pending.retry_after_s), 5.0))
            if time.monotonic() + delay > deadline:
                raise KsmechPending(pending)
            self._sleep(delay)

    def get_json(self, path: str, **params: Any) -> Any:
        """GET `path` as JSON (query parameters that are None are left out)."""
        return self._json(self.request("GET", path, params=_params(**params)))

    def post_json(self, path: str, body: Any = None, **params: Any) -> Any:
        return self._json(self.request("POST", path, params=_params(**params), json=body))

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            msg = f"{response.request.method} {response.request.url.path}: not JSON"
            raise KsmechAPIError(RESPONSE_INVALID, msg, status=response.status_code) from exc

    # ---- health and meta

    def health(self) -> S.HealthOut:
        return parse(S.HealthOut, self.get_json("/api/health"))

    def meta(self) -> S.MetaOut:
        return parse(S.MetaOut, self.get_json("/api/meta"))

    def whoami(self) -> S.PrincipalOut:
        """The principal the server sees for this client's token and `X-Ksmech-Via`."""
        return parse(S.PrincipalOut, self.get_json("/api/whoami"))

    def capabilities(self) -> dict[str, bool]:
        """{capability: enabled} of `/api/meta` (raw, so names a newer server adds are kept)."""
        raw = self.get_json("/api/meta").get("capabilities") or {}
        return {str(k): bool((v or {}).get("enabled")) for k, v in raw.items()}

    # ---- prediction

    def predictors(self) -> list[S.PredictorInfoOut]:
        return parse_list(S.PredictorInfoOut, self.get_json("/api/predictors"))

    def submit_predict(self, request: S.PredictIn | Mapping[str, Any]) -> S.JobCreatedOut:
        """POST /api/predict: the job (202), not its result (`wait_job`, or `predict`)."""
        body = _request_body(S.PredictIn, request)
        return parse(S.JobCreatedOut, self.post_json("/api/predict", body))

    def predict(
        self,
        reactants: Iterable[str],
        *,
        timeout: float = 600.0,
        poll: float = 0.5,
        **fields: Any,
    ) -> S.JobOut:
        """Submit a prediction (`PredictIn` fields: predictor, beam, max_steps, checkpoint,
        ...) and wait for the job; returns the finished `JobOut` (done, failed or cancelled),
        `KsmechTimeout` after `timeout` seconds."""
        body = {"reactants_smiles": list(reactants), **_body(**fields)}
        created = self.submit_predict(body)
        return self.wait_job(created.job_id, timeout=timeout, poll=poll)

    def wait_job(self, job_id: str, *, timeout: float = 600.0, poll: float = 0.5) -> S.JobOut:
        """Poll GET /api/jobs/{id} (the interval grows to 2 s) until the job finishes."""
        deadline = time.monotonic() + float(timeout)
        delay = max(0.05, float(poll))
        while True:
            job = self.job(job_id)
            if job.status in FINISHED:
                return job
            if time.monotonic() + delay > deadline:
                raise KsmechTimeout(f"job {job_id} still {job.status} after {timeout:g} s", job)
            self._sleep(delay)
            delay = min(2.0, delay * 1.5)

    def jobs(
        self, *, kind: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[S.JobSummaryOut]:
        data = self.get_json("/api/jobs", kind=kind, status=status, limit=limit)
        return parse_list(S.JobSummaryOut, data)

    def job(self, job_id: str, *, since_revision: int | None = None) -> S.JobOut:
        data = self.get_json(f"/api/jobs/{job_id}", since_revision=since_revision)
        return parse(S.JobOut, data)

    def cancel_job(self, job_id: str, *, force: bool = False) -> S.JobOut:
        return parse(S.JobOut, self.post_json(f"/api/jobs/{job_id}/cancel", force=force))

    def pathways(
        self, job_id: str, *, limit: int = 20, include_pruned: bool = False
    ) -> list[S.PathwayOut]:
        path = f"/api/jobs/{job_id}/pathways"
        data = self.get_json(path, limit=limit, include_pruned=include_pruned)
        return parse_list(S.PathwayOut, data)

    def expand(self, job_id: str, node_id: str, **fields: Any) -> S.JobCreatedOut:
        """POST /api/jobs/{id}/expand (`ExpandIn`: steps, predictor, beam, checkpoint)."""
        body = _request_body(S.ExpandIn, {"node_id": node_id, **_body(**fields)})
        return parse(S.JobCreatedOut, self.post_json(f"/api/jobs/{job_id}/expand", body))

    def override(
        self, job_id: str, parent_state_id: str, matrix: S.MatrixIO | Mapping[str, Any]
    ) -> S.OverrideResultOut:
        """A user step R on one state of the job's network (synchronous)."""
        body = _request_body(S.OverrideIn, {"parent_state_id": parent_state_id, "matrix": matrix})
        data = self.post_json(f"/api/jobs/{job_id}/override", body)
        return parse(S.OverrideResultOut, data)

    def step_check(self, state_id: str, matrix: S.MatrixIO | Mapping[str, Any]) -> S.StepCheckOut:
        """POST /api/step/check: validate a reaction matrix on a pool state (no job)."""
        body = _request_body(S.StepCheckIn, {"state_id": state_id, "matrix": matrix})
        return parse(S.StepCheckOut, self.post_json("/api/step/check", body))

    # ---- pools

    def normalize(
        self, reactants: Iterable[str], stoichiometry: list[int] | None = None
    ) -> S.NormalizeOut:
        body = _body(reactants_smiles=list(reactants), stoichiometry=stoichiometry)
        return parse(S.NormalizeOut, self.post_json("/api/normalize", body))

    def create_pool(
        self, reactants: Iterable[str], stoichiometry: list[int] | None = None
    ) -> S.PoolOut:
        """POST /api/pools: build and save a pool state (no job)."""
        body = _body(reactants_smiles=list(reactants), stoichiometry=stoichiometry)
        return parse(S.PoolOut, self.post_json("/api/pools", body))

    def pool(self, state_id: str) -> S.PoolOut:
        return parse(S.PoolOut, self.get_json(f"/api/pool/{state_id}"))

    def pool_tokens(self, state_id: str) -> S.TokensOut:
        return parse(S.TokensOut, self.get_json(f"/api/pool/{state_id}/tokens"))

    def pool_geometry(self, state_id: str) -> S.GeometryOut:
        return parse(S.GeometryOut, self.get_json(f"/api/pool/{state_id}/geometry"))

    def pool_depiction(
        self, state_id: str, *, keep_maps: Iterable[int] = (), server_svg: bool = False
    ) -> S.DepictionOut:
        maps = ",".join(str(int(m)) for m in keep_maps) or None
        path = f"/api/pool/{state_id}/depiction"
        data = self.get_json(path, keep_maps=maps, server_svg=server_svg or None)
        return parse(S.DepictionOut, data)

    def pool_orbital_cube(self, state_id: str, token_id: int, **params: Any) -> str:
        """The Gaussian cube text of one orbital token (spacing, padding)."""
        path = f"/api/pool/{state_id}/orbital/{int(token_id)}"
        return self.request("GET", path, params=_params(**params)).text

    # ---- molecule workbench

    def molecule_create(
        self, *, smiles: str | None = None, molblock: str | None = None, retry: bool = False
    ) -> S.MoleculeOut:
        """POST /api/molecule (exactly one of smiles / molblock): featurization starts."""
        body = _body(smiles=smiles, molblock=molblock, retry=retry or None)
        return parse(S.MoleculeOut, self.post_json("/api/molecule", body))

    def molecule(self, molecule_id: str) -> S.MoleculeOut:
        return parse(S.MoleculeOut, self.get_json(f"/api/molecule/{molecule_id}"))

    def molecule_ready(
        self, *, smiles: str | None = None, molblock: str | None = None, timeout: float = 300.0
    ) -> S.MoleculeOut:
        """Create the molecule and poll it until its featurization ends (ready, qm_only or
        failed); `KsmechTimeout` after `timeout` seconds."""
        mol = self.molecule_create(smiles=smiles, molblock=molblock)
        deadline = time.monotonic() + float(timeout)
        delay = 0.25
        while mol.status in ("missing", "running"):
            if time.monotonic() + delay > deadline:
                raise KsmechTimeout(f"molecule {mol.molecule_id} still {mol.status}", mol)
            self._sleep(delay)
            delay = min(2.0, delay * 1.5)
            mol = self.molecule(mol.molecule_id)
        return mol

    def molecule_geometry(self, molecule_id: str) -> S.MoleculeGeometryOut:
        path = f"/api/molecule/{molecule_id}/geometry"
        return parse(S.MoleculeGeometryOut, self.get_json(path))

    def molecule_properties(self, molecule_id: str) -> S.MoleculePropertiesOut:
        path = f"/api/molecule/{molecule_id}/properties"
        return parse(S.MoleculePropertiesOut, self.get_json(path))

    def molecule_symmetry(
        self, molecule_id: str, *, tolerance: float | None = None
    ) -> S.PointGroupOut:
        path = f"/api/molecule/{molecule_id}/symmetry"
        return parse(S.PointGroupOut, self.get_json(path, tolerance=tolerance))

    def molecule_orbitals(
        self, molecule_id: str, *, orbital_set: str | None = None, start: str | None = None
    ) -> S.OrbitalSetOut:
        path = f"/api/molecule/{molecule_id}/orbitals"
        return parse(S.OrbitalSetOut, self.get_json(path, set=orbital_set, start=start))

    def depict2d(self, molblock: str, **fields: Any) -> S.Depict2DOut:
        """POST /api/depict2d (`Depict2DIn`: mode, style, molecule_id, view)."""
        body = _request_body(S.Depict2DIn, {"molblock": molblock, **_body(**fields)})
        return parse(S.Depict2DOut, self.post_json("/api/depict2d", body))

    # ---- checkpoints, training runs and reports

    def models(self) -> S.ModelsOut:
        """The local checkpoints (`GET /api/models`)."""
        return parse(S.ModelsOut, self.get_json("/api/models"))

    def ml_runs(self) -> S.MlRunsOut:
        """The training runs (`GET /api/ml/runs`; the `ml` capability)."""
        return parse(S.MlRunsOut, self.get_json("/api/ml/runs"))

    def ml_run(self, source: str, run: str) -> S.MlRunDetailOut:
        """One training run with every epoch's metrics (`GET /api/ml/runs/{source}/{run}`)."""
        return parse(S.MlRunDetailOut, self.get_json(f"/api/ml/runs/{source}/{run}"))

    def ml_reports(self) -> S.MlReportsOut:
        """The physics floor and LightGBM reports (`GET /api/ml/reports`)."""
        return parse(S.MlReportsOut, self.get_json("/api/ml/reports"))

    # ---- embedding traces

    def embedding_create(self, *, run: str, **fields: Any) -> S.JobCreatedOut:
        """POST /api/embedding (`EmbeddingIn`: state_id or smiles, run, file, source)."""
        body = _request_body(S.EmbeddingIn, {"run": run, **_body(**fields)})
        return parse(S.JobCreatedOut, self.post_json("/api/embedding", body))

    def embedding(self, embedding_id: str) -> Any:
        """The stored payload of a trace (JSON)."""
        return self.get_json(f"/api/embedding/{embedding_id}")

    # ---- curation (read) and the dataset browser

    def curate_list(
        self,
        *,
        status: str | None = None,
        predictor: str | None = None,
        stale: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[S.MechanismSummaryOut]:
        params = _params(status=status, predictor=predictor, stale=stale)
        data = self.get_json("/api/curate", limit=limit, offset=offset, **params)
        return parse_list(S.MechanismSummaryOut, data)

    def curate_get(self, mechanism_id: int) -> S.MechanismOut:
        return parse(S.MechanismOut, self.get_json(f"/api/curate/{int(mechanism_id)}"))

    def browse_status(self) -> S.IndexStatusOut:
        return parse(S.IndexStatusOut, self.get_json("/api/browse/status"))

    def browse_search(self, q: str = "", **params: Any) -> S.SearchOut:
        """GET /api/browse/search: q, mode (exact | substructure | ...), side, source (a list),
        status, resonance, limit, offset."""
        return parse(S.SearchOut, self.get_json("/api/browse/search", q=q, **params))

    def browse_step(self, ref: str) -> S.StepViewOut:
        return parse(S.StepViewOut, self.get_json(f"/api/browse/step/{ref}"))


def _request_body(model: type[BaseModel], value: BaseModel | Mapping[str, Any]) -> dict:
    """A request body checked against its input model before it is sent: a misspelt field is
    a local `KsmechAPIError(REQUEST_INVALID)` (the server would answer 422 anyway)."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    try:
        model.model_validate(dict(value))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise KsmechAPIError("REQUEST_INVALID", f"{where}: {first.get('msg')}") from exc
    return {k: _dumped(v) for k, v in value.items()}


def _dumped(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value
