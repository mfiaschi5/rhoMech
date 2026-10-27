"""The 3D view and depiction services (design 6.4 `view3d`), API side.

Services take an `ApiContext` and return `schemas` models (or `str` for cube files). Heavy work
(cube grids) runs in the worker through `ctx.interactive`, which raises
`Pending` while a task runs; the API side only reads the files the worker wrote (and checks
that every path it reads lies under `data/serve`). Geometry and depictions are computed here:
RDKit and the token files already in memory.
"""

from __future__ import annotations

from pathlib import Path

from ksmech.serve.chem import cubes as cubemod
from ksmech.serve.chem import geometry
from ksmech.serve.core import schemas
from ksmech.serve.core.contracts import PoolState
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.depiction import depict
from ksmech.serve.runtime.context import ApiContext
from ksmech.serve.science.analysis import analysis_digest

CUBE_TIMEOUT_S = 180


def _tokens(ctx: ApiContext, state: PoolState):
    """The state's `ReactantTokens`; missing tokens are featurized first (`ensure_tokens`:
    `Pending` while it runs, POOL_NOT_ANALYZED when it failed). No oriented analysis."""
    try:
        return ctx.reactant_tokens(state)
    except ServeError as exc:
        missing = (exc.detail or {}).get("code") == str(Code.MISSING_TOKENS)
        if exc.code != Code.POOL_NOT_ANALYZED or not missing:
            raise
    return ctx.ensure_tokens(state)


def _digest(ctx: ApiContext, state: PoolState) -> str:
    digest = analysis_digest(ctx.loader, state)
    if digest is None:
        msg = f"pool {state.state_id} has fragments without tokens"
        detail = {"code": str(Code.MISSING_TOKENS), "keys": list(state.fragment_token_keys)}
        raise ServeError(Code.POOL_NOT_ANALYZED, msg, status=409, detail=detail)
    return digest


def _read_serve_file(ctx: ApiContext, path: str | Path) -> str:
    p = Path(path).resolve()
    if not p.is_relative_to(ctx.settings.serve.resolve()):
        raise ServeError(Code.CONSISTENCY_FAILED, f"worker file outside data/serve: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ServeError(Code.CONSISTENCY_FAILED, f"cannot read {p}: {exc}") from exc


def geometry_out(ctx: ApiContext, state_id: str) -> schemas.GeometryOut:
    """Every fragment's record geometry (needs the tokens, not the analysis)."""
    state = ctx.state(state_id)
    rt = _tokens(ctx, state)
    return schemas.GeometryOut(state_id=state.state_id, species=geometry.species_geometry(rt))


def orbital_cube_text(
    ctx: ApiContext,
    state_id: str,
    token_id: int,
    *,
    spacing: float = cubemod.DEFAULT_SPACING_ANG,
    padding: float = cubemod.DEFAULT_PADDING_ANG,
) -> str:
    """The Gaussian cube of one orbital token on its cap +- padding (the `orbital_cube`
    task; read from the cache when present)."""
    state = ctx.state(state_id)
    rt = _tokens(ctx, state)
    f, _ = geometry.orbital_position(rt, token_id)
    spacing, padding = float(spacing), float(padding)
    cubemod.check_box(spacing, padding)  # REQUEST_INVALID before any task
    digest = _digest(ctx, state)
    path = cubemod.cube_path(ctx.settings, state.state_id, int(token_id), spacing, padding, digest)
    if not path.is_file():
        payload = {"state": state.to_dict(), "token": int(token_id)}
        payload |= {"spacing": spacing, "padding": padding}
        label = f"orbital cube of token {token_id} ({rt.fragments[f].key})"
        res = ctx.interactive("orbital_cube", payload, label=label, timeout_s=CUBE_TIMEOUT_S)
        path = Path(res["path"])
    return _read_serve_file(ctx, path)


def depiction_out(
    ctx: ApiContext, state_id: str, *, keep_maps: set[int], server_svg: bool
) -> schemas.DepictionOut:
    """The 2D depiction of any saved state (RDKit only); global sites only when every
    fragment's tokens are present."""
    state = ctx.state(state_id)
    rt = None
    if all(ctx.loader.has(k) for k in state.fragment_token_keys):
        try:
            rt = ctx.reactant_tokens(state)
        except ServeError:
            rt = None  # a correspondence failure: depict without global sites
    body = depict.pool_depiction(state, rt, set(keep_maps), bool(server_svg))
    return schemas.DepictionOut(**body)
