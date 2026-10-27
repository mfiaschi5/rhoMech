"""Routes of one pool state: normalization, the pool, its tokens, geometry, orbital cubes and
depiction."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from ksmech.serve.core import schemas as S
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.http import (
    StateId,
    Svc,
    TokenId,
    check_id,
    cube_response,
)
from ksmech.serve.pool import service as pool_service
from ksmech.serve.pool import view3d

router = APIRouter(prefix="/api")


@router.post("/normalize", response_model=S.NormalizeOut)
def normalize(svc: Svc, body: S.NormalizeIn) -> S.NormalizeOut:
    return pool_service.normalize(svc.ctx, body)


@router.post("/pools", response_model=S.PoolOut)
def create_pool(svc: Svc, body: S.NormalizeIn) -> S.PoolOut:
    return pool_service.create_pool(svc.ctx, body)


@router.get("/pool/{state_id}", response_model=S.PoolOut)
def get_pool(svc: Svc, state_id: StateId) -> S.PoolOut:
    return pool_service.pool_out(svc.ctx, check_id("state_id", state_id))


@router.get("/pool/{state_id}/tokens", response_model=S.TokensOut)
def pool_tokens(svc: Svc, state_id: StateId) -> S.TokensOut:
    return pool_service.tokens_out(svc.ctx, check_id("state_id", state_id))


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
    return cube_response(text, f"{sid}-{token_id}.cube")


@router.get("/pool/{state_id}/depiction", response_model=S.DepictionOut)
def pool_depiction(
    svc: Svc, state_id: StateId, keep_maps: str | None = None, server_svg: bool = False
) -> S.DepictionOut:
    sid = check_id("state_id", state_id)
    maps = _keep_maps(keep_maps)
    return view3d.depiction_out(svc.ctx, sid, keep_maps=maps, server_svg=server_svg)
