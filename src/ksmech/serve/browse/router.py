"""Routes of the dataset browser: index status and build, search, the step view (the whole
router is gated by the `browse` capability: off in the public profile)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from ksmech.serve.core import schemas as S
from ksmech.serve.core.auth import Role, require_role
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.http import (
    Svc,
    blank,
    check_id,
)

router = APIRouter(prefix="/api", dependencies=[require("browse")])


@router.get("/browse/status", response_model=S.IndexStatusOut)
def browse_status(svc: Svc) -> S.IndexStatusOut:
    return svc.browse.status()


@router.post(
    "/browse/index",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require_role(Role.CURATOR)],
)
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
        status=blank(status),
        resonance=resonance,
        limit=limit,
        offset=offset,
        **kwargs,
    )


@router.get("/browse/step/{ref}", response_model=S.StepViewOut)
def browse_step(svc: Svc, ref: str) -> S.StepViewOut:
    return svc.browse.step(check_id("ref", ref))
