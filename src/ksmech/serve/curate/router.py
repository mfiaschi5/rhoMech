"""Routes of curation (static routes before /curate/{mid})."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Query
from fastapi.responses import PlainTextResponse

from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.http import (
    Mid,
    Svc,
    blank,
    check_id,
)

router = APIRouter(prefix="/api")


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
        status=blank(status), predictor=blank(predictor), stale=stale, limit=limit, offset=offset
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
