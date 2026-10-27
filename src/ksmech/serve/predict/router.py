"""Routes of prediction: predictors, predict jobs, overrides, expansions, the live step check,
pathways (the local checkpoints are `serve.ml`)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ksmech.serve.core import schemas as S
from ksmech.serve.core.auth import Role, require_role
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.http import (
    JobId,
    Svc,
    blank,
    check_id,
)

router = APIRouter(prefix="/api")


@router.get(
    "/predictors",
    response_model=list[S.PredictorInfoOut],
    dependencies=[require("predict", role=Role.READER)],
)
def predictors(svc: Svc) -> list[S.PredictorInfoOut]:
    spec = svc.ctx.spec_version
    return [
        S.PredictorInfoOut(**info.to_dict(), spec_version=spec) for info in svc.ctx.registry.info()
    ]


@router.post(
    "/predict",
    response_model=S.JobCreatedOut,
    status_code=202,
    dependencies=[require("predict")],
)
def predict(svc: Svc, body: S.PredictIn) -> S.JobCreatedOut:
    return svc.jobs.predict(body)


@router.get("/jobs", response_model=list[S.JobSummaryOut])
def list_jobs(
    svc: Svc,
    kind: str | None = None,
    status: str | None = None,
    limit: Annotated[int, Query(ge=0, le=1000)] = 50,
) -> list[S.JobSummaryOut]:
    return svc.jobs.list(kind=blank(kind), status=blank(status), limit=limit)


@router.get("/jobs/{job_id}", response_model=S.JobOut)
def get_job(
    svc: Svc, job_id: JobId, since_revision: Annotated[int | None, Query(ge=0)] = None
) -> S.JobOut:
    return svc.jobs.get(check_id("job_id", job_id), since_revision=since_revision)


@router.post(
    "/jobs/{job_id}/cancel", response_model=S.JobOut, dependencies=[require_role(Role.USER)]
)
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
