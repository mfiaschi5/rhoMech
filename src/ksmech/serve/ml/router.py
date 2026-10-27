"""Routes of the ML artifacts: the local checkpoints (`/models`, open: the predict form picks
one) and the lab's training runs and reports (`/ml/*`, gated by the `ml` capability: off in
the public profile, curator and above)."""

from __future__ import annotations

from fastapi import APIRouter

from ksmech.serve.core import schemas as S
from ksmech.serve.core.capabilities import require
from ksmech.serve.core.http import Svc, check_id
from ksmech.serve.ml import runs
from ksmech.serve.ml.checkpoints import list_checkpoints, model_roots

router = APIRouter(prefix="/api")
lab = APIRouter(prefix="/api/ml", dependencies=[require("ml")])


@router.get("/models", response_model=S.ModelsOut)
def list_models(svc: Svc) -> S.ModelsOut:
    """The local checkpoints and the embedding capability they give (never gated)."""
    from ksmech.serve.core import capabilities

    on, reason = capabilities.status("embedding", svc.settings)
    return S.ModelsOut(
        roots=[str(r) for r in model_roots(svc.settings)],
        checkpoints=[S.CheckpointOut(**c.to_out()) for c in list_checkpoints(svc.settings)],
        embedding=S.CapabilityOut(enabled=on, reason=reason),
    )


@lab.get("/runs", response_model=S.MlRunsOut)
def list_runs(svc: Svc) -> S.MlRunsOut:
    """Every training run under the model roots (run.json, metrics, checkpoints, reports)."""
    return S.MlRunsOut(
        roots=[str(r) for r in model_roots(svc.settings)],
        runs=[S.MlRunOut(**r) for r in runs.list_runs(svc.settings)],
    )


@lab.get("/runs/{source}/{run}", response_model=S.MlRunDetailOut)
def get_run(svc: Svc, source: str, run: str) -> S.MlRunDetailOut:
    """One run: run.json, the metrics of every epoch and the reports beside its checkpoints."""
    detail = runs.run_detail(svc.settings, check_id("run_name", source), check_id("run_name", run))
    return S.MlRunDetailOut.model_validate(detail)


@lab.get("/reports", response_model=S.MlReportsOut)
def list_reports(svc: Svc) -> S.MlReportsOut:
    """The physics floor and LightGBM reports under data/processed/<source>/."""
    return S.MlReportsOut(
        roots=[str(r) for r in runs.report_roots(svc.settings)],
        reports=[S.MlReportOut(**r) for r in runs.list_reports(svc.settings)],
    )
