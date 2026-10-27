"""ML tools: the local checkpoints, the training runs with their metrics, and the evaluation
reports (all read only; the `ml` capability). Also the resources `ksmech://checkpoints` and
`ksmech://runs/{source}/{run}`."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ksmech.client import KsmechClient
from ksmech.client import summary as sm
from ksmech.mcp.tools import READ, Enabled, add, blocking, dumps

Source = Annotated[str, Field(description="The data source of the run, e.g. 'pmechdb'.")]
Run = Annotated[str, Field(description="The run name (as list_training_runs gives it).")]


class MlTools:
    """The ML tools over one client."""

    def __init__(self, client: KsmechClient) -> None:
        self.client = client

    def checkpoints(self) -> list[dict]:
        models = self.client.models()
        return [sm.checkpoint_summary(c) for c in models.checkpoints]

    async def list_checkpoints(self) -> str:
        """The trained model checkpoints on this server ('<source>/<run>/<file>': best or
        last), with their epoch, encoder (dense, flex, sparse), validation metrics (anchor
        top-1/top-5 and losses), whether they are usable (a dirty run is refused) and
        whether they are smoke runs. predict_mechanism takes one with predictor='network'
        and checkpoint_run."""
        return dumps(await blocking(self.checkpoints))

    async def list_training_runs(self) -> str:
        """The training runs ('<source>/<run>') with their encoder, number of epochs, the
        metrics of the best epoch (validation anchor top-k, losses), the checkpoints present,
        whether the run is a smoke run and whether its code was clean (a dirty run's
        checkpoints are refused)."""
        out = await blocking(self.client.ml_runs)
        return dumps([sm.run_row(r) for r in out.runs])

    async def training_run_metrics(
        self,
        source: Source,
        run: Run,
        detail: Annotated[
            bool, Field(description="Up to 60 epochs instead of the first and last 5.")
        ] = False,
    ) -> str:
        """One training run: its configuration, the metrics of every epoch (losses,
        validation anchor top-k; the first and last 5 epochs unless detail) and the names
        of the evaluation reports beside its checkpoints."""
        d = await blocking(self.client.ml_run, source, run)
        return dumps(sm.run_detail_summary(d, detail=detail))

    async def list_reports(
        self,
        detail: Annotated[bool, Field(description="Each report's JSON (lists cut).")] = False,
    ) -> str:
        """The evaluation reports of the lab: the physics floor (anchors ranked by the
        Klopman-Salem gap factor, no learned parameter) and the LightGBM reference, per data
        split, with their headline numbers (val / test top-1, top-3, top-5, median rank)."""
        out = await blocking(self.client.ml_reports)
        return dumps([sm.report_row(r, detail=detail) for r in out.reports])


def register(app: FastMCP, client: KsmechClient, enabled: Enabled) -> list[str]:
    if not enabled("ml"):
        return []
    t = MlTools(client)
    names = [
        add(app, t.list_checkpoints, "List checkpoints", READ),
        add(app, t.list_training_runs, "List training runs", READ),
        add(app, t.training_run_metrics, "Training run metrics", READ),
        add(app, t.list_reports, "List evaluation reports", READ),
    ]
    register_resources(app, t)
    return names


def register_resources(app: FastMCP, t: MlTools) -> None:
    """`ksmech://checkpoints` and `ksmech://runs/{source}/{run}` (JSON)."""

    @app.resource(
        "ksmech://checkpoints",
        name="checkpoints",
        description="The local model checkpoints with their validation metrics.",
        mime_type="application/json",
    )
    async def checkpoints_resource() -> str:
        return dumps(await blocking(t.checkpoints))

    @app.resource(
        "ksmech://runs/{source}/{run}",
        name="run_metrics",
        description="One training run with its per-epoch metrics.",
        mime_type="application/json",
    )
    async def run_resource(source: str, run: str) -> str:
        d = await blocking(t.client.ml_run, source, run)
        return dumps(sm.run_detail_summary(d, detail=True))
