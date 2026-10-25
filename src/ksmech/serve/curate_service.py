"""The curation service (design 6.4): what the `/curate` routes call.

`CurateService(ctx, engine, *, networks, submit)` reads networks only through the injected
`NetworkSource` (job id -> (NetworkOut, the job's PredictIn)), never through `network.py`, and
runs export and re-check jobs through the injected `JobSubmitter`. Every method returns a
`schemas` model; errors are `ServeError`s with their HTTP status. Sessions are short (one unit
of work per call).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlmodel import col, select

from ksmech.serve import db
from ksmech.serve.contracts import JobSubmitter, NetworkSource
from ksmech.serve.curation import (
    as_model,
    create_mechanism,
    mechanism_out,
    set_order_groups,
    set_status,
    staleness,
    summary_out,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.schemas import (
    CurateIn,
    ExportIn,
    ExportReportOut,
    JobCreatedOut,
    MechanismOut,
    MechanismPatchIn,
    MechanismSummaryOut,
    RecheckIn,
    RecheckOneIn,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ksmech.serve.context import ApiContext

log = logging.getLogger(__name__)

MAX_LIST = 500


class CurateService:
    def __init__(
        self,
        ctx: ApiContext,
        engine: Engine,
        *,
        networks: NetworkSource,
        submit: JobSubmitter,
    ) -> None:
        self.ctx = ctx
        self.engine = engine
        self.networks = networks
        self.submit = submit

    def _out(self, session, m: db.Mechanism) -> MechanismOut:
        c = self.ctx
        return mechanism_out(session, m, c.registry, c.spec_version, c.physics_fp)

    @staticmethod
    def _mechanism(session, mid: int) -> db.Mechanism:
        m = session.get(db.Mechanism, int(mid))
        if m is None:
            raise ServeError(Code.MECHANISM_NOT_FOUND, f"no mechanism {mid}", status=404)
        return m

    def create(self, body: CurateIn | dict) -> MechanismOut:
        """POST /curate: one mechanism from a pathway or a chain of edges of a finished job."""
        body = as_model(CurateIn, body)
        if (body.pathway_id is None) == (body.edge_ids is None):
            raise ServeError(Code.REQUEST_INVALID, "give exactly one of pathway_id and edge_ids")
        network, request = self.networks(body.job_id)
        with db.session(self.engine) as s:
            m = create_mechanism(s, network, request, body, body.job_id)
            return self._out(s, m)

    def list(
        self,
        *,
        status: str | None = None,
        predictor: str | None = None,
        stale: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MechanismSummaryOut]:
        """GET /curate: newest first, filtered by status, predictor and staleness."""
        limit, offset = max(0, min(int(limit), MAX_LIST)), max(0, int(offset))
        q = select(db.Mechanism)
        if status is not None:
            q = q.where(db.Mechanism.status == status)
        if predictor is not None:
            q = q.where(db.Mechanism.predictor == predictor)
        q = q.order_by(col(db.Mechanism.created_at).desc(), col(db.Mechanism.id).desc())
        with db.session(self.engine) as s:
            mechanisms = list(s.exec(q).all())
            steps = _steps_by_mechanism(s, [m.id for m in mechanisms])
            last = _last_rechecks(s, [m.id for m in mechanisms])
        c = self.ctx
        out = []
        for m in mechanisms:
            mine = steps.get(m.id, [])
            is_stale = bool(staleness(m, mine, c.registry, c.spec_version, c.physics_fp))
            if stale is None or is_stale == bool(stale):
                out.append(summary_out(m, len(mine), is_stale, last.get(m.id)))
        return out[offset : offset + limit]

    def get(self, mid: int) -> MechanismOut:
        """GET /curate/{mid}."""
        with db.session(self.engine) as s:
            return self._out(s, self._mechanism(s, mid))

    def patch(self, mid: int, body: MechanismPatchIn | dict) -> MechanismOut:
        """PATCH /curate/{mid}: notes, then the status (transition table), then the order
        groups (checked against the new status); all or nothing."""
        body = as_model(MechanismPatchIn, body)
        with db.session(self.engine) as s:
            m = self._mechanism(s, mid)
            if body.notes is not None:
                m.notes = body.notes
                s.add(m)
            if body.status is not None:
                set_status(s, m, body.status)
            if body.order_groups is not None:
                set_order_groups(s, m, list(body.order_groups))
            s.flush()
            return self._out(s, m)

    def export(self, body: ExportIn | dict | None = None) -> JobCreatedOut:
        """POST /curate/export: an `export` job (result ExportReportOut)."""
        body = as_model(ExportIn, body or {})
        return JobCreatedOut.model_validate(self.submit("export", body.model_dump(mode="json")))

    def latest_export(self) -> ExportReportOut:
        """GET /curate/export/latest: the report of the most recent export."""
        q = select(db.ExportRecord).order_by(col(db.ExportRecord.created_at).desc())
        with db.session(self.engine) as s:
            rec = s.exec(q.limit(1)).first()
        if rec is None:
            raise ServeError(Code.EXPORT_NOT_FOUND, "no export has run yet", status=404)
        return ExportReportOut.model_validate(json.loads(rec.report_json))

    def recheck(self, body: RecheckIn | dict | None = None) -> JobCreatedOut:
        """POST /curate/recheck: a `recheck` job; an explicit predictor must be available."""
        body = as_model(RecheckIn, body or {})
        if body.predictor is not None:
            self.ctx.registry.get(body.predictor)  # 422 unknown / 409 unavailable
        return JobCreatedOut.model_validate(self.submit("recheck", body.model_dump(mode="json")))

    def recheck_one(self, mid: int, body: RecheckOneIn | dict | None = None) -> JobCreatedOut:
        """POST /curate/{mid}/recheck: re-check one mechanism now, stale or not."""
        body = as_model(RecheckOneIn, body or {})
        with db.session(self.engine) as s:
            m = self._mechanism(s, mid)
        self.ctx.registry.get(body.predictor or m.predictor)
        request = RecheckIn(mechanism_ids=[int(mid)], predictor=body.predictor, stale_only=False)
        return JobCreatedOut.model_validate(self.submit("recheck", request.model_dump(mode="json")))


def _steps_by_mechanism(session, ids: list[int]) -> dict[int, list[db.CuratedStep]]:
    out: dict[int, list[db.CuratedStep]] = {}
    if not ids:
        return out
    q = select(db.CuratedStep).where(col(db.CuratedStep.mechanism_id).in_(ids))
    for step in session.exec(q.order_by(col(db.CuratedStep.step_index))).all():
        out.setdefault(step.mechanism_id, []).append(step)
    return out


def _last_rechecks(session, ids: list[int]) -> dict[int, object]:
    if not ids:
        return {}
    q = (
        select(db.Recheck.mechanism_id, func.max(db.Recheck.created_at))
        .where(col(db.Recheck.mechanism_id).in_(ids))
        .group_by(db.Recheck.mechanism_id)
    )
    return {mid: _aware(t) for mid, t in session.exec(q).all()}


def _aware(t):
    """`func.max` bypasses the UTCDateTime decorator on SQLite: re-attach UTC."""
    from datetime import UTC, datetime

    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=UTC))
