"""Re-check of curated steps after retraining or a physics change (design 9.4, spec v9).

`run_recheck` (the `recheck` job handler, API side) picks the mechanisms (given ids, or all;
only the stale ones with `stale_only`) and, for each step:

1. runs one `featurize` task per reactant token key the loader lacks, at JOB priority, exactly
   as the chemistry does before an expansion (a failure is recorded with its code);
2. then runs the worker task `recheck_step` (no QM inside): the predictor proposes its
   candidate reaction matrices for the step's reactant state (`RECHECK_BEAM` anchors or
   pairs), each is applied, and the distinct products are ranked in the predictor's order;
   the curated step's rank is the position of its product, compared by the sorted multiset
   of product sigma keys (the pool key: resonance forms of one product are one product);
3. writes one `Recheck` row. The mechanism's status is never changed.

Row fields keep their database names: `first_arrow_rank` is the 1-based rank of the curated
product among the predictor's distinct products (None when it is not proposed), `n_valid` the
number of distinct products, `aligned` whether the reactant state could be analyzed,
`product_match` FOUND / NOT_PROPOSED / UNCHECKED, and `top1_reproduces_product` rank == 1.

`stale_only` also leaves out a stale mechanism whose every step's latest Recheck row succeeded
(reason None) with the predictor the job would use, at its current version, the current spec
version and the current physics fingerprint: it was re-checked already, and a repeat would give
the same rows. Each one is counted and reported as RECHECKED_AT_CURRENT_VERSIONS. The stored
versions of the mechanism are never rewritten, so `curation.staleness` still calls it stale
(design 9.4); an explicit `stale_only=False` re-checks it again.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from sqlmodel import col, select

from ksmech.serve import db
from ksmech.serve.contracts import JobContext, Runtime
from ksmech.serve.curation import rechecks_of, staleness, steps_of
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.schemas import RecheckIn, RecheckReportOut, RecheckRowOut
from ksmech.serve.worker import Priority

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ksmech.serve.context import ApiContext
    from ksmech.serve.predictors import PredictorRegistry

log = logging.getLogger(__name__)

RECHECK_TIMEOUT_S = 900
RECHECK_BEAM = 20  # anchors (floor) or orbital pairs (gbdt) the predictor completes
UNCHECKED = "UNCHECKED"  # product_match when the re-check could not run
FOUND, NOT_PROPOSED = "FOUND", "NOT_PROPOSED"
RECHECKED = "RECHECKED_AT_CURRENT_VERSIONS"  # stale_only: already re-checked, left out
ROW_FIELDS = tuple(RecheckRowOut.model_fields)


# ---- worker side


def _blank(predictor: str, rt_ctx: Runtime) -> dict:
    from ksmech.serve.versions import physics_fingerprint, spec_version

    return {
        "predictor": predictor,
        "predictor_version": "",
        "spec_version": spec_version(rt_ctx.settings.spec_path),
        "physics_fingerprint": physics_fingerprint(),
        "aligned": False,
        "first_arrow_rank": None,
        "n_valid": None,
        "top1": None,
        "top5": None,
        "product_match": UNCHECKED,
        "top1_reproduces_product": None,
        "reason": None,
        "detail": "",
    }


def _state(rt_ctx: Runtime, doc: dict):
    """The step's reactant state: the saved one, else rebuilt from its mapped SMILES."""
    from ksmech.serve.pools import load_state, save_state, state_from_mapped

    try:
        return load_state(str(doc["reactant_state_id"]), rt_ctx.settings)
    except (ServeError, KeyError):
        state = state_from_mapped(str(doc["reactants_mapped"]))
        save_state(state, rt_ctx.settings)
        return state


def product_ranking(rt_ctx: Runtime, predictor, state, beam: int) -> tuple[list[str], Counter]:
    """The distinct product pool keys of the predictor's candidates, in its order, and the
    skip counts."""
    from ksmech.serve.analysis import pool_work
    from ksmech.serve.stepapply import apply_matrix

    work = pool_work(rt_ctx, state)
    skips: Counter = Counter()
    keys: list[str] = []
    for c in predictor.propose(rt_ctx, work, beam, "all", skips):
        applied = apply_matrix(state, c.matrix)
        if not applied.ok:
            skips[f"APPLY_FAILED:{applied.issues[0].code}"] += 1
        elif applied.state.pool_key not in keys:
            keys.append(applied.state.pool_key)
    return keys, skips


def recheck_step(
    rt_ctx: Runtime,
    doc: dict,
    predictor: str,
    product_pool_key: str,
    *,
    step_id: str | None = None,
) -> dict:
    """Re-evaluate one curated step on the current tokens (worker; never runs QM: the caller
    featurized first, and missing tokens are the reason MISSING_TOKENS). Returns the
    RecheckRowOut fields except mechanism_id, step_index and created_at, plus `detail`."""
    out = _blank(predictor, rt_ctx)
    p = rt_ctx.scorers.get(predictor)
    if p is None:
        msg = f"predictor {predictor!r} is not available in the worker"
        return out | {"reason": str(Code.PREDICTOR_UNAVAILABLE), "detail": msg}
    out["predictor_version"] = p.info().version or ""
    try:
        keys, skips = product_ranking(rt_ctx, p, _state(rt_ctx, doc), RECHECK_BEAM)
    except ServeError as exc:
        return out | {"reason": str(exc.code), "detail": exc.message}
    rank = keys.index(product_pool_key) + 1 if product_pool_key in keys else None
    return out | {
        "aligned": True,
        "first_arrow_rank": rank,
        "n_valid": len(keys),
        "top1": rank == 1,
        "top5": rank is not None and rank <= 5,
        "product_match": FOUND if rank is not None else NOT_PROPOSED,
        "top1_reproduces_product": rank == 1,
        "detail": f"skipped {dict(skips)}" if skips else "",
    }


def t_recheck_step(rt: Runtime, payload: dict) -> dict:
    """{"doc", "predictor", "product_pool_key", "mechanism_id"?, "step_index"?} ->
    RecheckRowOut fields (without created_at) plus `detail`."""
    mid, k = payload.get("mechanism_id"), payload.get("step_index")
    step_id = None if mid is None or k is None else f"recheck-{mid}-{k}"
    out = recheck_step(
        rt,
        dict(payload["doc"]),
        str(payload["predictor"]),
        str(payload["product_pool_key"]),
        step_id=step_id,
    )
    if mid is not None and k is not None:
        out |= {"mechanism_id": int(mid), "step_index": int(k)}
    return out


# ---- API side: the job handler


def _targets(ctx: JobContext, engine: Engine, req: RecheckIn, api: ApiContext, skipped: Counter):
    """[(mechanism, steps)] to re-check; unknown ids are skipped MECHANISM_NOT_FOUND. With
    `stale_only`, the stale mechanisms not re-checked at the current versions already."""
    with db.session(engine) as s:
        q = select(db.Mechanism).order_by(col(db.Mechanism.id))
        if req.mechanism_ids is not None:
            q = q.where(col(db.Mechanism.id).in_(list(req.mechanism_ids)))
        mechanisms = list(s.exec(q).all())
        out = [(m, steps_of(s, m.id)) for m in mechanisms]
    for mid in sorted(set(req.mechanism_ids or []) - {m.id for m in mechanisms}):
        skipped[str(Code.MECHANISM_NOT_FOUND)] += 1
        ctx.skip(Code.MECHANISM_NOT_FOUND, f"no mechanism {mid}", mechanism_id=mid)
    if req.stale_only:
        reg, sv, fp = api.registry, api.spec_version, api.physics_fp
        out = [(m, steps) for m, steps in out if staleness(m, steps, reg, sv, fp)]
        out = _not_rechecked(ctx, engine, req, api, out, skipped)
    return out


def _latest(rechecks: list[db.Recheck]) -> dict[int, db.Recheck]:
    """step index -> its latest Recheck row (`rechecks` ordered by (created_at, id))."""
    return {r.step_index: r for r in rechecks}


def _current(steps: list[db.CuratedStep], latest: dict[int, db.Recheck], want: tuple) -> bool:
    """Whether every step's latest Recheck succeeded at `want` = (predictor, predictor
    version, spec version, physics fingerprint)."""
    rows = [latest.get(st.step_index) for st in steps]
    return bool(rows) and all(
        r is not None
        and r.reason is None
        and (r.predictor, r.predictor_version, r.spec_version, r.physics_fingerprint) == want
        for r in rows
    )


def _not_rechecked(
    ctx: JobContext, engine: Engine, req: RecheckIn, api: ApiContext, targets: list, skipped
) -> list:
    """`targets` without the mechanisms re-checked at the current versions (module doc), each
    one counted and reported as RECHECKED_AT_CURRENT_VERSIONS."""
    with db.session(engine) as s:
        latest = {m.id: _latest(rechecks_of(s, m.id)) for m, _ in targets}
    out = []
    for m, steps in targets:
        predictor = predictor_for(req, m)
        version = _version(api.registry, predictor)
        want = (predictor, version, api.spec_version, api.physics_fp)
        if version is None or not _current(steps, latest[m.id], want):
            out.append((m, steps))
            continue
        skipped[RECHECKED] += 1
        detail = f"every step re-checked with {predictor} {version} at spec {api.spec_version}"
        ctx.skip(RECHECKED, detail, mechanism_id=m.id)
    return out


def _failed(api: ApiContext, base: dict, code: Any, detail: str, version: str | None) -> dict:
    return {
        **base,
        "created_at": db.utcnow(),
        "predictor_version": version or "",
        "spec_version": api.spec_version,
        "physics_fingerprint": api.physics_fp,
        "aligned": False,
        "first_arrow_rank": None,
        "n_valid": None,
        "top1": None,
        "top5": None,
        "product_match": UNCHECKED,
        "top1_reproduces_product": None,
        "reason": str(code),
        "detail": detail,
    }


def _featurize(ctx: JobContext, api: ApiContext, doc: dict) -> None:
    """One `featurize` task per reactant token key the loader lacks (JOB priority)."""
    from ksmech.serve.featurize import featurize_label
    from ksmech.serve.keys import side_keys

    for key in dict.fromkeys(side_keys(doc["reactants_mapped"]).token_keys):
        if api.loader.has(key):
            continue
        label = featurize_label(api.featurizer, key)
        ctx.progress(phase="featurize", message=label)
        r = ctx.scheduler.run(
            "featurize",
            {"key": key},
            priority=Priority.JOB,
            timeout_s=ctx.settings.qm_timeout_s + 120,
            owner=ctx.job_id,
            label=label,
        )
        if not r.ok:
            raise ServeError.from_task(r)


def predictor_for(req: RecheckIn, m: db.Mechanism) -> str:
    """The request's predictor, else the mechanism's (the floor for a user's mechanism)."""
    from ksmech.serve.contracts import USER_PREDICTOR

    return req.predictor or ("floor" if m.predictor == USER_PREDICTOR else m.predictor)


def _version(registry: PredictorRegistry, name: str) -> str | None:
    try:
        return registry.info_of(name).version
    except ServeError:
        return None


def _recheck_one(ctx: JobContext, api: ApiContext, m: db.Mechanism, step, predictor: str) -> dict:
    """One Recheck row's fields (plus `detail`) for one curated step."""
    doc = json.loads(step.arrows_json)
    base = {"mechanism_id": m.id, "step_index": step.step_index, "predictor": predictor}
    version = _version(api.registry, predictor)
    try:
        api.registry.get(predictor)
        if not doc.get("matrix"):
            raise ServeError(Code.REQUEST_INVALID, "a spec-v3 curated step (no reaction matrix)")
        _featurize(ctx, api, doc)
    except ServeError as exc:
        return _failed(api, base, exc.code, exc.message, version)
    payload = {"doc": doc, "predictor": predictor, "product_pool_key": step.product_pool_key}
    payload |= {"mechanism_id": m.id, "step_index": step.step_index}
    label = f"re-check of mechanism {m.id} step {step.step_index}"
    ctx.progress(phase="recheck", message=label)
    try:
        r = ctx.scheduler.run(
            "recheck_step",
            payload,
            priority=Priority.JOB,
            timeout_s=RECHECK_TIMEOUT_S,
            owner=ctx.job_id,
            label=label,
        )
    except ServeError as exc:  # e.g. WORKER_UNSTABLE from the scheduler itself
        return _failed(api, base, exc.code, exc.message, version)
    if not r.ok:
        return _failed(api, base, r.code or Code.UNEXPECTED, r.detail, version)
    return {**r.result, **base, "created_at": db.utcnow()}


def _save(engine: Engine, row: dict) -> RecheckRowOut:
    fields = {k: row[k] for k in ROW_FIELDS}
    with db.session(engine) as s:
        s.add(db.Recheck(**fields))
    return RecheckRowOut.model_validate(fields)


def run_recheck(
    ctx: JobContext, request: dict, *, engine: Engine, registry: PredictorRegistry
) -> dict:
    """The `recheck` job (module doc). Returns the RecheckReportOut fields."""
    from ksmech.serve.context import build_context

    req = RecheckIn.model_validate(request or {})
    api = build_context(ctx.settings, scheduler=ctx.scheduler, registry=registry)
    skipped: Counter = Counter()
    targets = _targets(ctx, engine, req, api, skipped)
    total, rows = sum(len(steps) for _, steps in targets), []
    ctx.progress(phase="recheck", done=0, total=total, message=f"{len(targets)} mechanisms")
    for m, steps in targets:
        for step in steps:
            if ctx.cancelled():
                raise ServeError(Code.CANCELLED, f"re-check cancelled after {len(rows)} steps")
            row = _recheck_one(ctx, api, m, step, predictor_for(req, m))
            if row["reason"] is not None:
                skipped[str(row["reason"])] += 1
                where = {"mechanism_id": m.id, "step_index": step.step_index}
                ctx.skip(row["reason"], row.get("detail", ""), **where)
            rows.append(_save(engine, row))
            ctx.progress(phase="recheck", done=len(rows), total=total)
    return _report(api, req, rows, skipped)


def _report(api: ApiContext, req: RecheckIn, rows: list[RecheckRowOut], skipped) -> dict:
    checked = [r for r in rows if r.reason is None]
    n = len(checked)
    predictors = sorted({r.predictor for r in rows})
    name = req.predictor or (predictors[0] if len(predictors) == 1 else None)
    notes = [] if name or not predictors else [f"per-mechanism predictors: {predictors}"]
    report = RecheckReportOut(
        n_steps=len(rows),
        n_checked=n,
        top1=sum(bool(r.top1) for r in checked) / n if n else None,
        top5=sum(bool(r.top5) for r in checked) / n if n else None,
        product_match=dict(sorted(Counter(r.product_match for r in rows).items())),
        skipped=dict(sorted(skipped.items())),
        rows=rows,
        provenance=api.provenance(name, notes=notes),
    )
    log.info("re-check: %d steps, %d checked, skipped %s", len(rows), n, dict(skipped))
    return report.model_dump(mode="json")
