"""The worker side of one step: the predictor-driven beam scan of a pool state (design 4.3)
and a user-applied step (design 4.6, steps 2 and 7), with their task bodies `t_expand` and
`t_apply` (design 5.2).

`expand_state` never builds an arrow itself: it scores the pool (`an.score_<name>` when the
analysis recorded the predictor's current version for it, else the predictor's `score`),
masks it with `an.valid & scope_mask`, takes the softmax at T (`log_softmax_valid`), and
applies the steps `predictor.propose` returns, in their order.
Each scanned proposal either joins the group of its child pool key (the first member, the
best scoring, is the representative) or is counted under its reason code
(`APPLY_FAILED:<ApplyReason>`, `SPECIES_KEY_FAILED`, `RESONANCE_ONLY_STEP`); the grouping and
the stopping rule are `contracts.ChildScan`. A child's probability is the sum of its scanned
members' p, a lower bound (the true marginal is at most probability + unscanned mass).

Every state returned (each kept child, the product of a user step) is saved with
`pools.save_state` before the function returns (design 2.3).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ksmech.serve import stepapply as sa
from ksmech.serve.analysis import PoolWork, pool_work
from ksmech.serve.candidates import reason_name, scope_mask
from ksmech.serve.contracts import (
    ArrowSpec,
    ChildScan,
    ExpandRequest,
    Expansion,
    PoolScores,
    PoolState,
    ProposedChild,
    ProposedStep,
    Runtime,
    ScoringContext,
    StepOutcome,
    log_softmax_valid,
)
from ksmech.serve.errors import Code, ServeError, error_body
from ksmech.serve.physics_view import brief_physics, cell_rank, site_pairs
from ksmech.serve.pools import save_state

log = logging.getLogger(__name__)

MAX_ALTERNATIVES = 5
MAX_FAILED_EXAMPLES = 10
USER_SITES_FLAG = "user_sites_not_contact"
NO_SITE_PAIR_FLAG = "site_pair_unavailable"


@dataclass
class _Member:
    step: ProposedStep
    outcome: sa.ApplyOutcome
    p: float


@dataclass
class _Scan:
    scan: ChildScan
    groups: dict[str, list[_Member]] = field(default_factory=dict)
    failed: list[tuple[ProposedStep, float, str, str]] = field(default_factory=list)
    stop: str = "exhausted"


# ---- scores


def _predictor(rt_ctx: Runtime, name: str):
    predictor = rt_ctx.scorers.get(name)
    if predictor is None:
        msg = f"predictor {name!r} is not available in the worker ({sorted(rt_ctx.scorers)})"
        raise ServeError(Code.PREDICTOR_UNAVAILABLE, msg, status=409)
    return predictor


def _scores(rt_ctx: Runtime, work: PoolWork, predictor, ctx: ScoringContext) -> np.ndarray:
    """The stored `score_<name>` when the analysis recorded this predictor's current version
    for it, else `predictor.score(ctx)`."""
    stored = work.an.stored_scores(predictor.name, predictor.info().version)
    if stored is not None:
        return stored
    return np.asarray(predictor.score(ctx), dtype=np.float64)


def _temperature(predictor, temperature: float | None) -> float:
    return float(predictor.info().default_temperature if temperature is None else temperature)


def pool_scores(rt_ctx: Runtime, work: PoolWork, req: ExpandRequest):
    """(predictor, PoolScores, ScoringContext) of one state for the request (design 4.3)."""
    predictor = _predictor(rt_ctx, req.predictor)
    ctx = ScoringContext(work.sample, work.rt, rt_ctx.loader, work.comps)
    scores = _scores(rt_ctx, work, predictor, ctx)
    valid = np.asarray(work.an.valid, dtype=bool) & scope_mask(work.rt, req.candidate_scope)
    masked = np.where(valid, scores, -np.inf)
    t = _temperature(predictor, req.temperature)
    logp = log_softmax_valid(masked, valid, t)
    an = work.an
    pool = PoolScores(masked, valid, logp, an.contact_sites, an.rows, an.cols, t)
    return predictor, pool, ctx


# ---- arrows of a step


def _cell(work: PoolWork, arrow: ArrowSpec) -> tuple[int, int]:
    return work.an.cell(int(arrow.donor), int(arrow.acceptor))


def step_arrows(
    rt_ctx: Runtime,
    work: PoolWork,
    arrows: Sequence[ArrowSpec],
    sources: Sequence[str],
    product,
) -> tuple[list[dict], dict | None, str | None, list[int]]:
    """(ArrowOut dicts with their physics briefs, the encoding or None, the encoding failure
    or None, touched maps). Touched maps come from `edit_diff` against the Kekule base the
    encoding used (the drawn state when the encoding failed)."""
    rt = work.rt
    try:
        enc = sa.encode_dataset_arrows(rt, product, arrows)
        base, encoded, reason = enc.base, enc.to_dict(), None
    except sa.EncodeError as exc:
        enc, base, encoded, reason = None, rt.mol, None, exc.detail
    touched = sa.edit_diff(base, product).touched_maps
    out = []
    for k, (arrow, source) in enumerate(zip(arrows, sources, strict=True)):
        i, j = _cell(work, arrow)
        brief = brief_physics(work.an, rt, i, j, rt_ctx.occupations)
        info = None if enc is None else enc.arrow_info(k)
        out.append(sa.arrow_out(rt, arrow, source, info, brief))
    return out, encoded, reason, touched


def _plain_arrows(rt_ctx: Runtime, work: PoolWork, step: ProposedStep) -> list[dict]:
    """ArrowOut dicts of a proposal that gave no child (source kind unknown)."""
    out = []
    for arrow, source in zip(step.arrows, step.site_sources, strict=True):
        try:
            i, j = _cell(work, arrow)
            brief = brief_physics(work.an, work.rt, i, j, rt_ctx.occupations)
        except ServeError:
            brief = None
        try:
            out.append(sa.arrow_out(work.rt, arrow, source, None, brief))
        except (KeyError, IndexError, ValueError):
            out.append({"donor": arrow.donor, "acceptor": arrow.acceptor, "physics": None})
    return out


# ---- the scan


def _failure(outcome: sa.ApplyOutcome, parent_key: str) -> str | None:
    if not outcome.ok:
        return outcome.failure
    if outcome.state.pool_key == parent_key:
        return str(Code.RESONANCE_ONLY_STEP)
    return None


def _scan(work: PoolWork, steps: list[ProposedStep], req: ExpandRequest, n_valid: int) -> _Scan:
    out = _Scan(ChildScan(req.beam, req.max_scan))
    scan, parent_key = out.scan, work.state.pool_key
    for n, step in enumerate(steps):
        stop = scan.should_stop(len(steps) - n)
        if stop is not None:
            out.stop = stop
            return out
        p = math.exp(step.logp) if math.isfinite(step.logp) else 0.0
        outcome = sa.apply_step(work.rt, step.arrows)
        failure = _failure(outcome, parent_key)
        if failure is not None:
            scan.add(p, None, failure)
            if len(out.failed) < MAX_FAILED_EXAMPLES:
                detail = outcome.detail or f"child pool key {parent_key} is the parent's"
                out.failed.append((step, p, failure, detail))
            continue
        key = outcome.state.pool_key
        scan.add(p, key)
        out.groups.setdefault(key, []).append(_Member(step, outcome, p))
    capped = scan.n_scanned >= req.max_scan and n_valid > scan.n_scanned
    out.stop = "max_scan" if capped else "exhausted"
    return out


def _alternative(rt_ctx: Runtime, work: PoolWork, m: _Member) -> dict:
    arrows, *_ = step_arrows(rt_ctx, work, m.step.arrows, m.step.site_sources, m.outcome.product)
    return {"arrows": arrows, "score": m.step.score, "probability": m.p, "rank": m.step.rank}


def _child(rt_ctx: Runtime, work: PoolWork, members: list[_Member], mass: float) -> ProposedChild:
    rep = members[0]
    arrows, encoded, reason, touched = step_arrows(
        rt_ctx, work, rep.step.arrows, rep.step.site_sources, rep.outcome.product
    )
    save_state(rep.outcome.state, rt_ctx.settings)
    alternatives = [_alternative(rt_ctx, work, m) for m in members[1 : 1 + MAX_ALTERNATIVES]]
    return ProposedChild(
        state=rep.outcome.state,
        arrows=arrows,
        encoded=encoded,
        encode_reason=reason,
        touched_maps=touched,
        score=float(rep.step.score),
        probability=float(mass),
        logp=math.log(mass) if mass > 0 else float("-inf"),
        rank=int(rep.step.rank),
        n_alternatives=len(members) - 1,
        alternatives=alternatives,
    )


def _failed_example(rt_ctx: Runtime, work: PoolWork, item) -> dict:
    step, p, code, detail = item
    arrows = _plain_arrows(rt_ctx, work, step)
    return {"rank": step.rank, "arrows": arrows, "probability": p, "code": code, "detail": detail}


def _analysis_failed(exc: ServeError) -> Expansion:
    if str(exc.code) == Code.ANALYSIS_FAILED:
        return Expansion.failed(Code.ANALYSIS_FAILED, exc.message, exc.upstream)
    return Expansion.failed(Code.ANALYSIS_FAILED, exc.message, str(exc.code))


def _warnings(work: PoolWork, req: ExpandRequest) -> list[str]:
    errors = (work.an.meta or {}).get("predictor_errors") or {}
    out = []
    if req.predictor in errors:
        out.append(f"stored {req.predictor} scores missing: {errors[req.predictor]}")
    return out


def _no_candidates(work: PoolWork, req: ExpandRequest) -> Expansion:
    """No valid cell in scope: an ok expansion without children (the network's
    NO_CANDIDATES); with no distribution at all, both masses are 0."""
    return Expansion(
        ok=True,
        code=None,
        upstream=None,
        detail="no valid pair in scope",
        children=[],
        n_valid=0,
        n_scanned=0,
        scanned_mass=0.0,
        unscanned_mass=0.0,
        failures={},
        failed_examples=[],
        fermi_smeared=work.an.fermi_smeared,
        warnings=_warnings(work, req),
    )


def expand_state(rt_ctx: Runtime, state: PoolState, req: ExpandRequest) -> Expansion:
    """The beam expansion of one state (design 4.3). Never raises for a chemistry outcome:
    an analysis failure, an unavailable predictor or an invalid request is `Expansion.failed`."""
    try:
        work = pool_work(rt_ctx, state)
    except ServeError as exc:
        return _analysis_failed(exc)
    try:
        predictor, pool, ctx = pool_scores(rt_ctx, work, req)
        n_valid = int(pool.valid.sum())
        steps = predictor.propose(pool, ctx, k=req.max_scan) if n_valid else []
    except ServeError as exc:
        return Expansion.failed(str(exc.code), exc.message, exc.upstream)
    except ValueError as exc:  # a temperature <= 0 or a NaN score: never a silent probability
        return Expansion.failed(Code.REQUEST_INVALID, str(exc))
    an = work.an
    if not n_valid:
        return _no_candidates(work, req)
    sc = _scan(work, steps, req, n_valid)
    children = [_child(rt_ctx, work, sc.groups[k], sc.scan.mass[k]) for k in sc.scan.kept()]
    failures = Counter(sc.scan.failures)
    failures[Code.ARROW_ENCODING_FAILED] += sum(c.encode_reason is not None for c in children)
    failures = {str(k): int(v) for k, v in failures.items() if v}
    if failures:
        log.warning("expand %s (%s): %s", state.state_id, req.predictor, failures)
    return Expansion(
        ok=True,
        code=None,
        upstream=None,
        detail=f"scan stopped: {sc.stop}",
        children=children,
        n_valid=n_valid,
        n_scanned=sc.scan.n_scanned,
        scanned_mass=float(sc.scan.scanned_mass),
        unscanned_mass=float(sc.scan.unscanned_mass),
        failures=failures,
        failed_examples=[_failed_example(rt_ctx, work, x) for x in sc.failed],
        fermi_smeared=an.fermi_smeared,
        warnings=_warnings(work, req),
    )


# ---- a user step (override)


def _pair_label(an, i: int, j: int) -> dict:
    ok = bool(an.valid[i, j])
    return {"pair_valid": ok, "reason": None if ok else reason_name(int(an.reasons[i, j]))}


def _view_row(an, scores, valid, logp, i: int, j: int) -> dict:
    if not bool(valid[i, j]):
        return {"score": None, "probability": None, "rank": None, **_pair_label(an, i, j)}
    score = float(scores[i, j])
    return {
        "score": score if math.isfinite(score) else None,
        "probability": float(np.exp(logp[i, j])),
        "rank": cell_rank(scores, valid, i, j),
        **_pair_label(an, i, j),
    }


def predictor_view(
    rt_ctx: Runtime, work: PoolWork, arrows: Sequence[ArrowSpec], name: str, t: float | None
) -> tuple[list[dict], list[dict]]:
    """(PredictorViewOut dicts per user arrow, extra issues): what the predictor thought of
    each user arrow's cell, its softmax over every valid pair at T and its rank. An
    unavailable predictor gives null values and a PREDICTOR_UNAVAILABLE warning instead of
    failing the override."""
    an = work.an
    cells = [_cell(work, a) for a in arrows]
    predictor = rt_ctx.scorers.get(name)
    if predictor is None:
        empty = {"score": None, "probability": None, "rank": None}
        rows = [empty | _pair_label(an, i, j) for i, j in cells]
        msg = f"predictor {name!r} is not available in the worker: no predictor view"
        return rows, [error_body(Code.PREDICTOR_UNAVAILABLE, msg, detail={"severity": "warning"})]
    ctx = ScoringContext(work.sample, work.rt, rt_ctx.loader, work.comps)
    valid = np.asarray(an.valid, dtype=bool)
    masked = np.where(valid, _scores(rt_ctx, work, predictor, ctx), -np.inf)
    logp = log_softmax_valid(masked, valid, _temperature(predictor, t))
    return [_view_row(an, masked, valid, logp, i, j) for i, j in cells], []


def _attach_site_pair(rt_ctx: Runtime, work: PoolWork, arrow: ArrowSpec, out: dict) -> None:
    """Sites that differ from the cell's contact sites: the site-pair diagnostics of design
    7.4 (extra key `site_pair`) and a flag on the brief."""
    i, j = _cell(work, arrow)
    contact = tuple(int(x) for x in work.an.contact_sites[i, j])
    if (int(arrow.donor_site), int(arrow.acceptor_site)) == contact:
        return
    flags = out["physics"]["flags"]
    flags.append(USER_SITES_FLAG)
    try:
        block = site_pairs(
            work, rt_ctx.kernels, arrow.donor, arrow.acceptor, arrow.donor_site, arrow.acceptor_site
        )
        out["site_pair"] = block["site_pairs"][0]
    except ServeError as exc:  # a site that is not a centre (allow_invalid): no placement
        flags.append(NO_SITE_PAIR_FLAG)
        out["site_pair"] = None
        log.info("no site pair for %s: %s", arrow, exc.message)


def _checked(work: PoolWork, arrows: Sequence[ArrowSpec], allow_invalid: bool) -> list:
    an = work.an
    issues = sa.check_arrows(work.rt, arrows, an.reasons, an.rows, an.cols, allow_invalid)
    blocking = [x for x in issues if x.blocking]
    if blocking:
        raise blocking[0].error()
    return issues


def _applied(work: PoolWork, arrows: Sequence[ArrowSpec]) -> sa.ApplyOutcome:
    out = sa.apply_step(work.rt, arrows)
    if not out.ok:
        msg = f"{out.code}: {out.upstream}: {out.detail}"
        raise ServeError(out.code, msg, status=422, upstream=out.upstream)
    if out.state.pool_key == work.state.pool_key:
        msg = f"the arrows give the same pool key {work.state.pool_key} (a resonance-only step)"
        raise ServeError(Code.RESONANCE_ONLY_STEP, msg, status=422)
    return out


def apply_user_step(
    rt_ctx: Runtime,
    state: PoolState,
    arrows: Sequence[ArrowSpec],
    allow_invalid: bool,
    predictor: str,
    temperature: float | None,
) -> StepOutcome:
    """The user's arrows applied in their order (design 4.6 steps 2 and 7). Raises
    ServeError (422) for a blocking check, APPLY_FAILED (upstream ApplyReason),
    SPECIES_KEY_FAILED or RESONANCE_ONLY_STEP; the product state is saved."""
    arrows = [a if isinstance(a, ArrowSpec) else ArrowSpec.from_dict(a) for a in arrows]
    if not arrows:
        raise ServeError(Code.REQUEST_INVALID, "a step needs at least one arrow", status=422)
    work = pool_work(rt_ctx, state)
    issues = _checked(work, arrows, allow_invalid)
    out = _applied(work, arrows)
    view, view_issues = predictor_view(rt_ctx, work, arrows, predictor, temperature)
    sources = ["user"] * len(arrows)
    arrow_dicts, encoded, reason, touched = step_arrows(rt_ctx, work, arrows, sources, out.product)
    for arrow, d in zip(arrows, arrow_dicts, strict=True):
        _attach_site_pair(rt_ctx, work, arrow, d)
    save_state(out.state, rt_ctx.settings)
    return StepOutcome(
        state=out.state,
        arrows=arrow_dicts,
        encoded=encoded,
        encode_reason=reason,
        touched_maps=touched,
        predictor_view=view,
        issues=[x.to_body() for x in issues] + view_issues,
    )


# ---- task bodies


def t_expand(rt: Runtime, payload: dict) -> dict:
    """{"state", "req"} -> Expansion.to_dict()."""
    state = PoolState.from_dict(payload["state"])
    req = ExpandRequest.from_dict(payload["req"])
    return expand_state(rt, state, req).to_dict()


def t_apply(rt: Runtime, payload: dict) -> dict:
    """{"state", "arrows", "allow_invalid", "predictor", "temperature"} ->
    StepOutcome.to_dict()."""
    state = PoolState.from_dict(payload["state"])
    arrows = [ArrowSpec.from_dict(a) for a in payload["arrows"]]
    temperature = payload.get("temperature")
    outcome = apply_user_step(
        rt,
        state,
        arrows,
        bool(payload.get("allow_invalid", False)),
        str(payload.get("predictor") or "floor"),
        None if temperature is None else float(temperature),
    )
    return outcome.to_dict()
