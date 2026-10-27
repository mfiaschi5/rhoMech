"""The worker side of one step (spec v9, "Inference"): the predictor's candidates for a pool
state, applied, merged and cut to the beam, with the task body `t_expand`.

`expand_state` asks the predictor for its candidate reaction matrices (best first), applies each
one as B + R with RDKit's valence checks (`stepapply.apply_matrix`), and groups the products by
their pool key (the sorted multiset of their sigma keys): the first member of a group is its
representative, the others are alternatives. A candidate that gives no child is counted under
its reason code (`APPLY_FAILED:<issue code>`; `APPLY_FAILED:RESONANCE_ONLY_STEP` when the
product is the parent pool), and so is everything the predictor dropped (`COMPLETION_FAILED:...`).
The `beam` best groups are kept, in the predictor's order.

Each child carries its edge step (`edgestep.step_record`: R, anchor, arrows, orbital readings,
product SMILES and sigma keys), its score, its log-probability (None unless the predictor gives
probabilities) and a beam weight: the log-probability when there is one, else the log of the
child's share of the kept children's scores (|G_atom| of the floor) or a softmax of them (the
GBDT's ranking scores). The weight orders the beam and the pathways; it is not a probability
unless `scores_are_probabilities`.

Every state returned is saved with `pools.save_state` before the function returns.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

from ksmech.serve import edgestep, stepapply
from ksmech.serve.analysis import PoolWork, pool_work
from ksmech.serve.contracts import (
    ExpandRequest,
    Expansion,
    PoolState,
    ProposedChild,
    Runtime,
    log_softmax_valid,
)
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.pools import save_state

log = logging.getLogger(__name__)

MAX_ALTERNATIVES = 5
MAX_FAILED_EXAMPLES = 10


def _predictor(rt_ctx: Runtime, req: ExpandRequest):
    """The available predictor of the request, bound to it (`Predictor.bind`: the network
    takes the request's checkpoint)."""
    predictor = rt_ctx.scorers.get(req.predictor)
    if predictor is None:
        msg = (
            f"predictor {req.predictor!r} is not available in the worker ({sorted(rt_ctx.scorers)})"
        )
        raise ServeError(Code.PREDICTOR_UNAVAILABLE, msg, status=409)
    return predictor.bind(req)


def beam_weights(scores: list[float], logps: list, probabilities: bool) -> list[float]:
    """The log beam weight of each kept child (module docstring)."""
    if probabilities:
        return [float(x) if x is not None else float("-inf") for x in logps]
    s = np.asarray(scores, dtype=np.float64)
    if len(s) and (s >= 0).all() and s.sum() > 0:
        with np.errstate(divide="ignore"):
            return [float(x) for x in np.log(s / s.sum())]
    return [float(x) for x in log_softmax_valid(s, np.ones(len(s), dtype=bool), 1.0)]


def _group(state: PoolState, cands, failures: Counter, failed: list) -> dict[str, list]:
    groups: dict[str, list] = {}
    for rank, c in enumerate(cands, start=1):
        applied = stepapply.apply_matrix(state, c.matrix)
        if not applied.ok:
            code = f"APPLY_FAILED:{applied.issues[0].code}"
            failures[code] += 1
            if len(failed) < MAX_FAILED_EXAMPLES:
                failed.append(_failed_example(c, rank, code, applied.issues[0].message))
            continue
        groups.setdefault(applied.state.pool_key, []).append((rank, c, applied))
    return groups


def _failed_example(c, rank: int, code: str, detail: str) -> dict:
    return {
        "rank": rank,
        "matrix": stepapply.matrix_io(c.matrix),
        "score": c.score,
        "code": code,
        "detail": detail,
    }


def _alternative(rank: int, c) -> dict:
    return {
        "matrix": stepapply.matrix_io(c.matrix),
        "anchor": {"i": c.anchor[0], "j": c.anchor[1], "sign": c.anchor[2]},
        "score": c.score,
        "logp": c.logp,
        "rank": rank,
    }


def _child(rt_ctx: Runtime, work: PoolWork, members: list, weight: float, name: str):
    rank, c, applied = members[0]
    step = edgestep.step_record(
        work.state, c.matrix, pg=work.pg, rt=work.rt, occupations=rt_ctx.occupations,
        anchor=c.anchor,
    )  # fmt: skip
    step.pop("product_state")
    step |= {"channel": c.channel, "predictor": name}
    save_state(applied.state, rt_ctx.settings)
    return ProposedChild(
        state=applied.state,
        step=step,
        score=float(c.score),
        logp=None if c.logp is None else float(c.logp),
        weight=float(weight),
        rank=int(rank),
        n_alternatives=len(members) - 1,
        alternatives=[_alternative(r, a) for r, a, _ in members[1 : 1 + MAX_ALTERNATIVES]],
    )


def _failed(code, message: str, upstream=None) -> Expansion:
    return Expansion.failed(str(code), message, upstream)


def expand_state(rt_ctx: Runtime, state: PoolState, req: ExpandRequest) -> Expansion:
    """The beam expansion of one state. Never raises for a chemistry outcome: an analysis
    failure, an unavailable predictor or an invalid request is `Expansion.failed`."""
    try:
        work = pool_work(rt_ctx, state)
    except ServeError as exc:
        return _failed(Code.ANALYSIS_FAILED, exc.message, exc.upstream or str(exc.code))
    skips: Counter = Counter()
    try:
        predictor = _predictor(rt_ctx, req)
        cands = predictor.propose(rt_ctx, work, req.beam, req.candidate_scope, skips)
    except ServeError as exc:
        return _failed(exc.code, exc.message, exc.upstream)
    failed: list[dict] = []
    groups = _group(state, cands, skips, failed)
    kept = list(groups)[: req.beam]
    probabilities = bool(predictor.info().scores_are_probabilities)
    first = [groups[k][0][1] for k in kept]
    weights = beam_weights([c.score for c in first], [c.logp for c in first], probabilities)
    children = [
        _child(rt_ctx, work, groups[k], w, predictor.name)
        for k, w in zip(kept, weights, strict=True)
    ]
    failures = {str(k): int(v) for k, v in sorted(skips.items()) if v}
    if failures:
        log.warning("expand %s (%s): %s", state.state_id, req.predictor, failures)
    mass = float(sum(math.exp(w) for w in weights)) if probabilities else float("nan")
    return Expansion(
        ok=True,
        code=None,
        upstream=None,
        detail=f"{len(cands)} candidates, {len(groups)} distinct products, {len(kept)} kept",
        children=children,
        n_valid=len(cands),
        n_scanned=len(cands),
        scanned_mass=mass,
        unscanned_mass=float("nan") if not probabilities else max(0.0, 1.0 - mass),
        failures=failures,
        failed_examples=failed,
        fermi_smeared=work.an.fermi_smeared,
        warnings=[],
    )


def t_expand(rt: Runtime, payload: dict) -> dict:
    """{"state", "req"} -> Expansion.to_dict()."""
    state = PoolState.from_dict(payload["state"])
    req = ExpandRequest.from_dict(payload["req"])
    return expand_state(rt, state, req).to_dict()
