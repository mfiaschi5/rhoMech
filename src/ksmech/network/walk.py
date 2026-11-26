"""The walk along a benchmark's recorded step targets (`ksmech-network evaluate --walk`): a
single-step measure on a pathway benchmark that needs no search budget.

From the case's start pool, for each recorded step target in order: the pool is expanded once,
its distinct children ranked by the predictor's log P, and the step's `rank` is the 1-based
place of the first child that holds the target's species (EXACT, identity keys; SKELETON would
call every proton transfer done before it happens). The walk then moves to that child, so the
by-products are the model's own, and the next target is ranked from there. It stops at the
first target no child holds (NOT_PROPOSED), at a pool that cannot be expanded
(EXPAND_FAILED:<code>, e.g. a by-product without tokens) or at a target the benchmark does not
give (STEP_TARGET_MISSING, STEP_TARGET_UNPARSED). A target the pool already holds is `held`:
no step is ranked for it.

Every rank after the first is conditional on the recorded pathway having been followed so
far, with the model's by-products: the first step is the one number that is a plain top-k
over the cases.

With a `network.rerank.Rerank` each ranked step also holds `rerank`: the target child's rank
after the first `top` children are reranked, per lambda, and whether all of them were scored
(`full`). The walk still moves to the child that holds the target, so one walk reads every
lambda; the summary adds the first step's top-k per lambda (`first_step_rerank`).
"""

from __future__ import annotations

from collections import Counter

from ksmech.network.benchmarks import Case
from ksmech.network.pools import pool_state
from ksmech.network.target import EXACT, arrival, target_from_smiles

KS = (1, 3, 5)


def _holds(target, state) -> bool:
    return arrival(target, state.species, state.skeletons, (EXACT,)) is not None


def ranked_proposals(expander, expansion) -> list:
    """The best proposal of each distinct child state of an expansion, the most probable first."""
    seen, out = set(), []
    for p in sorted(expansion.proposals, key=lambda p: -p.logp):
        key = expander.key(p.child)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def ranked_children(expander, expansion) -> list:
    """The distinct child states of an expansion, the most probable first."""
    return [p.child for p in ranked_proposals(expander, expansion)]


def _step(expander, state, target, skips: Counter, rerank=None):
    """(the step's row, the state the walk moves to or None, the stop code or None)."""
    if _holds(target, state):
        return {"rank": None, "held": True}, state, None
    ex = expander.expand(state)
    skips.update(ex.skips)
    if ex.failure:
        return None, None, f"EXPAND_FAILED:{ex.failure}"
    proposals = ranked_proposals(expander, ex)
    hit = next((k for k, p in enumerate(proposals) if _holds(target, p.child)), None)
    row = {"rank": None if hit is None else hit + 1, "held": False, "children": len(proposals)}
    if rerank is not None:
        row["rerank"] = rerank.ranks(proposals, hit, skips)
    if hit is None:
        return row, None, "NOT_PROPOSED"
    return row, proposals[hit].child, None


def walk_case(expander, case: Case, rerank=None) -> dict:
    """One case's row: a row per step target reached (`rank`, `held`, `children`), the `stop`
    code (None: every target was reached) and the expander's skip counts."""
    try:
        state = pool_state(case.start)
    except Exception as exc:  # noqa: BLE001 - the case is reported, not dropped
        return {"case": case.case_id, "failure": f"UNPARSED:{type(exc).__name__}"}
    steps, stop, skips = [], None, Counter()
    for smiles in case.step_targets:
        if not smiles:
            stop = "STEP_TARGET_MISSING"
            break
        try:
            target = target_from_smiles(smiles)
        except Exception:  # noqa: BLE001 - counted
            stop = "STEP_TARGET_UNPARSED"
            break
        row, state, stop = _step(expander, state, target, skips, rerank)
        if row is not None:
            steps.append(row)
        if stop is not None:
            break
    return {"case": case.case_id, "n_targets": len(case.step_targets), "steps": steps,
            "stop": stop, "skips": dict(skips)}  # fmt: skip


def _top(ranks: list, ks=KS) -> dict:
    n = len(ranks)
    return {"n": n} | {f"top{k}": sum(r is not None and r <= k for r in ranks) / max(n, 1)
                       for k in ks}  # fmt: skip


def walk_summary(rows: list[dict], ks=KS) -> dict:
    """Over the cases: the first step's top-k (the cases whose first target was ranked), the
    top-k over every step ranked, the pathways walked whole (every target reached) and those
    walked with every step within k, the steps never reached, the stop and skip counts."""
    done = [r for r in rows if "failure" not in r]
    ranked = lambda r: [s["rank"] for s in r["steps"] if not s["held"]]  # noqa: E731
    first = [r["steps"][0]["rank"] for r in done if r["steps"] and not r["steps"][0]["held"]]
    whole = [r for r in done if r["stop"] is None and r["n_targets"]]
    n = len(done)
    out = {"n": n, "failed": len(rows) - n, "first_step": _top(first, ks),
           "steps": _top([x for r in done for x in ranked(r)], ks),
           "steps_held": sum(s["held"] for r in done for s in r["steps"]),
           "steps_unreached": sum(r["n_targets"] - len(r["steps"]) for r in done),
           "walked": len(whole) / max(n, 1)}  # fmt: skip
    for k in ks:
        out[f"walked_top{k}"] = sum(all(x <= k for x in ranked(r)) for r in whole) / max(n, 1)
    skips: Counter = Counter()
    for r in done:
        skips.update(r["skips"])
    out["stops"] = dict(sorted(Counter(r["stop"] for r in done if r["stop"]).items()))
    reranked = first_step_rerank(done, ks)
    return out | ({"first_step_rerank": reranked} if reranked else {}) | {
        "skips": dict(sorted(skips.items()))}  # fmt: skip


def first_step_rerank(rows: list[dict], ks=KS) -> dict:
    """Per lambda the first step's top-k after the rerank, over the cases whose first target
    was ranked and over those whose first `top` children were all scored (`full`); {} when
    no row was reranked."""
    firsts = [r["steps"][0] for r in rows if r["steps"] and not r["steps"][0]["held"]]
    firsts = [s for s in firsts if "rerank" in s]
    if not firsts:
        return {}
    out = {}
    for lam in firsts[0]["rerank"]["by_lambda"]:
        ranks = [s["rerank"]["by_lambda"][lam] for s in firsts]
        full = [s["rerank"]["by_lambda"][lam] for s in firsts if s["rerank"]["full"]]
        out[lam] = _top(ranks, ks) | {"full": _top(full, ks)}
    return out
