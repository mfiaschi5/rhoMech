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
"""

from __future__ import annotations

from collections import Counter

from ksmech.network.benchmarks import Case
from ksmech.network.pools import pool_state
from ksmech.network.target import EXACT, arrival, target_from_smiles

KS = (1, 3, 5)


def _holds(target, state) -> bool:
    return arrival(target, state.species, state.skeletons, (EXACT,)) is not None


def ranked_children(expander, expansion) -> list:
    """The distinct child states of an expansion, the most probable first."""
    seen, out = set(), []
    for p in sorted(expansion.proposals, key=lambda p: -p.logp):
        key = expander.key(p.child)
        if key not in seen:
            seen.add(key)
            out.append(p.child)
    return out


def _step(expander, state, target, skips: Counter):
    """(the step's row, the state the walk moves to or None, the stop code or None)."""
    if _holds(target, state):
        return {"rank": None, "held": True}, state, None
    ex = expander.expand(state)
    skips.update(ex.skips)
    if ex.failure:
        return None, None, f"EXPAND_FAILED:{ex.failure}"
    children = ranked_children(expander, ex)
    row = {"rank": None, "held": False, "children": len(children)}
    for rank, child in enumerate(children, start=1):
        if _holds(target, child):
            return row | {"rank": rank}, child, None
    return row, None, "NOT_PROPOSED"


def walk_case(expander, case: Case) -> dict:
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
        row, state, stop = _step(expander, state, target, skips)
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
    return out | {"skips": dict(sorted(skips.items()))}
