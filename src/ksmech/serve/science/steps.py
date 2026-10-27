"""The API side of a user's step: `POST /api/step/check` (live validation of an edited R on a
pool state, no job needed) and the record a user step adds to a network (`JobService.override`).

Both build `edgestep.step_record` on the API side: the product and the issues need RDKit only;
the arrow reading and the orbital readings need the pool's analysis (the pooled gap, computed
once in the worker and read from its npz) and tokens. `check_step` asks for the analysis with a
short wait (`CHECK_WAIT_S`): when it is not ready the check answers anyway, with the arrows read
without the gap and `readings_status` "pending" (the analysis keeps running; a later check has
the readings) or "unavailable" (the pool cannot be analyzed). A user step waits as long as any
interactive call (`settings.interactive_wait_s`, then 202 Pending).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ksmech.serve.core.errors import Pending, ServeError
from ksmech.serve.science import edgestep, stepapply

if TYPE_CHECKING:
    from ksmech.serve.core.contracts import PoolState
    from ksmech.serve.runtime.context import ApiContext

log = logging.getLogger(__name__)

CHECK_WAIT_S = 1.0
STEP_INVALID = "STEP_INVALID"


def _analysis(ctx: ApiContext, state: PoolState, wait_s: float | None):
    """(pooled gap, tokens, status) of the state: status ready | pending | unavailable."""
    try:
        an = ctx.ensure_analysis(state, wait_s=wait_s)
        return an.pg, ctx.reactant_tokens(state), "ready"
    except Pending:
        if wait_s is None:
            raise
        return None, None, "pending"
    except ServeError as exc:
        if str(exc.code) == "TOKENS_CHANGED":
            raise
        log.info("no readings for %s: %s %s", state.state_id, exc.code, exc.message)
        return None, None, "unavailable"


def _occupations(ctx: ApiContext):
    fn = getattr(ctx.featurizer, "occupations", None)
    return fn if callable(fn) else None


def check_step(ctx: ApiContext, state: PoolState, matrix: Any, *, wait_s=CHECK_WAIT_S) -> dict:
    """`edgestep.step_record` of R on `state`, with `readings_status` and, when the pooled
    gap is known, the user step's score |G_atom| of its anchor. `wait_s=None` waits the
    interactive time and lets `Pending` (202) through."""
    R = stepapply.matrix_of(matrix)
    pg, rt, status = _analysis(ctx, state, wait_s)
    rec = edgestep.step_record(state, R, pg=pg, rt=rt, occupations=_occupations(ctx))
    rec["readings_status"] = status
    rec["score"] = None
    anchor = rec.get("anchor")
    if anchor is not None and pg is not None and rt is not None:
        g = rt.global_atom
        rec["score"] = abs(float(pg.g_atom[g[anchor["i"]], g[anchor["j"]]]))
    return rec


def check_out(rec: dict) -> dict:
    """`StepCheckOut` fields of a step record."""
    child = rec.get("product_state")
    keys = ("ok", "issues", "product_smiles", "sigma_keys", "formal_charges", "arrows")
    out = {k: rec[k] for k in keys}
    out |= {
        "readings": rec["readings"],
        "anchor": rec["anchor"],
        "product_state_id": None if child is None or not rec["ok"] else child.state_id,
        "pmechdb_code": rec.get("pmechdb_code"),
        "readings_status": rec["readings_status"],
    }
    return out


def step_check(ctx: ApiContext, state_id: str, matrix: Any) -> dict:
    """POST /api/step/check. The product state of a valid R is saved, so the client can
    depict it (`/api/pool/{product_state_id}/depiction`)."""
    from ksmech.serve.chem.pools import save_state

    state = ctx.state(state_id)
    rec = check_step(ctx, state, matrix)
    if rec["ok"] and rec.get("product_state") is not None:
        save_state(rec["product_state"], ctx.settings)
    return check_out(rec)


def invalid_step(rec: dict) -> ServeError:
    """422 STEP_INVALID with the issues of a step record."""
    codes = ", ".join(x["code"] for x in rec["issues"]) or "unknown"
    detail = {"issues": rec["issues"]}
    return ServeError(
        STEP_INVALID, f"the matrix is no valid step: {codes}", status=422, detail=detail
    )
