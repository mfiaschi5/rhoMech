"""The chemistry a network is built with (design 2.3): `Chemistry` is the protocol `network.py`
and the jobs use; `MolecularChemistry` serves the real, cached and synthetic modes through the
scheduler (every heavy step runs in the compute worker: featurize, expand). A user's step is
applied on the API side (`serve.science.steps`), not through the chemistry.

Persistence contract: every `PoolState` returned by `root` or `expand` has been saved
with `pools.save_state` (the worker tasks save what they return); `MolecularChemistry` checks
that each returned state file exists, and a missing one is CONSISTENCY_FAILED.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from ksmech.serve.chem.pools import SpeciesInput, preflight_inputs, save_state, state_path
from ksmech.serve.compute.worker import Priority, Scheduler, TaskResult
from ksmech.serve.core.contracts import ExpandRequest, Expansion, JobContext, PoolState
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.settings import ServeSettings

if TYPE_CHECKING:
    from ksmech.serve.chem.featurize import Featurizer

log = logging.getLogger(__name__)

EXPAND_TIMEOUT_S = 900
FEATURIZE_SLACK_S = 120
QM_MEDIAN_S = 3.4  # per molecule, from the PMechDB QM manifest (p90 41 s)
QM_MEDIAN_ATOMS = 20
SYNTHETIC_S = 1.0


class Chemistry(Protocol):
    mode: str  # real | cached | synthetic | stub

    def root(self, inputs: list[SpeciesInput]) -> PoolState: ...  # API side, cheap; raises

    def expand(self, state: PoolState, req: ExpandRequest, ctx: JobContext) -> Expansion: ...


def _n_atoms(key: str) -> int | None:
    from ksmech.qm.embed import prepare_mol
    from ksmech.qm.records import QMError

    try:
        return int(prepare_mol(key).GetNumAtoms())
    except (QMError, ValueError):
        return None


def qm_eta_s(featurizer: Featurizer, keys: Sequence[str]) -> float:
    """A rough time estimate for featurizing `keys`: the median QM time scaled by the square of
    the atom count (synthetic EHT: about a second each)."""
    from ksmech.serve.chem.synthetic import SYNTHETIC_METHOD

    if featurizer.method == SYNTHETIC_METHOD:
        return SYNTHETIC_S * len(keys)
    eta = 0.0
    for key in keys:
        n = _n_atoms(key) or QM_MEDIAN_ATOMS
        eta += QM_MEDIAN_S * max(0.25, (n / QM_MEDIAN_ATOMS) ** 2)
    return eta


class MolecularChemistry:
    """Real, cached and synthetic chemistry behind the scheduler."""

    def __init__(self, settings: ServeSettings, scheduler: Scheduler, featurizer: Featurizer):
        self.settings = settings
        self.scheduler = scheduler
        self.featurizer = featurizer
        self.loader = featurizer.loader()
        self.mode = str(featurizer.mode)

    def root(self, inputs: list[SpeciesInput]) -> PoolState:
        """The preflighted, saved root state; the first input error raises (422)."""
        pf = preflight_inputs(inputs, self.featurizer, self.settings.limits)
        if pf.error is not None:
            err = pf.error
            raise ServeError(
                err.code, err.message, status=422, upstream=err.upstream, detail=err.detail
            )
        if pf.state is None:  # cannot happen: no error means a state
            raise ServeError(Code.CONSISTENCY_FAILED, "preflight gave no state", status=500)
        save_state(pf.state, self.settings)
        return pf.state

    def _featurize(self, state: PoolState, ctx: JobContext) -> Expansion | None:
        """Run one `featurize` task per distinct fragment key the loader lacks, in fragment
        order. Returns the failed expansion of the first failure, or None."""
        from ksmech.serve.chem.featurize import featurize_label

        missing = [k for k in dict.fromkeys(state.fragment_token_keys) if not self.loader.has(k)]
        timeout = float(self.settings.qm_timeout_s + FEATURIZE_SLACK_S)
        for done, key in enumerate(missing):
            if ctx.cancelled():
                return Expansion.failed(Code.CANCELLED, f"cancelled before featurizing {key}")
            label = featurize_label(self.featurizer, key)
            eta = qm_eta_s(self.featurizer, missing[done:])
            ctx.progress(phase="featurize", message=label, eta_s=eta)
            r = self.scheduler.run(
                "featurize",
                {"key": key},
                priority=Priority.JOB,
                timeout_s=timeout,
                owner=ctx.job_id,
                label=label,
            )
            self._forward_events(r, ctx)
            if not r.ok:
                code = str(r.code or Code.FEATURIZATION_FAILED)
                log.warning("featurize %s failed [%s/%s] %s", key, code, r.upstream, r.detail)
                return Expansion.failed(code, f"{key}: {r.detail}", r.upstream)
        return None

    def _forward_events(self, r: TaskResult, ctx: JobContext) -> None:
        """Reason-coded worker events: a skipped stale token file (TOKENS_STALE) is a job skip;
        anything else (ORIENTED_CACHE_WIPED, which the scheduler acts on) is logged."""
        for ev in r.events or []:
            code = str(ev.get("code") or "")
            rest = {k: v for k, v in ev.items() if k not in ("code", "detail")}
            if code == Code.TOKENS_STALE:
                ctx.skip(code, str(ev.get("detail") or "stale token file skipped"), **rest)
            else:
                log.warning(
                    "job %s: worker event %s %s %s", ctx.job_id, code, ev.get("detail"), rest
                )

    def _missing_states(self, states: Sequence[PoolState]) -> list[str]:
        return [s.state_id for s in states if not state_path(s.state_id, self.settings).exists()]

    def expand(self, state: PoolState, req: ExpandRequest, ctx: JobContext) -> Expansion:
        """Featurize, then the `expand` task (which builds or loads the analysis). Never
        raises: every failure is a failed expansion with its reason code."""
        failed = self._featurize(state, ctx)
        if failed is not None:
            return failed
        if ctx.cancelled():
            return Expansion.failed(Code.CANCELLED, f"cancelled before expanding {state.pool_key}")
        label = f"analysis and expansion of {state.pool_key}"
        ctx.progress(phase="expand", message=label)
        payload = {"state": state.to_dict(), "req": req.to_dict()}
        r = self.scheduler.run(
            "expand",
            payload,
            priority=Priority.JOB,
            timeout_s=EXPAND_TIMEOUT_S,
            owner=ctx.job_id,
            label=label,
        )
        self._forward_events(r, ctx)
        if not r.ok:
            return Expansion.failed(str(r.code or Code.UNEXPECTED), r.detail, r.upstream)
        try:
            ex = Expansion.from_dict(r.result or {})
        except (KeyError, TypeError, ValueError) as exc:
            detail = f"malformed expand result: {type(exc).__name__}: {exc}"
            return Expansion.failed(Code.CONSISTENCY_FAILED, detail)
        missing = self._missing_states([ch.state for ch in ex.children])
        if missing:
            return Expansion.failed(Code.CONSISTENCY_FAILED, f"state not persisted: {missing[0]}")
        return ex
