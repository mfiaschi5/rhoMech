"""Which serve features are on, and why the others are off (spec v8 port, Phase 0).

The app was written for the spec-v3 target: one orbital-pair arrow per step, ranked by the
Klopman-Salem floor and applied with `arrows.apply_predicted`. Spec v8 predicts each elementary
step as a reaction matrix R = Delta(BE) decoded around a forming anchor bond, and
`apply_predicted_arrows` now needs a leaving-bond choice per arrow. Every feature that ranks,
applies or re-checks orbital-pair arrows is therefore off until it is ported; features that only
read molecules (tokens, geometry, orbitals, depiction, PMechDB browse, the curation database,
the molecule workbench) stay on; `workbench` is listed as a capability that is always on.

`CAPABILITIES` is evaluated once at import: {name: (enabled, reason)}, reason None when on.
`require(name)` is the FastAPI dependency of a gated route: ServeError FEATURE_DISABLED (503)
with the reason. `/api/meta` lists them (`MetaOut.capabilities`) and the UI greys the pages.
"""

from __future__ import annotations

from typing import Any

from ksmech.serve.errors import Code, ServeError

_ARROW_TARGET = (
    "spec v8 predicts each step as a reaction matrix R = Delta(BE) decoded around a forming "
    "anchor bond; "
)
_ANALYSIS = (
    _ARROW_TARGET + "this view ranks spec-v3 orbital-pair arrow candidates from the oriented "
    "pair analysis (the -(KS3 + KS2) floor), which is not the v8 target"
)
_DISABLED: dict[str, str] = {
    "predict": _ARROW_TARGET
    + "the floor, GBDT and stub predictors rank single orbital-pair arrows, and the network "
    "expansion applies them with apply_predicted_arrows, which now needs a leaving-bond choice "
    "per arrow",
    "correct": _ARROW_TARGET
    + "an override applies orbital-pair arrows with apply_predicted_arrows, which now needs a "
    "leaving-bond choice per arrow",
    "candidates": _ANALYSIS,
    "physics": _ANALYSIS,
    "site_pairs": _ANALYSIS,
    "curate_edit": "curated mechanisms are pathways of predict jobs (predict is off); the "
    "curation database is read-only until predict is ported to spec v8",
    "curate_recheck": _ARROW_TARGET
    + "a re-check re-runs the predictors and apply_predicted_arrows on spec-v3 orbital-pair "
    "arrows",
    "export": "the user_curated source (data.user_curated, Source.USER_CURATED, curated steps "
    "pinned to train) was not ported: it conflicts with the current data layer",
}
ORDER = (
    "predict",
    "correct",
    "candidates",
    "physics",
    "site_pairs",
    "contact",
    "curate_edit",
    "curate_recheck",
    "export",
    "workbench",
)
ALWAYS_ON = frozenset({"workbench"})  # display-only features that read molecules


def _contact() -> tuple[bool, str | None]:
    """The placement replay is pinned to one oriented version (`placement`); off otherwise."""
    from ksmech.serve.placement import PINNED_ORIENTED_VERSION
    from ksmech.serve.upstream import ORIENTED_SOURCE
    from ksmech.serve.versions import source_constant

    try:
        version = source_constant(ORIENTED_SOURCE, "ORIENTED_VERSION")
    except LookupError:
        version = None
    if version == PINNED_ORIENTED_VERSION and "contact" not in _DISABLED:
        return True, None
    return False, (
        f"the contact placement replay is pinned to ORIENTED_VERSION {PINNED_ORIENTED_VERSION}; "
        f"{ORIENTED_SOURCE} has {version}; " + _ANALYSIS
    )


def _evaluate() -> dict[str, tuple[bool, str | None]]:
    out: dict[str, tuple[bool, str | None]] = {}
    for name in ORDER:
        if name in ALWAYS_ON:
            out[name] = (True, None)
        else:
            out[name] = _contact() if name == "contact" else (False, _DISABLED[name])
    return out


CAPABILITIES: dict[str, tuple[bool, str | None]] = _evaluate()


def enabled(name: str) -> bool:
    return CAPABILITIES[name][0]


def check(name: str) -> None:
    """Raise FEATURE_DISABLED (503) when `name` is off."""
    on, reason = CAPABILITIES[name]
    if not on:
        detail = {"capability": name, "reason": reason}
        raise ServeError(Code.FEATURE_DISABLED, f"{name} is disabled: {reason}", detail=detail)


def require(name: str) -> Any:
    """`Depends` for `dependencies=[require(name)]`: runs before the body and the service."""
    from fastapi import Depends

    if name not in CAPABILITIES:
        raise KeyError(f"unknown capability {name!r}")

    def _require() -> None:
        check(name)

    return Depends(_require)


def as_meta() -> dict[str, dict[str, Any]]:
    """`MetaOut.capabilities`: {name: {enabled, reason}}."""
    return {n: {"enabled": on, "reason": why} for n, (on, why) in CAPABILITIES.items()}
