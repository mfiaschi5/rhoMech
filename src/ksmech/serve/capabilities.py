"""Which serve features are on, and why the others are off (spec v9).

Spec v9 predicts each elementary step as a reaction matrix R = Delta(BE) decoded around a
forming anchor bond and reads the curved arrows back from R (`serve.reading`). The predict,
correct and curation features work on that target: network edges carry R, its anchor, its
arrow reading and the orbital readings of its forming bonds; `export` is a PMechDB-format
file download of the curated steps, never training data. The spec-v3 panels that ranked
single orbital-pair arrows from the Klopman-Salem contact terms (`candidates`, `physics`)
have no v9 counterpart yet and stay off; the contact placement and the site-pair view were removed
with the contact model (archived upstream on `archive/stage-b`). `workbench` is display-only
and always on.

`CAPABILITIES` is evaluated once at import: {name: (enabled, reason)}, reason None when on.
`require(name)` is the FastAPI dependency of a gated route: ServeError FEATURE_DISABLED (503)
with the reason. `/api/meta` lists them (`MetaOut.capabilities`) and the UI greys the pages.
"""

from __future__ import annotations

from typing import Any

from ksmech.serve.errors import Code, ServeError

_V3_PANEL = (
    "spec v9 predicts each step as a reaction matrix R; this panel ranked spec-v3 single "
    "orbital-pair arrows by the Klopman-Salem contact terms, which v9 does not compute. The "
    "geometry-free physics of a step (the pooled gap factor of each forming bond, its dominant "
    "orbital pair and canonical MO pair) is on every edge and in POST /api/step/check"
)
_DISABLED: dict[str, str] = {
    "candidates": _V3_PANEL,
    "physics": _V3_PANEL,
}
ORDER = (
    "predict",
    "correct",
    "candidates",
    "physics",
    "curate_edit",
    "curate_recheck",
    "export",
    "workbench",
)
ALWAYS_ON = frozenset({"workbench"})  # display-only features that read molecules


def _evaluate() -> dict[str, tuple[bool, str | None]]:
    out: dict[str, tuple[bool, str | None]] = {}
    for name in ORDER:
        if name in ALWAYS_ON or name not in _DISABLED:
            out[name] = (True, None)
        else:
            out[name] = (False, _DISABLED[name])
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
