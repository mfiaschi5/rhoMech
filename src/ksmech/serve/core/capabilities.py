"""Which serve features are on, and why the others are off (spec v9).

Spec v9 predicts each elementary step as a reaction matrix R = Delta(BE) decoded around a
forming anchor bond and reads the curved arrows back from R (`serve.science.reading`). The predict,
correct and curation features work on that target: network edges carry R, its anchor, its
arrow reading and the orbital readings of its forming bonds; `export` is a PMechDB-format
file download of the curated steps, never training data. The spec-v3 panels that ranked
single orbital-pair arrows from the Klopman-Salem contact terms (`candidates`, `physics`)
have no v9 counterpart yet and stay off; the contact placement and the site-pair view were removed
with the contact model (archived upstream on `archive/stage-b`). `workbench` is display-only
and always on.

`embedding` (the trace of one pool through a trained checkpoint, `serve.embedding.service`) also
needs
the machine: torch and the model code importable (checked without importing them) and one
usable local checkpoint (`serve.predict.models.embedding_status`), so it is evaluated per request
against the app's settings.

`CAPABILITIES` is evaluated once at import: {name: (enabled, reason)}, reason None when on;
for `embedding` it holds the import-time part only. `status(name, settings)` adds the
settings-dependent part. `require(name)` is the FastAPI dependency of a gated route:
ServeError FEATURE_DISABLED (503) with the reason. `/api/meta` lists them
(`MetaOut.capabilities`) and the UI greys the pages.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ksmech.serve.core.errors import Code, ServeError

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

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
    "embedding",
)
ALWAYS_ON = frozenset({"workbench"})  # display-only features that read molecules


def _embedding_imports() -> tuple[bool, str | None]:
    from ksmech.serve.predict.models import missing_modules

    missing = missing_modules()
    return (not missing, f"not importable here: {', '.join(missing)}" if missing else None)


def _embedding_settings(settings: ServeSettings) -> tuple[bool, str | None]:
    from ksmech.serve.predict.models import embedding_status

    return embedding_status(settings)


_STATIC: dict[str, Callable[[], tuple[bool, str | None]]] = {"embedding": _embedding_imports}
_DYNAMIC: dict[str, Callable[[ServeSettings], tuple[bool, str | None]]] = {
    "embedding": _embedding_settings
}


def _evaluate() -> dict[str, tuple[bool, str | None]]:
    out: dict[str, tuple[bool, str | None]] = {}
    for name in ORDER:
        if name in _STATIC:
            out[name] = _STATIC[name]()
        elif name in ALWAYS_ON or name not in _DISABLED:
            out[name] = (True, None)
        else:
            out[name] = (False, _DISABLED[name])
    return out


CAPABILITIES: dict[str, tuple[bool, str | None]] = _evaluate()


def status(name: str, settings: ServeSettings | None = None) -> tuple[bool, str | None]:
    """(enabled, reason) of `name`: `CAPABILITIES`, then, with settings, the part that depends
    on them (the local checkpoints of `embedding`)."""
    on, reason = CAPABILITIES[name]
    if on and settings is not None and name in _DYNAMIC:
        return _DYNAMIC[name](settings)
    return on, reason


def enabled(name: str, settings: ServeSettings | None = None) -> bool:
    return status(name, settings)[0]


def check(name: str, settings: ServeSettings | None = None) -> None:
    """Raise FEATURE_DISABLED (503) when `name` is off."""
    on, reason = status(name, settings)
    if not on:
        detail = {"capability": name, "reason": reason}
        raise ServeError(Code.FEATURE_DISABLED, f"{name} is disabled: {reason}", detail=detail)


def require(name: str) -> Any:
    """`Depends` for `dependencies=[require(name)]`: runs before the body and the service,
    against the settings of the app that serves the request."""
    from fastapi import Depends, Request

    if name not in CAPABILITIES:
        raise KeyError(f"unknown capability {name!r}")

    def _require(request) -> None:
        services = getattr(request.app.state, "services", None)
        check(name, getattr(services, "settings", None))

    _require.__annotations__ = {"request": Request, "return": None}  # resolved, not a string
    return Depends(_require)


def as_meta(settings: ServeSettings | None = None) -> dict[str, dict[str, Any]]:
    """`MetaOut.capabilities`: {name: {enabled, reason}}."""
    out = {}
    for name in CAPABILITIES:
        on, why = status(name, settings)
        out[name] = {"enabled": on, "reason": why}
    return out
