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
usable local checkpoint (`serve.ml.checkpoints.embedding_status`), so it is evaluated per request
against the app's settings.

`browse` (the PMechDB browser), `curate` (reading curated mechanisms) and `ml` (training runs and
the physics / eval reports) are always on in the `lab` profile. A profile (`PROFILES`,
`ServeSettings.profile`) is the set of capabilities a deployment admits: `public` drops every
route that reads the dataset or the lab's work (browse, curate and its edit / recheck / export,
embedding, which traces stored dataset samples, and ml), and those answer FEATURE_DISABLED with
the reason "not in the public profile".

`CAPABILITIES` is evaluated once at import: {name: (enabled, reason)}, reason None when on;
for `embedding` it holds the import-time part only. `status(name, settings)` adds the profile
and the settings-dependent part. `require(name)` is the FastAPI dependency of a gated route:
ServeError FEATURE_DISABLED (503) with the reason, then the least role of the capability
(`core.auth.CAPABILITY_ROLES`, or `role=`). `admit(name)` checks the profile and the role only
(a router of stored results that stays readable while the feature itself is off).
A POST through `require` is then charged to the principal's job quota when the app installed
one (`app.state.charge`, `serve.accounts.quotas`: predict and embedding jobs; QUOTA_EXCEEDED).
`/api/meta` lists them (`MetaOut.capabilities`) and the UI greys the pages.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ksmech.serve.core.auth import CAPABILITY_ROLES, Role, authorize, principal_of
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
    "browse",
    "curate",
    "ml",
)
ALWAYS_ON = frozenset({"workbench"})  # display-only features that read molecules
PROFILES: dict[str, frozenset[str]] = {
    "lab": frozenset(ORDER),
    "public": frozenset({"predict", "correct", "candidates", "physics", "workbench"}),
}
"""The capabilities each profile admits (`ServeSettings.profile`)."""


def _embedding_imports() -> tuple[bool, str | None]:
    from ksmech.serve.ml.checkpoints import missing_modules

    missing = missing_modules()
    return (not missing, f"not importable here: {', '.join(missing)}" if missing else None)


def _embedding_settings(settings: ServeSettings) -> tuple[bool, str | None]:
    from ksmech.serve.ml.checkpoints import embedding_status

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


def in_profile(name: str, settings: ServeSettings | None) -> tuple[bool, str | None]:
    """Whether the settings' profile admits `name` (lab when there are no settings)."""
    profile = getattr(settings, "profile", "lab")
    if name in PROFILES[profile]:
        return True, None
    return False, f"not in the {profile} profile"


def status(name: str, settings: ServeSettings | None = None) -> tuple[bool, str | None]:
    """(enabled, reason) of `name`: the profile, `CAPABILITIES`, then, with settings, the part
    that depends on them (the local checkpoints of `embedding`)."""
    admitted, why = in_profile(name, settings)
    if not admitted:
        return False, why
    on, reason = CAPABILITIES[name]
    if on and settings is not None and name in _DYNAMIC:
        return _DYNAMIC[name](settings)
    return on, reason


def enabled(name: str, settings: ServeSettings | None = None) -> bool:
    return status(name, settings)[0]


def _raise_off(name: str, reason: str | None) -> None:
    detail = {"capability": name, "reason": reason}
    raise ServeError(Code.FEATURE_DISABLED, f"{name} is disabled: {reason}", detail=detail)


def check(name: str, settings: ServeSettings | None = None) -> None:
    """Raise FEATURE_DISABLED (503) when `name` is off."""
    on, reason = status(name, settings)
    if not on:
        _raise_off(name, reason)


def _dependency(name: str, role: Role | str | None, full: bool) -> Any:
    from fastapi import Depends, Request

    if name not in CAPABILITIES:
        raise KeyError(f"unknown capability {name!r}")
    wanted = Role(role) if role is not None else CAPABILITY_ROLES[name]

    def _require(request) -> None:
        services = getattr(request.app.state, "services", None)
        settings = getattr(services, "settings", None)
        if full:
            check(name, settings)
        else:
            admitted, why = in_profile(name, settings)
            if not admitted:
                _raise_off(name, why)
        principal = principal_of(request)
        authorize(principal, wanted, name)
        charge = getattr(request.app.state, "charge", None)
        if full and charge is not None and request.method == "POST":
            charge(principal, name)

    _require.__annotations__ = {"request": Request, "return": None}  # resolved, not a string
    return Depends(_require)


def require(name: str, *, role: Role | str | None = None) -> Any:
    """`Depends` for `dependencies=[require(name)]`: runs before the body and the service,
    against the settings of the app that serves the request: FEATURE_DISABLED (503) when the
    capability is off, then the least role (`CAPABILITY_ROLES[name]` unless `role`)."""
    return _dependency(name, role, full=True)


def admit(name: str, *, role: Role | str = Role.READER) -> Any:
    """`Depends` of a whole router: FEATURE_DISABLED when the profile does not admit `name`,
    then `role`; the capability's own state is left to the routes that need it."""
    return _dependency(name, role, full=False)


def as_meta(settings: ServeSettings | None = None) -> dict[str, dict[str, Any]]:
    """`MetaOut.capabilities`: {name: {enabled, reason}}."""
    out = {}
    for name in CAPABILITIES:
        on, why = status(name, settings)
        out[name] = {"enabled": on, "reason": why}
    return out
