"""The 3D view and depiction services (design 6.4 `view3d`), API side.

Services take an `ApiContext` and return `schemas` models (or `str` for cube files). Heavy work
(cube grids, the contact placement) runs in the worker through `ctx.interactive`, which raises
`Pending` while a task runs; the API side only reads the files the worker wrote (and checks
that every path it reads lies under `data/serve`). Geometry and depictions are computed here:
RDKit and the token files already in memory.
"""

from __future__ import annotations

from pathlib import Path

from ksmech.serve import cubes as cubemod
from ksmech.serve import depict, geometry, placement, schemas
from ksmech.serve.analysis import analysis_digest
from ksmech.serve.context import ApiContext
from ksmech.serve.contracts import PoolState
from ksmech.serve.errors import Code, ServeError

CUBE_TIMEOUT_S = 180
CONTACT_TIMEOUT_S = 300
WHICH = ("donor", "acceptor")
CUBE_MODES = ("urls", "inline")
CONTACT_NOTE = (
    "the oriented contact placement behind the floor's KS3 (kernels.main, 2.3 Angstrom): the "
    "donor fragment and the acceptor's placed cap"
)


def _tokens(ctx: ApiContext, state: PoolState):
    """The state's `ReactantTokens`; missing tokens are featurized first (through the
    analysis, which raises `Pending` while it runs and POOL_NOT_ANALYZED when it failed)."""
    try:
        return ctx.reactant_tokens(state)
    except ServeError as exc:
        missing = (exc.detail or {}).get("code") == str(Code.MISSING_TOKENS)
        if exc.code != Code.POOL_NOT_ANALYZED or not missing:
            raise
    ctx.ensure_analysis(state)
    return ctx.reactant_tokens(state)


def _digest(ctx: ApiContext, state: PoolState) -> str:
    digest = analysis_digest(ctx.loader, state)
    if digest is None:
        msg = f"pool {state.state_id} has fragments without tokens"
        detail = {"code": str(Code.MISSING_TOKENS), "keys": list(state.fragment_token_keys)}
        raise ServeError(Code.POOL_NOT_ANALYZED, msg, status=409, detail=detail)
    return digest


def _read_serve_file(ctx: ApiContext, path: str | Path) -> str:
    p = Path(path).resolve()
    if not p.is_relative_to(ctx.settings.serve.resolve()):
        raise ServeError(Code.CONSISTENCY_FAILED, f"worker file outside data/serve: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ServeError(Code.CONSISTENCY_FAILED, f"cannot read {p}: {exc}") from exc


def geometry_out(ctx: ApiContext, state_id: str) -> schemas.GeometryOut:
    """Every fragment's record geometry (needs the tokens, not the analysis)."""
    state = ctx.state(state_id)
    rt = _tokens(ctx, state)
    return schemas.GeometryOut(state_id=state.state_id, species=geometry.species_geometry(rt))


def orbital_cube_text(
    ctx: ApiContext,
    state_id: str,
    token_id: int,
    *,
    spacing: float = cubemod.DEFAULT_SPACING_ANG,
    padding: float = cubemod.DEFAULT_PADDING_ANG,
) -> str:
    """The Gaussian cube of one orbital token on its cap +- padding (the `orbital_cube`
    task; read from the cache when present)."""
    state = ctx.state(state_id)
    rt = _tokens(ctx, state)
    f, _ = geometry.orbital_position(rt, token_id)
    spacing, padding = float(spacing), float(padding)
    cubemod.check_box(spacing, padding)  # REQUEST_INVALID before any task
    digest = _digest(ctx, state)
    path = cubemod.cube_path(ctx.settings, state.state_id, int(token_id), spacing, padding, digest)
    if not path.is_file():
        payload = {"state": state.to_dict(), "token": int(token_id)}
        payload |= {"spacing": spacing, "padding": padding}
        label = f"orbital cube of token {token_id} ({rt.fragments[f].key})"
        res = ctx.interactive("orbital_cube", payload, label=label, timeout_s=CUBE_TIMEOUT_S)
        path = Path(res["path"])
    return _read_serve_file(ctx, path)


def depiction_out(
    ctx: ApiContext, state_id: str, *, keep_maps: set[int], server_svg: bool
) -> schemas.DepictionOut:
    """The 2D depiction of any saved state (RDKit only); global sites only when every
    fragment's tokens are present."""
    state = ctx.state(state_id)
    rt = None
    if all(ctx.loader.has(k) for k in state.fragment_token_keys):
        try:
            rt = ctx.reactant_tokens(state)
        except ServeError:
            rt = None  # a correspondence failure: depict without global sites
    body = depict.pool_depiction(state, rt, set(keep_maps), bool(server_svg))
    return schemas.DepictionOut(**body)


def _contact_ids(donor, donor_site, acceptor, acceptor_site) -> tuple[int, int, int, int]:
    return int(donor), int(donor_site), int(acceptor), int(acceptor_site)


def _contact_record(ctx: ApiContext, state_id: str, ids: tuple[int, int, int, int]) -> dict:
    """The contact JSON the worker wrote (the `contact` task when it is missing). The ids are
    validated against the analysis first (NOT_AN_ORBITAL_TOKEN, SITE_NOT_A_CENTRE)."""
    import json

    state = ctx.state(state_id)
    placement.check_source_version()  # PHYSICS_VERSION_MISMATCH (409) before any work
    an = ctx.ensure_analysis(state)
    rt = ctx.reactant_tokens(state)
    placement.resolve_ids(rt, an, *ids)
    digest = _digest(ctx, state)
    path = placement.contact_path(ctx.settings, state.state_id, ids, ctx.physics_fp, digest)
    if not path.is_file():
        keys = ("donor", "donor_site", "acceptor", "acceptor_site")
        payload = {"state": state.to_dict(), **dict(zip(keys, ids, strict=True))}
        label = f"contact placement {ids[0]}@{ids[1]} -> {ids[2]}@{ids[3]} in {state.pool_key}"
        res = ctx.interactive("contact", payload, label=label, timeout_s=CONTACT_TIMEOUT_S)
        path = Path(res["path"])
    return json.loads(_read_serve_file(ctx, path))


def contact_url(state_id: str, ids: tuple[int, int, int, int], which: str) -> str:
    return f"/api/contact/{state_id}/{ids[0]}/{ids[1]}/{ids[2]}/{ids[3]}/cube/{which}"


def contact_out(
    ctx: ApiContext,
    state_id: str,
    donor: int,
    donor_site: int,
    acceptor: int,
    acceptor_site: int,
    *,
    cubes: str = "urls",
) -> schemas.ContactOut:
    """The oriented placement of (donor, donor_site, acceptor, acceptor_site), verified
    (`consistent`), with both cube URLs, or both cubes inline with `cubes="inline"`."""
    if cubes not in CUBE_MODES:
        raise ServeError(Code.REQUEST_INVALID, f"cubes must be one of {CUBE_MODES}, got {cubes!r}")
    ids = _contact_ids(donor, donor_site, acceptor, acceptor_site)
    record = _contact_record(ctx, state_id, ids)
    notes = [CONTACT_NOTE]
    if record.get("checks", {}).get(str(Code.ANALYTIC_OVERLAP_DRIFT)):
        notes.append(f"{Code.ANALYTIC_OVERLAP_DRIFT}: the analytic overlap differs from S")
    inline = None
    if cubes == "inline":
        inline = {w: _read_serve_file(ctx, record["cube_paths"][w]) for w in WHICH}
    record |= {
        "cube_urls": {w: contact_url(state_id, ids, w) for w in WHICH},
        "cubes": inline,
        "iso_default": float(cubemod.ISO_DEFAULT),
        "provenance": ctx.provenance(None, notes=notes),
    }
    return schemas.ContactOut(**record)


def contact_cube_text(
    ctx: ApiContext,
    state_id: str,
    donor: int,
    donor_site: int,
    acceptor: int,
    acceptor_site: int,
    which: str,
) -> str:
    """One cube of the contact view: "donor" (the whole donor LMO) or "acceptor" (the placed,
    cap-truncated acceptor LMO), both on one grid."""
    if which not in WHICH:
        raise ServeError(Code.POOL_NOT_FOUND, f"no contact cube {which!r}", status=404)
    ids = _contact_ids(donor, donor_site, acceptor, acceptor_site)
    record = _contact_record(ctx, state_id, ids)
    return _read_serve_file(ctx, record["cube_paths"][which])
