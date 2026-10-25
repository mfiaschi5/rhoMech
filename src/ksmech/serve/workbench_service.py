"""Molecule workbench services, API side (design 6.4 style): geometry, properties, orbital
sets and cubes of a registered molecule (`molecule.py`). Display only.

Services never compute: a molecule without a QM record is featurized first (the `featurize`
task; `Pending` while it runs, its failure passes through and is recorded in the registry),
heavy work runs in the worker (`wb_orbital_set`, `wb_cube`, `wb_properties`) through
`ctx.interactive` (202 Pending while it runs), and the API process only reads the files the
worker wrote, after checking that each lies under `data/serve`. Each cache path is computed
here first, so a cached result never touches the scheduler. The geometry is read with
`np.load` from the QM record (never `load_record`, which probes tblite).

The 2D depiction (`depict2d_out`, RDKit only, in the API process) reads the record geometry
the same way for a 3D->2D drawing and never featurizes.

The point group (`symmetry_out`, numpy in the API process, `symmetry.py`) reads the record
geometry like /geometry (featurizing first when the record is missing) and is memoized per
(molecule, tolerance, QM record digest).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ksmech.serve import (
    densities,
    depiction2d,
    geometry,
    molecule,
    orbital_sets,
    properties,
    symmetry,
)
from ksmech.serve import schemas as S
from ksmech.serve.context import ApiContext
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.featurize import qm_npz_path
from ksmech.serve.molecule import file_digest

WB_TIMEOUT_S = 300
FRAME_NOTE = "the QM record's own frame (Angstrom), record atom order"


@dataclass(frozen=True)
class CubeText:
    text: str
    meta: dict
    filename: str


def _read_serve_text(ctx: ApiContext, path: str | Path) -> str:
    p = Path(path).resolve()
    if not p.is_relative_to(ctx.settings.serve.resolve()):
        raise ServeError(Code.CONSISTENCY_FAILED, f"worker file outside data/serve: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ServeError(Code.CONSISTENCY_FAILED, f"cannot read {p}: {exc}") from exc


def _read_json(ctx: ApiContext, path: str | Path) -> dict:
    try:
        return json.loads(_read_serve_text(ctx, path))
    except ValueError as exc:
        raise ServeError(Code.CONSISTENCY_FAILED, f"unreadable worker file {path}: {exc}") from exc


def _qm(ctx: ApiContext, mid: str) -> tuple[dict, Path]:
    """(registry entry, QM record path); featurizes first when the record is missing. A
    featurize failure recorded for this method is raised again without a new run (POST the
    molecule with `retry` to run it again)."""
    entry = molecule.load_entry(ctx.settings, mid)
    method = ctx.featurizer.method
    path = qm_npz_path(ctx.settings, entry["key"], method)
    if path is None:
        failure = molecule.current_failure(ctx, entry)
        if failure is not None:
            detail = {"molecule_id": mid, "failure": failure}
            msg = f"featurize of {entry['key']} failed: {failure.get('message')}"
            raise ServeError(failure["code"], msg, upstream=failure.get("upstream"), detail=detail)
        molecule.run_featurize(ctx, entry)
        path = qm_npz_path(ctx.settings, entry["key"], method)
    if path is None:
        msg = f"no QM record for {entry['key']} after featurize"
        raise ServeError(Code.MOLECULE_NOT_READY, msg, status=409)
    return entry, path


def _task(ctx: ApiContext, name: str, payload: dict, label: str, path: Path) -> Path:
    """`path` when the worker already wrote it, else the task's (Pending while it runs)."""
    if path.is_file():
        return path
    res = ctx.interactive(name, payload, label=label, timeout_s=WB_TIMEOUT_S)
    return Path(res["path"])


# ---- geometry


def geometry_body(ctx: ApiContext, mid: str, key: str, qm_path: Path) -> dict:
    """`MoleculeGeometryOut` fields from the QM record's arrays (np.load only)."""
    with np.load(qm_path, allow_pickle=False) as z:
        zs = np.asarray(z["atomic_numbers"], dtype=int)
        xyz = np.asarray(z["coords_ang"], dtype=float).reshape(-1, 3)
        charge, unpaired = int(z["charge"]), int(z["unpaired"])
    record = geometry.record_mol(key)
    if record.GetNumAtoms() != len(zs):
        msg = f"the record of {key} has {len(zs)} atoms, prepare_mol {record.GetNumAtoms()}"
        raise ServeError(Code.CONSISTENCY_FAILED, msg)
    atoms = [
        {"global_site": i, "record_index": i, "map": None, "Z": int(zi), "x": float(p[0])}
        | {"symbol": a.GetSymbol(), "y": float(p[1]), "z": float(p[2])}
        for i, (zi, p, a) in enumerate(zip(zs, xyz, record.GetAtoms(), strict=True))
    ]
    comment = f"{key} charge={charge} mult={unpaired + 1} method={ctx.featurizer.method}"
    return {
        "molecule_id": mid,
        "key": key,
        "charge": charge,
        "multiplicity": unpaired + 1,
        "atoms": atoms,
        "bonds": geometry.bond_rows(record),
        "xyz": geometry.xyz_text(zs, xyz, comment),
        "molblock": geometry.molblock_with_coords(key, xyz),
        "frame_note": FRAME_NOTE,
    }


def geometry_out(ctx: ApiContext, mid: str) -> S.MoleculeGeometryOut:
    """GET /molecule/{id}/geometry: the record geometry as atoms, bonds, xyz and molblock."""
    entry, qm_path = _qm(ctx, mid)
    return S.MoleculeGeometryOut(**geometry_body(ctx, mid, entry["key"], qm_path))


# ---- properties


def properties_out(ctx: ApiContext, mid: str) -> S.MoleculePropertiesOut:
    """GET /molecule/{id}/properties: global, atom and bond properties (`wb_properties`)."""
    entry, qm_path = _qm(ctx, mid)
    key, method = entry["key"], ctx.featurizer.method
    path = properties.properties_path(
        ctx.settings, key, method, file_digest(qm_path), ctx.loader.digest(key)
    )
    label = f"workbench properties of {key}"
    body = _read_json(ctx, _task(ctx, "wb_properties", {"key": key}, label, path))
    body["geometry"] = geometry_body(ctx, mid, key, qm_path)
    return S.MoleculePropertiesOut(molecule_id=mid, **body)


# ---- orbital sets


def orbitals_out(
    ctx: ApiContext, mid: str, set_: str | None = None, start: str | None = None
) -> S.OrbitalSetOut:
    """GET /molecule/{id}/orbitals?set=&start=: the rows of one orbital set (`wb_orbital_set`)."""
    set_, start = orbital_sets.check_set(set_, start)  # REQUEST_INVALID before any work
    entry, qm_path = _qm(ctx, mid)
    key = entry["key"]
    token_digest = ctx.loader.digest(key) if orbital_sets.is_model_set(set_, start) else None
    path = orbital_sets.rows_path(
        ctx.settings, key, ctx.featurizer.method, file_digest(qm_path), set_, start, token_digest
    )
    payload = {"key": key, "set": set_, "start": start}
    label = f"{set_}{'' if start is None else '/' + start} orbitals of {key}"
    body = _read_json(ctx, _task(ctx, "wb_orbital_set", payload, label, path))
    return S.OrbitalSetOut(molecule_id=mid, **body)


# ---- cubes


def _check_index(ctx: ApiContext, key: str, req: densities.CubeRequest) -> None:
    """ORBITAL_OUT_OF_RANGE before any task: every set has one orbital per canonical MO."""
    if req.index is None:
        return
    occ = ctx.featurizer.occupations(key)
    n = None if occ is None else len(occ)
    if n is not None and not 0 <= req.index < n:
        msg = f"orbital {req.index} out of range: {key} has {n} orbitals (0..{n - 1})"
        raise ServeError(Code.ORBITAL_OUT_OF_RANGE, msg, detail={"index": req.index, "n": n})


def _cube_name(mid: str, req: densities.CubeRequest) -> str:
    parts = [mid, req.kind]
    if req.index is not None:
        parts += [str(req.set), *([req.start] if req.start else []), str(req.index)]
    return "-".join(parts) + ".cube"


def cube_out(
    ctx: ApiContext,
    mid: str,
    kind: str,
    *,
    set_: str | None = None,
    start: str | None = None,
    index: int | None = None,
    quality: str | None = None,
    spacing: float | None = None,
    padding: float | None = None,
) -> CubeText:
    """GET /molecule/{id}/cube: one surface on the whole-molecule grid (`wb_cube`)."""
    req = densities.cube_request(kind, set_, start, index, quality, spacing, padding)
    entry, qm_path = _qm(ctx, mid)
    key = entry["key"]
    _check_index(ctx, key, req)
    model = req.kind in densities.ORBITAL_KINDS and orbital_sets.is_model_set(req.set, req.start)
    token_digest = ctx.loader.digest(key) if model else None
    path, meta_path = densities.cube_paths(
        ctx.settings, key, ctx.featurizer.method, file_digest(qm_path), req, token_digest
    )
    if not (path.is_file() and meta_path.is_file()):
        payload = {"key": key, "request": req.to_payload()}
        label = f"{req.kind} cube of {key}"
        res = ctx.interactive("wb_cube", payload, label=label, timeout_s=WB_TIMEOUT_S)
        path, meta_path = Path(res["path"]), Path(res["meta"])
    return CubeText(_read_serve_text(ctx, path), _read_json(ctx, meta_path), _cube_name(mid, req))


def cube_headers(cube: CubeText, download: bool) -> dict[str, str]:
    """Content-Disposition plus the metadata the viewer shows next to the surface."""
    meta = cube.meta
    integral = meta.get("integral")
    expected = meta.get("expected")
    disposition = "attachment" if download else "inline"
    return {
        "Content-Disposition": f"{disposition}; filename={cube.filename}",
        "X-Cube-Kind": str(meta.get("kind")),
        "X-Cube-Units": str(meta.get("units")),
        "X-Cube-Integral": "none" if integral is None else f"{float(integral):.6g}",
        "X-Cube-Expected": "none" if expected is None else f"{float(expected):.6g}",
        "X-Cube-Spacing": f"{float(meta.get('spacing_ang', 0.0)):.6g}",
        "Access-Control-Expose-Headers": "X-Cube-Kind, X-Cube-Units, X-Cube-Integral, "
        "X-Cube-Expected, X-Cube-Spacing",
    }


# ---- 2D depiction


def _record_xyz(ctx: ApiContext, mid: str, drawn) -> np.ndarray:
    """The QM record coordinates of each drawn atom (sketch order); never computes."""
    entry = molecule.load_entry(ctx.settings, mid)
    path = qm_npz_path(ctx.settings, entry["key"], ctx.featurizer.method)
    if path is None:
        msg = f"no QM record for {entry['key']}: compute the molecule first"
        raise ServeError(Code.MOLECULE_NOT_READY, msg, detail={"molecule_id": mid})
    if drawn.key != entry["key"]:
        msg = f"the sketch is {drawn.key}, molecule {mid} is {entry['key']}"
        raise ServeError(Code.MAPPING_FAILED, msg, detail={"molecule_id": mid})
    mapping = molecule.sketch_mapping(drawn.mol, entry["key"])
    if mapping is None:
        msg = f"the sketch does not match the record of {entry['key']}"
        raise ServeError(Code.MAPPING_FAILED, msg, detail={"molecule_id": mid})
    with np.load(path, allow_pickle=False) as z:
        xyz = np.asarray(z["coords_ang"], dtype=float).reshape(-1, 3)
    return xyz[mapping["sketch_to_record"]]


def depict2d_out(ctx: ApiContext, body: S.Depict2DIn) -> S.Depict2DOut:
    """POST /depict2d: the 2D drawing of a sketch (`depiction2d`). match3d reads the record
    geometry of `molecule_id` (MOLECULE_NOT_READY without one; never featurizes)."""
    mol, raw = depiction2d.parse_molblock(body.molblock)
    if body.mode == "clean":
        out = depiction2d.clean(mol, raw, body.style)
        return S.Depict2DOut(**out, molecule_id=body.molecule_id)
    if body.molecule_id is None or body.view is None:
        msg = "match3d needs molecule_id and view"
        raise ServeError(Code.REQUEST_INVALID, msg, detail={"mode": body.mode})
    drawn = molecule.parse_input(molblock=body.molblock)
    ref = _record_xyz(ctx, body.molecule_id, drawn)
    out = depiction2d.match3d(mol, raw, ref, body.view.quaternion, body.style)
    return S.Depict2DOut(**out, molecule_id=body.molecule_id)


# ---- point group

SYMMETRY_CACHE_SIZE = 128
_SYMMETRY: OrderedDict[tuple, dict] = OrderedDict()
_SYMMETRY_LOCK = threading.Lock()


def _symmetry_body(qm_path: Path, tolerance: float) -> dict:
    with np.load(qm_path, allow_pickle=False) as z:
        zs = np.asarray(z["atomic_numbers"], dtype=int)
        xyz = np.asarray(z["coords_ang"], dtype=float).reshape(-1, 3)
    try:
        return symmetry.point_group(zs, xyz, tolerance)
    except symmetry.SymmetryError as exc:
        detail = {"n_atoms": len(zs), "max_atoms": symmetry.MAX_ATOMS, "tolerance": tolerance}
        raise ServeError(exc.code, exc.message, detail=detail) from exc
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ServeError(Code.SYMMETRY_FAILED, f"point-group search failed: {exc}") from exc


def symmetry_out(ctx: ApiContext, mid: str, tolerance: float | None = None) -> S.PointGroupOut:
    """GET /molecule/{id}/symmetry?tolerance=: the Schoenflies point group of the record."""
    tol = symmetry.DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    if not symmetry.MIN_TOLERANCE <= tol <= symmetry.MAX_TOLERANCE:
        msg = f"tolerance {tol} outside {symmetry.MIN_TOLERANCE}..{symmetry.MAX_TOLERANCE} Å"
        raise ServeError(Code.REQUEST_INVALID, msg, detail={"tolerance": tol})
    entry, qm_path = _qm(ctx, mid)
    digest = file_digest(qm_path)
    memo = (mid, round(tol, 4), digest)
    with _SYMMETRY_LOCK:
        body = _SYMMETRY.get(memo)
        if body is not None:
            _SYMMETRY.move_to_end(memo)
    if body is None:
        body = _symmetry_body(qm_path, tol)
        with _SYMMETRY_LOCK:
            _SYMMETRY[memo] = body
            while len(_SYMMETRY) > SYMMETRY_CACHE_SIZE:
                _SYMMETRY.popitem(last=False)
    return S.PointGroupOut(molecule_id=mid, key=entry["key"], qm_digest=digest, **body)
