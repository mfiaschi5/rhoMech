"""Volumetric surfaces of the molecule workbench (worker task `wb_cube`), display only.

One whole-molecule grid: `cubes.grid_around(all atoms, spacing, padding)` (default padding
3.0 Angstrom). The spacing is explicit, or set by `quality` so that the box holds about
120k (draft), 350k (normal) or 1.2M (fine) points, rounded up to 0.001 Angstrom;
`cubes.MAX_POINTS` still applies (CUBE_TOO_LARGE). Values come from
`A = physics.gto.ao_values(Z, xyz, points)`, chunked through `cubes.on_grid`:

- `orbital`: `A @ c` (bohr^-3/2) for MO / LMO `index` of a set (`orbital_sets`);
- `orbital_density`: `(A @ c)^2` (bohr^-3);
- `density`: the valence density `rowsum((A @ (C_occ sqrt(n)))^2)`, P = C diag(n) C^T;
- `fukui_minus` / `fukui_plus`: frozen-orbital Fukui functions, the mean |psi|^2 over the HOMO
  / LUMO degenerate block (`orbitals.frontier.frontier_blocks`, 0.05 eV); `NO_FRONTIER` when
  the block does not exist;
- `dual`: `fukui_plus - fukui_minus`;
- `esp`: sum_A q_A / max(|r - R_A|, 0.2 bohr) over the Mulliken charges, in hartree/e: an
  approximation (point charges, no electron density), labelled so.

The minimal basis is the STO-nG fit of GFN2's valence Slater functions; under real chemistry
its overlap differs slightly from tblite's, so every density cube reports its integral
(`integral`: the grid sum times the voxel volume; for `orbital` the integral of psi^2) next to
the exact value (`expected`: 1, N_valence or 0). The two comment lines of the cube carry the
kind, the units, the integral and the notes; a `<sha>.json` sidecar holds the same metadata.
Cubes are cached under `data/serve/workbench/cubes/<sha>.cube`, keyed by the key, the QM method
and content sha, the kind, set, start, index, spacing (or quality), padding, the model token
file's digest (model-set orbitals only), `DENSITY_VERSION`, `ORBITAL_SET_VERSION`,
`WORKBENCH_VERSION` and the oriented fingerprint.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ksmech.serve import cubes
from ksmech.serve.contracts import Runtime
from ksmech.serve.errors import Code, ServeError, jsonable
from ksmech.serve.molecule import WORKBENCH_VERSION, cache_sha

if TYPE_CHECKING:
    from ksmech.qm.records import QMRecord
    from ksmech.serve.settings import ServeSettings

DENSITY_VERSION = 1
KINDS = ("orbital", "orbital_density", "density", "fukui_minus", "fukui_plus", "dual", "esp")
ORBITAL_KINDS = frozenset({"orbital", "orbital_density"})
QUALITY_POINTS = {"draft": 120_000, "normal": 350_000, "fine": 1_200_000}
DEFAULT_QUALITY = "normal"
DEFAULT_PADDING_ANG = 3.0
ESP_FLOOR_BOHR = 0.2
SPACING_STEP_ANG = 0.001
UNITS = {
    "orbital": "bohr^-3/2",
    "orbital_density": "bohr^-3",
    "density": "e bohr^-3",
    "fukui_minus": "bohr^-3",
    "fukui_plus": "bohr^-3",
    "dual": "bohr^-3",
    "esp": "hartree/e",
}
NOTE_BASIS = (
    "valence electrons only, minimal basis (the STO-nG fit of the GFN2 valence Slater "
    "functions; its overlap differs slightly from tblite's, hence the integral)"
)
NOTE_PHASE = "phase arbitrary"
NOTE_ESP = (
    "approximate ESP: Mulliken point charges, sum q_A / max(|r - R_A|, 0.2 bohr); no electron "
    "density"
)


@dataclass(frozen=True)
class CubeRequest:
    kind: str
    set: str | None  # None for the set-independent kinds
    start: str | None
    index: int | None
    quality: str | None  # None when the spacing is explicit
    spacing: float | None
    padding: float

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, d: dict) -> CubeRequest:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


def _number(name: str, value, minimum: float, strict: bool) -> float:
    x = float(value)
    if not math.isfinite(x) or x < minimum or (strict and x == minimum):
        op = ">" if strict else ">="
        raise ServeError(Code.REQUEST_INVALID, f"{name} must be a number {op} {minimum}")
    return x


def cube_request(
    kind: str,
    set_: str | None = None,
    start: str | None = None,
    index: int | None = None,
    quality: str | None = None,
    spacing: float | None = None,
    padding: float | None = None,
) -> CubeRequest:
    """The normalized request (REQUEST_INVALID for a bad value): set, start and index only for
    the orbital kinds, quality only when the spacing is not explicit."""
    from ksmech.serve.orbital_sets import check_set

    if kind not in KINDS:
        raise ServeError(Code.REQUEST_INVALID, f"kind must be one of {list(KINDS)}, got {kind!r}")
    pad = DEFAULT_PADDING_ANG if padding is None else _number("padding", padding, 0.0, False)
    sp = None if spacing is None else _number("spacing", spacing, 0.0, True)
    q = None if sp is not None else (quality or DEFAULT_QUALITY)
    if q is not None and q not in QUALITY_POINTS:
        msg = f"quality must be one of {list(QUALITY_POINTS)}, got {quality!r}"
        raise ServeError(Code.REQUEST_INVALID, msg)
    if kind not in ORBITAL_KINDS:
        return CubeRequest(kind, None, None, None, q, sp, pad)
    if index is None or int(index) < 0:
        raise ServeError(Code.REQUEST_INVALID, f"kind {kind} needs an orbital index >= 0")
    set_, start = check_set(set_, start)
    return CubeRequest(kind, set_, start, int(index), q, sp, pad)


def spacing_for(xyz_ang: np.ndarray, quality: str, padding: float) -> float:
    """The spacing that puts about `QUALITY_POINTS[quality]` points in the padded box."""
    xyz = np.asarray(xyz_ang, dtype=float).reshape(-1, 3)
    extent = xyz.max(axis=0) - xyz.min(axis=0) + 2 * float(padding)
    volume = float(np.prod(np.maximum(extent, SPACING_STEP_ANG)))
    raw = (volume / QUALITY_POINTS[quality]) ** (1 / 3)
    steps = max(1, math.ceil(round(raw / SPACING_STEP_ANG, 6)))
    return round(steps * SPACING_STEP_ANG, 6)


def cube_paths(
    s: ServeSettings, key: str, method: str, qm_digest: str, req: CubeRequest, token_digest
) -> tuple[Path, Path]:
    """(cube, metadata JSON) of one surface."""
    from ksmech.serve.orbital_sets import ORBITAL_SET_VERSION
    from ksmech.serve.versions import oriented_fingerprint

    parts = ["wb_cube", key, method, qm_digest, req.to_payload(), token_digest]
    parts += [DENSITY_VERSION, ORBITAL_SET_VERSION, WORKBENCH_VERSION, oriented_fingerprint()]
    sha = cache_sha(parts)
    return s.workbench_cubes / f"{sha}.cube", s.workbench_cubes / f"{sha}.json"


# ---- evaluators (points in Angstrom)


def esp_at(z_xyz_ang: np.ndarray, charges: np.ndarray, points_ang: np.ndarray) -> np.ndarray:
    """Point-charge potential (hartree/e) at the points, |r - R_A| floored at 0.2 bohr."""
    b = cubes.bohr_ang()
    centres = np.asarray(z_xyz_ang, dtype=float).reshape(-1, 3) / b
    pts = np.asarray(points_ang, dtype=float).reshape(-1, 3) / b
    d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2)
    return np.maximum(d, ESP_FLOOR_BOHR) ** -1 @ np.asarray(charges, dtype=float)


def _ao(rec: QMRecord, pts: np.ndarray) -> np.ndarray:
    from ksmech.physics.gto import ao_values

    return ao_values(rec.atomic_numbers, rec.coords_ang, pts)


def _square_sum(rec: QMRecord, M: np.ndarray, scale: float = 1.0) -> Callable:
    def evaluate(pts: np.ndarray) -> np.ndarray:
        B = _ao(rec, pts) @ M
        return scale * np.einsum("pk,pk->p", B, B)

    return evaluate


def frontier_columns(rec: QMRecord, level: str) -> np.ndarray:
    """(nao, k) canonical MOs of the HOMO or LUMO degenerate block; NO_FRONTIER when absent."""
    from ksmech.orbitals.frontier import frontier_blocks

    blocks = frontier_blocks(np.asarray(rec.orbital_energies_ev), np.asarray(rec.occupations))
    if level not in blocks or not len(blocks[level]):
        msg = f"{rec.smiles} has no {level.upper()} (frontier blocks: {sorted(blocks)})"
        raise ServeError(Code.NO_FRONTIER, msg, detail={"level": level, "blocks": sorted(blocks)})
    return rec.C[:, np.asarray(blocks[level], dtype=int)]


def _fukui(rec: QMRecord, level: str) -> tuple[Callable, str]:
    Cb = frontier_columns(rec, level)
    note = f"frozen-orbital: mean |psi|^2 over the {level.upper()} block ({Cb.shape[1]} MOs)"
    return _square_sum(rec, Cb, 1.0 / Cb.shape[1]), note


def _dual(rec: QMRecord) -> tuple[Callable, str]:
    plus, minus = _fukui(rec, "lumo")[0], _fukui(rec, "homo")[0]
    return (lambda pts: plus(pts) - minus(pts)), "frozen-orbital f+ - f-"


def orbital_vector(rt: Runtime, key: str, rec: QMRecord, req: CubeRequest) -> np.ndarray:
    """The AO coefficients of orbital `req.index` of the set; ORBITAL_OUT_OF_RANGE."""
    from ksmech.serve.orbital_sets import CANONICAL, set_tokens

    if req.set == CANONICAL:
        C = np.asarray(rec.C)
    else:
        C = set_tokens(rt, key, req.set, req.start)[0].C_loc
    n = C.shape[1]
    if not 0 <= int(req.index) < n:
        msg = f"orbital {req.index} out of range: set {req.set} has {n} orbitals (0..{n - 1})"
        raise ServeError(Code.ORBITAL_OUT_OF_RANGE, msg, detail={"index": req.index, "n": n})
    return C[:, int(req.index)]


def surface(rt: Runtime, key: str, rec: QMRecord, req: CubeRequest) -> tuple[Callable, dict]:
    """(evaluate(points) -> values, {expected, notes, integrate}) of one request."""
    occ = np.asarray(rec.occupations, dtype=float)
    if req.kind in ORBITAL_KINDS:
        c = orbital_vector(rt, key, rec, req)
        where = "canonical MO" if req.start is None else f"{req.set}/{req.start} LMO"
        notes = [f"{where} {req.index}", NOTE_PHASE if req.kind == "orbital" else NOTE_BASIS]
        if req.kind == "orbital":
            return (lambda pts: _ao(rec, pts) @ c), {"expected": 1.0, "notes": notes}
        return _square_sum(rec, c[:, None]), {"expected": 1.0, "notes": notes}
    if req.kind == "density":
        M = rec.C[:, occ > 0] * np.sqrt(occ[occ > 0])
        return _square_sum(rec, M), {"expected": float(occ.sum()), "notes": [NOTE_BASIS]}
    if req.kind in ("fukui_minus", "fukui_plus"):
        evaluate, note = _fukui(rec, "homo" if req.kind == "fukui_minus" else "lumo")
        return evaluate, {"expected": 1.0, "notes": [note, NOTE_BASIS]}
    if req.kind == "dual":
        evaluate, note = _dual(rec)
        return evaluate, {"expected": 0.0, "notes": [note, NOTE_BASIS]}
    q = np.asarray(rec.charges, dtype=float)
    return (lambda pts: esp_at(rec.coords_ang, q, pts)), {"expected": None, "notes": [NOTE_ESP]}


def grid_integral(kind: str, values: np.ndarray, g: cubes.Grid) -> float | None:
    """The grid sum times the voxel volume (bohr^3); of psi^2 for `orbital`; None for `esp`."""
    if kind == "esp":
        return None
    dv = (g.spacing_ang / cubes.bohr_ang()) ** 3
    v = np.asarray(values, dtype=float)
    return float(np.sum(v * v if kind == "orbital" else v) * dv)


def _comments(key: str, req: CubeRequest, meta: dict) -> tuple[str, str]:
    first = f"ksmech-serve workbench cube v{DENSITY_VERSION} kind={req.kind} key={key}"
    if req.kind in ORBITAL_KINDS:
        first += f" set={req.set} start={req.start} index={req.index}"
    integral = "none" if meta["integral"] is None else f"{meta['integral']:.6f}"
    of = " (of psi^2)" if req.kind == "orbital" else ""
    second = f"units bohr; values {meta['units']}; integral{of}={integral}"
    second += f" expected={meta['expected']}; " + "; ".join(meta["notes"])
    return first, second


def render(rt: Runtime, key: str, rec: QMRecord, req: CubeRequest) -> tuple[str, dict]:
    """(cube text, metadata) of one request on the whole-molecule grid."""
    from ksmech.serve.context import NOTE_SYNTHETIC

    spacing = req.spacing or spacing_for(rec.coords_ang, req.quality, req.padding)
    g = cubes.grid_around(rec.coords_ang, spacing, req.padding)
    evaluate, info = surface(rt, key, rec, req)
    values = cubes.on_grid(g, evaluate)
    notes = info["notes"] + ([NOTE_SYNTHETIC] if rt.settings.chemistry == "synthetic" else [])
    meta = req.to_payload() | {
        "key": key,
        "units": UNITS[req.kind],
        "integral": grid_integral(req.kind, values, g),
        "expected": info["expected"],
        "notes": notes,
        "spacing_ang": float(spacing),
        "n": list(g.n),
        "n_points": g.n_points,
        "origin_ang": [float(x) for x in g.origin_ang],
        "min": float(values.min()),
        "max": float(values.max()),
    }
    z, xyz = rec.atomic_numbers, rec.coords_ang
    return cubes.write_cube(g, z, xyz, values, _comments(key, req, meta)), meta


def t_wb_cube(rt: Runtime, payload: dict) -> dict:
    """{"key", "request": CubeRequest} -> {"path", "meta"} (cube text and its metadata JSON)."""
    from ksmech.serve.orbital_sets import is_model_set, load_qm
    from ksmech.serve.pools import atomic_write_text

    key = str(payload["key"])
    r = CubeRequest.from_payload(payload["request"])
    req = cube_request(r.kind, r.set, r.start, r.index, r.quality, r.spacing, r.padding)
    rec, _, qm_digest = load_qm(rt, key)
    model = req.kind in ORBITAL_KINDS and is_model_set(req.set, req.start)
    token_digest = rt.loader.digest(key) if model else None
    method = rt.featurizer.method
    path, meta_path = cube_paths(rt.settings, key, method, qm_digest, req, token_digest)
    if not (path.is_file() and meta_path.is_file()):
        text, meta = render(rt, key, rec, req)
        atomic_write_text(path, text, rt.settings)
        atomic_write_text(meta_path, json.dumps(jsonable(meta), sort_keys=True) + "\n", rt.settings)
    return {"path": str(path), "meta": str(meta_path)}
