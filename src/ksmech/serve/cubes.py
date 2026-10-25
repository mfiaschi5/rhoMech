"""Orbital cubes: a localized orbital on a grid around its cap, as a Gaussian cube (adapter A15).

- **Grid** (`grid_around`): the token's cap atoms (`cap_atoms`: its centres and their
  bonded neighbours) +- `padding`, at `spacing` (default 0.2 / 4.0 Angstrom), so the box stays
  local to the LMO; `n_i = ceil((hi_i - lo_i) / spacing) + 1`. More than 4,000,000 points is
  `CUBE_TOO_LARGE` (422).
- **Values**: `gto.ao_values(Z, xyz, points) @ coeff_vector` over every AO of the fragment, in
  chunks of 20,000 points: the LMO exactly, in bohr^-3/2. The Pipek-Mezey phase is arbitrary.
- **Cube file**: bohr throughout; two comment lines, the atom count and origin, three diagonal
  axis lines, one `Z Z.0 x y z` line per atom, then the values in %13.5E, six per line, x
  outermost and z innermost, a line break after each z column.

Cube files are cached under `data/serve/cubes/<sha>.cube`, keyed by the state, the token, the
token fingerprint (the basis code and the token schema), the fragments' token digest, the
spacing, the padding and `CUBE_VERSION`. Upstream modules are imported inside the functions, so
the API side can import the path helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ksmech.serve.contracts import PoolState, Runtime
from ksmech.serve.errors import Code, ServeError

if TYPE_CHECKING:
    from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
    from ksmech.serve.settings import ServeSettings

CUBE_VERSION = 1
MAX_POINTS = 4_000_000
CHUNK_POINTS = 20_000
DEFAULT_SPACING_ANG = 0.2
DEFAULT_PADDING_ANG = 4.0
ISO_DEFAULT = 0.05
UNITS_COMMENT = "units bohr; values bohr^-3/2; phase arbitrary (Pipek-Mezey)"


def bohr_ang() -> float:
    from ksmech.qm.geometry import BOHR_ANG

    return float(BOHR_ANG)


@dataclass(frozen=True)
class Grid:
    origin_ang: np.ndarray  # (3,) the first grid point
    n: tuple[int, int, int]
    spacing_ang: float

    @property
    def n_points(self) -> int:
        return math.prod(self.n)


def check_box(spacing: float, padding: float) -> None:
    if not (math.isfinite(spacing) and spacing > 0):
        raise ServeError(Code.REQUEST_INVALID, f"spacing must be a number > 0, got {spacing!r}")
    if not (math.isfinite(padding) and padding >= 0):
        raise ServeError(Code.REQUEST_INVALID, f"padding must be a number >= 0, got {padding!r}")


def grid_around(
    points_ang: np.ndarray,
    spacing: float = DEFAULT_SPACING_ANG,
    padding: float = DEFAULT_PADDING_ANG,
) -> Grid:
    """The box of `points_ang` +- `padding` at `spacing` (Angstrom). Raises CUBE_TOO_LARGE
    above `MAX_POINTS` points (an extent / spacing that overflows included) and
    REQUEST_INVALID for a bad spacing or padding."""
    spacing, padding = float(spacing), float(padding)
    check_box(spacing, padding)
    pts = np.asarray(points_ang, dtype=float).reshape(-1, 3)
    if not len(pts):
        raise ServeError(Code.REQUEST_INVALID, "a grid needs at least one point")
    lo, hi = pts.min(axis=0) - padding, pts.max(axis=0) + padding
    with np.errstate(over="ignore", invalid="ignore"):
        ratios = (hi - lo) / spacing
    if not np.all(np.isfinite(ratios)) or np.any(ratios + 1 > MAX_POINTS):
        msg = f"a grid of {ratios.tolist()} steps per axis exceeds {MAX_POINTS} points"
        detail = {"spacing": spacing, "padding": padding, "max_points": MAX_POINTS}
        raise ServeError(Code.CUBE_TOO_LARGE, msg, detail=detail)
    # rounded to 1e-9 so that an extent that is a whole number of steps is not one step longer
    n = tuple(int(math.ceil(round(float(r), 9))) + 1 for r in ratios)
    total = math.prod(n)  # Python ints: never wraps like an int64 product
    if total > MAX_POINTS:
        msg = f"a {n[0]}x{n[1]}x{n[2]} grid has {total} points (at most {MAX_POINTS})"
        detail = {"n": list(n), "points": total, "max_points": MAX_POINTS}
        raise ServeError(Code.CUBE_TOO_LARGE, msg, detail=detail)
    return Grid(lo, n, spacing)


def grid_points(g: Grid, start: int, stop: int) -> np.ndarray:
    """(stop - start, 3) Angstrom positions of the flat grid indices [start, stop), x outermost
    and z innermost (the cube file's value order)."""
    idx = np.arange(int(start), int(stop))
    ix, iy, iz = np.unravel_index(idx, g.n)
    return g.origin_ang[None, :] + g.spacing_ang * np.column_stack([ix, iy, iz])


def on_grid(g: Grid, evaluate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """`evaluate(points)` on every grid point, in chunks of `CHUNK_POINTS` (flat cube order)."""
    out = np.empty(g.n_points)
    for start in range(0, g.n_points, CHUNK_POINTS):
        stop = min(g.n_points, start + CHUNK_POINTS)
        out[start:stop] = evaluate(grid_points(g, start, stop))
    return out


_on_grid = on_grid  # the name before the workbench made it public


def fragment_arrays(t: MoleculeTokens) -> tuple[np.ndarray, np.ndarray]:
    """(Z, positions in Angstrom) of a fragment's record."""
    z = np.array([a.atomic_number for a in t.atoms], dtype=int)
    return z, np.array([a.position for a in t.atoms], dtype=float).reshape(-1, 3)


def lmo_values_at(t: MoleculeTokens, k: int, points_ang: np.ndarray) -> np.ndarray:
    """The whole LMO `t.orbitals[k]` at the points (bohr^-3/2), every AO of the fragment."""
    from ksmech.serve.gto import ao_values

    z, xyz = fragment_arrays(t)
    return ao_values(z, xyz, points_ang) @ t.orbitals[k].coeff_vector


def lmo_values(t: MoleculeTokens, k: int, g: Grid) -> np.ndarray:
    """`lmo_values_at` on every grid point, in chunks of `CHUNK_POINTS`."""
    return on_grid(g, lambda pts: lmo_values_at(t, k, pts))


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def write_cube(
    g: Grid,
    z: np.ndarray,
    xyz_ang: np.ndarray,
    values: np.ndarray,
    comments: tuple[str, str],
) -> str:
    """The Gaussian cube text (bohr throughout) of `values` on `g` with the atoms (z, xyz)."""
    b = bohr_ang()
    z = np.asarray(z, dtype=int).ravel()
    xyz = np.asarray(xyz_ang, dtype=float).reshape(-1, 3) / b
    vals = np.asarray(values, dtype=float).reshape(g.n)
    o = np.asarray(g.origin_ang, dtype=float) / b
    lines = [_one_line(comments[0]), _one_line(comments[1])]
    lines.append(f"{len(z):5d}{o[0]:12.6f}{o[1]:12.6f}{o[2]:12.6f}")
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = g.spacing_ang / b
        lines.append(f"{g.n[axis]:5d}{step[0]:12.6f}{step[1]:12.6f}{step[2]:12.6f}")
    for zi, p in zip(z, xyz, strict=True):
        lines.append(f"{int(zi):5d}{float(zi):12.6f}{p[0]:12.6f}{p[1]:12.6f}{p[2]:12.6f}")
    nz = g.n[2]
    for ix in range(g.n[0]):
        for iy in range(g.n[1]):
            col = vals[ix, iy]
            lines += ["".join(f"{v:13.5E}" for v in col[k : k + 6]) for k in range(0, nz, 6)]
    return "\n".join(lines) + "\n"


def cube_path(
    s: ServeSettings, state_id: str, token: int, spacing: float, padding: float, digest: str
) -> Path:
    """`data/serve/cubes/<sha>.cube` of one orbital cube."""
    from ksmech.serve.versions import token_fingerprint

    key = ["orbital", state_id, int(token), token_fingerprint(), digest]
    key += [float(spacing), float(padding), CUBE_VERSION]
    sha = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]
    return s.cubes / f"{sha}.cube"


def orbital_cube(t: MoleculeTokens, k: int, spacing: float, padding: float, header: str) -> str:
    """The cube text of LMO k of one fragment on the grid around its cap."""
    from ksmech.orbitals.tokens import OrbitalKind

    o = t.orbitals[k]
    z, xyz = fragment_arrays(t)
    g = grid_around(xyz[cap_atoms(o, t)], spacing, padding)
    kind = "occ" if o.kind == OrbitalKind.OCCUPIED else "virt"
    first = f"{header} kind={kind} E_r={float(o.energy_ev):.6f}"
    return write_cube(g, z, xyz, lmo_values(t, k, g), (first, UNITS_COMMENT))


def cap_atoms(o: OrbitalToken, t: MoleculeTokens) -> np.ndarray:
    """The token's local cap: its centre atoms and their first bonded neighbours (hydrogens
    included), sorted. Vendored from `physics/contact/legacy_fixed23.cap_atoms` (46adf6b; the
    module is archived on `archive/stage-b`), with its `bonded_neighbours`
    (`tokens.bonded_matrix`: within 1.3x the covalent-radii sum)."""
    from ksmech.orbitals.tokens import bonded_matrix

    centres = {int(a) for a in o.descriptors.all_center_atoms}
    bonded = bonded_matrix(t)
    cap = set(centres)
    for a in centres:
        cap.update(int(n) for n in np.flatnonzero(bonded[a]))
    return np.array(sorted(cap))


def worker_tokens(rt: Runtime, state: PoolState):
    """`rt.reactant_tokens(state)` with MissingTokens / CorrespondenceError as ServeErrors."""
    from ksmech.arrows.reactants import CorrespondenceError, MissingTokens

    try:
        return rt.reactant_tokens(state)
    except MissingTokens as exc:
        detail = {"keys": list(exc.keys)}
        raise ServeError(Code.MISSING_TOKENS, str(exc), detail=detail) from exc
    except CorrespondenceError as exc:
        raise ServeError(Code.CORRESPONDENCE, str(exc)) from exc


def t_orbital_cube(rt: Runtime, payload: dict) -> dict:
    """{"state", "token", "spacing", "padding"} -> {"path"}: the cube of one orbital token."""
    from ksmech.serve.analysis import analysis_digest, tokens_changed
    from ksmech.serve.geometry import orbital_position
    from ksmech.serve.pools import atomic_write_text

    state = PoolState.from_dict(payload["state"])
    token = int(payload["token"])
    spacing = float(payload.get("spacing", DEFAULT_SPACING_ANG))
    padding = float(payload.get("padding", DEFAULT_PADDING_ANG))
    check_box(spacing, padding)
    digest = analysis_digest(rt.loader, state)
    if digest is None:
        raise ServeError(Code.MISSING_TOKENS, f"no tokens for every fragment of {state.pool_key}")
    path = cube_path(rt.settings, state.state_id, token, spacing, padding, digest)
    if not path.is_file():
        rtk = worker_tokens(rt, state)
        f, k = orbital_position(rtk, token)
        header = f"ksmech-serve cube v{CUBE_VERSION} state={state.state_id} token={token}"
        text = orbital_cube(rtk.fragments[f].tokens, k, spacing, padding, header)
        after = analysis_digest(rt.loader, state)
        if after != digest:  # the cube must hold the tokens its file name claims
            raise tokens_changed(state, digest, after)
        atomic_write_text(path, text, rt.settings)
    return {"path": str(path)}
