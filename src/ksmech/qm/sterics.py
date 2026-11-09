"""Per-atom buried volume (%Vbur): morfeus' `BuriedVolume` centred on every atom in turn.

morfeus builds the sphere's grid of points and a KD-tree over it for every centre. The grid
depends only on the sphere's radius and density, so one grid and one tree serve every centre of
every molecule: each centre asks the tree for the points within each other atom's radius, with
the coordinates taken relative to the centre as morfeus takes them. Same points, same test,
same values as morfeus (`tests/test_qm_sterics.py`), 3-4x faster (docs/decisions.md,
2026-11-09).
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
from morfeus.geometry import Sphere
from morfeus.utils import get_radii
from scipy.spatial import cKDTree

RADIUS_ANG = 3.5
# morfeus default is 0.001 A^-3; 0.004 is 4.5x faster and moves %Vbur by < 0.15 (benchmarked)
DENSITY = 0.004
RADII_TYPE, RADII_SCALE = "bondi", 1.17  # morfeus' BuriedVolume defaults


@cache
def sphere_tree(radius: float, density: float) -> tuple[int, cKDTree]:
    """(number of points, KD-tree) of morfeus' filled sphere grid centred at the origin."""
    points = Sphere(np.zeros(3), radius, method="projection", density=density, filled=True).points
    return len(points), cKDTree(points, compact_nodes=False, balanced_tree=False)


def buried_volume_per_atom(
    atomic_numbers: np.ndarray,
    coords_ang: np.ndarray,
    radius: float = RADIUS_ANG,
    density: float = DENSITY,
) -> np.ndarray:
    """%Vbur (sphere of `radius` Angstrom, hydrogens included) for each atom as the centre.

    The centre atom itself is excluded, as morfeus excludes it. A single atom has nothing
    around it: 0 %.
    """
    z = [int(a) for a in atomic_numbers]
    radii = np.asarray(get_radii(z, radii_type=RADII_TYPE, scale=RADII_SCALE), dtype=float)
    n_points, tree = sphere_tree(float(radius), float(density))
    xyz = np.asarray(coords_ang, dtype=float)
    out = np.zeros(len(z))
    for i in range(len(z)):
        rel = xyz - xyz[i]
        near = radii + radius > np.linalg.norm(rel, axis=1)  # morfeus' test for a reaching atom
        near[i] = False
        if not near.any():
            continue
        hits = tree.query_ball_point(rel[near], radii[near], return_sorted=False)
        buried = np.zeros(n_points, dtype=bool)
        buried[np.fromiter(itertools.chain.from_iterable(hits), dtype=np.intp)] = True
        out[i] = 100.0 * (int(buried.sum()) / n_points)
    return out
