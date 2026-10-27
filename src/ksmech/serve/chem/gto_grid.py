"""Orbitals on a cube grid, by the separability of Cartesian Gaussians (display surfaces).

`gto.ao_values(Z, xyz, points) @ C` is exact but evaluates every AO at every point: a
350k-point cube of a 33-atom molecule takes seconds. On the axis-aligned grid of a cube
(`cubes.Grid`) every primitive Cartesian component factorizes,

    x^lx y^ly z^lz exp(-a r^2) = [x^lx e^(-a x^2)] [y^ly e^(-a y^2)] [z^lz e^(-a z^2)],

(x, y, z relative to the atom, in bohr) so the grid of one orbital is a sum of outer products of
1-D factors. With the terms t = (atom, shell, primitive, Cartesian component), the 1-D factors
X (T, nx), Y (T, ny), Z (T, nz) and the term weights W = M @ C (M[t, ao]: the primitive's
contraction coefficient times tblite's spherical transformation `gto.SPH`, the same shells,
normalization and AO order as `gto.ao_values`),

    psi_j[x, y, z] = sum_t (X[t, x] Y[t, y]) (W[t, j] Z[t, z]),

one matrix product of (nx ny, T) by (T, k nz) for k orbitals at once, in blocks of terms and
columns so that no block exceeds `BLOCK_BYTES`. The values equal `ao_values @ C` to rounding
(`tests/test_serve_gto_grid.py`).
"""

from __future__ import annotations

from functools import cache

import numpy as np

from ksmech.qm.geometry import BOHR_ANG
from ksmech.serve.chem.gto import CART, SPH, element_shells

BLOCK_BYTES = 32 * 2**20


@cache
def _element_terms(z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(alpha (t,), exponents (t, 3), M (t, nao)) of one element's terms, `gto` AO order."""
    alphas, expo, rows = [], [], []
    nao = sum(s.n_sph for s in element_shells(z))
    ao = 0
    for sh in element_shells(z):
        for a, c in zip(sh.alpha, sh.coeff, strict=True):
            for k, e in enumerate(CART[sh.l]):
                row = np.zeros(nao)
                row[ao : ao + sh.n_sph] = c * SPH[sh.l][:, k]
                alphas.append(a)
                expo.append(e)
                rows.append(row)
        ao += sh.n_sph
    return np.array(alphas), np.array(expo, dtype=int), np.array(rows)


def _axis(origin_bohr: float, n: int, step_bohr: float, centre: float, alpha, power):
    """(t, n) 1-D factors d^power exp(-alpha d^2) along one axis, d = x - centre."""
    d = origin_bohr + step_bohr * np.arange(n) - centre
    return d[None, :] ** power[:, None] * np.exp(-alpha[:, None] * (d * d)[None, :])


def grid_factors(z, xyz_ang, origin_ang, n, spacing_ang):
    """(X, Y, Z, M): the 1-D factors (T, n_i) of every term and their AO map M (T, nao)."""
    z = np.asarray(z, dtype=int).ravel()
    centres = np.asarray(xyz_ang, dtype=float).reshape(-1, 3) / BOHR_ANG
    o = np.asarray(origin_ang, dtype=float) / BOHR_ANG
    h = float(spacing_ang) / BOHR_ANG
    terms = [_element_terms(int(zi)) for zi in z]
    nao = sum(t[2].shape[1] for t in terms)
    M = np.zeros((sum(len(t[0]) for t in terms), nao))
    facs: list[list[np.ndarray]] = [[], [], []]
    row = col = 0
    for (alpha, expo, m), c in zip(terms, centres, strict=True):
        M[row : row + len(alpha), col : col + m.shape[1]] = m
        for d in range(3):
            facs[d].append(_axis(o[d], int(n[d]), h, c[d], alpha, expo[:, d]))
        row += len(alpha)
        col += m.shape[1]
    X, Y, Zf = (np.concatenate(f, axis=0) for f in facs)
    return X, Y, Zf, M


def _blocks(total: int, per: int) -> list[slice]:
    size = max(1, BLOCK_BYTES // max(1, 8 * per))
    return [slice(i, min(total, i + size)) for i in range(0, total, size)]


def columns_on_grid(z, xyz_ang, C, origin_ang, n, spacing_ang) -> np.ndarray:
    """(k, nx, ny, nz) values (bohr^-3/2) of the k columns of C (nao, k) on the grid."""
    nx, ny, nz = (int(v) for v in n)
    C = np.asarray(C, dtype=float).reshape(np.shape(C)[0], -1)
    X, Y, Zf, M = grid_factors(z, xyz_ang, origin_ang, n, spacing_ang)
    W = M @ C  # (T, k)
    k = W.shape[1]
    out = np.zeros((nx * ny, k * nz))
    for ts in _blocks(len(W), nx * ny):
        P = (X[ts, :, None] * Y[ts, None, :]).reshape(-1, nx * ny).T  # (nx ny, t)
        Zw = (W[ts, :, None] * Zf[ts, None, :]).reshape(-1, k * nz)  # (t, k nz)
        out += P @ Zw
    return out.reshape(nx, ny, k, nz).transpose(2, 0, 1, 3)


def square_sum_on_grid(z, xyz_ang, C, origin_ang, n, spacing_ang, scale=1.0) -> np.ndarray:
    """(nx, ny, nz): scale * sum_j psi_j^2 over the columns of C, in blocks of columns."""
    C = np.asarray(C, dtype=float).reshape(np.shape(C)[0], -1)
    out = np.zeros(tuple(int(v) for v in n))
    for cs in _blocks(C.shape[1], int(np.prod(n))):
        vals = columns_on_grid(z, xyz_ang, C[:, cs], origin_ang, n, spacing_ang)
        out += np.einsum("kxyz,kxyz->xyz", vals, vals)
    return float(scale) * out
