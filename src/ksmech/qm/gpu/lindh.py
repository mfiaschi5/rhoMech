"""Lindh's model Hessian without Python loops: the same stretch, bend and torsion terms as
`ksmech.qm.anc.lindh_hessian` (equal to 1e-15), enumerated with array masks, about 20x faster.
The CPU optimizer keeps its own loops, so its records do not move."""

from __future__ import annotations

import numpy as np

from ksmech.qm.anc import K_BEND, K_STRETCH, K_TORSION, RHO_NEIGHBOUR, RHO_TERM, lindh_rho


def _accumulate(h: np.ndarray, atoms: np.ndarray, b: np.ndarray, k: np.ndarray) -> None:
    """h += sum_t k_t v_t v_t^T, v_t the Wilson vectors b_t (T, na, 3) on atoms_t (T, na)."""
    if not len(k):
        return
    idx = (3 * atoms[:, :, None] + np.arange(3)).reshape(len(k), -1)  # (T, 3 na)
    v = b.reshape(len(k), -1)
    vals = k[:, None, None] * v[:, :, None] * v[:, None, :]
    np.add.at(h, (idx[:, :, None], idx[:, None, :]), vals)


def _stretches(xyz, rho):
    i, j = np.nonzero(np.triu(rho > RHO_TERM, 1))
    d = xyz[i] - xyz[j]
    u = d / np.linalg.norm(d, axis=1)[:, None]
    return np.stack([i, j], 1), np.stack([u, -u], 1), K_STRETCH * rho[i, j]


def _bends(xyz, rho, nb):
    j, i, k = np.nonzero(nb[:, :, None] & nb[:, None, :])
    sel = i < k
    j, i, k = j[sel], i[sel], k[sel]
    w = rho[i, j] * rho[j, k]
    sel = w > RHO_TERM
    j, i, k, w = j[sel], i[sel], k[sel], w[sel]
    u, v = xyz[i] - xyz[j], xyz[k] - xyz[j]
    ru, rv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    c = (u * v).sum(1) / (ru * rv)
    s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    ok = s >= 1e-3
    u, v, ru, rv, c, s = u[ok], v[ok], ru[ok], rv[ok], c[ok], s[ok]
    bi = (c[:, None] * u / ru[:, None] - v / rv[:, None]) / (ru * s)[:, None]
    bk = (c[:, None] * v / rv[:, None] - u / ru[:, None]) / (rv * s)[:, None]
    atoms = np.stack([i[ok], j[ok], k[ok]], 1)
    return atoms, np.stack([bi, -(bi + bk), bk], 1), K_BEND * w[ok]


def _torsions(xyz, rho, nb):
    n = len(xyz)
    jj, kk = np.nonzero(np.triu(nb, 1))  # k in nb[j], k > j
    if not len(jj):
        return np.zeros((0, 4), int), np.zeros((0, 4, 3)), np.zeros(0)
    ar = np.arange(n)
    cand = (nb[jj][:, :, None] & nb[kk][:, None, :]  # i in nb[j], m in nb[k]
            & (ar[None, :, None] != kk[:, None, None])  # i != k
            & (ar[None, None, :] != jj[:, None, None])  # m != j
            & (ar[None, :, None] != ar[None, None, :]))  # m != i  # fmt: skip
    p, i, m = np.nonzero(cand)
    j, k = jj[p], kk[p]
    w = rho[i, j] * rho[j, k] * rho[k, m]
    sel = w > RHO_TERM
    i, j, k, m, w = i[sel], j[sel], k[sel], m[sel], w[sel]
    f, g, hh = xyz[i] - xyz[j], xyz[j] - xyz[k], xyz[m] - xyz[k]
    a, b = np.cross(f, g), np.cross(hh, g)
    a2, b2, gn = (a * a).sum(1), (b * b).sum(1), np.linalg.norm(g, axis=1)
    ok = (a2 >= 1e-6) & (b2 >= 1e-6) & (gn >= 1e-6)
    f, g, hh, a, b, a2, b2, gn = (x[ok] for x in (f, g, hh, a, b, a2, b2, gn))
    fg, hg = (f * g).sum(1) / (a2 * gn), (hh * g).sum(1) / (b2 * gn)
    bi, bl = -(gn / a2)[:, None] * a, (gn / b2)[:, None] * b
    bj = (gn / a2)[:, None] * a + fg[:, None] * a - hg[:, None] * b
    bk = -fg[:, None] * a + hg[:, None] * b - (gn / b2)[:, None] * b
    atoms = np.stack([i[ok], j[ok], k[ok], m[ok]], 1)
    return atoms, np.stack([bi, bj, bk, bl], 1), K_TORSION * w[ok]


def lindh_hessian_fast(z: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Lindh's model Hessian (3n, 3n), as `qm.anc.lindh_hessian`, without Python loops."""
    n = len(z)
    h = np.zeros((3 * n, 3 * n))
    rho = lindh_rho(z, xyz)
    nb = rho > RHO_NEIGHBOUR
    for atoms, b, k in (_stretches(xyz, rho), _bends(xyz, rho, nb), _torsions(xyz, rho, nb)):
        _accumulate(h, atoms, b, k)
    return h
