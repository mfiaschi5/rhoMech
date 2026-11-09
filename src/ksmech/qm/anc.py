"""Geometry optimization in approximate normal coordinates (ANC) with rational-function (RFO)
steps, the scheme of xtb's ANCopt, on any energy-and-gradient function.

The coordinates are the eigenvectors of a model Hessian with the translations and rotations
projected out: Lindh's (Lindh, Bernhardsson, Karlstrom, Malmqvist, Chem. Phys. Lett. 241 (1995)
423), a sum over atom pairs, triples and quadruples of stretch, bend and torsion terms damped by
rho_ij = exp(alpha_ij (r_ref,ij^2 - r_ij^2)). In that basis the start is well conditioned, so
the steps are near Newton steps from the first one; the Hessian is then updated by BFGS from the
gradients, and every `micro` steps the basis is rebuilt from it at the current geometry.

Each step is the RFO step of the current Hessian, cut to a trust radius that grows when the
energy falls as predicted and shrinks when it does not; a step that raises the energy is taken
back. Convergence is the largest Cartesian gradient component, as for the L-BFGS-B path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Lindh 1995, Table 1, by period: rows 1, 2 and 3 (and every heavier row) of the periodic table
ALPHA = np.array([[1.0, 0.3949, 0.3949], [0.3949, 0.28, 0.28], [0.3949, 0.28, 0.28]])
R_REF = np.array([[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]])  # Bohr
K_STRETCH, K_BEND, K_TORSION = 0.45, 0.15, 0.005  # Hartree / Bohr^2, Hartree / rad^2
RHO_NEIGHBOUR = 1e-2  # an atom pair that may take part in a bend or a torsion
RHO_TERM = 1e-4  # the smallest product of rho kept as a term
H_LOW = 0.01  # floor of the model Hessian's eigenvalues (xtb's hlow)

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]  # Bohr (n, 3) -> (E, grad (n, 3))


@dataclass(frozen=True)
class AncSettings:
    fmax: float = 3e-4  # converged when every gradient component is at most this (Ha / Bohr)
    maxiter: int = 500  # energy and gradient evaluations
    micro: int = 20  # steps between two rebuilds of the basis
    radius: float = 0.3  # first trust radius, Bohr
    radius_max: float = 1.0
    radius_min: float = 1e-3
    uphill: float = 1e-6  # Hartree: a step that raises the energy more than this is taken back


@dataclass(frozen=True)
class AncResult:
    xyz: np.ndarray  # Bohr
    converged: bool
    n_evals: int
    gmax: float


def _period(z: np.ndarray) -> np.ndarray:
    return np.where(z <= 2, 0, np.where(z <= 10, 1, 2))


def lindh_rho(z: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """rho_ij of every atom pair (zero on the diagonal)."""
    p = _period(np.asarray(z))
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    rho = np.exp(ALPHA[p[:, None], p[None, :]] * (R_REF[p[:, None], p[None, :]] ** 2 - d2))
    np.fill_diagonal(rho, 0.0)
    return rho


def b_bend(xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> np.ndarray | None:
    """Wilson B vectors (3, 3) of the angle i-j-k; None near 0 or 180 degrees."""
    u, v = xi - xj, xk - xj
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    c = float(u @ v / (ru * rv))
    s = np.sqrt(max(1.0 - c * c, 0.0))
    if s < 1e-3:
        return None
    bi = (c * u / ru - v / rv) / (ru * s)
    bk = (c * v / rv - u / ru) / (rv * s)
    return np.array([bi, -(bi + bk), bk])


def b_torsion(xi: np.ndarray, xj: np.ndarray, xk: np.ndarray, xl: np.ndarray) -> np.ndarray | None:
    """Wilson B vectors (4, 3) of the dihedral i-j-k-l (Blondel and Karplus, J. Comput. Chem.
    17 (1996) 1132); None when three of the atoms are collinear."""
    f, g, h = xi - xj, xj - xk, xl - xk
    a, b = np.cross(f, g), np.cross(h, g)
    a2, b2, gn = a @ a, b @ b, np.linalg.norm(g)
    if a2 < 1e-6 or b2 < 1e-6 or gn < 1e-6:
        return None
    fg, hg = (f @ g) / (a2 * gn), (h @ g) / (b2 * gn)
    bi, bl = -gn / a2 * a, gn / b2 * b
    bj = gn / a2 * a + fg * a - hg * b
    bk = -fg * a + hg * b - gn / b2 * b
    return np.array([bi, bj, bk, bl])


def _add_term(h: np.ndarray, atoms: list[int], b: np.ndarray, k: float) -> None:
    idx = (3 * np.asarray(atoms)[:, None] + np.arange(3)).ravel()
    v = b.ravel()
    h[np.ix_(idx, idx)] += k * np.outer(v, v)


def _bends(xyz, rho, nb):
    for j, near in enumerate(nb):
        for a, i in enumerate(near):
            for k in near[a + 1 :]:
                w = rho[i, j] * rho[j, k]
                if w > RHO_TERM and (b := b_bend(xyz[i], xyz[j], xyz[k])) is not None:
                    yield [i, j, k], b, K_BEND * w


def _torsions(xyz, rho, nb):
    for j, near in enumerate(nb):
        for k in near[near > j]:
            for i in near[near != k]:
                for m in nb[k][(nb[k] != j) & (nb[k] != i)]:
                    w = rho[i, j] * rho[j, k] * rho[k, m]
                    if w <= RHO_TERM:
                        continue
                    if (b := b_torsion(xyz[i], xyz[j], xyz[k], xyz[m])) is not None:
                        yield [i, j, k, m], b, K_TORSION * w


def lindh_hessian(z: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Lindh's model Hessian (3n, 3n), Hartree / Bohr^2, at `xyz` (Bohr)."""
    n = len(z)
    h = np.zeros((3 * n, 3 * n))
    rho = lindh_rho(z, xyz)
    for i, j in zip(*np.nonzero(np.triu(rho > RHO_TERM, 1)), strict=True):
        u = (xyz[i] - xyz[j]) / np.linalg.norm(xyz[i] - xyz[j])
        _add_term(h, [i, j], np.array([u, -u]), K_STRETCH * rho[i, j])
    nb = [np.flatnonzero(rho[j] > RHO_NEIGHBOUR) for j in range(n)]
    for atoms, b, k in (*_bends(xyz, rho, nb), *_torsions(xyz, rho, nb)):
        _add_term(h, atoms, b, k)
    return h


def rigid_motions(xyz: np.ndarray) -> np.ndarray:
    """Orthonormal translations and rotations of `xyz` (3n, 6), or (3n, 5) if it is linear."""
    n, c = len(xyz), xyz - xyz.mean(axis=0)
    vecs = [np.tile(e, n) for e in np.eye(3)] + [np.cross(e, c).ravel() for e in np.eye(3)]
    u, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def anc_basis(h_cart: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vectors (3n, m), eigenvalues (m,)) of `h_cart` without the rigid motions, eigenvalues
    floored at H_LOW; m = 3n - 6 (3n - 5 for a linear geometry)."""
    t = rigid_motions(xyz)
    p = np.eye(len(h_cart)) - t @ t.T
    shift = 1e3 + np.abs(h_cart).sum(axis=1).max()  # above every internal mode (Gershgorin)
    lam, vec = np.linalg.eigh(p @ h_cart @ p + shift * (t @ t.T))
    keep = np.argsort(lam)[: len(h_cart) - t.shape[1]]
    return vec[:, keep], np.maximum(lam[keep], H_LOW)


def rfo_step(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The rational-function step: the lowest eigenvector of [[H, g], [g, 0]], scaled."""
    m = len(g)
    aug = np.zeros((m + 1, m + 1))
    aug[:m, :m], aug[:m, m], aug[m, :m] = h, g, g
    v = np.linalg.eigh(aug)[1][:, 0]
    if abs(v[m]) < 1e-8:  # no usable eigenvector: a steepest-descent step
        return -g / np.maximum(np.diag(h), H_LOW)
    return v[:m] / v[m]


def _bfgs(h: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """BFGS update; skipped when the curvature condition fails (keeps H positive definite)."""
    sy, hs = s @ y, h @ s
    if sy <= 1e-10 or s @ hs <= 1e-12:
        return h
    return h + np.outer(y, y) / sy - np.outer(hs, hs) / (s @ hs)


class _State:
    """The current point of an optimization and its trust radius."""

    def __init__(self, fun: Objective, x0: np.ndarray, cfg: AncSettings) -> None:
        self.fun, self.cfg = fun, cfg
        self.x = np.asarray(x0, dtype=float).copy()
        self.e, self.g = fun(self.x)
        self.n_evals, self.radius = 1, cfg.radius

    @property
    def gmax(self) -> float:
        return float(np.abs(self.g).max())

    def done(self) -> bool:
        return self.gmax <= self.cfg.fmax or self.n_evals >= self.cfg.maxiter

    def step(self, vecs: np.ndarray, h: np.ndarray) -> np.ndarray:
        """One RFO step in the basis `vecs`; returns the updated Hessian."""
        cfg, gq = self.cfg, vecs.T @ self.g.ravel()
        s = rfo_step(h, gq)
        if (ns := np.linalg.norm(s)) > self.radius:
            s *= self.radius / ns
        predicted = gq @ s + 0.5 * s @ h @ s
        x_new = self.x + (vecs @ s).reshape(self.x.shape)
        e_new, g_new = self.fun(x_new)
        self.n_evals += 1
        de = e_new - self.e
        ratio = de / predicted if predicted < 0 else -1.0
        if de > cfg.uphill and ratio < 0:  # uphill: take it back, shrink the region
            self.radius = max(self.radius / 4, cfg.radius_min)
            return h
        h = _bfgs(h, s, vecs.T @ (g_new - self.g).ravel())
        if ratio > 0.75 and np.linalg.norm(s) > 0.8 * self.radius:
            self.radius = min(2 * self.radius, cfg.radius_max)
        elif ratio < 0.25:
            self.radius = max(self.radius / 4, cfg.radius_min)
        self.x, self.e, self.g = x_new, e_new, g_new
        return h


def optimize_anc(
    fun: Objective, z: np.ndarray, x0: np.ndarray, cfg: AncSettings | None = None
) -> AncResult:
    """Minimize `fun` from `x0` (Bohr, (n, 3)) for the atoms `z`."""
    cfg = cfg or AncSettings()
    state = _State(fun, x0, cfg)
    h_cart = lindh_hessian(np.asarray(z), state.x)
    while not state.done():
        vecs, lam = anc_basis(h_cart, state.x)
        h = np.diag(lam)
        for _ in range(cfg.micro):
            h = state.step(vecs, h)
            if state.done():
                break
        h_cart = vecs @ h @ vecs.T  # the learned curvature seeds the next basis
    return AncResult(state.x, state.gmax <= cfg.fmax, state.n_evals, state.gmax)
