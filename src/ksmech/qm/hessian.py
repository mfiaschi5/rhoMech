"""Semi-numerical Hessian and harmonic frequencies from tblite gradients (reaction-network plan,
item B; docs/decisions.md, 2026-11-09).

The Hessian is the central difference of tblite's analytic GFN2 gradients, H[k] =
(g(x + h e_k) - g(x - h e_k)) / 2h with h = 0.005 Bohr (xtb's `step_hess`), at the record's
geometry and with the record's calculator: GFN2, its ALPB solvent (or the gas phase), charge and
unpaired count (a species `qm.spin` reassigns carries the assigned count in its record; tblite
runs an open shell restricted with fractional occupations, as for the record). Every displaced
SCF starts from the reference point's converged wavefunction (from the guess when that start
fails), at SCC accuracy 0.2 (xtb's `--hess`). The Hessian is symmetrized.

Frequencies: the Cartesian Hessian is mass-weighted with the average atomic masses (RDKit's
table, as `qm.point_group`), the translations and rotations (rigid-body vectors in mass-weighted
coordinates, five for a linear molecule) are removed by transforming to their orthogonal
complement, the 3N-6 (3N-5) dimensional vibrational subspace, which is then diagonalized: the
same as xtb's projection, with the mode count exact by construction. A frequency is
sqrt(lambda) / 2 pi c in cm^-1, negative for a negative eigenvalue (an imaginary mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tblite.interface import Calculator, TBLiteRuntimeError

from ksmech.qm.geometry import SCF_MAXITER, new_calculator
from ksmech.qm.records import QMError, ReasonCode

FD_STEP_BOHR = 0.005  # xtb's step_hess
HESS_ACCURACY = 0.2  # xtb's SCC accuracy for --hess
# sqrt(Hartree / (Bohr^2 amu)) in cm^-1: sqrt(E_h / (a0^2 u)) / (2 pi c)
AU_TO_CM = 5140.4871
ROTATIONS_LINEAR = 2


@dataclass(frozen=True)
class HessianResult:
    hessian: np.ndarray  # (3N, 3N) Hartree / Bohr^2, Cartesian, symmetrized
    gradient: np.ndarray  # (N, 3) Hartree / Bohr at the reference geometry
    asymmetry: float  # max |H - H^T| before symmetrizing
    n_gradients: int
    fresh_starts: int  # displaced SCFs that failed from the reference wavefunction


def atomic_masses(z: np.ndarray) -> np.ndarray:
    """Average atomic masses (amu), RDKit's periodic table."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    return np.array([table.GetAtomicWeight(int(zi)) for zi in np.asarray(z)], dtype=float)


def _hessian_calculator(z, xyz_bohr, charge, unpaired, solvent, scf_maxiter, accuracy):
    calc = new_calculator(z, xyz_bohr, charge, unpaired, solvent, scf_maxiter=scf_maxiter)
    calc.set("accuracy", float(accuracy))
    return calc


def _gradient(calc: Calculator, xyz_bohr: np.ndarray, ref) -> tuple[np.ndarray, bool]:
    """(gradient at `xyz_bohr`, whether the SCF had to start from the guess)."""
    calc.update(positions=xyz_bohr)
    try:
        return np.array(calc.singlepoint(ref, copy=True).get("gradient")), False
    except TBLiteRuntimeError:
        pass
    try:
        return np.array(calc.singlepoint().get("gradient")), True
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, f"displaced geometry: {exc}") from exc


def fd_hessian(
    z: np.ndarray,
    xyz_bohr: np.ndarray,
    charge: int,
    unpaired: int,
    solvent: str | None,
    step: float = FD_STEP_BOHR,
    scf_maxiter: int = SCF_MAXITER,
    accuracy: float = HESS_ACCURACY,
) -> HessianResult:
    """The central-difference Hessian of the GFN2 energy (module docstring). Raises
    QMError(SCF_NOT_CONVERGED) when an SCF fails from both starts."""
    x0 = np.asarray(xyz_bohr, dtype=float).reshape(-1, 3)
    calc = _hessian_calculator(z, x0, charge, unpaired, solvent, scf_maxiter, accuracy)
    try:
        ref = calc.singlepoint()
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, f"reference geometry: {exc}") from exc
    g0 = np.array(ref.get("gradient"))
    n3 = x0.size
    hess, fresh = np.zeros((n3, n3)), 0
    for k in range(n3):
        rows = []
        for sign in (1.0, -1.0):
            x = x0.copy().reshape(-1)
            x[k] += sign * step
            g, restarted = _gradient(calc, x.reshape(-1, 3), ref)
            rows.append(g.reshape(-1))
            fresh += restarted
        hess[k] = (rows[0] - rows[1]) / (2.0 * step)
    asym = float(np.abs(hess - hess.T).max()) if n3 else 0.0
    return HessianResult(0.5 * (hess + hess.T), g0, asym, 2 * n3 + 1, fresh)


def rigid_body_basis(xyz: np.ndarray, masses: np.ndarray, linear: bool) -> np.ndarray:
    """(3N, k) orthonormal basis of the mass-weighted translations and rotations: k = 3 for an
    atom, 5 for a linear molecule, else 6 (the k largest singular directions of the six
    rigid-body vectors)."""
    x = np.asarray(xyz, dtype=float).reshape(-1, 3)
    m = np.asarray(masses, dtype=float)
    r = x - (m[:, None] * x).sum(0) / m.sum()
    sq = np.sqrt(m)[:, None]
    vecs = []
    for a in range(3):
        t = np.zeros_like(x)
        t[:, a] = 1.0
        vecs.append((t * sq).reshape(-1))
    for a in range(3):
        vecs.append((np.cross(np.eye(3)[a], r) * sq).reshape(-1))
    k = 3 if len(x) == 1 else 3 + (ROTATIONS_LINEAR if linear else 3)
    u, _, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, :k]


def vibrational_subspace(rigid: np.ndarray) -> np.ndarray:
    """(3N, 3N - k) orthonormal complement of the rigid-body basis."""
    q, _ = np.linalg.qr(rigid, mode="complete")
    return q[:, rigid.shape[1] :]


def mass_weighted(hessian: np.ndarray, masses: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    return hessian * np.outer(inv, inv)


def frequencies(
    hessian: np.ndarray, xyz: np.ndarray, masses: np.ndarray, linear: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies in cm^-1 ascending, negative = imaginary; their mass-weighted Cartesian
    normal modes as columns) of the vibrational subspace: 3N-6, 3N-5 or none."""
    rigid = rigid_body_basis(xyz, masses, linear)
    basis = vibrational_subspace(rigid)
    if not basis.shape[1]:
        return np.zeros(0), np.zeros((len(masses) * 3, 0))
    h_int = basis.T @ mass_weighted(hessian, masses) @ basis
    lam, vec = np.linalg.eigh(0.5 * (h_int + h_int.T))
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * AU_TO_CM
    return nu, basis @ vec


def principal_moments(xyz_ang: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Principal moments of inertia (amu A^2), ascending."""
    x = np.asarray(xyz_ang, dtype=float).reshape(-1, 3)
    m = np.asarray(masses, dtype=float)
    r = x - (m[:, None] * x).sum(0) / m.sum()
    t = np.eye(3) * (m * (r * r).sum(1)).sum() - np.einsum("i,ij,ik->jk", m, r, r)
    return np.linalg.eigvalsh(t)
