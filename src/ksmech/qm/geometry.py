"""Geometry optimization backends and the post-optimization connectivity check.

tblite has no optimizer, so the default backend runs scipy L-BFGS-B on tblite energies and
gradients (Cartesian, Bohr). The other backend shells out to the `xtb` binary (`--opt --gfn 2`).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from tblite.interface import Calculator, TBLiteRuntimeError

from ksmech.qm.records import QMError, ReasonCode

BOHR_ANG = 0.529177210903
_XTB_TIMEOUT_S = 600
# Bonds to these ionic/very electropositive elements are not tested: their contacts with the
# rest of the molecule are electrostatic, and covalent-radius ratios mean little for them.
SKIP_BOND_ELEMENTS = frozenset({3, 11, 12, 19})  # Li, Na, Mg, K


def new_calculator(
    z: np.ndarray,
    xyz_bohr: np.ndarray,
    charge: int,
    unpaired: int,
    solvent: str | None = None,
    save_integrals: bool = False,
) -> Calculator:
    calc = Calculator("GFN2-xTB", z, xyz_bohr, charge=float(charge), uhf=int(unpaired))
    calc.set("verbosity", 0)
    if solvent:
        calc.add("alpb-solvation", solvent)
    if save_integrals:
        calc.set("save-integrals", 1)
    return calc


@dataclass(frozen=True)
class WallRestraint:
    """Flat-bottom harmonic walls E = k * max(0, d - d0)^2 on selected atom pairs."""

    pairs: np.ndarray  # (m, 2) atom indices
    d0_bohr: np.ndarray  # (m,) wall onset distances
    k: float  # Hartree / Bohr^2

    def energy_gradient(self, xyz_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(xyz_bohr)
        if not len(self.pairs):
            return 0.0, grad
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        vec = xyz_bohr[i] - xyz_bohr[j]
        dist = np.linalg.norm(vec, axis=1)
        over = np.maximum(dist - self.d0_bohr, 0.0)
        pull = (2.0 * self.k * over / np.maximum(dist, 1e-12))[:, None] * vec
        np.add.at(grad, i, pull)
        np.add.at(grad, j, -pull)
        return float(self.k * (over**2).sum()), grad


def bond_walls(mol: Chem.Mol, wall_factor: float, k: float) -> WallRestraint:
    """Walls on every input bond (Li, Na, Mg, K bonds excluded) at `wall_factor` x radii sum."""
    table = Chem.GetPeriodicTable()
    pairs, d0 = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() in SKIP_BOND_ELEMENTS or b.GetAtomicNum() in SKIP_BOND_ELEMENTS:
            continue
        pairs.append((a.GetIdx(), b.GetIdx()))
        radii = table.GetRcovalent(a.GetAtomicNum()) + table.GetRcovalent(b.GetAtomicNum())
        d0.append(wall_factor * radii / BOHR_ANG)
    return WallRestraint(np.array(pairs, dtype=int).reshape(-1, 2), np.array(d0), k)


def optimize_tblite(
    z: np.ndarray,
    xyz_bohr: np.ndarray,
    charge: int,
    unpaired: int,
    fmax: float,
    maxiter: int,
    solvent: str | None = None,
    walls: WallRestraint | None = None,
) -> np.ndarray:
    """L-BFGS-B on GFN2 energies and gradients (plus optional bond walls).

    Returns Bohr coordinates, shape (n, 3). Convergence is on the total gradient.
    """
    if len(z) == 1:
        return xyz_bohr.copy()
    calc = new_calculator(z, xyz_bohr, charge, unpaired, solvent)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        calc.update(positions=flat.reshape(-1, 3))
        res = calc.singlepoint()
        energy, grad = float(res.get("energy")), np.asarray(res.get("gradient"))
        if walls is not None:
            e_wall, g_wall = walls.energy_gradient(flat.reshape(-1, 3))
            energy, grad = energy + e_wall, grad + g_wall
        return energy, grad.ravel()

    try:
        out = minimize(
            objective,
            xyz_bohr.ravel(),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=maxiter, maxfun=2 * maxiter, gtol=fmax, ftol=1e-12),
        )
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, f"during optimization: {exc}") from exc
    if np.abs(out.jac).max() > fmax:
        raise QMError(
            ReasonCode.OPT_NOT_CONVERGED,
            f"max|grad|={np.abs(out.jac).max():.2e} after {out.nit} iterations ({out.message})",
        )
    return out.x.reshape(-1, 3)


def _write_xyz(path: Path, z: np.ndarray, xyz_ang: np.ndarray) -> None:
    sym = Chem.GetPeriodicTable()
    lines = [str(len(z)), "ksmech"]
    lines += [
        f"{sym.GetElementSymbol(int(a))} {x:.10f} {y:.10f} {w:.10f}"
        for a, (x, y, w) in zip(z, xyz_ang, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


def optimize_xtb(
    z: np.ndarray, xyz_bohr: np.ndarray, charge: int, unpaired: int, solvent: str | None = None
) -> np.ndarray:
    """Optimize with the `xtb` binary (`--opt --gfn 2`). Returns Bohr coordinates."""
    exe = shutil.which("xtb")
    if exe is None:
        raise QMError(ReasonCode.XTB_NOT_FOUND, "optimizer='xtb' but no xtb executable on PATH")
    if len(z) == 1:
        return xyz_bohr.copy()
    with tempfile.TemporaryDirectory(prefix="ksmech-xtb-") as tmp:
        work = Path(tmp)
        _write_xyz(work / "in.xyz", z, xyz_bohr * BOHR_ANG)
        cmd = [exe, "in.xyz", "--opt", "--gfn", "2", "--chrg", str(charge), "--uhf", str(unpaired)]
        if solvent:
            cmd += ["--alpb", solvent]
        try:
            proc = subprocess.run(
                cmd, cwd=work, capture_output=True, text=True, timeout=_XTB_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise QMError(ReasonCode.TIMEOUT, "xtb --opt timed out") from exc
        out = work / "xtbopt.xyz"
        if proc.returncode != 0 or not out.is_file() or "FAILED TO CONVERGE" in proc.stdout:
            raise QMError(ReasonCode.OPT_NOT_CONVERGED, f"xtb exit {proc.returncode}")
        rows = [ln.split()[1:4] for ln in out.read_text().splitlines()[2:] if ln.strip()]
    return np.array(rows, dtype=float) / BOHR_ANG


def bond_stretch_ratios(mol: Chem.Mol, xyz_ang: np.ndarray) -> list[tuple[int, int, float]]:
    """(i, j, d_ij / (r_cov_i + r_cov_j)) for every input bond, skipping Li, Na, Mg and K bonds."""
    table = Chem.GetPeriodicTable()
    out = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        zi, zj = bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()
        if zi in SKIP_BOND_ELEMENTS or zj in SKIP_BOND_ELEMENTS:
            continue
        d = float(np.linalg.norm(xyz_ang[i] - xyz_ang[j]))
        out.append((i, j, d / (table.GetRcovalent(zi) + table.GetRcovalent(zj))))
    return out


def check_connectivity(mol: Chem.Mol, xyz_ang: np.ndarray, cov_factor: float) -> None:
    """Raise CONNECTIVITY_CHANGED if an input bond stretched beyond `cov_factor` x the sum of
    covalent radii. New close contacts are ignored.
    """
    broken = [(i, j, r) for i, j, r in bond_stretch_ratios(mol, xyz_ang) if r > cov_factor]
    if broken:
        shown = ", ".join(f"({i},{j}) {r:.2f}x" for i, j, r in broken[:6])
        raise QMError(
            ReasonCode.CONNECTIVITY_CHANGED, f"{len(broken)} bond(s) beyond {cov_factor}x: {shown}"
        )
