"""Geometry optimization backends, the post-optimization connectivity check and the two laws of
a tagged species' optimization (`kick`, `formed_bonds`; open-shell plan P4).

tblite has no optimizer, so the default backend runs scipy L-BFGS-B on tblite energies and
gradients (Cartesian, Bohr), each SCF from the guess. `optimize_tblite_anc` runs the ANC / RFO
steps of `speedchem.anc` instead, each SCF restarted from the previous point's wavefunction
(`TbliteEnergy`), and may stop a free run as soon as a bond has dissociated
(`DissociationGuard`). The other backend shells out to the `xtb` binary (`--opt --gfn 2`).
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize
from speedchem import bonds as _bonds
from speedchem.anc import AncSettings, optimize_anc
from speedchem.bonds import (  # noqa: F401 - re-exported: rhoMech's callers import them from here
    DISSOCIATED_PATIENCE,
    SKIP_BOND_ELEMENTS,
    WallRestraint,
    bond_stretch_ratios,
    bond_walls,
)
from speedchem.reasons import ConnectivityChanged
from speedchem.units import BOHR_ANG
from tblite.interface import Calculator, TBLiteRuntimeError

from ksmech.qm.records import QMError, ReasonCode

_XTB_TIMEOUT_S = 600
SCF_MAXITER = 250  # tblite's own default SCF cycle limit
# the fixed symmetry-breaking displacement of P2's scan (scripts/probes/spin_scan.py)
KICK_ANG = 0.02
KICK_SEED = 0
FORMED_FACTOR = 1.2  # a non-bonded pair closer than this x r_cov sum is a formed bond (the walls)


def new_calculator(
    z: np.ndarray,
    xyz_bohr: np.ndarray,
    charge: int,
    unpaired: int,
    solvent: str | None = None,
    save_integrals: bool = False,
    scf_maxiter: int = SCF_MAXITER,
) -> Calculator:
    calc = Calculator("GFN2-xTB", z, xyz_bohr, charge=float(charge), uhf=int(unpaired))
    calc.set("verbosity", 0)
    calc.set("max-iter", int(scf_maxiter))  # a convergence limit, not part of the method
    if solvent:
        calc.add("alpb-solvation", solvent)
    if save_integrals:
        calc.set("save-integrals", 1)
    return calc


def optimize_tblite(
    z: np.ndarray,
    xyz_bohr: np.ndarray,
    charge: int,
    unpaired: int,
    fmax: float,
    maxiter: int,
    solvent: str | None = None,
    walls: WallRestraint | None = None,
    scf_maxiter: int = SCF_MAXITER,
) -> np.ndarray:
    """L-BFGS-B on GFN2 energies and gradients (plus optional bond walls).

    Returns Bohr coordinates, shape (n, 3). Convergence is on the total gradient.
    """
    if len(z) == 1:
        return xyz_bohr.copy()
    calc = new_calculator(z, xyz_bohr, charge, unpaired, solvent, scf_maxiter=scf_maxiter)

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


class TbliteEnergy:
    """GFN2 energy and gradient along one optimization (`qm.anc`'s objective): each SCF starts
    from the previous point's wavefunction, or from the guess when that start fails, plus the
    optional bond walls. A `guard` sees every trial geometry first and may raise a QMError."""

    def __init__(
        self,
        z: np.ndarray,
        xyz_bohr: np.ndarray,
        charge: int,
        unpaired: int,
        solvent: str | None,
        scf_maxiter: int = SCF_MAXITER,
        walls: WallRestraint | None = None,
        guard: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.calc = new_calculator(z, xyz_bohr, charge, unpaired, solvent, scf_maxiter=scf_maxiter)
        self.walls, self.guard = walls, guard
        self.result = None  # the last point's tblite Result: the next SCF's start
        self.fresh_starts = 0

    def _singlepoint(self):
        try:
            return self.calc.singlepoint(self.result)
        except TBLiteRuntimeError:
            if self.result is None:
                raise
            self.result, self.fresh_starts = None, self.fresh_starts + 1
            return self.calc.singlepoint()

    def __call__(self, xyz_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        if self.guard is not None:
            self.guard(xyz_bohr)
        self.calc.update(positions=xyz_bohr)
        self.result = self._singlepoint()
        energy, grad = float(self.result.get("energy")), np.array(self.result.get("gradient"))
        if self.walls is not None:
            e_wall, g_wall = self.walls.energy_gradient(xyz_bohr)
            energy, grad = energy + e_wall, grad + g_wall
        return energy, grad


class DissociationGuard(_bonds.DissociationGuard):
    """`speedchem.bonds.DissociationGuard` raising `QMError(CONNECTIVITY_CHANGED)`, as every
    caller here expects (a `TbliteEnergy` guard)."""

    def __call__(self, xyz_bohr: np.ndarray) -> None:
        try:
            super().__call__(xyz_bohr)
        except ConnectivityChanged as exc:
            raise QMError(ReasonCode.CONNECTIVITY_CHANGED, exc.detail) from None


def optimize_tblite_anc(
    z: np.ndarray,
    xyz_bohr: np.ndarray,
    charge: int,
    unpaired: int,
    fmax: float,
    maxiter: int,
    solvent: str | None = None,
    walls: WallRestraint | None = None,
    scf_maxiter: int = SCF_MAXITER,
    guard: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """ANC / RFO steps (`qm.anc`) on GFN2 energies and gradients, each SCF restarted from the
    previous point, plus optional bond walls and a guard. Returns Bohr coordinates (n, 3)."""
    if len(z) == 1:
        return xyz_bohr.copy()
    energy = TbliteEnergy(z, xyz_bohr, charge, unpaired, solvent, scf_maxiter, walls, guard)
    try:
        out = optimize_anc(energy, z, xyz_bohr, AncSettings(fmax=fmax, maxiter=maxiter))
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, f"during optimization: {exc}") from exc
    if not out.converged:
        raise QMError(
            ReasonCode.OPT_NOT_CONVERGED,
            f"max|grad|={out.gmax:.2e} after {out.n_evals} energies (ANC)",
        )
    return out.xyz


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


def check_connectivity(mol: Chem.Mol, xyz_ang: np.ndarray, cov_factor: float) -> None:
    """`speedchem.bonds.check_connectivity`: `QMError(CONNECTIVITY_CHANGED)` if an input bond
    stretched beyond `cov_factor` x the sum of covalent radii. New close contacts are ignored."""
    try:
        _bonds.check_connectivity(mol, xyz_ang, cov_factor)
    except ConnectivityChanged as exc:
        raise QMError(ReasonCode.CONNECTIVITY_CHANGED, exc.detail) from None


def kick(shape: tuple[int, ...]) -> np.ndarray:
    """The fixed symmetry-breaking displacement (Bohr): normal, sigma KICK_ANG, seed KICK_SEED,
    the second start of P2's spin scan."""
    return np.random.default_rng(KICK_SEED).normal(0.0, KICK_ANG / BOHR_ANG, shape)


def _bond_tested(atom: Chem.Atom) -> bool:
    """Neither a transition metal (the coordination sphere of a ligand-truncated fragment moves
    freely) nor Li, Na, Mg or K (`SKIP_BOND_ELEMENTS`)."""
    from ksmech.data import domain

    d = domain.d_count(atom)
    tm = d is not None and 0 <= d <= 10
    return not tm and atom.GetAtomicNum() not in SKIP_BOND_ELEMENTS


def formed_bonds(
    mol: Chem.Mol, xyz_ang: np.ndarray, factor: float = FORMED_FACTOR
) -> list[tuple[int, int, float]]:
    """(i, j, d / r_cov sum) of the pairs not bonded in the drawing (graph distance 2 or more)
    that end within `factor` x their covalent-radius sum, where the bond walls start: a bond the
    optimization formed, which `check_connectivity` ignores (P2's BOND_FORMED law,
    scripts/probes/open_shell_probe.py). Metal-ligand contacts and Li, Na, Mg and K are left
    out (`_bond_tested`)."""
    table, dist = Chem.GetPeriodicTable(), Chem.GetDistanceMatrix(mol)
    keep = [a.GetIdx() for a in mol.GetAtoms() if _bond_tested(a)]
    out = []
    for i, j in itertools.combinations(keep, 2):
        if dist[i, j] < 2:
            continue
        zi, zj = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        rcov = table.GetRcovalent(zi) + table.GetRcovalent(zj)
        r = float(np.linalg.norm(xyz_ang[i] - xyz_ang[j])) / rcov
        if r < factor:
            out.append((i, j, round(r, 3)))
    return out
