"""Re-optimization along imaginary modes before the thermochemistry (`qm.thermo_cache`;
docs/decisions.md, 2026-11-10).

The records are L-BFGS-B optima at fmax 3e-4 Hartree/Bohr, and an optimizer started on a
symmetric geometry stays on it: some records are saddles of a methyl or ring torsion (acetone at
-76 cm^-1). xtb's rule (invert a mode above -20 cm^-1, drop one below) makes G jump when a mode
crosses the cutoff, so a mode below `ReoptSettings.threshold_cm` (-10 cm^-1; |nu| <= 1 stays a
numerical zero) is followed downhill instead, in rounds:

1. the geometry is displaced along the modes below the threshold: each mode's Cartesian
   displacement scaled to a largest atomic displacement of 1, their sum to `step_ang` (0.2 A),
   in both directions;
2. each displaced geometry is optimized on the record's own surface (GFN2, the record's ALPB
   solvent or the gas phase, its charge and unpaired count) by the ANC / RFO optimizer of
   `geometry.optimize_tblite_anc` at `fmax` (1e-4, three times tighter than the records'), the
   free run stopped once a bond of the drawing stays broken (`geometry.DissociationGuard`);
3. a direction is kept when it converged, breaks no bond of the drawing beyond 1.3x its
   covalent-radius sum (`geometry.bond_stretch_ratios`), forms no contact within 1.2x
   (`geometry.formed_bonds`) that the record's geometry lacks, and its energy (a single point
   with the record's SCF settings) is below the current one; the lower of the kept ones is the
   next geometry, whose Hessian the caller computes (`follow_imaginary`'s `evaluate`);
4. the rounds stop at a geometry with no mode below the threshold (MINIMUM untouched, else
   REOPTIMIZED) or with one left (PERSISTENT, why in `StopCode`): xtb's rule then applies to the
   last Hessian, as before; a Hessian that fails at a kept geometry leaves the last good one
   (REOPT_HESSIAN_FAILED). The record itself is never changed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from rdkit import Chem
from tblite.interface import TBLiteRuntimeError

from ksmech.qm.embed import prepare_mol
from ksmech.qm.geometry import (
    BOHR_ANG,
    FORMED_FACTOR,
    SCF_MAXITER,
    DissociationGuard,
    bond_stretch_ratios,
    formed_bonds,
    new_calculator,
    optimize_tblite_anc,
)
from ksmech.qm.records import QMError, QMRecord, ReasonCode
from ksmech.qm.xtb_calc import QMConfig, solvent_of

log = logging.getLogger(__name__)

REOPT_THRESHOLD_CM = -10.0
REOPT_ROUNDS = 3
REOPT_STEP_ANG = 0.2
REOPT_FMAX = 1e-4
REOPT_MAXITER = 1000  # energies per re-optimization (the ANC optimizer counts energies)
REOPT_OPTIMIZER = "tblite-anc"
ZERO_CM = 1.0  # |nu| at or below it is a numerical zero (`thermo.ThermoSettings.zero_cutoff_cm`)
ROUND_COST = 2.0  # a round's time, estimated as this x the last Hessian's
NOT_LOWER = "NOT_LOWER"  # a converged direction whose energy is not below the current one


class Outcome(StrEnum):
    MINIMUM = "MINIMUM"  # the record's Hessian has no mode below the threshold: untouched
    REOPTIMIZED = "REOPTIMIZED"  # re-optimized; no mode below the threshold is left
    PERSISTENT = "PERSISTENT"  # a mode below the threshold remains: xtb's rule on it


class StopCode(StrEnum):
    """Why the rounds ended with a mode below the threshold left."""

    OFF = "REOPT_OFF"  # rounds = 0
    ROUNDS = "REOPT_ROUNDS"  # the last round's Hessian still has one
    NO_LOWER_MINIMUM = "REOPT_NO_LOWER_MINIMUM"  # neither direction was kept
    BUDGET = "REOPT_BUDGET"  # the time left is short of another round
    HESSIAN_FAILED = "REOPT_HESSIAN_FAILED"  # a kept geometry's Hessian failed: the last good one
    NO_GRAPH = "REOPT_NO_GRAPH"  # the record's drawing does not give its atoms: no guard


@dataclass(frozen=True)
class ReoptSettings:
    threshold_cm: float = REOPT_THRESHOLD_CM
    rounds: int = REOPT_ROUNDS
    step_ang: float = REOPT_STEP_ANG
    fmax: float = REOPT_FMAX
    maxiter: int = REOPT_MAXITER  # a budget
    scf_maxiter: int = SCF_MAXITER  # a budget
    broken_factor: float = QMConfig.conn_cov_factor  # the QM run's 1.3
    formed_factor: float = FORMED_FACTOR

    def __post_init__(self) -> None:
        if not self.threshold_cm < -ZERO_CM:
            raise ValueError(f"the threshold must be below -{ZERO_CM} cm^-1: {self.threshold_cm}")
        if self.rounds < 0 or self.step_ang <= 0 or self.fmax <= 0:
            raise ValueError(f"rounds >= 0, step and fmax > 0: {self}")


def identity(settings: ReoptSettings) -> dict[str, Any]:
    """The settings a sidecar's content depends on (the budgets apart)."""
    return {
        "reopt_threshold_cm": float(settings.threshold_cm),
        "reopt_rounds": int(settings.rounds),
        "reopt_step_ang": float(settings.step_ang),
        "reopt_fmax": float(settings.fmax),
        "reopt_optimizer": REOPT_OPTIMIZER,
        "reopt_broken_factor": float(settings.broken_factor),
        "reopt_formed_factor": float(settings.formed_factor),
    }


@dataclass(frozen=True)
class Point:
    """A geometry with its energy and its Hessian's normal modes (`follow_imaginary`)."""

    xyz_ang: np.ndarray
    energy_ha: float
    frequencies_cm: np.ndarray  # ascending, negative = imaginary
    modes: np.ndarray  # (3N, n_modes) mass-weighted Cartesian normal modes as columns
    seconds: float  # the time its evaluation took
    info: Any = None  # the evaluator's own results (symmetry, gradient, ...)


@dataclass(frozen=True)
class ReoptResult:
    points: tuple[Point, ...]  # every evaluated geometry: the record's first, the kept last
    outcome: Outcome
    stop: str  # a StopCode for PERSISTENT, else ""
    rejected: tuple[str, ...]  # every direction not kept, by reason
    opt_seconds: float  # the re-optimizations' time

    @property
    def first(self) -> Point:
        return self.points[0]

    @property
    def final(self) -> Point:
        return self.points[-1]

    @property
    def rounds(self) -> int:
        return len(self.points) - 1


@dataclass(frozen=True)
class BondState:
    broken: frozenset  # drawn bonds beyond broken_factor x r_cov
    formed: frozenset  # undrawn pairs within formed_factor x r_cov


def below(point: Point, settings: ReoptSettings) -> np.ndarray:
    """Indices of the modes below the threshold."""
    return np.flatnonzero(np.asarray(point.frequencies_cm) < settings.threshold_cm)


def _largest_atom(v: np.ndarray) -> float:
    return float(np.linalg.norm(v.reshape(-1, 3), axis=1).max())


def mode_displacement(modes: np.ndarray, masses: np.ndarray, step_ang: float) -> np.ndarray:
    """(N, 3) Angstrom: the sum of the mass-weighted modes' Cartesian displacements, each scaled
    to a largest atomic displacement of 1, scaled to `step_ang`. In mass-weighted coordinates the
    sum has a positive component on every mode, so it leaves each saddle at once."""
    inv = 1.0 / np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    total = np.zeros(len(inv))
    for k in range(modes.shape[1]):
        v = inv * modes[:, k]
        total += v / _largest_atom(v)
    return (total * (step_ang / _largest_atom(total))).reshape(-1, 3)


def record_graph(record: QMRecord) -> Chem.Mol | None:
    """The record's drawing in its atom order (`embed.prepare_mol`), None when its atoms are
    not the record's."""
    try:
        mol = prepare_mol(record.atom_order_smiles)
    except QMError:
        return None
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return mol if z == [int(x) for x in record.atomic_numbers] else None


def bond_state(mol: Chem.Mol, xyz_ang: np.ndarray, settings: ReoptSettings) -> BondState:
    broken = ((i, j) for i, j, r in bond_stretch_ratios(mol, xyz_ang) if r > settings.broken_factor)
    formed = ((i, j) for i, j, _ in formed_bonds(mol, xyz_ang, settings.formed_factor))
    return BondState(frozenset(broken), frozenset(formed))


def check_bonds(mol: Chem.Mol, base: BondState, xyz_ang: np.ndarray, settings) -> None:
    """CONNECTIVITY_CHANGED for a bond broken, BOND_FORMED for a contact formed, relative to
    the record's geometry (`base`)."""
    now = bond_state(mol, xyz_ang, settings)
    if broken := sorted(now.broken - base.broken):
        raise QMError(ReasonCode.CONNECTIVITY_CHANGED, f"bonds broken {broken}")
    if formed := sorted(now.formed - base.formed):
        raise QMError(ReasonCode.BOND_FORMED, f"contacts formed {formed}")


def single_point_energy(record: QMRecord, xyz_bohr: np.ndarray, scf_maxiter: int) -> float:
    """The GFN2 energy at `xyz_bohr` with the record's calculator and SCF settings."""
    solvent = solvent_of(record.method)
    calc = new_calculator(
        record.atomic_numbers, xyz_bohr, record.charge, record.unpaired, solvent,
        scf_maxiter=scf_maxiter,
    )  # fmt: skip
    try:
        return float(calc.singlepoint().get("energy"))
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, f"re-optimized geometry: {exc}") from exc


def relax_direction(
    record: QMRecord, mol: Chem.Mol, base: BondState, start_ang: np.ndarray, settings
) -> tuple[np.ndarray, float]:
    """(Angstrom geometry, energy) of one displaced start, optimized and checked; QMError
    (OPT_NOT_CONVERGED, SCF_NOT_CONVERGED, CONNECTIVITY_CHANGED, BOND_FORMED) otherwise."""
    guard = None if base.broken else DissociationGuard(mol, settings.broken_factor)
    xyz_bohr = optimize_tblite_anc(
        record.atomic_numbers,
        np.asarray(start_ang, dtype=float) / BOHR_ANG,
        record.charge,
        record.unpaired,
        settings.fmax,
        settings.maxiter,
        solvent_of(record.method),
        None,
        settings.scf_maxiter,
        guard,
    )
    check_bonds(mol, base, xyz_bohr * BOHR_ANG, settings)
    return xyz_bohr * BOHR_ANG, single_point_energy(record, xyz_bohr, settings.scf_maxiter)


def lower_minimum(
    record: QMRecord, mol: Chem.Mol, base: BondState, point: Point, masses, settings
) -> tuple[tuple[np.ndarray, float] | None, list[str]]:
    """(the lower kept minimum of the two directions or None, the reasons of the others)."""
    step = mode_displacement(point.modes[:, below(point, settings)], masses, settings.step_ang)
    kept, rejected = [], []
    for sign in (1.0, -1.0):
        try:
            xyz, energy = relax_direction(record, mol, base, point.xyz_ang + sign * step, settings)
        except QMError as exc:
            rejected.append(str(exc.reason))
            continue
        if energy < point.energy_ha:
            kept.append((xyz, energy))
        else:
            rejected.append(NOT_LOWER)
    return (min(kept, key=lambda k: k[1]) if kept else None), rejected


def _no_round(n_rounds: int, point: Point, settings, deadline: float | None) -> str:
    """The StopCode that forbids another round, or ''."""
    if settings.rounds == 0:
        return str(StopCode.OFF)
    if n_rounds >= settings.rounds:
        return str(StopCode.ROUNDS)
    if deadline is not None and time.monotonic() + ROUND_COST * point.seconds > deadline:
        return str(StopCode.BUDGET)
    return ""


def follow_imaginary(
    record: QMRecord,
    evaluate: Callable[[np.ndarray, float], Point],
    masses: np.ndarray,
    settings: ReoptSettings,
    deadline: float | None = None,
) -> ReoptResult:
    """The rounds of the module docstring from the record's geometry. `evaluate(xyz_ang,
    energy_ha)` computes a geometry's Hessian; `deadline` (time.monotonic) ends the rounds
    early (BUDGET) when the next one would likely pass it."""
    points = [evaluate(np.asarray(record.coords_ang, dtype=float), float(record.energy_ha))]
    rejected, opt_seconds, stop = [], 0.0, ""
    mol = base = None
    while not stop and len(below(points[-1], settings)):
        stop = _no_round(len(points) - 1, points[-1], settings, deadline)
        if not stop and mol is None:
            mol = record_graph(record)
            stop = "" if mol is not None else str(StopCode.NO_GRAPH)
        if stop:
            break
        if base is None:
            base = bond_state(mol, points[0].xyz_ang, settings)
        t0 = time.perf_counter()
        found, why = lower_minimum(record, mol, base, points[-1], masses, settings)
        opt_seconds += time.perf_counter() - t0
        rejected += why
        if found is None:
            stop = str(StopCode.NO_LOWER_MINIMUM)
            continue
        try:
            points.append(evaluate(*found))
        except QMError as exc:  # the last good Hessian stays, the record's at worst
            log.warning("Hessian of %s's re-optimized geometry failed: %s", record.smiles, exc)
            rejected.append(str(exc.reason))
            stop = str(StopCode.HESSIAN_FAILED)
    if stop:
        outcome = Outcome.PERSISTENT
    else:
        outcome = Outcome.REOPTIMIZED if len(points) > 1 else Outcome.MINIMUM
    return ReoptResult(tuple(points), outcome, stop, tuple(rejected), opt_seconds)
