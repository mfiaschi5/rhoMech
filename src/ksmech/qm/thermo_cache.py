"""The thermochemistry sidecar of a QM record: harmonic frequencies, principal moments, symmetry
number and multiplicity, so G(T) is evaluated at any T without the Hessian (`qm.thermo`;
reaction-network plan, item B; docs/decisions.md, 2026-11-09).

One file per record, `data/cache/thermo/<cache key>.npz`, keyed like the record (its map-free
SMILES, conformer and method, spin tag included) and stamped with the record as its input, as the
Wiberg sidecars are (`qm.bond_index`). The header's config holds the record's method and
conformer and every setting of the Hessian (`ThermoConfig`: step, SCC accuracy, masses, symmetry
tolerance) and of the re-optimization along imaginary modes (`qm.thermo_reopt`: threshold,
rounds, step, fmax, optimizer, bond guard); a sidecar whose record or settings changed is stale,
and so is one written before the re-optimization existed (its header lacks those keys). The
thermochemistry settings (qRRHO cutoff, imaginary-mode rule, standard state) are applied when G
is evaluated and are not part of the file.

The species' thermo minimum is the record's geometry, or, when its Hessian has a mode below the
threshold, the lower minimum `qm.thermo_reopt.follow_imaginary` reaches from it: the sidecar holds
its geometry (`coords_ang`), energy (`energy_ha`, the E_el of G), frequencies, moments and
symmetry number, beside the record's energy and frequencies, the rounds, the energy drop and the
outcome. The record file is only read: tokens are built on its geometry.

A failure that a rerun with the same settings would repeat (an SCF that does not converge at a
displaced geometry, a timeout) leaves a tombstone `<cache key>.failed.json` beside the sidecars,
in the format of the QM failure memo (`qm.failures`) under the kind `thermo_failure`; a larger
budget (`timeout_s`, `scf_maxiter`) or another setting runs again, and a success removes it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, fields
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ksmech.data import provenance as prov
from ksmech.qm import blas, failures
from ksmech.qm import point_group as pg
from ksmech.qm import thermo_reopt as ro
from ksmech.qm.geometry import BOHR_ANG, SCF_MAXITER
from ksmech.qm.hessian import (
    FD_STEP_BOHR,
    HESS_ACCURACY,
    atomic_masses,
    fd_hessian,
    frequencies,
    principal_moments,
)
from ksmech.qm.records import (
    DEFAULT_CACHE_DIR,
    QMRecord,
    cache_key,
    is_valid_cache_file,
    record_config,
    record_inputs,
)
from ksmech.qm.thermo import (
    DEFAULTS,
    HARTREE_KCAL,
    MolecularThermo,
    ThermoSettings,
    standard_free_energy,
    usable_modes,
)
from ksmech.qm.xtb_calc import solvent_of

log = logging.getLogger(__name__)

KIND = "thermo"
FAILURE_KIND = "thermo_failure"
THERMO_VERSION = 1
DEFAULT_THERMO_DIR = Path("data/cache/thermo")
NOT_STATIONARY_GRAD = 1e-3  # Hartree / Bohr: xtb's 'normal' gradient threshold
ASYMMETRY_WARN = 1e-2  # Hartree / Bohr^2: xtb's warning on an asymmetric Hessian
MASSES = "rdkit-average"


class ThermoCode(StrEnum):
    """Why a record has no thermo sidecar (failures and skips) or what degraded (warnings)."""

    NO_RECORD = "NO_RECORD"  # no QM record for the species under this method
    RECORD_STALE = "RECORD_STALE"  # the record's header does not match its identity
    UNKNOWN_METHOD = "UNKNOWN_METHOD"  # the record's method names no medium
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    TIMEOUT = "TIMEOUT"
    UNEXPECTED = "UNEXPECTED"
    WORKER_CRASH = "WORKER_CRASH"
    # warnings: the sidecar is still written
    IMAGINARY_REOPTIMIZED = "IMAGINARY_REOPTIMIZED"  # the thermo minimum is not the record's
    IMAGINARY_PERSISTENT = "IMAGINARY_PERSISTENT"  # a mode below the threshold is left: xtb's rule
    IMAGINARY_INVERTED = "IMAGINARY_INVERTED"  # imaginary modes above -20 cm^-1, made real
    IMAGINARY_DROPPED = "IMAGINARY_DROPPED"  # imaginary modes at or below -20 cm^-1, left out
    NOT_STATIONARY = "NOT_STATIONARY"  # max |gradient| at the geometry above 1e-3 Eh/Bohr
    RESTRAINED_GEOM = "RESTRAINED_GEOM"  # the record's geometry is a bond-wall optimum
    HESSIAN_ASYMMETRIC = "HESSIAN_ASYMMETRIC"  # max |H - H^T| above 1e-2 Eh/Bohr^2
    SYMMETRY_FAILED = "SYMMETRY_FAILED"  # the point-group search raised: sigma = 1


MEMO_CODES = frozenset({ThermoCode.SCF_NOT_CONVERGED, ThermoCode.TIMEOUT})


@dataclass(frozen=True)
class ThermoConfig:
    """The Hessian's and the re-optimization's settings (part of a sidecar's identity, the
    directories and the budgets apart) and where records and sidecars live."""

    fd_step_bohr: float = FD_STEP_BOHR
    accuracy: float = HESS_ACCURACY
    scf_maxiter: int = SCF_MAXITER  # a budget
    timeout_s: int = 3600  # a budget, per molecule, enforced by the batch
    symmetry_tolerance: float = pg.DEFAULT_TOLERANCE  # Angstrom
    reopt_threshold_cm: float = ro.REOPT_THRESHOLD_CM  # follow the modes below it
    reopt_rounds: int = ro.REOPT_ROUNDS  # Hessians after the record's, at most; 0: xtb's rule
    reopt_step_ang: float = ro.REOPT_STEP_ANG  # largest atomic displacement along the modes
    reopt_fmax: float = ro.REOPT_FMAX  # Hartree / Bohr
    reopt_maxiter: int = ro.REOPT_MAXITER  # a budget: energies per re-optimization
    qm_dir: Path = DEFAULT_CACHE_DIR
    out_dir: Path = DEFAULT_THERMO_DIR

    def __post_init__(self) -> None:
        self.reopt_settings()  # ValueError on a threshold at or above -1 cm^-1

    def reopt_settings(self) -> ro.ReoptSettings:
        return ro.ReoptSettings(
            threshold_cm=self.reopt_threshold_cm,
            rounds=self.reopt_rounds,
            step_ang=self.reopt_step_ang,
            fmax=self.reopt_fmax,
            maxiter=self.reopt_maxiter,
            scf_maxiter=self.scf_maxiter,
        )


DEFAULT_CONFIG = ThermoConfig()


def sidecar_config(method: str, conformer_id: int, config: ThermoConfig) -> dict[str, Any]:
    """The configuration a sidecar is stamped with and checked against: the re-optimization's
    keys are absent from a sidecar written before it, which therefore reads as stale."""
    return {
        "method": method,
        "conformer_id": int(conformer_id),
        "fd_step_bohr": float(config.fd_step_bohr),
        "accuracy": float(config.accuracy),
        "masses": MASSES,
        "symmetry_tolerance": float(config.symmetry_tolerance),
        **ro.identity(config.reopt_settings()),
    }


def thermo_path(out_dir: Path, smiles: str, conformer_id: int, method: str) -> Path:
    return Path(out_dir) / f"{cache_key(smiles, conformer_id, method)}.npz"


def failure_path(out_dir: Path, smiles: str, conformer_id: int, method: str) -> Path:
    return Path(out_dir) / f"{cache_key(smiles, conformer_id, method)}{failures.SUFFIX}"


# ---- computing


def symmetry_of(z: np.ndarray, xyz_ang: np.ndarray, tol: float) -> tuple[str, int, bool]:
    """(Schoenflies symbol, rotational symmetry number, linear) by `qm.point_group`: sigma is
    the number of proper rotations of the group found (1 for C∞v and an atom, 2 for D∞h)."""
    found = pg.search(z, xyz_ang, tol)
    if found.name == "D∞h":
        return found.name, 2, True
    if found.name in ("C∞v", "Kh"):
        return found.name, 1, found.name == "C∞v"
    proper = sum(1 for op in found.operations if op.det == 1)
    return found.name, max(proper, 1), False


def _symmetry_or_c1(record: QMRecord, xyz_ang: np.ndarray, tol: float):
    """(group, sigma, linear, warnings) at `xyz_ang`; C1 and sigma 1 when the search raises."""
    try:
        return (*symmetry_of(record.atomic_numbers, xyz_ang, tol), ())
    except Exception as exc:  # noqa: BLE001 - a symmetry failure must not lose the Hessian
        log.warning("point group of %s failed (%s); sigma = 1", record.smiles, exc)
        linear = record.n_atoms == 2
        return "C1", 1, linear, (str(ThermoCode.SYMMETRY_FAILED),)


@dataclass(frozen=True)
class Vibrations:
    """What a geometry's evaluation gives besides its normal modes (`thermo_reopt.Point.info`)."""

    group: str
    sigma: int
    linear: bool
    symmetry_warnings: tuple[str, ...]
    grad_max: float
    asymmetry: float
    n_gradients: int
    fresh_starts: int


def _hessian(record: QMRecord, xyz_ang: np.ndarray, config: ThermoConfig):
    return fd_hessian(
        record.atomic_numbers,
        np.asarray(xyz_ang, dtype=float) / BOHR_ANG,
        record.charge,
        record.unpaired,
        solvent_of(record.method),
        config.fd_step_bohr,
        config.scf_maxiter,
        config.accuracy,
    )


def evaluate_point(
    record: QMRecord, masses: np.ndarray, config: ThermoConfig, xyz_ang: np.ndarray, energy: float
) -> ro.Point:
    """The point group and the Hessian's normal modes at `xyz_ang` (`qm.hessian`), with the
    record's calculator. An atom has neither."""
    t0 = time.perf_counter()
    group, sigma, linear, sym_warn = _symmetry_or_c1(record, xyz_ang, config.symmetry_tolerance)
    if record.n_atoms == 1:
        info = Vibrations(group, sigma, linear, sym_warn, 0.0, 0.0, 0, 0)
        return ro.Point(xyz_ang, energy, np.zeros(0), np.zeros((3, 0)), 0.0, info)
    hess = _hessian(record, xyz_ang, config)
    nu, modes = frequencies(hess.hessian, np.asarray(xyz_ang) / BOHR_ANG, masses, linear)
    gmax = float(np.abs(hess.gradient).max())
    info = Vibrations(
        group, sigma, linear, sym_warn, gmax, hess.asymmetry, hess.n_gradients, hess.fresh_starts
    )
    return ro.Point(xyz_ang, energy, nu, modes, time.perf_counter() - t0, info)


def _warnings(record: QMRecord, res: ro.ReoptResult) -> list[str]:
    final, info = res.final, res.final.info
    use = usable_modes(final.frequencies_cm, DEFAULTS)
    flags = [
        (res.rounds > 0, ThermoCode.IMAGINARY_REOPTIMIZED),
        (res.outcome is ro.Outcome.PERSISTENT, ThermoCode.IMAGINARY_PERSISTENT),
        (use.n_inverted > 0, ThermoCode.IMAGINARY_INVERTED),
        (use.n_dropped_imaginary > 0, ThermoCode.IMAGINARY_DROPPED),
        (info.grad_max > NOT_STATIONARY_GRAD, ThermoCode.NOT_STATIONARY),
        ("RESTRAINED_GEOM" in record.warnings.split(","), ThermoCode.RESTRAINED_GEOM),
        (info.asymmetry > ASYMMETRY_WARN, ThermoCode.HESSIAN_ASYMMETRIC),
    ]
    return list(info.symmetry_warnings) + [str(code) for on, code in flags if on]


def _reopt_arrays(record: QMRecord, res: ro.ReoptResult) -> dict[str, np.ndarray]:
    return {
        "coords_ang": np.asarray(res.final.xyz_ang, dtype=float),
        "record_energy_ha": np.array(float(record.energy_ha)),
        "energy_drop_ha": np.array(float(record.energy_ha) - float(res.final.energy_ha)),
        "record_frequencies_cm": np.asarray(res.first.frequencies_cm, dtype=float),
        "reopt_rounds": np.array(int(res.rounds)),
        "reopt_outcome": np.array(str(res.outcome)),
        "reopt_stop": np.array(res.stop),
        "reopt_rejected": np.array(",".join(res.rejected)),
        "reopt_seconds": np.array(float(res.opt_seconds)),
    }


def compute_thermo(record: QMRecord, config: ThermoConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """The sidecar's arrays for one record: the Hessian at its geometry (`qm.hessian`), the
    re-optimization along its imaginary modes (`qm.thermo_reopt`), and the frequencies, moments,
    symmetry number and energy of the thermo minimum it ends at, with the multiplicity. The
    rounds stop early (REOPT_BUDGET) when the next would likely pass `config.timeout_s`. Raises
    QMError on an SCF failure of a Hessian and ValueError on a method with no medium."""
    masses = atomic_masses(record.atomic_numbers)
    deadline = time.monotonic() + config.timeout_s
    evaluate = partial(evaluate_point, record, masses, config)
    res = ro.follow_imaginary(record, evaluate, masses, config.reopt_settings(), deadline)
    final, info = res.final, res.final.info
    return {
        "smiles": np.array(record.smiles),
        "conformer_id": np.array(int(record.conformer_id)),
        "method": np.array(record.method),
        "charge": np.array(int(record.charge)),
        "unpaired": np.array(int(record.unpaired)),
        "multiplicity": np.array(int(record.unpaired) + 1),
        "atomic_numbers": np.asarray(record.atomic_numbers, dtype=np.int32),
        "energy_ha": np.array(float(final.energy_ha)),
        "frequencies_cm": np.asarray(final.frequencies_cm, dtype=float),
        "moments_amu_ang2": principal_moments(final.xyz_ang, masses),
        "mass_amu": np.array(float(masses.sum())),
        "symmetry_number": np.array(int(info.sigma)),
        "point_group": np.array(info.group),
        "linear": np.array(bool(info.linear)),
        "grad_max": np.array(info.grad_max),
        "hessian_asymmetry": np.array(info.asymmetry),
        "n_gradients": np.array(sum(int(p.info.n_gradients) for p in res.points)),
        "fresh_starts": np.array(sum(int(p.info.fresh_starts) for p in res.points)),
        "warnings": np.array(",".join(_warnings(record, res))),
        **_reopt_arrays(record, res),
    }


def reopt_fields(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """A sidecar's re-optimization as plain values (the batch's rows and report)."""
    first = np.asarray(arrays["record_frequencies_cm"], dtype=float)
    final = np.asarray(arrays["frequencies_cm"], dtype=float)
    return {
        "reopt_outcome": str(arrays["reopt_outcome"]),
        "reopt_stop": str(arrays["reopt_stop"]),
        "reopt_rounds": int(arrays["reopt_rounds"]),
        "reopt_rejected": str(arrays["reopt_rejected"]),
        "energy_drop_kcal": float(arrays["energy_drop_ha"]) * HARTREE_KCAL,
        "lowest_record_cm": float(first.min()) if len(first) else None,
        "lowest_cm": float(final.min()) if len(final) else None,
    }


# ---- files


def save_thermo(
    arrays: dict[str, Any], record_path: Path, config: ThermoConfig, seconds: float = 0.0
) -> Path:
    """Write the sidecar beside the others, stamped with its record; the time spent and the BLAS
    are tracing notes (header `extra`), outside the content hash."""
    method, cid = str(arrays["method"]), int(arrays["conformer_id"])
    header = prov.stamp(
        KIND,
        THERMO_VERSION,
        prov.file_input(record_path),
        sidecar_config(method, cid, config),
        prov.array_digest(arrays),
        extra={"seconds": round(float(seconds), 3), "blas": blas.provider("dgemm_")},
    )
    path = thermo_path(config.out_dir, str(arrays["smiles"]), cid, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    return prov.save_npz(path, arrays, header)


def is_current(path: Path, method: str, conformer_id: int, config: ThermoConfig) -> bool:
    """A sidecar exists for these settings and its record has not changed."""
    if not is_valid_cache_file(path):
        return False
    try:
        prov.verify(path, KIND, THERMO_VERSION, sidecar_config(method, conformer_id, config))
        return True
    except (prov.StaleArtifact, prov.CorruptArtifact):
        return False


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != prov.HEADER_KEY}


def as_thermo(arrays: dict[str, np.ndarray]) -> MolecularThermo:
    return MolecularThermo(
        energy_ha=float(arrays["energy_ha"]),
        frequencies_cm=np.asarray(arrays["frequencies_cm"], dtype=float),
        moments_amu_ang2=np.asarray(arrays["moments_amu_ang2"], dtype=float),
        mass_amu=float(arrays["mass_amu"]),
        symmetry_number=int(arrays["symmetry_number"]),
        multiplicity=int(arrays["multiplicity"]),
        linear=bool(arrays["linear"]),
        n_atoms=len(arrays["atomic_numbers"]),
    )


def load_thermo(
    path: Path,
    identity: tuple[str, int] | None = None,
    config: ThermoConfig | None = None,
) -> MolecularThermo:
    """The sidecar at `path`. Its header is verified (kind, version, the record it was built
    from); with `identity` = (method, conformer_id) and `config` also the settings it was built
    with. Raises StaleArtifact."""
    want = None if identity is None else sidecar_config(*identity, config or DEFAULT_CONFIG)
    prov.verify(path, KIND, THERMO_VERSION, want)
    return as_thermo(read_arrays(path))


# ---- the failure memo


def failure_params(config: ThermoConfig) -> dict[str, Any]:
    """Every setting of a `ThermoConfig` but its directories, as read from JSON."""
    raw = {f.name: getattr(config, f.name) for f in fields(config) if not f.name.endswith("_dir")}
    return json.loads(json.dumps({**raw, "version": THERMO_VERSION}, sort_keys=True))


def known_failure(
    config: ThermoConfig, smiles: str, conformer_id: int, method: str
) -> failures.Failure | None:
    """The tombstone of this sidecar that covers `config` (`failures.covers`), None when there
    is none or a sidecar exists."""
    if is_valid_cache_file(thermo_path(config.out_dir, smiles, conformer_id, method)):
        return None
    hit = failures.read(failure_path(config.out_dir, smiles, conformer_id, method))
    if hit is None or (hit.smiles, hit.conformer_id, hit.method) != (smiles, conformer_id, method):
        return None
    return hit if failures.covers(hit.params, failure_params(config)) else None


def remember_failure(
    config: ThermoConfig,
    smiles: str,
    conformer_id: int,
    method: str,
    reason: str,
    detail: str,
    seconds: float,
) -> Path | None:
    """The tombstone of a failure that recurs (`MEMO_CODES`); None for any other reason."""
    if reason not in MEMO_CODES:
        return None
    body = dict(
        smiles=smiles,
        conformer_id=int(conformer_id),
        method=method,
        reason=str(reason),
        detail=str(detail),
        seconds=float(seconds),
        params=failure_params(config),
    )
    header = prov.stamp(
        FAILURE_KIND,
        THERMO_VERSION,
        record_inputs(smiles),
        record_config(method, conformer_id),
        prov.sha256_text(json.dumps(body, sort_keys=True)),
    )
    return prov.write_json(failure_path(config.out_dir, smiles, conformer_id, method), body, header)


def forget_failure(config: ThermoConfig, smiles: str, conformer_id: int, method: str) -> None:
    failure_path(config.out_dir, smiles, conformer_id, method).unlink(missing_ok=True)


# ---- lookup


def thermo_of(
    smiles: str,
    base_method: str | None = None,
    conformer_id: int = 0,
    config: ThermoConfig = DEFAULT_CONFIG,
) -> MolecularThermo:
    """The verified sidecar of one species (its map-free canonical SMILES) under `base_method`
    (`QMConfig().method` by default; the species' spin tag is added). FileNotFoundError when
    there is none, StaleArtifact when its record or settings changed."""
    from ksmech.qm.spin import method_for
    from ksmech.qm.xtb_calc import QMConfig

    method = method_for(smiles, base_method or QMConfig().method)
    path = thermo_path(config.out_dir, smiles, conformer_id, method)
    if not is_valid_cache_file(path):
        raise FileNotFoundError(f"no thermo sidecar for {smiles!r} ({method}): {path}")
    return load_thermo(path, (method, conformer_id), config)


def free_energy_of(
    smiles: str,
    temperature: float,
    settings: ThermoSettings = DEFAULTS,
    base_method: str | None = None,
    config: ThermoConfig = DEFAULT_CONFIG,
) -> float:
    """G_i(T) in Hartree of one species, in the standard state of `settings` (1 M), from its
    sidecar (`thermo_of`): the `g_standard` entry of `qm.thermo.pool_free_energy`."""
    return standard_free_energy(thermo_of(smiles, base_method, 0, config), temperature, settings)
