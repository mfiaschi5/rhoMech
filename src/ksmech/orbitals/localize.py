"""Pipek-Mezey localization of GFN2-xTB orbitals, in plain numpy.

The functional is the sum over orbitals and atoms of the squared Mulliken gross population,
P = sum_i sum_A (Q_A^i)^2, maximized by 2x2 Jacobi rotations (Pipek & Mezey, J. Chem. Phys.
90, 4916, 1989). One sweep visits every orbital pair once, as n-1 round-robin rounds of n/2
disjoint pairs: rotations on disjoint pairs do not change each other's angle, so a round is one
vectorized step (the angle needs only Q_ii, Q_jj and Q_ij) and the functional never decreases.
Occupied and virtual orbitals are localized as separate blocks. Within the occupied block the
doubly and the singly occupied MOs are localized separately, because a rotation that mixes MOs
of different occupation changes the density sum_i n_i |phi_i|^2: a SOMO stays occupation 1.

Foster-Boys is not available as a fallback: tblite's Python interface exposes overlap and
Hamiltonian integrals only (`save-integrals`), no dipole integrals.

Within one atom tblite orders the p functions as (py, pz, px), the real spherical harmonics
m = -1, 0, +1. Verified on HF placed along each Cartesian axis; see `P_AXIS_ORDER`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.qm.fukui import OCC_THRESHOLD
from ksmech.qm.records import QMRecord

# AO index within a p shell -> Cartesian axis: tblite's (py, pz, px) order.
P_AXIS_ORDER = (1, 2, 0)
DOUBLE_THRESHOLD = 1.5  # an occupied MO at or above this is doubly occupied, below it a SOMO
_N_VALENCE_AOS = frozenset({1, 4, 9})
_MIN_GAIN = 1e-14


class LocalizationError(RuntimeError):
    """The basis is not valence-only or the input block is malformed."""


@dataclass(frozen=True)
class PMConfig:
    tol: float = 1e-8  # stop when the functional changes by less than this over a sweep
    max_sweeps: int = 200


@dataclass(frozen=True)
class PMStats:
    n_orbitals: int
    n_sweeps: int
    converged: bool
    functional_start: float
    functional_end: float


def valence_electrons(z: int) -> int:
    """Valence electron count GFN2-xTB uses for element `z` (1..86).

    Main group: group number. Transition metals: group number, except Zn, Cd and Hg (d10 in the
    core: 2). Lanthanides: 3 (4f in the core). Checked against tblite single-atom occupations.
    """
    if z <= 2:
        return z
    if z <= 10:
        return z - 2
    if z <= 18:
        return z - 10
    if z in (30, 48, 80):
        return 2
    if z <= 30:
        return z - 18
    if z <= 36:
        return z - 28
    if z <= 48:
        return z - 36
    if z <= 54:
        return z - 46
    if z <= 56:
        return z - 54
    if z <= 71:
        return 3
    if z <= 80:
        return z - 68
    if z <= 86:
        return z - 78
    raise LocalizationError(f"no valence electron count for Z={z}")


def assert_valence_only(rec: QMRecord) -> None:
    """Raise unless the record's basis is a minimal valence basis with no core orbitals."""
    counts = np.bincount(rec.ao_atom, minlength=rec.n_atoms)
    bad = [int(i) for i, c in enumerate(counts) if c not in _N_VALENCE_AOS]
    if bad:
        raise LocalizationError(f"atoms {bad} carry a non-minimal AO count: {counts[bad]}")
    expected = sum(valence_electrons(int(z)) for z in rec.atomic_numbers) - rec.charge
    n_elec = float(rec.occupations.sum())
    if abs(n_elec - expected) > 1e-6:
        raise LocalizationError(
            f"{n_elec} electrons in the record, {expected} valence electrons expected: "
            "the basis is not valence-only"
        )


def _schedule(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Round-robin rounds of disjoint orbital pairs that cover every pair once per sweep."""
    n_pad = n + n % 2
    pos = list(range(n_pad))
    rounds = []
    for _ in range(n_pad - 1):
        left = np.array(pos[: n_pad // 2])
        right = np.array(pos[n_pad // 2 :][::-1])
        keep = (left < n) & (right < n)  # drop the padding index of an odd block
        rounds.append((left[keep], right[keep]))
        pos = [pos[0], pos[-1], *pos[1:-1]]
    return rounds


def _round(C, SC, U, D, P, left, right) -> None:
    """Rotate every (left[k], right[k]) pair by its optimal PM angle, in place.

    C and SC = S C are AO coefficient blocks, U the accumulated rotation, D the per-atom
    diagonal populations Q_A^ii (n_atoms, n) and P the (n_atoms, nao) AO-to-atom indicator.
    """
    X = C[:, left] * SC[:, right] + C[:, right] * SC[:, left]
    qij = 0.5 * (P @ X)
    qii, qjj = D[:, left], D[:, right]
    diff = qii - qjj
    a = np.sum(qij * qij - 0.25 * diff * diff, axis=0)
    b = np.sum(qij * diff, axis=0)
    gain = 0.5 * (np.hypot(a, b) + a)
    gamma = np.where(gain > _MIN_GAIN, 0.25 * np.arctan2(b, -a), 0.0)
    c, s = np.cos(gamma), np.sin(gamma)
    for M in (C, SC, U):
        Mi, Mj = M[:, left], M[:, right]
        M[:, left] = c * Mi + s * Mj
        M[:, right] = c * Mj - s * Mi
    D[:, left] = c * c * qii + s * s * qjj + 2 * c * s * qij
    D[:, right] = s * s * qii + c * c * qjj - 2 * c * s * qij


def pipek_mezey_stats(
    C_sub: np.ndarray, S: np.ndarray, ao_to_atom: np.ndarray, config: PMConfig | None = None
) -> tuple[np.ndarray, np.ndarray, PMStats]:
    """`pipek_mezey` that also reports the sweep count and convergence."""
    config = config or PMConfig()
    C = np.array(C_sub, dtype=float)
    ao_to_atom = np.asarray(ao_to_atom)
    nao = len(ao_to_atom)
    if C.ndim != 2 or C.shape[0] != nao or S.shape != (nao, nao):
        raise LocalizationError(f"shape mismatch: C {C.shape}, S {S.shape}, map {ao_to_atom.shape}")
    n = C.shape[1]
    U = np.eye(n)
    if n < 2:
        return C, U, PMStats(n, 0, True, 0.0, 0.0)
    n_atoms = int(ao_to_atom.max()) + 1
    P = np.zeros((n_atoms, nao))
    P[ao_to_atom, np.arange(nao)] = 1.0
    SC = S @ C
    D = P @ (C * SC)
    start = value = float(np.sum(D * D))
    rounds = _schedule(n)
    sweeps, converged = 0, False
    while sweeps < config.max_sweeps and not converged:
        sweeps += 1
        for left, right in rounds:
            _round(C, SC, U, D, P, left, right)
        new = float(np.sum(D * D))
        converged = abs(new - value) < config.tol
        value = new
    return C_sub @ U, U, PMStats(n, sweeps, converged, start, value)


def pipek_mezey(
    C_sub: np.ndarray, S: np.ndarray, ao_to_atom: np.ndarray, config: PMConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Localize the orbitals in the columns of `C_sub` (AO coefficients, S-orthonormal).

    Returns (C_loc, U) with C_loc = C_sub @ U and U orthogonal. Sweeps until the functional
    changes by less than `config.tol`, at most `config.max_sweeps` times; the result after the
    last sweep is returned either way (check `pipek_mezey_stats` for convergence).
    """
    C_loc, U, _ = pipek_mezey_stats(C_sub, S, ao_to_atom, config)
    return C_loc, U


@dataclass(frozen=True)
class LocalizedOrbitals:
    """Localized occupied and virtual blocks of one record, with their canonical-MO index sets."""

    occ_index: np.ndarray  # canonical MO indices of the occupied block
    virt_index: np.ndarray
    C_occ: np.ndarray  # (nao, n_occ) localized coefficients
    C_virt: np.ndarray
    U_occ: np.ndarray
    U_virt: np.ndarray
    stats_occ: PMStats
    stats_virt: PMStats

    @property
    def converged(self) -> bool:
        return self.stats_occ.converged and self.stats_virt.converged


def _merge_stats(a: PMStats, b: PMStats) -> PMStats:
    return PMStats(
        a.n_orbitals + b.n_orbitals,
        max(a.n_sweeps, b.n_sweeps),
        a.converged and b.converged,
        a.functional_start + b.functional_start,
        a.functional_end + b.functional_end,
    )


def _localize_occupied(rec: QMRecord, occ_idx: np.ndarray, config: PMConfig | None):
    """PM on the doubly and the singly occupied MOs of `occ_idx` separately.

    U is block-diagonal in the order of `occ_idx`, so the spectral weights of every LMO still
    run over the whole occupied block, and the stats cover both sub-blocks (sweeps: the max).
    """
    n = len(occ_idx)
    U = np.zeros((n, n))
    stats = PMStats(0, 0, True, 0.0, 0.0)
    double = rec.occupations[occ_idx] >= DOUBLE_THRESHOLD
    for sel in (np.flatnonzero(double), np.flatnonzero(~double)):
        if len(sel):
            _, U_sub, st = pipek_mezey_stats(rec.C[:, occ_idx[sel]], rec.S, rec.ao_atom, config)
            U[np.ix_(sel, sel)] = U_sub
            stats = _merge_stats(stats, st)
    return rec.C[:, occ_idx] @ U, U, stats


def localize_record(rec: QMRecord, config: PMConfig | None = None) -> LocalizedOrbitals:
    """PM-localize the occupied (occupation >= 0.5) and virtual blocks of a QM record; the
    SOMOs of an open-shell record are localized among themselves, apart from the rest."""
    assert_valence_only(rec)
    occ_idx = np.flatnonzero(rec.occupations >= OCC_THRESHOLD)
    virt_idx = np.flatnonzero(rec.occupations < OCC_THRESHOLD)
    C_occ, U_occ, st_occ = _localize_occupied(rec, occ_idx, config)
    C_virt, U_virt, st_virt = pipek_mezey_stats(rec.C[:, virt_idx], rec.S, rec.ao_atom, config)
    return LocalizedOrbitals(occ_idx, virt_idx, C_occ, C_virt, U_occ, U_virt, st_occ, st_virt)


def mulliken_populations(C_loc: np.ndarray, S: np.ndarray, ao_to_atom: np.ndarray, n_atoms: int):
    """Per-atom Mulliken gross population of every orbital: shape (n_atoms, n_orbitals)."""
    gross = C_loc * (S @ C_loc)
    out = np.zeros((n_atoms, C_loc.shape[1]))
    for k in range(C_loc.shape[1]):
        out[:, k] = np.bincount(ao_to_atom, weights=gross[:, k], minlength=n_atoms)
    return out
