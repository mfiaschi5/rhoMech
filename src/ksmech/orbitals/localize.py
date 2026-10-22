"""Population localization of GFN2-xTB orbitals, in plain numpy (spec v7.2, appendix).

Default: intrinsic bond orbitals (IBO; Knizia, J. Chem. Theory Comput. 9, 4834, 2013), the
functional P_4 = sum_i sum_A (Q_A^i)^4 over Lowdin populations Q_A^i = sum_{mu in A}
(S^1/2 L)_{mu i}^2, started from a pivoted Cholesky factorization of each block's density
matrix in the Lowdin basis. In GFN2-xTB's minimal valence basis the IAO reference basis is the
computational basis itself, so the IAO projector is the identity and the intrinsic atomic
orbitals are the symmetrically orthogonalized AOs S^1/2: IAO populations are Lowdin populations.
For comparison `method="pm_mulliken"` keeps Pipek-Mezey: the squared Mulliken gross population,
P_2 = sum_i sum_A (Q_A^i)^2 (Pipek & Mezey, J. Chem. Phys. 90, 4916, 1989); `start="canonical"`
starts either from the canonical MOs.

Both are maximized by 2x2 Jacobi rotations. One sweep visits every orbital pair once, as n-1
round-robin rounds of n/2 disjoint pairs: rotations on disjoint pairs do not change each
other's angle, so a round is one vectorized step (the angle needs only Q_ii, Q_jj and Q_ij)
and the functional never decreases (PM: closed-form angle; IBO: the exact exponent-4 gain
maximized per pair on a grid that contains 0, then Newton-refined).
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
SMEAR_TOL = 0.05  # an MO further than this from its aufbau occupation is Fermi-smeared
_N_VALENCE_AOS = frozenset({1, 4, 9})
_MIN_GAIN = 1e-14


class LocalizationError(RuntimeError):
    """The basis is not valence-only or the input block is malformed."""


METHODS = ("ibo", "pm_mulliken")
STARTS = ("cholesky", "canonical")
_IBO_GRID = np.linspace(-np.pi / 4, np.pi / 4, 33)[:-1]  # one period of the p=4 gain; has 0
_NEWTON_STEPS = 4


@dataclass(frozen=True)
class PMConfig:
    tol: float = 1e-8  # stop when the functional changes by less than this over a sweep
    max_sweeps: int = 200
    method: str = "ibo"  # "ibo" (Lowdin populations, exponent 4) | "pm_mulliken" (comparison)
    start: str = "cholesky"  # "cholesky" (pivoted, Lowdin basis) | "canonical" (the MOs)

    def __post_init__(self) -> None:
        if self.method not in METHODS or self.start not in STARTS:
            raise LocalizationError(f"unknown method/start {self.method!r}/{self.start!r}")


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


def aufbau_occupations(rec: QMRecord) -> np.ndarray:
    """Integer occupations of the record's spin state: 2 on the (N - u) / 2 lowest MOs, 1 on
    the next u (the SOMOs), 0 above; u = `rec.unpaired`."""
    n_elec = int(round(float(rec.occupations.sum())))
    n_double = (n_elec - rec.unpaired) // 2
    order = np.argsort(rec.orbital_energies_ev, kind="stable")
    ref = np.zeros(len(rec.occupations))
    ref[order[:n_double]] = 2.0
    ref[order[n_double : n_double + rec.unpaired]] = 1.0
    return ref


def is_fermi_smeared(rec: QMRecord, tol: float = SMEAR_TOL) -> bool:
    """Some MO's occupation lies more than `tol` from its aufbau value: the GFN2 gap is small
    enough for the electronic temperature to spread electrons over several levels. For a
    closed shell this is exactly "some MO has 0.05 < n < 1.95"; a radical's SOMO at n = 1 is
    its spin state, not smearing."""
    return bool(np.any(np.abs(rec.occupations - aufbau_occupations(rec)) > tol))


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


def _round(C, SC, U, D, atom_sum, left, right) -> None:
    """Rotate every (left[k], right[k]) pair by its optimal PM angle, in place.

    C and SC = S C are AO coefficient blocks, U the accumulated rotation, D the per-atom
    diagonal populations Q_A^ii (n_atoms, n) and atom_sum the AO-to-atom sum (`_atom_sum`).
    """
    X = C[:, left] * SC[:, right] + C[:, right] * SC[:, left]
    qij = 0.5 * atom_sum(X)
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
    atom_sum = _atom_sum(ao_to_atom)
    SC = S @ C
    D = atom_sum(C * SC)
    start = value = float(np.sum(D * D))
    rounds = _schedule(n)
    sweeps, converged = 0, False
    while sweeps < config.max_sweeps and not converged:
        sweeps += 1
        for left, right in rounds:
            _round(C, SC, U, D, atom_sum, left, right)
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


def _sqrt_psd(S: np.ndarray) -> np.ndarray:
    """S^1/2 of the symmetric positive-definite overlap matrix."""
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def cholesky_start(X: np.ndarray) -> np.ndarray:
    """Orthogonal U whose columns X U are the pivoted-Cholesky orbitals of the block, made
    orthonormal symmetrically. `X` (nao, n) holds the block's orthonormal Lowdin coefficients;
    the density X X^T is factorized with the pivot at the largest remaining diagonal (first
    index on ties), so the start is deterministic and already local (Aquilante et al. 2006)."""
    dens = X @ X.T
    n = X.shape[1]
    L = np.zeros((len(dens), n))
    diag = np.diag(dens).copy()
    for k in range(n):
        piv = int(np.argmax(diag))
        col = dens[:, piv] - L[:, :k] @ L[piv, :k]
        L[:, k] = col / np.sqrt(diag[piv])
        diag = diag - L[:, k] ** 2
        diag[piv] = -np.inf  # never pivot twice on one AO
    M = X.T @ L
    w, V = np.linalg.eigh(M.T @ M)
    return M @ ((V / np.sqrt(w)) @ V.T)


def _ibo_gain(m: np.ndarray, d: np.ndarray, q: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Change of sum_A [Q_ii^4 + Q_jj^4] when the pair is rotated by `gamma` (..., pairs):
    Q_ii = m + x, Q_jj = m - x with x = d cos 2g + q sin 2g, so the sum is
    2 m^4 + 12 m^2 x^2 + 2 x^4 and the gain is relative to x = d (g = 0)."""
    x = d * np.cos(2 * gamma[..., None, :]) + q * np.sin(2 * gamma[..., None, :])
    return np.sum(12 * m * m * (x * x - d * d) + 2 * (x**4 - d**4), axis=-2)


def ibo_coefficients(m: np.ndarray, d: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, ...]:
    """(a4, b4, a8, b8) per pair: the exponent-4 gain is exactly the trigonometric polynomial
    a4 (cos 4g - 1) + b4 sin 4g + a8 (cos 8g - 1) + b8 sin 8g. With x^2 = A0 + A1 cos 4g +
    B1 sin 4g (A0 = (d^2 + q^2)/2, A1 = (d^2 - q^2)/2, B1 = d q), 12 m^2 x^2 + 2 x^4 expands
    into these four atom sums, so a round sums over atoms once instead of once per angle."""
    a0, a1, b1 = 0.5 * (d * d + q * q), 0.5 * (d * d - q * q), d * q
    w = 12 * m * m + 4 * a0
    return (w * a1).sum(0), (w * b1).sum(0), (a1 * a1 - b1 * b1).sum(0), (2 * a1 * b1).sum(0)


def _poly_gain(c: tuple[np.ndarray, ...], g: np.ndarray) -> np.ndarray:
    a4, b4, a8, b8 = c
    c4, c8 = np.cos(4 * g) - 1, np.cos(8 * g) - 1
    return a4 * c4 + b4 * np.sin(4 * g) + a8 * c8 + b8 * np.sin(8 * g)


def ibo_angles(m: np.ndarray, d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-pair rotation angle that maximizes the exponent-4 gain: the best of a 32-point grid
    over the period (it contains 0, so no pair loses), refined by Newton steps that are kept
    only where they improve the gain. (m, d, q): (n_atoms, n_pairs)."""
    c = ibo_coefficients(m, d, q)
    grid = _poly_gain(c, _IBO_GRID[:, None])
    gamma = _IBO_GRID[np.argmax(grid, axis=0)]
    best = grid.max(axis=0)
    step_max = _IBO_GRID[1] - _IBO_GRID[0]
    a4, b4, a8, b8 = c
    for _ in range(_NEWTON_STEPS):
        s4, c4, s8, c8 = np.sin(4 * gamma), np.cos(4 * gamma), np.sin(8 * gamma), np.cos(8 * gamma)
        f1 = 4 * (b4 * c4 - a4 * s4) + 8 * (b8 * c8 - a8 * s8)
        f2 = -16 * (a4 * c4 + b4 * s4) - 64 * (a8 * c8 + b8 * s8)
        newton = np.where(f2 < 0, f1 / np.where(f2 < 0, f2, -1.0), 0.0)
        trial = gamma - np.clip(newton, -step_max, step_max)
        gain = _poly_gain(c, trial)
        better = gain > best
        gamma, best = np.where(better, trial, gamma), np.where(better, gain, best)
    return np.where(best > _MIN_GAIN, gamma, 0.0)


def _atom_sum(ao_to_atom: np.ndarray):
    """f(M) = per-atom sums of the AO rows of M: `np.add.reduceat` over the atom boundaries
    when the AO map is sorted (tblite records run atom by atom), else the dense indicator."""
    ao_to_atom = np.asarray(ao_to_atom)
    if np.all(np.diff(ao_to_atom) >= 0) and ao_to_atom[0] == 0 and np.all(np.diff(ao_to_atom) <= 1):
        starts = np.flatnonzero(np.r_[True, np.diff(ao_to_atom) > 0])
        return lambda M: np.add.reduceat(M, starts, axis=0)
    P = np.zeros((int(ao_to_atom.max()) + 1, len(ao_to_atom)))
    P[ao_to_atom, np.arange(len(ao_to_atom))] = 1.0
    return lambda M: P @ M


def _ibo_round(X, U, D, atom_sum, left, right) -> None:
    """Rotate every (left[k], right[k]) pair by its exponent-4 angle, in place. X: Lowdin
    coefficients, U: accumulated rotation, D: (n_atoms, n) populations, atom_sum: AO -> atom."""
    q = atom_sum(X[:, left] * X[:, right])
    m = 0.5 * (D[:, left] + D[:, right])
    d = 0.5 * (D[:, left] - D[:, right])
    gamma = ibo_angles(m, d, q)
    c, s = np.cos(gamma), np.sin(gamma)
    for M in (X, U):
        Mi, Mj = M[:, left], M[:, right]
        M[:, left] = c * Mi + s * Mj
        M[:, right] = c * Mj - s * Mi
    x = d * np.cos(2 * gamma) + q * np.sin(2 * gamma)
    D[:, left], D[:, right] = m + x, m - x


def ibo_stats(
    C_sub: np.ndarray,
    S: np.ndarray,
    ao_to_atom: np.ndarray,
    config: PMConfig | None = None,
    S12: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, PMStats]:
    """IBO localization of the columns of `C_sub` from where they are (the caller picks the
    start): (C_loc, U, stats), C_loc = C_sub @ U, functional sum (Lowdin population)^4."""
    config = config or PMConfig()
    ao_to_atom = np.asarray(ao_to_atom)
    n = C_sub.shape[1]
    if n < 2:
        return np.array(C_sub, dtype=float), np.eye(n), PMStats(n, 0, True, 0.0, 0.0)
    X = (S12 if S12 is not None else _sqrt_psd(S)) @ C_sub
    U = np.eye(n)
    atom_sum = _atom_sum(ao_to_atom)
    D = atom_sum(X * X)
    start = value = float(np.sum(D**4))
    rounds = _schedule(n)
    sweeps, converged = 0, False
    while sweeps < config.max_sweeps and not converged:
        sweeps += 1
        for left, right in rounds:
            _ibo_round(X, U, D, atom_sum, left, right)
        new = float(np.sum(D**4))
        converged = abs(new - value) < config.tol
        value = new
    return C_sub @ U, U, PMStats(n, sweeps, converged, start, value)


def localize_block(
    C_sub: np.ndarray, S: np.ndarray, ao_to_atom: np.ndarray, config: PMConfig | None = None
) -> tuple[np.ndarray, np.ndarray, PMStats]:
    """Localize one block with `config.method` from `config.start`: (C_loc, U, stats) with
    C_loc = C_sub @ U and U orthogonal (start rotation included)."""
    config = config or PMConfig()
    C_sub = np.asarray(C_sub, dtype=float)
    if C_sub.ndim != 2 or C_sub.shape[0] != len(ao_to_atom) or S.shape != (len(ao_to_atom),) * 2:
        raise LocalizationError(f"shape mismatch: C {C_sub.shape}, S {S.shape}")
    U0 = np.eye(C_sub.shape[1])
    S12 = _sqrt_psd(S) if (config.start == "cholesky" or config.method == "ibo") else None
    if config.start == "cholesky" and C_sub.shape[1] > 1:
        U0 = cholesky_start(S12 @ C_sub)
    if config.method == "ibo":
        _, U1, stats = ibo_stats(C_sub @ U0, S, ao_to_atom, config, S12)
    else:
        _, U1, stats = pipek_mezey_stats(C_sub @ U0, S, ao_to_atom, config)
    U = U0 @ U1
    return C_sub @ U, U, stats


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
    """Localization of the doubly and the singly occupied MOs of `occ_idx` separately.

    U is block-diagonal in the order of `occ_idx`, so the spectral weights of every LMO still
    run over the whole occupied block, and the stats cover both sub-blocks (sweeps: the max).
    """
    n = len(occ_idx)
    U = np.zeros((n, n))
    stats = PMStats(0, 0, True, 0.0, 0.0)
    double = rec.occupations[occ_idx] >= DOUBLE_THRESHOLD
    for sel in (np.flatnonzero(double), np.flatnonzero(~double)):
        if len(sel):
            _, U_sub, st = localize_block(rec.C[:, occ_idx[sel]], rec.S, rec.ao_atom, config)
            U[np.ix_(sel, sel)] = U_sub
            stats = _merge_stats(stats, st)
    return rec.C[:, occ_idx] @ U, U, stats


def localize_record(rec: QMRecord, config: PMConfig | None = None) -> LocalizedOrbitals:
    """Localize (IBO by default, `config.method`) the occupied (occupation >= 0.5) and virtual
    blocks of a QM record; the SOMOs of an open-shell record are localized among themselves."""
    assert_valence_only(rec)
    occ_idx = np.flatnonzero(rec.occupations >= OCC_THRESHOLD)
    virt_idx = np.flatnonzero(rec.occupations < OCC_THRESHOLD)
    C_occ, U_occ, st_occ = _localize_occupied(rec, occ_idx, config)
    C_virt, U_virt, st_virt = localize_block(rec.C[:, virt_idx], rec.S, rec.ao_atom, config)
    return LocalizedOrbitals(occ_idx, virt_idx, C_occ, C_virt, U_occ, U_virt, st_occ, st_virt)


def mulliken_populations(C_loc: np.ndarray, S: np.ndarray, ao_to_atom: np.ndarray, n_atoms: int):
    """Per-atom Mulliken gross population of every orbital: shape (n_atoms, n_orbitals)."""
    gross = C_loc * (S @ C_loc)
    out = np.zeros((n_atoms, C_loc.shape[1]))
    for k in range(C_loc.shape[1]):
        out[:, k] = np.bincount(ao_to_atom, weights=gross[:, k], minlength=n_atoms)
    return out
