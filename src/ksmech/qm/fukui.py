"""Condensed Fukui functions and global reactivity descriptors from GFN2 results.

Charge convention: q_A is the Mulliken charge (positive = electron-poor).

Finite difference (relaxed, same geometry):
    f+  = q(N)   - q(N+1)    where an added electron goes
    f-  = q(N-1) - q(N)      where an electron is removed from
    f0  = (f+ + f-) / 2
    dual = f+ - f-           (> 0 electrophilic site, < 0 nucleophilic site)
Each of f+, f-, f0 sums to 1 over the atoms.

Frozen-orbital (no extra calculation): f+_A and f-_A are the Mulliken gross populations of the
LUMO and HOMO on atom A, i.e. the sum over A's AOs of C_mu,k (S C_k)_mu.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.qm.ao import per_atom_sum

HARTREE_TO_EV = 27.211386245988
OCC_THRESHOLD = 0.5  # an MO is "occupied" when its occupation is at least this


@dataclass(frozen=True)
class FukuiFD:
    plus: np.ndarray
    minus: np.ndarray
    zero: np.ndarray
    dual: np.ndarray


def finite_difference(q_n: np.ndarray, q_plus: np.ndarray, q_minus: np.ndarray) -> FukuiFD:
    """Condensed Fukui functions from the charges of the N, N+1 and N-1 electron systems."""
    f_plus = q_n - q_plus
    f_minus = q_minus - q_n
    return FukuiFD(f_plus, f_minus, 0.5 * (f_plus + f_minus), f_plus - f_minus)


def nan_fukui(n_atoms: int) -> FukuiFD:
    nan = np.full(n_atoms, np.nan)
    return FukuiFD(nan, nan.copy(), nan.copy(), nan.copy())


def homo_index(occupations: np.ndarray) -> int | None:
    """Index of the highest MO with occupation >= 0.5; None for a zero-electron system."""
    occupied = np.flatnonzero(occupations >= OCC_THRESHOLD)
    return int(occupied.max()) if occupied.size else None


def unpaired_for_ionized(unpaired: int) -> int:
    """Unpaired electrons of the N+1 / N-1 system: one if N is closed shell, else one fewer."""
    return 1 if unpaired == 0 else unpaired - 1


def frozen_orbital_fukui(
    C: np.ndarray, S: np.ndarray, ao_atom: np.ndarray, n_atoms: int, homo: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """(f+, f-) as the Mulliken populations of the LUMO and HOMO. NaN when either is missing."""
    if homo is None or homo + 1 >= C.shape[1]:
        nan = np.full(n_atoms, np.nan)
        return nan, nan.copy()

    def population(k: int) -> np.ndarray:
        return per_atom_sum(C[:, k] * (S @ C[:, k]), ao_atom, n_atoms)

    return population(homo + 1), population(homo)


def global_descriptors(eps_ev: np.ndarray, homo: int | None) -> dict[str, float]:
    """HOMO, LUMO, chemical potential, hardness and electrophilicity (eV, Koopmans-style).

    mu = (E_HOMO + E_LUMO) / 2,  eta = (E_LUMO - E_HOMO) / 2,  omega = mu^2 / (2 eta).
    """
    nan = float("nan")
    out = dict(
        homo_ev=nan,
        lumo_ev=nan,
        chemical_potential_ev=nan,
        hardness_ev=nan,
        electrophilicity_ev=nan,
    )
    if homo is None or homo + 1 >= len(eps_ev):
        return out
    e_h, e_l = float(eps_ev[homo]), float(eps_ev[homo + 1])
    mu, eta = 0.5 * (e_h + e_l), 0.5 * (e_l - e_h)
    out.update(homo_ev=e_h, lumo_ev=e_l, chemical_potential_ev=mu, hardness_ev=eta)
    if eta > 0:
        out["electrophilicity_ev"] = mu**2 / (2 * eta)
    return out


def fock_from_mo(S: np.ndarray, C: np.ndarray, eps_ha: np.ndarray) -> np.ndarray:
    """Rebuild the converged Fock matrix: F = S C diag(eps) C^T S (valid since C^T S C = 1)."""
    SC = S @ C
    return (SC * eps_ha) @ SC.T
