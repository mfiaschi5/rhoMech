"""Klopman-Salem term 2: the Coulomb interaction between the charges on two sites.

    KS2(k, l) = q_k q_l / (eps R_kl)        (eV; q Mulliken partial charges, R in Angstrom)

The site of an atom token is the atom; the site of an orbital token is its main centre m(r).
Within one fragment R is the geometric distance between the sites; between fragments, which
have not met, it is the van der Waals contact distance R_vdW,k + R_vdW,l. Two tokens on the
same site have no Coulomb term (undefined, 0).
"""

from __future__ import annotations

import numpy as np

from ksmech.physics.params import COULOMB_EV_ANG, KSParams


def coulomb(q_k: float, q_l: float, r_ang: float, params: KSParams | None = None) -> float:
    """q_k q_l / (eps R) in eV; 0 for R = 0."""
    if r_ang <= 0:
        return 0.0
    return COULOMB_EV_ANG * q_k * q_l / ((params or KSParams()).eps * r_ang)


def coulomb_matrix(
    q: np.ndarray, distances: np.ndarray, defined: np.ndarray, params: KSParams | None = None
) -> np.ndarray:
    """KS2 for every site pair: `q` per token site, `distances` (N, N) in Angstrom."""
    eps = (params or KSParams()).eps
    out = np.zeros_like(distances, dtype=float)
    ok = defined & (distances > 0)
    out[ok] = (COULOMB_EV_ANG * np.outer(q, q) / eps)[ok] / distances[ok]
    return out
