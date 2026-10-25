"""The Coulomb interaction between the charges on two atoms, for the stored samples'
atom-atom matrix (`pairs.atom_coulomb_matrix`, `MechanismSample.atom_coulomb`).

    KS2(k, l) = q_k q_l / (eps R_kl)        (eV; q Mulliken partial charges, R in Angstrom)

Within one fragment R is the geometric distance; between fragments, which have not met, it is
the van der Waals contact distance R_vdW,k + R_vdW,l. The matrix is no model input: it stays in
the schema-15 sample store until the next schema bump.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from rdkit import Chem

from ksmech.physics.params import COULOMB_EV_ANG, KSParams


@cache
def vdw_sum(za: int, zb: int) -> float:
    """R_vdW,a + R_vdW,b (Angstrom, RDKit's periodic table): the contact distance between
    fragments."""
    table = Chem.GetPeriodicTable()
    return table.GetRvdw(int(za)) + table.GetRvdw(int(zb))


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
