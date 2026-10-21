"""Contact coupling between two localized orbitals, with no common AO basis required.

Two tokens of different molecules (or two tokens of one molecule that is already at its SCF
solution, where the converged coupling vanishes) are evaluated *as if brought into contact*:
one centre of each orbital at the sum of their covalent radii, lobes pointing at each other.
The AO overlap of that diatomic contact (s-s, s-p_sigma, p_sigma-s, p_sigma-p_sigma at the
contact distance) is probed from tblite itself for every element pair and cached, so no overlap
model is invented. The coupling of a centre pair is the maximal-overlap combination of the
orbital's |s| coefficient and p-vector norm on each centre; the token pair takes the best
centre pair. d functions and pi-type contacts are ignored.
"""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
from rdkit import Chem
from tblite.interface import Calculator

from ksmech.orbitals.descriptors import OrbitalDescriptors
from ksmech.orbitals.localize import P_AXIS_ORDER, valence_electrons
from ksmech.qm.ao import ao_layout
from ksmech.qm.geometry import BOHR_ANG

log = logging.getLogger(__name__)
_PZ = P_AXIS_ORDER.index(2)  # position of p_z inside a p shell (tblite order py, pz, px)


def contact_distance_ang(za: int, zb: int) -> float:
    table = Chem.GetPeriodicTable()
    return table.GetRcovalent(za) + table.GetRcovalent(zb)


@cache
def contact_overlap(za: int, zb: int) -> np.ndarray:
    """[[S_ss, S_s,pz], [S_pz,s, S_pz,pz]] of the A-B diatomic at the contact distance along z.

    Entries involving a missing p shell (hydrogen) are 0. Zeros with a logged warning if tblite
    cannot run the diatomic.
    """
    z = np.array([za, zb], dtype=np.int32)
    xyz = np.zeros((2, 3))
    xyz[1, 2] = contact_distance_ang(za, zb) / BOHR_ANG
    uhf = (valence_electrons(za) + valence_electrons(zb)) % 2
    try:
        calc = Calculator("GFN2-xTB", z, xyz, charge=0.0, uhf=uhf)
        calc.set("verbosity", 0)
        calc.set("save-integrals", 1)
        S = np.asarray(calc.singlepoint().get("overlap-matrix"))
    except Exception as exc:  # noqa: BLE001 - tblite raises its own error types
        log.warning("contact overlap probe failed for Z=%d, Z=%d: %s", za, zb, exc)
        return np.zeros((2, 2))
    ao_atom, ao_l = ao_layout(z)
    out = np.zeros((2, 2))
    for i, atom in enumerate((0, 1)):
        for j, other in enumerate((0, 1)):
            if i == j:
                continue
            s_i = np.flatnonzero((ao_atom == atom) & (ao_l == 0))[0]
            s_j = np.flatnonzero((ao_atom == other) & (ao_l == 0))[0]
            p_i = np.flatnonzero((ao_atom == atom) & (ao_l == 1))
            p_j = np.flatnonzero((ao_atom == other) & (ao_l == 1))
            out[0, 0] = abs(S[s_i, s_j])
            out[0, 1] = abs(S[s_i, p_j[_PZ]]) if len(p_j) else 0.0
            out[1, 0] = abs(S[p_i[_PZ], s_j]) if len(p_i) else 0.0
            out[1, 1] = abs(S[p_i[_PZ], p_j[_PZ]]) if len(p_i) and len(p_j) else 0.0
            return out  # the block is symmetric under A <-> B up to signs, which are dropped
    return out


def center_pair_coupling(s_a: float, p_a: float, za: int, s_b: float, p_b: float, zb: int) -> float:
    """Maximal-overlap contact coupling of one centre pair: every term constructive."""
    S = contact_overlap(int(za), int(zb))
    return float(
        s_a * s_b * S[0, 0] + s_a * p_b * S[0, 1] + p_a * s_b * S[1, 0] + p_a * p_b * S[1, 1]
    )


def contact_coupling(
    d_r: OrbitalDescriptors, z_r: np.ndarray, d_s: OrbitalDescriptors, z_s: np.ndarray
) -> float:
    """Best centre-pair contact coupling of two tokens (`z_*`: atomic numbers of each record)."""
    best = 0.0
    for i, a in enumerate(d_r.center_atoms):
        if a < 0:
            continue
        for j, b in enumerate(d_s.center_atoms):
            if b < 0:
                continue
            value = center_pair_coupling(
                d_r.center_s_coeff[i],
                d_r.center_p_norm[i],
                z_r[a],
                d_s.center_s_coeff[j],
                d_s.center_p_norm[j],
                z_s[b],
            )
            best = max(best, value)
    return best
