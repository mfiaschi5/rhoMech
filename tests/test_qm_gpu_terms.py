"""GPU xTB stage 2: the torch reference of every GFN2 term beyond the one-electron integrals.

`ksmech.qm.gpu.terms` is checked against tblite at tblite's own converged density: the CN
self-energies against tblite's H0 diagonal, the total energy (band energy, entropy, repulsion,
D4, ES2, ES3, AES, ALPB, CDS, shift) and the total gradient, in the gas phase and in water.
tblite's H0 gradient uses the potential of the last SCF input charges, so the gradient is
compared at a tight SCF (accuracy 1e-6), where that lag is below 1e-11.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from ksmech.qm.geometry import new_calculator
from ksmech.qm.gpu import terms as T
from ksmech.qm.gpu.basis import mol_basis
from tests.test_qm_gpu import CASES, _self_energies

KEYS = ("energy", "gradient", "density-matrix", "orbital-energies", "orbital-occupations",
        "orbital-coefficients", "hamiltonian-matrix", "overlap-matrix")  # fmt: skip
GRADIENT_CASES = ("ethanol", "chloromethane", "acrolein", "copper_dichloride")


@cache
def tblite_run(name: str, solvent: str | None, accuracy: float | None = None) -> dict:
    z, xyz, charge, uhf = CASES[name]
    calc = new_calculator(z, xyz, charge, uhf, solvent, save_integrals=True)
    if accuracy is not None:
        calc.set("accuracy", accuracy)
    res = calc.singlepoint()
    return {k: np.array(res.get(k)) for k in KEYS}


def spin_occupations(name: str, emo: np.ndarray) -> list[np.ndarray]:
    z, _, charge, uhf = CASES[name]
    nel = T.reference_occupations(mol_basis(z)).sum() - charge
    return [T.fermi_filling(emo, (nel + uhf) / 2), T.fermi_filling(emo, (nel - uhf) / 2)]


@pytest.mark.parametrize("name", list(CASES))
def test_selfenergies_match_tblite_h0_diagonal(name):
    z, xyz, _, _ = CASES[name]
    se = T.selfenergies(mol_basis(z), T.cn_dexp(z, T._tensor(xyz))).numpy()
    assert np.abs(se - _self_energies(z, tblite_run(name, None))).max() < 1e-12


@pytest.mark.parametrize("solvent", [None, "water"])
@pytest.mark.parametrize("name", list(CASES))
def test_total_energy_matches_tblite(name, solvent):
    z, xyz, _, _ = CASES[name]
    r = tblite_run(name, solvent)
    occ = spin_occupations(name, r["orbital-energies"])
    assert np.abs(occ[0] + occ[1] - r["orbital-occupations"]).max() < 1e-12
    e, terms = T.total_energy(mol_basis(z), xyz, r["density-matrix"], occ, solvent is not None)
    assert ("alpb" in terms) == (solvent is not None)
    assert abs(e - float(r["energy"])) < 1e-10


@pytest.mark.parametrize("solvent", [None, "water"])
@pytest.mark.parametrize("name", GRADIENT_CASES)
def test_total_gradient_matches_tblite(name, solvent):
    z, xyz, _, _ = CASES[name]
    r = tblite_run(name, solvent, 1e-6)
    c, e = r["orbital-coefficients"], r["orbital-energies"]
    wdensity = (c * (r["orbital-occupations"] * e)) @ c.T
    g = T.total_gradient(mol_basis(z), xyz, r["density-matrix"], wdensity, solvent is not None)
    assert np.abs(g - r["gradient"]).max() < 1e-9


def _water_pair(sep: float):
    """Two water molecules sep Bohr apart (O-O), the second rotated."""
    w = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
    w2 = w[:, [2, 0, 1]] + np.array([sep, 0.0, 0.0])
    return np.array([8, 1, 1, 8, 1, 1]), np.vstack([w, w2])


def _neon_triangle(d: float):
    """Ne2 (3 Bohr apart) and a third Ne d Bohr from both: two ATM pairs at distance d."""
    h = np.sqrt(d * d - 1.5**2)
    return np.array([10, 10, 10]), np.array([[-1.5, 0, 0], [1.5, 0, 0], [0, h, 0.0]])


WINDOWS = {  # each puts pairs where a cutoff or its smooth switch acts
    "d4_two_switch": _water_pair(49.97),  # disp2 = 50 Bohr, switch width 0.05
    "atm_switch": _neon_triangle(24.97),  # disp3 = 25 Bohr
    "born_inside": _water_pair(65.5),  # Born lrcut = 35 A = 66.14 Bohr
    "born_outside": _water_pair(66.8),
}


@pytest.mark.parametrize("solvent", [None, "water"])
@pytest.mark.parametrize("name", list(WINDOWS))
def test_cutoff_windows_match_tblite(name, solvent):
    z, xyz = WINDOWS[name]
    res = new_calculator(z, xyz, 0, 0, solvent)
    res.set("accuracy", 1e-6)
    r = {k: np.array(v) for k, v in zip(KEYS[:6], map(res.singlepoint().get, KEYS[:6]),
                                         strict=True)}  # fmt: skip
    basis = mol_basis(z)
    nel = T.reference_occupations(basis).sum()
    occ = [T.fermi_filling(r["orbital-energies"], nel / 2)] * 2
    e, _ = T.total_energy(basis, xyz, r["density-matrix"], occ, solvent is not None)
    assert abs(e - float(r["energy"])) < 1e-10
    c, emo = r["orbital-coefficients"], r["orbital-energies"]
    wdensity = (c * (r["orbital-occupations"] * emo)) @ c.T
    g = T.total_gradient(basis, xyz, r["density-matrix"], wdensity, solvent is not None)
    assert np.abs(g - r["gradient"]).max() < 1e-9


def element_hydrides() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Z-H for Z = 1..86 at 3/4 of the covalent distance: charged atoms in every element."""
    from ksmech.qm.gpu.basis import params

    rcov = params()["covalent_rad"]["covalent_rad_d3"]
    return {z: (np.array([z, 1]), np.array([[0.0, 0.0, 0.0],
                                            [0.3, -0.2, 0.75 * (rcov[z - 1] + rcov[0])]]))
            for z in range(1, 87)}  # fmt: skip


def _tblite_any_spin(z, xyz, solvent):
    """tblite at the lowest spin its electron count allows (GFN2's lanthanide valence counts
    differ from Z), returning the result and uhf."""
    from tblite.interface import TBLiteRuntimeError

    for uhf in ((int(z.sum()) % 2), 1 - int(z.sum()) % 2):
        try:
            return new_calculator(z, xyz, 0, uhf, solvent).singlepoint(), uhf
        except TBLiteRuntimeError:
            continue
    raise AssertionError(f"tblite failed for {z.tolist()} at both spin parities")


def test_total_energy_matches_tblite_for_every_element():
    """Water, so ALPB, the H-bond term and D4's charge scaling see every element's tables."""
    worst = 0.0
    for z, (zz, xyz) in element_hydrides().items():
        res, uhf = _tblite_any_spin(zz, xyz, "water")
        emo = np.array(res.get("orbital-energies"))
        nel = T.reference_occupations(mol_basis(zz)).sum()
        occ = [T.fermi_filling(emo, (nel + uhf) / 2), T.fermi_filling(emo, (nel - uhf) / 2)]
        e, _ = T.total_energy(mol_basis(zz), xyz, np.array(res.get("density-matrix")), occ, True)
        assert abs(e - float(res.get("energy"))) < 1e-10, z
        worst = max(worst, abs(e - float(res.get("energy"))))
    assert worst < 1e-10
