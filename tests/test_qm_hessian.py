"""`qm.hessian` and `qm.thermo_cache.compute_thermo` on optimized fixture records: the number of
vibrational modes, symmetry numbers, invariance under a rigid motion, open shells, single atoms
and the finite-difference step."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ksmech.qm import thermo as th
from ksmech.qm import thermo_cache as tc
from ksmech.qm.hessian import atomic_masses, fd_hessian, frequencies, rigid_body_basis
from ksmech.qm.thermo_cache import ThermoConfig, as_thermo, compute_thermo, symmetry_of
from tests.fixtures.molecules import MOLECULES

LINEAR = {"C#N": 1, "O=C=O": 2}  # HCN C∞v, CO2 D∞h


@pytest.fixture(scope="module")
def thermo_of(qm_record):
    """thermo_of(smiles) -> (record, sidecar arrays), memoized."""
    memo = {}

    def get(smiles: str):
        if smiles not in memo:
            rec = qm_record(smiles)
            memo[smiles] = rec, compute_thermo(rec, ThermoConfig())
        return memo[smiles]

    return get


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_a_nonlinear_molecule_has_3n_minus_6_modes(name, thermo_of):
    rec, arrays = thermo_of(MOLECULES[name])
    nu = arrays["frequencies_cm"]
    assert len(nu) == 3 * rec.n_atoms - 6 and not bool(arrays["linear"])
    assert nu.max() < 4000.0 and (nu > 100.0).sum() >= len(nu) - 2  # torsions may be soft
    assert int(arrays["multiplicity"]) == 1 and float(arrays["grad_max"]) < 1e-3


@pytest.mark.parametrize(("smiles", "sigma"), sorted(LINEAR.items()))
def test_a_linear_molecule_has_3n_minus_5_modes(smiles, sigma, thermo_of):
    rec, arrays = thermo_of(smiles)
    assert bool(arrays["linear"]) and int(arrays["symmetry_number"]) == sigma
    assert len(arrays["frequencies_cm"]) == 3 * rec.n_atoms - 5
    assert arrays["frequencies_cm"].min() > 300.0  # the bends are real, not rotations


def test_an_atom_has_no_mode_and_no_hessian(qm_record, monkeypatch):
    monkeypatch.setattr(tc, "fd_hessian", lambda *a, **k: pytest.fail("no Hessian for an atom"))
    rec = qm_record("[Br]")
    arrays = compute_thermo(rec)
    assert len(arrays["frequencies_cm"]) == 0 and int(arrays["multiplicity"]) == 2
    r = th.thermo_at(as_thermo(arrays), 298.15)
    assert r.entropy_parts["rot"] == 0.0 and r.zpe == 0.0


@pytest.mark.parametrize(
    ("name", "group", "sigma"),
    [
        ("benzene", "D6h", 12),
        ("ethane", "D3d", 6),
        ("chloromethane", "C3v", 3),
        ("acetonitrile", "C3v", 3),
        ("acetaldehyde", "Cs", 1),
    ],
)
def test_symmetry_numbers_of_optimized_records(name, group, sigma, thermo_of):
    _, arrays = thermo_of(MOLECULES[name])
    assert (str(arrays["point_group"]), int(arrays["symmetry_number"])) == (group, sigma)


def test_symmetry_numbers_of_ideal_geometries():
    c, h = 0.0, 0.629
    methane = np.array([[c, c, c], [h, h, h], [-h, -h, h], [-h, h, -h], [h, -h, -h]])
    assert symmetry_of(np.array([6, 1, 1, 1, 1]), methane, 0.1)[:2] == ("Td", 12)
    water = np.array([[0, 0, 0], [0.757, 0, 0.586], [-0.757, 0, 0.586]])
    assert symmetry_of(np.array([8, 1, 1]), water, 0.1) == ("C2v", 2, False)
    assert symmetry_of(np.array([35]), np.zeros((1, 3)), 0.1) == ("Kh", 1, False)


def test_frequencies_are_invariant_under_a_rigid_motion(thermo_of):
    """Gas phase: ALPB's surface grid is fixed in the frame, so a rotated molecule's solvation
    energy differs at the 1e-6 Hartree level; the gas-phase Hessian has no such term."""
    rec, _ = thermo_of(MOLECULES["ethanol"])
    rot = Rotation.from_euler("xyz", [37.0, -61.0, 112.0], degrees=True).as_matrix()
    moved = rec.coords_ang @ rot.T + np.array([3.1, -7.4, 0.9])
    masses = atomic_masses(rec.atomic_numbers)
    nus = []
    for xyz in (rec.coords_ang, moved):
        bohr = xyz / 0.529177210903
        hess = fd_hessian(rec.atomic_numbers, bohr, rec.charge, rec.unpaired, None)
        nus.append(frequencies(hess.hessian, bohr, masses, False)[0])
    assert np.abs(nus[0] - nus[1]).max() < 0.5  # cm^-1


def test_the_solvated_hessian_moves_little_under_a_rigid_motion(thermo_of):
    """ALPB's surface grid turns with the frame, not the molecule: over random orientations
    G - E_el of ethanol spreads by 0.15 kcal/mol and its stiff modes by 6 cm^-1, whatever the
    step (docs/decisions.md, 2026-11-09)."""
    rec, arrays = thermo_of(MOLECULES["ethanol"])
    rot = Rotation.from_euler("zyx", [80.0, 25.0, -140.0], degrees=True).as_matrix()
    moved = replace(rec, coords_ang=rec.coords_ang @ rot.T + 2.0)
    again = compute_thermo(moved)
    nu, nu2 = arrays["frequencies_cm"], again["frequencies_cm"]
    assert np.abs(nu - nu2)[nu > 500].max() < 10.0
    g = [th.thermo_at(as_thermo(a), 298.15).g_correction for a in (arrays, again)]
    assert abs(g[0] - g[1]) * th.HARTREE_KCAL < 0.25


def test_rigid_body_basis_spans_translations_and_rotations():
    xyz = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.3], [0.2, -0.3, 1.2]])
    masses = np.array([12.0, 1.0, 16.0, 1.0])
    basis = rigid_body_basis(xyz, masses, False)
    assert basis.shape == (12, 6) and np.allclose(basis.T @ basis, np.eye(6))
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]])
    assert rigid_body_basis(line, np.array([12.0, 16.0, 16.0]), True).shape == (9, 5)


def test_triplet_o2_is_open_shell_with_one_mode(thermo_of):
    rec, arrays = thermo_of("[O][O]")
    assert rec.unpaired == 2 and int(arrays["multiplicity"]) == 3
    assert len(arrays["frequencies_cm"]) == 1 and bool(arrays["linear"])
    r = th.thermo_at(as_thermo(arrays), 298.15)
    assert r.entropy_parts["el"] * th.HARTREE_J_MOL == pytest.approx(th.R_GAS * np.log(3))


def test_the_step_size_barely_matters(thermo_of):
    rec, arrays = thermo_of(MOLECULES["acetonitrile"])
    half = compute_thermo(rec, ThermoConfig(fd_step_bohr=0.0025))
    assert np.abs(arrays["frequencies_cm"] - half["frequencies_cm"]).max() < 2.0
