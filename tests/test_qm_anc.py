"""qm.anc (Lindh model Hessian, ANC / RFO steps) and its GFN2 path in qm.geometry."""

import numpy as np
import pytest
from tblite.interface import TBLiteRuntimeError

from ksmech.qm import anc
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import (
    BOHR_ANG,
    DissociationGuard,
    TbliteEnergy,
    new_calculator,
    optimize_tblite,
    optimize_tblite_anc,
)
from ksmech.qm.records import QMError, ReasonCode
from tests.fixtures.molecules import MOLECULES

FMAX = 3e-4
KCAL = 627.5095


def start(smiles: str):
    mol = prepare_mol(smiles)
    cid = embed(mol, 1, seed=0)[0]
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    return mol, z, mol.GetConformer(cid).GetPositions() / BOHR_ANG


def bend(x):
    u, v = x[0] - x[1], x[2] - x[1]
    return np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))


def dihedral(x):
    b0, b1, b2 = x[0] - x[1], x[2] - x[1], x[3] - x[2]
    n = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ n) * n, b2 - (b2 @ n) * n
    return np.arctan2(np.cross(n, v) @ w, v @ w)


@pytest.mark.parametrize(
    ("angle", "b_vectors", "k"), [(bend, anc.b_bend, 3), (dihedral, anc.b_torsion, 4)]
)
def test_wilson_b_vectors_are_the_angle_gradients(angle, b_vectors, k):
    x = np.random.default_rng(0).normal(size=(k, 3)) * 1.5
    num = np.zeros((k, 3))
    for a in range(k):
        for c in range(3):
            dx = np.zeros((k, 3))
            dx[a, c] = 1e-6
            num[a, c] = (angle(x + dx) - angle(x - dx)) / 2e-6
    b = b_vectors(*x)
    assert np.allclose(b, num, atol=1e-7) or np.allclose(b, -num, atol=1e-7)


def test_collinear_angles_have_no_b_vectors():
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 1.0, 0]])
    assert anc.b_bend(*x[:3]) is None and anc.b_torsion(*x) is None


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_lindh_hessian_is_symmetric_positive_and_blind_to_rigid_motions(name):
    _, z, xyz = start(MOLECULES[name])
    h = anc.lindh_hessian(z, xyz)
    t = anc.rigid_motions(xyz)
    assert np.allclose(h, h.T) and np.linalg.eigvalsh(h).min() > -1e-10
    assert t.shape == (3 * len(z), 6) and np.allclose(t.T @ t, np.eye(6))
    assert np.abs(h @ t).max() < 1e-8
    vecs, lam = anc.anc_basis(h, xyz)
    assert vecs.shape == (3 * len(z), 3 * len(z) - 6) and lam.min() >= anc.H_LOW
    assert np.abs(t.T @ vecs).max() < 1e-8


def test_a_linear_geometry_has_five_rigid_motions():
    xyz = np.array([[0.0, 0, 0], [2.2, 0, 0], [4.4, 0, 0]])
    vecs, _ = anc.anc_basis(anc.lindh_hessian(np.array([8, 6, 8]), xyz), xyz)
    assert anc.rigid_motions(xyz).shape[1] == 5 and vecs.shape[1] == 4


def test_optimize_anc_recovers_a_geometry_from_its_distances():
    """Springs on every atom pair at the embedded distances: the minimum is that geometry."""
    _, z, xyz = start(MOLECULES["acetone"])
    pairs = np.array([(i, j) for i in range(len(z)) for j in range(i + 1, len(z))])
    r0 = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)

    def springs(x):
        d = x[pairs[:, 0]] - x[pairs[:, 1]]
        r = np.linalg.norm(d, axis=1)
        f = (0.2 * (r - r0) / r)[:, None] * d
        g = np.zeros_like(x)
        np.add.at(g, pairs[:, 0], f)
        np.add.at(g, pairs[:, 1], -f)
        return float((0.1 * (r - r0) ** 2).sum()), g

    x0 = xyz + np.random.default_rng(1).normal(0, 0.2, xyz.shape)
    out = anc.optimize_anc(springs, z, x0, anc.AncSettings(fmax=1e-6))
    assert out.converged and out.gmax <= 1e-6 and out.n_evals < 100
    r = np.linalg.norm(out.xyz[pairs[:, 0]] - out.xyz[pairs[:, 1]], axis=1)
    assert np.abs(r - r0).max() < 1e-4


def test_an_evaluation_budget_is_reported_as_not_converged():
    _, z, xyz = start(MOLECULES["ethanol"])
    energy = TbliteEnergy(z, xyz, 0, 0, "water")
    out = anc.optimize_anc(energy, z, xyz, anc.AncSettings(maxiter=3))
    assert not out.converged and out.n_evals == 3


def energy_and_gmax(z, xyz):
    res = new_calculator(z, xyz, 0, 0, "water").singlepoint()
    return float(res.get("energy")), float(np.abs(np.asarray(res.get("gradient"))).max())


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_anc_reaches_the_lbfgs_minimum_on_the_fixture_molecules(name):
    _, z, xyz = start(MOLECULES[name])
    e_anc, g_anc = energy_and_gmax(z, optimize_tblite_anc(z, xyz, 0, 0, FMAX, 500, "water"))
    e_lbfgs, _ = energy_and_gmax(z, optimize_tblite(z, xyz, 0, 0, FMAX, 500, "water"))
    assert g_anc <= FMAX
    assert abs(e_anc - e_lbfgs) * KCAL < 0.05


def test_a_failed_restart_falls_back_to_the_guess(monkeypatch):
    _, z, xyz = start(MOLECULES["acetone"])
    energy = TbliteEnergy(z, xyz, 0, 0, "water")
    e0, _ = energy(xyz)
    real = type(energy.calc).singlepoint

    def no_restart(self, res=None, copy=False):
        if res is not None:
            raise TBLiteRuntimeError("restart refused")
        return real(self, res, copy)

    monkeypatch.setattr(type(energy.calc), "singlepoint", no_restart)
    e1, _ = energy(xyz)
    assert energy.fresh_starts == 1 and abs(e1 - e0) < 1e-8


def test_the_restart_reaches_the_same_energy():
    _, z, xyz = start(MOLECULES["acrolein"])
    energy = TbliteEnergy(z, xyz, 0, 0, "water")
    energy(xyz + 0.05)
    e_restart, g_restart = energy(xyz)
    e_fresh, g_fresh = TbliteEnergy(z, xyz, 0, 0, "water")(xyz)
    assert abs(e_restart - e_fresh) < 1e-7 and np.abs(g_restart - g_fresh).max() < 1e-5


def test_the_guard_stops_a_run_once_a_bond_stays_broken():
    mol, _, xyz = start(MOLECULES["chloromethane"])
    far = xyz.copy()
    far[1] = xyz[0] + 1.6 * (xyz[1] - xyz[0])  # C-Cl at 1.6x its length
    guard = DissociationGuard(mol, 1.3, patience=3)
    for x in (far, far, xyz, far, far):  # an intact geometry restarts the count
        guard(x)
    with pytest.raises(QMError) as exc:
        guard(far)
    assert exc.value.reason is ReasonCode.CONNECTIVITY_CHANGED and "(0,1)" in exc.value.detail


def test_lithium_bonds_are_not_guarded():
    mol, _, xyz = start("C[Li]")
    far = xyz.copy()
    far[1] = xyz[0] + 3.0 * (xyz[1] - xyz[0])
    guard = DissociationGuard(mol, 1.3, patience=1)
    guard(far)
    guard(far)
