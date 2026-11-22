"""GPU xTB stage 1: the GFN2 basis, the numpy reference integrals and the CUDA kernels.

The reference (`ksmech.qm.gpu.reference`) is checked against tblite on the fixture molecules
and two d-shell metal species of the open-shell list (the integrals do not depend on the spin):
S and H0 against tblite's saved matrices, D and Q through tblite's molecular dipole and
quadrupole rebuilt from its density. Its gradient contraction is checked by finite differences.
The CUDA kernels (marker `gpu`, skipped without a device or the nvcc toolchain) are checked
against the reference.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ksmech.qm.ao import element_ao_l
from ksmech.qm.geometry import BOHR_ANG, new_calculator
from ksmech.qm.gpu import reference as ref
from ksmech.qm.gpu.basis import element_shells, mol_basis
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL

# hand-placed d-shell metal species (Bohr): linear CuCl2, tetrahedral FeBr4-
_T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)
METALS = {
    "copper_dichloride": (np.array([29, 17, 17]),
                          np.array([[0, 0, 0], [0, 0, 3.97], [0, 0, -3.97]]), 0, 1),
    "tetrabromoferrate": (np.array([26, 35, 35, 35, 35]),
                          np.vstack([np.zeros(3), 4.44 * _T]), -1, 5),
}  # fmt: skip
assert set(METALS) <= set(OPEN_SHELL)


def _organic(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return z, mol.GetConformer().GetPositions() / BOHR_ANG, Chem.GetFormalCharge(mol), 0


CASES = {**{k: _organic(s) for k, s in MOLECULES.items()}, **METALS}


@cache
def tblite_ref(name: str) -> dict:
    z, xyz, charge, uhf = CASES[name]
    res = new_calculator(z, xyz, charge, uhf, None, save_integrals=True).singlepoint()
    return {k: np.array(res.get(k)) for k in (
        "overlap-matrix", "hamiltonian-matrix", "density-matrix", "charges", "dipole",
        "quadrupole")}  # fmt: skip


def _self_energies(z, r):
    basis = mol_basis(z)
    h, s = r["hamiltonian-matrix"], r["overlap-matrix"]
    return np.array([h[a, a] / s[a, a] for a in basis.sh_ao])


def test_basis_has_tblite_shell_layout():
    for z in range(1, 87):
        ls = [cg.l for cg in element_shells(z) for _ in range(2 * cg.l + 1)]
        assert tuple(ls) == element_ao_l(z), z


def _molecular_moments(xyz, basis, r, ints):
    """tblite's molecular dipole and quadrupole from its density and our integrals."""
    nat, P, q = len(basis.z), r["density-matrix"], r["charges"]
    dp = np.stack([np.bincount(basis.ao_atom, -(P * d).sum(0), nat) for d in ints.D])
    qp = np.stack([np.bincount(basis.ao_atom, -(P * c).sum(0), nat) for c in ints.Q])
    quad = qp.sum(axis=1)
    for a in range(nat):
        x, v, d = xyz[a], xyz[a] * q[a], dp[:, a]
        c = np.array([x[0] * (v[0] + 2 * d[0]), x[0] * (v[1] + d[1]) + d[0] * x[1],
                      x[1] * (v[1] + 2 * d[1]), x[0] * (v[2] + d[2]) + d[0] * x[2],
                      x[1] * (v[2] + d[2]) + d[1] * x[2], x[2] * (v[2] + 2 * d[2])])  # fmt: skip
        tr = 0.5 * (c[0] + c[2] + c[5])
        quad += np.array([1.5 * c[0] - tr, 3 * c[1], 1.5 * c[2] - tr, 3 * c[3], 3 * c[4],
                          1.5 * c[5] - tr])  # fmt: skip
    return xyz.T @ q + dp.sum(axis=1), quad


@pytest.mark.parametrize("name", list(CASES))
def test_reference_matches_tblite(name):
    z, xyz, _, _ = CASES[name]
    r, basis = tblite_ref(name), mol_basis(z)
    ints = ref.integrals(basis, xyz, _self_energies(z, r))
    assert np.abs(ints.S - r["overlap-matrix"]).max() < 1e-12
    assert np.abs(ints.H0 - r["hamiltonian-matrix"]).max() < 1e-11
    dip, quad = _molecular_moments(xyz, basis, r, ints)
    assert np.abs(dip - r["dipole"]).max() < 1e-10
    assert np.abs(quad - r["quadrupole"]).max() < 1e-10


def random_inputs(basis, seed: int = 0) -> ref.GradInputs:
    rng = np.random.default_rng(seed)
    n, nat, nsh = basis.nao, len(basis.z), len(basis.sh_atom)
    a, b = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    return ref.GradInputs(
        se=-0.5 + 0.1 * rng.normal(size=nsh), dsedcn=0.01 * rng.normal(size=nsh),
        P=(a + a.T) / 2, X=0.05 * (b + b.T), vao=0.1 * rng.normal(size=n),
        vdp=0.1 * rng.normal(size=(3, nat)), vqp=0.1 * rng.normal(size=(6, nat)),
    )  # fmt: skip


@pytest.mark.parametrize("name", ["chloromethane", "copper_dichloride"])
def test_reference_gradient_matches_finite_differences(name):
    z, xyz, _, _ = CASES[name]
    basis = mol_basis(z)
    g = random_inputs(basis)
    grad, _ = ref.h0_gradient(basis, xyz, g)
    assert np.abs(grad.sum(axis=0)).max() < 1e-12  # translation invariance
    h = 1e-4
    for a, k in [(0, 0), (1, 2), (len(z) - 1, 1)]:
        xp, xm = xyz.copy(), xyz.copy()
        xp[a, k] += h
        xm[a, k] -= h
        fd = (ref.h0_functional(basis, xp, g) - ref.h0_functional(basis, xm, g)) / (2 * h)
        assert abs(fd - grad[a, k]) < 1e-7


def _gpu_ready() -> bool:
    try:
        import torch
    except ImportError:
        return False
    from ksmech.qm.gpu import native

    nvcc = Path(os.environ.get("KSXTB_NVCC", str(native.DEFAULT_NVCC)))
    return torch.cuda.is_available() and (nvcc.exists() or native.LIB.exists())


gpu = pytest.mark.skipif(not _gpu_ready(), reason="no CUDA device or ksxtb toolchain")


@pytest.mark.gpu
@gpu
def test_cuda_kernels_match_reference():
    import torch

    from ksmech.qm.gpu import native

    # D, Q and the gradient on every angular class: s/p organics, Cl d shells, Cu and Fe d with
    # Br d (d-d pairs); S and H0 of all ten fixtures are checked against tblite below
    names = ["acetone", "chloromethane", "benzene", "copper_dichloride", "tetrabromoferrate"]
    hb = native.host_batch([CASES[k][:2] for k in names])
    db = native.DeviceBatch(hb)
    gins = [random_inputs(mol_basis(CASES[k][0]), seed=i) for i, k in enumerate(names)]
    db.t["sh_se"].copy_(torch.from_numpy(np.concatenate([g.se for g in gins])))
    db.t["sh_dsedcn"].copy_(torch.from_numpy(np.concatenate([g.dsedcn for g in gins])))
    db.forward()
    P, X = np.zeros(hb.mat_total), np.zeros(hb.mat_total)
    for m, g in enumerate(gins):
        o, n = int(hb.arrays["mol_mat"][m]), g.P.shape[0]
        P[o : o + n * n], X[o : o + n * n] = g.P.ravel(), g.X.ravel()
    dev = [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (
        P, X, np.concatenate([g.vao for g in gins]),
        np.concatenate([g.vdp.T.ravel() for g in gins]),
        np.concatenate([g.vqp.T.ravel() for g in gins]))]  # fmt: skip
    db.h0_gradient(*dev)
    grad = db.grad.cpu().numpy().reshape(-1, 3)
    dedcn = db.dedcn.cpu().numpy()
    a0 = 0
    for m, (name, g) in enumerate(zip(names, gins, strict=True)):
        z, xyz = CASES[name][:2]
        basis = mol_basis(z)
        ints = ref.integrals(basis, xyz, g.se)
        for key, nc in (("S", 1), ("H0", 1), ("D", 3), ("Q", 6)):
            want = getattr(ints, key).reshape(nc, basis.nao, basis.nao)
            assert np.abs(db.matrix(getattr(db, key), m, nc) - want).max() < 1e-12, (name, key)
        gr, dc = ref.h0_gradient(basis, xyz, g)
        assert np.abs(grad[a0 : a0 + len(z)] - gr).max() < 1e-11, name
        assert np.abs(dedcn[a0 : a0 + len(z)] - dc).max() < 1e-12, name
        a0 += len(z)


def _device_forward(mols, refs):
    """S and H0 on the GPU with the shell self-energies read from tblite's H0 diagonal."""
    import torch

    from ksmech.qm.gpu import native

    db = native.DeviceBatch(native.host_batch(mols))
    se = [_self_energies(z, r) for (z, _), r in zip(mols, refs, strict=True)]
    db.t["sh_se"].copy_(torch.from_numpy(np.concatenate(se)))
    db.forward()
    return db


@pytest.mark.gpu
@gpu
def test_cuda_s_and_h0_match_tblite_on_every_fixture():
    names = list(CASES)
    db = _device_forward([CASES[k][:2] for k in names], [tblite_ref(k) for k in names])
    for m, name in enumerate(names):
        r = tblite_ref(name)
        assert np.abs(db.matrix(db.S, m)[0] - r["overlap-matrix"]).max() < 1e-12, name
        assert np.abs(db.matrix(db.H0, m)[0] - r["hamiltonian-matrix"]).max() < 1e-11, name


@pytest.mark.gpu
@gpu
def test_cuda_s_and_h0_match_tblite_for_every_element():
    """Z-Z diatomics for Z = 1..86 (every GFN2 shell and its d-d pairs)."""
    from tblite.interface import TBLiteRuntimeError

    mols, refs, failed = [], [], []
    for z in range(1, 87):
        zz, xyz = np.array([z, z]), np.array([[0.0, 0.0, 0.0], [0.3, -0.4, 4.5]])
        try:
            res = new_calculator(zz, xyz, 0, 0, None, save_integrals=True).singlepoint()
        except TBLiteRuntimeError:
            failed.append(z)
            continue
        mols.append((zz, xyz))
        refs.append({k: np.array(res.get(k)) for k in ("overlap-matrix", "hamiltonian-matrix")})
    assert not failed, f"tblite failed for Z = {failed}"
    db = _device_forward(mols, refs)
    for m, r in enumerate(refs):
        assert np.abs(db.matrix(db.S, m)[0] - r["overlap-matrix"]).max() < 1e-12, m + 1
        assert np.abs(db.matrix(db.H0, m)[0] - r["hamiltonian-matrix"]).max() < 1e-11, m + 1
