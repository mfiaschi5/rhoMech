"""GPU xTB stage 2: the CUDA kernels of the non-integral GFN2 terms (marker `gpu`).

The kernels (`ksmech.qm.gpu.classical`) are checked against the torch reference
(`ksmech.qm.gpu.terms`, exact against tblite) at random SCF variables: every term's energy,
the four potentials and the gradient with its CN, multipole-radius, Born-radius and surface
chains, plus an external dE/dCN. The batch holds the fixtures, the cutoff windows and a long
alkane chain longer than the D4 (50 Bohr) and Born (66 Bohr) cutoffs and a
hydride Z-H of every element Z = 1..86. The full GPU pipeline
(stage 1 integrals and H0 gradient with stage 2's potentials) is checked against tblite's
energy and gradient at tblite's density.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ksmech.qm.geometry import BOHR_ANG
from ksmech.qm.gpu import terms as T
from ksmech.qm.gpu.basis import mol_basis
from tests.test_qm_gpu import CASES, gpu
from tests.test_qm_gpu_terms import WINDOWS, element_hydrides, spin_occupations, tblite_run


@cache
def long_chain() -> tuple[np.ndarray, np.ndarray]:
    """n-C30H62, all-trans: about 37 A end to end."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C" * 30))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    for k in range(mol.GetNumAtoms() - 3):  # straighten every C-C-C-C dihedral
        a, b, c, d = (k, k + 1, k + 2, k + 3)
        if all(mol.GetAtomWithIdx(x).GetAtomicNum() == 6 for x in (a, b, c, d)):
            Chem.rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, 180.0)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return z, conf.GetPositions() / BOHR_ANG


def batch_molecules() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    hydrides = {f"Z{z}H": m for z, m in element_hydrides().items()}
    return {**{k: v[:2] for k, v in CASES.items()}, **WINDOWS, "long_chain": long_chain(),
            **hydrides}  # fmt: skip


def random_variables(z: np.ndarray, rng) -> dict[str, np.ndarray]:
    nat, nsh = len(z), len(mol_basis(z).sh_atom)
    return {"qsh": 0.3 * rng.normal(size=nsh), "qat": 0.4 * rng.normal(size=nat),
            "dp": 0.2 * rng.normal(size=(nat, 3)), "qp": 0.2 * rng.normal(size=(nat, 6)),
            "dedcn": 0.05 * rng.normal(size=nat)}  # fmt: skip


def torch_reference(z, xyz, v, solvent: bool):
    """Per-term energies, then grad, vsh, vat, vdp, vqp by autograd (shift left out)."""
    import torch

    xt = T._tensor(xyz).requires_grad_()
    leaves = {k: T._tensor(v[k]).requires_grad_() for k in ("qsh", "qat", "dp", "qp")}
    terms = T.energy_terms(mol_basis(z), xt, leaves, solvent)
    terms.pop("shift", None)
    e = sum(terms.values()) + (T.cn_dexp(z, xt) * T._tensor(v["dedcn"])).sum()
    wrt = [xt, *leaves.values()]
    g = torch.autograd.grad(e, wrt, allow_unused=True)
    g = [np.zeros(t.shape) if d is None else d.numpy() for d, t in zip(g, wrt, strict=True)]
    return {k: float(t) for k, t in terms.items()}, dict(zip(("grad", "vsh", "vat", "vdp", "vqp"),
                                                             g, strict=True))  # fmt: skip


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("solvent", [False, True])
def test_cuda_terms_match_reference(solvent):
    import torch

    from ksmech.qm.gpu import classical as K

    mols = batch_molecules()
    rng = np.random.default_rng(5)
    vs = {k: random_variables(z, rng) for k, (z, _) in mols.items()}
    cb = K.ClassicalBatch(list(mols.values()), solvent)
    cb.geometry()

    def cat(key):
        return torch.from_numpy(np.concatenate([v[key].ravel() for v in vs.values()])).cuda()

    out = cb.evaluate(cat("qsh"), cat("qat"), cat("dp"), cat("qp"), cat("dedcn"))
    out = {k: t.cpu().numpy() for k, t in out.items()}
    a0 = s0 = 0
    for m, (name, (z, xyz)) in enumerate(mols.items()):
        terms, ref = torch_reference(z, xyz, vs[name], solvent)
        nat, nsh = len(z), len(mol_basis(z).sh_atom)
        for term, e in terms.items():
            assert abs(out["energy"][m, K.TERMS.index(term)] - e) < 1e-12, (name, term)
        for key, lo, n in (("grad", a0, nat), ("vsh", s0, nsh), ("vat", a0, nat),
                           ("vdp", a0, nat), ("vqp", a0, nat)):  # fmt: skip
            assert np.abs(out[key][lo : lo + n] - ref[key]).max() < 1e-12, (name, key)
        a0, s0 = a0 + nat, s0 + nsh


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("solvent", [None, "water"])
def test_cuda_total_matches_tblite(solvent):
    from ksmech.qm.gpu import classical as K

    names = list(CASES)
    runs = [tblite_run(n, solvent, 1e-6) for n in names]
    wdens = [(r["orbital-coefficients"] * (r["orbital-occupations"] * r["orbital-energies"]))
             @ r["orbital-coefficients"].T for r in runs]  # fmt: skip
    mols = [CASES[n][:2] for n in names]
    e, g = K.total_at_density(mols, [r["density-matrix"] for r in runs], wdens, bool(solvent))
    a0 = 0
    for name, r, em, (z, _) in zip(names, runs, e, mols, strict=True):
        ts = sum(T.electronic_entropy(f) for f in spin_occupations(name, r["orbital-energies"]))
        assert abs(em + ts - float(r["energy"])) < 1e-10, name
        assert np.abs(g[a0 : a0 + len(z)] - r["gradient"]).max() < 1e-9, name
        a0 += len(z)
