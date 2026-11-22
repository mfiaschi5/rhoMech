"""GPU xTB stage 3: the batched GFN2 SCF (marker `gpu`).

The Fock build at tblite's converged charges reproduces tblite's orbital energies and density;
the SCF from the SAD guess, run hard (accuracy 1e-6) on the fixtures in one batch, lands on
tblite's energy and gradient (the same fixed point), gas and water, CuCl2 and FeBr4- included.
"""

from __future__ import annotations

import numpy as np
import pytest

from ksmech.qm.gpu.basis import mol_basis
from tests.test_qm_gpu import CASES, gpu
from tests.test_qm_gpu_terms import tblite_run


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("solvent", [None, "water"])
def test_fock_at_tblite_charges_gives_tblite_orbitals(solvent):
    import torch

    from ksmech.qm.gpu import classical as K
    from ksmech.qm.gpu import scf as S

    names = list(CASES)
    mols = [CASES[n] for n in names]
    runs = [tblite_run(n, solvent, 1e-6) for n in names]
    sb = S.SCFBatch(mols, solvent is not None)
    p = torch.from_numpy(np.concatenate([r["density-matrix"].ravel() for r in runs])).cuda()
    qsh, dp, qp = K._mulliken(sb.db, sb.geo_mols, p)
    pot = sb._potentials(qsh, dp, qp)
    emo, c, _ = S.solve(sb.g, S.fock(sb.g, sb.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"]))
    for m, (name, r) in enumerate(zip(names, runs, strict=True)):
        k = mol_basis(CASES[name][0]).nao
        assert np.abs(emo[m, :k].cpu().numpy() - r["orbital-energies"]).max() < 1e-9, name
        cm = c[m, :k, :k]
        pm = ((cm * torch.from_numpy(r["orbital-occupations"]).cuda()) @ cm.T).cpu().numpy()
        assert np.abs(pm - r["density-matrix"]).max() < 1e-8, name


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("solvent", [None, "water"])
def test_scf_lands_on_tblite_energy_and_gradient(solvent):
    from ksmech.qm.gpu import scf as S

    names = list(CASES)
    sb = S.SCFBatch([CASES[n] for n in names], solvent is not None, accuracy=1e-6)
    r = sb.run()
    assert bool(r.converged.all())
    e, g = sb.gradient(r)
    e, g, a0 = e.cpu().numpy(), g.cpu().numpy(), 0
    for m, name in enumerate(names):
        ref, nat = tblite_run(name, solvent, 1e-6), len(CASES[name][0])
        assert abs(e[m] - float(ref["energy"])) < 1e-10, name
        assert np.abs(g[a0 : a0 + nat] - ref["gradient"]).max() < 1e-9, name
        a0 += nat


@pytest.mark.gpu
@gpu
def test_scf_kernels_match_torch_references():
    """scf.cu's Fock build, Mulliken sums and Fermi filling against their torch versions, with
    the open-shell metals' fractional occupations in the batch."""
    import torch

    from ksmech.qm.gpu import scf as S

    sb = S.SCFBatch(list(CASES.values()), True)
    r = sb.run()
    pot = sb._potentials(r.qsh, r.dp, r.qp)
    args = (sb.g, sb.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"])
    assert torch.abs(S.fock(*args) - S.fock_reference(*args)).max() < 1e-13
    for a, b in zip(S.mulliken(sb.g, sb.lay, r.density, sb.n0sh, sb.db.hb.nat),
                    S.mulliken_reference(sb.g, sb.lay, r.density, sb.n0sh, sb.db.hb.nat),
                    strict=True):  # fmt: skip
        assert torch.abs(a - b).max() < 1e-13
    occ, ts = S.occupations(r.emo, sb.nel_spin)
    ref = [S.fermi(r.emo, n) for n in sb.nel_spin]
    assert torch.abs(occ - ref[0] - ref[1]).max() < 1e-14
    assert torch.abs(ts - S.entropy(ref[0]) - S.entropy(ref[1])).max() < 1e-14
    fractional = ((occ - occ.round()).abs() > 1e-6).any(1)
    assert bool(fractional.any())  # the metals exercise fractional occupations


@pytest.mark.gpu
@gpu
def test_scf_gradient_follows_tblite_convention_at_production_accuracy():
    """At accuracy 1 the H0 gradient is contracted with the last input potentials, as tblite's
    is; the gradient then stays within 2e-5 of tblite's at the same accuracy."""
    from ksmech.qm.geometry import new_calculator
    from ksmech.qm.gpu import scf as S

    names = ["ethanol", "acetone", "acrolein", "chloromethane"]
    sb = S.SCFBatch([CASES[n] for n in names], True)
    _, g = sb.gradient(sb.run())
    g, a0 = g.cpu().numpy(), 0
    for name in names:
        z, xyz, charge, uhf = CASES[name]
        ref = np.array(new_calculator(z, xyz, charge, uhf, "water").singlepoint().get("gradient"))
        assert np.abs(g[a0 : a0 + len(z)] - ref).max() < 2e-5, name
        a0 += len(z)


@pytest.mark.gpu
@gpu
def test_a_poisoned_molecule_fails_alone():
    """Coincident atoms (S not positive definite) and a NaN coordinate fail their own molecule
    with a status; the rest of the batch reaches the clean batch's results."""
    import torch

    from ksmech.qm.gpu import scf as S

    names = ["acetone", "benzene", "chloromethane"]
    clean = [CASES[n] for n in names]
    z, xyz, charge, uhf = CASES["acetone"]
    coincident = xyz.copy()
    coincident[1] = coincident[0]
    nan = xyz.copy()
    nan[2, 1] = np.nan
    poisoned = clean + [(z, coincident, charge, uhf), (z, nan, charge, uhf)]
    ref = S.SCFBatch(clean, True)
    r_ref = ref.run()
    e_ref, g_ref = ref.gradient(r_ref)
    sb = S.SCFBatch(poisoned, True)
    r = sb.run()
    e, g = sb.gradient(r)
    assert r.converged.tolist() == [True, True, True, False, False]
    assert r.status[3:].tolist() == [S.SCF_BAD_GEOMETRY, S.SCF_BAD_GEOMETRY]
    assert torch.abs(e[:3] - e_ref).max() < 1e-12
    n = sum(len(CASES[k][0]) for k in names)
    assert torch.abs(g[:n] - g_ref).max() < 1e-12
