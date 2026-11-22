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
    emo, c = S.solve(sb.g, S.fock(sb.g, sb.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"]))
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
