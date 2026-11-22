"""GPU xTB stage 4: the batched ANC / RFO optimizer (marker `gpu`).

Its linear algebra (`anc_basis`, `rfo_step`, `bfgs`) is checked against `qm.anc`'s on padded
mixed batches; the optimizer itself against `qm.anc.optimize_anc` driven by the same GPU
energy, step by step (scripts/gpu_xtb/stage4_gate.py runs the production comparison).
"""

from __future__ import annotations

import numpy as np
import pytest

from ksmech.qm import anc
from tests.test_qm_gpu import CASES, gpu
from tests.test_qm_gpu_lindh import LINEAR


@pytest.mark.gpu
@gpu
def test_batched_anc_basis_matches_qm_anc():
    import torch

    from ksmech.qm.gpu import optimize as O

    mols = [CASES[k][:2] for k in ("ethane", "benzene", "acetonitrile")] + [LINEAR["co2"]]
    nmax = max(len(z) for z, _ in mols)
    pdim = 3 * nmax
    h = np.zeros((len(mols), pdim, pdim))
    x = np.zeros((len(mols), nmax, 3))
    real = np.zeros((len(mols), nmax), dtype=bool)
    for m, (z, xyz) in enumerate(mols):
        h[m, : 3 * len(z), : 3 * len(z)] = anc.lindh_hessian(z, xyz)
        x[m, : len(z)], real[m, : len(z)] = xyz, True
    vecs, lam, live = O.anc_basis(*(torch.as_tensor(a).cuda() for a in (h, x, real)))
    for m, (z, xyz) in enumerate(mols):
        v_ref, l_ref = anc.anc_basis(anc.lindh_hessian(z, xyz), xyz)
        k, nm = 3 * len(z), len(l_ref)
        assert int(live[m].sum()) == nm  # 3n - 6, or 3n - 5 for CO2
        assert np.abs(lam[m, :nm].cpu().numpy() - l_ref).max() < 1e-10
        span = vecs[m, :k, :nm].cpu().numpy()
        assert np.abs(span @ span.T - v_ref @ v_ref.T).max() < 1e-8
        assert np.abs(vecs[m, k:].cpu().numpy()).max(initial=0.0) == 0.0


@pytest.mark.gpu
@gpu
def test_batched_rfo_and_bfgs_match_qm_anc():
    import torch

    from ksmech.qm.gpu import optimize as O

    rng = np.random.default_rng(0)
    sizes, mmax = (7, 12, 4), 12
    hs, gs, ss, ys = [], [], [], []
    for m in sizes:
        a = rng.normal(size=(m, m))
        hs.append(a @ a.T / m + 0.05 * np.eye(m))
        gs.append(rng.normal(size=m) * 10.0 ** rng.uniform(-4, -1))
        ss.append(rng.normal(size=m) * 0.1)
        ys.append(hs[-1] @ ss[-1] + 0.01 * rng.normal(size=m))
    pad = lambda v, fill=0.0: np.pad(v, (0, mmax - len(v)), constant_values=fill)  # noqa: E731
    hb = np.stack([np.pad(h, (0, mmax - len(h))) + np.diag(pad(np.zeros(len(h)), 1.0))
                   for h in hs])  # fmt: skip
    t = [torch.as_tensor(a).cuda() for a in (hb, np.stack([pad(g) for g in gs]),
                                             np.stack([pad(s) for s in ss]),
                                             np.stack([pad(y) for y in ys]))]  # fmt: skip
    step = O.rfo_step(t[0], t[1]).cpu().numpy()
    upd = O.bfgs(t[0], t[2], t[3]).cpu().numpy()
    for m, n in enumerate(sizes):
        assert np.abs(step[m, :n] - anc.rfo_step(hs[m], gs[m])).max() < 1e-12
        assert np.abs(step[m, n:]).max(initial=0.0) < 1e-14  # padded modes: rounding only
        assert np.abs(upd[m, :n, :n] - anc._bfgs(hs[m], ss[m], ys[m])).max() < 1e-12
