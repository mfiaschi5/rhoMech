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


def _task(name: str, key: str | None = None):
    from ksmech.qm.embed import embed, prepare_mol
    from ksmech.qm.geometry import BOHR_ANG
    from ksmech.qm.gpu import optimize as O
    from tests.fixtures.molecules import MOLECULES

    mol = prepare_mol(MOLECULES[name])
    cid = embed(mol, 1, seed=0)[0]
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    start = mol.GetConformer(cid).GetPositions() / BOHR_ANG
    return O.RelaxTask(key or name, mol, z, start, 0, 0)


@pytest.mark.gpu
@gpu
def test_relax_many_converges_and_runs_the_walled_retry():
    """acetone and ethanol relax to fmax; with every bond counted broken (conn_cov_factor
    0.5) the guard stops the free run, the walled run follows, and the bonds still read broken:
    CONNECTIVITY_CHANGED after both runs' energies, as qm.xtb_calc._relax would end."""
    from ksmech.qm.gpu import optimize as O
    from ksmech.qm.xtb_calc import QMConfig

    tasks = [_task("acetone"), _task("ethanol")]
    out = O.relax_many(tasks)
    assert all(o.reason is None and o.gmax <= 3e-4 and not o.restrained for o in out)
    broken = O.relax_many([_task("ethane")], QMConfig(optimizer="tblite-anc", conn_cov_factor=0.5))
    assert broken[0].reason.value == "CONNECTIVITY_CHANGED"
    assert broken[0].n_evals > 5  # the guard's patience, then the walled run


@pytest.mark.gpu
@gpu
def test_a_resumed_optimization_reaches_the_same_minimum():
    """Stopped in one batch and resumed in another, a run lands where it lands alone (the
    padded fp32 SCF cycles differ with the batch: noise, not a different minimum)."""
    from ksmech.qm.embed import kabsch_rmsd
    from ksmech.qm.gpu import optimize as O

    tasks = [_task(n) for n in ("acetone", "methyl_acetate", "acrolein")]
    specs = [O.OptSpec(t.z, t.start, 0, 0) for t in tasks]
    alone = [O.BatchedANC([sp]).run()[0] for sp in specs]
    done, states = O.BatchedANC(specs).run(stop_below=3)
    assert states  # stopped with all three still running
    resumed = O.BatchedANC([O.OptSpec(sp.z, sp.start, 0, 0, state=states[m])
                            for m, sp in enumerate(specs) if m in states]).run()  # fmt: skip
    got = dict(done) | dict(zip(sorted(states), resumed, strict=True))
    for m, a in enumerate(alone):
        assert got[m].converged and a.converged
        assert abs(got[m].energy - a.energy) < 1e-5
        assert kabsch_rmsd(got[m].xyz, a.xyz) < 0.05
