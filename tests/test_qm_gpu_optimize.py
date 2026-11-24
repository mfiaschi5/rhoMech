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


# --- carry-over: SCFs that run on into the next round -----------------------------------------


class _GpuEnergy:
    """One molecule's GPU energy and gradient for `qm.anc.optimize_anc`, each SCF restarted
    from the last converged point (stage4_gate.py's `GpuObjective`)."""

    def __init__(self, z, x0) -> None:
        from ksmech.qm.gpu import scf as S

        self.sb, self.start, self.log = S.SCFBatch([(z, x0, 0, 0)], True), None, []

    def __call__(self, x):
        import torch

        self.sb.set_geometry(torch.as_tensor(x).cuda())
        r, _ = self.sb.run_with_fallback(self.start)
        e, g = self.sb.gradient(r)
        if bool(r.converged[0]):
            self.start = (r.qsh.clone(), r.dp.clone(), r.qp.clone())
        self.log.append(float(e[0]))
        return float(e[0]), g.cpu().numpy()


@pytest.mark.gpu
@gpu
def test_carry_none_reproduces_qm_anc_driven_by_the_gpu_energy():
    """Without a carry, one round is one step: every evaluation's energy is the CPU
    optimizer's on the same GPU energy (to rounding), and so are the energy count and the end;
    the per-molecule trace holds every step evaluation (the start's excluded)."""
    from ksmech.qm.gpu import optimize as O

    for name in ("methyl_acetate", "acrolein"):
        t = _task(name)
        obj = _GpuEnergy(t.z, t.start)
        cpu = anc.optimize_anc(obj, t.z, t.start)
        opt = O.BatchedANC([O.OptSpec(t.z, t.start, 0, 0)], trace=True)
        res = opt.run()[0]
        assert res.converged and cpu.converged and res.n_evals == cpu.n_evals
        assert opt.rounds == res.n_evals  # the first evaluation's round, then one per step
        gpu_e = np.array([e for e, _ in res.trace])
        assert len(gpu_e) == res.n_evals - 1
        assert np.abs(np.array(obj.log[1:]) - gpu_e).max() < 1e-9
        assert np.abs(res.xyz - cpu.xyz).max() < 1e-7


def _carry_specs(names):
    from ksmech.qm.gpu import optimize as O

    return [O.OptSpec(t.z, t.start, 0, 0) for t in (_task(n) for n in names)]


@pytest.mark.gpu
@gpu
def test_carry_over_reaches_the_same_minima_with_the_same_counts():
    """With carry 1 (one SCF cycle per round, so every evaluation waits several rounds while
    the others step) and carry 0.25, a batch of fast and slow SCFs ends where carry None ends.
    Per molecule the computation is the same; what differs is which molecules share a cycle's
    eigensolve, and a cycle with one molecule active takes cuSOLVER's single-matrix path (1e-7
    to 1e-6 Eh in the fp32 orbital energies), which can move an SCF's last cycle within its
    1e-6 Eh convergence threshold: energies agree to far below 1e-6 Eh (measured 7e-11), and a
    count may move by a step where such noise tips an accept/reject."""
    from ksmech.qm.gpu import optimize as O

    names = ("ethane", "acetone", "benzene", "acrolein", "methyl_acetate", "acetonitrile")
    ref_opt = O.BatchedANC(_carry_specs(names))
    ref = ref_opt.run()
    for carry in (1.0, 0.25):
        opt = O.BatchedANC(_carry_specs(names), carry=carry, trace=True)
        got = opt.run()
        for a, b in zip(ref, got, strict=True):
            assert a.converged and b.converged and a.reason == b.reason
            assert len(b.trace) == b.n_evals - 1  # every step evaluation traced once
            assert abs(a.energy - b.energy) < 1e-6
            assert abs(a.n_evals - b.n_evals) <= 2
            assert np.abs(a.xyz - b.xyz).max() < 1e-4
        if carry == 1.0:  # the SCFs did interleave: far more rounds than steps
            assert opt.rounds > 2 * ref_opt.rounds


@pytest.mark.gpu
@gpu
def test_the_guard_counts_a_waiting_trial_once():
    """Every bond counts broken (factor 0.5), patience 5: the guard stops the run at its 5th
    trial (the start included), carry or not. With carry 1 each evaluation waits several
    rounds; a guard that counted a waiting trial each round would stop after one energy."""
    from ksmech.qm.gpu import optimize as O

    t = _task("ethanol")
    guard = O._guard_spec(t.mol, 0.5)
    assert guard[3] == 5

    def specs():
        return [O.OptSpec(t.z, t.start, 0, 0, guard=guard)] + _carry_specs(("acetone", "benzene"))

    outs = {}
    for carry in (None, 1.0):
        opt = O.BatchedANC(specs(), carry=carry)
        outs[carry] = opt.run()
        if carry:  # the trials did wait: many more rounds than evaluations
            assert opt.rounds > 2 * outs[carry][0].n_evals
    for carry, out in outs.items():
        assert out[0].reason.value == "CONNECTIVITY_CHANGED", carry
        assert out[0].n_evals == 5, carry
        assert out[1].converged and out[2].converged
    assert outs[None][0].energy == pytest.approx(outs[1.0][0].energy, abs=1e-6)


@pytest.mark.gpu
@gpu
def test_a_refill_stop_finishes_the_evaluations_in_flight():
    """Stopped for a refill while evaluations are pending (carry 1), the batch first runs
    them to their end and takes their steps: resumed in another batch, every molecule ends
    where the uninterrupted batch ends, with the same energy count (to the batch-composition
    noise of the resume test above)."""
    from ksmech.qm.gpu import optimize as O

    names = ("acetone", "methyl_acetate", "acrolein")
    whole = O.BatchedANC(_carry_specs(names), carry=1.0).run()
    opt = O.BatchedANC(_carry_specs(names), carry=1.0)
    done, states = opt.run(stop_below=3)
    assert states and opt.drained > 0  # stopped with evaluations in flight
    assert all(s.n_evals > 1 for s in states.values())
    specs = _carry_specs(names)
    resumed = O.BatchedANC(
        [O.OptSpec(sp.z, sp.start, 0, 0, state=states[m]) for m, sp in enumerate(specs)
         if m in states], carry=1.0).run()  # fmt: skip
    got = dict(done) | dict(zip(sorted(states), resumed, strict=True))
    for m, a in enumerate(whole):
        assert got[m].converged and a.converged
        assert abs(got[m].energy - a.energy) < 1e-5
        assert abs(got[m].n_evals - a.n_evals) <= 2


@pytest.mark.gpu
@gpu
def test_a_guard_stopped_trial_traces_no_energy():
    """The trial the guard stops is never evaluated: its trace entry's energy is NaN, carry or
    not, also when another molecule's evaluation ends in the same round (methyl acetate and
    acrolein still step at ethanol's 5th trial) and the carried SCF state still holds ethanol's
    previous energy."""
    from ksmech.qm.gpu import optimize as O

    t = _task("ethanol")
    guard = O._guard_spec(t.mol, 0.5)
    partners = _carry_specs(("methyl_acetate", "acrolein", "ethanol"))
    for carry in (None, 1.0):
        opt = O.BatchedANC([O.OptSpec(t.z, t.start, 0, 0, guard=guard)] + partners, carry=carry,
                           trace=True)  # fmt: skip
        out = opt.run()[0]
        assert out.reason.value == "CONNECTIVITY_CHANGED", carry
        energies = np.array([e for e, _ in out.trace])
        assert len(energies) == out.n_evals - 1 and np.isnan(energies[-1]), carry
        assert np.isfinite(energies[:-1]).all(), carry


@pytest.mark.gpu
@gpu
def test_a_failed_first_evaluation_does_not_fall_back_to_the_same_guess():
    """A first evaluation starts from the SAD guess: when it fails there is nothing to fall back
    to (`TbliteEnergy` re-raises without a previous result), so it fails after one SCF, with no
    fresh start counted."""
    from ksmech.qm.gpu import optimize as O

    opt = O.BatchedANC(_carry_specs(("acetone", "benzene")))
    opt.sb.econv, opt.sb.max_iter = 0.0, 6  # no SCF converges
    cycles = {"n": 0}
    iterate = opt.sb._iterate

    def counted(s, idx):
        cycles["n"] += 1
        return iterate(s, idx)

    opt.sb._iterate = counted
    for out in opt.run():
        assert out.reason.value == "SCF_NOT_CONVERGED" and out.fresh_starts == 0
        assert out.n_evals == 1
    assert cycles["n"] == 6  # one SCF to its limit, not a second from the same guess
