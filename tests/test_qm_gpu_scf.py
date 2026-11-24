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


@pytest.mark.gpu
@gpu
def test_broyden_buffers_equal_the_list_form():
    """The preallocated, equilibrated mixer against tblite's arithmetic as written, over more
    iterations than one buffer chunk, with molecules dropping out."""
    import torch

    from ksmech.qm.gpu import scf as S

    gen = torch.Generator(device="cuda").manual_seed(0)
    q0 = torch.randn(5, 40, dtype=torch.float64, device="cuda", generator=gen)
    a, b = S.Broyden(q0, chunk=4), S.BroydenReference(q0)
    target = torch.randn(5, 40, dtype=torch.float64, device="cuda", generator=gen)
    active = torch.ones(5, dtype=torch.bool, device="cuda")
    for it in range(12):
        if it == 6:
            active[1] = False
        dq = torch.where(active[:, None], 0.3 * (target - a.q) + 0.05 * torch.sin(3 * a.q), 0.0)
        qa, qb = a.next(dq, active), b.next(dq, active)
        assert torch.abs(qa - qb).max() < 1e-11, it


@pytest.mark.gpu
@gpu
def test_refined_fp32_orbitals_reach_fp64_eigh():
    """At the converged Fock matrices of the fixtures, the fp32 eigensolve refined in fp64 gives
    the fp64 eigensolve's density, energy-weighted density and sum of occupied orbital energies
    (rotations among orbitals of equal occupation are left unresolved, so single energies are
    Rayleigh quotients); the metals (fractional occupations in near-degenerate pairs, whose split
    follows fp32 noise) are flagged for fp64 eigh."""
    import torch

    from ksmech.qm.gpu import scf as S

    sb = S.SCFBatch(list(CASES.values()), True)
    r = sb.run()
    pot = sb._potentials(r.qsh, r.dp, r.qp)
    idx = torch.arange(len(CASES), device="cuda")
    f = S.fock(sb.g, sb.lay, pot["vsh"], pot["vat"], pot["vdp"], pot["vqp"], idx)
    e32, c32, _ = S.eig_fp32(sb.g, f, idx)
    occ, _ = S.occupations(e32, sb.nel_spin)
    emo, cf, k, ok = S.refine_occupied(sb.g, f, c32, e32, occ, idx)
    e64, c64, _ = S.solve(sb.g, f, idx)
    occ64, _ = S.occupations(e64, sb.nel_spin)
    names = list(CASES)
    # the metals: fractionally occupied orbitals in near-degenerate pairs
    assert [n for n, o in zip(names, ok.tolist(), strict=True) if not o] == [
        "copper_dichloride", "tetrabromoferrate"]  # fmt: skip
    st = {"occ": occ64, "c": cf, "f": f, "x": torch.zeros_like(f)}
    sb._wdensity(st, idx)
    for m in torch.nonzero(ok).squeeze(1).tolist():
        nocc = int((occ64[m] > 0).sum())
        assert abs(float(emo[m, :nocc].sum() - e64[m, :nocc].sum())) < 1e-11
        p = (cf[m, :, :k] * occ64[m, :k]) @ cf[m, :, :k].T
        assert torch.abs(p - (c64[m] * occ64[m]) @ c64[m].T).max() < 5e-12  # eigh rounding
        x64 = (c64[m] * occ64[m] * e64[m]) @ c64[m].T
        assert torch.abs(st["x"][m] - x64).max() < 5e-12


def _toy_residual(q, target):
    import torch

    return 0.3 * (target - q) + 0.05 * torch.sin(3 * q)


@pytest.mark.gpu
@gpu
def test_broyden_molecules_at_their_own_iterations_equal_each_alone():
    """Molecules restarted at different iterations (staggered `reset`s, one of them paused for
    a while) match the list form run on each molecule alone, over more slots than a chunk."""
    import torch

    from ksmech.qm.gpu import scf as S

    gen = torch.Generator(device="cuda").manual_seed(1)
    b, v, n_it = 5, 30, 16
    q0 = torch.randn(b, v, dtype=torch.float64, device="cuda", generator=gen)
    q1 = torch.randn(b, v, dtype=torch.float64, device="cuda", generator=gen)
    target = torch.randn(b, v, dtype=torch.float64, device="cuda", generator=gen)
    resets = {2: [0], 5: [1, 3], 9: [0]}  # iteration -> molecules restarted from q1
    paused = {(4, 2), (5, 2), (6, 2)}  # (iteration, molecule) with no step
    mixer = S.Broyden(q0, chunk=4)
    refs = [S.BroydenReference(q0[m : m + 1]) for m in range(b)]
    for it in range(n_it):
        which = torch.zeros(b, dtype=torch.bool, device="cuda")
        for m in resets.get(it, []):
            which[m] = True
            refs[m] = S.BroydenReference(q1[m : m + 1])
        mixer.reset(which, q1)
        active = torch.tensor([(it, m) not in paused for m in range(b)], device="cuda")
        dq = torch.where(active[:, None], _toy_residual(mixer.q, target), 0.0)
        q = mixer.next(dq, active)
        for m in range(b):
            if bool(active[m]):
                r = refs[m].next(_toy_residual(refs[m].q, target[m : m + 1]), active[m : m + 1])
                assert torch.abs(q[m] - r[0]).max() < 1e-11, (it, m)
    assert int(mixer.h.max()) > 4  # the buffers grew


@pytest.mark.gpu
@gpu
def test_broyden_reset_after_nans_gives_finite_steps():
    """A molecule whose history went NaN, restarted by `reset`, steps as a fresh mixer does."""
    import torch

    from ksmech.qm.gpu import scf as S

    gen = torch.Generator(device="cuda").manual_seed(2)
    q0 = torch.randn(3, 20, dtype=torch.float64, device="cuda", generator=gen)
    target = torch.randn(3, 20, dtype=torch.float64, device="cuda", generator=gen)
    mixer = S.Broyden(q0)
    on = torch.ones(3, dtype=torch.bool, device="cuda")
    for it in range(5):
        dq = _toy_residual(mixer.q, target)
        if it >= 2:
            dq[1] = torch.nan
        mixer.next(dq, on)
    assert not bool(torch.isfinite(mixer.q[1]).all())
    mixer.reset(torch.tensor([False, True, False], device="cuda"), q0)
    ref = S.BroydenReference(q0[1:2])
    for _ in range(6):
        q = mixer.next(_toy_residual(mixer.q, target), on)
        r = ref.next(_toy_residual(ref.q, target[1:2]), on[1:2])
        assert bool(torch.isfinite(q).all())
        assert torch.abs(q[1] - r[0]).max() < 1e-11


def _carry_batch():
    from ksmech.qm.gpu import scf as S

    # the fp32 eigensolve of one matrix differs from that of a batch (another solver path), so
    # every phase below keeps two molecules of a kind running together: B twice, C twice
    names = ["acetone", "methyl_acetate", "methyl_acetate", "acrolein", "acrolein"]
    return S.SCFBatch([CASES[n] for n in names], True)


@pytest.mark.gpu
@gpu
def test_an_scf_started_beside_one_mid_run_ends_as_alone():
    """Molecules started while another one of the batch is mid-run (`advance` with a carry)
    end with the energy, cycle count and status they reach run alone from the same start."""
    import torch

    from ksmech.qm.gpu import scf as S

    sb = _carry_batch()
    mask = lambda *ms: torch.tensor([m in ms for m in range(5)], device="cuda")  # noqa: E731
    first = sb.run(active=mask(0, 3, 4))
    start = (first.qsh, first.dp, first.qp)  # acetone from its converged charges, the rest new
    alone = sb.run(start, mask(1, 2))
    state = sb.new_run()
    sb.start(state, mask(0, 3, 4))
    for _ in range(3):
        assert not bool(sb.advance(state, carry=1.0).any())  # exactly one cycle each
    assert state.iters.tolist() == [3, 0, 0, 3, 3]
    sb.start(state, mask(0, 1, 2), start)  # acetone restarted, the methyl acetates started
    sb.advance(state)
    r = sb.result(state)
    assert r.status.tolist() == [S.SCF_OK] * 5
    for m in (1, 2):
        assert abs(float(r.energy[m] - alone.energy[m])) < 1e-10
        assert int(r.iterations[m]) == int(alone.iterations[m])
    ref = sb.run(start, mask(0, 1, 2))
    assert int(r.iterations[0]) == int(ref.iterations[0])
    assert abs(float(r.energy[0] - ref.energy[0])) < 1e-10
    assert abs(float(r.energy[3] - first.energy[3])) < 1e-10  # the carried acroleins
    assert int(r.iterations[3]) == int(first.iterations[3])


@pytest.mark.gpu
@gpu
def test_advance_with_a_carry_returns_early_and_resumes():
    """advance(carry) stops with molecules still running; advance(None) then completes them,
    after a `set_geometry` at the same coordinates, with the results of one advance(None). The
    organic fixtures (they end after 7 to 12
    cycles): the metals' SCF is not reproducible run to run at 1e-10 (atomic sums; CuCl2 ends
    after 19 or 20 cycles), with or without a carry."""
    import torch

    from ksmech.qm.gpu import scf as S

    organics = [CASES[n] for n in CASES if n not in ("copper_dichloride", "tetrabromoferrate")]
    sb = S.SCFBatch(organics, True)
    b = len(organics)
    on = torch.ones(b, dtype=torch.bool, device="cuda")
    whole = sb.run()
    state = sb.new_run()
    sb.start(state, on)
    ended = sb.advance(state, carry=0.5)
    left = int(state.running.sum())
    assert 0 < left <= b // 2 and int(ended.sum()) == b - left
    assert not bool((ended & state.running).any())
    sb.set_geometry(None)  # the integrals again at unchanged coordinates, mid-SCF
    ended2 = sb.advance(state)
    assert bool((ended | ended2).all()) and not bool((ended & ended2).any())
    r = sb.result(state)
    assert r.status.tolist() == [S.SCF_OK] * b
    assert torch.abs(r.energy - whole.energy).max() < 1e-10
    assert r.iterations.tolist() == whole.iterations.tolist()
    assert torch.abs(r.density - whole.density).max() < 1e-10


@pytest.mark.gpu
@gpu
def test_fallback_restarts_from_the_guess_within_the_evaluation():
    """A start that fails (NaN variables for one molecule) falls back to the SAD guess in the
    same evaluation: its result is the guess run's, marked as fell back, the rest untouched."""
    import torch

    from ksmech.qm.gpu import scf as S

    sb = _carry_batch()
    guess = sb.run()
    start = [t.clone() for t in (guess.qsh, guess.dp, guess.qp)]
    shells = sb.atom_mol[sb.sh_atom] == 3
    start[0][shells] = torch.nan
    plain = sb.run(start)
    assert int(plain.status[3]) == S.SCF_NONFINITE
    r, fell = sb.run_with_fallback(start)
    assert fell.tolist() == [False, False, False, True, False]
    assert r.status.tolist() == [S.SCF_OK] * 5
    assert int(r.iterations[3]) == int(guess.iterations[3])
    assert abs(float(r.energy[3] - guess.energy[3])) < 1e-10
    keep = torch.arange(5, device="cuda") != 3
    assert torch.abs(r.energy[keep] - plain.energy[keep]).max() < 1e-10
    assert r.iterations[keep].tolist() == plain.iterations[keep].tolist()


@pytest.mark.gpu
@gpu
def test_the_cycle_limit_holds_per_scf_and_the_fallback_matches_two_runs():
    """With a limit of 5 cycles, SCFs started at different cycles each end after their own 5th
    with SCF_MAXITER, and leave the mixer's solve width to the SCFs after them; the fallback
    inside the evaluation gives what a run from the start merged with a run from the guess
    gives."""
    import torch

    from ksmech.qm.gpu import scf as S

    sb = _carry_batch()
    sb.max_iter = 5
    mask = lambda *ms: torch.tensor([m in ms for m in range(5)], device="cuda")  # noqa: E731
    state = sb.new_run()
    sb.start(state, mask(3, 4))
    sb.advance(state, carry=1.0)
    sb.advance(state, carry=1.0)
    sb.start(state, mask(1, 2))
    ended = sb.advance(state, carry=1.0)
    assert ended.tolist() == [False] * 5 and state.iters.tolist() == [0, 1, 1, 3, 3]
    ended = sb.advance(state, carry=1.0)
    ended |= sb.advance(state, carry=1.0)
    assert ended.tolist() == [False, False, False, True, True]
    assert state.status[3:].tolist() == [S.SCF_MAXITER] * 2
    assert sb.advance(state).tolist() == [False, True, True, False, False]
    assert state.iters.tolist() == [0, 5, 5, 5, 5]
    sb.start(state, mask(0))
    sb.advance(state, carry=1.0)
    sb.advance(state, carry=1.0)
    assert state.mixer.hbound == 1  # the ended SCFs' slots no longer count
    sb.advance(state)
    r = sb.result(state)
    assert r.iterations.tolist() == [5] * 5 and r.status.tolist() == [S.SCF_MAXITER] * 5
    start = sb.guess()
    got, fell = sb.run_with_fallback(start)
    assert fell.tolist() == [True] * 5
    ref = sb.merge(sb.run(start), sb.run(None, fell), fell)
    for k in ("energy", "density", "wdensity", "qsh", "dp", "qp"):
        assert torch.abs(getattr(got, k) - getattr(ref, k)).max() < 1e-12, k
    assert got.iterations.tolist() == ref.iterations.tolist()
    assert got.status.tolist() == ref.status.tolist()
    for k in got.vin:
        assert torch.abs(got.vin[k] - ref.vin[k]).max() < 1e-12, k


@pytest.mark.gpu
@gpu
def test_a_long_scf_leaves_the_mixer_narrow_for_the_scfs_after_it():
    """An SCF that runs to a 40-cycle limit grows the history to 64 slots; once it has ended,
    the next SCFs' solves are narrow again (hbound exact at once: the ended SCFs are retired
    before the solve reads its width) and the buffers shrink back to one chunk, and those SCFs
    end as a fresh state's."""
    import torch

    from ksmech.qm.gpu import scf as S

    sb = _carry_batch()
    econv, sb.max_iter = sb.econv, 40
    mask = lambda *ms: torch.tensor([m in ms for m in range(5)], device="cuda")  # noqa: E731
    state = sb.new_run()
    sb.econv = 0.0  # never converged: both acroleins run to the limit
    sb.start(state, mask(3, 4))
    sb.advance(state)
    assert state.status[3:].tolist() == [S.SCF_MAXITER] * 2
    assert state.mixer.df.shape[1] == 64 and state.mixer.hbound == 0
    sb.econv = econv
    sb.start(state, mask(0, 1, 2))
    sb.advance(state)
    assert state.mixer.df.shape[1] == state.mixer.chunk
    r, ref = sb.result(state), sb.run(None, mask(0, 1, 2))
    assert r.status[:3].tolist() == [S.SCF_OK] * 3
    assert r.iterations[:3].tolist() == ref.iterations[:3].tolist()
    assert torch.abs(r.energy[:3] - ref.energy[:3]).max() < 1e-10
    assert torch.abs(r.density[:3] - ref.density[:3]).max() < 1e-10


@pytest.mark.gpu
@gpu
def test_the_ended_counts_ride_on_the_mixers_host_read():
    """A cycle makes one stacked host read besides the running set (`advance`): the mixer's
    solve reads the ended and fallback counts with its own failure count and width; only a
    cycle without a solve (the first of a fresh mixer) reads the counts by themselves."""
    import torch

    sb = _carry_batch()
    state = sb.new_run()
    sb.start(state, torch.ones(5, dtype=torch.bool, device="cuda"))
    calls = {"n": 0}
    orig = torch.Tensor.tolist

    def counting(t):
        calls["n"] += 1
        return orig(t)

    cycles = 0
    torch.Tensor.tolist = counting
    try:
        while bool(state.running.any()):
            sb.advance(state, carry=1.0)
            cycles += 1
    finally:
        torch.Tensor.tolist = orig
    assert cycles > 5 and calls["n"] == cycles  # one read per cycle, not two
