"""Where a batched optimization step's time goes: the SCF tuning's measurement.

    uv run python scripts/gpu_xtb/scf_profile.py [--bin 51-80] [--seeds 4] [--per-bin 64]
        [--steps 0] [--torch-profile]

Embeds the bin's species as `stage4_gate.py time` does (cached in the scratch file `--cache`),
runs `relax_many` over them (production's path: refilled batches, walled retries; all steps, or
`--steps` energies per run), with the
SCF's sections wrapped in synchronized timers: geometry (integrals, CN, radii), each cycle's
input potentials, Fock + eigensolve + density (fp32 and fp64 phases apart), Mulliken, output
potentials and energy, the mixer, the gradient, and the optimizer's own work (RFO, BFGS, guard,
walls) as the remainder. Also the cycles per SCF by phase, the active molecules per cycle
against the batch size (work spent on molecules already finished), and the fallbacks.
The timers synchronize the device, so the totals are an upper bound of the unsynchronized run;
--torch-profile instead runs torch.profiler over the first few steps and prints the top CUDA
kernels and the host time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage4_gate import SAMPLE, embed_many  # noqa: E402

from ksmech.qm.gpu import optimize as O  # noqa: E402
from ksmech.qm.gpu import scf as S  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)


def timed(name, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t0
        N[name] += 1
        return out

    return wrap


def instrument() -> None:
    sb = S.SCFBatch
    sb.set_geometry = timed("geometry", sb.set_geometry)
    sb.gradient = timed("gradient", sb.gradient)
    sb.run = timed("scf_run", sb.run)
    orig_pot = sb._potentials

    def pot(self, *a, slot="in"):
        return timed(f"potentials_{slot}", orig_pot)(self, *a, slot=slot)

    sb._potentials = pot
    S.fock = timed("fock", S.fock)
    S.solve = timed("eigh_fp64", S.solve)
    S.SCFBatch._precise = timed("precise", S.SCFBatch._precise)
    S.refine_occupied = timed("refine", S.refine_occupied)
    S.eig_fp32 = timed("eigh_fp32_merged", S.eig_fp32)
    S.SCFBatch._wdensity = timed("wdensity", S.SCFBatch._wdensity)
    S.solve_fp32 = timed("eigh_fp32", S.solve_fp32)
    S.mulliken = timed("mulliken", S.mulliken)
    S.occupations = timed("fermi", S.occupations)
    S.Broyden.next = timed("broyden", S.Broyden.next)
    orig_cycle = sb._cycle

    def cycle(self, st, p, active, precise, idx=None):
        a = active
        N["cyc_fp32_mols"] += int((a & ~precise).sum())
        N["cyc_fp64_mols"] += int((a & precise).sum())
        N["cyc_batch_mols"] += len(a)
        N["cycles"] += 1
        return timed("cycle", orig_cycle)(self, st, p, active, precise, idx)

    sb._cycle = cycle
    O.BatchedANC._evaluate = timed("evaluate", O.BatchedANC._evaluate)
    O.BatchedANC._step = timed("step", O.BatchedANC._step)
    orig_rf = sb.run_with_fallback

    def rf(self, start, active=None):
        r, fell = orig_rf(self, start, active)
        N["evals"] += int(active.sum()) if active is not None else len(r.energy)
        N["fallbacks"] += int(fell.sum())
        it = r.iterations[active] if active is not None else r.iterations
        N["scf_cycles_sum"] += int(it.sum())
        return r, fell

    sb.run_with_fallback = rf


def evals_log() -> list:
    """Per `_evaluate`: CUDA events around it, the batch size, the live molecules and the SCF
    cycles of the batch, kept as device tensors and read once at the end (no extra sync)."""
    rows, orig = [], O.BatchedANC._evaluate

    def ev(self, x, active):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = orig(self, x, active)
        b.record()
        rows.append((self.b, active.sum(), a, b, self._last_cycles))
        return out

    orig_run = S.SCFBatch.run

    def run(self, start=None, active=None):
        r = orig_run(self, start, active)
        self._cycles = r.iterations.max()
        return r

    O.BatchedANC._evaluate = ev
    S.SCFBatch.run = run
    O.BatchedANC._last_cycles = property(lambda self: self.sb._cycles)
    return rows


def cycles_log() -> list:
    """Per SCF cycle: an event at its start (the input potentials), the batch size and the
    active molecules; a cycle's time is up to the next cycle's start in the same SCF."""
    rows, sb = [], S.SCFBatch
    orig_pot, orig_cycle, orig_run = sb._potentials, sb._cycle, sb.run

    def pot(self, *a, slot="in"):
        if slot == "in":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            rows.append([self._run_id, self.vars.b, None, ev])
        return orig_pot(self, *a, slot=slot)

    def cycle(self, st, p, active, precise, idx=None):
        rows[-1][2] = active.sum()
        return orig_cycle(self, st, p, active, precise, idx)

    def run(self, start=None, active=None):
        self._run_id = len(rows)
        return orig_run(self, start, active)

    S.SCFBatch._potentials, S.SCFBatch._cycle, S.SCFBatch.run = pot, cycle, run
    return rows


def cycle_costs(rows: list) -> dict:
    """ms per cycle by active share of the batch, and the share of SCF time in each."""
    torch.cuda.synchronize()
    recs = []
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        if a[0] == b[0] and a[2] is not None:
            recs.append((a[1], int(a[2]), a[3].elapsed_time(b[3])))
    total, out = sum(r[2] for r in recs), {}
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = [r for r in recs if lo <= r[1] / r[0] < hi]
        if sel:
            out[f"{lo}-{hi}"] = {
                "cycles": len(sel),
                "ms_per_cycle": round(sum(r[2] for r in sel) / len(sel), 2),
                "mean_active": round(float(np.mean([r[1] for r in sel])), 1),
                "mean_batch": round(float(np.mean([r[0] for r in sel])), 1),
                "share": round(sum(r[2] for r in sel) / total, 3),
            }
    return out


def drain(rows: list) -> dict:
    """Wall share and seconds per molecule-energy by the live molecules of the step."""
    torch.cuda.synchronize()
    edges = [0, 8, 16, 32, 64, 128, 256, 10**9]
    out = {}
    recs = [(b, int(n), a.elapsed_time(e) / 1e3, int(c)) for b, n, a, e, c in rows]
    total = sum(r[2] for r in recs)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = [r for r in recs if lo < r[1] <= hi]
        if not sel:
            continue
        sec, ev = sum(r[2] for r in sel), sum(r[1] for r in sel)
        out[f"{lo + 1}-{hi}"] = {
            "steps": len(sel),
            "seconds": round(sec, 2),
            "share": round(sec / total, 3),
            "ms_per_mol_eval": round(1e3 * sec / max(ev, 1), 2),
            "mean_batch": round(float(np.mean([r[0] for r in sel])), 1),
            "mean_cycles": round(float(np.mean([r[3] for r in sel])), 1),
        }
    out["total_eval_s"] = round(total, 2)
    out["mol_evals"] = sum(r[1] for r in recs)
    return out


def tasks_for(bin_: str, per_bin: int, seeds: int, cache: Path):
    if cache.exists():
        got = pickle.loads(cache.read_bytes())
    else:
        sample = json.loads((Path("/home/jacopo/Documents/ksmech") / SAMPLE).read_text())
        recs = sample[bin_][:per_bin]
        jobs = [(r["smiles"], s) for s in range(seeds) for r in recs]
        got = [g for g in embed_many(jobs) if g is not None]
        cache.write_bytes(pickle.dumps(got))
    return [O.RelaxTask(str(k), mol, z, x, c, 0) for k, (mol, z, x, c) in enumerate(got)]


def report(wall: float, nmol: int) -> dict:
    out = {"wall_s": wall, "n_mol": nmol, "per_mol_ms": 1e3 * wall / nmol}
    out["sections_s"] = {k: round(v, 3) for k, v in sorted(T.items(), key=lambda kv: -kv[1])}
    out["counts"] = dict(N)
    ev = max(N["evals"], 1)
    out["ms_per_mol_eval"] = 1e3 * T["evaluate"] / ev
    out["scf_cycles_per_eval"] = N["scf_cycles_sum"] / ev
    out["active_share_per_cycle"] = (N["cyc_fp32_mols"] + N["cyc_fp64_mols"]) / max(
        N["cyc_batch_mols"], 1
    )
    out["fp64_share"] = N["cyc_fp64_mols"] / max(N["cyc_fp32_mols"] + N["cyc_fp64_mols"], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", default="51-80")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--per-bin", type=int, default=64)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--torch-profile", action="store_true")
    ap.add_argument("--no-timers", action="store_true", help="the wall time alone")
    ap.add_argument("--limit", type=int, default=0, help="the first N tasks only")
    ap.add_argument(
        "--evals-log",
        action="store_true",
        help="per evaluation (batch, live, cycles, seconds), no other timer",
    )
    ap.add_argument("--cycles-log", action="store_true", help="per SCF cycle, no other timer")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tasks = tasks_for(args.bin, args.per_bin, args.seeds, args.cache)
    from ksmech.qm.xtb_calc import QMConfig

    config = QMConfig(optimizer="tblite-anc")
    if args.steps:  # the first steps only
        config = QMConfig(optimizer="tblite-anc", maxiter=args.steps)
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            O.relax_many(tasks, config)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
        return
    tasks = tasks[: args.limit] if args.limit else tasks
    log = evals_log() if args.evals_log else None
    clog = cycles_log() if args.cycles_log else None
    if not args.no_timers and not args.evals_log and not args.cycles_log:
        instrument()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = O.relax_many(tasks, config)
    torch.cuda.synchronize()
    res = report(time.perf_counter() - t0, len(tasks))
    res["outcomes"] = dict(Counter("ok" if r.reason is None else str(r.reason) for r in outs))
    res["walled"] = sum(bool(r.restrained) for r in outs)
    if log is not None:
        res["by_live"] = drain(log)
    if clog is not None:
        res["by_active_share"] = cycle_costs(clog)
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
