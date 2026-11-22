"""Stage 2 of the GPU xTB plan: the non-integral GFN2 terms against tblite and against the budget.

    uv run python scripts/gpu_xtb/stage2_gate.py check-tblite [--n 8]
    uv run python scripts/gpu_xtb/stage2_gate.py time [--batch 64 256 1024] [--reps 20]

Molecules: the spike's sample (`data/processed/gpu_qm_spike/mols.json`, docs/gpu_qm_spike.md),
ALPB water as in production.

- check-tblite: the whole GPU energy and gradient at tblite's converged density (SCF accuracy
  1e-6): stage 1's integrals and H0 gradient with stage 2's terms and potentials, against
  tblite's total energy and gradient.
- time, per molecule and energy, best of the batch sizes:
  - stage2_ms: the CN, multipole-radius and Born-radius pass plus one evaluation of every
    stage-2 term with potentials and gradient. Gate: budget / 3, as stage 1's.
  - nonscf_ms: everything outside the SCF for one energy and gradient (stage 2's geometry
    pass, self-energies, stage 1's forward, a potential evaluation, stage 1's H0 gradient,
    the gradient evaluation with the H0 dE/dCN). Gate: 2 x budget / 3.
  - evaluate_ms: one evaluation alone, an upper bound of the per-SCF-cycle potential cost
    (stage 3's budget).
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ksmech.qm.geometry import new_calculator
from ksmech.qm.gpu import classical as K
from ksmech.qm.gpu import native
from ksmech.qm.gpu import terms as T
from ksmech.qm.gpu.basis import mol_basis

SAMPLE = Path("data/processed/gpu_qm_spike/mols.json")
OUT = Path("data/processed/gpu_qm_spike")
BUDGET_MS = {"1-15": 0.06, "16-30": 0.8, "31-50": 2.3, "51-80": 6.8, "81-120": 25.0,
             "121-200": 37.0}  # fmt: skip


def _mols(records):
    return [(np.array(r["z"]), np.array(r["xyz"])) for r in records]


def check_tblite(sample: dict, n: int) -> dict:
    out = {}
    for key, recs in sample.items():
        recs = recs[:n]
        runs, wd = [], []
        for r in recs:
            calc = new_calculator(np.array(r["z"]), np.array(r["xyz"]), int(r["charge"]), 0,
                                  "water")  # fmt: skip
            calc.set("accuracy", 1e-6)
            res = calc.singlepoint()
            got = {k: np.array(res.get(k)) for k in ("energy", "gradient", "density-matrix",
                   "orbital-energies", "orbital-occupations", "orbital-coefficients")}  # fmt: skip
            c = got["orbital-coefficients"]
            wd.append((c * (got["orbital-occupations"] * got["orbital-energies"])) @ c.T)
            runs.append(got)
        e, g = K.total_at_density(_mols(recs), [x["density-matrix"] for x in runs], wd, True)
        de, dg, a0 = [], [], 0
        for r, x, em in zip(recs, runs, e, strict=True):
            nel = T.reference_occupations(mol_basis(np.array(r["z"]))).sum() - int(r["charge"])
            occ = T.fermi_filling(x["orbital-energies"], nel / 2)
            de.append(abs(em + 2 * T.electronic_entropy(occ) - float(x["energy"])))
            dg.append(float(np.abs(g[a0 : a0 + len(r["z"])] - x["gradient"]).max()))
            a0 += len(r["z"])
        out[key] = {"n": len(recs), "max_dE": max(de), "max_dg": max(dg)}
        print(key, out[key], flush=True)
    return out


def _time(fn, reps: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def _random_scf(cb: K.ClassicalBatch, rng) -> list[torch.Tensor]:
    def r(n, s):
        return torch.from_numpy(s * rng.normal(size=n)).cuda()

    return [r(cb.nsh, 0.3), r(cb.nat, 0.4), r(3 * cb.nat, 0.2), r(6 * cb.nat, 0.2),
            r(cb.nat, 0.05)]  # fmt: skip


def _nonscf(cb, db, se0, kcn, sh_atom, scf, h0in) -> None:
    """One energy and gradient's work outside the SCF, potentials from the given variables."""
    qsh, qat, dp, qp, _ = scf
    cb.geometry()
    db.t["sh_se"].copy_(se0 - kcn * cb.w["cn"][sh_atom])
    db.forward()
    cb.evaluate(qsh, qat, dp, qp)
    db.h0_gradient(*h0in)
    cb.evaluate(qsh, qat, dp, qp, db.dedcn)


def _stage2(cb, scf) -> None:
    cb.geometry()
    cb.evaluate(*scf)


def _bin_rows(recs, nb, reps, rng) -> dict:
    picked = [recs[i % len(recs)] for i in range(nb)]
    mols = _mols(picked)
    cb, db = K.ClassicalBatch(mols, True), native.DeviceBatch(native.host_batch(mols))
    scf = _random_scf(cb, rng)
    se0, kcn = (torch.from_numpy(a).cuda() for a in K._shell_selfenergy_tables(mols))
    sh_atom = cb.t["sh_atom"].long()
    mat, nao = db.hb.mat_total, db.hb.ao_total

    def rnd(n, s):
        return torch.from_numpy(s * rng.normal(size=n)).cuda()

    h0in = (rnd(mat, 0.1), rnd(mat, 0.05), rnd(nao, 0.1), rnd(3 * cb.nat, 0.1),
            rnd(6 * cb.nat, 0.1))  # fmt: skip
    row = {
        "stage2_ms": _time(partial(_stage2, cb, scf), reps) / nb,
        "evaluate_ms": _time(partial(cb.evaluate, *scf), reps) / nb,
        "nonscf_ms": _time(partial(_nonscf, cb, db, se0, kcn, sh_atom, scf, h0in), reps) / nb,
    }
    return row


def time_bins(sample: dict, batches: list[int], reps: int, mem_gb: float = 10.0) -> dict:
    rng, out = np.random.default_rng(0), {}
    for key, recs in sample.items():
        out[key] = {}
        for nb in batches:
            nao2 = sum(int(recs[i % len(recs)]["nao"]) ** 2 for i in range(nb))
            if nao2 * 8 * 15 > mem_gb * 1e9:
                print(key, nb, "skipped: memory", flush=True)
                continue
            row = out[key][nb] = _bin_rows(recs, nb, reps, rng)
            print(key, nb, " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
            torch.cuda.empty_cache()
        rows = [r for k, r in out[key].items() if isinstance(k, int)]
        best = {k: min(r[k] for r in rows) for k in ("stage2_ms", "evaluate_ms", "nonscf_ms")}
        gate = {"stage2_ms": BUDGET_MS[key] / 3, "nonscf_ms": 2 * BUDGET_MS[key] / 3}
        out[key].update(best=best, gate=gate)
        verdict = all(best[k] <= g for k, g in gate.items())
        print(key, "best", {k: round(v, 4) for k, v in best.items()}, "gate",
              {k: round(v, 3) for k, v in gate.items()}, "PASS" if verdict else "FAIL",
              flush=True)  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("check-tblite", "time"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--root", type=Path, default=Path("/home/jacopo/Documents/ksmech"))
    args = ap.parse_args()
    sample = json.loads((args.root / SAMPLE).read_text())
    t0 = time.time()
    if args.what == "check-tblite":
        res = check_tblite(sample, args.n)
    else:
        res = time_bins(sample, args.batch, args.reps)
    res = {"result": res, "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name()}
    path = args.root / OUT / f"stage2_{args.what}.json"
    path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
