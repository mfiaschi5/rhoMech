"""Stage 3 of the GPU xTB plan: the batched GFN2 SCF single point against tblite and the budget.

    uv run python scripts/gpu_xtb/stage3_gate.py check [--n 8]
    uv run python scripts/gpu_xtb/stage3_gate.py time [--batch 64 256 1024] [--reps 3]

Molecules: the spike's sample (`data/processed/gpu_qm_spike/mols.json`), closed shells, ALPB
water as in production.

- check, per bin:
  - tight: the GPU SCF and tblite both at accuracy 1e-6, energy and gradient compared (the same
    fixed point, so a wrong term shows here);
  - production: the GPU at accuracy 1 and tblite at accuracy 1, each against tblite at 1e-6
    (how far each lands from the converged answer at the production setting);
  - SCF cycles of the GPU at accuracy 1, and failures (not converged, non-finite).
- time, per molecule, best of the batch sizes, accuracy 1: a fresh single point with gradient
  (from the SAD guess) and a restarted one (the previous geometry's converged charges, every
  atom moved by up to 0.01 Bohr, as in an optimizer), and the host-side construction of the
  batch (build_ms, once per batch, outside both). Gate at 51-80 atoms: the fresh single point
  at most half of cpu12's (tblite's single point + gradient on one core / 12).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ksmech.qm.geometry import new_calculator
from ksmech.qm.gpu import scf as S

SAMPLE = Path("data/processed/gpu_qm_spike/mols.json")
OUT = Path("data/processed/gpu_qm_spike")
CPU12_MS = {
    "1-15": 0.4,
    "16-30": 2.8,
    "31-50": 7.6,
    "51-80": 21.0,
    "81-120": 77.0,
    "121-200": 119.0,
}  # fmt: skip  docs/gpu_qm_spike.md, measurement 1


def _mols(recs):
    return [(np.array(r["z"]), np.array(r["xyz"]), int(r["charge"]), 0) for r in recs]


def _tblite(m, accuracy):
    z, xyz, charge, uhf = m
    calc = new_calculator(z, xyz, charge, uhf, "water")
    calc.set("accuracy", accuracy)
    res = calc.singlepoint()
    return float(res.get("energy")), np.array(res.get("gradient"))


def _gpu(mols, accuracy):
    sb = S.SCFBatch(mols, True, accuracy=accuracy)
    r = sb.run()
    e, g = sb.gradient(r)
    return sb, r, e.cpu().numpy(), g.cpu().numpy()


def _split(g, mols):
    out, a0 = [], 0
    for z, *_ in mols:
        out.append(g[a0 : a0 + len(z)])
        a0 += len(z)
    return out


def check(sample: dict, n: int) -> dict:
    out = {}
    for key, recs in sample.items():
        mols = _mols(recs[:n])
        ref = [_tblite(m, 1e-6) for m in mols]
        prod = [_tblite(m, 1.0) for m in mols]
        _, rt, et, gt = _gpu(mols, 1e-6)
        _, rp, ep, gp = _gpu(mols, 1.0)
        gt, gp = _split(gt, mols), _split(gp, mols)

        def worst(es, gs, refs=ref):
            pairs = list(zip(es, gs, refs, strict=True))
            return (max(abs(e - r[0]) for e, _, r in pairs),
                    max(float(np.abs(g - r[1]).max()) for _, g, r in pairs))  # fmt: skip

        row = {
            "n": len(mols),
            "tight_dE_dg": worst(et, gt),
            "gpu_acc1_dE_dg": worst(ep, gp),
            "tblite_acc1_dE_dg": worst([p[0] for p in prod], [p[1] for p in prod]),
            "cycles_acc1": [int(x) for x in rp.iterations.tolist()],
            "failed": int((~rp.converged).sum()) + int((~rt.converged).sum()),
        }
        out[key] = row
        print(key, row, flush=True)
    return out


def _time_once(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def time_bins(sample: dict, batches: list[int], reps: int) -> dict:
    rng, out = np.random.default_rng(0), {}
    for key, recs in sample.items():
        out[key] = {}
        for nb in batches:
            picked = [recs[i % len(recs)] for i in range(nb)]
            try:
                t_build = time.perf_counter()
                sb = S.SCFBatch(_mols(picked), True)
                torch.cuda.synchronize()
                build_ms = (time.perf_counter() - t_build) * 1e3 / nb
            except torch.OutOfMemoryError:
                print(key, nb, "skipped: memory", flush=True)
                torch.cuda.empty_cache()
                continue
            xyz0 = sb.db.t["xyz"].clone()

            def fresh(sb=sb, xyz0=xyz0):
                sb.set_geometry(xyz0)
                sb.gradient(sb.run())

            r0 = sb.run()
            start = (r0.qsh, r0.dp, r0.qp)
            moved = xyz0 + torch.from_numpy(rng.uniform(-0.01, 0.01, xyz0.shape)).cuda()

            def restart(sb=sb, moved=moved, start=start):
                sb.set_geometry(moved)
                sb.gradient(sb.run(start))

            row = {"fresh_ms": min(_time_once(fresh) for _ in range(reps)) / nb,
                   "restart_ms": min(_time_once(restart) for _ in range(reps)) / nb,
                   "build_ms": build_ms,
                   "cycles_fresh": float(r0.iterations.double().mean())}  # fmt: skip
            out[key][nb] = row
            print(key, nb, {k: round(v, 3) for k, v in row.items()}, flush=True)
            sb = None
            torch.cuda.empty_cache()
        rows = [r for r in out[key].values()]
        if rows:
            best = min(r["fresh_ms"] for r in rows)
            out[key]["best_fresh_ms"], out[key]["cpu12_ms"] = best, CPU12_MS[key]
            print(key, f"best fresh {best:.3f} ms vs cpu12 {CPU12_MS[key]} ms "
                  f"(x{CPU12_MS[key] / best:.2f})", flush=True)  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("check", "time"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--root", type=Path, default=Path("/home/jacopo/Documents/ksmech"))
    args = ap.parse_args()
    sample = json.loads((args.root / SAMPLE).read_text())
    t0 = time.time()
    res = (
        check(sample, args.n) if args.what == "check" else time_bins(sample, args.batch, args.reps)
    )
    res = {"result": res, "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name()}
    path = args.root / OUT / f"stage3_{args.what}.json"
    path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
