"""Stage 1 of the GPU xTB plan: the batched integrals against tblite and against the budget.

    uv run python scripts/gpu_xtb/stage1_gate.py check-tblite [--n 16]
    uv run python scripts/gpu_xtb/stage1_gate.py check-fd [--n 2]
    uv run python scripts/gpu_xtb/stage1_gate.py time [--batch 64 256 1024] [--reps 20]

Molecules: the spike's sample, 64 cached closed-shell GFN2 records per size bin with their
optimized geometries (`data/processed/gpu_qm_spike/mols.json`, docs/gpu_qm_spike.md).

- check-tblite: S and H0 against tblite's saved integrals (the shell self-energies read from
  tblite's H0 diagonal), D and Q through tblite's molecular dipole and quadrupole rebuilt from
  its density and charges.
- check-fd: the H0 gradient contraction against central differences of the functional it
  differentiates, both from the GPU, random densities and potentials.
- time: forward (S, H0, D, Q) and gradient pass per molecule against the stage 1 gate,
  one third of the spike's non-SCF budget per energy.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from speedchem.gpu import native
from speedchem.gpu.basis import mol_basis

from ksmech.qm.geometry import new_calculator

SAMPLE = Path("data/processed/gpu_qm_spike/mols.json")
OUT = Path("data/processed/gpu_qm_spike")
# the spike's budget for everything outside the SCF, ms per molecule and energy
BUDGET_MS = {"1-15": 0.06, "16-30": 0.8, "31-50": 2.3, "51-80": 6.8, "81-120": 25.0,
             "121-200": 37.0}  # fmt: skip


def load_sample(root: Path) -> dict[str, list[dict]]:
    return json.loads((root / SAMPLE).read_text())


def _tblite_ref(z, xyz, charge):
    calc = new_calculator(np.array(z), np.array(xyz), charge, 0, None, save_integrals=True)
    res = calc.singlepoint()
    return {k: np.array(res.get(k)) for k in
            ("overlap-matrix", "hamiltonian-matrix", "density-matrix", "charges", "dipole",
             "quadrupole")}  # fmt: skip


def _atomic(P, M, ao_atom, nat):
    return np.stack([np.bincount(ao_atom, -(P * Mc).sum(axis=0), nat) for Mc in M])


def _molecular_quadrupole(xyz, q, dp, qp):
    """tblite's get_molecular_quadrupole_moment."""
    tot = np.zeros(6)
    for a in range(len(q)):
        r, dpa = xyz[a], dp[:, a]
        v = r * q[a]
        c = np.zeros(6)
        c[[0, 2, 5]] = r * (v + 2 * dpa)
        c[1] = r[0] * (v[1] + dpa[1]) + dpa[0] * r[1]
        c[3] = r[0] * (v[2] + dpa[2]) + dpa[0] * r[2]
        c[4] = r[1] * (v[2] + dpa[2]) + dpa[1] * r[2]
        tr = 0.5 * (c[0] + c[2] + c[5])
        c = np.array([1.5 * c[0] - tr, 3 * c[1], 1.5 * c[2] - tr, 3 * c[3], 3 * c[4],
                      1.5 * c[5] - tr])  # fmt: skip
        tot += qp[:, a] + c
    return tot


def _compare(db, m, ref, z, xyz) -> dict:
    nat, basis = len(z), mol_basis(np.array(z))
    S, H0 = db.matrix(db.S, m)[0], db.matrix(db.H0, m)[0]
    D, Q = db.matrix(db.D, m, 3), db.matrix(db.Q, m, 6)
    P, q = ref["density-matrix"], ref["charges"]
    dp = _atomic(P, D, basis.ao_atom, nat)
    qp = _atomic(P, Q, basis.ao_atom, nat)
    dip = np.asarray(xyz).T @ q + dp.sum(axis=1)
    return {
        "S": float(np.abs(S - ref["overlap-matrix"]).max()),
        "H0": float(np.abs(H0 - ref["hamiltonian-matrix"]).max()),
        "dipole": float(np.abs(dip - ref["dipole"]).max()),
        "quadrupole": float(np.abs(_molecular_quadrupole(np.asarray(xyz), q, dp, qp)
                                   - ref["quadrupole"]).max()),
    }  # fmt: skip


def check_tblite(sample: dict, n: int) -> dict:
    out = {}
    for key, mols in sample.items():
        mols = mols[:n]
        refs = [_tblite_ref(r["z"], r["xyz"], r["charge"]) for r in mols]
        hb = native.host_batch([(np.array(r["z"]), np.array(r["xyz"])) for r in mols])
        db = native.DeviceBatch(hb)
        se = []
        for r, ref in zip(mols, refs, strict=True):
            basis = mol_basis(np.array(r["z"]))
            h, s = ref["hamiltonian-matrix"], ref["overlap-matrix"]
            se.append([h[a, a] / s[a, a] for a in basis.sh_ao])
        db.t["sh_se"].copy_(torch.tensor(np.concatenate(se)))
        db.forward()
        torch.cuda.synchronize()
        errs = [_compare(db, m, ref, r["z"], r["xyz"]) for m, (r, ref) in
                enumerate(zip(mols, refs, strict=True))]  # fmt: skip
        out[key] = {k: max(e[k] for e in errs) for k in errs[0]}
        print(key, " ".join(f"{k} {v:.1e}" for k, v in out[key].items()), flush=True)
    return out


def _random_inputs(hb, rng):
    mat = np.zeros(hb.mat_total)
    P, X = mat.copy(), mat.copy()
    for m in range(hb.nmol):
        o, n = int(hb.arrays["mol_mat"][m]), int(hb.arrays["mol_nao"][m])
        a, b = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        P[o : o + n * n] = ((a + a.T) / 2).ravel()
        X[o : o + n * n] = (0.05 * (b + b.T)).ravel()
    tens = lambda a: torch.from_numpy(np.ascontiguousarray(a)).cuda()  # noqa: E731
    return {
        "P": tens(P), "X": tens(X), "vao": tens(0.1 * rng.normal(size=hb.ao_total)),
        "vdp": tens(0.1 * rng.normal(size=3 * hb.nat)),
        "vqp": tens(0.1 * rng.normal(size=6 * hb.nat)),
        "se": tens(-0.5 + 0.1 * rng.normal(size=hb.nsh)),
        "dsedcn": tens(0.01 * rng.normal(size=hb.nsh)),
    }  # fmt: skip


def gpu_functional(db, g, m: int, ao_atom: np.ndarray) -> float:
    """`reference.h0_functional` from the GPU matrices of molecule m."""
    S, H0 = db.matrix(db.S, m)[0], db.matrix(db.H0, m)[0]
    D, Q = db.matrix(db.D, m, 3), db.matrix(db.Q, m, 6)
    off, n = int(db.hb.arrays["mol_mat"][m]), int(db.hb.arrays["mol_nao"][m])
    ao0 = int(db.hb.arrays["mol_ao"][m])
    a0 = int(np.flatnonzero(db.hb.arrays["atom_mol"] == m)[0])
    P = g["P"][off : off + n * n].reshape(n, n).cpu().numpy()
    X = g["X"][off : off + n * n].reshape(n, n).cpu().numpy()
    vao = g["vao"][ao0 : ao0 + n].cpu().numpy()
    nat = int(ao_atom.max()) + 1
    vdp = g["vdp"].reshape(-1, 3)[a0 : a0 + nat].cpu().numpy().T
    vqp = g["vqp"].reshape(-1, 6)[a0 : a0 + nat].cpu().numpy().T
    off_site = ao_atom[:, None] != ao_atom[None, :]
    p, x = P * off_site, X * off_site
    e = 0.5 * float((-p * (vao[:, None] + vao[None, :]) * S).sum())
    e += float((p * H0).sum()) - float((x * S).sum())
    e -= float(np.einsum("mn,cmn,cn->", p, D, vdp[:, ao_atom]))
    e -= float(np.einsum("mn,cmn,cn->", p, Q, vqp[:, ao_atom]))
    return e


def check_fd(sample: dict, n: int, h: float = 1e-4, ncoord: int = 6) -> dict:
    rng, out = np.random.default_rng(0), {}
    for key, mols in sample.items():
        worst = 0.0
        for r in mols[:n]:
            z, xyz = np.array(r["z"]), np.array(r["xyz"])
            hb = native.host_batch([(z, xyz)])
            db, g = native.DeviceBatch(hb), _random_inputs(hb, rng)
            db.t["sh_se"].copy_(g["se"])
            db.t["sh_dsedcn"].copy_(g["dsedcn"])
            db.h0_gradient(g["P"], g["X"], g["vao"], g["vdp"], g["vqp"])
            grad = db.grad.cpu().numpy().reshape(-1, 3)
            ao_atom = mol_basis(z).ao_atom
            for _ in range(ncoord):
                a, k = int(rng.integers(len(z))), int(rng.integers(3))
                vals = []
                for sgn in (1, -1):
                    x2 = xyz.copy()
                    x2[a, k] += sgn * h
                    db.set_xyz(x2)
                    db.forward()
                    vals.append(gpu_functional(db, g, 0, ao_atom))
                worst = max(worst, abs((vals[0] - vals[1]) / (2 * h) - grad[a, k]))
        out[key] = worst
        print(key, f"max |analytic - FD| {worst:.1e}", flush=True)
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


def time_bins(sample: dict, batches: list[int], reps: int, mem_gb: float = 10.0) -> dict:
    rng, out = np.random.default_rng(0), {}
    for key, mols in sample.items():
        out[key] = {}
        for nb in batches:
            picked = [mols[i % len(mols)] for i in range(nb)]
            nao2 = sum(int(r["nao"]) ** 2 for r in picked)
            if nao2 * 8 * 13 > mem_gb * 1e9:
                print(key, nb, "skipped: memory", flush=True)
                continue
            hb = native.host_batch([(np.array(r["z"]), np.array(r["xyz"])) for r in picked])
            db, g = native.DeviceBatch(hb), _random_inputs(hb, rng)
            db.t["sh_se"].copy_(g["se"])
            fwd = _time(db.forward, reps)
            grd = _time(partial(db.h0_gradient, g["P"], g["X"], g["vao"], g["vdp"], g["vqp"]), reps)
            row = {"forward_ms": fwd / nb, "gradient_ms": grd / nb, "total_ms": (fwd + grd) / nb}
            out[key][nb] = row
            print(key, nb, " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
            db = g = None
            torch.cuda.empty_cache()
        best = min((r["total_ms"] for r in out[key].values()), default=float("nan"))
        gate = BUDGET_MS[key] / 3
        out[key]["gate_ms"], out[key]["best_ms"] = gate, best
        print(key, f"best {best:.4f} ms vs gate {gate:.3f} ms:",
              "PASS" if best <= gate else "FAIL", flush=True)  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("check-tblite", "check-fd", "time"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--root", type=Path, default=Path("/home/jacopo/Documents/ksmech"))
    args = ap.parse_args()
    sample = load_sample(args.root)
    t0 = time.time()
    if args.what == "check-tblite":
        res = check_tblite(sample, args.n or 16)
    elif args.what == "check-fd":
        res = check_fd(sample, args.n or 2)
    else:
        res = time_bins(sample, args.batch, args.reps)
    res = {"result": res, "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name()}
    path = args.root / OUT / f"stage1_{args.what}.json"
    path.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
