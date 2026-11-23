"""Stage 4 of the GPU xTB plan: the batched ANC / RFO optimizer.

    uv run python scripts/gpu_xtb/stage4_gate.py steps [--per-bin 2] [--rfo fp64|fp32]
    uv run python scripts/gpu_xtb/stage4_gate.py check [--rfo fp64|fp32] [--capacity 256]
    uv run python scripts/gpu_xtb/stage4_gate.py time [--per-bin 64] [--seeds 1] [--bins 51-80]

Molecules: the spike sample's species (`data/processed/gpu_qm_spike/mols.json`), embedded as
production does (`prepare_mol`, `embed(mol, 1, seed=0)`), untagged closed shells, ALPB water.

- steps: `qm.anc.optimize_anc` on the CPU driven by the GPU energy (one molecule, each SCF
  restarted from the last evaluated point), against `BatchedANC` on the same molecule alone:
  the energies of every evaluation, the energy counts, the end points. Isolates the optimizer
  from the energy (the energy itself is stage 3's).
- check: `relax_many` (the GPU's `_relax`: free run with the guard, walls on a broken bond,
  refilled batches) from the exact starts of stage4_reference.json (production's tblite-anc
  `_relax`, scripts/gpu_xtb/stage4_reference.py), per species: the outcome against the
  reference's; at the GPU's end point tblite's energy and max|gradient| (fmax 3e-4: the share
  above it is what a CPU polish would take); same minimum (Kabsch RMSD < 0.05 A and |dE| <
  1e-5 Eh against the reference's end) or a different one, lower or higher; the energies used;
  the GPU's wall time over the whole sample against the reference's CPU seconds.
- time: per size bin, `relax_many` on `--per-bin` species of the spike sample, each embedded
  with `--seeds` seeds (production uses seed 0; more seeds put more molecules in flight), alone
  on the GPU: seconds and energies per molecule, and every failure with its reason. Against
  the reference's CPU seconds per molecule of that bin / 12, from stage4_reference_preload.json
  (workers started as qm.batch starts them, OpenBLAS preloaded) when it exists: an estimate of
  `--workers 12`, which stage 6 measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ksmech.qm import anc
from ksmech.qm.geometry import BOHR_ANG
from ksmech.qm.gpu import optimize as O
from ksmech.qm.gpu import scf as S

SAMPLE = Path("data/processed/gpu_qm_spike/mols.json")
OUT = Path("data/processed/gpu_qm_spike")


def embedded(smiles: str, seed: int = 0):
    """(z, start Bohr, charge) as qm.batch._compute embeds it; None for a tagged or open shell."""
    from rdkit import Chem

    from ksmech.qm import spin as qm_spin
    from ksmech.qm.embed import embed, prepare_mol

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)  # keep the embed notes

    mol = prepare_mol(smiles)
    ids = embed(mol, 1, seed=seed)
    charge, unpaired, tag = qm_spin.spin_for_qm(mol, smiles)
    if not ids or tag is not None or unpaired:
        return None
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return mol, z, mol.GetConformer(ids[0]).GetPositions() / BOHR_ANG, charge


def _embedded_job(job: tuple[str, int]):
    return embedded(*job)


def embed_many(jobs: list[tuple[str, int]], workers: int = 12) -> list:
    """`embedded` of every (smiles, seed) on `workers` CPU processes (spawned, one thread each:
    the parent holds a CUDA context), in order. The embedding is outside every timing."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_embedded_job, jobs, chunksize=4))


class GpuObjective:
    """One molecule's GPU energy and gradient, restarted from the last evaluated point."""

    def __init__(self, z, x0, charge) -> None:
        self.sb, self.start, self.log = S.SCFBatch([(z, x0, charge, 0)], True), None, []

    def __call__(self, x):
        self.sb.set_geometry(torch.as_tensor(x).cuda())
        r, _ = self.sb.run_with_fallback(self.start)
        e, g = self.sb.gradient(r)
        if bool(r.converged[0]):
            self.start = (r.qsh.clone(), r.dp.clone(), r.qp.clone())
        self.log.append(float(e[0]))
        return float(e[0]), g.cpu().numpy()


def steps(sample: dict, per_bin: int, rfo: str) -> dict:
    out = {}
    dtype = torch.float32 if rfo == "fp32" else torch.float64
    for key, recs in sample.items():
        rows = []
        for rec in recs[:per_bin]:
            got = embedded(rec["smiles"])
            if got is None:
                continue
            _, z, x0, charge = got
            obj = GpuObjective(z, x0, charge)
            cpu = anc.optimize_anc(obj, z, x0)
            opt = O.BatchedANC([O.OptSpec(z, x0, charge, 0)], rfo_dtype=dtype, trace=True)
            res = opt.run()[0]
            gpu_e = np.array([float(t[0][0]) for t in opt.trace])
            n = min(len(obj.log) - 1, len(gpu_e))
            de = np.abs(np.array(obj.log[1 : n + 1]) - gpu_e[:n])
            rows.append({
                "nat": len(z), "cpu_evals": cpu.n_evals, "gpu_evals": res.n_evals,
                "cpu_conv": bool(cpu.converged), "gpu_conv": res.converged,
                "max_dE_steps": float(de.max()) if n else 0.0,
                "first_step_over_1e-9": int(np.argmax(de > 1e-9)) if (de > 1e-9).any() else None,
                "end_dE": res.energy - min(obj.log),
                "end_rmsd_ang": float(np.sqrt(((res.xyz - cpu.xyz) ** 2).sum(1).mean()) * BOHR_ANG),
            })  # fmt: skip
            print(key, rows[-1], flush=True)
        out[key] = rows
    return out


def _tblite_at(z, xyz, charge):
    from ksmech.qm.geometry import new_calculator

    res = new_calculator(z, xyz, charge, 0, "water").singlepoint()
    return float(res.get("energy")), float(np.abs(np.array(res.get("gradient"))).max())


def _compare(row: dict, out) -> dict:
    from ksmech.qm.embed import kabsch_rmsd

    z, ref_ok = np.array(row["z"]), row["outcome"] == "ok"
    gpu_ok = out.reason is None
    c = {"bin": row["bin"], "nat": len(z), "ref": row["outcome"],
         "gpu": "ok" if gpu_ok else str(out.reason), "ref_evals": row["n_evals"],
         "gpu_evals": out.n_evals, "gpu_restrained": out.restrained,
         "ref_restrained": row.get("restrained"), "fresh_starts": out.fresh_starts}  # fmt: skip
    if gpu_ok:
        c["tb_energy"], c["tb_gmax"] = _tblite_at(z, out.xyz, row["charge"])
    if gpu_ok and ref_ok:
        c["rmsd_ang"] = kabsch_rmsd(np.array(row["end"]) * BOHR_ANG, out.xyz * BOHR_ANG)
        c["dE"] = c["tb_energy"] - row["energy"]
        same = c["rmsd_ang"] < 0.05 and abs(c["dE"]) < 1e-5
        c["minimum"] = "same" if same else ("lower" if c["dE"] < 0 else "higher")
    return c


def check(root: Path, rfo: str, capacity: int) -> dict:
    import time

    from ksmech.qm.embed import embed, prepare_mol

    ref = json.loads((root / OUT / "stage4_reference.json").read_text())
    rows = [r for rs in ref.values() for r in rs if r["outcome"] != "skipped"]
    tasks = []
    for k, row in enumerate(rows):
        mol = prepare_mol(row["smiles"])
        embed(mol, 1, seed=0)
        z, start = np.array(row["z"]), np.array(row["start"])
        tasks.append(O.RelaxTask(str(k), mol, z, start, row["charge"], 0))
    dtype = torch.float32 if rfo == "fp32" else torch.float64
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = O.relax_many(tasks, capacity=capacity, rfo_dtype=dtype,
                        on_batch=lambda *a: print("batch", a, flush=True))  # fmt: skip
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    per = [_compare(row, out) for row, out in zip(rows, outs, strict=True)]
    for c in per:
        print(c, flush=True)
    cpu_s = sum(r["seconds"] for r in rows)
    return {"per_species": per, "gpu_seconds": seconds, "cpu_seconds_one_core": cpu_s,
            "n": len(rows)}  # fmt: skip


def _cpu12(root: Path, key: str) -> tuple[float, str]:
    for name, label in (("stage4_reference_preload.json", "preload"),
                        ("stage4_reference.json", "no preload")):  # fmt: skip
        path = root / OUT / name
        if path.exists():
            ok = [r for r in json.loads(path.read_text())[key] if r["outcome"] == "ok"]
            return float(np.mean([r["seconds"] for r in ok])) / 12, label
    raise FileNotFoundError("no stage 4 reference")


def time_bins(root: Path, per_bin: int, seeds: int, bins: list[str] | None) -> dict:
    import time

    sample = json.loads((root / SAMPLE).read_text())
    out = {}
    for key, recs in sample.items():
        if bins and key not in bins:
            continue
        jobs = [(f"{key}:{seed}:{k}", rec["smiles"], seed)
                for seed in range(seeds) for k, rec in enumerate(recs[:per_bin])]  # fmt: skip
        tasks = [O.RelaxTask(name, *got, 0) for name, got in zip(
            [j[0] for j in jobs], embed_many([(j[1], j[2]) for j in jobs]), strict=True)
            if got is not None]  # fmt: skip
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = O.relax_many(tasks)
        torch.cuda.synchronize()
        gpu = (time.perf_counter() - t0) / len(tasks)
        cpu12, label = _cpu12(root, key)
        fails = [(r.key, str(r.reason), r.detail, r.n_evals) for r in res if r.reason is not None]
        out[key] = {"n": len(tasks), "gpu_s_per_mol": gpu, "cpu12_s_per_mol": cpu12,
                    "cpu_reference": label, "speedup": cpu12 / gpu,
                    "gpu_evals_mean": float(np.mean([r.n_evals for r in res])),
                    "failures": fails}  # fmt: skip
        print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[key].items()},
              flush=True)  # fmt: skip
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("steps", "check", "time"))
    ap.add_argument("--capacity", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--bins", nargs="+", default=None)
    ap.add_argument("--per-bin", type=int, default=2)
    ap.add_argument("--rfo", choices=("fp64", "fp32"), default="fp64")
    ap.add_argument("--root", type=Path, default=Path("/home/jacopo/Documents/ksmech"))
    args = ap.parse_args()
    if args.what == "check":
        res = check(args.root, args.rfo, args.capacity)
    elif args.what == "time":
        res = time_bins(args.root, args.per_bin, args.seeds, args.bins)
    else:
        res = steps(json.loads((args.root / SAMPLE).read_text()), args.per_bin, args.rfo)
    tag = f"_s{args.seeds}" if args.what == "time" and args.seeds > 1 else ""
    path = args.root / OUT / f"stage4_{args.what}_{args.rfo}{tag}.json"
    path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
