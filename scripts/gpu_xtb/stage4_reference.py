"""Stage 4's reference: production's CPU optimization (tblite-anc) of the spike sample's species.

    uv run python scripts/gpu_xtb/stage4_reference.py [--per-bin 16] [--workers 8] [--preload]
        [--bins 31-50 51-80] [--out NAME]

For each species: `prepare_mol` and `embed(mol, 1, seed=0)` as `qm.batch._compute` does, then
`GFN2Calculator._relax` with `QMConfig(optimizer="tblite-anc")` (free run with the dissociation
guard, a walled retry from the start when a bond breaks, `check_connectivity` after each). Only
untagged closed shells (`spin_for_qm`), the GPU engine's scope. Per species: the start and end
geometries (Bohr), the outcome (ok or the reason code), whether walls were needed, the number of
energies, the seconds, and tblite's energy and max|gradient| at the end point. Written once to
data/processed/gpu_qm_spike/stage4_reference.json (the GPU runs compare against it).
With --preload the workers start as `qm.batch.execute` starts them (spawned, one thread, the
OpenBLAS `qm.blas` chooses preloaded): production's CPU speed, written to
stage4_reference_preload.json; each row records the library behind tblite's dsygvd.
--bins limits the run to those size bins and --out names the file (in the same folder).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SAMPLE = Path("data/processed/gpu_qm_spike/mols.json")
OUT = Path("data/processed/gpu_qm_spike/stage4_reference.json")


def _one(smiles: str) -> dict:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ksmech.qm import geometry
    from ksmech.qm import spin as qm_spin
    from ksmech.qm.embed import embed, prepare_mol
    from ksmech.qm.geometry import BOHR_ANG, new_calculator
    from ksmech.qm.records import QMError
    from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

    row: dict = {"smiles": smiles}
    mol = prepare_mol(smiles)
    ids = embed(mol, 1, seed=0)
    charge, unpaired, tag = qm_spin.spin_for_qm(mol, smiles)
    row.update(charge=charge, unpaired=unpaired, tag=tag, n_embedded=len(ids))
    if not ids or tag is not None or unpaired:
        row["outcome"] = "skipped"
        return row
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    start = mol.GetConformer(ids[0]).GetPositions() / BOHR_ANG
    row.update(z=z.tolist(), start=start.tolist())
    calls = {"n": 0}
    orig = geometry.TbliteEnergy.__call__

    def counted(self, x):
        calls["n"] += 1
        return orig(self, x)

    geometry.TbliteEnergy.__call__ = counted
    calc = GFN2Calculator(QMConfig(optimizer="tblite-anc"))
    t0 = time.perf_counter()
    try:
        xyz, restrained = calc._relax(mol, z, start, charge, unpaired)
        row.update(outcome="ok", restrained=restrained, end=np.asarray(xyz).tolist())
        res = new_calculator(z, xyz, charge, unpaired, "water").singlepoint()
        row.update(energy=float(res.get("energy")),
                   gmax=float(np.abs(np.array(res.get("gradient"))).max()))  # fmt: skip
    except QMError as exc:
        row.update(outcome=str(exc.reason), detail=exc.detail)
    row.update(n_evals=calls["n"], seconds=time.perf_counter() - t0)
    from ksmech.qm import blas

    row["blas"] = blas.provider()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-bin", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", type=Path, default=Path("/home/jacopo/Documents/ksmech"))
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--bins", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sample = json.loads((args.root / SAMPLE).read_text())
    smiles = [(key, r["smiles"]) for key, recs in sample.items() for r in recs[: args.per_bin]
              if not args.bins or key in args.bins]  # fmt: skip
    out = {}
    with _pool(args.workers, args.preload) as pool:
        for (key, _smi), row in zip(smiles, pool.map(_one, [s for _, s in smiles]), strict=True):
            row["bin"] = key
            out.setdefault(key, []).append(row)
            print(key, row["outcome"], row.get("n_evals"), round(row.get("seconds", 0), 1),
                  row.get("blas"), flush=True)  # fmt: skip
    path = args.root / (OUT.with_name("stage4_reference_preload.json") if args.preload else OUT)
    if args.out:
        path = args.root / OUT.with_name(args.out)
    path.write_text(json.dumps(out) + "\n")
    print("wrote", path)


@contextmanager
def _pool(n: int, preload: bool):
    if not preload:
        with ProcessPoolExecutor(n) as pool:
            yield pool
        return
    from ksmech.processes import single_threaded_children
    from ksmech.qm import blas

    ctx = multiprocessing.get_context("spawn")
    with (
        single_threaded_children(),
        blas.preloaded_children(),
        ProcessPoolExecutor(n, mp_context=ctx) as pool,
    ):
        yield pool


if __name__ == "__main__":
    main()
