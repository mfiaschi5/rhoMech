"""Stage 6 inputs: the spike's 64 species per bin (gpu_smoke.json, in bin order) as one file per
bin, and 1,000 random species of the full ReactMech v19 QM manifest (any status, seed 0, the
spike's species left out). Atom counts with H from RDKit."""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
BINS = [(1, 15), (16, 30), (31, 50), (51, 80), (81, 120), (121, 200)]
OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/jacopo/Documents/ksmech/data/processed/gpu_qm_benchmark"
)


def n_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.AddHs(mol).GetNumAtoms() if mol is not None else -1


def bin_of(n):
    return next((f"{a}-{b}" for a, b in BINS if a <= n <= b), "other")


spike = json.loads(
    Path("/home/jacopo/Documents/ksmech/data/processed/gpu_qm_spike/gpu_smoke.json").read_text()
)
for i, (a, b) in enumerate(BINS):
    part = spike[64 * i : 64 * (i + 1)]
    assert all(a <= n_atoms(s) <= b for s in part), (a, b)
    Path(f"{OUT}/spike_{a}-{b}.json").write_text(json.dumps(part))
m = pd.read_parquet(
    "/home/jacopo/Documents/ksmech/data/processed_full/reactmech/qm_manifest.parquet"
)
m = m[~m.smiles.isin(set(spike))].drop_duplicates("smiles")
pick = m.sample(n=1000, random_state=0).copy()
pick["n_atoms"] = [n_atoms(s) for s in pick.smiles]
pick["bin"] = [bin_of(n) for n in pick.n_atoms]
Path(f"{OUT}/reactmech_1000.json").write_text(json.dumps(pick.smiles.tolist()))
pick[["smiles", "status", "reason", "seconds", "n_atoms", "bin"]].to_parquet(
    f"{OUT}/reactmech_1000_prod.parquet"
)
print(Counter(pick.bin), Counter(pick.status), Counter(pick.reason[pick.status != "ok"]))
total = pick.seconds.sum()
print(f"production lbfgs seconds sum {total:.0f} -> /12 = {total / 12:.0f} s")
print(pick.groupby("bin").seconds.agg(["count", "sum", "median"]))
