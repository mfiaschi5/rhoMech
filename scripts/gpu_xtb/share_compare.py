"""gpu-xtb work sharing: read the runs of share_job.sh (runs/<set>__<arm>/) with stage 6's
reader (stage6_compare.py: walls, rates, outcomes, routes, GPU use, validity, the record
differences against the tblite-anc arm, per size bin), then per gpu-anc arm the species idle
CPU workers took from the GPU's queue: their count and sizes, their records against the anc arm
(the CPU path, so the same to the last bits) and the GPU-optimized ones'. Writes
share_summary.json beside the runs. Exit 1 if an arm is invalid."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage6_compare as s6  # noqa: E402

SHARE_NOTE = "shared from the GPU's queue"


def _subset(recs, keep):
    return {s: v for s, v in recs.items() if s in keep}


def share_rows(arms):
    """Per gpu-anc arm: the shared and GPU-optimized species and their records against anc."""
    if "anc" not in arms:
        return {}
    ref, out = arms["anc"][2], {}
    for name, (a, m, recs) in arms.items():
        if not name.startswith("gpu"):
            continue
        det = m.detail.fillna("")
        shared = set(m.smiles[det.str.contains(SHARE_NOTE, regex=False)])
        on_gpu = set(m.smiles[det.str.startswith("optimized on the GPU")])
        natoms = s6.n_atoms_of(shared) if shared else {}
        sizes = np.array(list(natoms.values()) or [0])
        out[name] = dict(
            wall_s=a["wall_s"],
            speedup_vs_anc=a.get("speedup_vs_anc"),
            n_shared=len(shared),
            n_gpu_rows=len(on_gpu),
            gpu=a["gpu"],
            shared_atoms_median=float(np.median(sizes)),
            shared_atoms_max=int(sizes.max()),
            shared_by_bin={
                f"{lo}-{hi}": int(((sizes >= lo) & (sizes <= hi)).sum())
                for lo, hi in s6.BINS
                if shared
            },  # fmt: skip
            shared_vs_anc=s6.diff(_subset(recs, shared), ref),
            gpu_rows_vs_anc=s6.diff(_subset(recs, on_gpu), ref),
        )
    return out


def main(root):
    status = s6.main(root)
    by_set = {}
    for d in sorted(p for p in (Path(root) / "runs").iterdir() if (p / "wall.json").exists()):
        try:
            a, m = s6.load_arm(d)
        except FileNotFoundError:
            continue
        by_set.setdefault(a["set"], {})[a["arm"]] = (a, m, s6.records(m))
    summary = {}
    for s, arms in by_set.items():
        if "anc" in arms:
            for a, _, _ in arms.values():
                a["speedup_vs_anc"] = round(arms["anc"][0]["wall_s"] / a["wall_s"], 3)
        summary[s] = share_rows(arms)
        print(f"\n== sharing, {s}")
        for name, row in summary[s].items():
            print(
                f"  {name:14s} wall {row['wall_s']:7.1f} s  x{row['speedup_vs_anc']} vs anc  "
                f"shared {row['n_shared']}  GPU rows {row['n_gpu_rows']}  gpu {row['gpu']}"
            )
            if row["n_shared"]:
                print(
                    f"    shared atoms median {row['shared_atoms_median']} max "
                    f"{row['shared_atoms_max']} by bin {row['shared_by_bin']}"
                )
                print(f"    shared vs anc: {row['shared_vs_anc']}")
            print(f"    GPU rows vs anc: {row['gpu_rows_vs_anc']}")
    (Path(root) / "share_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
