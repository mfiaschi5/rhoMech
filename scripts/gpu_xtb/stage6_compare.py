"""Stage 6: read the benchmark's runs (runs/<set>__<arm>/) and print, per set and arm, the wall
time, species and ok records per hour, outcomes and failure codes, the GPU routes, polish and
retry counts, GPU use, and the validity checks (every species computed in the arm itself, no
foreign GPU process or pytest during it). Then the record differences against the tblite-anc arm
on the common-ok species: energy and the Kabsch RMSD of all atoms (lbfgs vs anc as the reference
spread). Writes bench_summary.json beside the runs. Exit 1 if a finished arm is invalid."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jacopo/Documents/ksmech/.claude/worktrees/gpu-xtb/src")
from ksmech.qm.records import load_record  # noqa: E402

IN = Path("/home/jacopo/Documents/ksmech/data/processed/gpu_qm_benchmark")
BINS = [(1, 15), (16, 30), (31, 50), (51, 80), (81, 120), (121, 200)]


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def gpu_stats(path):
    try:
        g = pd.read_csv(
            path, header=None, names=["t", "util", "mem", "power"], skipinitialspace=True
        )
        util = g.util.str.rstrip(" %").astype(float)
        mem = g.mem.str.rstrip(" MiB").astype(float)
        return dict(
            util_mean=round(util.mean(), 1),
            util_busy=round((util > 0).mean(), 3),
            mem_max_mib=int(mem.max()),
        )
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e)[:80])


def foreign(state_path):
    """Lines of state.txt that show a GPU process other than the desktop, or a pytest/fit."""
    out = []
    for line in Path(state_path).read_text().splitlines():
        if line.startswith(("start ", "end ")):
            continue
        if "kwin" in line or "Xwayland" in line:
            continue
        out.append(line[:120])
    fp = Path(state_path).with_name("foreign.txt")
    if fp.exists():
        out += [x[:120] for x in fp.read_text().splitlines() if x.strip()]
    return out


def load_arm(d):
    w = json.loads((d / "wall.json").read_text())
    s = w["set"]
    m = pd.read_parquet(d / "proc" / "species" / s / "qm_manifest.parquet")
    n_in = len(json.loads((IN / f"{s}.json").read_text()))
    ok = m[m.status == "ok"]
    stale = [p for p in ok.cache_path if os.path.getmtime(p) < w["start"]]
    det = m.detail.fillna("")
    warn = Counter(x for ws in m.warnings.fillna("") for x in str(ws).split(",") if x)
    a = dict(
        set=s,
        arm=w["arm"],
        wall_s=round(w["wall_s"], 1),
        exit=w["exit"],
        n_in=n_in,
        n_rows=len(m),
        n_ok=len(ok),
        species_per_h=round(len(m) / w["wall_s"] * 3600, 1),
        ok_per_h=round(len(ok) / w["wall_s"] * 3600, 1),
        failures=dict(Counter(m.reason[m.status != "ok"])),
        gpu_rows=int(det.str.startswith("optimized on the GPU").sum()),
        gpu_retried=int(det.str.contains("GPU retried on the CPU").sum()),
        polished=warn.get("GPU_POLISHED", 0),
        restrained=warn.get("RESTRAINED_GEOM", 0),
        memo_hits=int(det.str.startswith("known failure").sum()),
        stale_records=len(stale),
        foreign=foreign(d / "state.txt"),
        gpu=gpu_stats(d / "gpu.csv"),
    )
    routes = [x for x in (d / "run.log").read_text().splitlines() if "GPU routes" in x]
    a["routes"] = routes[-1][len("GPU routes and outcomes: ") :] if routes else ""
    bp = d / "busy.txt"
    busy = (
        Counter(" ".join(x.split()[4:])[:60] for x in bp.read_text().splitlines())
        if bp.exists()
        else {}
    )
    a["load_start_end"] = [
        x.split("loadavg ")[1].split()[0]
        for x in (d / "state.txt").read_text().splitlines()
        if x.startswith(("start ", "end "))
    ]
    a["busy_other"] = dict(busy.most_common(5))  # soft: other CPU-heavy processes (samples)
    a["valid"] = w["exit"] == 0 and len(m) == n_in and not stale and not a["memo_hits"]
    return a, m


def records(m):
    out = {}
    for r in m[m.status == "ok"].itertuples():
        rec = load_record(r.cache_path)
        out[r.smiles] = (rec.energy_ha, rec.coords_ang, rec.atomic_numbers)
    return out


def diff(ra, rb):
    common = sorted(set(ra) & set(rb))
    de, rmsd = [], []
    for s in common:
        ea, xa, za = ra[s]
        eb, xb, zb = rb[s]
        if len(za) != len(zb) or (za != zb).any():
            continue
        de.append(ea - eb)
        rmsd.append(kabsch_rmsd(xa, xb))
    de, rmsd = np.abs(np.array(de)), np.array(rmsd)
    if not len(de):
        return dict(n=0)
    return dict(
        n=len(de),
        dE_median=float(np.median(de)),
        dE_p90=float(np.quantile(de, 0.9)),
        dE_max=float(de.max()),
        lt_1e5=int((de < 1e-5).sum()),
        lt_1e4=int((de < 1e-4).sum()),
        lt_1e3=int((de < 1e-3).sum()),
        rmsd_median=float(np.median(rmsd)),
        rmsd_p90=float(np.quantile(rmsd, 0.9)),
        rmsd_lt_0p1=int((rmsd < 0.1).sum()),
    )


def n_atoms_of(m_sets):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return {s: Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms() for s in m_sets}


def per_bin(arms, ref, natoms):
    """Outcomes and the record differences by size bin (the ReactMech slice)."""
    out = {}
    for a, b in BINS:
        key, keep = f"{a}-{b}", {s for s, n in natoms.items() if a <= n <= b}
        row = {}
        for name, (_, m, recs) in arms.items():
            mm = m[m.smiles.isin(keep)]
            row[name] = dict(n=len(mm), ok=int((mm.status == "ok").sum()))
            if name != "anc" and ref is not None:
                row[name]["vs_anc"] = diff(
                    {s: v for s, v in recs.items() if s in keep},
                    {s: v for s, v in ref.items() if s in keep},
                )
        out[key] = row
    return out


def main(root):
    runs = sorted(p for p in (Path(root) / "runs").iterdir() if (p / "wall.json").exists())
    by_set, bad = {}, []
    for d in runs:
        try:
            a, m = load_arm(d)
        except FileNotFoundError as e:
            bad.append((d.name, False, [f"no manifest: {e.filename}"]))
            print(
                f"\n!! {d.name}: no manifest "
                f"(exit {json.loads((d / 'wall.json').read_text())['exit']})"
            )
            continue
        by_set.setdefault(a["set"], {})[a["arm"]] = (a, m, records(m))
        if not a["valid"] or a["foreign"]:
            bad.append((d.name, a["valid"], a["foreign"]))
    summary = {}
    for s, arms in by_set.items():
        ref = arms["anc"][2] if "anc" in arms else None
        print(f"\n== {s}")
        rows = {}
        for name, (a, _m, recs) in arms.items():
            if ref is not None and name != "anc":
                a["vs_anc"] = diff(recs, ref)
            rows[name] = a
            print(
                f"  {name:7s} wall {a['wall_s']:8.1f} s  {a['species_per_h']:8.1f} species/h  "
                f"ok {a['n_ok']}/{a['n_in']}  fail {a['failures']}  valid {a['valid']}  "
                f"load {a['load_start_end']}"
            )
            if a["busy_other"]:
                print(f"          other busy processes (30 s samples): {a['busy_other']}")
            if a["gpu_rows"] or name.startswith("gpu"):
                print(
                    f"          routes: {a['routes']}  retried {a['gpu_retried']}  "
                    f"polished {a['polished']}  gpu {a['gpu']}"
                )
            if "vs_anc" in a:
                print(f"          vs anc: {a['vs_anc']}")
        for name in [k for k in rows if k != "anc" and "anc" in rows]:
            rows[name]["speedup_vs_anc"] = round(rows["anc"]["wall_s"] / rows[name]["wall_s"], 3)
        if "lbfgs" in rows:
            for name in rows:
                rows[name]["speedup_vs_lbfgs"] = round(
                    rows["lbfgs"]["wall_s"] / rows[name]["wall_s"], 3
                )
        print(
            "  speed-ups:",
            {k: (v.get("speedup_vs_lbfgs"), v.get("speedup_vs_anc")) for k, v in rows.items()},
        )
        if s.startswith("reactmech"):
            natoms = n_atoms_of(set().union(*(set(m.smiles) for _, m, _ in arms.values())))
            rows["_per_bin"] = per_bin(arms, ref, natoms)
            for k, v in rows["_per_bin"].items():
                print(f"  bin {k}: {v}")
        summary[s] = rows
    (Path(root) / "bench_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if bad:
        print("\nINVALID OR CONTAMINATED ARMS:")
        for b in bad:
            print(" ", b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(IN)))
