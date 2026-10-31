"""The spin-state scan (open-shell plan P2 steps 2 to 4): A, B and their combination into
spin_theory_v1 and spin_table_v1.

    uv run python scripts/probes/spin_scan.py [--workers 8] [--write]

Species: every u >= 2 fragment of PMechDB and ReactMech (either side of any step; the
placeholder law's shapes, `domain.is_placeholder`, are counted and left out) and of FlowER's
RS/RC lines (scripts/probes/flower_species_census.py), the PMechDB Cu(II)/Fe(III) species and
the METAL_SUSPECT fragments of open_shell_probe.py. (A): each species at every unpaired count
its electron count and d shells allow (u, u - 2, ... ; Fe(III) d5 5, 3, 1; Cu(II) d9 1; Ni(II)
d8 and Cr(IV) d2 2, 0) is optimized with spGFN2 in ALPB water from its production geometry and
from it displaced by a fixed kick (bond walls when a bond breaks); a start whose optimum formed a
bond the drawing lacks is a state of another molecule and fails (BOND_FORMED). The lower energies
at the optima are compared against the method's error bar (`data.spin_table.ERROR_BARS`). (B):
`data.spin_theory` on the restricted lowest-spin record at the production geometry; the report
adds B at A's displaced optimum as a diagnostic. (C): `data.spin_table.combine`. With --write the
two tables go to src/ksmech/data/spin_tables/ (commit them with no build running); the report,
with every A-B disagreement, goes to data/cache/probe_open_shell/scan_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import flower_species_census as census
import numpy as np
import open_shell_probe as probe
from rdkit import RDLogger

from ksmech.data import chem, domain, spin_table
from ksmech.data.io import read_steps, steps_path
from ksmech.processes import in_processes
from ksmech.qm.embed import prepare_mol
from ksmech.qm.geometry import BOHR_ANG


def dataset_species(processed: Path = probe.PROCESSED) -> tuple[list[probe.Species], Counter]:
    """u >= 2 fragments of every PMechDB and ReactMech step, and the placeholders left out."""
    found: dict[str, Counter] = {}
    left_out: Counter = Counter()
    for src in ("pmechdb", "reactmech"):
        for step in read_steps(steps_path(src, processed)):
            for side in (step.reactants_smiles, step.products_smiles):
                for frag in chem.fragments(chem.parse_mapped(side)):
                    if sum(a.GetNumRadicalElectrons() for a in frag.GetAtoms()) < 2:
                        continue
                    smi = chem.unmapped_canonical(frag)
                    if domain.is_placeholder(frag):
                        left_out[smi] += 1
                        continue
                    found.setdefault(smi, Counter())[src] += 1
    out = [probe.species_for(s, "organic", origin=dict(c)) for s, c in sorted(found.items())]
    return out, left_out


def flower_species() -> list[probe.Species]:
    rep = census.report(census.radical_lines())
    return [
        probe.species_for(r["smiles"], "organic", origin={"flower": dict(r["lines"])})
        for r in rep["multi_radical"]["species"]
    ]


# A starts every state twice: from the production geometry and from it displaced by a fixed
# Gaussian kick (seed 0, 0.02 Angstrom per coordinate). A gradient optimizer started on a
# symmetry element cannot leave it (the gradient across it vanishes by symmetry), so the
# production start alone can stop at a symmetric saddle: the ETKDG embedding draws Cl[C]Cl
# linear. A state's energy is the lowest of its converged starts (docs/decisions.md).
KICK_ANG = 0.02
KICK_SEED = 0
STARTS = ("production", "displaced")
NO_DISPLACED = "no displaced optimum"


def kick(shape: tuple[int, ...]) -> np.ndarray:
    """The fixed symmetry-breaking displacement (Bohr)."""
    return np.random.default_rng(KICK_SEED).normal(0.0, KICK_ANG / BOHR_ANG, shape)


def scan_start(task: tuple[probe.Species, int, str]) -> dict:
    """One state of one species from one start: spGFN2 optimization, then the energy at the
    optimum. Never raises: a failure comes back with its reason."""
    sp, u, start = task
    tag = {"u": u, "smiles": sp.smiles, "start": start}
    try:
        rec = probe.production(sp)
    except Exception as exc:  # noqa: BLE001 - counted per state
        return {"status": "failed", "reason": "NO_PRODUCTION", "detail": str(exc)} | tag
    mol = prepare_mol(sp.smiles)
    method = f"GFN2-xTB/{probe.medium(probe.SOLVENT)}/sp/u{u}/opt-from-prod-u{rec.unpaired}"
    x0 = rec.coords_ang / BOHR_ANG
    if start == "displaced":
        method += f"/kick{KICK_ANG}-s{KICK_SEED}"
        x0 = x0 + kick(x0.shape)

    def compute() -> dict:
        t0 = time.perf_counter()
        xyz, walls = probe.optimize(mol, x0, rec.charge, u, probe.SOLVENT, True)
        point = probe.single_point(rec.atomic_numbers, xyz, rec.charge, u, probe.SOLVENT, True)
        return {
            "u": u,
            "energy_ha": point["energy_ha"],
            "restrained": walls,
            "coords_ang": (xyz * BOHR_ANG).tolist(),
            "frontier": probe.sp_frontier(point),
            "seconds": time.perf_counter() - t0,
        }

    return probe.cached(probe.PROBE_ROOT / "scan", sp.smiles, method, compute) | tag


def drawing_kept(run: dict) -> dict:
    """A run whose optimum formed a bond the drawing lacks (`probe.formed_bonds`) is a state of
    another molecule, not of the drawing: it fails with BOND_FORMED, the pairs in `detail`."""
    if run["status"] != "ok":
        return run
    bonds = probe.formed_bonds(prepare_mol(run["smiles"]), np.asarray(run["coords_ang"]))
    return run | {"status": "failed", "reason": "BOND_FORMED", "detail": bonds} if bonds else run


def scan_state(runs: list[dict]) -> dict:
    """One state from its starts: the lowest energy among the converged starts that kept the
    drawing, with every start kept."""
    runs = [drawing_kept(r) for r in runs]
    keep = ("status", "energy_ha", "restrained", "reason", "detail")
    starts = {r["start"]: {k: r.get(k) for k in keep} for r in runs}
    ok = [r for r in runs if r["status"] == "ok"]
    if not ok:
        reason = "/".join(sorted({str(r.get("reason")) for r in runs}))
        head = {"u": runs[0]["u"], "smiles": runs[0]["smiles"], "status": "failed"}
        return head | {"reason": reason, "starts": starts, "displaced_coords": None}
    best = min(ok, key=lambda r: r["energy_ha"])
    shifted = [r.get("coords_ang") for r in ok if r["start"] == "displaced"]
    return best | {"starts": starts, "displaced_coords": shifted[0] if shifted else None}


def a_verdict(sp: probe.Species, states: list[dict]) -> dict:
    """A for one species: every state's energy relative to the lowest (eV) and the verdict at
    the error bar of its kind (a transition metal present or not)."""
    tm = any(probe._tm(a) for a in prepare_mol(sp.smiles).GetAtoms())
    kind = "transition_metal" if tm else "organic"
    energies = {r["u"]: r.get("energy_ha") if r["status"] == "ok" else None for r in states}
    ev = {u: None if e is None else e * probe.HA_EV for u, e in energies.items()}
    bar = spin_table.ERROR_BARS[kind]["ev"]
    verdict = spin_table.scan_verdict(ev, bar)
    base = min((e for e in ev.values() if e is not None), default=None)
    return {
        "kind": kind,
        "bar_ev": bar,
        "lowest": verdict.lowest,
        "gap_ev": verdict.gap_ev,
        "conclusive": verdict.conclusive,
        "states": {
            str(r["u"]): {
                "energy_ha": energies[r["u"]],
                "relative_ev": None if ev[r["u"]] is None else ev[r["u"]] - base,
                "restrained": r.get("restrained"),
                "status": r["status"],
                "reason": r.get("reason"),
                "start": r.get("start") if r["status"] == "ok" else None,
                "starts": r.get("starts"),
            }
            for r in states
        },
    }


def b_verdict(sp: probe.Species) -> dict:
    """B for one species on its restricted lowest-spin record at the production geometry."""
    from ksmech.data import spin_theory

    try:
        rec = probe.theory_record(sp, probe.production(sp))
    except Exception as exc:  # noqa: BLE001 - B is then undetermined, with the reason
        return {"unpaired": None, "rule_chain": ["no restricted record"], "detail": str(exc)}
    return spin_theory.theory_verdict(prepare_mol(sp.smiles), rec).as_dict()


def b_displaced(task: tuple[probe.Species, list | None]) -> dict:
    """Diagnostic, not the table's B: B on a restricted lowest-spin record at A's optimum of
    the lowest state from the displaced start (a geometry off any symmetric saddle)."""
    from ksmech.data import spin_theory

    sp, coords = task
    if coords is None:
        return {"smiles": sp.smiles, "unpaired": None, "rule_chain": [NO_DISPLACED]}
    u0 = min(sp.states)
    try:
        root = probe.PROBE_ROOT / "b_displaced"
        rec = probe.record_for(sp.smiles, u0, probe.SOLVENT, root, given=np.asarray(coords))
    except Exception as exc:  # noqa: BLE001 - reported with the reason
        return {"smiles": sp.smiles, "unpaired": None, "rule_chain": [f"no record: {exc}"]}
    verdict = spin_theory.theory_verdict(prepare_mol(sp.smiles), rec).as_dict()
    return verdict | {"smiles": sp.smiles}


def table_entry(sp: probe.Species, a: dict, b: dict) -> dict:
    """One spin_table_v1 entry: rule C on A and B, every state, both verdicts."""
    scan = spin_table.ScanVerdict(a["lowest"], a["gap_ev"], a["conclusive"])
    unpaired, case = spin_table.combine(scan, b.get("unpaired"))
    mult = unpaired + 1 if isinstance(unpaired, int) else unpaired
    a_out = {k: v for k, v in a.items() if k != "states"}
    return {
        "unpaired": unpaired,
        "multiplicity": mult,
        "case": case,
        "states": a["states"],
        "a": a_out,
        "b": b,
        "groups": list(sp.groups),
        "origin": sp.origin,
    }


def run_scan(species: list[probe.Species], workers: int) -> dict[str, list[dict]]:
    """Every state of every species from both starts, combined per state and grouped by
    SMILES: the species the table serves (organic, Cu/Fe) before the METAL_SUSPECT
    diagnostics, each longest first."""
    tasks = [(sp, u, s) for sp in species for u in sp.states for s in STARTS]
    tasks.sort(key=lambda t: (t[0].groups == ("suspect",), -prepare_mol(t[0].smiles).GetNumAtoms()))
    runs: dict[tuple[str, int], list[dict]] = {}
    for k, res in enumerate(in_processes(scan_start, tasks, workers), 1):
        runs.setdefault((res["smiles"], res["u"]), []).append(res)
        tag = f"{res['status']} u={res['u']} {res['start']}"
        print(f"step {k} / {len(tasks)} {tag} {res['smiles']}", flush=True)
    out: dict[str, list[dict]] = {sp.smiles: [] for sp in species}
    for (smiles, _), state_runs in runs.items():
        out[smiles].append(scan_state(state_runs))
    return out


def displaced_coords(sp: probe.Species, states: list[dict]) -> list | None:
    """A's optimum of the lowest state from the displaced start, when it converged."""
    return next((r["displaced_coords"] for r in states if r["u"] == min(sp.states)), None)


def gather(processed: Path = probe.PROCESSED) -> tuple[list[probe.Species], dict]:
    data, left_out = dataset_species(processed)
    groups = [data, flower_species(), probe.metal_species(processed)]
    species = probe.merge_species([*groups, probe.suspect_species(processed)])
    counts = {"placeholders_left_out": dict(left_out), "n_species": len(species)}
    counts["by_group"] = dict(Counter(g for sp in species for g in sp.groups))
    return species, counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--write", action="store_true", help="write the two tracked tables")
    parser.add_argument("--report", type=Path, default=probe.PROBE_ROOT / "scan_report.json")
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    species, counts = gather()
    species.sort(key=lambda s: -prepare_mol(s.smiles).GetNumAtoms())
    rows = {r["smiles"]: r for r in in_processes(probe.probe_species, species, args.workers)}
    ready = [sp for sp in species if rows[sp.smiles]["status"] == "ok"]
    scans = run_scan(ready, args.workers)
    shifted = [(sp, displaced_coords(sp, scans[sp.smiles])) for sp in ready]
    b_alt = {r["smiles"]: r for r in in_processes(b_displaced, shifted, args.workers)}
    entries, theory = {}, {}
    for sp in species:
        b = b_verdict(sp)
        theory[sp.smiles] = b
        none = [{"u": u, "status": "failed", "reason": "NO_PRODUCTION"} for u in sp.states]
        a = a_verdict(sp, scans.get(sp.smiles) or none)
        entries[sp.smiles] = table_entry(sp, a, b)
    counts["start_and_geometry"] = start_effects(entries, b_alt)
    counts["bond_formed"] = bond_formed(scans)
    write_outputs(entries, theory, rows, counts, args)


def bond_formed(scans: dict[str, list[dict]]) -> dict:
    """Every start whose optimum formed a bond the drawing lacks, by species and state: the
    pairs (atom indices, d / covalent-radius sum)."""
    out: dict[str, dict] = {}
    for smiles, states in scans.items():
        for r in states:
            hits = {
                start: s["detail"]
                for start, s in (r.get("starts") or {}).items()
                if s.get("reason") == "BOND_FORMED"
            }
            if hits:
                out.setdefault(smiles, {})[str(r["u"])] = hits
    return out


def start_effects(entries: dict, b_alt: dict) -> dict:
    """Diagnostics of the two starts: the species whose A changes when only the production
    start is kept, the species whose B or C would change with B read at A's displaced
    optimum instead of the production geometry, and those without that optimum."""
    a_changed, b_changed, no_optimum = {}, {}, []
    for smiles, e in entries.items():
        prod = {
            int(u): s["starts"]["production"]["energy_ha"]
            if s.get("starts") and s["starts"]["production"]["status"] == "ok"
            else None
            for u, s in e["states"].items()
        }
        ev = {u: None if x is None else x * probe.HA_EV for u, x in prod.items()}
        only = spin_table.scan_verdict(ev, e["a"]["bar_ev"])
        if (only.lowest, only.conclusive) != (e["a"]["lowest"], e["a"]["conclusive"]):
            a_changed[smiles] = {"production_only": only.__dict__, "both": e["a"]}
        alt = b_alt.get(smiles)
        if alt is None:
            continue
        if alt.get("rule_chain") == [NO_DISPLACED]:
            no_optimum.append(smiles)
            continue
        scan = spin_table.ScanVerdict(e["a"]["lowest"], e["a"]["gap_ev"], e["a"]["conclusive"])
        c_alt = spin_table.combine(scan, alt.get("unpaired"))
        if alt.get("unpaired") != e["b"].get("unpaired") or c_alt[0] != e["unpaired"]:
            b_changed[smiles] = {
                "b": e["b"].get("unpaired"),
                "b_displaced": alt.get("unpaired"),
                "rule_chain_displaced": alt.get("rule_chain"),
                "c": [e["unpaired"], e["case"]],
                "c_displaced": list(c_alt),
            }
    return {
        "a_changed_by_displaced_start": a_changed,
        "b_changed_off_saddle": b_changed,
        "b_no_displaced_optimum": no_optimum,
    }


def write_outputs(entries, theory, rows, counts, args) -> None:
    from ksmech.data import spin_theory

    theory_table = spin_theory.theory_table(theory)
    table = spin_table.with_hash(
        {
            "version": spin_table.VERSION,
            "schema": spin_table.SCHEMA,
            "method": "spGFN2-xTB (tblite 0.7.0, spin-polarization 1.0), every state optimized "
            "from the production geometry (restricted GFN2 at the drawn or metal_v1 "
            "multiplicity, the engine's embedding) and from it displaced by a fixed Gaussian "
            f"kick ({KICK_ANG} Angstrom, seed {KICK_SEED}); the lower energy among the "
            "converged starts that formed no bond the drawing lacks (non-metal pairs within "
            "the wall factor, 1.2 x the covalent-radius sum; else the state fails, BOND_FORMED)",
            "solvent": f"ALPB {probe.SOLVENT}",
            "theory_digest": theory_table["digest"],
            "error_bars": spin_table.ERROR_BARS,
            "entries": dict(sorted(entries.items())),
        }
    )
    spin_table.validate(table, theory_table)
    cases = Counter(e["case"] for e in entries.values())
    disagree = {s: e for s, e in entries.items() if e["case"] == spin_table.DISAGREE}
    report = {"counts": counts, "cases": dict(cases), "disagreements": disagree}
    report |= {"probe_failures": {s: r for s, r in rows.items() if r["status"] != "ok"}}
    probe._write_json(args.report, report | {"table": table, "theory": theory_table})
    if args.write:
        spin_table.TABLE_DIR.mkdir(exist_ok=True)
        spin_theory.write_theory_table(theory, spin_table.THEORY_V1)
        spin_table.SPECIES_V1.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    effects = counts["start_and_geometry"]
    short = {k: v for k, v in counts.items() if k not in ("start_and_geometry", "bond_formed")}
    short |= {k: len(v) for k, v in effects.items()} | {"bond_formed": len(counts["bond_formed"])}
    print(json.dumps({"counts": short, "cases": dict(cases), "n_disagree": len(disagree)}))


if __name__ == "__main__":
    main()
