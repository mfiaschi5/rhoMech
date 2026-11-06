"""The open-shell QM probe (open-shell plan P2 steps 1 and 5) and the QM helpers of the spin
scan (step 2, scripts/probes/spin_scan.py).

    uv run python scripts/probes/open_shell_probe.py [--workers 8] [--out FILE]

Step 1: restricted GFN2 and spGFN2 (tblite's spin-polarization term) at the geometry production
uses, for three groups: the OPEN_SHELL fixtures plus ethane (the closed-shell check); the
PMechDB Cu(II)/Fe(III) species of the SPIN_PARITY failures at the table multiplicity (metal_v1:
Cu(II) d9 one unpaired electron, Fe(III) d5 five); and the METAL_SUSPECT fragments of kept steps
(H2CrO3, the Ni(II) d8 fragments of the dative repair) at their production records. The
production geometry is the engine's: `prepare_mol`, `embed(mol, 1, seed=0)` and
`GFN2Calculator.run` (restricted tblite L-BFGS, bond walls when a bond breaks) at that
multiplicity. Per species: both energies; the restricted occupations (integral, a degenerate
plateau or smeared) and max|n - aufbau|; which block (occupation >= 0.5, today's rule) each
open MO lands in; the restricted vs spGFN2 charges (max |dq| over all atoms, the non-metal
atoms, and the non-metal atoms the dataset arrows touch); eps_SOMO against the spGFN2 alpha
HOMO / beta LUMO midpoint; the vertical spGFN2 energy of every allowed multiplicity; the times.

Step 5: gas-phase vertical delta-SCF of twelve small radicals at the radical's restricted
gas-phase geometry, restricted and spGFN2: IE = E(N-1) - E(N), EA = E(N) - E(N+1), each ion in
the lower of its two lowest spin states.

Every QM result is cached under data/cache/probe_open_shell/, a scratch root no pipeline command
reads (docs/decisions.md): records with `save_record`, the rest as JSON keyed like records, and
a failure as a `.failed.json` beside its key. Failures are counted by reason, never dropped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from scipy.optimize import minimize
from tblite.interface import TBLiteRuntimeError, TBLiteValueError

from ksmech.data import chem, domain
from ksmech.orbitals.localize import SMEAR_TOL
from ksmech.processes import in_processes
from ksmech.qm import fukui
from ksmech.qm.embed import NOTES_PROP, embed, prepare_mol
from ksmech.qm.geometry import (
    BOHR_ANG,
    SKIP_BOND_ELEMENTS,
    bond_walls,
    check_connectivity,
    new_calculator,
)
from ksmech.qm.records import QMError, QMRecord, ReasonCode, cache_path, load_record, save_record
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.fixtures.molecules import OPEN_SHELL  # noqa: E402

PROBE_ROOT = Path("data/cache/probe_open_shell")
PROCESSED = Path("data/processed")
SOLVENT = "water"
SP_WEIGHT = 1.0  # tblite's spin-polarization scaling (its default)
HA_EV = fukui.HARTREE_TO_EV
OCC_TOL = 1e-6  # numerical equality of occupations, not a chemistry threshold
OPEN_TOL = SMEAR_TOL  # an MO this far from 0 and 2 is open: the engine's fermi_smeared bound
# metal_v1 (P4): the unpaired electrons the table gives a metal at its oxidation state
TABLE_U = {(29, 2): 1, (26, 3): 5}
FIXTURE_EXTRA = {"ethane": ("CC", 0)}  # the closed-shell check: spGFN2 dE must be 0
IPEA_RADICALS = {
    "CH3": "[CH3]", "OH": "[OH]", "HO2": "O[O]", "NO": "[N]=O", "NO2": "[O]N=O",
    "NO3": "[O][N+](=O)[O-]", "CH3O": "C[O]", "Br": "[Br]", "Cl": "[Cl]", "O2": "[O][O]",
    "allyl": "C=C[CH2]", "phenoxyl": "[O]c1ccccc1",
}  # fmt: skip

# Gas-phase literature values (eV) for step 5, typed from the sources (no download): the
# adiabatic ionization energy and electron affinity of each radical with its citation. They
# are compared only with the gas-phase vertical delta-SCF levels of `ipea_species` (U22).
NIST_IE = "NIST Chemistry WebBook, SRD 69, ion energetics: evaluated ionization energy"
NIST_EA = "NIST Chemistry WebBook, SRD 69, ion energetics: electron affinity"
IPEA_LITERATURE = {
    "CH3": (9.843, NIST_IE, 0.080, NIST_EA),
    "OH": (13.017, NIST_IE, 1.8277, NIST_EA),
    "HO2": (11.35, NIST_IE, 1.078, NIST_EA),
    "NO": (9.2642, NIST_IE, 0.026, "Travers, Cowles, Ellison 1989 (NO- photoelectron "
           "spectrum), as quoted by the Wikipedia article on nitric oxide"),
    "NO2": (9.586, NIST_IE, 2.273, NIST_EA),
    "NO3": (12.57, NIST_IE, 3.937, NIST_EA),
    "CH3O": (10.72, NIST_IE + " (a photoelectron value of 7.37 eV is also listed)", 1.5689,
             "Weichman et al. 2017 (cryo-SEVI of CH3O-)"),
    "Br": (11.8138, NIST_IE, 3.3636, NIST_EA),
    "Cl": (12.9676, NIST_IE, 3.6127, NIST_EA),
    "O2": (12.0697, NIST_IE, 0.448, NIST_EA),
    "allyl": (8.13, NIST_IE + " (a ZEKE value of 8.153 eV is also listed)", 0.481, NIST_EA),
    "phenoxyl": (8.56, NIST_IE, 2.2538, NIST_EA),
}  # fmt: skip


@dataclass
class Species:
    smiles: str
    groups: tuple[str, ...]
    u: int  # the production multiplicity (unpaired electrons)
    states: tuple[int, ...]  # every unpaired count the electron count and d shells allow
    record: str | None = None  # an existing production record
    arrow_atoms: tuple[int, ...] = ()  # record indices of the atoms dataset arrows touch
    origin: dict = field(default_factory=dict)


# ---------------------------------------------------------------- multiplicities


def d_states(n: int) -> tuple[int, ...]:
    """Unpaired counts a d^n ion allows: n, n - 2, ... for n <= 5, else by holes."""
    top = n if n <= 5 else 10 - n
    return tuple(range(top, -1, -2))


def allowed_states(mol: Chem.Mol, u: int) -> tuple[int, ...]:
    """Every unpaired count from the drawn radicals and the open d shells, highest first: the
    non-metal radicals r give r, r - 2, ...; each transition metal its d_states; summed."""
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms() if not _tm(a))
    sets = [tuple(range(radicals, -1, -2))]
    sets += [d_states(domain.d_count(a)) for a in mol.GetAtoms() if _tm(a)]
    sums = {sum(c) for c in itertools.product(*sets)}
    if not any(_tm(a) for a in mol.GetAtoms()):
        sums = set(range(u, -1, -2))
    return tuple(sorted(sums, reverse=True))


def _tm(atom: Chem.Atom) -> bool:
    return domain.d_count(atom) is not None and 0 <= domain.d_count(atom) <= 10


def table_u(mol: Chem.Mol) -> int:
    """Drawn radicals plus metal_v1's unpaired electrons of each Cu(II) and Fe(III)."""
    u = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    key = lambda a: (a.GetAtomicNum(), domain.oxidation_state(a))  # noqa: E731
    return u + sum(TABLE_U.get(key(a), 0) for a in mol.GetAtoms())


def electron_count(mol: Chem.Mol) -> int:
    return sum(a.GetAtomicNum() for a in mol.GetAtoms()) - chem.formal_charge(mol)


# ---------------------------------------------------------------- species


def species_for(smiles: str, group: str, u: int | None = None, **kw) -> Species:
    mol = prepare_mol(smiles)
    u = table_u(mol) if u is None else u
    return Species(smiles, (group,), u, allowed_states(mol, u), **kw)


def fixture_species() -> list[Species]:
    out = []
    for name, (drawn, _) in {**OPEN_SHELL, **FIXTURE_EXTRA}.items():
        smi = chem.unmapped_canonical(Chem.MolFromSmiles(drawn))
        out.append(species_for(smi, "fixture", origin={"fixture": name}))
    return out


def metal_species(processed: Path = PROCESSED) -> list[Species]:
    """The PMechDB SPIN_PARITY failures that hold a Cu or an Fe (the survey's kinds), with the
    record atoms the arrows of their steps touch."""
    import pandas as pd

    from ksmech.data.survey import spin_parity_kind

    man = pd.read_parquet(processed / "pmechdb" / "qm_manifest.parquet")
    failed = man[(man["status"] == "failed") & (man["reason"] == "SPIN_PARITY")]
    smiles = sorted(s for s in failed["smiles"].unique() if spin_parity_kind(s) in ("Cu", "Fe"))
    touched = arrow_atoms("pmechdb", set(smiles), processed)
    return [
        species_for(s, "metal", arrow_atoms=tuple(sorted(touched[s][0])), origin=touched[s][1])
        for s in smiles
    ]


def suspect_species(processed: Path = PROCESSED) -> list[Species]:
    """The METAL_SUSPECT fragments of either side of kept steps (both sources), at their
    production records (the pipeline at u = 0 where none exists)."""
    import pandas as pd

    from ksmech.data.io import read_steps, steps_path

    out = {}
    for src in ("pmechdb", "reactmech"):
        man = pd.read_parquet(processed / src / "qm_manifest.parquet").drop_duplicates("smiles")
        rows = man.set_index("smiles")
        for step in read_steps(steps_path(src, processed)):
            if step.domain_exclusion or "METAL_SUSPECT" not in (step.domain_class or ""):
                continue
            sides = step.reactants_smiles + "." + step.products_smiles
            for frag in chem.fragments(chem.parse_mapped(sides)):
                smi = chem.unmapped_canonical(frag)
                if domain.is_metal_suspect(frag) and smi not in out:
                    row = rows.loc[smi] if smi in rows.index else None
                    ok = row is not None and row["status"] == "ok"
                    origin = {"source": src, "manifest": None if row is None else row["status"]}
                    out[smi] = species_for(smi, "suspect", 0, origin=origin)
                    out[smi].record = row["cache_path"] if ok else None
    return [out[s] for s in sorted(out)]


def record_index_map(frag: Chem.Mol, smiles: str) -> dict[int, int] | None:
    """Atom map -> heavy-atom index of `prepare_mol(smiles)` for one mapped fragment, read
    through RDKit's SMILES output order; None when the fragment does not write as `smiles`."""
    heavy = Chem.RWMol(frag)
    for a in heavy.GetAtoms():
        a.SetIntProp("map", a.GetAtomMapNum())
        a.SetAtomMapNum(0)
    heavy = Chem.RemoveHs(heavy, sanitize=False)
    first = Chem.MolToSmiles(heavy)
    order1 = list(heavy.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
    again = Chem.MolFromSmiles(first)
    if again is None or Chem.MolToSmiles(again) != smiles:
        return None
    order2 = list(again.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
    return {heavy.GetAtomWithIdx(order1[k]).GetIntProp("map"): i for i, k in enumerate(order2)}


def arrow_atoms(source: str, smiles: set[str], processed: Path) -> dict[str, tuple[set, dict]]:
    """Per species: the record indices any dataset arrow touches on a reactant side, and the
    steps read (with how many could not be mapped)."""
    from ksmech.data.io import read_steps, steps_path

    out = {s: (set(), Counter()) for s in smiles}
    for step in read_steps(steps_path(source, processed)):
        maps = {m for arrow in step.arrows for m in arrow.atoms}
        for frag in chem.fragments(chem.parse_mapped(step.reactants_smiles)):
            smi = chem.unmapped_canonical(frag)
            if smi not in out:
                continue
            index = record_index_map(frag, smi)
            out[smi][1]["unmapped" if index is None else "steps"] += 1
            out[smi][0].update(index[m] for m in maps if index and m in index)
    return {s: (a, dict(c)) for s, (a, c) in out.items()}


# ---------------------------------------------------------------- QM


def calculator(z, xyz_bohr, charge: int, u: int, solvent: str | None, spin_polarized: bool):
    calc = new_calculator(z, xyz_bohr, charge, u, solvent)
    if spin_polarized:
        calc.add("spin-polarization", SP_WEIGHT)
    return calc


def single_point(z, xyz_bohr, charge, u, solvent, spin_polarized) -> dict:
    """Energy (Ha), orbital energies (eV), occupations and charges; for spGFN2 the orbital
    arrays have an alpha and a beta row."""
    t0 = time.perf_counter()
    try:
        res = calculator(z, xyz_bohr, charge, u, solvent, spin_polarized).singlepoint()
    except (TBLiteRuntimeError, TBLiteValueError) as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, str(exc)) from exc
    return {
        "energy_ha": float(res.get("energy")),
        "eps_ev": (np.asarray(res.get("orbital-energies")) * HA_EV).tolist(),
        "occ": np.asarray(res.get("orbital-occupations")).tolist(),
        "charges": np.asarray(res.get("charges")).tolist(),
        "seconds": time.perf_counter() - t0,
    }


def _lbfgs(calc, xyz_bohr: np.ndarray, walls, cfg: QMConfig) -> np.ndarray:
    """L-BFGS-B on one calculator's energies and gradients (`geometry.optimize_tblite`'s loop)."""

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        calc.update(positions=flat.reshape(-1, 3))
        res = calc.singlepoint()
        energy, grad = float(res.get("energy")), np.asarray(res.get("gradient"))
        if walls is not None:
            e_wall, g_wall = walls.energy_gradient(flat.reshape(-1, 3))
            energy, grad = energy + e_wall, grad + g_wall
        return energy, grad.ravel()

    opts = dict(maxiter=cfg.maxiter, maxfun=2 * cfg.maxiter, gtol=cfg.fmax, ftol=1e-12)
    try:
        out = minimize(objective, xyz_bohr.ravel(), jac=True, method="L-BFGS-B", options=opts)
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, f"during optimization: {exc}") from exc
    if np.abs(out.jac).max() > cfg.fmax:
        detail = f"max|grad|={np.abs(out.jac).max():.2e} after {out.nit} iterations"
        raise QMError(ReasonCode.OPT_NOT_CONVERGED, detail)
    return out.x.reshape(-1, 3)


def optimize(mol, xyz_bohr, charge, u, solvent, spin_polarized) -> tuple[np.ndarray, bool]:
    """Free optimization, then from the same start with bond walls when an input bond breaks
    (`GFN2Calculator._relax`'s rule). Returns (Bohr coordinates, walls needed)."""
    cfg = QMConfig(solvent=solvent)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    if len(z) == 1:
        return xyz_bohr.copy(), False
    calc = lambda: calculator(z, xyz_bohr, charge, u, solvent, spin_polarized)  # noqa: E731
    out = _lbfgs(calc(), xyz_bohr, None, cfg)
    try:
        check_connectivity(mol, out * BOHR_ANG, cfg.conn_cov_factor)
        return out, False
    except QMError:
        walls = bond_walls(mol, cfg.wall_factor, cfg.wall_k)
    out = _lbfgs(calc(), xyz_bohr, walls, cfg)
    check_connectivity(mol, out * BOHR_ANG, cfg.conn_cov_factor)
    return out, True


def formed_bonds(mol: Chem.Mol, xyz_ang: np.ndarray, factor: float = QMConfig.wall_factor) -> list:
    """Pairs of non-metal atoms, not bonded in the drawing, that end within `factor` x their
    covalent-radius sum (where the bond walls start): a bond the optimization formed, which
    `check_connectivity` ignores. Metal-ligand contacts are left out (the coordination sphere
    of a ligand-truncated fragment moves freely), as are Li, Na, Mg and K like there."""
    table, dist = Chem.GetPeriodicTable(), Chem.GetDistanceMatrix(mol)
    keep = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if not _tm(a) and a.GetAtomicNum() not in SKIP_BOND_ELEMENTS
    ]
    out = []
    for i, j in itertools.combinations(keep, 2):
        if dist[i, j] < 2:
            continue
        zi, zj = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        r = float(np.linalg.norm(xyz_ang[i] - xyz_ang[j])) / (
            table.GetRcovalent(zi) + table.GetRcovalent(zj)
        )
        if r < factor:
            out.append((i, j, round(r, 3)))
    return out


# ---------------------------------------------------------------- cache


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def cached(root: Path, smiles: str, method: str, compute) -> dict:
    """`compute()` once per (smiles, method) under `root`; a QMError is cached as a failure."""
    path = cache_path(root, smiles, 0, method).with_suffix(".json")
    failed = path.with_suffix(".failed.json")
    for p in (path, failed):
        if p.exists():
            return json.loads(p.read_text())
    try:
        out = {"status": "ok", "smiles": smiles, "method": method, **compute()}
        _write_json(path, out)
    except QMError as exc:
        out = {"status": "failed", "reason": str(exc.reason), "detail": exc.detail}
        _write_json(failed, out | {"smiles": smiles, "method": method})
    except Exception as exc:  # noqa: BLE001 - a probe run must survive anything, counted
        out = {"status": "failed", "reason": "UNEXPECTED", "detail": f"{type(exc).__name__}: {exc}"}
        _write_json(failed, out | {"smiles": smiles, "method": method})
    return out


def medium(solvent: str | None) -> str:
    return f"ALPB-{solvent}" if solvent else "gas"


def record_for(smiles: str, u: int, solvent: str | None, root: Path, given=None) -> QMRecord:
    """The restricted record of `run_at` at u: from the engine's embedding (the
    production pipeline), or at the `given` Angstrom geometry. Raises QMError."""
    tag = "given" if given is not None else "production"
    cfg = QMConfig(solvent=solvent, given_geometry=given is not None)
    cfg = replace(cfg, cache_dir=root / f"{tag}_{medium(solvent)}_u{u}")
    path = cache_path(cfg.cache_dir, smiles, 0, cfg.method)
    failed = path.with_suffix(".failed.json")
    if path.exists():
        return load_record(path)
    if failed.exists():
        note = json.loads(failed.read_text())
        raise QMError(ReasonCode(note["reason"]), note["detail"])
    mol = prepare_mol(smiles)
    try:
        _place(mol, given)
        rec = run_at(mol, smiles, u, cfg)
    except QMError as exc:
        _write_json(failed, {"reason": str(exc.reason), "detail": exc.detail, "smiles": smiles})
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    save_record(rec, path)
    return rec


def run_at(mol: Chem.Mol, smiles: str, u: int, cfg: QMConfig) -> QMRecord:
    """`GFN2Calculator.run` on conformer 0 with the unpaired count given rather than read from
    RDKit's radicals (which would change the SMILES the record is keyed by)."""
    from ksmech.qm import xtb_calc

    if (electron_count(mol) - u) % 2:
        raise QMError(ReasonCode.SPIN_PARITY, f"{electron_count(mol)} electrons, {u} unpaired")
    calc = GFN2Calculator(cfg)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    start, charge = mol.GetConformer(0).GetPositions() / BOHR_ANG, chem.formal_charge(mol)
    if cfg.given_geometry:
        check_connectivity(mol, start * BOHR_ANG, cfg.conn_cov_factor)
        xyz, restrained = start, False
    else:
        xyz, restrained = calc._relax(mol, z, start, charge, u)
    res = xtb_calc._single_point(z, xyz, charge, u, cfg.solvent, True, cfg.scf_maxiter)
    rec = calc._assemble(res, z, xyz * BOHR_ANG, smiles, 0, charge, u)
    if restrained:
        rec.add_warning(ReasonCode.RESTRAINED_GEOM)
    for note in mol.GetProp(NOTES_PROP).split(",") if mol.HasProp(NOTES_PROP) else []:
        rec.add_warning(ReasonCode(note))
    calc._add_fukui_fd(rec, z, xyz)
    calc._add_sterics(rec)
    return rec


def _place(mol: Chem.Mol, given) -> None:
    """Conformer 0: the engine's embedding, or the given coordinates."""
    if given is None:
        embed(mol, 1, seed=0)
        return
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(np.asarray(given)):
        conf.SetAtomPosition(i, p.tolist())
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)


def production(sp: Species, root: Path = PROBE_ROOT) -> QMRecord:
    """The species' production record: its own when one exists, else the pipeline at sp.u."""
    if sp.record:
        return load_record(Path(sp.record))
    return record_for(sp.smiles, sp.u, SOLVENT, root)


def theory_record(sp: Species, prod: QMRecord, root: Path = PROBE_ROOT) -> QMRecord:
    """Restricted, at the production geometry, with the lowest spin the electron count allows
    (spin_theory's input convention)."""
    u0 = electron_count(prepare_mol(sp.smiles)) % 2
    return record_for(sp.smiles, u0, SOLVENT, root, given=prod.coords_ang)


# ---------------------------------------------------------------- step 1 diagnostics


def aufbau(n_elec: float, u: int, n_mo: int) -> np.ndarray:
    alpha, beta = (round(n_elec) + u) // 2, (round(n_elec) - u) // 2
    idx = np.arange(n_mo)
    return (idx < alpha).astype(float) + (idx < beta)


def plateaus(occ: np.ndarray) -> list[list[int]]:
    """Consecutive MOs (energy order) with equal occupation."""
    groups = [[0]]
    for i in range(1, len(occ)):
        same = abs(occ[i] - occ[groups[-1][-1]]) < OCC_TOL
        groups[-1].append(i) if same else groups.append([i])
    return groups


def occupation_class(occ: np.ndarray) -> str:
    """integral (every MO within OPEN_TOL of an integer); degenerate (every MO further than that
    in an equal-occupation plateau of two or more); smeared otherwise."""
    frac = np.abs(occ - np.round(occ)) > OPEN_TOL
    if not frac.any():
        return "integral"
    lone = [g for g in plateaus(occ) if len(g) == 1 and frac[g[0]]]
    return "smeared" if lone else "degenerate"


def open_blocks(occ: np.ndarray) -> dict:
    """The open MOs (0 < n < 2) and the block today's 0.5 rule sends each to; a plateau split
    across the two blocks is counted."""
    open_ = [i for i in range(len(occ)) if OPEN_TOL < occ[i] < 2 - OPEN_TOL]
    block = {i: "occupied" if occ[i] >= fukui.OCC_THRESHOLD else "virtual" for i in open_}
    split = sum(1 for g in plateaus(occ) if len({block.get(i) for i in g} - {None}) > 1)
    return {"open": [[i, float(occ[i]), block[i]] for i in open_], "plateaus_split": split}


def sp_frontier(point: dict) -> dict:
    """alpha HOMO (the highest alpha level holding charge) and beta LUMO (the lowest beta level
    not full: Br.'s three beta p levels at 2/3 each) of a spGFN2 point, eV."""
    eps, occ = np.asarray(point["eps_ev"]), np.asarray(point["occ"])
    a_occ, b_virt = eps[0][occ[0] > OPEN_TOL], eps[1][occ[1] < 1 - OPEN_TOL]
    homo = float(a_occ.max()) if len(a_occ) else None
    lumo = float(b_virt.min()) if len(b_virt) else None
    return {"alpha_homo": homo, "beta_lumo": lumo}


def charge_shifts(rec: QMRecord, sp_point: dict, arrows: tuple[int, ...]) -> dict:
    dq = np.abs(np.asarray(sp_point["charges"]) - rec.charges)
    metal = np.isin(rec.atomic_numbers, list(domain.METALS))
    pick = [i for i in arrows if not metal[i]]
    return {
        "max_dq": float(dq.max()),
        "max_dq_nonmetal": float(dq[~metal].max()) if (~metal).any() else None,
        "max_dq_arrow_nonmetal": float(dq[pick].max()) if pick else None,
        "arrow_nonmetal_atoms": pick,
    }


def diagnostics(sp: Species, rec: QMRecord, sp_point: dict) -> dict:
    occ = np.asarray(rec.occupations)
    ref = aufbau(occ.sum(), rec.unpaired, len(occ))
    open_ = open_blocks(occ)
    eps = np.asarray(rec.orbital_energies_ev)
    somo = [eps[i] for i, _, _ in open_["open"]]
    front = sp_frontier(sp_point)
    mid = None
    if front["alpha_homo"] is not None and front["beta_lumo"] is not None:
        mid = (front["alpha_homo"] + front["beta_lumo"]) / 2
    return {
        "occupation_class": occupation_class(occ),
        "max_dev_aufbau": float(np.abs(occ - ref).max()),
        "open_mos": open_,
        "eps_somo_ev": float(np.mean(somo)) if somo else None,
        "sp_frontier": front,
        "sp_midpoint_ev": mid,
        "energy_restricted_ha": rec.energy_ha,
        "energy_sp_ha": sp_point["energy_ha"],
        "dE_sp_restricted_ev": (sp_point["energy_ha"] - rec.energy_ha) * HA_EV,
        **charge_shifts(rec, sp_point, sp.arrow_atoms),
    }


def vertical(rec: QMRecord, sp: Species, root: Path, tag: str = "prod") -> dict:
    """spGFN2 single points of every allowed state at the record's geometry."""
    z, xyz = rec.atomic_numbers, rec.coords_ang / BOHR_ANG
    out = {}
    for u in sp.states:
        method = f"GFN2-xTB/{medium(SOLVENT)}/sp/u{u}/{tag}-u{rec.unpaired}"
        res = cached(
            root / "vertical",
            sp.smiles,
            method,
            lambda u=u: single_point(z, xyz, rec.charge, u, SOLVENT, True),
        )
        out[u] = {k: res.get(k) for k in ("status", "reason", "energy_ha")}
    return out


def probe_species(sp: Species, root: Path = PROBE_ROOT) -> dict:
    """Step 1 for one species: production and theory records, the spGFN2 point at sp.u, the
    diagnostics and the vertical states. Never raises: a failure is returned with its reason."""
    row = {"smiles": sp.smiles, "groups": list(sp.groups), "u": sp.u, "states": list(sp.states)}
    row["origin"] = sp.origin
    t0 = time.perf_counter()
    try:
        rec = production(sp, root)
        theory_record(sp, rec, root)
        z, xyz = rec.atomic_numbers, rec.coords_ang / BOHR_ANG
        method = f"GFN2-xTB/{medium(SOLVENT)}/sp/u{sp.u}/prod"
        point = cached(
            root / "sp_single",
            sp.smiles,
            method,
            lambda: single_point(z, xyz, rec.charge, sp.u, SOLVENT, True),
        )
        if point["status"] != "ok":
            return row | {"status": "failed", "stage": "sp_single", "reason": point["reason"]}
        row |= {"status": "ok", "n_atoms": len(z), "warnings": list(rec.warnings)}
        row |= {"diagnostics": diagnostics(sp, rec, point), "vertical": vertical(rec, sp, root)}
        row["sp_seconds"] = point["seconds"]
    except QMError as exc:
        row |= {"status": "failed", "stage": "record", "reason": str(exc.reason)}
        row["detail"] = exc.detail
    except Exception as exc:  # noqa: BLE001 - counted, never fatal to the run
        row |= {"status": "failed", "stage": "record", "reason": "UNEXPECTED"}
        row["detail"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - t0
    return row


# ---------------------------------------------------------------- step 5


def ion_energy(z, xyz, charge, u_neutral, solvent, spin_polarized, root, smiles) -> dict:
    """The lower of the ion's two lowest spin states (u_neutral - 1 and + 1, at least 0)."""
    best = None
    for u in sorted({abs(u_neutral - 1), u_neutral + 1}):
        method = f"GFN2-xTB/{medium(solvent)}/{'sp' if spin_polarized else 'restricted'}/u{u}"
        method += f"/q{charge}/vertical"
        res = cached(
            root / "ipea",
            smiles,
            method,
            lambda u=u: single_point(z, xyz, charge, u, solvent, spin_polarized),
        )
        if res["status"] == "ok" and (best is None or res["energy_ha"] < best["energy_ha"]):
            best = {"u": u, "energy_ha": res["energy_ha"]}
    return best or {"u": None, "energy_ha": None}


def ipea_species(name: str, root: Path = PROBE_ROOT) -> dict:
    """Vertical gas-phase IE and EA (eV) of one radical, restricted and spGFN2."""
    smi = chem.unmapped_canonical(Chem.MolFromSmiles(IPEA_RADICALS[name]))
    u = table_u(prepare_mol(smi))
    row = {"name": name, "smiles": smi, "u": u}
    try:
        rec = record_for(smi, u, None, root)
    except QMError as exc:
        return row | {"status": "failed", "reason": str(exc.reason)}
    z, xyz = rec.atomic_numbers, rec.coords_ang / BOHR_ANG
    for label, sp_flag in (("restricted", False), ("sp", True)):
        method = f"GFN2-xTB/gas/{label}/u{u}/q{rec.charge}/vertical"
        neutral = cached(
            root / "ipea",
            smi,
            method,
            lambda f=sp_flag: single_point(z, xyz, rec.charge, u, None, f),
        )
        cation = ion_energy(z, xyz, rec.charge + 1, u, None, sp_flag, root, smi)
        anion = ion_energy(z, xyz, rec.charge - 1, u, None, sp_flag, root, smi)
        e0 = neutral.get("energy_ha")
        ie = None if None in (e0, cation["energy_ha"]) else (cation["energy_ha"] - e0) * HA_EV
        ea = None if None in (e0, anion["energy_ha"]) else (e0 - anion["energy_ha"]) * HA_EV
        row[label] = {"IE_ev": ie, "EA_ev": ea, "cation_u": cation["u"], "anion_u": anion["u"]}
    return row | {"status": "ok"}


# ---------------------------------------------------------------- report


def merge_species(groups: list[list[Species]]) -> list[Species]:
    """One entry per SMILES; groups and origins merged, the first entry's record and u kept."""
    out: dict[str, Species] = {}
    for sp in itertools.chain(*groups):
        if sp.smiles in out:
            have = out[sp.smiles]
            have.groups = tuple(dict.fromkeys(have.groups + sp.groups))
            have.origin = {**sp.origin, **have.origin}
            have.arrow_atoms = tuple(sorted(set(have.arrow_atoms) | set(sp.arrow_atoms)))
        else:
            out[sp.smiles] = sp
    return list(out.values())


def ipea_comparison(ipea: list[dict]) -> dict:
    """The delta-SCF levels against the literature, per level of theory and quantity: the mean
    signed error, its SD, the MAD once the mean shift is removed and the Spearman rank
    correlation (eV)."""
    from scipy.stats import spearmanr

    out = {}
    for label in ("restricted", "sp"):
        for qty, col in (("IE_ev", 0), ("EA_ev", 2)):
            pairs = [
                (r[label][qty], IPEA_LITERATURE[r["name"]][col])
                for r in ipea
                if r["status"] == "ok" and r[label][qty] is not None
            ]
            calc, lit = (np.array(v) for v in zip(*pairs, strict=True))
            err = calc - lit
            out[f"{label}_{qty}"] = {
                "n": len(err),
                "mean_signed_error": float(err.mean()),
                "sd": float(err.std(ddof=1)),
                "mad_shift_removed": float(np.abs(err - err.mean()).mean()),
                "spearman": float(spearmanr(calc, lit).statistic),
            }
    return out


def summary(rows: list[dict]) -> dict:
    """Per group: failures by reason, occupation classes, the max|n - aufbau| distribution, open
    MOs per block, split plateaus, |dq| and eps_SOMO - midpoint distributions."""
    out = {}
    for group in sorted({g for r in rows for g in r["groups"]}):
        sel = [r for r in rows if group in r["groups"]]
        ok = [r["diagnostics"] for r in sel if r["status"] == "ok"]
        blocks = Counter(b for d in ok for _, _, b in d["open_mos"]["open"])
        gap = [
            d["eps_somo_ev"] - d["sp_midpoint_ev"]
            for d in ok
            if d["eps_somo_ev"] is not None and d["sp_midpoint_ev"] is not None
        ]
        out[group] = {
            "n": len(sel),
            "failed": dict(Counter(r["reason"] for r in sel if r["status"] != "ok")),
            "occupation_class": dict(Counter(d["occupation_class"] for d in ok)),
            "max_dev_aufbau": _quantiles([d["max_dev_aufbau"] for d in ok]),
            "open_mo_blocks": dict(blocks),
            "plateaus_split": sum(d["open_mos"]["plateaus_split"] for d in ok),
            "max_dq": _quantiles([d["max_dq"] for d in ok]),
            "max_dq_arrow_nonmetal": _quantiles(
                [d["max_dq_arrow_nonmetal"] for d in ok if d["max_dq_arrow_nonmetal"] is not None]
            ),
            "eps_somo_minus_midpoint_ev": _quantiles(gap),
            "seconds": _quantiles([r["seconds"] for r in sel]),
        }
    return out


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    q = np.quantile(values, [0, 0.5, 0.9, 1])
    return {"n": len(values), "min": q[0], "median": q[1], "p90": q[2], "max": q[3]}


def ch3_known(root: Path = PROBE_ROOT) -> dict:
    """[CH3] at the engine's embedding (ETKDG + MMFF, no GFN2 optimization), where the values
    the plan quotes were measured: restricted SOMO and the spGFN2 alpha HOMO / beta LUMO (eV)."""
    mol = prepare_mol("[CH3]")
    embed(mol, 1, seed=0)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    xyz = mol.GetConformer(0).GetPositions() / BOHR_ANG
    points = {}
    for label, flag in (("restricted", False), ("sp", True)):
        method = f"GFN2-xTB/{medium(SOLVENT)}/{label}/u1/embedding"
        points[label] = cached(root / "checks", "[CH3]", method,
                               lambda f=flag: single_point(z, xyz, 0, 1, SOLVENT, f))  # fmt: skip
    eps, occ = np.asarray(points["restricted"]["eps_ev"]), np.asarray(points["restricted"]["occ"])
    return {"somo": float(eps[np.argmin(np.abs(occ - 1))]), **sp_frontier(points["sp"])}


def checks(rows: list[dict]) -> dict:
    """The plan's reproduction checks: [CH3] restricted SOMO -8.911 eV, spGFN2 alpha HOMO -9.687
    and beta LUMO -8.150 eV (to 0.0005 eV, at the embedding); ethane spGFN2 dE = 0.00."""
    known = {"somo": -8.911, "alpha_homo": -9.687, "beta_lumo": -8.150}
    out = {"CH3": ch3_known()}
    out["CH3_ok"] = all(abs(out["CH3"][k] - v) < 0.0005 for k, v in known.items())
    ethane = next((r for r in rows if r["smiles"] == "CC" and r["status"] == "ok"), None)
    if ethane:
        out["ethane_dE_ev"] = ethane["diagnostics"]["dE_sp_restricted_ev"]
        out["ethane_ok"] = abs(out["ethane_dE_ev"]) < 0.005
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=PROBE_ROOT / "probe_report.json")
    parser.add_argument("--groups", nargs="*", default=["fixture", "metal", "suspect"])
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    pick = {"fixture": fixture_species, "metal": metal_species, "suspect": suspect_species}
    species = merge_species([pick[g]() for g in args.groups])
    species.sort(key=lambda s: -prepare_mol(s.smiles).GetNumAtoms())  # longest first
    rows = []
    for k, row in enumerate(in_processes(probe_species, species, args.workers), 1):
        rows.append(row)
        print(f"step {k} / {len(species)} {row['status']} {row['smiles']}", flush=True)
    ipea = list(in_processes(ipea_species, list(IPEA_RADICALS), args.workers))
    report = {"summary": summary(rows), "checks": checks(rows), "ipea": ipea, "rows": rows}
    report["ipea_vs_literature"] = ipea_comparison(ipea)
    _write_json(args.out, report)
    print(json.dumps({"summary": report["summary"], "checks": report["checks"]}, default=str))
    by_reason = Counter(r.get("reason") for r in rows if r["status"] != "ok")
    print(f"species {len(rows)}, failed {sum(by_reason.values())}: {dict(by_reason)}")


if __name__ == "__main__":
    main()
