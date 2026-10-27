"""GFN2-xTB of a transition-metal complex from the coordination template (workbench adapter).

`process_metal_molecule` is the metal counterpart of `qm.batch.process_molecule` for one
conformer: the start geometry comes from `serve.chem.metal_geometry` (the standard ETKDG + MMFF/UFF
embed only when no template applies, with the warning METAL_TEMPLATE_SKIPPED), and
`MetalGFN2Calculator` relaxes it with a metal-aware connectivity check: a metal-ligand bond is
broken beyond `ml_cov_factor` (1.45) x the covalent radii sum, every other bond beyond the
standard 1.3. Only if a bond still breaks is the optimization redone with walls, at 1.35x on
metal-ligand bonds and the standard 1.2x elsewhere (RESTRAINED_GEOM), so GFN2 relaxes the
polyhedron freely. `qm/` is not modified: the record's method is the standard one plus
`+metal-geom-v1` (`MetalQMConfig.method`), a distinct cache key and provenance config.

Record warnings added here (strings, comma-separated like the `ReasonCode` ones):
METAL_TEMPLATE_SKIPPED, METAL_POLYHEDRON_CHANGED, METAL_ISOMER_CHANGED, FERMI_SMEARED (an MO
occupation more than 0.05 from 0 or 2: a near-degenerate, possibly open-shell ground state).
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from ksmech.qm import batch
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import (
    BOHR_ANG,
    SKIP_BOND_ELEMENTS,
    WallRestraint,
    optimize_tblite,
)
from ksmech.qm.records import QMError, ReasonCode, cache_path, save_record
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig
from ksmech.serve.chem import metal_geometry as mg

log = logging.getLogger(__name__)

FERMI_TOL = 0.05


@dataclass(frozen=True)
class MetalQMConfig(QMConfig):
    ml_cov_factor: float = mg.ML_COV_FACTOR
    ml_wall_factor: float = mg.ML_WALL_FACTOR

    @property
    def method(self) -> str:
        return mg.metal_method(QMConfig.method.fget(self))


def _limits(mol: Chem.Mol, organic: float, metal: float):
    """(i, j, d0) per input bond: d0 = factor x r_cov sum, `metal` for metal-ligand bonds."""
    table = Chem.GetPeriodicTable()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() in SKIP_BOND_ELEMENTS or b.GetAtomicNum() in SKIP_BOND_ELEMENTS:
            continue
        i, j = a.GetIdx(), b.GetIdx()
        factor = metal if mg.is_metal_bond(mol, i, j) else organic
        yield (
            i,
            j,
            factor * (table.GetRcovalent(a.GetAtomicNum()) + table.GetRcovalent(b.GetAtomicNum())),
        )


def broken_bonds(mol: Chem.Mol, xyz_ang: np.ndarray, organic: float, metal: float):
    """[(i, j, d / limit)] of every input bond longer than its limit (`_limits`)."""
    out = []
    for i, j, d0 in _limits(mol, organic, metal):
        d = float(np.linalg.norm(xyz_ang[i] - xyz_ang[j]))
        if d > d0:
            out.append((i, j, d / d0))
    return out


def check_metal_connectivity(mol, xyz_ang, organic: float, metal: float) -> None:
    """CONNECTIVITY_CHANGED when a bond is broken by the metal-aware limits."""
    broken = broken_bonds(mol, xyz_ang, organic, metal)
    if broken:
        shown = ", ".join(f"({i},{j}) {r:.2f}x limit" for i, j, r in broken[:6])
        msg = f"{len(broken)} bond(s) broken (M-L {metal}x, other {organic}x): {shown}"
        raise QMError(ReasonCode.CONNECTIVITY_CHANGED, msg)


def metal_bond_walls(mol: Chem.Mol, organic: float, metal: float, k: float) -> WallRestraint:
    rows = list(_limits(mol, organic, metal))
    pairs = np.array([(i, j) for i, j, _ in rows], dtype=int).reshape(-1, 2)
    return WallRestraint(pairs, np.array([d0 for *_, d0 in rows]) / BOHR_ANG, k)


class MetalGFN2Calculator(GFN2Calculator):
    """`GFN2Calculator` with the metal-aware check and walls (`run` itself is inherited)."""

    config: MetalQMConfig

    def _relax(self, mol, z, start_bohr, charge, unpaired) -> tuple[np.ndarray, bool]:
        cfg = self.config
        xyz = self._optimize(z, start_bohr, charge, unpaired)
        try:
            check_metal_connectivity(mol, xyz * BOHR_ANG, cfg.conn_cov_factor, cfg.ml_cov_factor)
            return xyz, False
        except QMError as exc:
            if not (cfg.restrain and cfg.optimizer == "tblite"):
                raise
            log.info("free optimization broke a bond (%s); retrying with walls", exc.detail)
        walls = metal_bond_walls(mol, cfg.wall_factor, cfg.ml_wall_factor, cfg.wall_k)
        xyz = optimize_tblite(
            z, start_bohr, charge, unpaired, cfg.fmax, cfg.maxiter, cfg.solvent, walls,
            scf_maxiter=cfg.scf_maxiter,
        )  # fmt: skip
        check_metal_connectivity(mol, xyz * BOHR_ANG, cfg.conn_cov_factor, cfg.ml_cov_factor)
        return xyz, True


def start_geometry(mol: Chem.Mol) -> tuple[list[mg.CentrePlan], list[str]]:
    """Put the template start geometry on `mol` as conformer 0. Returns (plans, warnings);
    without a template the standard embed is used and the plans are empty."""
    try:
        xyz, plans = mg.build_start_geometry(mol)
    except mg.MetalTemplateError as exc:
        log.info("no metal template (%s): standard embedding", exc)
        embed(mol, 1, seed=0)
        return [], ["METAL_TEMPLATE_SKIPPED"]
    mg.set_conformer(mol, xyz)
    return plans, []


def _final_warnings(mol, plans, rec) -> list[str]:
    out = [w for p in plans for w in mg.check_geometry(mol, p, rec.coords_ang)]
    occ = np.asarray(rec.occupations, dtype=float)
    if np.any((occ > FERMI_TOL) & (occ < 2.0 - FERMI_TOL)):
        out.append("FERMI_SMEARED")
    return out


def _compute(key: str, config: MetalQMConfig) -> dict:
    mol = prepare_mol(key)
    plans, notes = start_geometry(mol)
    t0 = time.perf_counter()
    out = MetalGFN2Calculator(config).try_run(mol, 0)
    seconds = time.perf_counter() - t0
    if out.record is None:
        return batch._row(key, 0, reason=str(out.reason), detail=out.detail, seconds=seconds)
    rec = out.record
    for w in [*notes, *_final_warnings(mol, plans, rec)]:
        rec.add_warning(w)  # type: ignore[arg-type] - metal warnings are plain strings
    path = cache_path(config.cache_dir, key, 0, config.method)
    save_record(rec, path)
    detail = "; ".join(p.describe() for p in plans)
    return batch._row(
        key, 0, cache_path=str(path), status="ok", warnings=rec.warnings, seconds=seconds,
        n_embedded=1, method=config.method, detail=detail,
    )  # fmt: skip


def process_metal_molecule(key: str, config: MetalQMConfig) -> dict:
    """One manifest-style row (`qm.batch._row`) for `key` on the metal path. Never raises:
    the SIGALRM limit is `config.timeout_s` (the handler is `qm.batch._init_worker`'s)."""
    t0 = time.perf_counter()
    signal.alarm(config.timeout_s)
    try:
        return _compute(key, config)
    except batch._Timeout:
        reason, detail = ReasonCode.TIMEOUT, f"> {config.timeout_s}s"
    except QMError as exc:
        reason, detail = exc.reason, exc.detail
    except Exception as exc:  # noqa: BLE001 - reported with a reason code, never raised
        log.exception("unexpected error for %s", key)
        reason, detail = ReasonCode.UNEXPECTED, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    return batch._row(key, 0, reason=str(reason), detail=detail, seconds=time.perf_counter() - t0)
