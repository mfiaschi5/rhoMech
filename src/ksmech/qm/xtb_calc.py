"""`GFN2Calculator`: optimize, single point, descriptors and sterics for one conformer.

With `QMConfig.given_geometry` the conformer's geometry is taken as it is (a dataset's own
GFN2-xTB optimization, e.g. QMugs): no optimization, only a check that every bond of the Lewis
structure is still a bond there, then the same single point, N+-1 Fukui and sterics. Its method
string ends in `given`, so those records never share a cache key with optimized ones.

Charge and multiplicity come from RDKit: total charge = sum of formal charges, unpaired electrons
= sum of radical electrons. tblite runs open-shell systems restricted with fractional occupations
(no spin polarization), so the SOMO of a doublet has occupation 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from tblite.interface import TBLiteRuntimeError, TBLiteValueError

from ksmech.data import chem
from ksmech.qm import fukui
from ksmech.qm.ao import ao_layout
from ksmech.qm.embed import NOTES_PROP, prepare_mol
from ksmech.qm.geometry import (
    BOHR_ANG,
    bond_walls,
    check_connectivity,
    new_calculator,
    optimize_tblite,
    optimize_xtb,
)
from ksmech.qm.records import DEFAULT_CACHE_DIR, QMError, QMRecord, ReasonCode
from ksmech.qm.sterics import buried_volume_per_atom

log = logging.getLogger(__name__)
_OPTIMIZERS = ("tblite", "xtb")


@dataclass(frozen=True)
class QMConfig:
    optimizer: str = "tblite"  # "tblite": scipy L-BFGS on tblite; "xtb": the xtb binary
    solvent: str | None = "water"  # ALPB implicit solvent for every calculation; None = gas phase
    fmax: float = 3e-4  # max |gradient| in Hartree/Bohr
    maxiter: int = 500  # optimizer iterations
    scf_maxiter: int = 250  # SCF cycles per energy (tblite's default); not part of `method`
    timeout_s: int = 900  # per molecule, enforced by the batch runner (103 atoms take ~300 s)
    conn_cov_factor: float = 1.3  # an input bond stretched beyond this x r_cov sum = broken
    restrain: bool = True  # retry a dissociating species with bond walls (tblite backend only)
    wall_factor: float = 1.2  # walls start at this x r_cov sum
    wall_k: float = 1.0  # Hartree / Bohr^2
    cache_dir: Path = DEFAULT_CACHE_DIR
    given_geometry: bool = False  # single points at the conformer's own geometry, no optimizer

    def __post_init__(self) -> None:
        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {self.optimizer!r}")

    @property
    def method(self) -> str:
        """Part of the cache key: results from different backends must not collide."""
        medium = f"ALPB-{self.solvent}" if self.solvent else "gas"
        opt = "tblite-lbfgs" if self.optimizer == "tblite" else "xtb-opt"
        return f"GFN2-xTB/{medium}/{'given' if self.given_geometry else opt}"


@dataclass
class CalcOutcome:
    record: QMRecord | None
    reason: ReasonCode | None = None
    detail: str = ""


def resolve_charge_spin(mol: Chem.Mol) -> tuple[int, int]:
    """(total charge, unpaired electrons) from RDKit. Raises SPIN_PARITY on an impossible pair."""
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    n_elec = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if n_elec < unpaired or (n_elec - unpaired) % 2:
        raise QMError(ReasonCode.SPIN_PARITY, f"{n_elec} electrons cannot have {unpaired} unpaired")
    return charge, unpaired


def _check_atom_order(mol: Chem.Mol, smiles: str) -> None:
    """The mol must come from `prepare_mol(smiles)`: same elements and bonds in the same order."""
    ref = prepare_mol(smiles)
    same_z = [a.GetAtomicNum() for a in ref.GetAtoms()] == [
        a.GetAtomicNum() for a in mol.GetAtoms()
    ]
    edges = lambda m: sorted(  # noqa: E731
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in m.GetBonds()
    )
    if not (same_z and edges(ref) == edges(mol)):
        raise ValueError(f"mol atom order differs from prepare_mol({smiles!r}); build it with that")


def _single_point(z, xyz_bohr, charge, unpaired, solvent, integrals, scf_maxiter=250):
    calc = new_calculator(
        z, xyz_bohr, charge, unpaired, solvent, save_integrals=integrals, scf_maxiter=scf_maxiter
    )
    try:
        return calc.singlepoint()
    except TBLiteRuntimeError as exc:
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, str(exc)) from exc


def _fukui_fd(z, xyz_bohr, charge, unpaired, q_n, solvent, scf_maxiter=250) -> fukui.FukuiFD | None:
    """Finite-difference Fukui from the N+1 and N-1 systems; None if either SCF fails."""
    u_ion = fukui.unpaired_for_ionized(unpaired)
    try:
        plus = _single_point(z, xyz_bohr, charge - 1, u_ion, solvent, False, scf_maxiter)
        minus = _single_point(z, xyz_bohr, charge + 1, u_ion, solvent, False, scf_maxiter)
        q_plus, q_minus = plus.get("charges"), minus.get("charges")
    except (QMError, TBLiteValueError, TBLiteRuntimeError) as exc:
        log.info("Fukui N+-1 failed (%s)", exc)
        return None
    return fukui.finite_difference(q_n, np.asarray(q_plus), np.asarray(q_minus))


class GFN2Calculator:
    def __init__(self, config: QMConfig | None = None) -> None:
        self.config = config or QMConfig()

    def _optimize(self, z, xyz_bohr, charge, unpaired) -> np.ndarray:
        cfg = self.config
        if cfg.optimizer == "xtb":
            return optimize_xtb(z, xyz_bohr, charge, unpaired, cfg.solvent)
        return optimize_tblite(
            z,
            xyz_bohr,
            charge,
            unpaired,
            cfg.fmax,
            cfg.maxiter,
            cfg.solvent,
            scf_maxiter=cfg.scf_maxiter,
        )

    def _relax(self, mol, z, start_bohr, charge, unpaired) -> tuple[np.ndarray, bool]:
        """Free optimization; if an input bond breaks, redo it from the start geometry with walls.

        Returns (Bohr coordinates, whether the walls were needed). Raises CONNECTIVITY_CHANGED if
        a bond still breaks (or restraining is unavailable or switched off).
        """
        cfg = self.config
        xyz_bohr = self._optimize(z, start_bohr, charge, unpaired)
        try:
            check_connectivity(mol, xyz_bohr * BOHR_ANG, cfg.conn_cov_factor)
            return xyz_bohr, False
        except QMError as exc:
            if not (cfg.restrain and cfg.optimizer == "tblite"):
                raise
            log.info("free optimization broke a bond (%s); retrying with walls", exc.detail)
        walls = bond_walls(mol, cfg.wall_factor, cfg.wall_k)
        xyz_bohr = optimize_tblite(
            z,
            start_bohr,
            charge,
            unpaired,
            cfg.fmax,
            cfg.maxiter,
            cfg.solvent,
            walls,
            scf_maxiter=cfg.scf_maxiter,
        )
        check_connectivity(mol, xyz_bohr * BOHR_ANG, cfg.conn_cov_factor)
        return xyz_bohr, True

    def run(self, mol: Chem.Mol, conf_id: int = 0) -> QMRecord:
        """Full pipeline for one conformer. Raises QMError(reason) on failure."""
        smiles = chem.unmapped_canonical(mol)
        _check_atom_order(mol, smiles)
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
        start = mol.GetConformer(conf_id).GetPositions() / BOHR_ANG
        charge, unpaired = resolve_charge_spin(mol)
        if self.config.given_geometry:
            check_connectivity(mol, start * BOHR_ANG, self.config.conn_cov_factor)
            xyz_bohr, restrained = start, False
        else:
            xyz_bohr, restrained = self._relax(mol, z, start, charge, unpaired)
        xyz_ang = xyz_bohr * BOHR_ANG
        res = _single_point(
            z, xyz_bohr, charge, unpaired, self.config.solvent, True, self.config.scf_maxiter
        )
        rec = self._assemble(res, z, xyz_ang, smiles, conf_id, charge, unpaired)
        if restrained:
            rec.add_warning(ReasonCode.RESTRAINED_GEOM)
        for note in mol.GetProp(NOTES_PROP).split(",") if mol.HasProp(NOTES_PROP) else []:
            rec.add_warning(ReasonCode(note))
        self._add_fukui_fd(rec, z, xyz_bohr)
        self._add_sterics(rec)
        return rec

    def _assemble(self, res, z, xyz_ang, smiles, conf_id, charge, unpaired) -> QMRecord:
        eps_ha = np.asarray(res.get("orbital-energies"))
        C, S = np.asarray(res.get("orbital-coefficients")), np.asarray(res.get("overlap-matrix"))
        occ = np.asarray(res.get("orbital-occupations"))
        ao_atom, ao_l = ao_layout(z)
        if len(ao_atom) != S.shape[0]:
            raise QMError(ReasonCode.UNEXPECTED, f"AO map {len(ao_atom)} != {S.shape[0]}")
        homo = fukui.homo_index(occ)
        eps_ev = eps_ha * fukui.HARTREE_TO_EV
        f_plus, f_minus = fukui.frozen_orbital_fukui(C, S, ao_atom, len(z), homo)
        nan = np.full(len(z), np.nan)
        rec = QMRecord(
            smiles=smiles,
            atom_order_smiles=smiles,
            conformer_id=conf_id,
            method=self.config.method,
            charge=charge,
            unpaired=unpaired,
            atomic_numbers=z,
            coords_ang=xyz_ang,
            energy_ha=float(res.get("energy")),
            orbital_energies_ev=eps_ev,
            occupations=occ,
            C=C,
            S=S,
            H_core=np.asarray(res.get("hamiltonian-matrix")),
            fock=fukui.fock_from_mo(S, C, eps_ha),
            charges=np.asarray(res.get("charges")),
            ao_atom=ao_atom,
            ao_l=ao_l,
            fukui_plus=nan.copy(),
            fukui_minus=nan.copy(),
            fukui_zero=nan.copy(),
            fukui_dual=nan.copy(),
            fukui_fmo_plus=f_plus,
            fukui_fmo_minus=f_minus,
            buried_volume=nan.copy(),
            **fukui.global_descriptors(eps_ev, homo),
        )
        if homo is None:
            rec.add_warning(ReasonCode.NO_FRONTIER)
        return rec

    def _add_fukui_fd(self, rec: QMRecord, z, xyz_bohr) -> None:
        if ReasonCode.NO_FRONTIER in rec.warnings:
            return
        cfg = self.config
        fd = _fukui_fd(
            z, xyz_bohr, rec.charge, rec.unpaired, rec.charges, cfg.solvent, cfg.scf_maxiter
        )
        if fd is None:
            rec.add_warning(ReasonCode.FUKUI_SCF_FAIL)
            return
        rec.fukui_plus, rec.fukui_minus = fd.plus, fd.minus
        rec.fukui_zero, rec.fukui_dual = fd.zero, fd.dual

    def _add_sterics(self, rec: QMRecord) -> None:
        try:
            rec.buried_volume = buried_volume_per_atom(rec.atomic_numbers, rec.coords_ang)
        except Exception as exc:  # noqa: BLE001 - morfeus errors are not a molecule failure
            log.info("buried volume failed for %s: %s", rec.smiles, exc)
            rec.add_warning(ReasonCode.VBUR_FAIL)

    def try_run(self, mol: Chem.Mol, conf_id: int = 0) -> CalcOutcome:
        """`run` that never raises: failures come back as a reason code and are logged."""
        try:
            return CalcOutcome(self.run(mol, conf_id))
        except QMError as exc:
            log.warning("QM failed: %s", exc)
            return CalcOutcome(None, exc.reason, exc.detail)
        except Exception as exc:  # noqa: BLE001 - the batch runner must survive anything
            log.exception("unexpected QM error")
            return CalcOutcome(None, ReasonCode.UNEXPECTED, f"{type(exc).__name__}: {exc}")
