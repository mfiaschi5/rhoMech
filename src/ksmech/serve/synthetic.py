"""Synthetic chemistry: charge-iterated extended Hueckel on the GFN2 basis -> a `QMRecord`
(adapter A6), so tests, the e2e run and UI development exercise the real token, physics and
arrows code with no SCF code and no xTB.

The basis is tblite's GFN2 valence basis (shell list exported by tblite's parameter table,
`qm.ao.element_ao_l`; no Hamiltonian is ever built by tblite), the overlap matrix is the STO-nG
one the oriented kernel uses (`physics.gto.overlap_matrix`), and the Hamiltonian is
Wolfsberg-Helmholz with valence-state ionization energies shifted by the atomic charges:

    h_mu  = h0[Z_A][l_mu] - OMEGA_EV * q_A        (eV)
    H_mn  = K_EHT * S_mn * (h_mu + h_nu) / 2,  H_mm = h_mu
    H C = S C eps (scipy.linalg.eigh), aufbau occupations, Mulliken charges, damped iteration.

Deterministic: ETKDG seed 0 with one thread, LAPACK on one thread, PM from the canonical MOs.
Every record is labelled: method `SYNTHETIC-EHT/v1` (part of every cache key, the oriented
keys included) and warning `SYNTHETIC`. Synthetic physics is a test vehicle, never a chemistry
claim.

The API process imports only the constants of this module; `eht_record` runs in the worker.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

SYNTHETIC_METHOD = "SYNTHETIC-EHT/v1"
K_EHT = 1.75
OMEGA_EV = 2.0
D_LEVEL_DEFAULT_EV = -2.0
MAX_ROUNDS = 50
CHARGE_TOL = 1e-6
NOT_CONVERGED = "SYNTHETIC_SCC_NOT_CONVERGED"
# valence-state ionization energies (eV) per atomic number and angular momentum
EHT_LEVELS: dict[int, dict[int, float]] = {
    1: {0: -13.60},
    3: {0: -5.40, 1: -3.50},
    5: {0: -15.20, 1: -8.50},
    6: {0: -21.40, 1: -11.40},
    7: {0: -26.00, 1: -13.40},
    8: {0: -32.30, 1: -14.80},
    9: {0: -40.00, 1: -18.10},
    11: {0: -5.10, 1: -3.00},
    12: {0: -9.00, 1: -4.50, 2: -2.00},
    14: {0: -17.30, 1: -9.20, 2: -2.00},
    15: {0: -18.60, 1: -14.00, 2: -2.00},
    16: {0: -20.00, 1: -11.00, 2: -2.00},
    17: {0: -26.30, 1: -14.20, 2: -2.00},
    19: {0: -4.34, 1: -2.73},
    35: {0: -22.07, 1: -13.10, 2: -2.00},
    53: {0: -18.00, 1: -12.70, 2: -2.00},
}


class SyntheticError(ValueError):
    """A species the synthetic chemistry cannot describe; `.code` is the reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def supported_elements() -> frozenset[int]:
    return frozenset(EHT_LEVELS)


def _molecule(key: str):
    """(mol with conformer 0, Z, xyz in Angstrom, charge, embed notes) of a token key."""
    from ksmech.qm.embed import NOTES_PROP, embed, prepare_mol
    from ksmech.qm.records import QMError
    from ksmech.qm.xtb_calc import resolve_charge_spin

    try:
        mol = prepare_mol(key)
        charge, unpaired = resolve_charge_spin(mol)
    except QMError as exc:
        raise SyntheticError(str(exc.reason), exc.detail) from exc
    if unpaired > 0:
        raise SyntheticError("RADICAL_NOT_SUPPORTED", f"{key}: {unpaired} unpaired electrons")
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    bad = sorted({int(x) for x in z} - supported_elements())
    if bad:
        raise SyntheticError("UNSUPPORTED_ELEMENT", f"no EHT levels for Z={bad}")
    try:
        embed(mol, 1, seed=0)
    except QMError as exc:
        raise SyntheticError(str(exc.reason), exc.detail) from exc
    notes = mol.GetProp(NOTES_PROP).split(",") if mol.HasProp(NOTES_PROP) else []
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)
    return mol, z, xyz, int(charge), [n for n in notes if n]


def basis(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ao_atom, ao_l, h0 in eV) in tblite's AO order (atom by atom, shell by shell)."""
    from ksmech.qm.ao import element_ao_l

    ao_atom, ao_l, h0 = [], [], []
    for i, zi in enumerate(int(x) for x in z):
        levels = EHT_LEVELS[zi]
        for ell in element_ao_l(zi):
            if ell not in levels and ell < 2:
                raise SyntheticError("UNSUPPORTED_ELEMENT", f"Z={zi} has no EHT level for l={ell}")
            ao_atom.append(i)
            ao_l.append(ell)
            h0.append(levels.get(ell, D_LEVEL_DEFAULT_EV))
    return np.array(ao_atom, np.int32), np.array(ao_l, np.int32), np.array(h0, float)


def _hamiltonian(S: np.ndarray, h: np.ndarray) -> np.ndarray:
    H = K_EHT * S * 0.5 * (h[:, None] + h[None, :])
    np.fill_diagonal(H, h)
    return H


def scc(
    S, h0, ao_atom, z, n_occ: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Charge-iterated EHT: (eps eV, C, H eV, Mulliken charges, converged)."""
    from scipy.linalg import eigh

    from ksmech.orbitals.localize import valence_electrons

    n_atoms = len(z)
    z_val = np.array([valence_electrons(int(x)) for x in z], dtype=float)
    q = np.zeros(n_atoms)
    converged = False
    for _ in range(MAX_ROUNDS):
        H = _hamiltonian(S, h0 - OMEGA_EV * q[ao_atom])
        eps, C = eigh(H, S)
        occ = C[:, :n_occ]
        P = 2.0 * occ @ occ.T
        q_new = z_val - np.bincount(ao_atom, weights=np.diag(P @ S), minlength=n_atoms)
        if np.max(np.abs(q_new - q), initial=0.0) < CHARGE_TOL:
            q, converged = q_new, True
            break
        q = 0.5 * q + 0.5 * q_new
    return eps, C, H, q, converged


def _electron_count(z: np.ndarray, charge: int, nao: int) -> int:
    from ksmech.orbitals.localize import valence_electrons

    n_val = sum(valence_electrons(int(x)) for x in z) - charge
    if n_val < 0 or n_val % 2 or n_val // 2 > nao:
        raise SyntheticError("ELECTRON_COUNT", f"{n_val} valence electrons in {nao} AOs")
    return n_val // 2


def _vbur(z: np.ndarray, xyz: np.ndarray, warnings: list[str]) -> np.ndarray:
    from ksmech.qm.sterics import buried_volume_per_atom

    try:
        return buried_volume_per_atom(z, xyz)
    except Exception as exc:  # noqa: BLE001 - morfeus errors are not a molecule failure
        log.info("buried volume failed: %s", exc)
        warnings.append("VBUR_FAIL")
        return np.full(len(z), np.nan)


def eht_record(key: str):
    """A complete `QMRecord` for a map-free canonical SMILES, filled like
    `xtb_calc.GFN2Calculator._assemble` (finite-difference Fukui fields NaN)."""
    from ksmech.physics.gto import overlap_matrix
    from ksmech.qm import fukui
    from ksmech.qm.records import QMRecord

    _, z, xyz, charge, notes = _molecule(key)
    ao_atom, ao_l, h0 = basis(z)
    S = overlap_matrix(z, xyz)
    n_occ = _electron_count(z, charge, len(h0))
    eps, C, H, q, converged = scc(S, h0, ao_atom, z, n_occ)
    occupations = np.zeros(len(eps))
    occupations[:n_occ] = 2.0
    warnings = ["SYNTHETIC", *([] if converged else [NOT_CONVERGED]), *notes]
    homo = fukui.homo_index(occupations)
    f_plus, f_minus = fukui.frozen_orbital_fukui(C, S, ao_atom, len(z), homo)
    eps_ha = eps / fukui.HARTREE_TO_EV
    nan = np.full(len(z), np.nan)
    rec = QMRecord(
        smiles=key,
        atom_order_smiles=key,
        conformer_id=0,
        method=SYNTHETIC_METHOD,
        charge=charge,
        unpaired=0,
        atomic_numbers=z,
        coords_ang=xyz,
        energy_ha=float(2.0 * eps_ha[:n_occ].sum()),
        orbital_energies_ev=eps,
        occupations=occupations,
        C=C,
        S=S,
        H_core=H / fukui.HARTREE_TO_EV,
        fock=fukui.fock_from_mo(S, C, eps_ha),
        charges=q,
        ao_atom=ao_atom,
        ao_l=ao_l,
        fukui_plus=nan.copy(),
        fukui_minus=nan.copy(),
        fukui_zero=nan.copy(),
        fukui_dual=nan.copy(),
        fukui_fmo_plus=f_plus,
        fukui_fmo_minus=f_minus,
        buried_volume=_vbur(z, xyz, warnings),
        **fukui.global_descriptors(eps, homo),
    )
    if homo is None:
        warnings.append("NO_FRONTIER")
    rec.warnings = ",".join(dict.fromkeys(warnings))
    return rec
