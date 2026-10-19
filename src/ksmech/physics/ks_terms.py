"""Klopman-Salem pair terms between orbital tokens: term 3 (filled-empty) in spectral form.

    KS3(r, s) = kappa(r, s)^2 * G(r, s)
    G(r, s)   = sum_k sum_m 2 p_rk p_sm / (E_k - E_m)

`G` is the gap kernel over the *full* spectral decompositions of the two tokens on the
canonical MOs of their blocks (never the truncated 8-entry model input): canonical energies in
the denominator. `kappa` is a coupling of the localized coefficient vectors:

- `contact_coupling` (ksmech.physics.contact), used for **every** pair: the two orbitals are
  evaluated as if brought into contact, which needs no common AO basis and is non-zero within
  one molecule too. `ks3_contact` is the unified term.
- `coupling(l_r, B, l_s)` with a matrix `B` in a common AO basis is the converged form. For
  occupied and virtual LMOs of one converged molecule it is identically zero with the
  molecule's own `S` or `F` (they are S-orthogonal and F-decoupled); it is kept for reference
  and for intermolecular matrices computed on a complex.
- For intramolecular pairs two extra pair features: the centroid distance and the
  cross-centre coupling, `l_r^T B l_s` restricted to the AO blocks of the donor's centres
  (rows) and the acceptor's centres (columns), which does not vanish.

For an occupied r and a virtual s every E_k - E_m is negative, so G < 0, and the more negative
G the stronger the stabilizing interaction. Energies are in eV, G in 1/eV.

Terms 1 (closed-shell repulsion) and 2 (Coulomb) are defined in the physics phase and are not
part of this module.
"""

from __future__ import annotations

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.contact import contact_coupling

MIN_GAP_EV = 1e-3  # a canonical pair closer than this is clamped (same-block degeneracies)


def coupling(l_r: np.ndarray, B: np.ndarray, l_s: np.ndarray) -> float:
    """l_r^T B l_s in a common AO basis."""
    return float(l_r @ B @ l_s)


def gap_kernel(
    energies_r: np.ndarray,
    weights_r: np.ndarray,
    energies_s: np.ndarray,
    weights_s: np.ndarray,
    min_gap_ev: float = MIN_GAP_EV,
) -> float:
    """G = sum_k sum_m 2 p_rk p_sm / (E_k - E_m) over two spectral sets (zero weights skip)."""
    mask_r, mask_s = weights_r > 0, weights_s > 0
    e_r, p_r = energies_r[mask_r], weights_r[mask_r]
    e_s, p_s = energies_s[mask_s], weights_s[mask_s]
    gap = e_r[:, None] - e_s[None, :]
    sign = np.where(gap >= 0, 1.0, -1.0)
    gap = sign * np.maximum(np.abs(gap), min_gap_ev)
    return float(np.sum(2.0 * p_r[:, None] * p_s[None, :] / gap))


def token_gap_kernel(r: OrbitalToken, s: OrbitalToken, min_gap_ev: float = MIN_GAP_EV) -> float:
    """G from the full spectral vectors of the two tokens."""
    d_r, d_s = r.descriptors, s.descriptors
    return gap_kernel(
        d_r.full_energies, d_r.full_weights, d_s.full_energies, d_s.full_weights, min_gap_ev
    )


def token_gap_kernel_truncated(r: OrbitalToken, s: OrbitalToken) -> float:
    """G from the 8-entry model-input spectra: for diagnostics only (see the truncation test)."""
    d_r, d_s = r.descriptors, s.descriptors
    return gap_kernel(
        d_r.spectral_energies, d_r.spectral_weights, d_s.spectral_energies, d_s.spectral_weights
    )


def ks3(
    l_r: np.ndarray,
    l_s: np.ndarray,
    B: np.ndarray,
    energies_r: np.ndarray,
    weights_r: np.ndarray,
    energies_s: np.ndarray,
    weights_s: np.ndarray,
) -> float:
    """Converged form of term 3 for one pair: (l_r^T B l_s)^2 * G(r, s)."""
    return coupling(l_r, B, l_s) ** 2 * gap_kernel(energies_r, weights_r, energies_s, weights_s)


def gap_kernel_matrix(rows: list[OrbitalToken], cols: list[OrbitalToken]) -> np.ndarray:
    """G for every pair of tokens; works across molecules (full spectral sets only)."""
    out = np.empty((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, s in enumerate(cols):
            out[i, j] = token_gap_kernel(r, s)
    return out


def ks3_matrix(rows: list[OrbitalToken], cols: list[OrbitalToken], B: np.ndarray) -> np.ndarray:
    """Converged term 3 for every pair of tokens of one molecule (`B` in its AO basis)."""
    L_r = np.column_stack([r.coeff_vector for r in rows])
    L_s = np.column_stack([s.coeff_vector for s in cols])
    return (L_r.T @ B @ L_s) ** 2 * gap_kernel_matrix(rows, cols)


def _atomic_numbers(t: MoleculeTokens) -> np.ndarray:
    return np.array([a.atomic_number for a in t.atoms])


def token_contact_coupling(
    r: OrbitalToken, t_r: MoleculeTokens, s: OrbitalToken, t_s: MoleculeTokens
):
    """kappa(r, s): best centre-pair contact coupling (tokens of any two molecules)."""
    return contact_coupling(
        r.descriptors, _atomic_numbers(t_r), s.descriptors, _atomic_numbers(t_s)
    )


def ks3_contact(
    r: OrbitalToken, t_r: MoleculeTokens, s: OrbitalToken, t_s: MoleculeTokens
) -> float:
    """Unified term 3: contact coupling squared times the full-spectrum gap kernel."""
    return token_contact_coupling(r, t_r, s, t_s) ** 2 * token_gap_kernel(r, s)


def ks3_contact_matrix(
    rows: list[OrbitalToken], t_r: MoleculeTokens, cols: list[OrbitalToken], t_s: MoleculeTokens
) -> np.ndarray:
    out = np.empty((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, s in enumerate(cols):
            out[i, j] = ks3_contact(r, t_r, s, t_s)
    return out


def centroid_distance(r: OrbitalToken, s: OrbitalToken) -> float:
    """Intramolecular pair feature: distance between the two orbital centroids (Angstrom)."""
    return float(np.linalg.norm(r.descriptors.centroid - s.descriptors.centroid))


def cross_center_coupling(r: OrbitalToken, s: OrbitalToken, B: np.ndarray, ao_atom: np.ndarray):
    """l_r^T B l_s restricted to AO rows on r's centres and AO columns on s's centres."""
    rows = np.isin(ao_atom, r.descriptors.center_atoms[r.descriptors.center_atoms >= 0])
    cols = np.isin(ao_atom, s.descriptors.center_atoms[s.descriptors.center_atoms >= 0])
    return float(r.coeff_vector[rows] @ B[np.ix_(rows, cols)] @ s.coeff_vector[cols])


def intramolecular_pair_features(
    r: OrbitalToken, s: OrbitalToken, t: MoleculeTokens, B: np.ndarray | None = None
) -> dict[str, float]:
    """The unified term plus the two intramolecular extras for a pair of one molecule."""
    B = t.S if B is None else B
    return dict(
        gap_kernel=token_gap_kernel(r, s),
        contact_coupling=token_contact_coupling(r, t, s, t),
        ks3_contact=ks3_contact(r, t, s, t),
        centroid_distance=centroid_distance(r, s),
        cross_center_coupling=cross_center_coupling(r, s, B, t.ao_atom),
    )
