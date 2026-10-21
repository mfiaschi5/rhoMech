"""Klopman-Salem pair terms between orbital tokens: term 3 (filled-empty) in spectral form.

    KS3(r, s) = beta(r, s)^2 * G(r, s)
    G(r, s)   = sum_k sum_m 2 p_rk p_sm / (E_k - E_m)

`G` is the gap kernel over the *full* spectral decompositions of the two tokens on the
canonical MOs of their blocks (never the truncated 8-entry model input): canonical energies in
the denominator. `beta` couples the two localized orbitals:

- the contact resonance integral (`ksmech.physics.contact_kernel`), used for **every** pair,
  within one molecule or across two: the two orbitals are evaluated as if brought into
  contact, which needs no common AO basis. `ks3_contact` is the unified term (eV).
- `coupling(l_r, B, l_s)` with a matrix `B` in a common AO basis is the converged form. For
  occupied and virtual LMOs of one converged molecule it is identically zero with the
  molecule's own `S` or `F` (they are S-orthogonal and F-decoupled).
- For intramolecular pairs two extra pair features: the centroid distance and the
  cross-centre coupling `l_r^T B l_s` restricted to the AO blocks between the donor's centres
  and the acceptor's centres, with the spec's B: the Wolfsberg-Helmholz resonance integral
  `resonance_integral(S, H_aa, H_bb, K)` on AO pairs of different atoms, 0 within an atom.
  H is the diagonal of the converged GFN2 Hamiltonian (the Fock matrix the token files store).
  Unlike S or F the restricted coupling does not vanish between converged occupied and
  virtual LMOs; it measures how much the two orbitals share atoms and bonds in the computed
  geometry, not whether they can meet.

For an occupied r and a virtual s of one molecule every E_k - E_m is negative, so G < 0, and
the more negative G the stronger the stabilizing interaction. Between two molecules a donor
level can lie above an acceptor level (an anion and a cation), and those terms are positive.
Energies are in eV, G in 1/eV. Terms 1 and 2 live in `ksmech.physics.repulsion` and
`ksmech.physics.coulomb`.
"""

from __future__ import annotations

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.contact.legacy_fixed23 import ao_energies
from ksmech.physics.contact_kernel import ContactKernel
from ksmech.physics.params import KSParams
from ksmech.qm.fukui import HARTREE_TO_EV

MIN_GAP_EV = 1e-3  # a canonical pair closer than this is clamped (same-block degeneracies)


def resonance_integral(S_ab, H_aa, H_bb, K: float = 1.75):
    """Wolfsberg-Helmholz: beta_ab = K S_ab (H_aa + H_bb) / 2 (broadcasts; eV in, eV out)."""
    return K * np.asarray(S_ab) * 0.5 * (np.asarray(H_aa) + np.asarray(H_bb))


def wolfsberg_helmholz(
    S: np.ndarray, F: np.ndarray, ao_atom: np.ndarray, K: float, h: np.ndarray | None = None
) -> np.ndarray:
    """B_mu,nu = beta_mu,nu (eV) for AOs on different atoms, 0 within an atom.

    `F` is the converged GFN2 Fock matrix in Hartree; only its diagonal enters. `h`: the AO
    energies (eV) to use instead, as `wh_matrix` passes the shell averages (`ao_energies`).
    """
    h = np.diag(F) * HARTREE_TO_EV if h is None else h
    B = resonance_integral(S, h[:, None], h[None, :], K)
    B[ao_atom[:, None] == ao_atom[None, :]] = 0.0
    return B


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


def _blocks(tokens: list[OrbitalToken]):
    """(indices, block energies, stacked full weights) per spectral block among `tokens`."""
    groups: dict[bytes, list[int]] = {}
    for i, o in enumerate(tokens):
        groups.setdefault(np.asarray(o.descriptors.full_energies).tobytes(), []).append(i)
    for idx in groups.values():
        energies = tokens[idx[0]].descriptors.full_energies
        weights = np.array([tokens[i].descriptors.full_weights for i in idx])
        yield np.array(idx), energies, weights


def gap_kernel_matrix(
    rows: list[OrbitalToken], cols: list[OrbitalToken], min_gap_ev: float = MIN_GAP_EV
) -> np.ndarray:
    """G for every pair of tokens; works across molecules (full spectral sets only).

    Tokens of one block share its canonical energies, so each block pair is one product
    G = 2 P_r D P_s^T with D_km = 1 / (E_k - E_m).
    """
    out = np.empty((len(rows), len(cols)))
    for ri, e_r, P_r in _blocks(rows):
        for ci, e_s, P_s in _blocks(cols):
            gap = e_r[:, None] - e_s[None, :]
            gap = np.where(gap >= 0, 1.0, -1.0) * np.maximum(np.abs(gap), min_gap_ev)
            out[np.ix_(ri, ci)] = 2.0 * P_r @ (1.0 / gap) @ P_s.T
    return out


def ks3_matrix(rows: list[OrbitalToken], cols: list[OrbitalToken], B: np.ndarray) -> np.ndarray:
    """Converged term 3 for every pair of tokens of one molecule (`B` in its AO basis)."""
    L_r = np.column_stack([r.coeff_vector for r in rows])
    L_s = np.column_stack([s.coeff_vector for s in cols])
    return (L_r.T @ B @ L_s) ** 2 * gap_kernel_matrix(rows, cols)


def ks3_contact(
    r: OrbitalToken,
    t_r: MoleculeTokens,
    s: OrbitalToken,
    t_s: MoleculeTokens,
    kernel: ContactKernel,
    params: KSParams | None = None,
) -> float:
    """Unified term 3 (eV): contact resonance integral squared times the gap kernel."""
    return kernel.beta(r, t_r, s, t_s, params) ** 2 * token_gap_kernel(r, s)


def contact_beta_matrix(rows, t_r, cols, t_s, kernel: ContactKernel, params=None) -> np.ndarray:
    """Contact resonance integral (eV) for every (row, col) token pair."""
    return kernel.beta_matrix(rows, t_r, cols, t_s, params)


def ks3_contact_matrix(rows, t_r, cols, t_s, kernel: ContactKernel, params=None) -> np.ndarray:
    beta = contact_beta_matrix(rows, t_r, cols, t_s, kernel, params)
    return beta**2 * gap_kernel_matrix(rows, cols)


def centroid_distance(r: OrbitalToken, s: OrbitalToken) -> float:
    """Intramolecular pair feature: distance between the two orbital centroids (Angstrom)."""
    return float(np.linalg.norm(r.descriptors.centroid - s.descriptors.centroid))


def wh_matrix(t: MoleculeTokens, params: KSParams | None = None) -> np.ndarray:
    """The molecule's Wolfsberg-Helmholz B (eV), zero within an atom, on the shell-averaged
    AO energies (rotation-invariant, `oriented.ao_energies`)."""
    K = (params or KSParams()).K
    return wolfsberg_helmholz(t.S, t.F, t.ao_atom, K, ao_energies(t))


def centre_restricted(tokens: list[OrbitalToken], ao_atom: np.ndarray) -> np.ndarray:
    """(nao, n) coefficient columns with every AO off the token's own centres A_r set to 0."""
    L = np.column_stack([o.coeff_vector for o in tokens]).reshape(len(ao_atom), -1)
    for k, o in enumerate(tokens):
        L[~np.isin(ao_atom, o.descriptors.all_center_atoms), k] = 0.0
    return L


def cross_center_coupling_matrix(
    rows: list[OrbitalToken], cols: list[OrbitalToken], ao_atom: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """l_r^T B l_s (eV) over the AO blocks between r's centres and s's centres, every pair."""
    return centre_restricted(rows, ao_atom).T @ B @ centre_restricted(cols, ao_atom)


def cross_center_coupling(
    r: OrbitalToken, s: OrbitalToken, B: np.ndarray, ao_atom: np.ndarray
) -> float:
    """The intramolecular coupling extra of one pair (B zero within an atom)."""
    return float(cross_center_coupling_matrix([r], [s], ao_atom, B)[0, 0])


def intramolecular_pair_features(
    r: OrbitalToken,
    s: OrbitalToken,
    t: MoleculeTokens,
    kernel: ContactKernel,
    params: KSParams | None = None,
) -> dict[str, float]:
    """The unified term plus the two intramolecular extras for a pair of one molecule."""
    return dict(
        gap_kernel=token_gap_kernel(r, s),
        contact_beta=kernel.beta(r, t, s, t, params, same=True),
        ks3_contact=kernel.beta(r, t, s, t, params, same=True) ** 2 * token_gap_kernel(r, s),
        centroid_distance=centroid_distance(r, s),
        cross_center_coupling=cross_center_coupling(r, s, wh_matrix(t, params), t.ao_atom),
    )
