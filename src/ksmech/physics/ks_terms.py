"""The geometry-free Klopman-Salem quantities between orbital tokens.

    G(r, s) = sum_k sum_m 2 p_rk p_sm / (E_k - E_m)

`G` is the gap kernel, the energy denominator of the filled-empty term, over the *full*
spectral decompositions of the two tokens on the canonical MOs of their blocks (never the
truncated 8-entry model input): canonical energies in the denominator. It needs no geometry,
so it is defined within one molecule and across two.

For intramolecular pairs one more quantity: the cross-centre coupling `l_r^T B l_s` restricted
to the AO blocks between the donor's centres and the acceptor's centres, with B the
Wolfsberg-Helmholz resonance integral `resonance_integral(S, H_aa, H_bb, K)` on AO pairs of
different atoms, 0 within an atom. H is the diagonal of the converged GFN2 Hamiltonian (the
Fock matrix the token files store), averaged per shell (`ao_energies`). The molecule's own S or
F couple none of its converged occupied and virtual LMOs (they are S-orthogonal and
F-decoupled); the restricted coupling does not vanish: it measures how much the two orbitals
share atoms and bonds in the computed geometry, not whether they can meet.

For an occupied r and a virtual s of one molecule every E_k - E_m is negative, so G < 0, and
the more negative G the stronger the stabilizing interaction. Between two molecules a donor
level can lie above an acceptor level (an anion and a cation), and those terms are positive.
Energies are in eV, G in 1/eV. The contact resonance integral, the Coulomb and the repulsion
terms of the archived contact model are on branch `archive/stage-b`.
"""

from __future__ import annotations

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.params import KSParams
from ksmech.qm.ao import gfn2_shells
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


def ao_energies(t: MoleculeTokens) -> np.ndarray:
    """(nao,) the AO energies of the Wolfsberg-Helmholz resonance integral (eV): the molecule's
    own GFN2 Fock diagonal, averaged over the 2l + 1 components of each shell (tblite's AO
    order, `qm.ao.gfn2_shells`). GFN2's anisotropic terms split the lab-frame diagonal of a p
    or d shell (0.06-0.2 eV on C and O p, 0.4 eV on Br d) in a way that depends on the record's
    orientation; the shell average is rotation-invariant, one valence-state energy per shell."""
    f = np.diag(t.F) * HARTREE_TO_EV
    out, k = f.copy(), 0
    shells = gfn2_shells()
    for atom in t.atoms:
        for _, ell, _, _ in shells[int(atom.atomic_number)]:
            n = 2 * ell + 1
            out[k : k + n] = f[k : k + n].mean()
            k += n
    if k != len(f):
        raise ValueError(f"{t.smiles}: {k} shell AOs for {len(f)} AOs")
    return out


def wh_matrix(t: MoleculeTokens, params: KSParams | None = None) -> np.ndarray:
    """The molecule's Wolfsberg-Helmholz B (eV), zero within an atom, on the shell-averaged
    AO energies (rotation-invariant, `ao_energies`)."""
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
