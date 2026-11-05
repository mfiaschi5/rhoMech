"""The geometry-free Klopman-Salem quantities between orbital tokens.

    G(r, s) = sum_k sum_m (n_k - n_m) p_rk p_sm / (E_k - E_m)

with n_k - n_m = 2 for a closed-shell occupied token and an empty one (the constant-2 fast path,
bitwise the G of every model before P7), and the token occupations where an open-shell token in
its donor role or a hole takes part (open-shell plan P7; after P2b every MO of a token's
spectrum has the token's occupation): a SOMO to an empty token and a doubly occupied token to a
hole weigh 1, Br.'s 4p (5/3) to an empty token 5/3, and an open-shell row to a hole column
weighs 0, where G is undefined (`gap_weights`). `gap_kernel_two_level` is g2, the same sum with
each canonical denominator replaced by the exact two-level one.

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


def electron_counts(tokens: list[OrbitalToken], occupied: bool) -> np.ndarray:
    """n of each token in G's weight: its occupation for an open-shell token in its donor role
    or a hole (`spin_role` != 0), else 2 for an occupied token and 0 for a virtual one."""
    closed = 2.0 if occupied else 0.0
    return np.array(
        [o.descriptors.occupation if o.spin_role else closed for o in tokens], dtype=float
    )


WEIGHT_TOL = 1e-4  # |n_r - n_s| below this is 0: one plateau's occupations agree to ~1e-6


def gap_weights(rows: list[OrbitalToken], cols: list[OrbitalToken]) -> np.ndarray:
    """(len(rows), len(cols)) n_r - n_s, rows occupied and columns virtual (`electron_counts`):
    2 on every closed-shell cell, 0 on an open-shell row with a hole column (two SOMOs of one
    plateau differ by ~1e-7 in their GFN2 occupations: below `WEIGHT_TOL` the weight is 0)."""
    w = electron_counts(rows, True)[:, None] - electron_counts(cols, False)[None, :]
    return np.where(np.abs(w) < WEIGHT_TOL, 0.0, w)


def gap_kernel_matrix(
    rows: list[OrbitalToken], cols: list[OrbitalToken], min_gap_ev: float = MIN_GAP_EV
) -> np.ndarray:
    """G for every pair of tokens; works across molecules (full spectral sets only).

    Tokens of one block share its canonical energies, so each block pair is one product
    G = 2 P_r D P_s^T with D_km = 1 / (E_k - E_m). Where a row or a column has a spin role the
    weight 2 is replaced by `gap_weights` (0 on an open-shell row with a hole column, where G
    is undefined); every other cell is computed as before, bit for bit.
    """
    out = np.empty((len(rows), len(cols)))
    open_r = np.array([o.spin_role != 0 for o in rows], dtype=bool)
    open_c = np.array([o.spin_role != 0 for o in cols], dtype=bool)
    weights = gap_weights(rows, cols) if open_r.any() or open_c.any() else None
    for ri, e_r, P_r in _blocks(rows):
        for ci, e_s, P_s in _blocks(cols):
            gap = e_r[:, None] - e_s[None, :]
            gap = np.where(gap >= 0, 1.0, -1.0) * np.maximum(np.abs(gap), min_gap_ev)
            out[np.ix_(ri, ci)] = 2.0 * P_r @ (1.0 / gap) @ P_s.T
            if weights is not None and (open_r[ri].any() or open_c[ci].any()):
                base = P_r @ (1.0 / gap) @ P_s.T
                w = weights[np.ix_(ri, ci)]
                touched = open_r[ri][:, None] | open_c[ci][None, :]
                out[np.ix_(ri, ci)] = np.where(touched, w * base, out[np.ix_(ri, ci)])
    return out


def _two_level(gap: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """1 / (sign(dE) D) with D = (|dE| + sqrt(dE^2 + 4 beta^2)) / 2 and sign(0) = -1: the exact
    two-level denominator, |dE| for a large gap and |beta| at degeneracy."""
    sign = np.where(gap > 0, 1.0, -1.0)
    d = 0.5 * (np.abs(gap) + np.sqrt(gap * gap + 4.0 * beta * beta))
    return sign / d


def gap_kernel_two_level(
    rows: list[OrbitalToken], cols: list[OrbitalToken], beta: np.ndarray
) -> np.ndarray:
    """g2 for every pair of tokens: sum_km (n_k - n_m) p_rk p_sm / (sign(dE_km) D_km) with the
    two-level D of the cell's coupling `beta` (eV, (len(rows), len(cols)), one value per cell;
    open-shell plan P7, U23). Bounded by sum |n_k - n_m| p_rk p_sm / |beta|, finite through a
    degeneracy, and G itself as beta -> 0 (without G's clamp at MIN_GAP_EV). The weights are
    `gap_weights`; a cell of weight 0 is 0."""
    out = np.zeros((len(rows), len(cols)))
    weights = gap_weights(rows, cols)
    for ri, e_r, P_r in _blocks(rows):
        for ci, e_s, P_s in _blocks(cols):
            gap = e_r[:, None] - e_s[None, :]
            for a, i in enumerate(ri):
                b = beta[i, ci]
                inv = _two_level(gap[None, :, :], b[:, None, None])  # (cells, K, M)
                vals = np.einsum("k,ckm,cm->c", P_r[a], inv, P_s)
                out[i, ci] = weights[i, ci] * vals
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
