"""The pooled gap factor of atom pairs (spec v9, "Edit decoder", Eq. Gatom) on a pool's tokens.

    G_atom[i, j] = sum_{r occ, s virt valid} (w_i^r w_j^s + w_j^r w_i^s) G(r, s)

with w the centre weights of the tokens (`ReactantTokens.centres`: every atom carrying at least
tau_atom of the token's population) and G the geometry-free gap factor over the full canonical
spectra (`physics.ks_terms.gap_kernel_matrix`, in float64). Only the pairs the arrow-candidate
rule allows (`physics.pairs.valid_pairs`: a doubly occupied donor, an empty acceptor that is
neither a d polarization function nor the antibond of the donor's own bond) contribute.

`directed[a, b] = sum_rs w_a^r w_b^s G(r, s)` is the part with the donor on a and the acceptor
on b, so `g_atom = directed + directed.T`. The arrow reading of a reaction matrix uses the
directed part to choose which end of a forming bond gives the pair (`serve.reading`); a
channel (r, s) of a pair is one term of the sum, and the orbital reading of a forming bond is
the channel with the largest |term| (spec, "Orbital interpretation").

Atoms here are global atoms of the `ReactantTokens` index space; serve converts them to pool
atoms (the atom indices of `parse_mapped(state.mapped_smiles)`) with `rt.mol_atom`.

Implemented in serve until upstream has it: move upstream when `model.inputs.pooled_gap` lands
(docs/decisions.md, 2026-10-25).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.orbitals.tokens import OrbitalToken


@dataclass
class PooledGap:
    """The token-pair block and the centre weights of one pool, in global indices."""

    rows: np.ndarray  # (n_occ,) global token id of each occupied token
    cols: np.ndarray  # (n_virt,) global token id of each virtual token
    valid: np.ndarray  # (n_occ, n_virt) bool, the arrow-candidate rule
    G: np.ndarray  # (n_occ, n_virt) float64 gap factor (1/eV), every pair
    w_occ: np.ndarray  # (n_occ, n_atoms) centre weights
    w_virt: np.ndarray  # (n_virt, n_atoms)

    @property
    def n_atoms(self) -> int:
        return int(self.w_occ.shape[1])

    @cached_property
    def g_valid(self) -> np.ndarray:
        return np.where(self.valid, self.G, 0.0)

    @cached_property
    def directed(self) -> np.ndarray:
        """(A, A): donor on the row atom, acceptor on the column atom."""
        return self.w_occ.T @ self.g_valid @ self.w_virt

    @cached_property
    def g_atom(self) -> np.ndarray:
        """(A, A) symmetric pooled gap factor."""
        return self.directed + self.directed.T

    def channel_terms(self, i: int, j: int) -> np.ndarray:
        """(n_occ, n_virt) the terms (w_i^r w_j^s + w_j^r w_i^s) G(r, s) of G_atom[i, j]."""
        wo, wv = self.w_occ, self.w_virt
        omega = np.outer(wo[:, i], wv[:, j]) + np.outer(wo[:, j], wv[:, i])
        return omega * self.g_valid

    def directed_terms(self, a: int, b: int) -> np.ndarray:
        """(n_occ, n_virt) the terms w_a^r w_b^s G(r, s): donor on a, acceptor on b."""
        return np.outer(self.w_occ[:, a], self.w_virt[:, b]) * self.g_valid

    def best_channel(self, i: int, j: int) -> tuple[int, int, float] | None:
        """(row, col, term) of the channel with the largest |term| of G_atom[i, j] (ties: the
        lowest row, then col); None when no valid channel has weight on both atoms."""
        terms = self.channel_terms(i, j)
        if not np.any(terms):
            return None
        r, s = np.unravel_index(int(np.argmax(np.abs(terms))), terms.shape)
        return int(r), int(s), float(terms[r, s])

    def top_directed(self, a: int, b: int, k: int) -> list[tuple[int, int, float]]:
        """The k channels (row, col, term) with the largest |w_a^r w_b^s G| (non-zero only)."""
        terms = self.directed_terms(a, b)
        flat = np.flatnonzero(terms.ravel())
        order = flat[np.argsort(-np.abs(terms.ravel()[flat]), kind="stable")][:k]
        n = terms.shape[1]
        return [(int(x // n), int(x % n), float(terms.ravel()[x])) for x in order]

    def to_members(self) -> dict[str, np.ndarray]:
        return {
            "rows": self.rows.astype(np.int64),
            "cols": self.cols.astype(np.int64),
            "valid": self.valid.astype(bool),
            "G": self.G.astype(np.float64),
            "w_occ": self.w_occ.astype(np.float64),
            "w_virt": self.w_virt.astype(np.float64),
        }

    @classmethod
    def from_members(cls, m) -> PooledGap:
        return cls(
            np.asarray(m["rows"]),
            np.asarray(m["cols"]),
            np.asarray(m["valid"], dtype=bool),
            np.asarray(m["G"], dtype=np.float64),
            np.asarray(m["w_occ"], dtype=np.float64),
            np.asarray(m["w_virt"], dtype=np.float64),
        )


def _weights(rt: ReactantTokens, tokens: np.ndarray) -> np.ndarray:
    out = np.zeros((len(tokens), rt.n_atoms))
    for k, t in enumerate(tokens):
        for g, w in rt.centres(int(t)).items():
            out[k, g] = w
    return out


def pooled_gap(rt: ReactantTokens) -> PooledGap:
    """The pooled-gap block of a reactant side (every fragment, one global index space)."""
    from ksmech.physics.ks_terms import gap_kernel_matrix
    from ksmech.physics.pairs import valid_pairs

    occ_ids = np.flatnonzero(rt.occupied)
    virt_ids = np.flatnonzero(~rt.occupied)
    occ = [rt.orbitals[int(j)] for j in occ_ids]
    virt = [rt.orbitals[int(j)] for j in virt_ids]
    G = gap_kernel_matrix(occ, virt) if occ and virt else np.zeros((len(occ), len(virt)))
    valid = valid_pairs([f.tokens for f in rt.fragments], rt.candidate)
    rows, cols = rt.n_atoms + occ_ids, rt.n_atoms + virt_ids
    return PooledGap(rows, cols, valid, G, _weights(rt, rows), _weights(rt, cols))


# ---- the dominant canonical MO pair of G(r, s)


def term_matrix(r: OrbitalToken, s: OrbitalToken) -> np.ndarray:
    """T[k, m] = 2 p_rk p_sm / (E_k - E_m) over the full spectra (`ks_terms.gap_kernel`'s
    expression and clamp), block-local indices; G(r, s) = T.sum()."""
    from ksmech.physics.ks_terms import MIN_GAP_EV

    er = np.asarray(r.descriptors.full_energies, dtype=np.float64)
    pr = np.asarray(r.descriptors.full_weights, dtype=np.float64)
    es = np.asarray(s.descriptors.full_energies, dtype=np.float64)
    ps = np.asarray(s.descriptors.full_weights, dtype=np.float64)
    gap = er[:, None] - es[None, :]
    gap = np.where(gap >= 0, 1.0, -1.0) * np.maximum(np.abs(gap), MIN_GAP_EV)
    return 2.0 * pr[:, None] * ps[None, :] / gap


def block_indices(n_occ: int, n_virt: int, occupations: np.ndarray | None) -> tuple:
    """(canonical index of every occupied-block position, of every virtual one, source):
    from the QM occupations when they fit the blocks, else the aufbau order."""
    if occupations is not None:
        from ksmech.qm.fukui import OCC_THRESHOLD

        occ = np.asarray(occupations, dtype=float)
        i_occ, i_virt = np.flatnonzero(occ >= OCC_THRESHOLD), np.flatnonzero(occ < OCC_THRESHOLD)
        if len(i_occ) == n_occ and len(i_virt) == n_virt:
            return i_occ, i_virt, "qm_occupations"
    return np.arange(n_occ), n_occ + np.arange(n_virt), "aufbau"


def _block_len(tokens: list) -> int:
    return len(tokens[0].descriptors.full_energies) if tokens else 0


def homo_label(canonical: int, homo: int | None) -> str:
    if homo is None or canonical == homo:
        return "HOMO"
    return f"HOMO-{homo - canonical}" if canonical < homo else f"HOMO+{canonical - homo}"


def lumo_label(canonical: int, lumo: int | None) -> str:
    if lumo is None or canonical == lumo:
        return "LUMO"
    return f"LUMO+{canonical - lumo}" if canonical > lumo else f"LUMO-{lumo - canonical}"


def dominant_pair(r: OrbitalToken, s: OrbitalToken, t_r, t_s, occ_r, occ_s) -> dict | None:
    """The canonical MO pair (k, m) with the largest |T[k, m]| of G(r, s) (ties: lowest k,
    then m): k a canonical MO of the donor's molecule `t_r`, m one of the acceptor's `t_s`
    (indices from their QM occupations `occ_r` / `occ_s`, else the aufbau order), with
    their energies (eV), HOMO/LUMO labels, the term (1/eV) and its share of G."""
    T = term_matrix(r, s)
    if T.size == 0 or not np.any(T):
        return None
    k, m = (int(x) for x in np.unravel_index(int(np.argmax(np.abs(T))), T.shape))
    occ_idx, _, src_r = block_indices(_block_len(t_r.occupied()), _block_len(t_r.virtual()), occ_r)
    _, virt_idx, src_s = block_indices(_block_len(t_s.occupied()), _block_len(t_s.virtual()), occ_s)
    if len(occ_idx) != T.shape[0] or len(virt_idx) != T.shape[1]:
        occ_idx, virt_idx, src_r = np.arange(T.shape[0]), np.arange(T.shape[1]), "block_local"
    k_can, m_can = int(occ_idx[k]), int(virt_idx[m])
    total = float(T.sum())
    homo = int(occ_idx.max()) if len(occ_idx) else None
    lumo = int(virt_idx.min()) if len(virt_idx) else None
    return {
        "k": k_can,
        "m": m_can,
        "e_k_ev": float(r.descriptors.full_energies[k]),
        "e_m_ev": float(s.descriptors.full_energies[m]),
        "k_label": homo_label(k_can, homo),
        "m_label": lumo_label(m_can, lumo),
        "term": float(T[k, m]),
        "share": float(T[k, m] / total) if total and math.isfinite(total) else None,
        "index_source": src_r if src_r == src_s else f"{src_r}/{src_s}",
    }
