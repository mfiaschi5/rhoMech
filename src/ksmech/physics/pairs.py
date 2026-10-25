"""Pair-level helpers shared by the sample build and the rankers: the arrow-candidate filter of
the (occupied, virtual) token pairs, the atom-atom Coulomb matrix of the stored samples, the
block of pair features, and their standardization.

The pair features themselves are geometry-free (`physics.stage_a`). Each entry is standardized
with statistics of the defined entries of a training set (`fit_pair_stats`). The contact
features, the Klopman-Salem attention bias and the frozen head of the archived contact model
are on branch `archive/stage-b`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ksmech.orbitals.tokens import (
    CandidateConfig,
    MoleculeTokens,
    antibonding_partner_matrix,
    site_excluded_matrix,
)
from ksmech.physics.coulomb import coulomb_matrix, vdw_sum
from ksmech.physics.params import KSParams

LOG_FLOOR = 1e-12


@dataclass
class PairBlock:
    z: np.ndarray  # (n_occ, n_virt, n_features), 0 where undefined
    defined: np.ndarray  # same shape, bool
    gap_sign: np.ndarray  # (n_occ, n_virt) int8: sign of G (bookkeeping, not a feature)
    sites: np.ndarray  # (n_occ, n_virt, 2) int64, always -1: kept for the schema-15 store
    n_non_finite: int = 0


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def valid_pairs(fragments: list[MoleculeTokens], config: CandidateConfig | None = None):
    """(n_occ, n_virt) arrow-candidate filter: a doubly occupied donor, an empty acceptor that
    is not d-type (unless allowed), and, within one fragment, not the donor's own antibond and
    not a pair whose every site pair is a bond interacting with itself (sites identical or
    bonded, and the acceptor carries >= `config.site_weight` on the donor's site or the donor
    on the acceptor's: `tokens.site_excluded_matrix`). Every criterion is read from descriptors
    and the record geometry (`ksmech.orbitals.tokens`)."""
    config = config or CandidateConfig()
    donors = {id(o) for t in fragments for o in t.donor_candidates(config)}
    acceptors = {id(o) for t in fragments for o in t.acceptor_candidates(config)}
    occ = [o for t in fragments for o in t.occupied()]
    virt = [o for t in fragments for o in t.virtual()]
    ok_d = np.array([id(o) in donors for o in occ], dtype=bool)
    ok_a = np.array([id(o) in acceptors for o in virt], dtype=bool)
    out = ok_d[:, None] & ok_a[None, :]
    r0 = np.cumsum([0, *[len(t.occupied()) for t in fragments]])
    c0 = np.cumsum([0, *[len(t.virtual()) for t in fragments]])
    for f, t in enumerate(fragments):  # centres are atoms of one fragment
        o_f, v_f = t.occupied(), t.virtual()
        if not o_f or not v_f:
            continue
        own = antibonding_partner_matrix(o_f, v_f)
        bad = own | site_excluded_matrix(t, o_f, v_f, config.site_weight)
        out[r0[f] : r0[f + 1], c0[f] : c0[f + 1]] &= ~bad
    return out


def atom_coulomb_matrix(fragments: list[MoleculeTokens], params: KSParams | None = None):
    """(A, A) symmetric KS2 between atoms, in the sample's atom order (fragment by fragment):
    the geometric distance within a fragment, the van der Waals contact across fragments,
    0 on the diagonal."""
    atoms = [a for t in fragments for a in t.atoms]
    frag = np.repeat(np.arange(len(fragments)), [len(t.atoms) for t in fragments])
    pos = np.array([a.position for a in atoms], dtype=float).reshape(-1, 3)
    z = [a.atomic_number for a in atoms]
    q = np.array([a.mulliken_charge for a in atoms], dtype=float)
    inter = frag[:, None] != frag[None, :]
    vdw = np.array([[vdw_sum(a, b) for b in z] for a in z]).reshape(inter.shape)
    distance = np.where(inter, vdw, _pairwise(pos, pos))
    return coulomb_matrix(q, distance, ~np.eye(len(atoms), dtype=bool), params)


# ---- standardization


@dataclass(frozen=True)
class PairStats:
    """Mean and standard deviation of each z entry over the defined entries of a training set,
    at the physical K and eps, then frozen."""

    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    names: tuple[str, ...]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            n: {"mean": m, "std": s, "count": int(c)}
            for n, m, s, c in zip(
                self.names, self.mean.tolist(), self.std.tolist(), self.count, strict=True
            )
        }

    @classmethod
    def from_dict(cls, d: dict, names: tuple[str, ...]) -> PairStats:
        if tuple(d) != tuple(names):
            raise ValueError(f"pair statistics for {tuple(d)}, expected {tuple(names)}")
        values = (np.array([d[n][k] for n in names]) for k in ("mean", "std", "count"))
        return cls(*values, names=tuple(names))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_json(cls, text: str, names: tuple[str, ...]) -> PairStats:
        return cls.from_dict(json.loads(text), names)


def fit_pair_stats(blocks, names: tuple[str, ...]) -> PairStats:
    """Streaming mean / std over the defined entries of many (z, defined) pairs whose last
    axis holds the features `names`."""
    k = len(names)
    n, s1, s2 = np.zeros(k), np.zeros(k), np.zeros(k)
    for z, defined in blocks:
        for j in range(k):
            v = z[..., j][defined[..., j]].astype(float)
            n[j], s1[j], s2[j] = n[j] + v.size, s1[j] + v.sum(), s2[j] + (v * v).sum()
    mean = np.divide(s1, n, out=np.zeros(k), where=n > 0)
    var = np.divide(s2, n, out=np.zeros(k), where=n > 0) - mean**2
    return PairStats(mean, np.sqrt(np.clip(var, 0.0, None)), n, tuple(names))


def standardize(z: np.ndarray, defined: np.ndarray, stats: PairStats) -> np.ndarray:
    """(z - mean) / std on defined entries, 0 elsewhere; a constant entry is only centred."""
    std = np.where(stats.std > 0, stats.std, 1.0)
    return np.where(defined, (z - stats.mean) / std, 0.0)
