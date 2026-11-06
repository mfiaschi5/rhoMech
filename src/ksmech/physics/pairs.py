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


PAIR_FLAGS = ("d_acceptor", "own_antibond", "site_self")  # stored 0/1 pair features (v18)
PAIR_VALID = "pair_valid"  # the legacy mask read as a 0/1 feature (pair_cells 'all')
PAIR_CELLS = ("valid", "all")  # `ModelConfig.pair_cells`: the cells a reader sees


def cell_columns(z, defined, valid, stored, names, cells: str = "valid"):
    """(z, defined), (..., len(names)): the features `names` selected by name from the `stored`
    columns of (z, defined), so a reader built for fewer features reads a larger store.
    `PAIR_VALID` is the legacy mask `valid` as a 0/1 column, defined on every cell. With
    `cells` 'valid' every column is undefined outside `valid` (what a v17 store held); with
    'all' every cell keeps its entries."""
    if cells not in PAIR_CELLS:
        raise ValueError(f"pair cells {cells!r}, expected one of {PAIR_CELLS}")
    missing = [n for n in names if n != PAIR_VALID and n not in stored]
    if missing:
        raise ValueError(f"pair features {missing} are not stored (stored: {tuple(stored)})")
    zs, ds = [], []
    for n in names:
        if n == PAIR_VALID:
            zs.append(valid.astype(z.dtype))
            ds.append(np.ones(valid.shape, dtype=bool))
        else:
            zs.append(z[..., stored.index(n)])
            ds.append(defined[..., stored.index(n)])
    z_out, d_out = np.stack(zs, -1), np.stack(ds, -1)
    if cells == "valid":
        d_out &= valid[..., None]
    return z_out, d_out


def _fragment_blocks(fragments: list[MoleculeTokens], config: CandidateConfig):
    """(own_antibond, site_self), both (n_occ, n_virt): the two within-fragment exclusions of
    every (occupied, virtual) cell, False across fragments (centres are atoms of one
    fragment)."""
    n_occ = sum(len(t.occupied()) for t in fragments)
    n_virt = sum(len(t.virtual()) for t in fragments)
    own = np.zeros((n_occ, n_virt), dtype=bool)
    site = np.zeros((n_occ, n_virt), dtype=bool)
    r0 = np.cumsum([0, *[len(t.occupied()) for t in fragments]])
    c0 = np.cumsum([0, *[len(t.virtual()) for t in fragments]])
    for f, t in enumerate(fragments):
        o_f, v_f = t.occupied(), t.virtual()
        if not o_f or not v_f:
            continue
        block = (slice(r0[f], r0[f + 1]), slice(c0[f], c0[f + 1]))
        own[block] = antibonding_partner_matrix(o_f, v_f)
        site[block] = site_excluded_matrix(t, o_f, v_f, config.site_weight)
    return own, site


def pair_flags(
    fragments: list[MoleculeTokens], config: CandidateConfig | None = None
) -> dict[str, np.ndarray]:
    """The three exclusions of `valid_pairs` as (n_occ, n_virt) bool arrays named by
    `PAIR_FLAGS`, on every cell: `d_acceptor` (the virtual token is a d polarization function,
    `tokens.is_d_type`, whatever `allow_d_acceptors` says), `own_antibond` (the donor's own
    antibond) and `site_self` (every site pair is a bond interacting with itself). Stored as
    pair features, so a model may read them as channels instead of having the cells deleted
    (open-shell plan P0a, item 1)."""
    config = config or CandidateConfig()
    occ = [o for t in fragments for o in t.occupied()]
    d_type = np.array([t.is_d_type(o) for t in fragments for o in t.virtual()], dtype=bool)
    own, site = _fragment_blocks(fragments, config)
    d_acc = np.broadcast_to(d_type[None, :], (len(occ), len(d_type))).copy()
    return {"d_acceptor": d_acc, "own_antibond": own, "site_self": site}


def open_self_cells(fragments: list[MoleculeTokens]) -> np.ndarray:
    """(n_occ, n_virt) the open-shell rows x hole columns of one fragment (`tokens.open_self`):
    an excitation inside the open shell, a token with its own hole among them. Across
    fragments the same kind of cell is a recombination (`somo_pair`), not masked."""

    def roles(tokens, value):
        return np.array([o.spin_role == value for o in tokens], dtype=bool)

    occ_f = np.repeat(np.arange(len(fragments)), [len(t.occupied()) for t in fragments])
    virt_f = np.repeat(np.arange(len(fragments)), [len(t.virtual()) for t in fragments])
    rows = roles([o for t in fragments for o in t.occupied()], 1)
    cols = roles([o for t in fragments for o in t.virtual()], -1)
    return rows[:, None] & cols[None, :] & (occ_f[:, None] == virt_f[None, :])


def valid_pairs(fragments: list[MoleculeTokens], config: CandidateConfig | None = None):
    """(n_occ, n_virt) arrow-candidate filter: a donor (doubly occupied, or an open-shell token
    in its donor role), an acceptor (empty and not d-type unless allowed, or a hole), and,
    within one fragment, not the donor's own antibond, not a pair whose every site pair is a
    bond interacting with itself (sites identical or bonded, and the acceptor carries >=
    `config.site_weight` on the donor's site or the donor on the acceptor's:
    `tokens.site_excluded_matrix`) and not an open-shell row with a hole column
    (`open_self_cells`). Every criterion is read from descriptors and the record geometry
    (`ksmech.orbitals.tokens`). Since v18 it is the legacy mask of `ModelConfig.pair_cells =
    'valid'`: the store keeps every cell and `pair_flags`."""
    config = config or CandidateConfig()
    donors = {id(o) for t in fragments for o in t.donor_candidates(config)}
    acceptors = {id(o) for t in fragments for o in t.acceptor_candidates(config)}
    occ = [o for t in fragments for o in t.occupied()]
    virt = [o for t in fragments for o in t.virtual()]
    ok_d = np.array([id(o) in donors for o in occ], dtype=bool)
    ok_a = np.array([id(o) in acceptors for o in virt], dtype=bool)
    own, site = _fragment_blocks(fragments, config)
    return ok_d[:, None] & ok_a[None, :] & ~(own | site | open_self_cells(fragments))


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
