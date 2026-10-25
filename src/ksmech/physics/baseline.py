"""The candidate pairs of each step's first arrow, ranked without a neural network: the
geometry-free features of every candidate, the untrained gap-kernel score and the rank
convention shared by the LightGBM ranker (`physics.lgbm`) and the textbook suite.

Candidates of a step: every (occupied, virtual) token pair of its reactant side that passes
`MechanismSample.pair_valid` (doubly occupied donor, empty acceptor, no d polarization function,
within a fragment not the donor's own antibond nor a pair whose every site pair is a bond
interacting with itself; read from descriptors and the record geometry). Target: the first
labeled arrow of the step (smallest `arrow_index`), whose labeled pairs are its donor set x
acceptor set. The labeled pairs of the step's other arrows are left out of training (neither
positive nor negative), never out of the evaluation. Metric: top-k accuracy under multi-label
marginalization: a hit when any labeled pair ranks among the k best candidates. Ties count
against the label (the rank is the number of other candidates scoring at least as well as the
best labeled pair). A step whose labeled pairs are all filtered out is a miss, counted as
`n_no_valid_label`.

Features of a candidate pair (donor r, acceptor s), no type one-hot (`FEATURE_NAMES`):

- the geometry-free pair features as stored (`physics.stage_a.STAGE_A_FEATURES`), NaN where
  undefined;
- the continuous descriptors of r and of s (the model inputs `orbital_features`);
- the frozen-orbital f- on r's main centre and f+ on s's main centre;
- the diagnostic LMO energies E_r and E_s.

The untrained score is -G (`gap_score`): the more negative the gap kernel, the stronger the
filled-empty interaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.arrows.align import ArrowLabel
from ksmech.arrows.reactants import TokenLoader
from ksmech.arrows.steps import MechanismSample
from ksmech.orbitals.descriptors import feature_names
from ksmech.orbitals.tokens import atom_feature_names
from ksmech.physics.stage_a import STAGE_A_FEATURES

TOP_K = (1, 3, 5)
TOKEN_FEATURES = (
    [f"donor_{n}" for n in feature_names()]
    + [f"acceptor_{n}" for n in feature_names()]
    + ["donor_site_f_minus", "acceptor_site_f_plus", "donor_energy_ev", "acceptor_energy_ev"]
)
FEATURE_NAMES = list(STAGE_A_FEATURES) + TOKEN_FEATURES


def first_label(sample: MechanismSample) -> ArrowLabel:
    return min(sample.labels, key=lambda lab: lab.arrow_index)


def orbital_positions(sample: MechanismSample) -> tuple[np.ndarray, np.ndarray]:
    """Sample orbital index -> row (occupied) or column (virtual) of the pair block, -1 else."""
    occ = sample.orbital_occupied
    row = np.full(sample.n_orbitals, -1)
    col = np.full(sample.n_orbitals, -1)
    row[occ] = np.arange(occ.sum())
    col[~occ] = np.arange((~occ).sum())
    return row, col


def label_pairs(sample: MechanismSample, label: ArrowLabel) -> np.ndarray:
    """(n_occ, n_virt) True on the labeled pairs (before the validity filter)."""
    row, col = orbital_positions(sample)
    out = np.zeros(sample.pair_valid.shape, dtype=bool)
    rows = [row[t - sample.n_atoms] for t in label.donor_ids if t >= sample.n_atoms]
    cols = [col[t - sample.n_atoms] for t in label.acceptor_ids if t >= sample.n_atoms]
    rows, cols = [r for r in rows if r >= 0], [c for c in cols if c >= 0]
    if rows and cols:
        out[np.ix_(rows, cols)] = True
    return out


def label_rank(scores: np.ndarray, valid: np.ndarray, positive: np.ndarray) -> int | None:
    """Number of valid non-labeled candidates scoring at least as well as the best valid
    labeled pair (0 = top-1); None when no labeled pair is valid."""
    pos = positive & valid
    if not pos.any():
        return None
    best = scores[pos].max()
    return int(((scores >= best) & valid & ~pos).sum())


def summarize_ranks(ranks: list[int | None], ks=TOP_K) -> dict:
    """Top-k accuracy (a missing label is a miss), median rank of the valid ones, counts."""
    valid = np.array([r for r in ranks if r is not None])
    nan = float("nan")
    out = {
        f"top{k}": float(np.mean([r is not None and r < k for r in ranks])) if ranks else nan
        for k in ks
    }
    out["median_rank"] = float(np.median(valid)) if len(valid) else float("nan")
    out["n"] = len(ranks)
    out["n_no_valid_label"] = sum(r is None for r in ranks)
    return out


@dataclass
class CandidateSet:
    """The valid candidate pairs of one step and everything needed to rank them."""

    step_id: str
    X: np.ndarray  # (n, n_features)
    rows: np.ndarray  # (n,) occupied row of the pair block
    cols: np.ndarray  # (n,) virtual column
    positive: np.ndarray  # (n,) a labeled pair of the first arrow
    ignore: np.ndarray  # (n,) a labeled pair of another arrow only (left out of training)
    valid: np.ndarray  # (n_occ, n_virt) the pair filter
    label: np.ndarray  # (n_occ, n_virt) the first arrow's labeled pairs, unfiltered

    def rank(self, scores: np.ndarray) -> int | None:
        matrix = np.full(self.valid.shape, -np.inf)
        matrix[self.rows, self.cols] = scores
        return label_rank(matrix, self.valid, self.label)


def main_centres(sample: MechanismSample) -> np.ndarray:
    """(n_orbitals,) atom index of each orbital's largest centre weight (from the incidence)."""
    out = np.full(sample.n_orbitals, -1)
    best = np.full(sample.n_orbitals, -np.inf)
    for j, g, w in sample.incidence:
        if w > best[int(j)]:
            best[int(j)], out[int(j)] = w, int(g)
    return out


def _pair_columns(sample: MechanismSample) -> np.ndarray:
    """(n_occ, n_virt, F) the geometry-free pair features as stored, NaN where undefined."""
    if tuple(sample.pair_features) != STAGE_A_FEATURES:
        raise ValueError(f"{sample.step_id}: pair features {sample.pair_features[:2]}...")
    return np.where(sample.pair_defined, sample.pair_z.astype(np.float64), np.nan)


def candidate_set(sample: MechanismSample, loader: TokenLoader) -> CandidateSet:
    """The valid pairs of `sample` with the columns of `FEATURE_NAMES`."""
    tokens = sample.tokens(loader)
    e_occ = np.array([o.energy_ev for t in tokens for o in t.occupied()])
    e_virt = np.array([o.energy_ev for t in tokens for o in t.virtual()])
    occ_idx = np.flatnonzero(sample.orbital_occupied)
    virt_idx = np.flatnonzero(~sample.orbital_occupied)
    rows, cols = np.nonzero(sample.pair_valid)
    main = main_centres(sample)
    names = atom_feature_names()
    f_minus = sample.atom_features[:, names.index("f_minus_fmo")]
    f_plus = sample.atom_features[:, names.index("f_plus_fmo")]
    desc = sample.orbital_features.astype(np.float64)
    X = np.column_stack(
        [
            _pair_columns(sample)[rows, cols],
            desc[occ_idx[rows]],
            desc[virt_idx[cols]],
            f_minus[main[occ_idx[rows]]],
            f_plus[main[virt_idx[cols]]],
            e_occ[rows],
            e_virt[cols],
        ]
    )
    first = label_pairs(sample, first_label(sample)) if sample.labels else sample.pair_valid & False
    others = np.zeros_like(first)
    for lab in sample.labels:
        others |= label_pairs(sample, lab)
    return CandidateSet(
        sample.step_id,
        X,
        rows,
        cols,
        first[rows, cols],
        (others & ~first)[rows, cols],
        sample.pair_valid,
        first,
    )


def gap_score(cs: CandidateSet) -> np.ndarray:
    """-G, the untrained "G alone" score (the stored features hold G itself)."""
    return -cs.X[:, FEATURE_NAMES.index("g")]


def evaluate_scores(sets: list[CandidateSet], scores: list[np.ndarray]) -> dict:
    ranks = [cs.rank(s) for cs, s in zip(sets, scores, strict=True)]
    return {"ranks": ranks, **summarize_ranks(ranks)}
