"""Ranking baselines with no neural network (Prompt 5): a logistic regression and a
gradient-boosted ranker that score the candidate pairs of each step's first arrow.

Features of a candidate pair (donor r, acceptor s), no type one-hot (`FEATURE_NAMES`, 61):

- the 11 pair-track features z_rs at the physical K and eps (`physics.pairs.PAIR_FEATURES`,
  `ranking.physical_pair_z`; log beta^2 at three distances, log|G|, log|KS3|, KS2, log|KS1|,
  the fragment flag, centroid distance, cross-centre coupling, CLASH);
- the 23 continuous descriptors of r and of s (the model inputs `orbital_features`);
- the frozen-orbital f- on r's main centre and f+ on s's main centre;
- the diagnostic LMO energies E_r and E_s.

Targets as in `ksmech.physics.ranking`: the positives are the labeled pairs of the first arrow;
the labeled pairs of the step's other arrows are left out of training (neither positive nor
negative), never out of the evaluation. Steps whose first arrow has no valid labeled pair are
not trained on and count as misses.

- `LogisticRanker`: pointwise logistic regression on asinh-scaled, standardized features, each
  step weighted equally (its positives sum to 1, its negatives sum to 1), L2 strength chosen on
  the validation split by top-5.
- `GBDTRanker`: gradient-boosted regression trees in numpy (histogram splits on at most 64
  quantile bins, depth 4, Newton leaves with L2 1.0, shrinkage 0.1, half the rows per tree) on
  the listwise loss -log sum_{i in P} softmax(scores)_i of each step (multi-label marginalized),
  early stopping on validation top-5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ksmech.arrows.reactants import TokenLoader
from ksmech.arrows.steps import MechanismSample
from ksmech.data import provenance as prov
from ksmech.orbitals.descriptors import feature_names
from ksmech.orbitals.tokens import atom_feature_names
from ksmech.physics.pairs import PAIR_FEATURES, frozen_head
from ksmech.physics.ranking import (
    first_label,
    label_pairs,
    label_rank,
    physical_pair_z,
    summarize_ranks,
)
from ksmech.physics.stage_a import STAGE_A_FEATURES

TOKEN_FEATURES = (
    [f"donor_{n}" for n in feature_names()]
    + [f"acceptor_{n}" for n in feature_names()]
    + ["donor_site_f_minus", "acceptor_site_f_plus", "donor_energy_ev", "acceptor_energy_ev"]
)
FEATURE_NAMES = list(PAIR_FEATURES) + TOKEN_FEATURES
# the geometry-free ranker (`physics.lgbm`): the stage-A pair features, NaN where undefined
GEOMFREE_FEATURE_NAMES = list(STAGE_A_FEATURES) + TOKEN_FEATURES
PAIR_SETS = {"legacy": FEATURE_NAMES, "stage_a": GEOMFREE_FEATURE_NAMES}
_COL = {name: k for k, name in enumerate(FEATURE_NAMES)}


@dataclass
class CandidateSet:
    """The valid candidate pairs of one step and everything needed to rank them."""

    step_id: str
    X: np.ndarray  # (n, n_features)
    rows: np.ndarray  # (n,) occupied row of the pair block
    cols: np.ndarray  # (n,) virtual column
    positive: np.ndarray  # (n,) a labeled pair of the first arrow
    ignore: np.ndarray  # (n,) a labeled pair of another arrow only (left out of training)
    gap_sign: np.ndarray  # (n,) sign of G (the logs drop it; the untrained floor needs it)
    defined: np.ndarray  # (n, n_pair_features) which pair features are defined
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


def _pair_columns(sample: MechanismSample, pairs: str) -> np.ndarray:
    """(n_occ, n_virt, F) pair block of the ranker: the legacy contact features at the
    physical K and eps, or the stage-A features as stored with NaN where undefined."""
    if pairs == "legacy":
        return physical_pair_z(sample)
    if pairs != "stage_a" or tuple(sample.pair_features) != STAGE_A_FEATURES:
        raise ValueError(f"{sample.step_id}: pair set {pairs!r} over {sample.pair_features[:2]}")
    return np.where(sample.pair_defined, sample.pair_z.astype(np.float64), np.nan)


def candidate_set(
    sample: MechanismSample, loader: TokenLoader, pairs: str = "legacy"
) -> CandidateSet:
    """The valid pairs of `sample` with the columns of `PAIR_SETS[pairs]`."""
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
            _pair_columns(sample, pairs)[rows, cols],
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
        sample.pair_gap_sign[rows, cols].astype(float),
        sample.pair_defined[rows, cols],
        sample.pair_valid,
        first,
    )


def untrained_scores(cs: CandidateSet) -> dict[str, np.ndarray]:
    """The untrained KS scores of candidate pairs, from their own feature rows (signed KS3 and
    G restored with the stored sign of G), the frozen head's logits among them."""
    X = cs.X
    fbar = 0.5 * (X[:, _COL["donor_energy_ev"]] + X[:, _COL["acceptor_energy_ev"]])
    G = cs.gap_sign * np.exp(X[:, _COL["log_abs_g"]])
    ks3 = cs.gap_sign * np.exp(X[:, _COL["log_abs_ks3"]])
    ks2 = X[:, _COL["ks2"]]
    out = {
        "ks3+ks2": -(ks3 + ks2),
        "ks3": -ks3,
        "fbar2_g": -(fbar**2) * G,
        "fbar2_g+ks2": -((fbar**2) * G + ks2),
        "gap_kernel": -G,
        "frozen_head": frozen_head(X[:, : len(PAIR_FEATURES)], cs.gap_sign, cs.defined),
    }
    return out


def gap_score(cs: CandidateSet, pairs: str = "legacy") -> np.ndarray:
    """-G, the untrained "G alone" floor, from either pair set (the legacy logs drop the sign
    of G, restored from `gap_sign`; stage A stores G itself)."""
    if pairs == "stage_a":
        return -cs.X[:, GEOMFREE_FEATURE_NAMES.index("g")]
    return -cs.gap_sign * np.exp(cs.X[:, _COL["log_abs_g"]])


def evaluate_scores(sets: list[CandidateSet], scores: list[np.ndarray]) -> dict:
    ranks = [cs.rank(s) for cs, s in zip(sets, scores, strict=True)]
    return {"ranks": ranks, **summarize_ranks(ranks)}


def _trainable(sets: list[CandidateSet]) -> list[CandidateSet]:
    return [cs for cs in sets if cs.positive.any()]


# ---- (a) logistic regression


@dataclass
class LogisticRanker:
    scale: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray  # (n_features + 1,), the bias last
    l2: float = 0.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.arcsinh(X / self.scale) - self.mean) / self.std

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X) @ self.weights[:-1] + self.weights[-1]

    @classmethod
    def prepare(cls, train: list[CandidateSet]) -> LogisticRanker:
        X = np.concatenate([cs.X for cs in train])
        absx = np.abs(X)
        scale = np.array(
            [np.median(c[c > 0]) if (c > 0).any() else 1.0 for c in absx.T], dtype=float
        )
        T = np.arcsinh(X / scale)
        std = T.std(axis=0)
        return cls(scale, T.mean(axis=0), np.where(std > 0, std, 1.0), np.zeros(X.shape[1] + 1))


def _logistic_data(model: LogisticRanker, sets: list[CandidateSet]):
    Z, y, w = [], [], []
    for cs in _trainable(sets):
        keep = ~cs.ignore
        pos = cs.positive[keep]
        Z.append(model.transform(cs.X[keep]))
        y.append(pos.astype(float))
        w.append(np.where(pos, 1.0 / pos.sum(), 1.0 / max(1, (~pos).sum())))
    return np.concatenate(Z), np.concatenate(y), np.concatenate(w)


def fit_logistic(train: list[CandidateSet], l2: float) -> LogisticRanker:
    model = LogisticRanker.prepare(train)
    Z, y, w = _logistic_data(model, train)
    Z1 = np.column_stack([Z, np.ones(len(Z))])

    def loss(beta):
        s = Z1 @ beta
        value = np.sum(w * (np.logaddexp(0.0, s) - y * s)) + 0.5 * l2 * beta[:-1] @ beta[:-1]
        grad = Z1.T @ (w * (1.0 / (1.0 + np.exp(-s)) - y))
        grad[:-1] += l2 * beta[:-1]
        return value, grad

    res = minimize(loss, model.weights, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    model.weights, model.l2 = res.x, l2
    return model


def select_logistic(train, val, grid=(0.01, 0.1, 1.0, 10.0)) -> tuple[LogisticRanker, dict]:
    """The L2 strength with the best validation top-5 (ties: top-1, then median rank)."""
    best, best_key, scan = None, None, {}
    for l2 in grid:
        model = fit_logistic(train, l2)
        r = evaluate_scores(val, [model.score(cs.X) for cs in val])
        scan[l2] = {k: v for k, v in r.items() if k != "ranks"}
        key = (r["top5"], r["top1"], -r["median_rank"])
        if best_key is None or key > best_key:
            best, best_key = model, key
    return best, scan


# ---- (b) gradient-boosted listwise ranker


@dataclass
class Tree:
    feature: np.ndarray  # (n_nodes,) split feature, -1 for a leaf
    threshold: np.ndarray  # (n_nodes,) go left when bin <= threshold
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray  # leaf value

    def predict(self, B: np.ndarray) -> np.ndarray:
        node = np.zeros(len(B), dtype=int)
        while True:
            f = self.feature[node]
            inner = f >= 0
            if not inner.any():
                return self.value[node]
            go_left = B[np.flatnonzero(inner), f[inner]] <= self.threshold[node[inner]]
            node[inner] = np.where(go_left, self.left[node[inner]], self.right[node[inner]])


@dataclass
class TreeConfig:
    depth: int = 4
    n_bins: int = 64
    l2: float = 1.0
    min_hess: float = 1e-3
    learning_rate: float = 0.1
    subsample: float = 0.5
    max_rounds: int = 400
    patience: int = 60
    check_every: int = 10
    seed: int = 0


def quantile_edges(X: np.ndarray, n_bins: int) -> list[np.ndarray]:
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [np.unique(np.quantile(col, qs)) for col in X.T]


def binned(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [np.searchsorted(e, col, side="right") for e, col in zip(edges, X.T, strict=True)]
    ).astype(np.uint8)


def _best_splits(G, H, l2: float, min_hess: float):
    """Per node: (gain, feature, threshold) of the best split from (n_nodes, F, NB) histograms."""
    g_left, h_left = np.cumsum(G, axis=2)[..., :-1], np.cumsum(H, axis=2)[..., :-1]
    g_all, h_all = G.sum(axis=2, keepdims=True), H.sum(axis=2, keepdims=True)
    g_right, h_right = g_all - g_left, h_all - h_left
    gain = g_left**2 / (h_left + l2) + g_right**2 / (h_right + l2) - g_all**2 / (h_all + l2)
    gain = np.where((h_left >= min_hess) & (h_right >= min_hess), gain, -np.inf)
    flat = gain.reshape(len(G), -1)
    best = np.argmax(flat, axis=1)
    nb = G.shape[2] - 1
    return flat[np.arange(len(G)), best], best // nb, best % nb


def fit_tree(B: np.ndarray, g: np.ndarray, h: np.ndarray, cfg: TreeConfig) -> Tree:
    """Level-wise regression tree on binned features with Newton leaf values."""
    n, F = B.shape
    NB = cfg.n_bins
    base = (np.arange(F) * NB)[None, :] + B
    feature, threshold, left, right, value = [-1], [0], [-1], [-1], [0.0]
    node = np.zeros(n, dtype=int)
    frontier = [0]
    for _ in range(cfg.depth):
        local = {nd: k for k, nd in enumerate(frontier)}
        slot = np.array([local.get(x, -1) for x in range(len(feature))])[node]
        active = slot >= 0
        idx = (slot[active, None] * F * NB + base[active]).ravel()
        size = len(frontier) * F * NB
        G = np.bincount(idx, np.repeat(g[active], F), size).reshape(len(frontier), F, NB)
        H = np.bincount(idx, np.repeat(h[active], F), size).reshape(len(frontier), F, NB)
        gains, feats, ths = _best_splits(G, H, cfg.l2, cfg.min_hess)
        new_frontier = []
        for k, nd in enumerate(frontier):
            Gt, Ht = G[k, 0].sum(), H[k, 0].sum()
            value[nd] = -Gt / (Ht + cfg.l2)
            if not np.isfinite(gains[k]) or gains[k] <= 1e-12:
                continue
            feature[nd], threshold[nd] = int(feats[k]), int(ths[k])
            for side in (left, right):
                side[nd] = len(feature)
                feature.append(-1), threshold.append(0), left.append(-1), right.append(-1)
                value.append(0.0)
                new_frontier.append(side[nd])
            mine = node == nd
            go_left = B[mine, feature[nd]] <= threshold[nd]
            node[np.flatnonzero(mine)] = np.where(go_left, left[nd], right[nd])
        frontier = new_frontier
        if not frontier:
            break
    for nd in frontier:
        mine = node == nd
        value[nd] = -g[mine].sum() / (h[mine].sum() + cfg.l2)
    return Tree(*(np.array(a) for a in (feature, threshold, left, right, value)))


def _softmax_grad(scores, positive, starts, sizes):
    """Gradient and diagonal Hessian of -log sum_P softmax, per contiguous group."""
    group = np.repeat(np.arange(len(sizes)), sizes)
    peak = np.maximum.reduceat(scores, starts)[group]
    e = np.exp(scores - peak)
    p = e / np.add.reduceat(e, starts)[group]
    ep = np.where(positive, e, 0.0)
    q = ep / np.add.reduceat(ep, starts)[group]
    return p - q, np.maximum(p * (1.0 - p), 1e-6)


@dataclass
class GBDTRanker:
    edges: list[np.ndarray]
    trees: list[Tree] = field(default_factory=list)
    learning_rate: float = 0.1
    history: list[dict] = field(default_factory=list)

    def score(self, X: np.ndarray) -> np.ndarray:
        B = binned(X, self.edges)
        out = np.zeros(len(X))
        for tree in self.trees:
            out += self.learning_rate * tree.predict(B)
        return out


def _stack(sets: list[CandidateSet], edges):
    sets = _trainable(sets)
    keep = [~cs.ignore for cs in sets]
    B = np.concatenate([binned(cs.X[k], edges) for cs, k in zip(sets, keep, strict=True)])
    pos = np.concatenate([cs.positive[k] for cs, k in zip(sets, keep, strict=True)])
    sizes = np.array([int(k.sum()) for k in keep])
    return B, pos, sizes, np.concatenate([[0], np.cumsum(sizes)[:-1]])


def fit_gbdt(train: list[CandidateSet], val: list[CandidateSet], cfg: TreeConfig) -> GBDTRanker:
    edges = quantile_edges(np.concatenate([cs.X for cs in train]), cfg.n_bins)
    B, pos, sizes, starts = _stack(train, edges)
    B_val = [binned(cs.X, edges) for cs in val]
    model = GBDTRanker(edges, learning_rate=cfg.learning_rate)
    rng = np.random.default_rng(cfg.seed)
    scores, val_scores = np.zeros(len(B)), [np.zeros(len(b)) for b in B_val]
    best_key, best_n = None, 0
    for k in range(cfg.max_rounds):
        g, h = _softmax_grad(scores, pos, starts, sizes)
        rows = rng.random(len(B)) < cfg.subsample
        tree = fit_tree(B[rows], g[rows], h[rows], cfg)
        model.trees.append(tree)
        scores += cfg.learning_rate * tree.predict(B)
        val_scores = [
            v + cfg.learning_rate * tree.predict(b) for v, b in zip(val_scores, B_val, strict=True)
        ]
        if (k + 1) % cfg.check_every:
            continue
        r = evaluate_scores(val, val_scores)
        key = (r["top5"], r["top1"], -r["median_rank"])
        model.history.append({"rounds": k + 1, **{x: r[x] for x in ("top1", "top3", "top5")}})
        if best_key is None or key > best_key:
            best_key, best_n = key, k + 1
        elif k + 1 - best_n >= cfg.patience:
            break
    model.trees = model.trees[:best_n]
    return model


# ---- persistence


BASELINE_VERSION = 1


def _save_model(
    path: Path, arrays: dict, kind: str, inputs: dict[str, str] | None, extra: dict | None = None
) -> Path:
    """Write a fitted ranker with a provenance header: its inputs are the sample files it was
    fitted and selected on, its config the feature list; `extra` notes, e.g. which textbook
    reactant pools its training data contains."""
    header = prov.stamp(
        "baseline",
        BASELINE_VERSION,
        inputs or {},
        {"model": kind, "features": FEATURE_NAMES},
        prov.array_digest(arrays),
        extra,
    )
    return prov.save_npz(path, arrays, header, compressed=False)


def save_gbdt(
    model: GBDTRanker, path: Path, inputs: dict[str, str] | None = None, extra: dict | None = None
) -> Path:
    arrays = {f"edge{i}": e for i, e in enumerate(model.edges)}
    for t, tree in enumerate(model.trees):
        for name in ("feature", "threshold", "left", "right", "value"):
            arrays[f"t{t}_{name}"] = getattr(tree, name)
    meta = {"n_trees": len(model.trees), "lr": model.learning_rate, "features": FEATURE_NAMES}
    arrays["meta"] = np.array(json.dumps(meta))
    return _save_model(Path(path), arrays, "gbdt", inputs, extra)


def save_logistic(model: LogisticRanker, path: Path, inputs: dict[str, str] | None = None):
    arrays = {k: np.asarray(getattr(model, k)) for k in ("scale", "mean", "std", "weights", "l2")}
    return _save_model(Path(path), arrays, "logistic", inputs)


def load_gbdt(path: Path, allow_stale: bool = False) -> GBDTRanker:
    """The fitted GBDT, after verifying that its sample files and features are unchanged
    (StaleArtifact otherwise; `allow_stale` only logs it, for diagnostics)."""
    config = {"model": "gbdt", "features": FEATURE_NAMES}
    prov.verify_clean(path, "baseline", BASELINE_VERSION, config, allow_stale=allow_stale)
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta["features"] != FEATURE_NAMES:
            raise ValueError(f"{path}: features differ from FEATURE_NAMES; re-run the baseline")
        edges = [z[f"edge{i}"] for i in range(len(FEATURE_NAMES))]
        names = ("feature", "threshold", "left", "right", "value")
        trees = [Tree(*(z[f"t{t}_{n}"] for n in names)) for t in range(meta["n_trees"])]
    return GBDTRanker(edges, trees, meta["lr"])
