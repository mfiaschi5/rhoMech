"""The fast GBDT ranker: LightGBM lambdarank over the candidate pairs of each step's first
arrow (`physics.baseline.CandidateSet`), one query per step.

- Training rows per step: every positive, the `n_hard` non-positive pairs the untrained scorer
  ranks highest (the floor -(KS3 + KS2) on the legacy contact features, -G on the
  geometry-free ones) and `n_random` others drawn uniformly, from a generator seeded by
  (seed, step id), so the draw does not depend on the order steps are read in. The pairs of
  the step's other arrows are left out (`CandidateSet.ignore`), as in the numpy GBDT.
- Validation and test keep every valid pair. Early stopping reads the validation top-5 with
  the `label_rank` convention (non-labeled pairs scoring at least the best labeled one), top-1
  breaking ties, as the numpy GBDT does.
- The model is saved as LightGBM's text model in an npz with a provenance header (kind
  `baseline_lgbm`): the sample files, the features and the configuration.
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ksmech.data import provenance as prov
from ksmech.physics.baseline import CandidateSet

log = logging.getLogger(__name__)

LGBM_VERSION = 1
TIE_WEIGHT = 1e-3  # early-stopping key: top-5 + TIE_WEIGHT * top-1


@dataclass
class LGBMConfig:
    n_hard: int = 20
    n_random: int = 30
    seed: int = 0
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    truncation_level: int = 10
    max_rounds: int = 2000
    patience: int = 100
    n_jobs: int = 2

    def params(self) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "None",
            "lambdarank_truncation_level": self.truncation_level,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "num_threads": self.n_jobs,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }


def step_rng(seed: int, step_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(step_id.encode())])


def sample_negatives(cs: CandidateSet, score: np.ndarray, cfg: LGBMConfig) -> np.ndarray:
    """Sorted row indices kept for training: every positive, the `n_hard` best-scoring
    negatives and `n_random` of the remaining ones (ignored pairs never)."""
    pos = np.flatnonzero(cs.positive)
    neg = np.flatnonzero(~cs.positive & ~cs.ignore)
    order = neg[np.argsort(-score[neg], kind="stable")]
    hard, rest = order[: cfg.n_hard], order[cfg.n_hard :]
    rng = step_rng(cfg.seed, cs.step_id)
    drawn = rng.choice(rest, size=min(cfg.n_random, len(rest)), replace=False)
    return np.sort(np.concatenate([pos, hard, drawn]).astype(np.int64))


@dataclass
class TrainRows:
    """The sampled training rows, one query per step with a positive."""

    X: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)
    groups: list[int] = field(default_factory=list)
    n_untrainable: int = 0  # steps whose first arrow has no valid labeled pair

    def add(self, cs: CandidateSet, score: np.ndarray, cfg: LGBMConfig) -> None:
        if not cs.positive.any():
            self.n_untrainable += 1
            return
        keep = sample_negatives(cs, score, cfg)
        self.X.append(cs.X[keep].astype(np.float32))
        self.y.append(cs.positive[keep].astype(np.int32))
        self.groups.append(len(keep))

    def extend(self, other: TrainRows) -> None:
        """Append the rows of `other` (a chunk of a parallel read), in its order."""
        self.X += other.X
        self.y += other.y
        self.groups += other.groups
        self.n_untrainable += other.n_untrainable

    def dataset(self) -> lgb.Dataset:
        return lgb.Dataset(
            np.concatenate(self.X), np.concatenate(self.y), group=self.groups, free_raw_data=True
        )

    @property
    def n_rows(self) -> int:
        return int(sum(self.groups))


def group_ranks(scores: np.ndarray, positive: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Per group, the `label_rank` of its best positive (non-positives scoring at least as
    well); -1 when the group has no positive (a miss), empty groups included."""
    out = np.full(len(sizes), -1, dtype=np.int64)
    group = np.repeat(np.arange(len(sizes)), sizes)
    has = np.bincount(group[positive], minlength=len(sizes)) > 0
    if not has.any():
        return out
    best = np.full(len(sizes), -np.inf)
    np.maximum.at(best, group[positive], scores[positive])
    beats = (scores >= best[group]) & ~positive
    counts = np.bincount(group[beats], minlength=len(sizes))
    out[has] = counts[has]
    return out


def top_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean((ranks >= 0) & (ranks < k))) if len(ranks) else float("nan")


@dataclass
class EvalRows:
    """Every valid pair of every step of an evaluation part, concatenated."""

    X: np.ndarray
    positive: np.ndarray
    sizes: np.ndarray

    @classmethod
    def of(cls, sets: list[CandidateSet]) -> EvalRows:
        sizes = np.array([len(cs.X) for cs in sets], dtype=np.int64)
        width = sets[0].X.shape[1] if sets else 0
        X = np.concatenate([cs.X for cs in sets]) if sets else np.zeros((0, width))
        pos = np.concatenate([cs.positive for cs in sets]) if sets else np.zeros(0, bool)
        return cls(X.astype(np.float32), pos.astype(bool), sizes)


def _feval(rows: EvalRows):
    def top5(preds: np.ndarray, _data) -> tuple[str, float, bool]:
        ranks = group_ranks(np.asarray(preds), rows.positive, rows.sizes)
        return "top5", top_k(ranks, 5) + TIE_WEIGHT * top_k(ranks, 1), True

    return top5


@dataclass
class LGBMRanker:
    booster: lgb.Booster
    features: list[str]
    history: list[dict] = field(default_factory=list)

    def score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.booster.predict(np.asarray(X, dtype=np.float32))

    def score_sets(
        self, Xs: list[np.ndarray], n_jobs: int = 1, batch_rows: int = 1_000_000
    ) -> list[np.ndarray]:
        """`score` of every matrix in `Xs`, in batches of about `batch_rows` rows predicted on
        `n_jobs` threads: one call per batch, not one per step."""
        out: list[np.ndarray] = []
        start = 0
        while start < len(Xs):
            stop, rows = start, 0
            while stop < len(Xs) and (stop == start or rows + len(Xs[stop]) <= batch_rows):
                rows += len(Xs[stop])
                stop += 1
            batch = [np.asarray(X, dtype=np.float32) for X in Xs[start:stop]]
            flat = (
                self.booster.predict(np.concatenate(batch), num_threads=n_jobs)
                if rows
                else np.zeros(0)
            )
            out.extend(np.split(flat, np.cumsum([len(X) for X in batch])[:-1]))
            start = stop
        return out

    @property
    def n_trees(self) -> int:
        return self.booster.num_trees()


def fit_lgbm(train: TrainRows, val: EvalRows, features: list[str], cfg: LGBMConfig) -> LGBMRanker:
    """lambdarank on the sampled rows; early stopping on the validation top-5 (all
    `max_rounds`, with a warning, when the validation part has no candidate)."""
    train_ds = train.dataset()
    record: dict = {}
    stopping = {}
    if len(val.X):
        val_ds = lgb.Dataset(val.X, val.positive.astype(np.int32), reference=train_ds)
        stopping = {"valid_sets": [val_ds], "valid_names": ["val"], "feval": _feval(val)}
        stopping["callbacks"] = [
            lgb.early_stopping(cfg.patience, first_metric_only=True, verbose=False),
            lgb.record_evaluation(record),
        ]
    else:
        log.warning("no validation candidates: %d rounds without early stopping", cfg.max_rounds)
    booster = lgb.train(cfg.params(), train_ds, num_boost_round=cfg.max_rounds, **stopping)
    best = booster.best_iteration or booster.current_iteration()
    curve = record.get("val", {}).get("top5", [])
    history = [{"rounds": k + 1, "key": v} for k, v in enumerate(curve) if (k + 1) % 10 == 0]
    model = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))
    return LGBMRanker(model, list(features), history)


# ---- persistence


def model_config(features: list[str], cfg: LGBMConfig | None) -> dict:
    """The configuration a model is verified against; without `cfg` only the features (a
    scorer that just uses the model, whatever it was fitted with). The thread count is part of
    it: `deterministic` and `force_row_wise` fix the trees for one thread count, not across
    them (the scaffold geometry-free ranker grew 1,273 trees on 2 threads, 1,295 on 14)."""
    out = {"model": "lgbm_lambdarank", "features": list(features)}
    return out | ({"lgbm": asdict(cfg)} if cfg is not None else {})


def save_lgbm(
    model: LGBMRanker,
    path: Path,
    cfg: LGBMConfig,
    inputs: dict[str, str] | None = None,
    extra: dict | None = None,
) -> Path:
    arrays = {
        "model": np.array(model.booster.model_to_string()),
        "meta": np.array(json.dumps({"features": model.features, "history": model.history})),
    }
    header = prov.stamp(
        "baseline_lgbm",
        LGBM_VERSION,
        inputs or {},
        model_config(model.features, cfg),
        prov.array_digest(arrays),
        extra,
    )
    return prov.save_npz(Path(path), arrays, header, compressed=True)


def load_lgbm(
    path: Path, features: list[str], cfg: LGBMConfig | None = None, allow_stale: bool = False
) -> LGBMRanker:
    """The fitted ranker, after verifying its sample files, features and (given `cfg`) its
    configuration (StaleArtifact otherwise; `allow_stale` only logs it, for diagnostics)."""
    config = model_config(features, cfg)
    prov.verify_clean(path, "baseline_lgbm", LGBM_VERSION, config, allow_stale=allow_stale)
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        booster = lgb.Booster(model_str=str(z["model"]))
    if meta["features"] != list(features):
        raise ValueError(f"{path}: features differ; re-run `ksmech-physics gbdt`")
    return LGBMRanker(booster, meta["features"], meta["history"])
