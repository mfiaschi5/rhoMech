"""Paired comparison of two arms of decode runs (open-shell plan P0b, step 4).

An arm is the per-sample hit files (`decode_hits_*.json`) of one configuration's seeds, decoded
on the same samples. The decision is one-sided: arm A (GENERAL) fails against arm B (LEGACY)
when its mean top-k is lower than B's by more than the tolerance max(2 x pooled SD, 1 point),
the pooled SD being the within-arm SD of every run (df = runs - 2). Beside it, the exact
McNemar test of H1 "A is worse": on the discordant samples, n_ab = A misses where B hits and
n_ba = A hits where B misses, p = P(X >= n_ab) for X ~ Binomial(n_ab + n_ba, 1/2). It is run on
the majority-of-seeds hits of each arm (a sample is a hit when more than half the arm's runs hit
it) and on every seed pair. Arms decoded on different samples, parts, cohorts, caps, beams or
decode versions are refused: the comparison would not be paired.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ksmech.data import provenance as prov

TOLERANCE_FLOOR = 1.0  # percentage points
PAIRED_KEYS = ("source", "split", "part", "cohort", "scope", "beam", "limit", "caps",
               "max_bonds", "max_atoms", "max_candidates")  # fmt: skip


def mcnemar_one_sided(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact one-sided McNemar test of 'a is worse than b' on paired 0/1 hit vectors."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("the hit vectors are not paired")
    n_ab, n_ba = int((~a & b).sum()), int((a & ~b).sum())
    n = n_ab + n_ba
    tail = sum(math.comb(n, k) for k in range(n_ab, n + 1))
    p = tail / 2**n if n else 1.0
    return {"a_miss_b_hit": n_ab, "a_hit_b_miss": n_ba, "p": float(p)}


def majority(hits: np.ndarray) -> np.ndarray:
    """(runs, samples) 0/1 -> a sample's hit when more than half the runs hit it."""
    hits = np.asarray(hits, dtype=int)
    return 2 * hits.sum(axis=0) > hits.shape[0]


def pooled_sd(arms: list[list[float]]) -> float:
    """The within-arm SD pooled over arms (df = runs - arms); 0 when df < 1."""
    ss = sum(((np.asarray(x) - np.mean(x)) ** 2).sum() for x in arms)
    df = sum(len(x) for x in arms) - len(arms)
    return float(math.sqrt(ss / df)) if df >= 1 else 0.0


def decide(top_a: list[float], top_b: list[float]) -> dict:
    """The one-sided rule on per-run top-k (percent): A fails when mean A < mean B - tolerance."""
    sd = pooled_sd([top_a, top_b])
    tol = max(2 * sd, TOLERANCE_FLOOR)
    mean_a, mean_b = float(np.mean(top_a)), float(np.mean(top_b))
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": mean_a - mean_b,
        "pooled_sd": sd,
        "tolerance": tol,
        "a_fails": bool(mean_a < mean_b - tol),
    }


def read_hits(path: Path) -> tuple[dict, dict[str, int | None]]:
    """(the decode configuration and version of a hit file, step id -> 0-based rank)."""
    header = prov.read_header(path)
    if header is None or header.kind != "decode_hits":
        raise ValueError(f"{path} is not a decode_hits file")
    rows = json.loads(Path(path).read_text())["per_sample"]
    config = {k: header.config.get(k) for k in PAIRED_KEYS} | {"version": header.schema_version}
    return config, {r["step_id"]: r["rank"] for r in rows}


def paired_hits(
    files: dict[str, list[Path]], k: int = 1
) -> tuple[list[str], dict[str, np.ndarray]]:
    """(the shared step ids, arm -> (runs, samples) hits at k), every file checked to be
    decoded the same way on the same samples."""
    configs, ranks = [], {}
    for arm, paths in files.items():
        ranks[arm] = []
        for path in paths:
            config, r = read_hits(path)
            configs.append((path, config))
            ranks[arm].append(r)
    first_path, first = configs[0]
    for path, config in configs[1:]:
        if diff := sorted(key for key in first if first[key] != config.get(key)):
            raise ValueError(f"{path} and {first_path} differ in {diff}: not paired")
    ids = sorted(next(iter(ranks.values()))[0])
    for arm, runs in ranks.items():
        if any(sorted(r) != ids for r in runs):
            raise ValueError(f"arm {arm}: a run was decoded on other samples")
    hits = {
        arm: np.array([[r[i] is not None and r[i] < k for i in ids] for r in runs])
        for arm, runs in ranks.items()
    }
    return ids, hits


def compare_arms(a: list[Path], b: list[Path], k: int = 1) -> dict:
    """The decision, the majority-of-seeds McNemar test and every seed pair's, per-run top-k."""
    ids, hits = paired_hits({"a": a, "b": b}, k)
    top = {arm: [100.0 * float(h.mean()) for h in runs] for arm, runs in hits.items()}
    pairs = [
        {"a": str(pa), "b": str(pb)} | mcnemar_one_sided(ha, hb)
        for pa, ha in zip(a, hits["a"], strict=True)
        for pb, hb in zip(b, hits["b"], strict=True)
    ]
    return {
        "k": k,
        "n": len(ids),
        "runs": {"a": [str(p) for p in a], "b": [str(p) for p in b]},
        "top_k": top,
        "decision": decide(top["a"], top["b"]),
        "majority": mcnemar_one_sided(majority(hits["a"]), majority(hits["b"])),
        "seed_pairs": pairs,
    }
