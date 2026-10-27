"""contracts.py: value-type round trips, log_softmax_valid, LRU, ChildScan, LocalJobContext."""

import json
import threading

import numpy as np
import pytest

from ksmech.serve.core.contracts import (
    LRU,
    ExpandRequest,
    Expansion,
    Limits,
    LocalJobContext,
    PoolState,
    PredictorInfo,
    PredictorRef,
    ProposedChild,
    SpeciesCount,
    log_softmax_valid,
    to_jsonable,
)
from tests.test_serve_fixtures import no_xtb  # noqa: F401 (fixture)


def _state() -> PoolState:
    species = (
        SpeciesCount("CCl", ("CCl",), 1, 0, "OK"),
        SpeciesCount("C[O-]", ("C[O-]",), 2, -1, "OK"),
    )
    return PoolState(
        "s" + "a" * 20,
        "[C:1][O-:2]",
        "CCl.C[O-].C[O-]",
        "p" + "b" * 20,
        species,
        ("C[O-]", "CCl", "C[O-]"),
        15,
    )


def _child(weight: float = -0.5, logp: float | None = None) -> ProposedChild:
    return ProposedChild(
        state=_state(),
        step={"matrix": {"bonds": [[1, 5, 1], [5, 6, -1]], "lone": [[1, -2], [6, 2]]}},
        score=0.75,
        logp=logp,
        weight=weight,
        rank=1,
        n_alternatives=2,
        alternatives=[{"score": 0.5, "logp": None, "rank": 3, "matrix": {}}],
    )


def _roundtrip(obj):
    d = json.loads(json.dumps(obj.to_dict(), allow_nan=False))  # strictly JSON-able
    return type(obj).from_dict(d)


INFO = PredictorInfo(
    "gbdt",
    "GBDT",
    "gbdt",
    False,
    "GBDT_MISSING",
    "no file",
    None,
    None,
    "rank score",
    1.0,
    ("scores rank candidates",),
    "a ranker",
    False,
)

CASES = [
    SpeciesCount("C=C[O-]", ("C=C[O-]", "[CH2-]C=O"), 1, -1, "OK"),
    _state(),
    ExpandRequest("floor", 3, None, "all", 64),
    ExpandRequest("gbdt", 2, 0.5, "inter", 64),
    Limits(6, 4, 150, 10, 12, 200),
    _child(),
    Expansion(
        True,
        None,
        None,
        "",
        [_child(), _child(-1.2, -0.3)],
        98,
        7,
        0.9,
        0.1,
        {"APPLY_FAILED:UNKNOWN_ARROW": 2},
        [{"rank": 3, "code": "X"}],
        False,
        ["w"],
    ),
    Expansion.failed("FEATURIZATION_FAILED", "scf", upstream="SCF_NOT_CONVERGED"),
    INFO,
    PredictorRef("floor", "floor/g-atom/abc", None, 1.0),
]


@pytest.mark.parametrize("obj", CASES, ids=lambda o: type(o).__name__)
def test_roundtrip(obj):
    assert _roundtrip(obj) == obj


def test_tuples_become_lists_and_nonfinite_become_none():
    d = _child(weight=float("-inf")).to_dict()
    assert isinstance(d["state"]["species"], list) and isinstance(d["state"]["species"][0], dict)
    assert d["weight"] is None and d["logp"] is None
    back = ProposedChild.from_dict(json.loads(json.dumps(d)))
    assert back.weight == float("-inf") and back.logp is None


def test_to_jsonable_numpy():
    out = to_jsonable(
        {
            "a": np.float64(1.5),
            "b": np.int64(3),
            "c": np.array([1.0, np.nan]),
            "d": np.bool_(True),
            "e": (1, 2),
            "f": {3, 1},
        }
    )
    assert out == {"a": 1.5, "b": 3, "c": [1.0, None], "d": True, "e": [1, 2], "f": [1, 3]}
    assert type(out["b"]) is int and type(out["d"]) is bool
    with pytest.raises(TypeError):
        to_jsonable(object())


def test_expansion_failed():
    ex = Expansion.failed("ANALYSIS_FAILED", "boom")
    assert not ex.ok and ex.code == "ANALYSIS_FAILED" and ex.detail == "boom"
    assert ex.children == [] and ex.n_valid == 0 and ex.failures == {}


# ---- log_softmax_valid


def test_log_softmax_matches_hand_softmax():
    scores = np.array([[1.0, 2.0, -np.inf], [0.5, 3.0, 0.0]])
    valid = np.array([[True, True, False], [True, True, False]])
    out = log_softmax_valid(scores, valid, 1.0)
    v = np.array([1.0, 2.0, 0.5, 3.0])
    p = np.exp(v) / np.exp(v).sum()
    assert np.allclose(np.exp(out[valid]), p, rtol=1e-14)
    assert np.isneginf(out[~valid]).all()
    assert np.isclose(np.exp(out[valid]).sum(), 1.0, rtol=1e-14)
    assert out.dtype == np.float64


def test_log_softmax_temperature():
    scores = np.array([0.0, 1.0, 2.0])
    valid = np.ones(3, dtype=bool)
    for t in (0.25, 1.0, 4.0):
        p = np.exp(scores / t) / np.exp(scores / t).sum()
        assert np.allclose(np.exp(log_softmax_valid(scores, valid, t)), p, rtol=1e-13)
    sharp = np.exp(log_softmax_valid(scores, valid, 0.01))
    assert sharp[2] > 0.999


def test_log_softmax_large_scores_stable():
    out = log_softmax_valid(np.array([1000.0, 1001.0]), np.array([True, True]), 1.0)
    assert np.isfinite(out).all()
    assert np.isclose(np.exp(out).sum(), 1.0)


def test_log_softmax_all_invalid_and_errors():
    out = log_softmax_valid(np.zeros((2, 2)), np.zeros((2, 2), dtype=bool), 1.0)
    assert np.isneginf(out).all()
    for t in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            log_softmax_valid(np.zeros(2), np.ones(2, dtype=bool), t)
    with pytest.raises(ValueError):
        log_softmax_valid(np.array([np.nan, 0.0]), np.ones(2, dtype=bool), 1.0)
    with pytest.raises(ValueError):
        log_softmax_valid(np.zeros(2), np.ones(3, dtype=bool), 1.0)
    assert np.isneginf(log_softmax_valid(np.full(2, -np.inf), np.ones(2, dtype=bool), 1.0)).all()


# ---- LRU


def test_lru():
    lru: LRU[str, int] = LRU(2)
    assert lru.get("a") is None
    lru.put("a", 1)
    lru.put("b", 2)
    assert lru.get("a") == 1  # a is now the most recent
    lru.put("c", 3)  # evicts b
    assert lru.get("b") is None and lru.get("a") == 1 and lru.get("c") == 3
    assert len(lru) == 2 and "a" in lru
    lru.put("a", 10)
    assert lru.get("a") == 10
    assert lru.pop("a") == 10 and "a" not in lru
    lru.clear()
    assert len(lru) == 0
    with pytest.raises(ValueError):
        LRU(0)


def test_lru_threads():
    lru: LRU[int, int] = LRU(64)

    def work(base):
        for i in range(500):
            lru.put(base + i % 100, i)
            lru.get(base + (i * 7) % 100)

    threads = [threading.Thread(target=work, args=(k * 1000,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(lru) == 64


# ---- ChildScan


# ---- LocalJobContext


def test_local_job_context(caplog):
    ctx = LocalJobContext("j" + "0" * 16, settings=None, scheduler=None)
    ctx.progress(phase="featurize", done=1, total=3)
    ctx.progress(phase="expand")
    assert ctx.progress_log == [{"phase": "featurize", "done": 1, "total": 3}, {"phase": "expand"}]
    assert ctx.cancelled() is False
    ctx.cancel_flag = True
    assert ctx.cancelled() is True
    with caplog.at_level("WARNING"):
        ctx.skip("APPLY_FAILED:UNKNOWN_ARROW", "no bond", node="p1")
        ctx.skip("APPLY_FAILED:UNKNOWN_ARROW")
    assert ctx.skipped == {"APPLY_FAILED:UNKNOWN_ARROW": 2}
    assert ctx.skip_log[0] == {
        "code": "APPLY_FAILED:UNKNOWN_ARROW",
        "detail": "no bond",
        "node": "p1",
    }
    assert "APPLY_FAILED:UNKNOWN_ARROW" in caplog.text
