"""contracts.py: value-type round trips, log_softmax_valid, LRU, ChildScan, LocalJobContext."""

import json
import math
import threading

import numpy as np
import pytest

from ksmech.serve.contracts import (
    LRU,
    ArrowSpec,
    ChildScan,
    ExpandRequest,
    Expansion,
    Limits,
    LocalJobContext,
    PoolState,
    PredictorInfo,
    PredictorRef,
    ProposedChild,
    ProposedStep,
    SpeciesCount,
    StepOutcome,
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


def _child(logp: float = -0.5) -> ProposedChild:
    return ProposedChild(
        state=_state(),
        arrows=[{"donor": 16, "physics": {"ks3": -0.1, "flags": ["synthetic"]}}],
        encoded={"code": "a=b;b,d=d"},
        encode_reason=None,
        touched_maps=[2, 6, 7],
        score=0.75,
        probability=0.6,
        logp=logp,
        rank=1,
        n_alternatives=2,
        alternatives=[{"score": 0.5, "probability": 0.2, "rank": 3, "arrows": []}],
    )


def _roundtrip(obj):
    d = json.loads(json.dumps(obj.to_dict(), allow_nan=False))  # strictly JSON-able
    return type(obj).from_dict(d)


INFO = PredictorInfo(
    "gbdt",
    "GBDT",
    "baseline",
    False,
    "GBDT_MISSING",
    "no file",
    None,
    None,
    "logit",
    1.0,
    ("trained on sample-v5",),
    "a ranker",
)

CASES = [
    SpeciesCount("C=C[O-]", ("C=C[O-]", "[CH2-]C=O"), 1, -1, "OK"),
    _state(),
    ArrowSpec(16, 1, 28, 5),
    ExpandRequest("floor", 3, None, "all", 64),
    ExpandRequest("stub", 2, 0.5, "inter", 64),
    Limits(6, 4, 150, 10, 12, 200),
    _child(),
    Expansion(
        True,
        None,
        None,
        "",
        [_child(), _child(-1.2)],
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
    StepOutcome(_state(), [{"donor": 1}], None, "no base", [1, 2], [{"rank": 4}], [{"code": "W"}]),
    INFO,
    PredictorRef("floor", "floor/ks3+ks2/d2.3/abc", None, 1.0),
    ProposedStep((ArrowSpec(16, 1, 28, 5),), ("rule",), 50.4, -0.01, 1, (6, 4)),
    ProposedStep(
        (ArrowSpec(1, 2, 3, 4), ArrowSpec(5, 6, 7, 8)), ("user", "user"), 1.0, -2.0, 2, None
    ),
]


@pytest.mark.parametrize("obj", CASES, ids=lambda o: type(o).__name__)
def test_roundtrip(obj):
    assert _roundtrip(obj) == obj


def test_tuples_become_lists_and_nonfinite_become_none():
    d = _child(logp=float("-inf")).to_dict()
    assert isinstance(d["state"]["species"], list) and isinstance(d["state"]["species"][0], dict)
    assert d["logp"] is None
    back = ProposedChild.from_dict(json.loads(json.dumps(d)))
    assert back.logp == float("-inf")
    step = ProposedStep(
        (ArrowSpec(1, 2, 3, 4),), ("contact",), float("nan"), float("-inf"), 1, None
    )
    s = step.to_dict()
    assert s["score"] is None and s["logp"] is None
    back = ProposedStep.from_dict(s)
    assert math.isnan(back.score) and back.logp == float("-inf")


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


def test_childscan_grouping_and_kept_order():
    scan = ChildScan(beam=2, max_scan=10)
    assert scan.add(0.25, "B") is True
    assert scan.add(0.25, "A") is True
    assert scan.add(0.125, None, "APPLY_FAILED:X") is False
    assert scan.add(0.125, "C") is True
    assert scan.add(0.0625, "C") is False  # second member of C
    assert scan.mass == {"B": 0.25, "A": 0.25, "C": 0.1875}
    assert scan.kept() == ["A", "B"]  # tie by key asc
    assert scan.ranked() == ["A", "B", "C"]
    assert scan.n_scanned == 5 and scan.scanned_mass == 0.8125
    assert scan.failures == {"APPLY_FAILED:X": 1}
    with pytest.raises(ValueError):
        scan.add(0.1, None)
    with pytest.raises(ValueError):
        scan.add(0.1, "A", "BOTH")


def test_childscan_exhausted_and_max_scan():
    scan = ChildScan(beam=3, max_scan=2)
    assert scan.should_stop(0) == "exhausted"
    scan.add(0.1, "A")
    assert scan.should_stop(5) is None
    scan.add(0.1, "B")
    assert scan.should_stop(5) == "max_scan"
    assert scan.should_stop(0) == "exhausted"  # exhausted is checked first


def test_childscan_bound_is_strict():
    """beam 1: after A = 0.5 the unscanned mass U = 0.5 equals p_1 - p_2 = 0.5 - 0: keep
    scanning; after B = 0.25, U = 0.25 == 0.5 - 0.25: keep scanning; after A += 0.125,
    U = 0.125 < 0.625 - 0.25: stop on the bound."""
    scan = ChildScan(beam=1, max_scan=100)
    scan.add(0.5, "A")
    assert scan.unscanned_mass == 0.5
    assert scan.should_stop(10) is None
    scan.add(0.25, "B")
    assert scan.unscanned_mass == 0.25
    assert scan.mass["A"] - scan.mass["B"] == 0.25
    assert scan.should_stop(10) is None
    scan.add(0.125, "A")
    assert scan.should_stop(10) == "bound"
    assert scan.kept() == ["A"]


def test_childscan_bound_needs_beam_children():
    scan = ChildScan(beam=2, max_scan=100)
    scan.add(0.75, "A")  # U = 0.25 < 0.75, but only one child exists
    assert scan.should_stop(3) is None
    scan.add(0.125, "B")  # U = 0.125 == p_2 - p_3 = 0.125 - 0: keep scanning
    assert scan.should_stop(3) is None
    scan.add(0.0625, None, "RESONANCE_ONLY_STEP")  # a failure still counts: U = 0.0625 < 0.125
    assert scan.should_stop(3) == "bound"


def test_childscan_lower_bound_property():
    """A kept child's mass is a lower bound: the true marginal is at most mass + U."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(40))
    children = rng.integers(0, 6, size=40).astype(str)
    order = np.argsort(-p, kind="stable")
    scan = ChildScan(beam=2, max_scan=40)
    for n_done, k in enumerate(order):
        if scan.should_stop(len(order) - n_done):
            break
        scan.add(float(p[k]), str(children[k]))
    true = {c: float(p[children == c].sum()) for c in set(children)}
    for c, m in scan.mass.items():
        assert m <= true[c] + 1e-12
        assert true[c] <= m + scan.unscanned_mass + 1e-12
    full = sorted(true, key=lambda c: (-true[c], c))[:2]
    assert set(scan.kept()) == set(full)  # the kept set is final once the bound stops the scan


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
