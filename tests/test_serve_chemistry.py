"""MolecularChemistry against a fake scheduler: the featurize prelude (order, dedup, cached
keys, failure codes) for expand and apply, WORKER_TIMEOUT / WORKER_CRASH, the persistence
check (CONSISTENCY_FAILED when a returned state file is missing) and the root preflight."""

import json

import pytest

from ksmech.serve.chemistry import MolecularChemistry
from ksmech.serve.contracts import (
    ExpandRequest,
    Expansion,
    LocalJobContext,
    ProposedChild,
)
from ksmech.serve.errors import ServeError
from ksmech.serve.featurize import SyntheticFeaturizer
from ksmech.serve.pools import SpeciesInput, initial_state, save_state, state_path
from ksmech.serve.worker import Priority
from tests.test_serve_fixtures_net import (  # noqa: F401 (fixtures)
    FakeScheduler,
    fail,
    no_xtb,
    ok,
    serve_settings,
)

pytestmark = pytest.mark.serve

JOB = "j" + "1" * 16
REQ = ExpandRequest("stub", 2, None, "all", 64)
FEATURIZE_CODES = (
    "FEATURIZATION_FAILED",
    "FEATURIZATION_DISABLED",
    "TOKENS_FAILED",
    "SYNTHETIC_FAILED",
    "WORKER_TIMEOUT",
    "WORKER_CRASH",
)


class _Loader:
    def __init__(self, inner, cached: set[str]):
        self.inner, self.cached = inner, cached

    def has(self, key: str) -> bool:
        return key in self.cached

    def __getattr__(self, name):
        return getattr(self.inner, name)


class PartlyCached:
    """A synthetic featurizer whose loader reports `cached` keys as present."""

    def __init__(self, settings, cached=()):
        self.inner = SyntheticFeaturizer(settings)
        self.mode, self.method = self.inner.mode, self.inner.method
        self._loader = _Loader(self.inner.loader(), set(cached))

    def loader(self):
        return self._loader

    def preflight(self, key):
        return self.inner.preflight(key)

    def occupations(self, key):
        return None


@pytest.fixture
def setup(serve_settings):  # noqa: F811
    """factory make(responses, cached=()) -> (chem, scheduler, ctx, settings)."""

    def make(responses=None, cached=()):
        s = serve_settings(chemistry="synthetic")
        sched = FakeScheduler(responses)
        chem = MolecularChemistry(s, sched, PartlyCached(s, cached))
        return chem, sched, LocalJobContext(JOB, s, sched), s

    return make


def sn2(settings, n_ccl=2):
    """MeO- + 2 MeCl (fragment keys C[O-], CCl, CCl), saved."""
    state = initial_state([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", n_ccl)], settings.limits)
    save_state(state, settings)
    return state


def product(settings, saved=True):
    state = initial_state([SpeciesInput("COC", 1), SpeciesInput("[Cl-]", 1)], settings.limits)
    if saved:
        save_state(state, settings)
    return state


def expansion(child_state) -> dict:
    child = ProposedChild(
        child_state, {"matrix": {"bonds": [], "lone": []}}, 0.5, None, -0.69, 1, 0, []
    )
    return Expansion(True, None, None, "", [child], 3, 2, 0.75, 0.25, {}, [], False, []).to_dict()


def test_featurize_loop_order_dedup_and_task_options(setup):
    chem, sched, ctx, s = setup()
    state = sn2(s)
    assert state.fragment_token_keys.count("CCl") == 2 and chem.mode == "synthetic"
    sched.responses = {"featurize": ok({"status": "built"}), "expand": ok(expansion(product(s)))}
    ex = chem.expand(state, REQ, ctx)
    assert ex.ok and [c.state.pool_key for c in ex.children] == ["[CH3][O][CH3]|0.[Cl]|-1"]
    assert sched.names() == ["featurize", "featurize", "expand"]
    keys = [c["payload"]["key"] for c in sched.calls[:2]]
    assert keys == list(dict.fromkeys(state.fragment_token_keys))  # fragment order, distinct
    for call in sched.calls[:2]:
        assert call["priority"] == Priority.JOB and call["owner"] == JOB
        assert call["timeout_s"] == s.qm_timeout_s + 120
        assert call["label"].startswith("synthetic EHT for ") and "atoms)" in call["label"]
    expand = sched.calls[2]
    assert expand["payload"] == {"state": state.to_dict(), "req": REQ.to_dict()}
    assert (expand["priority"], expand["timeout_s"], expand["owner"]) == (Priority.JOB, 900, JOB)
    json.dumps(expand["payload"])  # JSON-able across the worker boundary
    phases = [p["phase"] for p in ctx.progress_log]
    assert phases == ["featurize", "featurize", "expand"] and ctx.progress_log[0]["eta_s"] > 0


def test_cached_keys_are_not_featurized(setup):
    chem, sched, ctx, s = setup(cached={"CCl"})
    sched.responses = {"featurize": ok(), "expand": ok(expansion(product(s)))}
    assert chem.expand(sn2(s), REQ, ctx).ok
    assert [c["payload"].get("key") for c in sched.calls] == ["C[O-]", None]


@pytest.mark.parametrize("code", FEATURIZE_CODES)
def test_featurize_failure_codes(setup, code):
    chem, sched, ctx, s = setup()
    sched.responses = {"featurize": fail(code, "no record", upstream="SCF_FAILED")}
    ex = chem.expand(sn2(s), REQ, ctx)
    assert not ex.ok and (ex.code, ex.upstream) == (code, "SCF_FAILED")
    assert ex.children == [] and sched.names() == ["featurize"]  # the first failure stops


@pytest.mark.parametrize("code", ["WORKER_TIMEOUT", "WORKER_CRASH", "ANALYSIS_FAILED"])
def test_expand_task_failures_become_failed_expansions(setup, code):
    chem, sched, ctx, s = setup(cached={"C[O-]", "CCl"})
    sched.responses = {"expand": fail(code, "the worker died")}
    ex = chem.expand(sn2(s), REQ, ctx)
    assert not ex.ok and ex.code == code and ex.detail == "the worker died"
    assert sched.names() == ["expand"]


def test_expand_persistence_check(setup):
    chem, sched, ctx, s = setup(cached={"C[O-]", "CCl"})
    child = product(s, saved=False)
    sched.responses = {"expand": ok(expansion(child))}
    ex = chem.expand(sn2(s), REQ, ctx)
    assert not ex.ok and ex.code == "CONSISTENCY_FAILED"
    assert ex.detail == f"state not persisted: {child.state_id}"
    save_state(child, s)
    assert state_path(child.state_id, s).exists() and chem.expand(sn2(s), REQ, ctx).ok


def test_malformed_expand_result(setup):
    chem, sched, ctx, s = setup(cached={"C[O-]", "CCl"})
    sched.responses = {"expand": ok({"ok": True})}
    ex = chem.expand(sn2(s), REQ, ctx)
    assert ex.code == "CONSISTENCY_FAILED" and "malformed expand result" in ex.detail


def test_cancelled_before_featurizing(setup):
    chem, sched, ctx, s = setup()
    ctx.cancel_flag = True
    ex = chem.expand(sn2(s), REQ, ctx)
    assert ex.code == "CANCELLED" and sched.calls == []


def test_root_preflight(setup):
    chem, sched, ctx, s = setup()
    with pytest.raises(ServeError) as e:
        chem.root([SpeciesInput("C1CC", 1)])
    assert (e.value.code, e.value.status, e.value.detail) == ("INVALID_SMILES", 422, {"field": 0})
    with pytest.raises(ServeError) as e:
        chem.root([SpeciesInput("C[O-]", 1), SpeciesInput("[CH3]", 1)])
    assert (e.value.code, e.value.detail) == ("RADICAL_NOT_SUPPORTED", {"field": 1})
    with pytest.raises(ServeError) as e:
        chem.root([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 9)])
    assert e.value.code == "LIMIT_EXCEEDED"
    root = chem.root([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 1)])
    assert root.pool_key == "[CH3][Cl]|0.[CH3][O]|-1" and state_path(root.state_id, s).is_file()
    assert sched.calls == []  # the root never computes


def test_worker_events_are_forwarded(setup, caplog):
    chem, sched, ctx, s = setup(cached={"C[O-]"})
    stale = {"code": "TOKENS_STALE", "key": "CCl", "path": "/x/CCl.npz"}
    wiped = {"code": "ORIENTED_CACHE_WIPED", "detail": "digests changed"}
    feat = ok({"status": "built"})
    feat.events = [stale]
    exp = ok(expansion(product(s)))
    exp.events = [wiped]
    sched.responses = {"featurize": feat, "expand": exp}
    with caplog.at_level("WARNING", logger="ksmech.serve.chemistry"):
        assert chem.expand(sn2(s), REQ, ctx).ok
    assert ctx.skipped == {"TOKENS_STALE": 1} and ctx.skip_log[0]["key"] == "CCl"
    assert "ORIENTED_CACHE_WIPED" in caplog.text
