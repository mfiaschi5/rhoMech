"""predictors.py: floor, GBDT, transformer and stub predictors, default `propose`, registry."""

import re

import numpy as np
import pytest

from ksmech.serve.contracts import PoolScores, ScoringContext, log_softmax_valid
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.predictors import (
    DEFAULT_STUB_RULES,
    GBDT_SAMPLE_WARNING,
    FloorPredictor,
    GBDTPredictor,
    PredictorRegistry,
    StubPredictor,
    StubRule,
    TransformerPredictor,
)
from ksmech.serve.testing import requires
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    sn2_state,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = [pytest.mark.serve, requires("predict")]

FP = "0123456789ab"


@pytest.fixture(scope="module")
def work(synthetic_runtime, sn2_state):  # noqa: F811
    from ksmech.serve.analysis import pool_work

    return pool_work(synthetic_runtime, sn2_state)


@pytest.fixture(scope="module")
def ctx(work, synthetic_runtime):  # noqa: F811
    return ScoringContext(work.sample, work.rt, synthetic_runtime.loader, work.comps)


def _pool(scores, valid, sites=None, t=1.0) -> PoolScores:
    scores = np.where(valid, scores, -np.inf)
    n_occ, n_virt = valid.shape
    if sites is None:
        sites = np.stack(np.meshgrid(np.arange(n_occ), 100 + np.arange(n_virt), indexing="ij"), -1)
    return PoolScores(
        scores,
        valid,
        log_softmax_valid(scores, valid, t),
        sites,
        np.arange(n_occ) + 10,
        np.arange(n_virt) + 20,
        t,
    )


def _elements(rt, token_id):
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    z = [a.atomic_number for f in rt.fragments for a in f.tokens.atoms]
    return {pt.GetElementSymbol(int(z[a])) for a in rt.centres(token_id)}


def test_untrained_score_is_the_floor():
    from ksmech.physics.cli import UNTRAINED_SCORE

    assert UNTRAINED_SCORE == "ks3+ks2"


def test_floor_equals_scorer_masked(ctx):
    from ksmech.physics.ranking import SCORERS

    got = FloorPredictor(FP).score(ctx)
    want = np.where(ctx.sample.pair_valid, SCORERS["ks3+ks2"](ctx.comps), -np.inf)
    assert np.array_equal(got, want) and got.dtype == np.float64
    assert np.array_equal(
        got[ctx.sample.pair_valid], -(ctx.comps["ks3"] + ctx.comps["ks2"])[ctx.sample.pair_valid]
    )


def test_default_propose_order_ties_and_k():
    valid = np.array([[True, True, False], [True, True, True]])
    scores = np.array([[1.0, 3.0, 9.0], [3.0, 0.5, 3.0]])  # three-way tie at 3.0
    pool = _pool(scores, valid)
    steps = FloorPredictor(FP).propose(pool, None, k=10)
    assert [s.cell for s in steps] == [(0, 1), (1, 0), (1, 2), (0, 0), (1, 1)]
    assert [s.rank for s in steps] == [1, 2, 3, 4, 5]
    assert [s.score for s in steps] == [3.0, 3.0, 3.0, 1.0, 0.5]
    assert all(s.logp == pool.logp[s.cell] for s in steps)
    assert np.isclose(sum(np.exp(s.logp) for s in steps), 1.0)
    first = steps[0]
    assert len(first.arrows) == 1 and first.site_sources == ("contact",)
    a = first.arrows[0]
    assert (a.donor, a.acceptor) == (10, 21) and (a.donor_site, a.acceptor_site) == (0, 101)
    assert [s.cell for s in FloorPredictor(FP).propose(pool, None, k=2)] == [(0, 1), (1, 0)]
    assert FloorPredictor(FP).propose(pool, None, k=0) == []


def test_propose_main_centre_fallback(ctx, work):
    from ksmech.serve.predictors import main_centre_site

    an = work.an
    pool = PoolScores(
        an.floor,
        an.valid,
        log_softmax_valid(np.where(an.valid, an.floor, -np.inf), an.valid, 1.0),
        an.contact_sites.copy(),
        an.rows,
        an.cols,
        1.0,
    )
    top = FloorPredictor(FP).propose(pool, ctx, k=1)[0]
    i, j = top.cell
    pool.contact_sites[i, j] = (-1, -1)
    fallback = FloorPredictor(FP).propose(pool, ctx, k=1)[0]
    assert fallback.site_sources == ("main_centre",)
    arrow = fallback.arrows[0]
    assert arrow.donor_site == main_centre_site(ctx.rt, arrow.donor)
    assert arrow.acceptor_site == main_centre_site(ctx.rt, arrow.acceptor)


def test_propose_uses_contact_sites(ctx, work):
    an = work.an
    scores = FloorPredictor(FP).score(ctx)
    pool = PoolScores(
        scores,
        an.valid,
        log_softmax_valid(scores, an.valid, 1.0),
        an.contact_sites,
        an.rows,
        an.cols,
        1.0,
    )
    steps = FloorPredictor(FP).propose(pool, ctx, k=5)
    assert len(steps) == 5
    for s in steps:
        i, j = s.cell
        a = s.arrows[0]
        assert (a.donor, a.acceptor) == (an.rows[i], an.cols[j])
        assert (a.donor_site, a.acceptor_site) == tuple(an.contact_sites[i, j])


# ---- GBDT


def _tiny(path, trees=()):
    from ksmech.physics import baseline as bl

    model = bl.GBDTRanker(edges=[np.array([0.0])] * len(bl.FEATURE_NAMES), trees=list(trees))
    return bl.save_gbdt(model, path)


def test_gbdt_missing(tmp_path):
    info = GBDTPredictor(tmp_path / "none.npz", FP).info()
    assert not info.available and info.code == "GBDT_MISSING"
    assert info.version is None and info.checkpoint_hash is None
    assert info.warnings == (GBDT_SAMPLE_WARNING,)


def test_gbdt_valid_tiny_model(tmp_path, ctx):
    from ksmech.serve.versions import file_sha256

    path = _tiny(tmp_path / "m.npz")
    p = GBDTPredictor(path, FP)
    info = p.info()
    assert info.available and info.code is None and info.kind == "baseline"
    sha = file_sha256(path)
    assert info.checkpoint_hash == sha and re.match(r"^[0-9a-f]{64}$", sha)
    assert info.version == f"gbdt/{sha[:16]}/{FP}"
    assert "records no SAMPLE_VERSION" in info.warnings[0]
    scores = p.score(ctx)
    assert np.array_equal(scores, np.where(ctx.sample.pair_valid, 0.0, -np.inf))


def test_gbdt_warning_states_measured_values_only():
    from ksmech.serve.versions import source_constant

    current = source_constant("arrows/steps.py", "SAMPLE_VERSION")
    assert GBDT_SAMPLE_WARNING.startswith("the GBDT file records no SAMPLE_VERSION")
    assert f"SAMPLE_VERSION {current})" in GBDT_SAMPLE_WARNING
    assert "sample-v5" not in GBDT_SAMPLE_WARNING and "trained on" not in GBDT_SAMPLE_WARNING


def test_gbdt_follows_its_file(tmp_path, monkeypatch, ctx):
    import os

    from ksmech.physics import baseline as bl
    from ksmech.serve.versions import file_sha256

    path = _tiny(tmp_path / "m.npz")
    p = GBDTPredictor(path, FP)
    first = p.info()
    checks = []
    real = GBDTPredictor._check
    monkeypatch.setattr(GBDTPredictor, "_check", lambda self: checks.append(1) or real(self))
    assert p.info() == first and p.info() == first and checks == []  # unchanged: never re-read
    leaf = bl.Tree(
        np.array([0, -1, -1]),
        np.array([0, 0, 0]),
        np.array([1, -1, -1]),
        np.array([2, -1, -1]),
        np.array([0.0, 1.0, -1.0]),
    )
    st = path.stat()
    _tiny(path, [leaf])  # a retrained model in place: other bytes
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    second = p.info()
    assert checks == [1] and second.available
    assert second.checkpoint_hash == file_sha256(path) != first.checkpoint_hash
    assert second.version == f"gbdt/{file_sha256(path)[:16]}/{FP}" != first.version
    assert p.info() == second and checks == [1]
    valid = ctx.sample.pair_valid
    assert np.all(np.abs(p.score(ctx)[valid]) == pytest.approx(0.1))  # the new model scores
    path.unlink()
    gone = p.info()
    assert not gone.available and gone.code == "GBDT_MISSING" and gone.checkpoint_hash is None


def test_gbdt_features_mismatch(tmp_path):
    import json

    path = _tiny(tmp_path / "m.npz")
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta["features"] = meta["features"][:-1]
    arrays["meta"] = np.array(json.dumps(meta))
    edited = tmp_path / "edited.npz"
    np.savez(edited, **arrays)
    info = GBDTPredictor(edited, FP).info()
    assert not info.available and info.code == "GBDT_FEATURES_MISMATCH"
    assert info.version is not None  # the file exists: its hash is still reported


def test_gbdt_tree_feature_out_of_range(tmp_path):
    from ksmech.physics import baseline as bl

    tree = bl.Tree(
        np.array([70, -1, -1]),
        np.array([0, 0, 0]),
        np.array([1, -1, -1]),
        np.array([2, -1, -1]),
        np.array([0.0, 1.0, -1.0]),
    )
    info = GBDTPredictor(_tiny(tmp_path / "m.npz", [tree]), FP).info()
    assert info.code == "GBDT_FEATURES_MISMATCH"


def test_gbdt_load_failed(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip file")
    info = GBDTPredictor(bad, FP).info()
    assert not info.available and info.code == "GBDT_LOAD_FAILED"


def test_gbdt_unavailable_score_raises(tmp_path, ctx):
    with pytest.raises(ServeError) as e:
        GBDTPredictor(tmp_path / "none.npz", FP).score(ctx)
    assert e.value.code == Code.PREDICTOR_UNAVAILABLE and e.value.status == 409


# ---- transformer, versions


def test_transformer_unavailable(ctx):
    p = TransformerPredictor()
    info = p.info()
    assert not info.available and info.code == "TRANSFORMER_NOT_AVAILABLE"
    assert "no checkpoint" in info.reason
    with pytest.raises(ServeError) as e:
        p.score(ctx)
    assert e.value.status == 409 and e.value.detail["code"] == "TRANSFORMER_NOT_AVAILABLE"


def test_version_strings():
    assert FloorPredictor(FP).info().version == f"floor/ks3+ks2/d2.3/{FP}"
    assert FloorPredictor(FP).info().checkpoint_hash is None
    stub = StubPredictor(FP).info().version
    assert re.match(rf"^stub/[0-9a-f]{{12}}/{FP}$", stub)
    other = StubPredictor(FP, (StubRule(None, "O", "C", None, 1.0),)).info().version
    assert other != stub
    # the version does not depend on the hash seed (frozenset order)
    swapped = StubRule(frozenset({"O"}), "O", "C", frozenset(["Cl", "C"]), 50.0)
    assert StubPredictor(FP, (swapped,)).info().version == stub


def test_floor_default_temperature():
    info = FloorPredictor(FP).info()
    assert info.default_temperature == 1.0 and info.score_unit == "eV"
    assert info.kind == "physics_floor" and info.available


# ---- stub


def test_stub_rule_matching(ctx):
    rt = ctx.rt
    stub = StubPredictor(FP)
    rows = rt.n_atoms + np.flatnonzero(rt.occupied)
    cols = rt.n_atoms + np.flatnonzero(~rt.occupied)
    sigma_star = [c for c in cols if _elements(rt, c) == {"C", "Cl"}]
    lone_pairs = [r for r in rows if _elements(rt, r) == {"O"}]
    sigma_co = [r for r in rows if _elements(rt, r) == {"C", "O"}]
    assert sigma_star and lone_pairs and sigma_co
    assert stub.first_rule(rt, lone_pairs[0], sigma_star[0]) == DEFAULT_STUB_RULES[0]
    assert stub.first_rule(rt, sigma_co[0], sigma_star[0]) is None  # centres {C, O}
    bonus = stub.bonus(ctx)
    i = int(np.flatnonzero(rows == lone_pairs[0])[0])
    j = int(np.flatnonzero(cols == sigma_star[0])[0])
    k = int(np.flatnonzero(rows == sigma_co[0])[0])
    assert bonus[i, j] == 50.0 and bonus[k, j] == 0.0
    scores = stub.score(ctx)
    floor = FloorPredictor(FP).score(ctx)
    assert np.array_equal(scores, np.where(ctx.sample.pair_valid, floor + bonus, -np.inf))


def test_stub_propose_rule_sites(ctx, work):
    an = work.an
    stub = StubPredictor(FP)
    scores = stub.score(ctx)
    pool = PoolScores(
        scores,
        an.valid,
        log_softmax_valid(scores, an.valid, 1.0),
        an.contact_sites,
        an.rows,
        an.cols,
        1.0,
    )
    top = stub.propose(pool, ctx, k=3)
    assert top[0].site_sources == ("rule",)
    a = top[0].arrows[0]
    z = [x.atomic_number for f in ctx.rt.fragments for x in f.tokens.atoms]
    assert z[a.donor_site] == 8 and z[a.acceptor_site] == 6  # sites (O, C)
    assert _elements(ctx.rt, a.acceptor) == {"C", "Cl"}
    assert _elements(ctx.rt, a.donor) == {"O"}
    assert top[0].rank == 1 and top[0].score > 40


# ---- registry


def test_registry(serve_settings, tmp_path):  # noqa: F811
    s = serve_settings()
    reg = PredictorRegistry(s, FP)
    assert reg.names() == ["floor", "gbdt", "transformer"]  # no stub unless enabled
    assert [i.name for i in reg.info()] == reg.names()
    assert reg.get("floor").name == "floor"
    with pytest.raises(ServeError) as e:
        reg.get("stub")
    assert e.value.code == Code.PREDICTOR_UNKNOWN and e.value.status == 422
    with pytest.raises(ServeError) as e:
        reg.info_of("nope")
    assert e.value.code == Code.PREDICTOR_UNKNOWN
    for name, code in (("transformer", "TRANSFORMER_NOT_AVAILABLE"), ("gbdt", "GBDT_MISSING")):
        with pytest.raises(ServeError) as e:
            reg.get(name)
        assert e.value.code == Code.PREDICTOR_UNAVAILABLE and e.value.status == 409
        assert e.value.detail["code"] == code and e.value.detail["reason"]
    assert set(reg.available()) == {"floor"}
    ref = reg.ref("floor")
    assert ref.temperature == 1.0 and ref.version == f"floor/ks3+ks2/d2.3/{FP}"
    assert reg.ref("floor", 0.5).temperature == 0.5


def test_registry_with_stub_and_gbdt(serve_settings, tmp_path):  # noqa: F811
    path = _tiny(tmp_path / "model.npz")
    reg = PredictorRegistry(serve_settings(enable_stub_predictor=True, gbdt_path=path), FP)
    assert reg.names() == ["floor", "gbdt", "transformer", "stub"]
    assert set(reg.available()) == {"floor", "gbdt", "stub"}
    assert reg.get("gbdt").info().available
    assert reg.ref("stub").version.startswith("stub/")
