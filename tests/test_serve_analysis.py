"""analysis.py and candidates.py on MeO- + MeCl (synthetic chemistry): the identities, the pair
reasons, beta and beta^2, the comp_* rebuild, persistence, the digest key, the failure sidecar."""

import copy
import json
import os
import shutil
import threading

import numpy as np
import pytest

from ksmech.serve import analysis
from ksmech.serve.candidates import PairReason, pair_reasons, reason_name, scope_mask
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    sn2_state,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

META_KEYS = {
    "state_id",
    "physics_fingerprint",
    "token_digests",
    "versions",
    "chemistry",
    "ks_params",
    "candidate_config",
    "predictor_versions",
    "n_non_finite",
    "seconds",
    "created_at",
}


@pytest.fixture(scope="module")
def work(synthetic_runtime, sn2_state):  # noqa: F811
    return analysis.pool_work(synthetic_runtime, sn2_state)


def test_floor_identity(work):
    an = work.an
    assert np.array_equal(an.floor, -(an.ks3 + an.ks2))
    assert np.array_equal(an.score_floor, np.where(an.valid, an.floor, -np.inf))
    assert np.array_equal(an.frozen, -(an.ks3 + an.ks2 + an.ks1) / 1.0)


def test_reasons_match_valid_pairs(work):
    from ksmech.physics.pairs import valid_pairs

    an, rt = work.an, work.rt
    tokens = [f.tokens for f in rt.fragments]
    assert np.array_equal(an.reasons == PairReason.VALID, an.valid)
    assert np.array_equal(an.valid, valid_pairs(tokens, rt.candidate))
    assert np.array_equal(an.valid, work.sample.pair_valid)
    assert an.reasons.dtype == np.int8 and an.n_valid == int(an.valid.sum()) > 0


def test_rows_cols_and_cell(work):
    an, rt = work.an, work.rt
    assert np.array_equal(an.rows, rt.n_atoms + np.flatnonzero(rt.occupied))
    assert np.array_equal(an.cols, rt.n_atoms + np.flatnonzero(~rt.occupied))
    assert an.cell(int(an.rows[3]), int(an.cols[2])) == (3, 2)
    for donor, acceptor in ((0, int(an.cols[0])), (int(an.cols[0]), int(an.cols[0]))):
        with pytest.raises(ServeError) as e:
            an.cell(donor, acceptor)
        assert e.value.code == Code.NOT_AN_ORBITAL_TOKEN


def _cl_tokens(loader):
    return loader("CCl")


def _elements(t, o):
    return {t.symbol(int(a)) for a in o.descriptors.all_center_atoms}


def test_pair_reason_codes_on_hand_cases(synthetic_loader):  # noqa: F811
    from ksmech.orbitals.tokens import CandidateConfig
    from ksmech.physics.pairs import valid_pairs

    t = _cl_tokens(synthetic_loader)
    occ, virt = t.occupied(), t.virtual()
    r = pair_reasons([t], CandidateConfig())
    s_cl = next(i for i, o in enumerate(occ) if _elements(t, o) == {"C", "Cl"})
    n_cl = next(i for i, o in enumerate(occ) if _elements(t, o) == {"Cl"})
    ss_cl = next(j for j, o in enumerate(virt) if _elements(t, o) == {"C", "Cl"})
    d_cl = next(j for j, o in enumerate(virt) if t.is_d_type(o))
    assert r[s_cl, ss_cl] == PairReason.OWN_ANTIBOND  # sigma(C-Cl) -> its own sigma*
    assert r[n_cl, ss_cl] == PairReason.SELF_SITE  # n(Cl) -> sigma*(C-Cl): a bond with itself
    assert r[0, d_cl] == PairReason.ACCEPTOR_D_TYPE
    allow_d = CandidateConfig(allow_d_acceptors=True)
    assert pair_reasons([t], allow_d)[0, d_cl] != PairReason.ACCEPTOR_D_TYPE
    assert np.array_equal(pair_reasons([t], allow_d) == 0, valid_pairs([t], allow_d))
    assert reason_name(int(r[s_cl, ss_cl])) == "OWN_ANTIBOND"


def test_pair_reason_occupation_codes(synthetic_loader):  # noqa: F811
    from ksmech.orbitals.tokens import CandidateConfig
    from ksmech.physics.pairs import valid_pairs

    t = copy.deepcopy(_cl_tokens(synthetic_loader))
    occ, virt = t.occupied(), t.virtual()
    occ[1].descriptors.occupation = 1.0  # a SOMO-like donor
    virt[-1].descriptors.occupation = 0.6  # a smeared acceptor
    r = pair_reasons([t], CandidateConfig())
    assert (r[1] == PairReason.DONOR_FRACTIONAL).all()
    assert r[0, -1] == PairReason.ACCEPTOR_FRACTIONAL
    loose = CandidateConfig(exclude_fractional=False)
    r2 = pair_reasons([t], loose)
    assert (r2[1] == PairReason.DONOR_NOT_DOUBLY_OCCUPIED).all()
    assert r2[0, -1] == PairReason.ACCEPTOR_NOT_EMPTY
    for config in (CandidateConfig(), loose):
        assert np.array_equal(pair_reasons([t], config) == 0, valid_pairs([t], config))


def test_scope_mask(work):
    rt = work.rt
    full = scope_mask(rt, "all")
    inter = scope_mask(rt, "inter")
    assert full.shape == work.an.valid.shape and full.all()
    frag = rt.orbital_fragment
    rows, cols = np.flatnonzero(rt.occupied), np.flatnonzero(~rt.occupied)
    assert np.array_equal(inter, frag[rows][:, None] != frag[cols][None, :])
    assert inter.any() and not inter.all()
    assert np.array_equal(inter, work.an.inter)  # the pair-track flag: 1 across fragments
    with pytest.raises(ServeError) as e:
        scope_mask(rt, "intra")
    assert e.value.code == Code.REQUEST_INVALID


def test_beta2_matches_stored_logs(work):
    an = work.an
    for d in ("2.0", "2.3", "3.0"):
        b2 = an[f"beta2_d{d}"]
        defined = work.sample.pair_defined[..., _feature(f"log_beta2_d{d}")]
        assert np.array_equal(~np.isnan(b2), defined)  # NaN exactly where undefined
    b2 = an["beta2_d2.3"]
    ok = ~np.isnan(b2)
    assert np.allclose(an.beta[ok] ** 2, b2[ok], rtol=1e-5, atol=1e-11)


def _feature(name):
    from ksmech.physics.pairs import F

    return F[name]


def test_beta2_log_floor_passes_the_check(work, synthetic_runtime):  # noqa: F811
    """A beta ~ 0 pair stores log(1e-12) at K = 1, so beta2_d2.3 reads 1e-12 * K^2 ~ 3.06e-12,
    not 0; the panel check (rtol 1e-5, atol 1e-11) still passes."""
    import dataclasses

    from ksmech.physics.pairs import LOG_FLOOR

    k = _feature("log_beta2_d2.3")
    z = work.sample.pair_z.copy()
    z[0, 0, k] = np.log(LOG_FLOOR)
    sample = dataclasses.replace(work.sample, pair_z=z)
    members = analysis._physics_members(work.rt, sample, work.comps, synthetic_runtime.kernels)
    floored = members["beta2_d2.3"][0, 0]
    assert floored == pytest.approx(3.0625e-12, rel=1e-6)
    assert np.isclose(0.0, floored, rtol=1e-5, atol=1e-11)


def test_signed_beta_equals_the_kernel_call(work, synthetic_runtime):  # noqa: F811
    from ksmech.physics.params import KSParams

    an, rt = work.an, work.rt
    tokens = [f.tokens for f in rt.fragments]
    kernel = synthetic_runtime.kernels.main
    blocks = []
    for f, t_f in enumerate(tokens):
        row = [
            kernel.beta_matrix(t_f.occupied(), t_f, t_g.virtual(), t_g, KSParams(), same=f == g)
            for g, t_g in enumerate(tokens)
        ]
        blocks.append(np.concatenate(row, axis=1))
    assert np.array_equal(an.beta, np.concatenate(blocks, axis=0))
    assert (an.beta < 0).any() or (an.beta > 0).any()
    assert np.allclose(an.beta**2 * an.gap_kernel, an.ks3, rtol=1e-12, atol=0)


def test_comp_members_rebuild_comps(work, synthetic_runtime):  # noqa: F811
    from ksmech.physics.ranking import pair_components

    fresh = pair_components(work.sample, synthetic_runtime.loader, synthetic_runtime.kernels.main)
    rebuilt = work.an.comps()
    assert set(rebuilt) == set(fresh)
    assert {k for k in work.an.members if k.startswith("comp_")} == {
        "comp_fbar",
        "comp_log_abs_ks1",
    }
    for k in fresh:
        assert np.array_equal(rebuilt[k], fresh[k]), k


def test_save_load_roundtrip(work, tmp_path):
    an = work.an
    npz, meta = tmp_path / "a.npz", tmp_path / "a.json"
    analysis.save_analysis(an, npz, meta)
    back = analysis.load_analysis(npz)
    assert set(back.members) == set(an.members)
    for k, v in an.members.items():
        assert np.array_equal(back[k], v, equal_nan=v.dtype.kind == "f"), k
        assert back[k].dtype == v.dtype
    assert back.meta == json.loads(meta.read_text()) and back.state_id == an.state_id


def test_sidecar_meta(work, synthetic_settings, sn2_state):  # noqa: F811
    meta = json.loads(work.npz.with_suffix(".json").read_text())
    assert set(meta) >= META_KEYS
    assert meta["chemistry"] == "synthetic" and meta["state_id"] == sn2_state.state_id
    assert set(meta["predictor_versions"]) == {"floor", "stub"}
    assert meta["predictor_errors"] == {} and meta["n_non_finite"] == 0
    assert set(meta["token_digests"]) == {"C[O-]", "CCl"}
    fp = meta["physics_fingerprint"]
    assert work.npz.name == f"analysis-{fp}-{meta['analysis_digest']}.npz"
    assert work.npz.parent == synthetic_settings.pools / sn2_state.state_id


def test_consistency_failed(work, synthetic_runtime, monkeypatch):  # noqa: F811
    import ksmech.serve.candidates as candidates

    def all_valid(fragments, config):
        return np.zeros(work.an.valid.shape, dtype=np.int8)

    monkeypatch.setattr(candidates, "pair_reasons", all_valid)
    with pytest.raises(ServeError) as e:
        analysis.analyze_pool(
            work.state,
            work.rt,
            work.sample,
            work.comps,
            None,
            {},
            kernels=synthetic_runtime.kernels,
        )
    assert e.value.code == Code.CONSISTENCY_FAILED


def test_digest_changes_with_token_bytes(serve_settings, synthetic_settings):  # noqa: F811
    import shutil

    from ksmech.orbitals.tokens import load_tokens, save_tokens
    from ksmech.serve.featurize import SyntheticFeaturizer
    from ksmech.serve.pools import SpeciesInput, initial_state

    s = serve_settings(chemistry="synthetic")
    shutil.copytree(synthetic_settings.serve_tokens, s.serve_tokens)
    loader = SyntheticFeaturizer(s).loader()
    state = initial_state([SpeciesInput("C[O-]", 1), SpeciesInput("CCl", 1)], s.limits)
    before = analysis.analysis_digest(loader, state)
    assert before is not None and len(before) == 12
    path = loader.path_of("CCl")
    t = load_tokens(path)
    t.qm_warnings += ",EDITED"
    save_tokens(t, path)
    after = analysis.analysis_digest(loader, state)
    assert after != before
    fp = "f" * 12
    assert (
        analysis.analysis_paths(s, state.state_id, fp, before)[0]
        != analysis.analysis_paths(s, state.state_id, fp, after)[0]
    )
    missing = initial_state([SpeciesInput("CCC", 1)], s.limits)
    assert analysis.analysis_digest(loader, missing) is None


def test_failure_sidecar_on_missing_tokens(serve_settings):  # noqa: F811
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.tasks import runtime
    from ksmech.serve.versions import physics_fingerprint

    s = serve_settings(chemistry="synthetic")
    state = initial_state([SpeciesInput("CCC", 1)], s.limits)
    save_state(state, s)
    rt = runtime(TaskEnv(s))
    with pytest.raises(ServeError) as e:
        analysis.ensure_analysis(rt, state)
    assert e.value.code == Code.MISSING_TOKENS and e.value.detail == {"keys": ["CCC"]}
    failure = analysis.read_failure(s, state.state_id, physics_fingerprint())
    assert failure["code"] == "MISSING_TOKENS" and failure["digest"] is None
    assert failure["stage"] == "analyze"
    path = analysis.failure_path(s, state.state_id, physics_fingerprint())
    assert path.name == f"analysis-failure-{physics_fingerprint()}.json"
    assert analysis.current_failure(s, rt.loader, state, physics_fingerprint()) == failure
    rt.featurizer.ensure("CCC")  # tokens appear: the next analysis succeeds and clears it
    npz = analysis.ensure_analysis(rt, state)
    assert npz.is_file() and not path.exists()


def test_worker_tasks_analyze_and_score(synthetic_settings, sn2_state, work):  # noqa: F811
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.worker import InlineScheduler, Priority

    sched = InlineScheduler(TaskEnv(synthetic_settings))
    r = sched.run("analyze", {"state": sn2_state.to_dict()}, priority=Priority.JOB, timeout_s=900)
    assert r.ok, r
    assert r.result["npz"] == str(work.npz) and r.result["n_valid"] == work.an.n_valid
    assert r.result["fermi_smeared"] is False
    r = sched.run(
        "score",
        {"state": sn2_state.to_dict(), "predictor": "floor"},
        priority=Priority.INTERACTIVE,
        timeout_s=300,
    )
    assert r.ok, r
    assert r.result["version"] == work.an.meta["predictor_versions"]["floor"]
    with np.load(r.result["path"]) as z:
        assert np.array_equal(z["scores"], work.an.score_floor)
    r = sched.run(
        "score",
        {"state": sn2_state.to_dict(), "predictor": "gbdt"},
        priority=Priority.INTERACTIVE,
        timeout_s=300,
    )
    assert not r.ok and r.code == "PREDICTOR_UNAVAILABLE"


# ---- token rebuilds, stored score versions, atomic writes


def _rewrite_tokens(path):
    """Rewrite a token file the way a rebuild does (tmp file, os.replace): the same members,
    other bytes (uncompressed), a newer mtime."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    st = path.stat()
    tmp = path.with_name(f".{path.name}.rebuild.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def fresh(serve_settings, synthetic_settings, sn2_state):  # noqa: F811
    """(settings, runtime) on a new data root holding a copy of the synthetic tokens, QM
    records and oriented blocks, with the SN2 state saved (its token files may be rewritten)."""
    from ksmech.serve.pools import save_state
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.tasks import runtime

    s = serve_settings(chemistry="synthetic", enable_stub_predictor=True)
    for name in ("tokens", "qm", "oriented"):
        shutil.copytree(synthetic_settings.serve / name, s.serve / name)
    save_state(sn2_state, s)
    return s, runtime(TaskEnv(s))


def test_tokens_changed_during_the_analysis(fresh, sn2_state, monkeypatch):  # noqa: F811
    from ksmech.serve.versions import physics_fingerprint

    s, rt = fresh
    path, fp = rt.loader.path_of("CCl"), physics_fingerprint()
    before = analysis.analysis_digest(rt.loader, sn2_state)
    real = analysis.analyze_pool

    def rebuild_meanwhile(*args, **kwargs):
        out = real(*args, **kwargs)
        _rewrite_tokens(path)
        return out

    monkeypatch.setattr(analysis, "analyze_pool", rebuild_meanwhile)
    with pytest.raises(ServeError) as e:
        analysis.ensure_analysis(rt, sn2_state)
    after = analysis.analysis_digest(rt.loader, sn2_state)
    assert (e.value.code, e.value.status) == ("TOKENS_CHANGED", 503) and after != before
    assert e.value.detail["digest_before"] == before and e.value.detail["digest_after"] == after
    for digest in (before, after):  # nothing saved under either name, nothing recorded
        assert not analysis.analysis_paths(s, sn2_state.state_id, fp, digest)[0].exists()
    assert analysis.read_failure(s, sn2_state.state_id, fp) is None
    monkeypatch.undo()
    npz = analysis.ensure_analysis(rt, sn2_state)
    assert npz == analysis.analysis_paths(s, sn2_state.state_id, fp, after)[0]
    assert analysis.load_analysis(npz).meta["analysis_digest"] == after


def test_pool_work_refuses_a_bundle_from_other_tokens(fresh, sn2_state, monkeypatch):  # noqa: F811
    s, rt = fresh
    path = rt.loader.path_of("CCl")
    analysis.ensure_analysis(rt, sn2_state)
    real = analysis._checked_rt

    def rebuild_first(rt_ctx, state):
        _rewrite_tokens(path)  # after the npz was chosen, before rt is read
        return real(rt_ctx, state)

    monkeypatch.setattr(analysis, "_checked_rt", rebuild_first)
    with pytest.raises(ServeError) as e:
        analysis.pool_work(rt, sn2_state)
    assert (e.value.code, e.value.status) == ("TOKENS_CHANGED", 503)
    monkeypatch.undo()
    work = analysis.pool_work(rt, sn2_state)
    digest = analysis.analysis_digest(rt.loader, sn2_state)
    assert work.npz.name.endswith(f"-{digest}.npz") and work.rt is rt.reactant_tokens(sn2_state)


def test_success_clears_a_stale_failure_sidecar(fresh, sn2_state):  # noqa: F811
    from ksmech.serve.versions import physics_fingerprint

    s, rt = fresh
    fp = physics_fingerprint()
    npz = analysis.ensure_analysis(rt, sn2_state)
    err = ServeError(Code.ANALYSIS_FAILED, "an older failure")
    analysis.write_failure(s, sn2_state, fp, err, digest="0" * 12, stage="analyze")
    assert analysis.read_failure(s, sn2_state.state_id, fp) is not None
    assert analysis.ensure_analysis(rt, sn2_state) == npz  # the cached npz path
    assert analysis.read_failure(s, sn2_state.state_id, fp) is None
    analysis.write_failure(s, sn2_state, fp, analysis.tokens_changed(sn2_state, "a", "b"))
    assert analysis.read_failure(s, sn2_state.state_id, fp) is None  # transient: not recorded


def test_stored_scores_need_the_recorded_version(work):
    an = work.an
    v = an.meta["predictor_versions"]["stub"]
    assert np.array_equal(an.stored_scores("stub", v), an.score_stub)
    assert an.stored_scores("stub", v + "x") is None and an.stored_scores("stub", None) is None
    assert an.stored_scores("gbdt", v) is None  # no member, no recorded version
    bare = analysis.PoolAnalysis(an.state_id, an.members, {})
    assert bare.stored_scores("stub", v) is None


def _hammer(write, path, rounds: int = 200) -> list[BaseException]:
    """Two threads write `path` `rounds` times in lock step; each reads it back after every
    write. Returns the exceptions raised."""
    barrier, errors = threading.Barrier(2), []

    def run(t: int) -> None:
        for r in range(rounds):
            try:
                barrier.wait(timeout=30)
                write(path, json.dumps({"thread": t, "round": r, "pad": "x" * 2048}))
                json.loads(path.read_text(encoding="utf-8"))
            except BaseException as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(exc)

    threads = [threading.Thread(target=run, args=(t,)) for t in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


def test_atomic_writes_are_thread_safe(serve_settings):  # noqa: F811
    from ksmech.serve.pools import atomic_write_text

    s = serve_settings()
    text_path = s.pools / "s" / "analysis-failure-x.json"
    bytes_path = s.pools / "s" / "meta.json"
    assert _hammer(lambda p, t: atomic_write_text(p, t, s), text_path) == []
    assert _hammer(lambda p, t: analysis._atomic_bytes(p, t.encode(), s), bytes_path) == []
    names = sorted(p.name for p in text_path.parent.iterdir())
    assert names == sorted([bytes_path.name, text_path.name])  # no tmp file left


def test_atomic_write_survives_a_writer_in_between(serve_settings, monkeypatch):  # noqa: F811
    """Writer B runs completely while writer A sits between its write and its os.replace:
    with a per-process tmp name, B consumed A's tmp file and A's replace failed."""
    from ksmech.serve import pools

    s = serve_settings()
    path = s.pools / "s" / "state.json"
    real_replace, nested = os.replace, []

    class OsProxy:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def replace(src, dst):
            if not nested:
                nested.append(1)
                other = threading.Thread(target=pools.atomic_write_text, args=(dst, "B", s))
                other.start()
                other.join()
            real_replace(src, dst)

    monkeypatch.setattr(pools, "os", OsProxy())
    pools.atomic_write_text(path, "A", s)
    assert path.read_text() == "A" and nested == [1]
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no tmp file left
