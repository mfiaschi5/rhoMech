"""tasks.py, worker.py, context.py, versions.py, settings.py and errors.py: run_task error
mapping and lazy references, InlineScheduler, interactive, WorkerRuntime, ApiContext, meta,
the version strings and fingerprints, the settings."""

import json
import os
import pickle
import re
import shutil

import numpy as np
import pytest

from ksmech.serve import tasks
from ksmech.serve.errors import STATUS, Code, Pending, ServeError, error_body, jsonable
from ksmech.serve.settings import TaskEnv, assert_writable, load_settings, pin_threads
from ksmech.serve.testing import requires
from ksmech.serve.worker import (
    InlineScheduler,
    Priority,
    TaskHandle,
    TaskResult,
    WorkerStatus,
    interactive,
    task_key,
)
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    serve_settings,
    sn2_state,
    synthetic_context,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

SPEC_RE = re.compile(r"^v9\+[0-9a-f]{12};tok\d+;smp\d+;qm\d+$")
PROVENANCE_KEYS = {
    "predictor",
    "predictor_version",
    "checkpoint",
    "checkpoint_hash",
    "spec_version",
    "physics_fingerprint",
    "artifact_versions",
    "chemistry",
    "ks_params",
    "temperature",
    "candidate_scope",
    "candidate_config",
    "notes",
}
META_KEYS = {
    "api_schema",
    "capabilities",
    "app_version",
    "spec_label",
    "spec_sha256_12",
    "spec_version",
    "artifact_versions",
    "physics_fingerprint",
    "chemistry",
    "qm_method",
    "token_method",
    "ks_params",
    "candidate_config",
    "limits",
    "worker",
    "data_root",
}
WORKER_KEYS = {
    "alive",
    "n_workers",
    "pids",
    "busy",
    "current_task",
    "current_label",
    "elapsed_s",
    "queue_depth",
    "restarts",
    "unstable",
}


@pytest.fixture
def env(serve_settings):  # noqa: F811
    return TaskEnv(serve_settings(chemistry="synthetic"))


@pytest.fixture
def extra_task(monkeypatch):
    """Install a temporary task body under a name."""

    def install(name, body):
        monkeypatch.setitem(tasks.TASKS, name, body)
        monkeypatch.setattr(tasks, "_BODIES", {})

    return install


# ---- run_task


def test_run_task_maps_unexpected_errors(env, extra_task):
    def boom(rt, payload):
        raise ZeroDivisionError("division by zero")

    extra_task("boom", boom)
    out = tasks.run_task("boom", env, {}, 1)
    assert out["ok"] is False and out["code"] == "UNEXPECTED"
    assert out["detail"] == "ZeroDivisionError: division by zero"
    assert set(out) == {"ok", "result", "code", "upstream", "detail", "events", "pid", "seconds"}
    assert out["pid"] == os.getpid() and out["seconds"] >= 0


def test_run_task_ships_serve_errors(env, extra_task):
    def refuse(rt, payload):
        raise ServeError(Code.APPLY_FAILED, "no bond", upstream="UNKNOWN_ARROW", detail={"i": 0})

    extra_task("refuse", refuse)
    out = tasks.run_task("refuse", env, {}, 2)
    assert out["code"] == "APPLY_FAILED" and out["upstream"] == "UNKNOWN_ARROW"
    assert out["result"] == {"error": {"message": "no bond", "status": 422, "detail": {"i": 0}}}
    err = ServeError.from_task(TaskResult.from_dict(out))
    assert (err.code, err.status, err.upstream, err.detail) == (
        Code.APPLY_FAILED,
        422,
        "UNKNOWN_ARROW",
        {"i": 0},
    )


def test_run_task_lazy_references(env, extra_task):
    extra_task("ghost", "ksmech.serve.does_not_exist:t_ghost")
    out = tasks.run_task("ghost", env, {}, 3)
    assert out["code"] == "UNEXPECTED"
    assert out["detail"] == "task body not installed: ksmech.serve.does_not_exist:t_ghost"
    extra_task("ghost2", "ksmech.serve.contracts:t_missing")
    out = tasks.run_task("ghost2", env, {}, 4)
    assert out["detail"] == "task body not installed: ksmech.serve.contracts:t_missing"
    out = tasks.run_task("no_such_task", env, {}, 5)
    assert out["code"] == "UNEXPECTED" and "unknown task" in out["detail"]


def test_task_registry_is_fixed():
    assert set(tasks.TASKS) == {
        "ping",
        "diag_sleep",
        "diag_crash",
        "featurize",
        "analyze",
        "expand",
        "orbital_cube",
        "index_chunk",
        "recheck_step",
        "wb_orbital_set",
        "wb_cube",
        "wb_properties",
        "embedding",
    }
    lazy = {k: v for k, v in tasks.TASKS.items() if isinstance(v, str)}
    assert lazy["expand"] == "ksmech.serve.expansion:t_expand"
    assert lazy["recheck_step"] == "ksmech.serve.recheck:t_recheck_step"
    assert tasks.resolve("analyze").__name__ == "t_analyze"


def test_run_task_pid_file(env, extra_task):
    seen = {}

    def peek(rt, payload):
        path = env.settings.run / "task-7.pid"
        seen["exists"] = path.is_file()
        seen["pid"] = int(path.read_text())
        return {"ok": True}

    extra_task("peek", peek)
    out = tasks.run_task("peek", env, {}, 7)
    assert out["ok"] and out["result"] == {"ok": True}
    assert seen == {"exists": True, "pid": os.getpid()}
    assert not (env.settings.run / "task-7.pid").exists()


def test_diag_tasks_refused_unless_enabled(env):
    for name in ("diag_sleep", "diag_crash"):
        out = tasks.run_task(name, env, {"seconds": 0}, 8)
        assert out["code"] == "UNEXPECTED" and "disabled" in out["detail"]
    on = TaskEnv(env.settings, enable_diag_tasks=True)
    out = tasks.run_task("diag_sleep", on, {"seconds": 0.01}, 9)
    assert out["ok"] and out["result"] == {"slept": 0.01}
    out = tasks.run_task("diag_sleep", on, {"seconds": 601}, 10)
    assert out["code"] == "REQUEST_INVALID"


def test_ping(env):
    out = tasks.run_task("ping", env, {}, 11)
    assert out["ok"], out
    r = out["result"]
    assert r["pid"] == os.getpid() and r["torch_threads"] == 1
    assert set(r["threads"]) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    assert isinstance(r["sigalrm"], str)


def test_featurize_task(env):
    out = tasks.run_task("featurize", env, {"key": "CO"}, 12)
    assert out["ok"] and out["result"]["status"] == "built" and out["result"]["key"] == "CO"
    out = tasks.run_task("featurize", env, {"key": "CO"}, 13)
    assert out["result"]["status"] == "cached"
    out = tasks.run_task("featurize", env, {"key": "[SeH2]"}, 14)
    assert not out["ok"] and out["code"] == "SYNTHETIC_FAILED"
    assert out["upstream"] == "UNSUPPORTED_ELEMENT"
    assert out["result"]["error"]["detail"]["status"] == "failed"


# ---- InlineScheduler and interactive


def test_inline_scheduler(env):
    sched = InlineScheduler(env)
    h = sched.submit("ping", {}, priority=Priority.INTERACTIVE, timeout_s=60, label="ping it")
    assert h.done() and h.label == "ping it" and h.task_id.startswith("inline-")
    r = h.wait(0)
    assert isinstance(r, TaskResult) and r.ok
    assert sched.run("ping", {}, priority=Priority.JOB, timeout_s=60).ok
    assert sched.find(task_key("ping", {})) is None
    st = sched.status()
    assert isinstance(st, WorkerStatus) and set(st.to_dict()) == WORKER_KEYS
    assert st.alive and st.pids == [os.getpid()] and not st.busy and st.queue_depth == 0
    assert sched.kill_owner("j1") is False and sched.close() is None
    assert sched.history == [("ping", "ping it"), ("ping", "")]
    bad = sched.run("ping", {"x": object()}, priority=Priority.JOB, timeout_s=60)
    assert not bad.ok and bad.code == "UNEXPECTED" and "JSON" in bad.detail


def test_inline_results_cross_a_json_boundary(env, extra_task):
    extra_task("np_result", lambda rt, payload: {"x": np.int64(3)})
    r = InlineScheduler(env).run("np_result", {}, priority=Priority.JOB, timeout_s=10)
    assert not r.ok and "JSON" in r.detail
    extra_task("tuple_result", lambda rt, payload: {"x": (1, 2)})
    r = InlineScheduler(env).run("tuple_result", {}, priority=Priority.JOB, timeout_s=10)
    assert r.ok and r.result == {"x": [1, 2]}


def test_interactive_ok_and_error(env):
    sched = InlineScheduler(env)
    out = interactive(sched, "featurize", {"key": "CO"}, label="CO", wait_s=1, timeout_s=60)
    assert out["key"] == "CO"
    with pytest.raises(ServeError) as e:
        interactive(sched, "featurize", {"key": "[SeH2]"}, label="x", wait_s=1, timeout_s=60)
    assert e.value.code == "SYNTHETIC_FAILED" and e.value.status == 422
    assert e.value.detail["status"] == "failed"


class _SlowScheduler:
    """A scheduler whose tasks never finish (for the Pending path) and that dedups by key."""

    def __init__(self):
        self.submitted = []
        self.handles = {}

    def submit(self, name, payload, *, priority, timeout_s, owner="", key=None, label=""):
        self.submitted.append((name, key, priority))
        self.handles[key] = TaskHandle(f"t{len(self.submitted)}", label)
        return self.handles[key]

    def find(self, key):
        return self.handles.get(key)

    def status(self):
        return WorkerStatus(True, 1, [1], True, "featurize", "QM", 2.0, 3, 0, False)


def test_interactive_pending_and_dedup():
    sched = _SlowScheduler()
    with pytest.raises(Pending) as e:
        interactive(sched, "analyze", {"state": 1}, label="analysis", wait_s=0.01, timeout_s=9)
    p = e.value
    assert p.task_id == "t1" and p.label == "analysis" and p.retry_after_s == 1.0
    assert p.worker["queue_depth"] == 3
    body = p.to_dict()
    assert body["status"] == "pending" and set(body["worker"]) == WORKER_KEYS
    with pytest.raises(Pending) as e:
        interactive(sched, "analyze", {"state": 1}, label="analysis", wait_s=0.01, timeout_s=9)
    assert e.value.task_id == "t1" and len(sched.submitted) == 1  # joined, not resubmitted
    assert sched.submitted[0][1] == task_key("analyze", {"state": 1})
    assert sched.submitted[0][2] == Priority.INTERACTIVE
    assert task_key("analyze", {"state": 1}).startswith("analyze:")


# ---- WorkerRuntime


# ---- ApiContext


@requires("predict")  # the pool analysis serves predict and correct
def test_api_context_ensure_analysis(synthetic_context, sn2_state):  # noqa: F811
    from ksmech.serve.analysis import PoolAnalysis

    ctx = synthetic_context
    an = ctx.ensure_analysis(sn2_state)
    assert isinstance(an, PoolAnalysis) and an.state_id == sn2_state.state_id
    assert ctx.ensure_analysis(sn2_state) is an  # LRU by npz path
    assert ctx.state(sn2_state.state_id) == sn2_state
    rt = ctx.reactant_tokens(sn2_state)
    assert ctx.reactant_tokens(sn2_state) is rt and rt.n_atoms == sn2_state.n_atoms
    with pytest.raises(ServeError) as e:
        ctx.state("s" + "9" * 20)
    assert e.value.code == Code.POOL_NOT_FOUND and e.value.status == 404


def test_api_context_not_analyzed(serve_settings):  # noqa: F811
    from ksmech.serve.context import build_context
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    s = serve_settings(chemistry="synthetic")
    ctx = build_context(s)
    state = initial_state([SpeciesInput("[SeH2]", 1)], s.limits)  # no EHT levels for Se
    save_state(state, s)
    with pytest.raises(ServeError) as e:
        ctx.ensure_analysis(state)
    err = e.value
    assert err.code == Code.POOL_NOT_ANALYZED and err.status == 409
    assert err.detail["code"] == "SYNTHETIC_FAILED" and err.detail["stage"] == "featurize"
    n_tasks = len(ctx.scheduler.history)
    with pytest.raises(ServeError) as e:
        ctx.ensure_analysis(state)  # the failure is recorded: no second attempt
    assert e.value.code == Code.POOL_NOT_ANALYZED and len(ctx.scheduler.history) == n_tasks
    with pytest.raises(ServeError) as e:
        ctx.reactant_tokens(state)
    assert e.value.code == Code.POOL_NOT_ANALYZED and e.value.detail["code"] == "MISSING_TOKENS"


@requires("predict")  # the pool analysis serves predict and correct
def test_api_context_featurizes_then_analyzes(serve_settings):  # noqa: F811
    from ksmech.serve.context import build_context
    from ksmech.serve.pools import SpeciesInput, initial_state, save_state

    s = serve_settings(chemistry="synthetic")
    ctx = build_context(s)
    state = initial_state([SpeciesInput("[OH-]", 1), SpeciesInput("CO", 1)], s.limits)
    save_state(state, s)
    an = ctx.ensure_analysis(state)
    assert an.n_valid > 0
    names = [name for name, _ in ctx.scheduler.history]
    assert names == ["featurize", "featurize", "analyze"]
    labels = [label for _, label in ctx.scheduler.history]
    assert labels[0].startswith("synthetic EHT for ") and "atoms)" in labels[0]


def test_provenance(synthetic_context):  # noqa: F811
    ctx = synthetic_context
    p = ctx.provenance("floor")
    assert set(p) == PROVENANCE_KEYS
    assert p["predictor"] == "floor" and p["predictor_version"].startswith("floor/g-atom/")
    assert p["checkpoint_hash"] is None and SPEC_RE.match(p["spec_version"])
    assert p["chemistry"] == "synthetic" and p["temperature"] == 1.0
    assert p["ks_params"] == {"K": 1.75, "eps": 4.0}
    assert p["candidate_scope"] == "all"
    assert p["candidate_config"]["site_weight"] == 0.20  # upstream CandidateConfig default
    assert any("END_MECH" in n for n in p["notes"]) and any("synthetic" in n for n in p["notes"])
    q = ctx.provenance(None, scope="inter", notes=("hello",))
    assert q["predictor"] == "none" and q["temperature"] is None and q["notes"][0] == "hello"
    assert any("inter" in n for n in q["notes"])
    assert ctx.provenance("floor", temperature=0.5)["temperature"] == 0.5
    json.dumps(p, allow_nan=False)


def test_meta(synthetic_context):  # noqa: F811
    from ksmech.serve.context import meta

    m = meta(synthetic_context)
    assert set(m) == META_KEYS
    assert m["api_schema"] == 1 and m["spec_label"] == "v9" and SPEC_RE.match(m["spec_version"])
    assert m["spec_version"].startswith(f"v9+{m['spec_sha256_12']};")
    assert set(m["capabilities"]) >= {"predict", "correct", "export"}
    assert "contact" not in m["capabilities"] and "site_pairs" not in m["capabilities"]
    assert m["token_method"] == "SYNTHETIC-EHT/v1" and m["chemistry"] == "synthetic"
    assert set(m["worker"]) == WORKER_KEYS and m["limits"]["max_atoms"] == 150
    json.dumps(m, allow_nan=False)


# ---- versions


def test_spec_version_and_artifacts(synthetic_settings):  # noqa: F811
    from ksmech.arrows.steps import SAMPLE_VERSION
    from ksmech.orbitals.tokens import SCHEMA_VERSION as TOK
    from ksmech.qm.records import SCHEMA_VERSION as QM
    from ksmech.qm.xtb_calc import QMConfig
    from ksmech.serve import versions

    v = versions.artifact_versions()
    assert v == {
        "tokens": TOK,
        "samples": SAMPLE_VERSION,
        "qm": QM,
        "qm_method": QMConfig().method,
    }
    sv = versions.spec_version(synthetic_settings.spec_path)
    assert SPEC_RE.match(sv), sv
    assert sv.endswith(f";tok{TOK};smp{SAMPLE_VERSION};qm{QM}")
    assert versions.spec_sha12(synthetic_settings.spec_path) == sv[3:15]
    assert versions.spec_sha12(synthetic_settings.data_root / "missing.tex") is None
    assert versions.spec_version(synthetic_settings.data_root / "missing.tex").startswith(
        "v9+unknown;"
    )
    assert versions.floor_version("abc") == "floor/g-atom/abc"
    assert re.match(r"^[0-9a-f]{12}$", versions.token_fingerprint())
    assert re.match(r"^[0-9a-f]{12}$", versions.physics_fingerprint())
    assert versions.physics_fingerprint() != versions.token_fingerprint()


def _tree(tmp_path):
    from ksmech.serve import versions

    src = versions.SRC_ROOT
    root = tmp_path / "src"
    for kind in ("token", "physics"):
        for rel in versions.fingerprint_files(kind):
            if (src / rel).is_file():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, root / rel)
    return root


def test_fingerprints_change_only_with_their_own_files(tmp_path):
    from ksmech.serve.versions import compute_fingerprint

    root = _tree(tmp_path)

    def both():
        return (
            compute_fingerprint("token", root, "M"),
            compute_fingerprint("physics", root, "M"),
        )

    base = both()
    assert base == (
        compute_fingerprint("token", root, "M"),
        compute_fingerprint("physics", root, "M"),
    )
    edits = {
        "serve/gto.py": (True, True),
        "orbitals/tokens.py": (True, True),
        "physics/pairs.py": (False, True),
        "orbitals/classify.py": (False, True),
        "arrows/steps.py": (False, True),
        "serve/analysis.py": (False, True),
        "serve/reading.py": (False, True),
        "serve/geometry.py": (False, False),  # display-only: invalidates nothing
    }
    for rel, (ori, phy) in edits.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        old = path.read_bytes() if path.exists() else None
        path.write_bytes((old or b"") + b"\n# edited\n")
        now = both()
        assert (now[0] != base[0], now[1] != base[1]) == (ori, phy), rel
        if old is None:
            path.unlink()
        else:
            path.write_bytes(old)
        assert both() == base, rel
    assert compute_fingerprint("physics", root, "other method") != base[1]
    assert compute_fingerprint("token", root, "other method") == base[0]


def test_fingerprint_tolerates_missing_files(tmp_path):
    from ksmech.serve.versions import compute_fingerprint, fingerprint_files

    root = _tree(tmp_path)
    before = compute_fingerprint("physics", root, "M")
    (root / "serve" / "synthetic.py").unlink()
    gone = compute_fingerprint("physics", root, "M")
    assert gone != before and re.match(r"^[0-9a-f]{12}$", gone)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert re.match(r"^[0-9a-f]{12}$", compute_fingerprint("physics", empty, "M"))
    assert "physics/*.py" in fingerprint_files("physics", empty)  # unmatched glob kept
    with pytest.raises(ValueError):
        fingerprint_files("other")


# ---- settings


def test_settings_defaults(tmp_path):
    s = load_settings({}, data_root=tmp_path)
    assert s.data_root == tmp_path.resolve() and s.data_root.is_absolute()
    assert s.shared_cache_root == s.data_root and s.auto_index is False
    assert s.chemistry == "real" and s.workers == 1
    assert s.shared_qm == s.data_root / "data" / "cache" / "qm"
    assert s.shared_tokens == s.data_root / "data" / "cache" / "tokens"
    assert s.serve_qm == s.data_root / "data" / "serve" / "qm"
    assert s.gbdt_path == s.processed / "pmechdb" / "gbdt_lgbm_molecule_geomfree.npz"
    assert s.frontend_dist == s.repo_root / "apps" / "ui" / "dist"
    assert s.spec_path == s.repo_root / "docs" / "ksmech_architecture.tex"
    assert s.curation_db == s.data_root / "data" / "curation" / "curation.sqlite"
    assert s.serve_bond_index == s.serve / "bond_index"
    assert s.shared_bond_index == s.data_root / "data" / "cache" / "bond_index"
    assert s.limits.max_beam == 10 and s.limits.max_nodes == 200
    assert pickle.loads(pickle.dumps(TaskEnv(s))) == TaskEnv(s)


def test_settings_env_parsing(tmp_path):
    env = {
        "KSMECH_DATA_ROOT": str(tmp_path),
        "KSMECH_SERVE_CHEMISTRY": "synthetic",
        "KSMECH_SERVE_WORKERS": "7",
        "KSMECH_SERVE_AUTO_INDEX": "true",
        "KSMECH_SHARED_CACHE_ROOT": str(tmp_path / "shared"),
        "KSMECH_SERVE_GBDT": str(tmp_path / "m.npz"),
    }
    s = load_settings(env)
    assert s.chemistry == "synthetic" and s.workers == 2  # clamped to [1, 2]
    assert s.auto_index
    assert s.shared_qm == (tmp_path / "shared").resolve() / "data" / "cache" / "qm"
    assert s.gbdt_path == (tmp_path / "m.npz").resolve()
    assert load_settings({**env, "KSMECH_SERVE_WORKERS": "0"}).workers == 1
    assert load_settings({**env, "KSMECH_SERVE_AUTO_INDEX": "0"}).auto_index is False
    assert load_settings(env, chemistry="cached").chemistry == "cached"  # overrides win
    with pytest.raises(ValueError):
        load_settings({**env, "KSMECH_SERVE_CHEMISTRY": "quantum"})
    with pytest.raises(ValueError):
        load_settings({**env, "KSMECH_SERVE_AUTO_INDEX": "maybe"})
    with pytest.raises(ValueError):
        load_settings({**env, "KSMECH_SERVE_WORKERS": "two"})
    with pytest.raises(TypeError):
        load_settings({}, data_root=tmp_path, colour="blue")


def test_assert_writable(tmp_path):
    shared = tmp_path / "shared"
    (shared / "data" / "cache" / "qm").mkdir(parents=True)
    (shared / "data" / "processed" / "pmechdb").mkdir(parents=True)
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    os.symlink(shared / "data" / "cache", root / "data" / "linked")
    s = load_settings({}, data_root=root, shared_cache_root=shared)
    assert_writable(s.serve_qm / "x.npz", s)
    assert_writable(s.pools / "s1" / "state.json", s)
    bad_paths = (
        shared / "data" / "cache" / "qm" / "x.npz",
        root / "data" / "linked" / "qm" / "y",  # through a symlink into the shared cache
        shared / "data" / "processed" / "pmechdb" / "steps.parquet",
        root / "data" / "cache" / "tokens" / "z.npz",
        root / "data" / "raw" / "f.csv",
    )
    for bad in bad_paths:
        with pytest.raises(ValueError):
            assert_writable(bad, s)


def test_pin_threads(monkeypatch):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(var, "8")
    pin_threads()
    pin_threads()  # idempotent
    assert all(
        os.environ[v] == "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
    import torch

    assert torch.get_num_threads() == 1


# ---- errors


def test_errors():
    assert STATUS[Code.POOL_NOT_FOUND] == 404 and STATUS[Code.PREDICTOR_UNAVAILABLE] == 409
    assert STATUS[Code.CONSISTENCY_FAILED] == 500 and STATUS[Code.WORKER_UNSTABLE] == 503
    assert STATUS[Code.CUBE_TOO_LARGE] == 422
    assert ServeError(Code.MISSING_TOKENS, "x").status == 422  # job reasons default to 422
    assert ServeError("APPLY_FAILED:UNKNOWN_ARROW", "x").code == "APPLY_FAILED:UNKNOWN_ARROW"
    err = ServeError(Code.EDGE_NOT_FOUND, "no edge", detail={"v": float("nan")})
    assert err.status == 404 and str(err) == "EDGE_NOT_FOUND: no edge"
    body = err.body("r1")
    assert body == {
        "code": "EDGE_NOT_FOUND",
        "message": "no edge",
        "upstream": None,
        "detail": {"v": None},
        "request_id": "r1",
    }
    assert error_body("X", "m") == {
        "code": "X",
        "message": "m",
        "upstream": None,
        "detail": None,
        "request_id": "",
    }
    plain = ServeError.from_task(TaskResult(False, None, "WORKER_TIMEOUT", None, "deadline"))
    assert plain.code == Code.WORKER_TIMEOUT and plain.message == "deadline"
    assert ServeError.from_task(TaskResult(False, None, None, None, "")).code == Code.UNEXPECTED
    from pathlib import Path

    assert jsonable({"a": [np.float32(1.5), np.inf], "p": Path("/x"), 3: (1,)}) == {
        "a": [1.5, None],
        "p": "/x",
        "3": [1],
    }


# ---- featurizers and the layered loader


def _synthetic_files(s, key):
    """(QM npz, token npz) of `key` in the synthetic root of settings `s`."""
    from ksmech.orbitals.build import tokens_path
    from ksmech.qm.records import cache_path
    from ksmech.serve.synthetic import SYNTHETIC_METHOD

    return (
        cache_path(s.serve_qm, key, 0, SYNTHETIC_METHOD),
        tokens_path(s.serve_tokens, key, 0, SYNTHETIC_METHOD),
    )


def test_qm_featurizer_cached_mode(serve_settings, synthetic_settings):  # noqa: F811
    from ksmech.orbitals.build import tokens_path
    from ksmech.qm.records import cache_path
    from ksmech.qm.xtb_calc import QMConfig
    from ksmech.serve.featurize import FeaturizeStatus, make_featurizer

    s = serve_settings(chemistry="cached")
    feat = make_featurizer(s)
    method = QMConfig().method
    assert feat.mode == "cached" and feat.method == method
    res = feat.ensure("CO")
    assert res.status == FeaturizeStatus.FAILED and res.code == "FEATURIZATION_DISABLED"
    qm_src, tok_src = _synthetic_files(synthetic_settings, "CO")
    qm_dst = cache_path(s.shared_qm, "CO", 0, method)
    qm_dst.parent.mkdir(parents=True)
    shutil.copy(qm_src, qm_dst)
    res = feat.ensure("CO")  # a QM record but no tokens: cached mode never builds them
    assert res.code == "FEATURIZATION_DISABLED" and res.qm_path == str(qm_dst)
    tok_dst = tokens_path(s.shared_tokens, "CO", 0, method)
    tok_dst.parent.mkdir(parents=True)
    shutil.copy(tok_src, tok_dst)
    res = feat.ensure("CO")
    assert res.status == FeaturizeStatus.CACHED and res.tokens_path == str(tok_dst)
    assert feat.loader()("CO") is not None
    sp = feat.preflight("CO")
    assert sp.tokens_cached and sp.qm_cached and sp.errors == []
    assert feat.occupations("CO").sum() == 14
    assert not s.serve_qm.exists() and not s.serve_tokens.exists()  # nothing written


def test_qm_featurizer_real_mode_wiring(serve_settings, synthetic_settings, monkeypatch):  # noqa: F811
    from ksmech.orbitals import build
    from ksmech.qm import batch
    from ksmech.qm.records import cache_path
    from ksmech.qm.xtb_calc import QMConfig
    from ksmech.serve.featurize import FeaturizeStatus, make_featurizer

    s = serve_settings(chemistry="real", qm_timeout_s=123)
    feat = make_featurizer(s)
    qm_src, tok_src = _synthetic_files(synthetic_settings, "CO")
    calls = {}

    def fake_qm(task):
        calls["task"] = task
        path = cache_path(task.config.cache_dir, task.smiles, 0, task.config.method)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(qm_src, path)
        return [
            {"status": "ok", "cache_path": str(path), "reason": "", "detail": "", "warnings": "W"}
        ]

    def fake_build(key, config):
        from ksmech.qm.bond_index import bond_index_path
        from ksmech.serve.synthetic import SYNTHETIC_METHOD

        calls["build"] = config
        out = build.tokens_path(config.out_dir, key, 0, config.method)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tok_src, out)
        side = bond_index_path(s.serve_bond_index, key, 0, config.method)  # the Wiberg sidecar
        side.parent.mkdir(parents=True, exist_ok=True)
        src = bond_index_path(synthetic_settings.serve_bond_index, "CO", 0, SYNTHETIC_METHOD)
        shutil.copy(src, side)
        return build.BuildRow(key, "built")

    monkeypatch.setattr(batch, "process_molecule", fake_qm)
    monkeypatch.setattr(build, "process_molecule", fake_build)
    res = feat.ensure("CO")
    assert res.status == FeaturizeStatus.BUILT and res.warnings == "W"
    task = calls["task"]
    assert task.smiles == "CO" and task.n_conformers == 1
    assert task.config.cache_dir == s.serve_qm and task.config.timeout_s == 123
    assert task.config.method == QMConfig().method
    cfg = calls["build"]
    assert cfg.qm_cache_dir == s.serve_qm and cfg.out_dir == s.serve_tokens
    assert feat.ensure("CO").status == FeaturizeStatus.CACHED

    def failing(task):
        return [
            {"status": "failed", "reason": "SCF_NOT_CONVERGED", "detail": "scf", "cache_path": ""}
        ]

    monkeypatch.setattr(batch, "process_molecule", failing)
    res = feat.ensure("CCO")
    assert res.code == "FEATURIZATION_FAILED" and res.upstream == "SCF_NOT_CONVERGED"
    assert "SCF_NOT_CONVERGED" in s.failures_log.read_text()
    monkeypatch.setattr(batch, "process_molecule", fake_qm)
    monkeypatch.setattr(build, "process_molecule", lambda k, c: build.BuildRow(k, "failed", "x"))
    assert feat.ensure("CC").code == "TOKENS_FAILED"


def test_layered_loader(tmp_path, synthetic_settings):  # noqa: F811
    from ksmech.orbitals.build import tokens_path
    from ksmech.serve.loader import LayeredTokenLoader
    from ksmech.serve.synthetic import SYNTHETIC_METHOD
    from ksmech.serve.versions import file_sha256

    _, src = _synthetic_files(synthetic_settings, "CCl")
    shared, serve = tmp_path / "shared", tmp_path / "serve"
    loader = LayeredTokenLoader([shared, serve], SYNTHETIC_METHOD)
    assert not loader.has("CCl") and loader("CCl") is None and loader.digest("CCl") is None
    in_serve = tokens_path(serve, "CCl", 0, SYNTHETIC_METHOD)
    in_serve.parent.mkdir(parents=True)
    shutil.copy(src, in_serve)
    assert loader.has("CCl")  # a miss is never memoized
    assert loader.path_of("CCl") == in_serve and loader.digest("CCl") == file_sha256(in_serve)[:16]
    t = loader("CCl")
    assert t is not None and loader("CCl") is t
    stale = tokens_path(shared, "CCl", 0, SYNTHETIC_METHOD)
    stale.parent.mkdir(parents=True)
    with np.load(src) as z:
        arrays = {k: z[k] for k in z.files if k != "__provenance__"}
    np.savez_compressed(stale, **arrays)  # written before provenance headers: stale
    assert loader.path_of("CCl") == in_serve  # the stale shared file is skipped
    loader.path_of("CCl")
    events = loader.drain_events()
    assert [e["code"] for e in events] == ["TOKENS_STALE"] and loader.stale == {"CCl": str(stale)}
    assert loader.drain_events() == []
    shutil.copy(src, stale)  # a fresh shared file wins (first layer) once the memo is dropped
    assert loader.path_of("CCl") == stale
    loader.forget("CCl")
    assert loader("CCl") is not t


def test_layered_loader_resolves_inputs_against_the_layer_root(tmp_path, synthetic_settings):  # noqa: F811
    """A token file names its QM record relative to the checkout that built it (upstream
    provenance): the shared layer resolves that path against its own root, so a shared cache
    outside this checkout is not stale; a changed or missing record still is."""
    from dataclasses import replace
    from pathlib import Path

    from ksmech.data import provenance as prov
    from ksmech.orbitals.build import tokens_path
    from ksmech.serve.loader import LayeredTokenLoader, token_header_problems
    from ksmech.serve.synthetic import SYNTHETIC_METHOD

    qm, src = _synthetic_files(synthetic_settings, "CCl")
    root = tmp_path / "other-checkout"
    rel = Path("data/cache/qm") / f"serve-test-{qm.name}"
    (root / rel).parent.mkdir(parents=True)
    shutil.copy(qm, root / rel)
    shared = root / "data" / "cache" / "tokens"
    out = tokens_path(shared, "CCl", 0, SYNTHETIC_METHOD)
    with np.load(src) as z:
        arrays = {k: z[k] for k in z.files if k != prov.HEADER_KEY}
    header = replace(prov.read_header(src), inputs={f"file:{rel}": prov.current_hash(qm)})
    prov.save_npz(out, arrays, header)
    assert token_header_problems(out, root) == []
    assert any("input missing" in p for p in token_header_problems(out, None))
    loader = LayeredTokenLoader([shared], SYNTHETIC_METHOD, roots=[root])
    assert loader.path_of("CCl") == out and loader("CCl") is not None
    with pytest.raises(ValueError):
        LayeredTokenLoader([shared], SYNTHETIC_METHOD, roots=[root, None])
    (root / rel).unlink()
    assert LayeredTokenLoader([shared], SYNTHETIC_METHOD, roots=[root]).path_of("CCl") is None


# ---- a token rebuild under the running server


def _rewrite_tokens(path):
    """Rewrite a token file the way a rebuild does (tmp file, os.replace): the same members,
    other bytes (uncompressed), a newer mtime."""
    before = path.read_bytes()
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    st = path.stat()
    tmp = path.with_name(f".{path.name}.rebuild.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert path.read_bytes() != before


@pytest.fixture
def fresh_settings(serve_settings, synthetic_settings, sn2_state):  # noqa: F811
    """Synthetic settings (stub on) on a new data root holding a copy of the synthetic
    tokens, QM records and Wiberg sidecars, with the SN2 state saved: these token files may
    be rewritten (the session's synthetic root never is)."""
    from ksmech.serve.pools import save_state

    s = serve_settings(chemistry="synthetic")
    for name in ("tokens", "qm", "bond_index"):
        if (synthetic_settings.serve / name).is_dir():
            shutil.copytree(synthetic_settings.serve / name, s.serve / name)
    save_state(sn2_state, s)
    return s


def test_layered_loader_follows_a_rewritten_file(tmp_path, synthetic_settings):  # noqa: F811
    from ksmech.orbitals.build import tokens_path
    from ksmech.serve.loader import LayeredTokenLoader
    from ksmech.serve.synthetic import SYNTHETIC_METHOD

    _, src = _synthetic_files(synthetic_settings, "CCl")
    shared, serve = tmp_path / "shared", tmp_path / "serve"
    path = tokens_path(serve, "CCl", 0, SYNTHETIC_METHOD)
    path.parent.mkdir(parents=True)
    shutil.copy(src, path)
    loader = LayeredTokenLoader([shared, serve], SYNTHETIC_METHOD)
    t, digest = loader("CCl"), loader.digest("CCl")
    assert t is not None and loader("CCl") is t  # an unchanged file: the same object
    _rewrite_tokens(path)
    assert loader.digest("CCl") != digest
    t2 = loader("CCl")  # no forget(): the memo follows the file
    assert t2 is not None and t2 is not t and loader("CCl") is t2
    in_shared = tokens_path(shared, "CCl", 0, SYNTHETIC_METHOD)
    in_shared.parent.mkdir(parents=True)
    shutil.copy(src, in_shared)  # an earlier layer gains a usable file: it wins
    t3 = loader("CCl")
    assert t3 is not t2 and loader("CCl") is t3 and loader.path_of("CCl") == in_shared
    in_shared.unlink()  # the memoized file vanishes: the next layer is loaded again
    assert loader.has("CCl")
    t4 = loader("CCl")
    assert t4 is not t3 and loader("CCl") is t4
    path.unlink()
    assert not loader.has("CCl") and loader("CCl") is None


@requires("predict")  # the pool analysis serves predict and correct
def test_api_context_follows_a_token_rebuild(fresh_settings, sn2_state):  # noqa: F811
    from ksmech.serve.context import build_context

    ctx = build_context(fresh_settings)
    rt1 = ctx.reactant_tokens(sn2_state)
    an1 = ctx.ensure_analysis(sn2_state)
    assert ctx.reactant_tokens(sn2_state) is rt1
    _rewrite_tokens(ctx.loader.path_of("CCl"))
    rt2 = ctx.reactant_tokens(sn2_state)
    assert rt2 is not rt1 and ctx.reactant_tokens(sn2_state) is rt2
    ccl = [f.key for f in rt2.fragments].index("CCl")
    assert rt2.fragments[ccl].tokens is not rt1.fragments[ccl].tokens
    an2 = ctx.ensure_analysis(sn2_state)
    assert an2.path != an1.path and an2.meta["analysis_digest"] != an1.meta["analysis_digest"]


@requires("predict")  # the pool analysis serves predict and correct
def test_api_context_passes_tokens_changed_through(fresh_settings, sn2_state, monkeypatch):  # noqa: F811
    from ksmech.serve import analysis
    from ksmech.serve.context import build_context

    ctx = build_context(fresh_settings)
    path = ctx.loader.path_of("CCl")
    real = analysis.analyze_pool

    def rebuild_meanwhile(*args, **kwargs):
        out = real(*args, **kwargs)
        _rewrite_tokens(path)
        return out

    monkeypatch.setattr(analysis, "analyze_pool", rebuild_meanwhile)
    with pytest.raises(ServeError) as e:
        ctx.ensure_analysis(sn2_state)
    assert (e.value.code, e.value.status) == ("TOKENS_CHANGED", 503)
    assert analysis.read_failure(fresh_settings, sn2_state.state_id, ctx.physics_fp) is None
    monkeypatch.undo()
    an = ctx.ensure_analysis(sn2_state)  # the retry succeeds under the new digest
    assert an.meta["analysis_digest"] == analysis.analysis_digest(ctx.loader, sn2_state)
