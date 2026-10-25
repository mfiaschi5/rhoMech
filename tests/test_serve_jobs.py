"""jobs.py: atomic job files, restart recovery (SERVER_RESTART), retention, soft and hard
cancel, failing jobs (the runner survives), inline mode, the handler registry, RunnerJobContext
(progress throttling, skips in events.jsonl and failures.jsonl), NETWORK_BUSY, and the predict
and override handlers on StubChemistry."""

import json
import logging
import os
import threading
import time

import pytest

from ksmech.serve import jobs as jobs_mod
from ksmech.serve.errors import ServeError
from ksmech.serve.jobs import (
    Cancelled,
    Job,
    JobRunner,
    JobStore,
    curated_job_ids,
    new_job_id,
)
from tests.test_serve_fixtures_net import (  # noqa: F401 (fixtures)
    FakeScheduler,
    branching,
    no_xtb,
    serve_settings,
)

pytestmark = pytest.mark.serve

TERMINAL = ("done", "failed", "cancelled")


@pytest.fixture
def env(serve_settings):  # noqa: F811
    """(settings, store, scheduler) on a tmp data root."""
    s = serve_settings()
    return s, JobStore(s.jobs, s), FakeScheduler()


def runner_of(env, **kw) -> JobRunner:
    s, store, sched = env
    r = JobRunner(store, s, sched, **kw)
    r.register("echo", lambda ctx, req: {"echo": req})
    return r


def wait_for(runner, job_id, statuses=TERMINAL, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        job = runner.get(job_id)
        if job.status in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} still {runner.get(job_id).status}")


def old_job(job_id, status, created_at="2026-10-01T00:00:00+00:00"):
    return Job(job_id, "predict", status, created_at, {}, {"phase": "x"}, network_id=job_id)


def test_atomic_job_and_network_files(env, monkeypatch):
    s, store, _ = env
    job = old_job(new_job_id(), "done")
    path = store.save(job)
    assert json.loads(path.read_text()) == job.to_dict()
    assert store.load(job.id) == job
    store.write_network(job.id, {"nodes": [1, 2], "revision": 3})
    assert store.read_network(job.id) == {"nodes": [1, 2], "revision": 3}
    assert sorted(p.name for p in path.parent.iterdir()) == ["job.json", "network.json"]

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(jobs_mod.os, "replace", broken_replace)
    job.status = "failed"
    with pytest.raises(OSError):
        store.save(job)
    monkeypatch.undo()
    assert store.load(job.id).status == "done"  # the old file is intact
    assert sorted(p.name for p in path.parent.iterdir()) == ["job.json", "network.json"]
    with pytest.raises(ServeError) as e:
        store.dir("../x")
    assert (e.value.code, e.value.status) == ("JOB_NOT_FOUND", 404)


def test_restart_recovery(env):
    s, store, _ = env
    ids = {st: new_job_id() for st in ("queued", "running", "done")}
    for st, job_id in ids.items():
        store.save(old_job(job_id, st))
    store.write_network(ids["running"], {"partial": True})
    runner = runner_of(env, inline=True)
    for st in ("queued", "running"):
        job = runner.get(ids[st])
        assert job.status == "failed" and job.error["code"] == "SERVER_RESTART"
        assert st in job.error["message"] and job.finished_at is not None
        assert store.load(ids[st]).status == "failed"  # persisted
    assert runner.get(ids["done"]).status == "done" and runner.get(ids["done"]).error is None
    assert store.read_network(ids["running"]) == {"partial": True}  # still viewable


def test_retention_keeps_newest_and_curated(env):
    s, store, _ = env
    ids = [new_job_id() for _ in range(8)]
    for i, job_id in enumerate(ids):
        store.save(old_job(job_id, "done", f"2026-10-0{i + 1}T00:00:00+00:00"))
    runner = runner_of(env, inline=True, retention=5, protected=lambda: {ids[0]})
    kept = {j.id for j in runner.list(limit=100)}
    assert kept == {ids[0], *ids[3:]}
    assert not store.dir(ids[1]).exists() and store.dir(ids[0]).exists()
    runner_of(env, inline=True, retention=1, protected=lambda: None)  # unknown: keep all
    assert len(store.load_all()) == 6


def test_curated_job_ids_read_the_curation_db(env):
    from ksmech.serve import db

    s, _, _ = env
    assert curated_job_ids(s) == set()
    engine = db.make_engine(s.curation_db)
    db.init_db(engine)
    job_id = new_job_id()
    with db.session(engine) as ses:
        m = db.Mechanism(
            reactants="[]",
            stoichiometry="[]",
            predictor="stub",
            checkpoint_hash="x",
            spec_version="v3",
            status="validated",
        )
        ses.add(m)
        ses.flush()
        doc = json.dumps({"job_id": job_id, "arrows": []})
        step = db.CuratedStep(
            mechanism_id=m.id,
            step_index=0,
            order_group=0,
            reactant_pool_key="A",
            product_pool_key="B",
            arrows_json=doc,
        )
        ses.add(step)
    engine.dispose()
    assert curated_job_ids(s) == {job_id}


def test_inline_mode_and_result_file(env):
    runner = runner_of(env, inline=True)
    job = runner.submit("echo", {"x": 1}, label="echo it")
    assert job.status == "done" and job.started_at and job.finished_at
    assert runner.store.read_json(job.id, "result.json") == {"echo": {"x": 1}}
    assert job.result_path.endswith("result.json") and job.label == "echo it"
    assert job.created_out() == {
        "job_id": job.id,
        "kind": "echo",
        "status": "done",
        "status_url": f"/api/jobs/{job.id}",
        "parent_job_id": None,
    }


def test_unknown_kind_is_422(env):
    with pytest.raises(ServeError) as e:
        runner_of(env, inline=True).submit("nope", {})
    assert (e.value.code, e.value.status) == ("REQUEST_INVALID", 422)
    with pytest.raises(ValueError):
        runner_of(env, inline=True).register("x", lambda c, r: {}, thread="gpu")


def test_failing_jobs_and_the_runner_survives(env):
    runner = runner_of(env)
    try:

        def boom(ctx, req):
            raise RuntimeError("kaboom")

        def coded(ctx, req):
            raise ServeError("APPLY_FAILED", "no", upstream="VALENCE", detail={"i": 1})

        runner.register("boom", boom)
        runner.register("coded", coded)
        a, b, c = (runner.submit(k, {}).id for k in ("boom", "coded", "echo"))
        ja, jb, jc = (wait_for(runner, x) for x in (a, b, c))
        assert ja.status == "failed" and ja.error["code"] == "UNEXPECTED"
        assert ja.error["message"] == "RuntimeError: kaboom"
        assert (jb.error["code"], jb.error["upstream"]) == ("APPLY_FAILED", "VALENCE")
        assert jb.error["detail"] == {"i": 1} and jb.progress["phase"] == "failed"
        assert jc.status == "done"  # the runner thread survived
    finally:
        runner.close()


def test_soft_and_hard_cancel(env):
    runner = runner_of(env)
    started = threading.Event()

    def loop(ctx, req):
        started.set()
        while not ctx.cancelled():
            time.sleep(0.005)
        if req.get("raise"):
            raise Cancelled
        return {"stopped": True}

    runner.register("loop", loop)
    try:
        for force, raise_ in ((False, False), (True, True)):
            started.clear()
            job = runner.submit("loop", {"raise": raise_})
            assert started.wait(5.0)
            assert runner.cancel(job.id, force=force).cancel_requested
            assert wait_for(runner, job.id).status == "cancelled"
        assert runner.scheduler.killed == [job.id]  # only the hard cancel kills
        job = runner.submit("echo", {})
        wait_for(runner, job.id)
        assert runner.cancel(job.id).status == "done"  # a finished job is left as it is
    finally:
        runner.close()


def test_cancel_a_queued_job(env):
    runner = runner_of(env)
    gate, calls = threading.Event(), []
    runner.register("gate", lambda ctx, req: gate.wait(10.0) and {})
    runner.register("count", lambda ctx, req: calls.append(1) or {})
    try:
        blocker = runner.submit("gate", {})
        queued = runner.submit("count", {}, network_id="j" + "a" * 16)
        assert runner.get(queued.id).status == "queued"
        assert runner.busy_job("j" + "a" * 16) == queued.id
        with pytest.raises(ServeError) as e:
            runner.submit("count", {}, network_id="j" + "a" * 16, exclusive=True)
        assert (e.value.code, e.value.status) == ("NETWORK_BUSY", 409)
        assert e.value.detail["job_id"] == queued.id
        assert runner.cancel(queued.id).status == "cancelled"
        gate.set()
        wait_for(runner, blocker.id)
        runner.submit("echo", {})
        time.sleep(0.05)
        assert calls == [] and runner.get(queued.id).status == "cancelled"
        assert runner.busy_job("j" + "a" * 16) is None
    finally:
        gate.set()
        runner.close()


def test_progress_throttling(env, monkeypatch):
    runner = runner_of(env, inline=True)
    writes = []
    real_save = runner.store.save
    monkeypatch.setattr(
        runner.store, "save", lambda job: writes.append(job.progress) or real_save(job)
    )

    def chatty(ctx, req):
        for i in range(50):
            ctx.progress(phase="expand", done=i)
        ctx.progress(phase="pathways")
        return {}

    runner.register("chatty", chatty)
    job = runner.submit("chatty", {})
    during = [w for w in writes if w.get("phase") in ("expand", "pathways")]
    assert 2 <= len(during) <= 4  # first expand, the phase change, maybe one tick
    assert during[0]["done"] == 0 and during[-1]["phase"] == "pathways"
    assert runner.get(job.id).progress["done"] == 49  # every call reaches the live job


def test_skips_go_to_events_and_failures(env, caplog):
    s, store, _ = env
    runner = runner_of(env, inline=True)

    def skipper(ctx, req):
        ctx.skip("APPLY_FAILED:VALENCE", "3 proposals", node="p" + "1" * 20, count=3)
        ctx.skip("RESONANCE_ONLY_STEP")
        return {}

    runner.register("skipper", skipper)
    with caplog.at_level(logging.WARNING, logger="ksmech.serve.jobs"):
        job = runner.submit("skipper", {})
    assert runner.get(job.id).skipped == {"APPLY_FAILED:VALENCE": 3, "RESONANCE_ONLY_STEP": 1}
    events = store.events(job.id)
    assert [(e["code"], e["count"]) for e in events] == [
        ("APPLY_FAILED:VALENCE", 3),
        ("RESONANCE_ONLY_STEP", 1),
    ]
    assert events[0]["node"] == "p" + "1" * 20
    failures = [json.loads(x) for x in s.failures_log.read_text().splitlines()]
    assert [f["job_id"] for f in failures] == [job.id, job.id]
    assert "APPLY_FAILED:VALENCE" in caplog.text


def test_snapshot_throttle(env):
    runner = runner_of(env, inline=True)
    built = []

    def snap(ctx, req):
        for _ in range(5):
            ctx.snapshot_network(lambda: built.append(1) or {"n": len(built)})
        assert ctx.snapshot_network({"n": "final"}, force=True)
        return {}

    runner.register("snap", snap)
    job = runner.submit("snap", {}, network_id=None)
    assert built == [1]  # the thunk is not even built while throttled
    assert runner.store.read_network(job.id) == {"n": "final"}
    assert not [p for p in os.listdir(runner.store.dir(job.id)) if p.endswith(".tmp")]
