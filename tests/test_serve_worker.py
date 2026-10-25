"""procsched.ProcessScheduler, the ONLY serve test that spawns a process: exactly one worker,
`TaskEnv(settings, enable_diag_tasks=True)`, only the ping, diag_sleep and diag_crash tasks,
deadlines of a few seconds, and `close()` in a finally block.

The tests share one module-scoped scheduler and run in file order (ping, priorities, dedup,
timeout, crash, hard cancel, recycle, close); the circuit breaker uses a second scheduler with
a zero failure budget, so a single crash trips it. No ksmech compute runs in the worker.
"""

import os
import time

import pytest

from ksmech.serve.procsched import ProcessScheduler, _is_live_child, rss_mb
from ksmech.serve.settings import THREAD_VARS, TaskEnv, load_settings
from ksmech.serve.worker import Priority, interactive, task_key
from tests.test_serve_fixtures import no_xtb  # noqa: F401 (fixture)

pytestmark = pytest.mark.serve

PING_TIMEOUT_S = 120.0  # a fresh worker imports numpy, RDKit and torch on one core


def _env(tmp_path_factory, name):
    root = tmp_path_factory.mktemp(name)
    s = load_settings({}, data_root=root, chemistry="synthetic")
    return TaskEnv(s, enable_diag_tasks=True)


@pytest.fixture(scope="module")
def sched(tmp_path_factory):
    env = _env(tmp_path_factory, "serve-worker")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OMP_NUM_THREADS", "3")  # single_threaded_children() must pin it to 1
        sc = ProcessScheduler(env, workers=1)
        try:
            yield sc
        finally:
            sc.close()


def ping(sc) -> dict:
    r = sc.run("ping", {}, priority=Priority.INTERACTIVE, timeout_s=PING_TIMEOUT_S)
    assert r.ok, (r.code, r.detail)
    return r.result


def until(cond, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def sleep_task(sc, seconds, label, priority=Priority.JOB, timeout_s=30.0, **kw):
    payload = {"seconds": seconds, "tag": label}
    return sc.submit(
        "diag_sleep", payload, priority=priority, timeout_s=timeout_s, label=label, **kw
    )


def test_ping_threads_and_sigalrm(sched):
    out = ping(sched)
    assert out["threads"] == dict.fromkeys(THREAD_VARS, "1")
    assert out["torch_threads"] == 1
    assert out["sigalrm"] == "ksmech.qm.batch._on_alarm"
    st = sched.status()
    assert out["pid"] != os.getpid() and out["pid"] in st.pids and _is_live_child(out["pid"])
    assert st.alive and st.n_workers == 1 and not st.unstable and st.queue_depth == 0


def test_priority_order_and_fifo(sched):
    blocker = sleep_task(sched, 1.0, "blocker")
    assert until(lambda: sched.status().busy and sched.status().queue_depth == 0)
    order = [
        (Priority.BACKGROUND, "bg1"),
        (Priority.JOB, "job1"),
        (Priority.INTERACTIVE, "int1"),
        (Priority.JOB, "job2"),
        (Priority.INTERACTIVE, "int2"),
    ]
    handles = [sleep_task(sched, 0.0, label, priority=p) for p, label in order]
    assert sched.status().queue_depth == 5 and sched.status().current_label == "blocker"
    assert blocker.wait(30.0).ok and all(h.wait(30.0).ok for h in handles)
    labels = {label for _, label in order}
    assert [lab for _, lab in sched.history if lab in labels] == [
        "int1",
        "int2",
        "job1",
        "job2",
        "bg1",
    ]


def test_dedup_by_key_and_find(sched):
    a = sleep_task(sched, 0.5, "a", key="k-dedup")
    b = sleep_task(sched, 0.5, "b", key="k-dedup")
    assert a is b and sched.find("k-dedup") is a
    assert a.wait(30.0).ok and sched.find("k-dedup") is None
    payload = {"seconds": 0.3, "tag": "same"}
    c = sched.submit("diag_sleep", payload, priority=Priority.JOB, timeout_s=30.0)
    d = sched.submit("diag_sleep", dict(payload), priority=Priority.JOB, timeout_s=30.0)
    assert c is d and sched.find(task_key("diag_sleep", payload)) is c
    joined = interactive(sched, "diag_sleep", payload, label="same", wait_s=30.0, timeout_s=30.0)
    assert joined == {"slept": 0.3} and c.wait(0).result == {"slept": 0.3}


def test_deadline_sigkills_by_pid_file(sched):
    before = ping(sched)["pid"]
    t0 = time.monotonic()
    r = sched.run("diag_sleep", {"seconds": 60.0}, priority=Priority.JOB, timeout_s=2.0)
    assert r.code == "WORKER_TIMEOUT" and f"SIGKILL [{before}]" in r.detail, r.detail
    assert time.monotonic() - t0 < 15.0
    assert until(lambda: not _is_live_child(before))
    after = ping(sched)["pid"]
    assert after != before and sched.status().restarts >= 1
    assert list(sched.env.settings.run.glob("task-*.pid")) == []


def test_crash_and_recovery(sched):
    before = ping(sched)["pid"]
    r = sched.run("diag_crash", {}, priority=Priority.JOB, timeout_s=30.0)
    assert r.code == "WORKER_CRASH" and "diag_crash" in r.detail
    assert until(lambda: not _is_live_child(before))
    assert ping(sched)["pid"] != before and not sched.status().unstable
    assert list(sched.env.settings.run.glob("task-*.pid")) == []


def test_kill_owner_is_a_hard_cancel(sched):
    pid = ping(sched)["pid"]
    h = sleep_task(sched, 60.0, "owned", timeout_s=120.0, owner="j-owner")
    pid_file = sched.env.settings.run / f"task-{h.task_id[1:]}.pid"
    assert until(lambda: sched.status().current_label == "owned" and pid_file.is_file())
    assert not sched.kill_owner("j-nobody")
    assert sched.kill_owner("j-owner")
    r = h.wait(15.0)
    assert r is not None and r.code == "CANCELLED"
    assert until(lambda: not _is_live_child(pid))
    assert not sched.status().unstable  # a cancel is not a worker failure


def test_recycle_after_n_tasks(sched):
    restarts = sched.status().restarts
    sched.recycle_after = 1
    try:
        first = ping(sched)["pid"]
    finally:
        sched.recycle_after = 500
    second = ping(sched)["pid"]
    assert second != first and sched.status().restarts == restarts + 1
    assert rss_mb(os.getpid()) > 1.0 and rss_mb(2**22 + 12345) is None


def test_close_stops_the_worker(sched):
    pid = ping(sched)["pid"]
    queued = sleep_task(sched, 30.0, "closing")
    sched.close()
    assert queued.wait(5.0).code == "UNEXPECTED"
    assert until(lambda: not _is_live_child(pid))
    assert not sched.status().alive
    late = sched.run("ping", {}, priority=Priority.INTERACTIVE, timeout_s=5.0)
    assert late.code == "UNEXPECTED" and "closed" in late.detail


def test_circuit_breaker(tmp_path_factory):
    sc = ProcessScheduler(_env(tmp_path_factory, "serve-breaker"), workers=1, breaker_failures=0)
    try:
        r = sc.run("diag_crash", {}, priority=Priority.JOB, timeout_s=PING_TIMEOUT_S)
        assert r.code == "WORKER_CRASH"
        assert sc.status().unstable
        t0 = time.monotonic()
        r = sc.run("ping", {}, priority=Priority.INTERACTIVE, timeout_s=PING_TIMEOUT_S)
        assert r.code == "WORKER_UNSTABLE" and time.monotonic() - t0 < 1.0  # never spawned
        assert sc.status().pids == []
    finally:
        sc.close()


def test_recycle_flags_and_breaker_window_without_a_worker(tmp_path_factory):
    from ksmech.serve.procsched import _Item
    from ksmech.serve.worker import TaskHandle

    sc = ProcessScheduler(_env(tmp_path_factory, "serve-flags"), workers=5, breaker_failures=1)
    try:
        assert sc.n_workers == 2  # clamped to [1, 2]
        item = _Item(1, "expand", {}, int(Priority.JOB), 1.0, "k", "x", TaskHandle("t1"))
        out = {"ok": True, "result": {}, "code": None, "upstream": None, "detail": ""}
        out |= {"events": [{"code": "ORIENTED_CACHE_WIPED"}], "pid": None, "seconds": 0.0}
        with sc._lock:
            sc._completed(item, out)
            assert sc._recycle and item.handle.done() and item.handle.wait(0).ok
            sc._failures.extend([time.monotonic() - 3600.0] * 5)  # outside the window
            assert not sc._unstable()
            sc._failures.extend([time.monotonic()] * 2)
            assert sc._unstable() and sc.status().unstable is True
        assert sc.status().pids == [] and sc.history == []  # nothing was ever spawned
    finally:
        sc.close()
