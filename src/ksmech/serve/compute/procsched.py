"""The production scheduler (design 2.5, 5.1, 5.2): one dispatcher thread feeding a spawned
`ProcessPoolExecutor` that runs `tasks.run_task` in one (at most two) compute workers.

- The executor has `max_workers` = `workers` clamped to [1, 2], the `spawn` context and the
  initializer `ksmech.qm.batch._init_worker` (the SIGALRM handler of the QM timeout). It is
  created, and its first n submits (warm-up pings) made, inside
  `ksmech.processes.single_threaded_children()`, so every worker starts with one BLAS /
  OpenMP thread.
- Queued tasks wait in a (priority, seq) heap: INTERACTIVE before JOB before BACKGROUND, FIFO
  within a priority; at most n tasks are in flight. Tasks with the same key are deduplicated
  (the default key is `worker.task_key(name, payload)`), and `find(key)` returns the handle of
  a queued or running one.
- A task past its deadline is SIGKILLed through the pid its `run_task` wrote to
  `<run>/task-<seq>.pid` (WORKER_TIMEOUT); a dead worker (`BrokenProcessPool`) is
  WORKER_CRASH. Either way the executor is discarded and the broken in-flight siblings are
  resubmitted once. Only processes that are children of this process are ever signalled.
- More than 5 timeouts or crashes within 10 minutes trip the circuit breaker: every task then
  fails at once with WORKER_UNSTABLE (503) until old failures age out.
- The worker is recycled gracefully (when idle) after 500 tasks, when a worker's RSS exceeds
  2048 MB, or when a task reports ORIENTED_CACHE_WIPED (the upstream in-memory oriented caches
  are stale).

The private `executor._processes` is never used: worker pids come from the children that
appear when the executor spawns, from the pid files, and from every task result.
"""

from __future__ import annotations

import collections
import contextlib
import heapq
import itertools
import logging
import os
import signal
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

from ksmech.serve.compute.worker import Priority, TaskHandle, TaskResult, WorkerStatus, task_key
from ksmech.serve.core.errors import Code
from ksmech.serve.core.settings import TaskEnv

log = logging.getLogger(__name__)

MAX_WORKERS = 2
RECYCLE_AFTER = 500
MAX_RSS_MB = 2048.0
BREAKER_FAILURES = 5
BREAKER_WINDOW_S = 600.0
MAX_WAIT_S = 1.0
CLOSE_WAIT_S = 10.0
_PROC = Path("/proc")


def _proc_stat(pid: int) -> tuple[str, int] | None:
    """(state, ppid) of a process from /proc/<pid>/stat, or None when it is gone."""
    try:
        raw = (_PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    rest = raw[raw.rfind(")") + 2 :].split()
    return rest[0], int(rest[1])


def _is_live_child(pid: int) -> bool:
    st = _proc_stat(pid)
    return st is not None and st[0] != "Z" and st[1] == os.getpid()


def _spawned_children() -> set[int]:
    """Pids of this process's live multiprocessing spawn children (/proc scan)."""
    out = set()
    with contextlib.suppress(OSError):
        for entry in _PROC.iterdir():
            if not entry.name.isdigit() or not _is_live_child(int(entry.name)):
                continue
            try:
                cmd = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if b"spawn_main" in cmd:
                out.add(int(entry.name))
    return out


def rss_mb(pid: int) -> float | None:
    """Resident set size of a process in MB (/proc/<pid>/status VmRSS), or None."""
    try:
        for line in (_PROC / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass(eq=False)
class _Item:
    seq: int
    name: str
    payload: dict
    priority: int
    timeout_s: float
    key: str
    label: str
    handle: TaskHandle
    owners: list[str] = field(default_factory=list)
    attempts: int = 0
    future: Future | None = None
    gen: int = 0
    dispatched: float | None = None
    deadline: float | None = None
    kill_code: str | None = None
    kill_detail: str = ""


class ProcessScheduler:
    """The production `Scheduler` (see the module docstring)."""

    def __init__(
        self,
        env: TaskEnv,
        workers: int = 1,
        *,
        recycle_after: int = RECYCLE_AFTER,
        max_rss_mb: float = MAX_RSS_MB,
        breaker_failures: int = BREAKER_FAILURES,
        breaker_window_s: float = BREAKER_WINDOW_S,
    ) -> None:
        self.env = env
        self.n_workers = min(MAX_WORKERS, max(1, int(workers)))
        self.recycle_after = int(recycle_after)
        self.max_rss_mb = float(max_rss_mb)
        self.breaker_failures = int(breaker_failures)
        self.breaker_window_s = float(breaker_window_s)
        self.n_done = 0
        self.history: list[tuple[str, str]] = []  # (name, label) of every dispatched task
        self._lock = threading.RLock()
        self._kick = threading.Event()
        self._seq = itertools.count(1)
        self._heap: list[tuple[int, int, _Item]] = []
        self._by_key: dict[str, _Item] = {}
        self._inflight: dict[int, _Item] = {}
        self._executor: ProcessPoolExecutor | None = None
        self._gen = 0
        self._gen_done = 0
        self._pids: set[int] = set()
        self._warm: list[Future] = []
        self._restarts = 0
        self._failures: collections.deque[float] = collections.deque()
        self._recycle = False
        self._closing = False
        self._next_wait = MAX_WAIT_S
        self._thread = threading.Thread(target=self._loop, name="ksmech-dispatcher", daemon=True)
        self._thread.start()

    # ---- the Scheduler protocol

    def submit(
        self,
        name: str,
        payload: dict,
        *,
        priority: Priority,
        timeout_s: float,
        owner: str = "",
        key: str | None = None,
        label: str = "",
    ) -> TaskHandle:
        key = key or task_key(name, payload)
        with self._lock:
            item = self._by_key.get(key)
            if item is not None and not item.handle.done():
                if owner and owner not in item.owners:
                    item.owners.append(owner)
                return item.handle
            seq = next(self._seq)
            handle = TaskHandle(f"t{seq}", label or name)
            if self._closing:
                handle.set_result(TaskResult.failure(Code.UNEXPECTED, "the scheduler is closed"))
                return handle
            if self._unstable():
                handle.set_result(self._unstable_result())
                return handle
            item = _Item(seq, name, payload, int(priority), float(timeout_s), key, label, handle)
            item.owners = [owner] if owner else []
            self._by_key[key] = item
            heapq.heappush(self._heap, (item.priority, seq, item))
        self._kick.set()
        return handle

    def run(
        self,
        name: str,
        payload: dict,
        *,
        priority: Priority,
        timeout_s: float,
        owner: str = "",
        label: str = "",
    ) -> TaskResult:
        handle = self.submit(
            name, payload, priority=priority, timeout_s=timeout_s, owner=owner, label=label
        )
        return handle.wait(None)

    def find(self, key: str) -> TaskHandle | None:
        with self._lock:
            item = self._by_key.get(key)
            return None if item is None or item.handle.done() else item.handle

    def status(self) -> WorkerStatus:
        with self._lock:
            cur = self._current_item()
            now = time.monotonic()
            return WorkerStatus(
                alive=not self._closing and self._thread.is_alive(),
                n_workers=self.n_workers,
                pids=sorted(self._pids),
                busy=bool(self._inflight),
                current_task=None if cur is None else cur.name,
                current_label=None if cur is None else cur.label,
                elapsed_s=None if cur is None else now - (cur.dispatched or now),
                queue_depth=self._queue_depth(),
                restarts=self._restarts,
                unstable=self._unstable(),
            )

    def current(self) -> dict | None:
        """The running task (the oldest in flight): {name, label, elapsed_s, queue_depth,
        priority, owners}, or None when idle."""
        with self._lock:
            cur = self._current_item()
            if cur is None:
                return None
            return {
                "name": cur.name,
                "label": cur.label,
                "elapsed_s": time.monotonic() - (cur.dispatched or time.monotonic()),
                "queue_depth": self._queue_depth(),
                "priority": Priority(cur.priority).name,
                "owners": list(cur.owners),
            }

    def kill_owner(self, owner: str) -> bool:
        """Hard cancel: SIGKILL the worker running a task of `owner` (its result is CANCELLED)
        and cancel its queued tasks. A task shared with another owner keeps running for it."""
        if not owner:
            return False
        with self._lock:
            hit = False
            for _, _, item in self._heap:
                if owner in item.owners and not item.handle.done():
                    if item.owners == [owner]:
                        self._finish(
                            item, TaskResult.failure(Code.CANCELLED, f"cancelled by {owner}")
                        )
                        hit = True
                    else:
                        item.owners.remove(owner)
            for item in list(self._inflight.values()):
                if item.owners == [owner]:
                    item.kill_code, item.kill_detail = Code.CANCELLED, f"killed for {owner}"
                    self._kill_item(item)
                    hit = True
                elif owner in item.owners:
                    item.owners.remove(owner)
            if any(it.kill_code for it in self._inflight.values()):
                self._broken_pool()
        self._kick.set()
        return hit

    def close(self) -> None:
        """Fail every queued and running task, stop the dispatcher, stop the workers."""
        with self._lock:
            if self._closing:
                return
            self._closing = True
            closed = TaskResult.failure(Code.UNEXPECTED, "the scheduler was closed")
            for _, _, item in self._heap:
                self._finish(item, closed)
            self._heap.clear()
            for item in list(self._inflight.values()):
                self._kill_item(item)
                self._finish(item, closed)
            ex, pids = self._executor, set(self._pids)
            self._executor = None
        self._kick.set()
        self._thread.join(timeout=5.0)
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + CLOSE_WAIT_S
        while any(_is_live_child(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        for pid in pids:
            if _is_live_child(pid):
                log.warning("worker %d did not exit after close: SIGKILL", pid)
                self._signal(pid)

    # ---- dispatcher

    def _loop(self) -> None:
        while True:
            self._kick.wait(self._next_wait)
            self._kick.clear()
            with self._lock:
                if self._closing:
                    return
                try:
                    self._step()
                except Exception:  # noqa: BLE001 - the dispatcher must never die
                    log.exception("dispatcher step failed")
                    self._next_wait = MAX_WAIT_S

    def _step(self) -> None:
        self._reap()
        self._expire()
        self._maybe_recycle()
        self._fill()
        now = time.monotonic()
        deadlines = [it.deadline for it in self._inflight.values() if it.deadline is not None]
        wait = min([d - now for d in deadlines], default=MAX_WAIT_S)
        self._next_wait = min(MAX_WAIT_S, max(0.01, wait))

    def _fill(self) -> None:
        while self._heap and len(self._inflight) < self.n_workers and not self._closing:
            _, _, item = heapq.heappop(self._heap)
            if item.handle.done():
                continue
            if self._unstable():
                self._finish(item, self._unstable_result())
                continue
            self._dispatch(item)

    def _dispatch(self, item: _Item) -> None:
        from ksmech.serve.runtime.tasks import run_task

        item.attempts += 1
        try:
            ex = self._executor_or_spawn()
            fut = ex.submit(run_task, item.name, self.env, item.payload, item.seq)
        except (BrokenProcessPool, RuntimeError, OSError) as exc:
            log.error("cannot submit %s to the worker: %s", item.name, exc)
            if item.attempts >= 2:
                self._finish(item, TaskResult.failure(Code.WORKER_CRASH, f"submit failed: {exc}"))
            else:
                self._requeue(item)
            self._broken_pool()  # the in-flight siblings, the executor and the breaker
            return
        now = time.monotonic()
        item.future, item.gen = fut, self._gen
        item.dispatched, item.deadline = now, now + item.timeout_s
        self._inflight[item.seq] = item
        self.history.append((item.name, item.label))
        fut.add_done_callback(self._on_done)

    def _on_done(self, _fut: Future) -> None:
        self._kick.set()  # never takes the scheduler lock (called from executor threads)

    def _executor_or_spawn(self) -> ProcessPoolExecutor:
        if self._executor is not None:
            return self._executor
        from ksmech.processes import single_threaded_children
        from ksmech.qm import batch
        from ksmech.serve.runtime.tasks import run_task

        before = _spawned_children()
        with single_threaded_children():
            ex = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=get_context("spawn"),
                initializer=batch._init_worker,
            )
            self._gen += 1
            self._warm = []
            for _ in range(self.n_workers):  # warm-up pings: the workers spawn here
                fut = ex.submit(run_task, "ping", self.env, {}, next(self._seq))
                fut.add_done_callback(self._on_done)
                self._warm.append(fut)
        self._executor, self._gen_done = ex, 0
        self._pids = _spawned_children() - before
        log.info("compute worker generation %d started: pids %s", self._gen, sorted(self._pids))
        return ex

    # ---- completions, deadlines, breakage

    def _reap(self) -> None:
        broken = False
        for fut in list(self._warm):
            if fut.done():
                self._warm.remove(fut)
                broken |= self._absorb_warm(fut)
        for item in list(self._inflight.values()):
            fut = item.future
            if fut is None or not fut.done():
                continue
            if fut.cancelled():
                del self._inflight[item.seq]
                self._requeue(item)
                continue
            exc = fut.exception()
            if isinstance(exc, BrokenProcessPool):
                broken = True
            elif exc is not None:
                detail = f"{type(exc).__name__}: {exc}"
                self._finish(item, TaskResult.failure(Code.UNEXPECTED, detail))
            else:
                self._completed(item, fut.result())
        if broken:
            self._broken_pool()

    def _absorb_warm(self, fut: Future) -> bool:
        """Record a warm-up ping's pid; True when it found the pool broken."""
        if fut.cancelled():
            return False
        exc = fut.exception()
        if exc is not None:
            log.error("worker warm-up failed: %s", exc)
            return isinstance(exc, BrokenProcessPool)
        pid = (fut.result() or {}).get("pid")
        if pid:
            self._pids.add(int(pid))
        return False

    def _completed(self, item: _Item, out: dict) -> None:
        try:
            res = TaskResult.from_dict(out)
        except (KeyError, TypeError, ValueError) as exc:
            res = TaskResult.failure(Code.UNEXPECTED, f"malformed task result: {exc}")
        self.n_done += 1
        self._gen_done += 1
        if res.pid:
            self._pids.add(int(res.pid))
        if any(ev.get("code") == Code.ORIENTED_CACHE_WIPED for ev in res.events):
            log.warning("oriented cache wiped in the worker: recycling it when idle")
            self._recycle = True
        if self._gen_done >= self.recycle_after:
            self._recycle = True
        for pid in self._pids:
            rss = rss_mb(pid)
            if rss is not None and rss > self.max_rss_mb:
                log.warning("worker %d RSS %.0f MB > %.0f MB: recycling", pid, rss, self.max_rss_mb)
                self._recycle = True
        self._finish(item, res)

    def _expire(self) -> None:
        now = time.monotonic()
        expired = [
            it for it in self._inflight.values() if it.deadline is not None and now >= it.deadline
        ]
        for item in expired:
            item.kill_code = Code.WORKER_TIMEOUT
            item.kill_detail = f"{item.name} exceeded its {item.timeout_s:g} s deadline"
            self._kill_item(item)
        if expired:
            self._broken_pool()

    def _broken_pool(self) -> None:
        """The current executor is broken (a crash, or a kill of ours): discard it; killed
        items fail with their code, the other in-flight items are resubmitted once."""
        items = list(self._inflight.values())  # every in-flight task runs on this executor
        self._discard_executor()
        for it in items:  # a dead task never reached the `finally` that removes its pid file
            with contextlib.suppress(OSError):
                self._pid_file(it).unlink()
        killed = [it for it in items if it.kill_code]
        for it in killed:
            self._finish(it, TaskResult.failure(it.kill_code, it.kill_detail))
            if it.kill_code == Code.WORKER_TIMEOUT:
                self._record_failure()
        siblings = [it for it in items if not it.kill_code]
        if not killed:
            self._record_failure()
            if len(siblings) == 1:  # the only task in flight is the culprit
                detail = f"the worker died while running {siblings[0].name}"
                self._finish(siblings[0], TaskResult.failure(Code.WORKER_CRASH, detail))
                return
        for it in siblings:
            self._inflight.pop(it.seq, None)
            if it.attempts >= 2:
                detail = f"the worker died twice while running {it.name}"
                self._finish(it, TaskResult.failure(Code.WORKER_CRASH, detail))
            else:
                self._requeue(it)

    def _pid_file(self, item: _Item) -> Path:
        return self.env.settings.run / f"task-{item.seq}.pid"

    def _kill_item(self, item: _Item) -> None:
        pid_file = self._pid_file(item)
        try:
            pids = {int(pid_file.read_text().strip())}
        except (OSError, ValueError):
            pids = set(self._pids)  # the task has not started: stop the whole generation
            log.warning("no pid file for %s (seq %d): killing %s", item.name, item.seq, pids)
        for pid in pids:
            self._signal(pid)
        with contextlib.suppress(OSError):
            pid_file.unlink()
        item.kill_detail += f" (SIGKILL {sorted(pids)})"

    def _signal(self, pid: int) -> None:
        if not _is_live_child(pid):  # never signal anything but our own live worker
            return
        try:
            os.kill(pid, signal.SIGKILL)
            log.warning("sent SIGKILL to compute worker %d", pid)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            log.error("cannot kill worker %d: %s", pid, exc)

    def _discard_executor(self) -> None:
        ex, self._executor = self._executor, None
        self._warm = []
        self._pids = set()
        if ex is not None:
            self._restarts += 1
            ex.shutdown(wait=False, cancel_futures=True)

    def _maybe_recycle(self) -> None:
        if not self._recycle or self._executor is None or self._inflight:
            return
        ex, self._executor = self._executor, None
        log.info("recycling compute worker generation %d (%d tasks)", self._gen, self._gen_done)
        ex.shutdown(wait=True, cancel_futures=True)  # idle: the workers exit on the sentinel
        self._recycle, self._warm, self._pids = False, [], set()
        self._restarts += 1

    # ---- book-keeping

    def _finish(self, item: _Item, res: TaskResult) -> None:
        self._inflight.pop(item.seq, None)
        if self._by_key.get(item.key) is item:
            del self._by_key[item.key]
        if not item.handle.done():
            item.handle.set_result(res)

    def _requeue(self, item: _Item) -> None:
        item.future, item.dispatched, item.deadline = None, None, None
        heapq.heappush(self._heap, (item.priority, item.seq, item))

    def _current_item(self) -> _Item | None:
        running = [it for it in self._inflight.values() if it.dispatched is not None]
        return min(running, key=lambda it: it.dispatched, default=None)

    def _queue_depth(self) -> int:
        return sum(not it.handle.done() for _, _, it in self._heap)

    def _record_failure(self) -> None:
        self._failures.append(time.monotonic())

    def _unstable(self) -> bool:
        cutoff = time.monotonic() - self.breaker_window_s
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        return len(self._failures) > self.breaker_failures

    def _unstable_result(self) -> TaskResult:
        n, window = len(self._failures), self.breaker_window_s
        detail = f"{n} worker timeouts or crashes within {window:g} s: the worker is unstable"
        return TaskResult.failure(Code.WORKER_UNSTABLE, detail)
