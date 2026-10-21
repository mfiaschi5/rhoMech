"""The compute seam: a `Scheduler` runs named worker tasks (`tasks.TASKS`) and hands back
`TaskResult`s. `InlineScheduler` runs them synchronously in the caller's thread (pytest);
`procsched.ProcessScheduler` is the production one (one spawned worker process).

`interactive(...)` is how every API-side compute call waits: it deduplicates by key, waits at
most `wait_s`, then raises `errors.Pending` (202, retry the same URL) or, on failure,
`ServeError.from_task`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Protocol

from ksmech.serve.errors import Code, Pending, ServeError
from ksmech.serve.settings import TaskEnv


class Priority(IntEnum):
    INTERACTIVE = 0
    JOB = 1
    BACKGROUND = 2


@dataclass
class TaskResult:
    ok: bool
    result: dict | None
    code: str | None
    upstream: str | None
    detail: str
    events: list[dict] = field(default_factory=list)
    pid: int | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TaskResult:
        return cls(
            ok=bool(d["ok"]),
            result=d.get("result"),
            code=d.get("code"),
            upstream=d.get("upstream"),
            detail=str(d.get("detail") or ""),
            events=list(d.get("events") or []),
            pid=d.get("pid"),
            seconds=float(d.get("seconds") or 0.0),
        )

    @classmethod
    def failure(cls, code: str, detail: str, upstream: str | None = None) -> TaskResult:
        return cls(False, None, str(code), upstream, detail)


@dataclass
class WorkerStatus:
    alive: bool
    n_workers: int
    pids: list[int]
    busy: bool
    current_task: str | None
    current_label: str | None
    elapsed_s: float | None
    queue_depth: int
    restarts: int
    unstable: bool

    def to_dict(self) -> dict:
        """`WorkerStatusOut` fields."""
        return asdict(self)


class TaskHandle:
    """A submitted task. Schedulers complete it with `set_result`; `wait` returns None on
    timeout."""

    def __init__(self, task_id: str, label: str = "") -> None:
        self.task_id = task_id
        self.label = label
        self._done = threading.Event()
        self._result: TaskResult | None = None

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None) -> TaskResult | None:
        if not self._done.wait(timeout):
            return None
        return self._result

    def set_result(self, result: TaskResult) -> None:
        self._result = result
        self._done.set()


class Scheduler(Protocol):
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
    ) -> TaskHandle: ...

    def run(
        self,
        name: str,
        payload: dict,
        *,
        priority: Priority,
        timeout_s: float,
        owner: str = "",
        label: str = "",
    ) -> TaskResult: ...  # submit + wait

    def find(self, key: str) -> TaskHandle | None: ...  # a queued or running task

    def status(self) -> WorkerStatus: ...

    def kill_owner(self, owner: str) -> bool: ...

    def close(self) -> None: ...


def _json_roundtrip(value, what: str):
    """The value as it would arrive through a JSON-able worker boundary (strict NaN-tolerant
    JSON). Raises TypeError for values that are not JSON-able."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"task {what} is not JSON-able: {exc}") from exc


class InlineScheduler:
    """Runs every task synchronously in the caller's thread (`tasks.run_task`); the handle it
    returns is already done. Payloads and results go through a JSON round trip, so a test
    catches anything that would not cross the production worker boundary."""

    def __init__(self, env: TaskEnv) -> None:
        self.env = env
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self._current: tuple[str, str, float] | None = None
        self.n_done = 0
        self.history: list[tuple[str, str]] = []  # (name, label) of every task run

    def submit(self, name, payload, *, priority, timeout_s, owner="", key=None, label=""):
        from ksmech.serve.tasks import run_task

        seq = next(self._seq)
        handle = TaskHandle(f"inline-{seq}", label)
        try:
            payload = _json_roundtrip(payload, "payload")
        except TypeError as exc:
            handle.set_result(TaskResult.failure(Code.UNEXPECTED, str(exc)))
            return handle
        with self._lock:
            self._current = (name, label, time.monotonic())
        try:
            out = run_task(name, self.env, payload, seq)
        finally:
            with self._lock:
                self._current = None
                self.n_done += 1
                self.history.append((name, label))
        try:
            handle.set_result(TaskResult.from_dict(_json_roundtrip(out, "result")))
        except TypeError as exc:
            handle.set_result(TaskResult.failure(Code.UNEXPECTED, str(exc)))
        return handle

    def run(self, name, payload, *, priority, timeout_s, owner="", label=""):
        handle = self.submit(
            name, payload, priority=priority, timeout_s=timeout_s, owner=owner, label=label
        )
        return handle.wait(None)

    def find(self, key: str) -> TaskHandle | None:
        return None

    def status(self) -> WorkerStatus:
        with self._lock:
            current = self._current
        return WorkerStatus(
            alive=True,
            n_workers=1,
            pids=[os.getpid()],
            busy=current is not None,
            current_task=None if current is None else current[0],
            current_label=None if current is None else current[1],
            elapsed_s=None if current is None else time.monotonic() - current[2],
            queue_depth=0,
            restarts=0,
            unstable=False,
        )

    def kill_owner(self, owner: str) -> bool:
        return False

    def close(self) -> None:
        return None


def task_key(name: str, payload: dict) -> str:
    """The default dedup key: name + ":" + sha256(json(payload, sorted keys))[:12]."""
    raw = json.dumps(payload, sort_keys=True)
    return f"{name}:{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def interactive(
    scheduler: Scheduler,
    name: str,
    payload: dict,
    *,
    label: str,
    wait_s: float,
    timeout_s: float,
    priority: Priority = Priority.INTERACTIVE,
    key: str | None = None,
) -> dict:
    """Run (or join) a task and wait at most `wait_s`. Returns the task's result dict; raises
    `Pending` while it runs and `ServeError.from_task` when it failed."""
    key = key or task_key(name, payload)
    handle = scheduler.find(key) or scheduler.submit(
        name, payload, priority=priority, timeout_s=timeout_s, key=key, label=label
    )
    result = handle.wait(wait_s)
    if result is None:
        raise Pending(handle.task_id, label, 1.0, scheduler.status().to_dict())
    if result.ok:
        return result.result or {}
    raise ServeError.from_task(result)
