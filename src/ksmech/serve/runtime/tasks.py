"""Worker tasks: the registry, the per-process runtime and `run_task` (design 5.2).

`run_task(name, env, payload, seq)` is top-level and picklable and never raises: it writes
`<run_dir>/task-<seq>.pid` (the scheduler kills a hung task by that pid), runs
`TASKS[name](runtime(env), payload)` and returns the `TaskResult` fields. A `ServeError` keeps
its code, and its status, message and detail travel in `result["error"]`; anything else is
`UNEXPECTED` with "Type: message" (traceback logged).

`TASKS` is fixed in wave 1: local bodies, or lazy "module:function" references resolved with
importlib on first use, so a package that has not landed yet gives
`UNEXPECTED: task body not installed: <ref>`, never an import error.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ksmech.serve.chem.featurize import FeaturizeStatus, make_featurizer
from ksmech.serve.core.contracts import LRU, PoolState
from ksmech.serve.core.errors import Code, ServeError, jsonable
from ksmech.serve.core.settings import THREAD_VARS, TaskEnv, assert_writable

if TYPE_CHECKING:
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.serve.predict.predictors import Predictor, PredictorRegistry

log = logging.getLogger(__name__)

DIAG_TASKS = frozenset({"diag_sleep", "diag_crash"})
MAX_DIAG_SLEEP_S = 600.0
_RUNTIMES: dict[TaskEnv, WorkerRuntime] = {}
_RUNTIME_LOCK = threading.Lock()
_BODIES: dict[str, Callable] = {}


def _torch_one_thread() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(1)
    return int(torch.get_num_threads())


class WorkerRuntime:
    """The worker-side per-process state (`contracts.Runtime`): settings, featurizer, loader,
    the lazy scorers, the ReactantTokens and sample LRUs (by state_id,
    served only while the state's token digest is the one they were built with)."""

    def __init__(self, env: TaskEnv) -> None:
        _torch_one_thread()
        self.env = env
        self.settings = env.settings
        self.featurizer = make_featurizer(self.settings)
        self.loader = self.featurizer.loader()
        self.rt_cache: LRU[str, ReactantTokens] = LRU(16)
        self.sample_cache: LRU[str, Any] = LRU(16)
        self._rt_digests: dict[str, str | None] = {}  # state_id -> analysis digest of rt_cache
        self._registry: PredictorRegistry | None = None
        self._lock = threading.RLock()
        self._events: list[dict] = []

    @property
    def scorers(self) -> dict[str, Predictor]:
        """Available predictors by name. The registry is built once; availability is asked on
        every call, so a GBDT file that appears, changes or goes away is followed (the GBDT
        predictor re-checks its file only when its mtime or size changed)."""
        with self._lock:
            if self._registry is None:
                from ksmech.serve.core.versions import physics_fingerprint
                from ksmech.serve.predict.predictors import PredictorRegistry

                self._registry = PredictorRegistry(self.settings, physics_fingerprint())
            return self._registry.available()

    def reactant_tokens(self, state: PoolState) -> ReactantTokens:
        """From `rt_cache` while the state's token digest is the one it was built with, else
        `ReactantTokens.from_smiles` (MissingTokens propagates). A changed digest evicts the
        state from `rt_cache` and `sample_cache` and its keys from the loader memo."""
        from ksmech.serve.science.analysis import analysis_digest  # a lazy task module, like TASKS

        sid = state.state_id
        dig = analysis_digest(self.loader, state)
        with self._lock:
            if self._rt_digests.get(sid) != dig:
                self.rt_cache.pop(sid)
                self.sample_cache.pop(sid)
                if sid in self._rt_digests:  # built before with other token files
                    for k in set(state.fragment_token_keys):
                        self.loader.forget(k)
        rt = self.rt_cache.get(sid)
        if rt is None:
            from ksmech.arrows.reactants import ReactantTokens

            rt = ReactantTokens.from_smiles(state.mapped_smiles, self.loader)
            with self._lock:
                self.rt_cache.put(sid, rt)
                self._rt_digests[sid] = dig
        return rt

    def occupations(self, key: str):
        return self.featurizer.occupations(key)

    def event(self, code: str, detail: str = "", **fields: Any) -> None:
        with self._lock:
            self._events.append({"code": str(code), "detail": detail, **fields})

    def drain_events(self) -> list[dict]:
        with self._lock:
            out, self._events = self._events, []
        return out + self.loader.drain_events()


def runtime(env: TaskEnv) -> WorkerRuntime:
    """The process's runtime for `env`, created on first use (one per process and env)."""
    with _RUNTIME_LOCK:
        rt = _RUNTIMES.get(env)
        if rt is None:
            rt = _RUNTIMES[env] = WorkerRuntime(env)
        return rt


# ---- local task bodies


def t_ping(rt: WorkerRuntime, payload: dict) -> dict:
    """{} -> {pid, threads (OMP/OPENBLAS/MKL vars), torch_threads, sigalrm handler name}."""
    handler = signal.getsignal(signal.SIGALRM)
    name = f"{handler.__module__}.{handler.__qualname__}" if callable(handler) else str(handler)
    return {
        "pid": os.getpid(),
        "threads": {v: os.environ.get(v) for v in THREAD_VARS},
        "torch_threads": _torch_one_thread(),
        "sigalrm": name,
    }


def t_diag_sleep(rt: WorkerRuntime, payload: dict) -> dict:
    """Test only. {"seconds" <= 600} -> {"slept"}."""
    seconds = float(payload.get("seconds", 0.0))
    if not 0.0 <= seconds <= MAX_DIAG_SLEEP_S:
        raise ServeError(Code.REQUEST_INVALID, f"seconds must lie in [0, {MAX_DIAG_SLEEP_S}]")
    time.sleep(seconds)
    return {"slept": seconds}


def t_diag_crash(rt: WorkerRuntime, payload: dict) -> dict:
    """Test only: the worker process dies (os._exit(3))."""
    os._exit(3)


def t_featurize(rt: WorkerRuntime, payload: dict) -> dict:
    """{"key"} -> FeaturizeResult as dict; a failed featurization fails the task with its
    code (the result travels in the error detail)."""
    res = rt.featurizer.ensure(str(payload["key"]))
    out = res.to_dict()
    if res.status == FeaturizeStatus.FAILED:
        raise ServeError(
            res.code or Code.FEATURIZATION_FAILED, res.detail, upstream=res.upstream, detail=out
        )
    return out


TASKS: dict[str, str | Callable[[Any, dict], dict]] = {
    "ping": t_ping,
    "diag_sleep": t_diag_sleep,
    "diag_crash": t_diag_crash,
    "featurize": t_featurize,
    "analyze": "ksmech.serve.science.analysis:t_analyze",
    "expand": "ksmech.serve.predict.expansion:t_expand",
    "orbital_cube": "ksmech.serve.chem.cubes:t_orbital_cube",
    "index_chunk": "ksmech.serve.browse.pmindex:t_index_chunk",
    "recheck_step": "ksmech.serve.curate.recheck:t_recheck_step",
    "wb_orbital_set": "ksmech.serve.workbench.orbital_sets:t_wb_orbital_set",
    "wb_cube": "ksmech.serve.workbench.densities:t_wb_cube",
    "wb_properties": "ksmech.serve.workbench.properties:t_wb_properties",
    "embedding": "ksmech.serve.embedding.service:t_embedding",
}


def resolve(name: str) -> Callable[[Any, dict], dict]:
    """The body of task `name`. Raises ServeError(UNEXPECTED) for an unknown name or a lazy
    reference whose module or function does not exist (yet)."""
    if name in _BODIES:
        return _BODIES[name]
    entry = TASKS.get(name)
    if entry is None:
        raise ServeError(Code.UNEXPECTED, f"unknown task {name!r}")
    if callable(entry):
        _BODIES[name] = entry
        return entry
    module, _, func = entry.partition(":")
    try:
        found = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        raise ServeError(Code.UNEXPECTED, f"task body not installed: {entry}")
    body = getattr(importlib.import_module(module), func, None)  # import errors propagate
    if not callable(body):
        raise ServeError(Code.UNEXPECTED, f"task body not installed: {entry}")
    _BODIES[name] = body
    return body


def _pid_file(env: TaskEnv, seq: int) -> Path | None:
    s = env.settings
    path = s.run / f"task-{seq}.pid"
    try:
        assert_writable(path, s)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(f"{os.getpid()}\n")
        os.replace(tmp, path)
        return path
    except (OSError, ValueError) as exc:
        log.error("cannot write the pid file %s: %s", path, exc)
        return None


def _outcome(ok, result, code, upstream, detail, events, t0) -> dict:
    return {
        "ok": ok,
        "result": result,
        "code": None if code is None else str(code),
        "upstream": upstream,
        "detail": detail,
        "events": events,
        "pid": os.getpid(),
        "seconds": time.perf_counter() - t0,
    }


def run_task(name: str, env: TaskEnv, payload: dict, seq: int) -> dict:
    """Run one task; never raises. Returns {ok, result, code, upstream, detail, events, pid,
    seconds}."""
    t0 = time.perf_counter()
    pid_path = _pid_file(env, seq)
    rt: WorkerRuntime | None = None
    try:
        if name in DIAG_TASKS and not env.enable_diag_tasks:
            raise ServeError(Code.UNEXPECTED, f"diagnostic task {name!r} is disabled")
        body = resolve(name)
        rt = runtime(env)
        result = body(rt, payload)
        return _outcome(True, result, None, None, "", rt.drain_events(), t0)
    except ServeError as exc:
        error = {"message": exc.message, "status": exc.status, "detail": jsonable(exc.detail)}
        events = rt.drain_events() if rt is not None else []
        return _outcome(False, {"error": error}, exc.code, exc.upstream, exc.message, events, t0)
    except Exception as exc:  # noqa: BLE001 - a task never raises
        log.exception("task %s (seq %d) failed", name, seq)
        events = rt.drain_events() if rt is not None else []
        detail = f"{type(exc).__name__}: {exc}"
        return _outcome(False, None, Code.UNEXPECTED, None, detail, events, t0)
    finally:
        if pid_path is not None:
            with contextlib.suppress(FileNotFoundError):
                pid_path.unlink()
