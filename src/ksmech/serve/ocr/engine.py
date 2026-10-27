"""The OCR engines: a small protocol, and MolScribe behind a persistent JSON-lines worker.

`OcrEngine` is what the service needs from an engine: a name, `recognize(png, timeout_s)` and
`close()`. Only MolScribe is implemented; OSRA or DECIMER would be another class with the same
three members (their own worker, or a command line run per image).

`JsonLinesWorker` holds one subprocess that answers one JSON line per request line
(`tools/ocr/ocr_worker.py` is the protocol's reference). It is spawned lazily by the first
request and waits for the worker's ready line (the model loads once, `ready_timeout_s`). One
request runs at a time, under a lock; a request that cannot take the lock within
`busy_timeout_s` is OCR_BUSY (503). A request that gets no answer within its timeout kills the
worker (OCR_TIMEOUT, 504); a worker that exits, or a pipe that breaks, is OCR_ENGINE_FAILED
(500). Either way the next request spawns a new worker (`restarts` counts them). The worker's
stdout carries only protocol lines; a line that is not JSON, or answers another id, is skipped.
Its stderr goes to `log_path`; on a failure its tail is logged by the server, never sent to the
client.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.ocr.install import ENGINE, OcrPaths

log = logging.getLogger(__name__)

THREADS_ENV = "KSMECH_OCR_THREADS"
DEFAULT_THREADS = 2
READY_TIMEOUT_S = 300.0  # loading the 1.1 GB checkpoint on a CPU
BUSY_TIMEOUT_S = 120.0
TAIL_BYTES = 2000
_EOF = object()
_DROPPED_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class RawOcr:
    """What an engine read in one image, before RDKit checks it."""

    smiles: str | None
    molfile: str | None
    confidence: float | None
    width: int | None = None  # the image the coordinates refer to
    height: int | None = None


class OcrEngine(Protocol):
    name: str

    def recognize(self, png: Path, timeout_s: float) -> RawOcr: ...

    def close(self) -> None: ...


def _tail(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, fh.seek(0, os.SEEK_END) - TAIL_BYTES))
            return fh.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _pump(stream: IO[str], lines: queue.Queue) -> None:
    """The reader thread: every stdout line into the queue, then the EOF sentinel."""
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass
    lines.put(_EOF)


class JsonLinesWorker:
    """One persistent subprocess speaking JSON lines (module docstring)."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        log_path: Path | None = None,
        ready_timeout_s: float = READY_TIMEOUT_S,
        busy_timeout_s: float = BUSY_TIMEOUT_S,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.argv = [str(a) for a in argv]
        self.env = dict(env) if env is not None else None
        self.log_path = log_path
        self.ready_timeout_s = float(ready_timeout_s)
        self.busy_timeout_s = float(busy_timeout_s)
        self.popen = popen
        self.restarts = 0
        self.ready: dict = {}
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._spawned = 0

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _fail(self, code: Code, message: str) -> ServeError:
        """The error of a failed worker; the tail of its stderr goes to the server log only
        (library tracebacks and paths are not for the client)."""
        tail = _tail(self.log_path)
        log.error("ocr worker: %s: %s%s", code, message, f"\n{tail}" if tail else "")
        return ServeError(code, message, detail={"restarts": self.restarts})

    def _spawn(self) -> None:
        if self._spawned:
            self.restarts += 1
            log.warning("ocr worker: restarting (%d restarts so far)", self.restarts)
        self._spawned += 1
        stderr: Any = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr = open(self.log_path, "ab")  # noqa: SIM115 - the child holds it
        try:
            self._proc = self.popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=self.env,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._proc = None
            raise self._fail(Code.OCR_ENGINE_FAILED, f"cannot start the OCR worker: {exc}") from exc
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()
        self._lines = queue.Queue()
        threading.Thread(target=_pump, args=(self._proc.stdout, self._lines), daemon=True).start()

    def _start(self) -> None:
        """Spawn the worker and wait for its ready line."""
        self._spawn()
        msg = self._read(time.monotonic() + self.ready_timeout_s, lambda m: "ready" in m)
        if msg is None:
            self.kill()
            what = f"no ready line within {self.ready_timeout_s:.0f} s"
            raise self._fail(Code.OCR_TIMEOUT, f"the OCR worker did not start: {what}")
        if not msg.get("ready"):
            self.kill()
            why = str(msg.get("error") or "unknown error")
            raise self._fail(Code.OCR_ENGINE_FAILED, f"the OCR worker could not start: {why}")
        self.ready = msg
        log.info("ocr worker: ready (pid %s): %s", self.pid, msg)

    def _read(self, deadline: float, wanted: Callable[[dict], bool]) -> dict | None:
        """The next JSON line `wanted` accepts; None at the deadline. EOF: OCR_ENGINE_FAILED."""
        while (left := deadline - time.monotonic()) > 0:
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                return None
            if line is _EOF:
                proc = self._proc
                self.kill(grace_s=2.0)  # reaps it: the exit code is known
                code = None if proc is None else proc.returncode
                raise self._fail(Code.OCR_ENGINE_FAILED, f"the OCR worker exited (code {code})")
            try:
                msg = json.loads(line)
            except ValueError:
                log.debug("ocr worker: not a protocol line: %.200s", line)
                continue
            if isinstance(msg, dict) and wanted(msg):
                return msg
        return None

    def _send(self, payload: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.kill()
            raise self._fail(Code.OCR_ENGINE_FAILED, f"the OCR worker is gone: {exc}") from exc

    def request(self, payload: Mapping[str, Any], timeout_s: float) -> dict:
        """The worker's answer to one request (its `id` is added here)."""
        if not self._lock.acquire(timeout=self.busy_timeout_s):
            msg = f"the OCR engine stayed busy for {self.busy_timeout_s:.0f} s: try again"
            raise ServeError(Code.OCR_BUSY, msg)
        try:
            if not self.alive():
                self._start()
            self._next_id += 1
            rid = self._next_id
            self._send({**payload, "id": rid})
            answer = self._read(time.monotonic() + timeout_s, lambda m: m.get("id") == rid)
            if answer is None:
                self.kill()
                msg = f"no answer within {timeout_s:.0f} s: the OCR worker was stopped"
                raise self._fail(Code.OCR_TIMEOUT, msg)
            return answer
        finally:
            self._lock.release()

    def kill(self, grace_s: float = 0.0) -> None:
        """Stop the worker (after `grace_s` to exit on its own) and reap it."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if grace_s > 0:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=grace_s)
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.error("ocr worker %s did not exit after SIGKILL", proc.pid)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    def close(self) -> None:
        """Ask the worker to quit, then make sure it did."""
        with self._lock:
            if self.alive():
                try:
                    self._send({"op": "quit"})
                    assert self._proc is not None
                    self._proc.wait(timeout=5)
                except (ServeError, subprocess.TimeoutExpired):
                    pass
            self.kill()


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def raw_of(answer: Mapping[str, Any]) -> RawOcr:
    """The `RawOcr` of one worker answer (`ok: true`); an `ok: false` answer is OCR_FAILED."""
    if not answer.get("ok"):
        why = str(answer.get("error") or "the engine gave no reason")
        raise ServeError(Code.OCR_FAILED, f"the OCR engine could not read the image: {why}")
    return RawOcr(
        smiles=_str_or_none(answer.get("smiles")),
        molfile=_str_or_none(answer.get("molfile")),
        confidence=_float_or_none(answer.get("confidence")),
        width=_int_or_none(answer.get("width")),
        height=_int_or_none(answer.get("height")),
    )


def worker_env(threads: int) -> dict[str, str]:
    """The worker's environment: this process's, without the variables that would point its
    python at the server's environment, with `threads` BLAS / OpenMP / torch threads."""
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    env |= dict.fromkeys(_THREAD_VARS, str(threads)) | {THREADS_ENV: str(threads)}
    return env | {"PYTHONUNBUFFERED": "1"}


def default_threads() -> int:
    raw = os.environ.get(THREADS_ENV, "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_THREADS
    except ValueError:
        log.warning("%s=%r is not an integer: using %d", THREADS_ENV, raw, DEFAULT_THREADS)
        return DEFAULT_THREADS


class MolScribeEngine:
    """MolScribe 1.1.1 in `data/ocr/venv`, through `tools/ocr/ocr_worker.py`."""

    name = ENGINE

    def __init__(self, paths: OcrPaths, threads: int | None = None, **worker: Any) -> None:
        self.paths = paths
        self.threads = default_threads() if threads is None else max(1, int(threads))
        argv = [paths.python, "-I", paths.worker, "--weights", paths.checkpoint]
        argv += ["--threads", str(self.threads)]
        worker.setdefault("env", worker_env(self.threads))
        worker.setdefault("log_path", paths.log)
        self.worker = JsonLinesWorker(argv, **worker)

    def recognize(self, png: Path, timeout_s: float) -> RawOcr:
        return raw_of(self.worker.request({"image": str(png)}, timeout_s))

    def close(self) -> None:
        self.worker.close()
