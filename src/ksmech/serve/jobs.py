"""Background jobs (design 5.3, 5.5): the job store under `data/serve/jobs/<job_id>/`, the
runner threads, the job context handlers see, and the predict and override handlers.

Lifecycle: queued -> running -> done | failed | cancelled. A queued job is cancelled at once;
a running one is cancelled softly (checked between worker tasks) or, with `force`, also by
killing its running worker task (`scheduler.kill_owner`). At start-up every job still queued
or running becomes failed(SERVER_RESTART); its partial network is kept and viewable.

A job directory holds `job.json` (the `Job`), `network.json` (predict jobs: the network,
written atomically after every node expansion, throttled to 1 s, and at the end),
`result.json` (the result of the other kinds) and `events.jsonl` (reason-coded events).
Every write is atomic (tmp file in the same directory, fsync, os.replace).

Exactly one "batch" thread runs predict, override, recheck and export jobs in FIFO order (it
is the only writer of networks; each network also has its own lock), one "background"
thread runs index jobs and one "trace" thread runs embedding jobs (they read a pool and write
only under `data/serve/embedding/`, so they never wait behind a long prediction on the API
side; their worker task still queues behind the running task). Inline mode (tests) runs a job
in the submitting thread.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import queue
import re
import secrets
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ksmech.serve.contracts import USER_PREDICTOR, JobContext, PredictorRef
from ksmech.serve.errors import Code, ServeError, jsonable
from ksmech.serve.pools import SpeciesInput
from ksmech.serve.settings import ServeSettings, append_failure, assert_writable
from ksmech.serve.worker import Scheduler

if TYPE_CHECKING:
    from ksmech.serve.chemistry import Chemistry
    from ksmech.serve.network import Network

log = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"^j[0-9a-f]{16}$")
PROGRESS_INTERVAL_S = 1.0
NETWORK_INTERVAL_S = 1.0
RETENTION = 200
THREADS = ("batch", "background", "trace")
JOIN_TIMEOUT_S = 5.0

Handler = Callable[[JobContext, dict], dict | None]


class JobKind(StrEnum):
    PREDICT = "predict"
    OVERRIDE = "override"  # spec-v3 jobs only (a user step is synchronous since spec v9)
    EXPAND = "expand"
    RECHECK = "recheck"
    EXPORT = "export"
    INDEX = "index"
    EMBEDDING = "embedding"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE = (JobStatus.QUEUED, JobStatus.RUNNING)


class Cancelled(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised by a handler that stops because its job was cancelled."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_job_id() -> str:
    return "j" + secrets.token_hex(8)


@dataclass
class Job:
    id: str
    kind: str
    status: str
    created_at: str  # ISO 8601, aware UTC
    request: dict
    progress: dict
    started_at: str | None = None
    finished_at: str | None = None
    error: dict | None = None  # ErrorBody fields
    parent_job_id: str | None = None
    network_id: str | None = None
    result_path: str | None = None
    label: str = ""
    skipped: dict[str, int] = field(default_factory=dict)
    provenance: dict | None = None  # Provenance fields at submission (prediction jobs)
    cancel_requested: bool = False

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def created_out(self) -> dict:
        """`JobCreatedOut` fields. A prediction or override job carries its provenance at
        submission (predictor, its version or checkpoint hash, the spec version); the key is
        left out for the other jobs, whose `JobCreatedOut.provenance` defaults to None."""
        out = {
            "job_id": self.id,
            "kind": str(self.kind),
            "status": str(self.status),
            "status_url": f"/api/jobs/{self.id}",
            "parent_job_id": self.parent_job_id,
        }
        if self.provenance is not None:
            out["provenance"] = self.provenance
        return out

    def to_dict(self) -> dict:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping) -> Job:
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        kw["skipped"] = {str(k): int(v) for k, v in (kw.get("skipped") or {}).items()}
        return cls(**kw)


def _atomic_write(path: Path, text: str, settings: ServeSettings | None) -> None:
    if settings is not None:
        assert_writable(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


class JobStore:
    """The files of every job under `jobs_dir` (normally `data/serve/jobs`)."""

    def __init__(self, jobs_dir: Path, settings: ServeSettings | None = None) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.settings = settings
        self._lock = threading.Lock()

    def dir(self, job_id: str) -> Path:
        """The job's directory. A malformed id is JOB_NOT_FOUND (404), never a file access."""
        if not JOB_ID_RE.match(str(job_id)):
            raise ServeError(Code.JOB_NOT_FOUND, f"malformed job id {job_id!r}", status=404)
        return self.jobs_dir / job_id

    def network_path(self, job_id: str) -> Path:
        return self.dir(job_id) / "network.json"

    def write_json(self, job_id: str, name: str, data: Any) -> Path:
        path = self.dir(job_id) / name
        text = json.dumps(jsonable(data), sort_keys=True, indent=1) + "\n"
        with self._lock:
            _atomic_write(path, text, self.settings)
        return path

    def read_json(self, job_id: str, name: str) -> dict | None:
        try:
            return json.loads((self.dir(job_id) / name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def save(self, job: Job) -> Path:
        return self.write_json(job.id, "job.json", job.to_dict())

    def load(self, job_id: str) -> Job:
        try:
            d = self.read_json(job_id, "job.json")
        except (OSError, ValueError) as exc:
            msg = f"unreadable job {job_id}: {exc}"
            raise ServeError(Code.JOB_NOT_FOUND, msg, status=404) from exc
        if d is None:
            raise ServeError(Code.JOB_NOT_FOUND, f"no job {job_id}", status=404)
        return Job.from_dict(d)

    def load_all(self) -> list[Job]:
        """Every readable job; an unreadable one is logged and counted, never silent."""
        jobs, bad = [], 0
        if not self.jobs_dir.is_dir():
            return jobs
        for entry in sorted(self.jobs_dir.iterdir()):
            if not JOB_ID_RE.match(entry.name) or not (entry / "job.json").is_file():
                continue
            try:
                jobs.append(self.load(entry.name))
            except (ServeError, KeyError, TypeError, ValueError) as exc:
                bad += 1
                log.warning("skipping unreadable job %s [JOB_UNREADABLE]: %s", entry.name, exc)
        if bad:
            log.warning("%d unreadable job directories skipped", bad)
        return jobs

    def write_network(self, network_id: str, net: dict) -> Path:
        path = self.network_path(network_id)
        text = json.dumps(net, sort_keys=True) + "\n"
        with self._lock:
            _atomic_write(path, text, self.settings)
        return path

    def read_network(self, network_id: str) -> dict | None:
        try:
            return json.loads(self.network_path(network_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def append_event(self, job_id: str, record: dict) -> None:
        path = self.dir(job_id) / "events.jsonl"
        if self.settings is not None:
            assert_writable(path, self.settings)
        line = json.dumps(jsonable(record), sort_keys=True)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def events(self, job_id: str) -> list[dict]:
        path = self.dir(job_id) / "events.jsonl"
        if not path.is_file():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.dir(job_id), ignore_errors=True)


def curated_job_ids(settings: ServeSettings) -> set[str] | None:
    """Job ids referenced by curated steps (StepDoc `job_id` in `curatedstep.arrows_json`),
    read-only. None when the database exists but cannot be read (then nothing is deleted)."""
    path = settings.curation_db
    if not path.is_file():
        return set()
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = con.execute("SELECT arrows_json FROM curatedstep").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        log.warning("cannot read curated job ids from %s: %s", path, exc)
        return None
    ids = set()
    for (text,) in rows:
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            job_id = json.loads(text).get("job_id")
            if job_id:
                ids.add(str(job_id))
    return ids


class RunnerJobContext:
    """`contracts.JobContext` of a running job: throttled progress into `job.json`, the
    job's cancel event, and reason-coded skips (job `skipped`, `events.jsonl`,
    `data/serve/logs/failures.jsonl` and a WARNING)."""

    def __init__(self, runner: JobRunner, job_id: str) -> None:
        self.runner = runner
        self.job_id = job_id
        self.settings = runner.settings
        self.scheduler = runner.scheduler
        self._event = runner._cancel_event(job_id)
        self._last_write = float("-inf")
        self._last_network = float("-inf")

    def progress(self, **fields: Any) -> None:
        now = time.monotonic()
        with self.runner._lock:
            job = self.runner._jobs[self.job_id]
            phase = fields.get("phase")
            changed = phase is not None and phase != job.progress.get("phase")
            job.progress = {**job.progress, **jsonable(fields), "updated_at": now_iso()}
            if changed or now - self._last_write >= PROGRESS_INTERVAL_S:
                self.runner.store.save(job)
                self._last_write = now

    def cancelled(self) -> bool:
        return self._event.is_set()

    def skip(self, code: str, detail: str = "", **where: Any) -> None:
        n = max(1, int(where.pop("count", 1) or 1))
        with self.runner._lock:
            job = self.runner._jobs[self.job_id]
            job.skipped[str(code)] = job.skipped.get(str(code), 0) + n
        record = {"time": now_iso(), "code": str(code), "detail": detail, "count": n, **where}
        try:
            self.runner.store.append_event(self.job_id, record)
        except (OSError, ValueError) as exc:
            log.error("cannot append to the events of job %s: %s", self.job_id, exc)
        append_failure(self.settings, str(code), detail, job_id=self.job_id, count=n, **where)
        log.warning("job %s skipped [%s] x%d %s %s", self.job_id, code, n, detail, where or "")

    def snapshot_network(
        self, net: dict | Callable[[], dict], *, network_id: str | None = None, force=False
    ) -> bool:
        """Write the network (a dict, or a thunk building it) at most once per second unless
        `force`. Returns True when it wrote."""
        now = time.monotonic()
        if not force and now - self._last_network < NETWORK_INTERVAL_S:
            return False
        data = net() if callable(net) else net
        target = network_id or self.runner.get(self.job_id).network_id or self.job_id
        self.runner.store.write_network(target, data)
        self._last_network = now
        return True


class JobRunner:
    """The two runner threads and the in-memory view of every job (`job.json` persists it)."""

    def __init__(
        self,
        store: JobStore,
        settings: ServeSettings,
        scheduler: Scheduler,
        *,
        inline: bool = False,
        retention: int = RETENTION,
        protected: Callable[[], set[str] | None] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.scheduler = scheduler
        self.inline = bool(inline)
        self._handlers: dict[str, tuple[Handler, str]] = {}
        self._jobs: dict[str, Job] = {}
        self._events: dict[str, threading.Event] = {}
        self._net_locks: dict[str, threading.Lock] = {}
        self._lock = threading.RLock()
        self._queues: dict[str, queue.Queue] = {t: queue.Queue() for t in THREADS}
        self._threads: dict[str, threading.Thread] = {}
        self._recover()
        self._retain(retention, protected or (lambda: curated_job_ids(settings)))
        if not self.inline:
            for name in THREADS:
                t = threading.Thread(target=self._loop, args=(name,), name=f"jobs-{name}")
                t.daemon = True
                t.start()
                self._threads[name] = t

    # ---- start-up

    def _recover(self) -> None:
        """Every job left queued or running by a previous process: failed(SERVER_RESTART)."""
        for job in self.store.load_all():
            if job.active:
                was = job.status
                job.status, job.finished_at = JobStatus.FAILED.value, now_iso()
                job.error = {
                    "code": str(Code.SERVER_RESTART),
                    "message": f"the server restarted while the job was {was}",
                    "upstream": None,
                    "detail": None,
                    "request_id": "",
                }
                self.store.save(job)
                log.warning("job %s was %s at start-up: failed(SERVER_RESTART)", job.id, was)
            self._jobs[job.id] = job

    def _retain(self, keep_n: int, protected: Callable[[], set[str] | None]) -> None:
        """Keep the newest `keep_n` jobs plus every job a curated mechanism references (and
        the networks they belong to); delete the rest."""
        order = sorted(self._jobs.values(), key=lambda j: (j.created_at, j.id), reverse=True)
        if len(order) <= keep_n:
            return
        keep_ids = protected()
        if keep_ids is None:
            log.warning("job retention skipped: the curated job ids are unknown")
            return
        keep = {j.id for j in order[:keep_n]} | set(keep_ids)
        keep |= {self._jobs[i].network_id for i in list(keep) if i in self._jobs} - {None}
        dropped = [j.id for j in order if j.id not in keep]
        for job_id in dropped:
            self.store.delete(job_id)
            del self._jobs[job_id]
        log.info("job retention: deleted %d old jobs, kept %d", len(dropped), len(self._jobs))

    # ---- public API

    def register(self, kind: str, handler: Handler, *, thread: str = "batch") -> None:
        if thread not in THREADS:
            raise ValueError(f"thread must be one of {THREADS}, got {thread!r}")
        self._handlers[str(kind)] = (handler, thread)

    def submit(
        self,
        kind: str,
        request: dict,
        *,
        parent_job_id: str | None = None,
        network_id: str | None = None,
        label: str = "",
        exclusive: bool = False,
        provenance: dict | None = None,
    ) -> Job:
        """Queue a job (inline mode: run it now). With `exclusive`, a queued or running job
        on the same network is 409 NETWORK_BUSY. An unknown kind is 422."""
        if str(kind) not in self._handlers:
            msg = f"no handler for job kind {kind!r}; one of {sorted(self._handlers)}"
            raise ServeError(Code.REQUEST_INVALID, msg, status=422)
        with self._lock:
            if exclusive and network_id is not None:
                self._check_idle(network_id)
            job_id = new_job_id()
            if network_id is None and str(kind) == JobKind.PREDICT:
                network_id = job_id
            job = Job(
                id=job_id,
                kind=str(kind),
                status=JobStatus.QUEUED.value,
                created_at=now_iso(),
                request=copy.deepcopy(jsonable(request)),
                progress={"phase": "queued", "updated_at": now_iso()},
                parent_job_id=parent_job_id,
                network_id=network_id,
                label=label,
                provenance=provenance,
            )
            self._jobs[job_id] = job
            self._events[job_id] = threading.Event()
            self.store.save(job)
        if self.inline:
            self._run_one(job_id)
        else:
            self._queues[self._handlers[str(kind)][1]].put(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return copy.deepcopy(job)
        self.store.dir(job_id)  # a malformed id is 404 before any file access
        raise ServeError(Code.JOB_NOT_FOUND, f"no job {job_id}", status=404)

    def list(self, *, kind: str | None = None, status: str | None = None, limit: int = 50):
        with self._lock:
            jobs = [copy.deepcopy(j) for j in self._jobs.values()]
        jobs = [j for j in jobs if (kind is None or j.kind == kind)]
        jobs = [j for j in jobs if (status is None or j.status == status)]
        jobs.sort(key=lambda j: (j.created_at, j.id), reverse=True)
        return jobs[: max(0, int(limit))]

    def cancel(self, job_id: str, force: bool = False) -> Job:
        """A queued job ends cancelled at once; a running one is asked to stop (and, with
        `force`, its running worker task is killed). A finished job is left as it is."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ServeError(Code.JOB_NOT_FOUND, f"no job {job_id}", status=404)
            running = job.status == JobStatus.RUNNING
            if job.status == JobStatus.QUEUED:
                job.status, job.finished_at = JobStatus.CANCELLED.value, now_iso()
                job.progress = {**job.progress, "phase": "cancelled", "updated_at": now_iso()}
            elif running:
                job.cancel_requested = True
                self._cancel_event(job_id).set()
                msg = "hard cancel requested" if force else "cancel requested"
                job.progress = {**job.progress, "message": msg, "updated_at": now_iso()}
            self.store.save(job)
        if running and force:
            self.scheduler.kill_owner(job_id)
        return self.get(job_id)

    def network_lock(self, network_id: str) -> threading.Lock:
        with self._lock:
            return self._net_locks.setdefault(network_id, threading.Lock())

    def busy_job(self, network_id: str) -> str | None:
        """The id of a queued or running job on `network_id`, if any."""
        with self._lock:
            for job in self._jobs.values():
                if job.network_id == network_id and job.active:
                    return job.id
        return None

    def running(self) -> int:
        with self._lock:
            return sum(j.status == JobStatus.RUNNING for j in self._jobs.values())

    def close(self) -> None:
        """Ask running jobs to stop, stop the threads (bounded wait)."""
        with self._lock:
            for job in self._jobs.values():
                if job.status == JobStatus.RUNNING:
                    self._cancel_event(job.id).set()
        for name, t in self._threads.items():
            self._queues[name].put(None)
            t.join(timeout=JOIN_TIMEOUT_S)

    # ---- execution

    def _check_idle(self, network_id: str) -> None:
        busy = self.busy_job(network_id)
        if busy is not None:
            msg = f"job {busy} is queued or running on network {network_id}"
            detail = {"network_id": network_id, "job_id": busy}
            raise ServeError(Code.NETWORK_BUSY, msg, status=409, detail=detail)

    def _cancel_event(self, job_id: str) -> threading.Event:
        with self._lock:
            return self._events.setdefault(job_id, threading.Event())

    def _loop(self, name: str) -> None:
        q = self._queues[name]
        while True:
            job_id = q.get()
            if job_id is None:
                return
            try:
                self._run_one(job_id)
            except Exception:  # noqa: BLE001 - the runner thread never dies
                log.exception("runner %s: job %s escaped its wrapper", name, job_id)

    def _start(self, job_id: str) -> Handler | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                return None  # cancelled while queued
            job.status, job.started_at = JobStatus.RUNNING.value, now_iso()
            job.progress = {**job.progress, "phase": "starting", "updated_at": now_iso()}
            self.store.save(job)
            return self._handlers[job.kind][0]

    def _run_one(self, job_id: str) -> None:
        """Run a job with the wrapping of design 5.3: Cancelled (or a CANCELLED ServeError)
        gives cancelled, a ServeError failed with its code, anything else failed(UNEXPECTED)."""
        handler = self._start(job_id)
        if handler is None:
            return
        ctx = RunnerJobContext(self, job_id)
        try:
            result = handler(ctx, copy.deepcopy(self._jobs[job_id].request))
            self._store_result(job_id, result)
            self._end(job_id, JobStatus.CANCELLED if ctx.cancelled() else JobStatus.DONE)
        except Cancelled:
            self._end(job_id, JobStatus.CANCELLED)
        except ServeError as exc:
            if exc.code == Code.CANCELLED:
                self._end(job_id, JobStatus.CANCELLED)
            else:
                log.warning("job %s failed [%s] %s", job_id, exc.code, exc.message)
                self._end(job_id, JobStatus.FAILED, exc.body())
        except Exception as exc:  # noqa: BLE001 - reported as UNEXPECTED, never raised
            log.exception("job %s failed unexpectedly", job_id)
            err = ServeError(Code.UNEXPECTED, f"{type(exc).__name__}: {exc}", status=500)
            self._end(job_id, JobStatus.FAILED, err.body())

    def _store_result(self, job_id: str, result: dict | None) -> None:
        job = self._jobs[job_id]
        if job.kind == JobKind.PREDICT:
            path = self.store.network_path(job.network_id or job_id)
        elif result is not None:
            path = self.store.write_json(job_id, "result.json", result)
        else:
            return
        with self._lock:
            job.result_path = str(path)

    def _end(self, job_id: str, status: JobStatus, error: dict | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.finished_at, job.error = status.value, now_iso(), error
            job.progress = {**job.progress, "phase": status.value, "updated_at": now_iso()}
            self.store.save(job)


# ---- the predict and override handlers


def species_inputs(request: Mapping) -> list[SpeciesInput]:
    """The `PredictIn` / `NormalizeIn` species with their counts (default 1 each)."""
    smiles = list(request.get("reactants_smiles") or [])
    counts = request.get("stoichiometry")
    if counts is None:
        counts = [1] * len(smiles)
    if len(counts) != len(smiles):
        msg = f"{len(smiles)} reactants but {len(counts)} stoichiometric counts"
        raise ServeError(Code.STOICHIOMETRY_MISMATCH, msg, status=422)
    return [SpeciesInput(str(s), int(n)) for s, n in zip(smiles, counts, strict=True)]


def net_params(request: Mapping):
    """`NetParams` of a `PredictIn` dict (its defaults where a field is absent)."""
    from ksmech.serve.network import NetParams

    d = NetParams()
    t = request.get("temperature")
    return NetParams(
        beam=int(request.get("beam", d.beam)),
        max_steps=int(request.get("max_steps", d.max_steps)),
        temperature=None if t is None else float(t),
        stop_at_turnover=bool(request.get("stop_at_turnover", d.stop_at_turnover)),
        candidate_scope=str(request.get("candidate_scope", d.candidate_scope)),
        max_nodes=int(request.get("max_nodes", d.max_nodes)),
        n_pathways=int(request.get("n_pathways", d.n_pathways)),
    )


class _NetworkWriter:
    """The `NetHooks.snapshot` of a job: throttled atomic writes of `network.json`."""

    def __init__(self, ctx: JobContext, network_id: str) -> None:
        self.ctx = ctx
        self.network_id = network_id
        self.last: Network | None = None

    def __call__(self, net: Network) -> None:
        self.last = net
        self.write(net)

    def write(self, net: Network, force: bool = False) -> None:
        from ksmech.serve.network import network_to_dict

        snapshot = getattr(self.ctx, "snapshot_network", None)
        if snapshot is not None:
            snapshot(lambda: network_to_dict(net), network_id=self.network_id, force=force)

    def flush(self) -> None:
        if self.last is not None:
            self.write(self.last, force=True)


def predict_handler(
    chem: Chemistry, ref_of: Callable[[str, float | None], PredictorRef]
) -> Handler:
    """The `predict` job: root, `build_network`, `network.json`. `ref_of(name, T)` gives the
    predictor reference (`PredictorRegistry.ref`). Predictor "user" builds the root alone (a
    network the user extends with steps and expansions)."""

    def handle(ctx: JobContext, request: dict) -> dict:
        from ksmech.serve.network import NetHooks, Network, build_network

        ctx.progress(phase="normalize", message="building the root pool")
        root = chem.root(species_inputs(request))
        ref = ref_of(str(request.get("predictor", "floor")), request.get("temperature"))
        writer = _NetworkWriter(ctx, ctx.job_id)
        params = net_params(request)
        if ref.name == USER_PREDICTOR:
            net = Network.new(root, ref, params, ctx.job_id, getattr(chem, "mode", ""))
            net.complete = True
            net.touch()
        else:
            try:
                net = build_network(chem, root, ref, params, NetHooks(ctx, writer))
            except BaseException:
                writer.flush()  # keep the partial network viewable
                raise
        writer.write(net, force=True)
        return {"network_id": ctx.job_id, "revision": net.revision, "complete": net.complete}

    return handle


def expand_handler(
    chem: Chemistry, runner: JobRunner, ref_of: Callable[[str, float | None], PredictorRef]
) -> Handler:
    """The `expand` job: load the network, `continue_from` one node under the network's
    lock, write it back. Request: {network_id, parent_job_id, expand: ExpandIn dict}."""

    def handle(ctx: JobContext, request: dict) -> dict:
        from dataclasses import replace

        from ksmech.serve.network import NetHooks, continue_from, network_from_dict, rank_pathways

        net_id, body = str(request["network_id"]), dict(request["expand"])
        with runner.network_lock(net_id):
            data = runner.store.read_network(net_id)
            if data is None:
                raise ServeError(Code.JOB_NOT_DONE, f"network {net_id} has no data", status=409)
            net = network_from_dict(data)
            node_id = str(body["node_id"])
            if node_id not in net.nodes:
                raise ServeError(Code.NODE_NOT_FOUND, f"no node {node_id} in {net_id}", status=404)
            default = "floor" if net.predictor.name == USER_PREDICTOR else net.predictor.name
            ref = ref_of(str(body.get("predictor") or default), net.predictor.temperature)
            params = replace(net.params, beam=int(body.get("beam") or net.params.beam))
            writer = _NetworkWriter(ctx, net_id)
            try:
                steps = int(body.get("steps") or 1)
                continue_from(chem, net, node_id, steps, params, NetHooks(ctx, writer), ref)
            except BaseException:
                writer.flush()
                raise
            net.pathways = rank_pathways(net, net.params.n_pathways)
            net.touch()
            writer.write(net, force=True)
        return {
            "network_id": net_id,
            "network_revision": net.revision,
            "node_id": node_id,
            "predictor": ref.name,
            "n_nodes": len(net.nodes),
            "n_edges": len(net.edges),
        }

    return handle
