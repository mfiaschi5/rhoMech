"""`ksmech-qm run --optimizer gpu-anc`: tblite-anc's geometry optimization batched on the GPU
(`qm.gpu.optimize.relax_many`), the records written by the CPU workers as `qm.batch` writes them.

- **Routing**, per task in a worker (`prepare`). A failure the memo holds is answered from it.
  A species the GPU engine does not take goes to the CPU path (`batch.process_molecule`,
  tblite-anc), counted by route: more than one conformer, a given geometry, a spin tag or an
  unpaired electron, an element above Rn, fewer than `min_atoms` atoms (below about 30 atoms
  the CPU is faster, docs/decisions.md 2026-11-23), or a preparation that fails (the CPU path
  then reports it). The rest are embedded there (`embed(mol, 1, seed=0)`, as
  `batch._compute`).
- **The GPU** optimizes them as one queue, smallest first, that grows while the workers prepare
  the rest (`relax_many`'s `more`, `FEED` items at a time): a long queue is what makes it fast
  (1024 in flight against 256: 1.79x against 1.03x the 12 CPU workers at 31-50 atoms), and the
  GPU starts before every task is prepared. It runs in a thread of this process; the workers
  never import torch. The workers' priorities: finishing, preparing, the CPU path.
- **Finishing**, per outcome as soon as it is final, in a worker (`finish`): tblite's single
  point at the GPU's geometry; where its gradient (walls included) is above fmax, a tblite-anc
  polish from there (record warning GPU_POLISHED); Fukui, sterics, the record. Every record
  carries the run's one method, `GFN2-xTB/ALPB-water/gpu-anc`, the CPU-optimized ones too: the
  token build reads one method. The manifest's detail of a record says where it was optimized.
- **Failures.** A GPU failure, or a failure while finishing, is retried on the CPU path from
  the start (`batch._run_molecule`), counted; only that outcome reaches the failure memo, so
  gpu-anc's tombstones hold tblite-anc's failures, never the GPU's.
- **Budgets.** `maxiter`, `fmax` and `scf_maxiter` apply on the GPU too; `timeout_s` applies to
  every CPU part (preparation, finishing, CPU path), not to the GPU's batched optimization.

Water only (the GPU engine's ALPB); a run with another medium is refused.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from multiprocessing import get_context

import numpy as np
from rdkit import Chem

from ksmech.processes import single_threaded_children
from ksmech.qm import batch, blas, failures
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import BOHR_ANG
from ksmech.qm.records import QMError, ReasonCode, cache_path, save_record
from ksmech.qm.spin import method_for, spin_for_qm
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

log = logging.getLogger(__name__)

OPTIMIZER = "gpu-anc"
GPU_MIN_ATOMS = 31  # atoms with H; below, the CPU workers are faster
MAX_Z = 86  # the GPU engine's GFN2 parameters
FEED = 256  # prepared GPU items sent to its queue at a time (all that are left at the end)
_POLL_S = 0.2
_DONE = object()
RETRY_NOTE = "GPU retried on the CPU"


@dataclass(frozen=True)
class GpuItem:
    """A task routed to the GPU and embedded: the mol as an RDKit binary (its conformer and
    properties, the embed notes, kept), the start in Bohr."""

    task: batch.MolTask
    mol: bytes
    conf_id: int
    z: np.ndarray
    start: np.ndarray
    charge: int


@dataclass(frozen=True)
class Finish:
    """One GPU outcome to finish on a worker: the geometry (None after a GPU failure)."""

    item: GpuItem
    xyz: np.ndarray | None
    restrained: bool
    failure: str
    n_evals: int


def check_config(config: QMConfig) -> None:
    """ValueError for a configuration the GPU path cannot run."""
    if config.optimizer != OPTIMIZER:
        raise ValueError(f"the GPU path runs optimizer {OPTIMIZER!r}, not {config.optimizer!r}")
    if config.solvent != "water" or config.given_geometry:
        raise ValueError("gpu-anc runs ALPB water optimizations only")


def check_device() -> None:
    """ValueError when this process sees no CUDA device (torch imported here, not in workers)."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("gpu-anc needs a CUDA device; none is visible")


# --- workers -------------------------------------------------------------------------------


def _species_route(mol, smiles: str, min_atoms: int) -> tuple[str, int, int]:
    """(the CPU route or "", charge, unpaired) of a prepared species."""
    charge, unpaired, tag = spin_for_qm(mol, smiles)
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if tag is not None:
        return "spin_tag", charge, unpaired
    if unpaired:
        return "open_shell", charge, unpaired
    if max(z) > MAX_Z:
        return "element", charge, unpaired
    if len(z) < min_atoms:
        return "small", charge, unpaired
    return "", charge, unpaired


def prepare(job: tuple[batch.MolTask, int]):
    """Worker: a task's route. ("rows", rows) for a memo hit, ("cpu", route) for the CPU path,
    ("gpu", GpuItem) for an embedded GPU task."""
    task, min_atoms = job
    known = batch._known_failure(task)
    if known is not None:
        return "rows", [known]
    if task.n_conformers != 1:
        return "cpu", "conformers"
    signal.alarm(task.config.timeout_s)
    try:
        mol = prepare_mol(task.smiles)
        route, charge, _ = _species_route(mol, task.smiles, min_atoms)
        if route:
            return "cpu", route
        cid = embed(mol, 1, seed=0)[0]
    except (Exception, batch._Timeout) as exc:  # noqa: BLE001 - the CPU path reports it
        log.info("GPU preparation of %s failed (%s): CPU path", task.smiles, exc)
        return "cpu", "prepare_failed"
    finally:
        signal.alarm(0)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    start = mol.GetConformer(cid).GetPositions() / BOHR_ANG
    blob = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    return "gpu", GpuItem(task, blob, cid, z, start, charge)


def _record_rows(job: Finish, t0: float) -> list[dict] | str:
    """The record of a GPU geometry, saved: its rows, or the failure (str) to retry."""
    item, task = job.item, job.item.task
    mol = Chem.Mol(item.mol)
    try:
        rec = GFN2Calculator(task.config).run_from(mol, item.conf_id, job.xyz, job.restrained)
    except batch._Timeout:
        detail = f"> {task.config.timeout_s}s finishing the GPU geometry"
        seconds = time.perf_counter() - t0
        row = batch._row(task.smiles, item.conf_id, reason=str(ReasonCode.TIMEOUT), detail=detail)
        return [{**row, "seconds": seconds}]
    except QMError as exc:
        return f"finishing: {exc.reason}: {exc.detail}"
    except Exception as exc:  # noqa: BLE001 - retried on the CPU path
        log.exception("finishing %s failed", task.smiles)
        return f"finishing: {type(exc).__name__}: {exc}"
    if rec.smiles != task.smiles:
        return f"finishing: {ReasonCode.RECORD_KEY_MISMATCH}: record {rec.smiles!r}"
    path = cache_path(task.config.cache_dir, task.smiles, item.conf_id, method_for(
        task.smiles, task.config.method))  # fmt: skip
    save_record(rec, path)
    detail = f"optimized on the GPU, {job.n_evals} energies"
    row = dict(cache_path=str(path), status="ok", warnings=rec.warnings, n_embedded=1)
    row.update(detail=detail, seconds=time.perf_counter() - t0)
    return [batch._row(task.smiles, item.conf_id, **row)]


def finish(job: Finish) -> list[dict]:
    """Worker: the record of one GPU outcome, or the CPU path's rows when the GPU's run or the
    finishing failed (`RETRY_NOTE` in their detail). The memo learns only these rows."""
    task, failure = job.item.task, job.failure
    if job.xyz is not None:
        t0 = time.perf_counter()
        signal.alarm(task.config.timeout_s)
        try:
            got = _record_rows(job, t0)
        finally:
            signal.alarm(0)
        if not isinstance(got, str):
            failures.record_outcome(got, task.config)
            return got
        failure = got
    rows = batch._run_molecule(task)
    note = f"{RETRY_NOTE} ({failure})"
    for r in rows:
        r["detail"] = f"{r['detail']}; {note}" if r["detail"] else note
    failures.record_outcome(rows, task.config)
    return rows


# --- the driver ----------------------------------------------------------------------------


class _Pool:
    """A spawned worker pool that can be replaced after a crash."""

    def __init__(self, n: int) -> None:
        self.n, self.ex = n, self._new(n)

    @staticmethod
    def _new(n: int) -> ProcessPoolExecutor:
        ctx = get_context("spawn")
        return ProcessPoolExecutor(n, mp_context=ctx, initializer=batch._init_worker)

    def submit(self, fn, job):
        return self.ex.submit(fn, job)

    def restart(self) -> None:
        self.close()
        self.ex = self._new(self.n)

    def close(self) -> None:
        self.ex.shutdown(wait=False, cancel_futures=True)


class _Inbox:
    """The GPU thread's side of the items the workers prepare: `relax_many`'s `more`."""

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.closed = False

    def put(self, items: list[GpuItem] | None) -> None:
        self.q.put(items)

    def more(self, block: bool):
        """The RelaxTasks put since the last call ([] if none; waiting for some when `block`),
        or None once closed and empty."""
        from ksmech.qm.gpu.optimize import RelaxTask

        if self.closed:
            return None
        got: list[GpuItem] = []
        while True:
            try:
                items = self.q.get(timeout=_POLL_S) if block and not got else self.q.get_nowait()
            except queue.Empty:
                break
            if items is None:
                self.closed = True
                break
            got += items
        if self.closed and not got:
            return None
        return [RelaxTask(it.task.smiles, Chem.Mol(it.mol), it.z, it.start, it.charge, 0)
                for it in got]  # fmt: skip


def _gpu_thread(inbox: _Inbox, config: QMConfig, out: queue.Queue) -> None:
    """`relax_many` over the GPU items as they arrive in `inbox`, each outcome put on `out` as
    it is final; an exception is put there too, and `_DONE` last."""
    try:
        from ksmech.qm.gpu.optimize import relax_many

        relax_many([], config, on_result=out.put, more=inbox.more)
    except Exception as exc:  # noqa: BLE001 - its molecules are retried on the CPU path
        log.exception("the GPU optimization failed")
        out.put(exc)
    finally:
        out.put(_DONE)


class _Scheduler:
    """The pool's work, in this order of priority: finishing the GPU's outcomes, preparing the
    tasks (the GPU's queue grows meanwhile, `FEED` items at a time), the CPU tasks."""

    def __init__(self, pool: _Pool, on_rows, counts: Counter, min_atoms: int) -> None:
        self.pool, self.on_rows, self.counts, self.min_atoms = pool, on_rows, counts, min_atoms
        self.finish: deque = deque()
        self.prep: deque = deque()
        self.cpu: deque = deque()
        self.in_flight: dict = {}
        self.out: queue.Queue = queue.Queue()
        self.inbox = _Inbox()
        self.feed: list[GpuItem] = []  # prepared, not yet sent
        self.waiting: dict[str, GpuItem] = {}  # sent, no outcome yet
        self.gpu = "idle"  # "running", then "closed" (no more items), then "done"

    def run(self, tasks: list, config: QMConfig) -> None:
        self.config = config
        self.prep.extend(tasks)
        while self._busy():
            self._feed_gpu()
            if self.gpu in ("running", "closed"):
                idle = not (self.finish or self.prep or self.cpu or self.in_flight)
                self._take_outcomes(block=idle)
            self._fill()
            self._collect()

    def _busy(self) -> bool:
        queued = self.finish or self.prep or self.cpu or self.in_flight or self.feed
        return bool(queued) or self.gpu in ("running", "closed")

    def _preparing(self) -> bool:
        return bool(self.prep) or any(fn is prepare for fn, _ in self.in_flight.values())

    def _feed_gpu(self) -> None:
        """Send the prepared items to the GPU (starting it), and close its inbox once every
        task is prepared."""
        preparing = self._preparing()
        if self.feed and (len(self.feed) >= FEED or not preparing):
            if self.gpu == "done":  # the GPU run ended: retry them on the CPU path
                self.finish.extend(Finish(it, None, False, "GPU run ended early", 0)
                                   for it in self.feed)  # fmt: skip
                self.counts["gpu_unfinished"] += len(self.feed)
            else:
                self._start_gpu()
                self.waiting.update((it.task.smiles, it) for it in self.feed)
                self.inbox.put(self.feed)
            self.feed = []
        if not preparing and not self.feed and self.gpu == "running":
            self.inbox.put(None)
            self.gpu = "closed"

    def _start_gpu(self) -> None:
        if self.gpu == "idle":
            args = (self.inbox, self.config, self.out)
            threading.Thread(target=_gpu_thread, args=args, daemon=True).start()
            self.gpu = "running"

    def _take_outcomes(self, block: bool) -> None:
        """Move the GPU's final outcomes into finishing jobs."""
        while True:
            try:
                got = self.out.get(timeout=_POLL_S) if block else self.out.get_nowait()
            except queue.Empty:
                return
            block = False
            if got is _DONE:
                for item in self.waiting.values():  # an error ended the GPU run before them
                    self.finish.append(Finish(item, None, False, "GPU run ended early", 0))
                self.counts["gpu_unfinished"] += len(self.waiting)
                self.waiting, self.gpu = {}, "done"
                return
            if isinstance(got, Exception):
                self.counts["gpu_error"] += 1
                continue
            item = self.waiting.pop(got.key)
            failure = "" if got.reason is None else f"GPU: {got.reason}: {got.detail}"
            self.finish.append(Finish(item, got.xyz, got.restrained, failure, got.n_evals))

    def _next(self):
        if self.finish:
            return finish, self.finish.popleft()
        if self.prep:
            return prepare, (self.prep.popleft(), self.min_atoms)
        return batch.process_molecule, self.cpu.popleft()

    def _fill(self) -> None:
        while (self.finish or self.prep or self.cpu) and len(self.in_flight) < self.pool.n + 2:
            fn, job = self._next()
            self.in_flight[self.pool.submit(fn, job)] = (fn, job)

    def _collect(self) -> None:
        if not self.in_flight:
            return
        done, _ = wait(self.in_flight, timeout=_POLL_S, return_when=FIRST_COMPLETED)
        lost = []
        for fut in done:
            fn, job = self.in_flight.pop(fut)
            try:
                got = fut.result()
            except BrokenProcessPool:
                lost.append((fn, job))
                continue
            self._take(fn, job, got)
        if lost:  # every other job in flight is lost too
            self._recover(lost + list(self.in_flight.values()))

    def _take(self, fn, job, got) -> None:
        if fn is not prepare:
            self._count(fn, got)
            self.on_rows(got)
            return
        kind, value = got
        if kind == "rows":
            self.counts["memo"] += 1
            self.on_rows(value)
        elif kind == "cpu":
            self.counts[f"cpu_{value}"] += 1
            self.cpu.append(job[0])
        else:
            self.counts["gpu"] += 1
            self.feed.append(value)

    def _recover(self, lost: list) -> None:
        """After a pool crash: a new pool; a lost preparation goes to the CPU path, every other
        lost job runs alone (`batch._run_isolated`, a crash pinned on its molecule)."""
        self.in_flight = {}
        log.error("process pool broke; re-running %d jobs", len(lost))
        self.pool.restart()
        for fn, job in lost:
            if fn is prepare:
                self.counts["cpu_prepare_crash"] += 1
                self.cpu.append(job[0])
                continue
            task = job.item.task if isinstance(job, Finish) else job
            batch._run_isolated(job, self.on_rows, fn, lambda _j, t=task: batch._crash_row(t))

    def _count(self, fn, rows: list[dict]) -> None:
        if fn is not finish:
            return
        for r in rows:
            retried = RETRY_NOTE in str(r["detail"])
            ok = r["status"] == "ok"
            self.counts[f"gpu_{'retried_' if retried else ''}{'ok' if ok else 'failed'}"] += 1
            if ReasonCode.GPU_POLISHED in str(r["warnings"]).split(","):
                self.counts["gpu_polished"] += 1


def execute(tasks: list, n_workers: int, on_rows, min_atoms: int = GPU_MIN_ATOMS) -> Counter:
    """`batch.execute` for a gpu-anc run (module docstring): every task's rows reach `on_rows`
    once. Returns the counts by route (`gpu`, `cpu_<route>`, `memo`) and by GPU outcome
    (`gpu_ok`, `gpu_failed`, `gpu_retried_ok`, `gpu_retried_failed`, `gpu_polished`, ...)."""
    counts: Counter = Counter()
    if not tasks:
        return counts
    check_config(tasks[0].config)
    config = replace(tasks[0].config, spin_tag=None)  # the GPU's species are untagged
    with single_threaded_children(), blas.preloaded_children():
        pool = _Pool(max(n_workers, 1))
        try:
            _Scheduler(pool, on_rows, counts, min_atoms).run(tasks, config)
        finally:
            pool.close()
    return counts
