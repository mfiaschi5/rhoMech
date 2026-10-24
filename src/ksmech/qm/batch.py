"""Batch GFN2 featurization of every unique molecule in a dataset's `steps.parquet`.

Molecules are deduplicated on the map-free canonical SMILES. Each task embeds one molecule and
runs the calculator on its conformers. The disk cache is the source of truth for resuming; the
manifest parquet records the outcome of every (smiles, conformer) pair, failures included.
"""

from __future__ import annotations

import logging
import os
import random
import signal
import time
from collections import Counter
from collections.abc import Callable, Collection
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from ksmech.data import chem
from ksmech.data.io import DEFAULT_PROCESSED_DIR, source_dir, steps_path
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.records import (
    QMError,
    ReasonCode,
    cache_path,
    is_valid_cache_file,
    save_record,
)
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

log = logging.getLogger(__name__)

MANIFEST_NAME = "qm_manifest.parquet"
MANIFEST_COLUMNS = [
    "smiles",
    "conformer_id",
    "cache_path",
    "status",
    "reason",
    "detail",
    "warnings",
    "seconds",
    "n_embedded",
    "method",
]
_FLUSH_EVERY = 500
_WINDOW_PER_WORKER = 4
OnRows = Callable[[list[dict]], None]


@dataclass(frozen=True)
class MolTask:
    smiles: str
    n_conformers: int
    config: QMConfig


class _Timeout(BaseException):  # noqa: N818 - must pass through `except Exception` in the calculator
    pass


def _on_alarm(signum, frame) -> None:  # noqa: ARG001
    raise _Timeout


def _init_worker() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
    except ValueError:  # not the main thread: no per-molecule timeout
        log.warning("cannot install SIGALRM handler; timeouts disabled")


def _row(smiles: str, conf_id: int, **kw) -> dict:
    row = dict(
        smiles=smiles,
        conformer_id=conf_id,
        cache_path="",
        status="failed",
        reason="",
        detail="",
        warnings="",
        seconds=0.0,
        n_embedded=0,
        method="",
    )
    row.update(kw)
    return row


def _compute(task: MolTask) -> list[dict]:
    mol = prepare_mol(task.smiles)
    ids = embed(mol, task.n_conformers, seed=0)
    calc = GFN2Calculator(task.config)
    rows = []
    for cid in ids:
        t0 = time.perf_counter()
        out = calc.try_run(mol, cid)
        if out.record is None:
            seconds = time.perf_counter() - t0
            rows.append(
                _row(task.smiles, cid, reason=str(out.reason), detail=out.detail, seconds=seconds)
            )
            continue
        path = cache_path(task.config.cache_dir, task.smiles, cid, task.config.method)
        save_record(out.record, path)
        rows.append(
            _row(
                task.smiles,
                cid,
                cache_path=str(path),
                status="ok",
                warnings=out.record.warnings,
                seconds=time.perf_counter() - t0,
                n_embedded=len(ids),
            )
        )
    return rows


def process_molecule(task: MolTask) -> list[dict]:
    """Worker entry point. Never raises: every outcome becomes manifest rows."""
    t0 = time.perf_counter()
    signal.alarm(task.config.timeout_s)
    try:
        return _compute(task)
    except _Timeout:
        reason, detail = ReasonCode.TIMEOUT, f"> {task.config.timeout_s}s"
    except QMError as exc:
        reason, detail = exc.reason, exc.detail
    except Exception as exc:  # noqa: BLE001 - one bad molecule must not stop the batch
        log.exception("unexpected error for %s", task.smiles)
        reason, detail = ReasonCode.UNEXPECTED, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    seconds = time.perf_counter() - t0
    return [_row(task.smiles, 0, reason=str(reason), detail=detail, seconds=seconds)]


def collect_molecules(
    source: str, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> tuple[list[str], list[dict]]:
    """Unique map-free canonical SMILES of every fragment, plus rows for fragments that fail."""
    table = pq.read_table(
        steps_path(source, processed_dir), columns=["reactants_smiles", "products_smiles"]
    )
    sides = set(table.column("reactants_smiles").to_pylist())
    sides |= set(table.column("products_smiles").to_pylist())
    keys: set[str] = set()
    failures: dict[str, dict] = {}
    for side in sorted(sides):
        for frag in chem.fragments(chem.parse_mapped(side)):
            try:
                keys.add(chem.unmapped_canonical(frag))
            except (chem.SanitizeError, ValueError) as exc:
                raw = Chem.MolToSmiles(frag)
                failures[raw] = _row(
                    raw, -1, reason=str(ReasonCode.CANONICALIZE_FAILED), detail=str(exc)
                )
    return sorted(keys), list(failures.values())


def _is_done(
    task: MolTask,
    prev: pd.DataFrame | None,
    retry_failed: bool,
    retry_reasons: Collection[str] | None = None,
) -> bool:
    """Every conformer is cached, or the molecule failed before and is not to be retried:
    with `retry_reasons`, a failure is retried only when its reason code is one of them."""
    rows = None
    if prev is not None:
        rows = prev[(prev["smiles"] == task.smiles) & (prev["method"] == task.config.method)]
    n_expected = task.n_conformers
    if rows is not None and (rows["status"] == "ok").any():
        embedded = int(rows.loc[rows["status"] == "ok", "n_embedded"].max())
        n_expected = min(n_expected, embedded) if embedded else n_expected
    if all(
        is_valid_cache_file(cache_path(task.config.cache_dir, task.smiles, c, task.config.method))
        for c in range(n_expected)
    ):
        return True
    failed = None if rows is None else rows["status"] == "failed"
    if failed is None or not failed.any():
        return False
    if retry_reasons:
        return not rows.loc[failed, "reason"].isin(set(retry_reasons)).any()
    return not retry_failed


def _run_isolated(task: MolTask, on_rows: OnRows) -> None:
    """Run one task in its own pool, so a hard crash is pinned on this molecule."""
    with ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=_init_worker) as ex:
        try:
            on_rows(ex.submit(process_molecule, task).result())
        except BrokenProcessPool:
            log.error("worker crashed on %s", task.smiles)
            row = _row(task.smiles, 0, reason=str(ReasonCode.WORKER_CRASH), detail="pool broke")
            on_rows([row])


def _run_window(pending: list[MolTask], n_workers: int, on_rows: OnRows) -> list[MolTask]:
    """Keep a window of tasks in flight. Returns the tasks lost if the pool broke, else []."""
    in_flight: dict[Future, MolTask] = {}
    ex = ProcessPoolExecutor(n_workers, mp_context=get_context("spawn"), initializer=_init_worker)
    try:
        while pending or in_flight:
            while pending and len(in_flight) < n_workers * _WINDOW_PER_WORKER:
                task = pending.pop()
                in_flight[ex.submit(process_molecule, task)] = task
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            lost = []
            for fut in done:
                task = in_flight.pop(fut)
                try:
                    on_rows(fut.result())
                except BrokenProcessPool:
                    lost.append(task)
            if lost:  # every other in-flight future is dead too
                return lost + list(in_flight.values())
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return []


def _run_pool(tasks: list[MolTask], n_workers: int, on_rows: OnRows) -> None:
    pending = list(reversed(tasks))  # pop() takes from the end
    while pending:
        lost = _run_window(pending, n_workers, on_rows)
        if lost:
            log.error("process pool broke; re-running %d in-flight tasks one by one", len(lost))
        for task in lost:
            _run_isolated(task, on_rows)


def manifest_path(source: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return source_dir(source, processed_dir) / MANIFEST_NAME


def load_manifest(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if Path(path).is_file() else None


def build_manifest(
    smiles_list: list[str],
    n_conformers: int,
    config: QMConfig,
    rows: list[dict],
    prev: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge previous rows, this run's rows and cache hits. Rows of other methods are kept."""
    current: dict[tuple[str, int, str], dict] = {}
    if prev is not None:
        for r in prev.to_dict("records"):
            current[(r["smiles"], int(r["conformer_id"]), r["method"])] = r
    for r in rows:
        r["method"] = config.method
        current[(r["smiles"], int(r["conformer_id"]), config.method)] = r
    for smiles in smiles_list:
        for cid in range(n_conformers):
            path = cache_path(config.cache_dir, smiles, cid, config.method)
            key = (smiles, cid, config.method)
            if is_valid_cache_file(path) and (key not in current or current[key]["status"] != "ok"):
                current[key] = _row(
                    smiles, cid, cache_path=str(path), status="ok", method=config.method
                )
    out = pd.DataFrame(list(current.values()), columns=MANIFEST_COLUMNS)
    return out.sort_values(["method", "smiles", "conformer_id"], ignore_index=True)


def run_view(manifest: pd.DataFrame, smiles_list: list[str], config: QMConfig) -> pd.DataFrame:
    """Rows of one run: this method, the molecules asked for, plus canonicalization failures."""
    in_scope = manifest["smiles"].isin(set(smiles_list)) | (manifest["conformer_id"] == -1)
    return manifest[in_scope & (manifest["method"] == config.method)].reset_index(drop=True)


def summarize(manifest: pd.DataFrame) -> dict:
    ok = manifest["status"] == "ok"
    warn = Counter(w for ws in manifest.loc[ok, "warnings"] for w in ws.split(",") if w)
    return dict(
        n_total=len(manifest),
        n_ok=int(ok.sum()),
        success_fraction=float(ok.mean()) if len(manifest) else float("nan"),
        reasons=Counter(manifest.loc[~ok, "reason"]),
        warnings=warn,
    )


def print_summary(manifest: pd.DataFrame, console: Console) -> dict:
    s = summarize(manifest)
    console.print(
        f"[bold]QM featurization[/bold]: {s['n_ok']}/{s['n_total']} ok "
        f"({s['success_fraction']:.2%}), {s['n_total'] - s['n_ok']} failed"
    )
    failed = manifest[manifest["status"] != "ok"]
    table = Table(title="Failure reasons")
    for col, just in (("reason", "left"), ("count", "right"), ("% of molecules", "right")):
        table.add_column(col, justify=just)
    table.add_column("example")
    for reason, count in s["reasons"].most_common():
        ex = failed.loc[failed["reason"] == reason].iloc[0]
        table.add_row(reason, str(count), f"{count / s['n_total']:.2%}", f"{ex['smiles']}"[:60])
    if not s["reasons"]:
        table.add_row("(none)", "0", "0.00%", "")
    console.print(table)
    if s["warnings"]:
        wt = Table(title="Warnings on successful records")
        wt.add_column("warning")
        wt.add_column("count", justify="right")
        for w, count in s["warnings"].most_common():
            wt.add_row(w, str(count))
        console.print(wt)
    return s


def _execute(tasks: list[MolTask], n_workers: int, on_rows: OnRows) -> None:
    if n_workers <= 1:
        _init_worker()
        for task in tasks:
            on_rows(process_molecule(task))
        return
    prev_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"  # one thread per worker; children inherit this
    try:
        _run_pool(tasks, n_workers, on_rows)
    finally:
        if prev_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev_omp


def featurize_dataset(
    source: str,
    n_workers: int = 1,
    n_conformers: int = 1,
    config: QMConfig | None = None,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    limit: int | None = None,
    retry_failed: bool = True,
    console: Console | None = None,
    retry_reasons: Collection[str] | None = None,
) -> pd.DataFrame:
    """Featurize every unique molecule of `source`; resumable. Returns this run's manifest rows.
    Earlier failures are retried (`retry_failed`), or only those whose reason code is in
    `retry_reasons`."""
    config = config or QMConfig()
    console = console or Console()
    smiles_list, canon_failures = collect_molecules(source, processed_dir)
    if limit is not None and limit < len(smiles_list):
        smiles_list = sorted(random.Random(0).sample(smiles_list, limit))
    path = manifest_path(source, processed_dir)
    prev = load_manifest(path)
    if prev is not None and "method" not in prev.columns:
        prev = prev.assign(method="")  # manifests written before the method column existed
    tasks = [MolTask(s, n_conformers, config) for s in smiles_list]
    todo = [t for t in tasks if not _is_done(t, prev, retry_failed, retry_reasons)]
    console.print(
        f"{len(smiles_list)} unique molecules, {len(smiles_list) - len(todo)} done already, "
        f"{len(todo)} to compute ({n_workers} workers, {config.method})"
    )
    rows: list[dict] = list(canon_failures)
    bar = tqdm(total=len(todo), unit="mol", disable=not todo)

    def on_rows(new: list[dict]) -> None:
        rows.extend(new)
        bar.update(1)
        if bar.n % _FLUSH_EVERY == 0:
            _write(path, build_manifest(smiles_list, n_conformers, config, rows, prev))

    try:
        _execute(todo, n_workers, on_rows)
    finally:
        bar.close()
        full = build_manifest(smiles_list, n_conformers, config, rows, prev)
        _write(path, full)
    view = run_view(full, smiles_list, config)
    print_summary(view, console)
    console.print(f"wrote manifest: {path}")
    return view


def _write(path: Path, manifest: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
