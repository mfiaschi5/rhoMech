"""`ksmech-qm thermo`: the thermochemistry sidecar (`qm.thermo_cache`) of every cached record of
a set of species, in spawned workers (`qm.batch.execute`: one thread each, tblite on the BLAS
`qm.blas` chooses, a crashed worker's task re-run alone).

Every species ends in one row with a status: `ok` (computed now), `up_to_date` (a current
sidecar exists), `known_failure` (a tombstone covers these settings), `skipped` (NO_RECORD,
RECORD_STALE, UNKNOWN_METHOD) or `failed` (SCF_NOT_CONVERGED, TIMEOUT, UNEXPECTED,
WORKER_CRASH). The counts by status and reason, the warnings, the re-optimizations along
imaginary modes by outcome and stop code (`qm.thermo_reopt`: MINIMUM, REOPTIMIZED, PERSISTENT
with REOPT_ROUNDS, REOPT_NO_LOWER_MINIMUM, REOPT_BUDGET, REOPT_HESSIAN_FAILED, REOPT_OFF,
REOPT_NO_GRAPH) with their energy drops and the directions not kept by reason, and the time
per molecule against its atom count are printed at the end; `--report` writes every row as
JSON, with ZPE, H, S and G at the asked temperatures and the re-optimization's fields.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from ksmech.data import provenance as prov
from ksmech.qm import batch
from ksmech.qm.records import QMError, cache_path, is_valid_cache_file, load_record
from ksmech.qm.spin import method_for
from ksmech.qm.thermo import DEFAULTS, HARTREE_KCAL, ThermoSettings, thermo_table
from ksmech.qm.thermo_cache import (
    ThermoCode,
    ThermoConfig,
    compute_thermo,
    forget_failure,
    is_current,
    known_failure,
    load_thermo,
    read_arrays,
    remember_failure,
    reopt_fields,
    save_thermo,
    thermo_path,
)
from ksmech.qm.xtb_calc import solvent_of

log = logging.getLogger(__name__)

ATOM_BINS = (1, 2, 10, 20, 40, 60, 80, 120, 10_000)


@dataclass(frozen=True)
class ThermoTask:
    smiles: str
    conformer_id: int
    method: str  # the record's method, spin tag included
    config: ThermoConfig
    memo: bool = True  # consult the tombstones; off for an asked-for retry


def make_tasks(
    smiles: Iterable[str],
    base_method: str,
    config: ThermoConfig,
    conformer_id: int = 0,
    memo: bool = True,
) -> list[ThermoTask]:
    """One task per species, under its own record method (`qm.spin.method_for`)."""
    return [
        ThermoTask(s, conformer_id, method_for(s, base_method), config, memo)
        for s in sorted(set(smiles))
    ]


def _row(task: ThermoTask, status: str, **kw) -> dict[str, Any]:
    row = dict(
        smiles=task.smiles,
        conformer_id=task.conformer_id,
        method=task.method,
        status=status,
        reason="",
        detail="",
        warnings="",
        seconds=0.0,
        n_atoms=0,
        path="",
    )
    row.update(kw)
    return row


def precheck(task: ThermoTask) -> dict[str, Any] | None:
    """The row of a task that needs no calculation (no record, a current sidecar, a covering
    tombstone), else None."""
    cfg = task.config
    record = cache_path(cfg.qm_dir, task.smiles, task.conformer_id, task.method)
    if not is_valid_cache_file(record):
        return _row(task, "skipped", reason=str(ThermoCode.NO_RECORD), detail=str(record))
    out = thermo_path(cfg.out_dir, task.smiles, task.conformer_id, task.method)
    if is_current(out, task.method, task.conformer_id, cfg):
        return _row(task, "up_to_date", path=str(out))
    hit = known_failure(cfg, task.smiles, task.conformer_id, task.method) if task.memo else None
    if hit is not None:
        detail = f"known failure ({hit.path.name}, not recomputed): {hit.detail}"
        return _row(task, "known_failure", reason=hit.reason, detail=detail)
    return None


def _compute(task: ThermoTask) -> dict[str, Any]:
    cfg = task.config
    path = cache_path(cfg.qm_dir, task.smiles, task.conformer_id, task.method)
    try:
        record = load_record(path, (task.smiles, task.conformer_id, task.method))
    except (prov.StaleArtifact, prov.CorruptArtifact, KeyError, ValueError) as exc:
        return _row(task, "skipped", reason=str(ThermoCode.RECORD_STALE), detail=str(exc))
    try:
        solvent_of(record.method)
    except ValueError as exc:
        return _row(task, "skipped", reason=str(ThermoCode.UNKNOWN_METHOD), detail=str(exc))
    t0 = time.perf_counter()
    arrays = compute_thermo(record, cfg)
    seconds = time.perf_counter() - t0
    out = save_thermo(arrays, path, cfg, seconds)
    return _row(
        task,
        "ok",
        warnings=str(arrays["warnings"]),
        seconds=seconds,
        n_atoms=record.n_atoms,
        path=str(out),
        **reopt_fields(arrays),
    )


def _failed(task: ThermoTask, reason: str, detail: str, t0: float) -> dict[str, Any]:
    return _row(task, "failed", reason=reason, detail=detail, seconds=time.perf_counter() - t0)


def _run_task(task: ThermoTask) -> dict[str, Any]:
    t0 = time.perf_counter()
    signal.alarm(task.config.timeout_s)
    try:
        return _compute(task)
    except batch._Timeout:  # noqa: SLF001 - the QM batch's SIGALRM exception
        return _failed(task, str(ThermoCode.TIMEOUT), f"> {task.config.timeout_s}s", t0)
    except QMError as exc:
        return _failed(task, str(exc.reason), exc.detail, t0)
    except Exception as exc:  # noqa: BLE001 - one bad molecule must not stop the batch
        log.exception("unexpected thermo error for %s", task.smiles)
        return _failed(task, str(ThermoCode.UNEXPECTED), f"{type(exc).__name__}: {exc}", t0)
    finally:
        signal.alarm(0)


def process_task(task: ThermoTask) -> list[dict[str, Any]]:
    """Worker entry point; never raises. A recurring failure leaves a tombstone, a success
    removes it."""
    row = precheck(task) or _run_task(task)
    try:
        if row["status"] == "ok":
            forget_failure(task.config, task.smiles, task.conformer_id, task.method)
        elif row["status"] == "failed":
            remember_failure(
                task.config,
                task.smiles,
                task.conformer_id,
                task.method,
                row["reason"],
                row["detail"],
                row["seconds"],
            )
    except OSError as exc:
        log.warning("thermo failure memo of %s not updated: %s", task.smiles, exc)
    return [row]


def _crash_row(task: ThermoTask) -> dict[str, Any]:
    return _row(task, "failed", reason=str(ThermoCode.WORKER_CRASH), detail="pool broke")


def run_thermo(
    tasks: Sequence[ThermoTask], n_workers: int = 1, console: Console | None = None
) -> list[dict[str, Any]]:
    """Every task's row: the ones that need no calculation answered here, the others computed
    in `n_workers` processes. Failures that recur leave tombstones (`process_task`)."""
    console = console or Console()
    rows, todo = [], []
    for task in tasks:
        pre = precheck(task)
        if pre is None:
            todo.append(task)
        else:
            rows.append(pre)
    console.print(
        f"{len(tasks)} species, {len(tasks) - len(todo)} answered without a calculation, "
        f"{len(todo)} Hessians to compute ({n_workers} workers, {batch.blas_note(n_workers)})"
    )
    bar = tqdm(total=len(todo), unit="mol", disable=not todo)

    def on_rows(new: list[dict]) -> None:
        rows.extend(new)
        bar.update(1)

    try:
        batch.execute(todo, n_workers, on_rows, worker=process_task, crash=_crash_row)
    finally:
        bar.close()
    return sorted(rows, key=lambda r: r["smiles"])


# ---- the summary and the report


def _status_table(rows: list[dict]) -> Table:
    table = Table(title="thermo: outcome by status and reason")
    for col, just in (("status", "left"), ("reason", "left"), ("count", "right")):
        table.add_column(col, justify=just)
    table.add_column("example")
    counts = Counter((r["status"], r["reason"]) for r in rows)
    for (status, reason), n in sorted(counts.items()):
        ex = next(r["smiles"] for r in rows if (r["status"], r["reason"]) == (status, reason))
        table.add_row(status, reason or "-", str(n), ex[:60])
    return table


def timing_bins(rows: list[dict]) -> list[tuple[str, int, float, float, float]]:
    """(atom range, molecules computed, mean s, median s, max s) of the `ok` rows by atom
    count."""
    out = []
    done = [r for r in rows if r["status"] == "ok"]
    for lo, hi in zip(ATOM_BINS[:-1], ATOM_BINS[1:], strict=True):
        s = np.array([r["seconds"] for r in done if lo <= r["n_atoms"] < hi])
        if len(s):
            label = f"{lo}-{hi - 1}" if hi < ATOM_BINS[-1] else f">={lo}"
            out.append((label, len(s), float(s.mean()), float(np.median(s)), float(s.max())))
    return out


def reopt_counts(rows: list[dict]) -> list[tuple[str, str, int, float, float, float, str]]:
    """(outcome, stop code, molecules, mean rounds, mean and max energy drop in kcal/mol, an
    example) of the `ok` rows by the re-optimization's outcome and stop code."""
    done = [r for r in rows if r["status"] == "ok" and "reopt_outcome" in r]
    out = []
    for key in sorted({(r["reopt_outcome"], r["reopt_stop"]) for r in done}):
        group = [r for r in done if (r["reopt_outcome"], r["reopt_stop"]) == key]
        drops = np.array([r["energy_drop_kcal"] for r in group])
        rounds = float(np.mean([r["reopt_rounds"] for r in group]))
        out.append((*key, len(group), rounds, float(drops.mean()), float(drops.max()),
                    group[0]["smiles"]))  # fmt: skip
    return out


def _reopt_table(rows: list[dict]) -> Table:
    table = Table(title="re-optimization along imaginary modes (computed now)")
    table.add_column("outcome", no_wrap=True, min_width=11)
    table.add_column("stop", no_wrap=True, min_width=22)
    for col in ("count", "rounds", "drop", "max"):
        table.add_column(col, justify="right", no_wrap=True)
    for outcome, stop, n, rounds, mean, top, _ in reopt_counts(rows):
        cells = (str(n), f"{rounds:.2f}", f"{mean:.3f}", f"{top:.3f}")
        table.add_row(outcome, stop or "-", *cells)
    return table


def print_summary(rows: list[dict], console: Console) -> None:
    console.print(_status_table(rows))
    warn = Counter(w for r in rows if r["status"] == "ok" for w in r["warnings"].split(",") if w)
    for w, n in warn.most_common():
        console.print(f"  warning {w}: {n}")
    if reopt_counts(rows):
        console.print(_reopt_table(rows))
        console.print("  (rounds kept; drops in kcal/mol, record minus thermo minimum: mean, max)")
    rejected = Counter(
        w for r in rows if r["status"] == "ok" for w in r.get("reopt_rejected", "").split(",") if w
    )
    for w, n in rejected.most_common():
        console.print(f"  re-optimized direction not kept, {w}: {n}")
    bins = timing_bins(rows)
    if bins:
        table = Table(title="time per Hessian by atom count (computed now, re-optimizations in)")
        for col in ("atoms", "molecules", "mean s", "median s", "max s"):
            table.add_column(col, justify="right")
        for label, n, mean, med, top in bins:
            table.add_row(label, str(n), f"{mean:.1f}", f"{med:.1f}", f"{top:.1f}")
        console.print(table)


def _thermo_rows(path: str, temperatures: Sequence[float], settings: ThermoSettings) -> list:
    data = load_thermo(Path(path))
    return [
        {
            "T": r.temperature,
            "zpe_ha": r.zpe,
            "h_ha": r.enthalpy,
            "s_ha_per_k": r.entropy,
            "g_ha": r.free_energy,
            "g_minus_e_kcal": r.g_correction * HARTREE_KCAL,
            "n_inverted": r.modes.n_inverted,
            "n_dropped_imaginary": r.modes.n_dropped_imaginary,
        }
        for r in thermo_table(data, temperatures, settings)
    ]


def write_report(
    rows: list[dict],
    path: Path,
    temperatures: Sequence[float],
    settings: ThermoSettings = DEFAULTS,
) -> Path:
    """Every row, with ZPE, H, S and G at `temperatures` and the re-optimization's fields for
    the species that have a sidecar."""
    out = []
    for r in rows:
        extra = {}
        if r["status"] in ("ok", "up_to_date"):
            extra = reopt_fields(read_arrays(Path(r["path"])))
            extra["thermo"] = _thermo_rows(r["path"], temperatures, settings)
        out.append({**r, **extra})
    body = {
        "temperatures": list(temperatures),
        "settings": settings.__dict__,
        "counts": dict(Counter(f"{r['status']}:{r['reason']}" for r in rows)),
        "reopt": reopt_counts(rows),
        "timing": timing_bins(rows),
        "rows": out,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1, default=str) + "\n")
    return path
