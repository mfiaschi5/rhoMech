"""GFN2-xTB single points at a dataset's own geometries: QMugs (`data.qmugs`).

Each conformer of `structures.tar.gz` is put in record order (`qmugs.record_mol`) and run with
`QMConfig(given_geometry=True)`: no optimization, only a check that every bond of the Lewis
structure is a bond in the QMugs geometry, then the single point, the N+-1 Fukui and the
sterics of every QM record. Records go to their own cache (`data/cache/qmugs/qm`), keyed like
every record by the map-free canonical SMILES, the conformer id and the method
(`GFN2-xTB/ALPB-water/given`). A molecule that QMugs lists under two ChEMBL ids is computed
once.

The archive is read as a stream, in chunks of `chunk` conformers, so no more than one chunk of
molfiles is in memory. `limit` takes the first conformers in archive order (ChEMBL ids sorted
as strings), so a larger limit later extends a run. The manifest
(`data/processed/qmugs/qm_manifest.parquet`) has a row per (ChEMBL id, conformer) with its
status, reason code and time; conformers already in it are skipped (failures too, unless
`retry_failed`).
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from tqdm import tqdm

from ksmech.arrows.reactants import CorrespondenceError
from ksmech.data import chem, qmugs
from ksmech.qm.batch import _Timeout, execute, print_summary
from ksmech.qm.records import QMError, ReasonCode, cache_path, is_valid_cache_file, save_record
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "chembl_id",
    "conformer_id",
    "smiles",
    "n_atoms",
    "cache_path",
    "status",
    "reason",
    "detail",
    "warnings",
    "seconds",
    "method",
]


@dataclass(frozen=True)
class GivenTask:
    chembl_id: str
    conformer_id: int
    molblock: str
    config: QMConfig

    def __str__(self) -> str:
        return f"{self.chembl_id}/conf_{self.conformer_id:02d}"


def _row(task: GivenTask, **kw) -> dict:
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row |= {"chembl_id": task.chembl_id, "conformer_id": task.conformer_id, "n_atoms": 0}
    row |= {"status": "failed", "seconds": 0.0, "method": task.config.method}
    return row | kw


def _compute(task: GivenTask, row: dict) -> None:
    """Fill `row` for one conformer; raises QMError, ValueError or CorrespondenceError."""
    key, mol = qmugs.record_mol(task.molblock, task.conformer_id)
    row |= {"smiles": key, "n_atoms": mol.GetNumAtoms()}
    path = cache_path(task.config.cache_dir, key, task.conformer_id, task.config.method)
    if not is_valid_cache_file(path):
        out = GFN2Calculator(task.config).try_run(mol, task.conformer_id)
        if out.record is None:
            row |= {"reason": str(out.reason), "detail": out.detail}
            return
        save_record(out.record, path)
        row["warnings"] = out.record.warnings
    row |= {"status": "ok", "cache_path": str(path)}


def run_given(task: GivenTask) -> list[dict]:
    """Worker entry point. Never raises: every outcome becomes one manifest row."""
    t0 = time.perf_counter()
    row = _row(task)
    signal.alarm(task.config.timeout_s)
    try:
        _compute(task, row)
    except _Timeout:
        row |= {"reason": str(ReasonCode.TIMEOUT), "detail": f"> {task.config.timeout_s}s"}
    except QMError as exc:
        row |= {"reason": str(exc.reason), "detail": exc.detail}
    except (ValueError, chem.SanitizeError) as exc:
        row |= {"reason": str(ReasonCode.SDF_UNREADABLE), "detail": str(exc)}
    except CorrespondenceError as exc:
        row |= {"reason": str(ReasonCode.RECORD_ORDER), "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001 - one bad conformer must not stop the batch
        log.exception("unexpected error for %s", task)
        row |= {"reason": str(ReasonCode.UNEXPECTED), "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        signal.alarm(0)
    row["seconds"] = time.perf_counter() - t0
    return [row]


def _crash(task: GivenTask) -> dict:
    return _row(task, reason=str(ReasonCode.WORKER_CRASH), detail="pool broke")


def _done(prev: pd.DataFrame | None, method: str, retry_failed: bool) -> set[tuple[str, int]]:
    if prev is None:
        return set()
    rows = prev[prev["method"] == method]
    if retry_failed:
        rows = rows[rows["status"] == "ok"]
    return set(zip(rows["chembl_id"], rows["conformer_id"].astype(int), strict=True))


def _write(path: Path, prev: pd.DataFrame | None, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out = new if prev is None else pd.concat([prev, new], ignore_index=True)
    out = out.drop_duplicates(["chembl_id", "conformer_id", "method"], keep="last")
    out = out.sort_values(["method", "chembl_id", "conformer_id"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def featurize_qmugs(
    tar_path: Path = qmugs.DEFAULT_TAR,
    conformers: Collection[int] = (0,),
    n_workers: int = 1,
    limit: int | None = None,
    chunk: int = 2000,
    config: QMConfig | None = None,
    out_dir: Path = qmugs.PROCESSED_DIR,
    retry_failed: bool = False,
    console: Console | None = None,
) -> pd.DataFrame:
    """Single points of the first `limit` conformers of the archive (all when None) that the
    manifest does not have yet; resumable. Returns the manifest rows of those conformers."""
    config = config or QMConfig(given_geometry=True, cache_dir=qmugs.CACHE_DIR)
    console = console or Console()
    path = Path(out_dir) / "qm_manifest.parquet"
    prev = pd.read_parquet(path) if path.is_file() else None
    done = _done(prev, config.method, retry_failed)
    rows: list[dict] = []
    wanted: list[tuple[str, int]] = []
    bar = tqdm(total=limit, unit="conf")
    pending: list[GivenTask] = []

    def on_rows(new: list[dict]) -> None:
        rows.extend(new)
        bar.update(len(new))

    def run(tasks: list[GivenTask]) -> None:
        execute(tasks, n_workers, on_rows, run_given, _crash)
        _write(path, prev, rows)

    for chembl_id, conf_id, block in qmugs.iter_conformers(tar_path, conformers):
        if limit is not None and len(wanted) >= limit:
            break
        wanted.append((chembl_id, conf_id))
        if (chembl_id, conf_id) in done:
            bar.update(1)
            continue
        pending.append(GivenTask(chembl_id, conf_id, block, config))
        if len(pending) == chunk:
            run(pending)
            pending = []
    if pending:
        run(pending)
    bar.close()
    full = _write(path, prev, rows)
    keys = set(wanted)
    view = full[[k in keys for k in zip(full["chembl_id"], full["conformer_id"], strict=True)]]
    console.print(f"{len(wanted)} conformers, {len(rows)} computed now ({n_workers} workers)")
    print_summary(view, console)
    console.print(f"wrote manifest: {path}")
    return view.reset_index(drop=True)
