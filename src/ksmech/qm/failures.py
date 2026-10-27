"""The failure memo: a QM calculation that failed for a reason that would recur leaves a
tombstone next to the records, `<cache key>.failed.json`, so no later run repeats it with the
same settings (docs/decisions.md, 2026-10-27).

The tombstone holds the molecule, the reason, the detail, the time spent and every setting of
the `QMConfig` it ran with (`params`, the cache directory excepted), under the provenance header
of a QM record (kind `qm_failure`). A later calculation of the same (SMILES, conformer, method)
is skipped while the tombstone *covers* its settings: the same settings, with no budget
(`timeout_s`, `maxiter`, `scf_maxiter`) larger than the one that failed. A larger budget, or
any other changed setting, runs again; a success removes the tombstone, and a record on disk
always wins over a tombstone.

Only reasons a rerun with the same settings would repeat are remembered (`MEMO_REASONS`); a
crashed worker, an unexpected exception or a missing binary are not. Writers are the functions
that run the calculation (`qm.batch.process_molecule`, serve's metal path), so the batch and
the web app share one memo per cache directory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ksmech.data import provenance as prov
from ksmech.qm.records import (
    ReasonCode,
    cache_key,
    cache_path,
    is_valid_cache_file,
    record_config,
    record_inputs,
)

log = logging.getLogger(__name__)

KIND = "qm_failure"
SCHEMA_VERSION = 1
SUFFIX = ".failed.json"
KNOWN_PREFIX = "known failure"
MEMO_REASONS = frozenset(
    {
        ReasonCode.EMBED_FAILED,
        ReasonCode.SPIN_PARITY,
        ReasonCode.OPT_NOT_CONVERGED,
        ReasonCode.SCF_NOT_CONVERGED,
        ReasonCode.CONNECTIVITY_CHANGED,
        ReasonCode.TIMEOUT,
        ReasonCode.UNSUPPORTED_ELEMENT,
    }
)
BUDGETS = ("timeout_s", "maxiter", "scf_maxiter")  # a larger value may succeed where one failed


@dataclass(frozen=True)
class Failure:
    smiles: str
    conformer_id: int
    method: str
    reason: str
    detail: str
    seconds: float
    params: dict[str, Any]
    path: Path


def tombstone_path(cache_dir: Path, smiles: str, conformer_id: int, method: str) -> Path:
    return Path(cache_dir) / f"{cache_key(smiles, conformer_id, method)}{SUFFIX}"


def config_params(config: Any) -> dict[str, Any]:
    """Every setting of a `QMConfig` (or subclass) but its cache directory, as read from JSON."""
    raw = {f.name: getattr(config, f.name) for f in fields(config) if f.name != "cache_dir"}
    return json.loads(json.dumps(raw, sort_keys=True, default=str))


def covers(failed: Mapping[str, Any], asked: Mapping[str, Any]) -> bool:
    """Whether a failure with settings `failed` predicts one with settings `asked`: the same
    settings, with no budget larger than the one that failed."""
    if set(failed) != set(asked):
        return False
    return all(
        _within(value, failed[name]) if name in BUDGETS else value == failed[name]
        for name, value in asked.items()
    )


def _within(asked: Any, failed: Any) -> bool:
    numbers = isinstance(asked, int | float) and isinstance(failed, int | float)
    return numbers and asked <= failed


def read(path: Path) -> Failure | None:
    """The tombstone at `path`, or None when it is missing or unreadable (logged)."""
    try:
        d = json.loads(Path(path).read_text())
        return Failure(
            str(d["smiles"]),
            int(d["conformer_id"]),
            str(d["method"]),
            str(d["reason"]),
            str(d["detail"]),
            float(d["seconds"]),
            dict(d["params"]),
            Path(path),
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("unreadable failure tombstone %s: %s", path, exc)
        return None


def lookup(cache_dir: Path, smiles: str, conformer_id: int, config: Any) -> Failure | None:
    """The remembered failure of this molecule that covers `config`'s settings, looked up in
    `cache_dir` (not `config.cache_dir`, so a caller can consult several layers); None when
    there is none or a record exists."""
    method = config.method
    if is_valid_cache_file(cache_path(cache_dir, smiles, conformer_id, method)):
        return None
    hit = read(tombstone_path(cache_dir, smiles, conformer_id, method))
    if hit is None or (hit.smiles, hit.conformer_id, hit.method) != (smiles, conformer_id, method):
        return None
    return hit if covers(hit.params, config_params(config)) else None


def known_detail(hit: Failure) -> str:
    """The detail of a row or result answered from the memo instead of computed."""
    return f"{KNOWN_PREFIX} ({hit.path.name}, not recomputed): {hit.detail}"


def remember(
    config: Any, smiles: str, conformer_id: int, reason: str, detail: str, seconds: float
) -> Path | None:
    """Write the tombstone of a failure whose reason recurs; None for any other reason. An
    existing tombstone that already covers these settings (it failed with larger budgets) is
    kept."""
    if reason not in MEMO_REASONS:
        return None
    params = config_params(config)
    path = tombstone_path(config.cache_dir, smiles, conformer_id, config.method)
    old = read(path) if path.is_file() else None
    if old is not None and old.reason == reason and covers(old.params, params):
        return path
    body = dict(
        smiles=smiles,
        conformer_id=int(conformer_id),
        method=config.method,
        reason=str(reason),
        detail=str(detail),
        seconds=float(seconds),
        params=params,
    )
    header = prov.stamp(
        KIND,
        SCHEMA_VERSION,
        record_inputs(smiles),
        record_config(config.method, conformer_id),
        prov.sha256_text(json.dumps(body, sort_keys=True)),
    )
    return prov.write_json(path, body, header)


def forget(config: Any, smiles: str, conformer_id: int) -> None:
    tombstone_path(config.cache_dir, smiles, conformer_id, config.method).unlink(missing_ok=True)


def record_outcome(rows: Iterable[Mapping[str, Any]], config: Any) -> None:
    """Tombstones for the failed rows a calculation returned, none for the conformers that
    succeeded. A memo write never fails the calculation: an OSError is logged."""
    for row in rows:
        cid = int(row["conformer_id"])
        if cid < 0:
            continue
        try:
            if row["status"] == "ok":
                forget(config, str(row["smiles"]), cid)
            else:
                remember(
                    config,
                    str(row["smiles"]),
                    cid,
                    str(row["reason"]),
                    str(row["detail"]),
                    float(row["seconds"]),
                )
        except OSError as exc:
            log.warning("failure memo of %s not updated: %s", row["smiles"], exc)
