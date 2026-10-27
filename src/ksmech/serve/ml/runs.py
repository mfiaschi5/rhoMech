"""The training runs and the physics / eval reports on disk, read-only, without torch (lab only).

A run is a directory `data/models/<source>/<run>/` of `checkpoints.model_roots` holding a
`run.json`, a `metrics.jsonl` or a checkpoint (a run trained on a pod may have no `.pt` here).
The first root wins for a (source, run) present in both. Its summary takes the facts the
checkpoint listing reads (`checkpoints.run_facts`) and passes the rest through as it is:
`provenance.config` of run.json, the metrics lines of the best and the last epoch, and the
detail adds run.json whole, every metrics line, and the report files beside the checkpoints
(`*_report_*.json`, such as `ksmech-eval decode`'s `decode_report_<split>_<part>.json`).

The reports of `ksmech-eval floor` and `ksmech-physics gbdt` are
`data/processed/<source>/{anchor_floor,gbdt_lgbm}_report_*.json` under the shared root, then the
data root (`report_roots`). A report is a file envelope (name, kind, size, time) around its JSON
as it is; a file larger than `MAX_REPORT_BYTES` or unreadable has `report` None and a reason,
and is logged with its reason code.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.ml.checkpoints import NAME_RE, epoch_metrics, model_roots, read_json, run_facts

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

log = logging.getLogger(__name__)

RUN_MARKERS = ("run.json", "metrics.jsonl")
RUN_LOGS = ("train.log", "stdout.log")
RUN_REPORTS = "*_report_*.json"
PROCESSED_REPORTS = ("anchor_floor_report_*.json", "gbdt_lgbm_report_*.json")
MAX_REPORT_BYTES = 4 << 20


def _iso(mtime_ns: int) -> datetime:
    return datetime.fromtimestamp(mtime_ns / 1e9, UTC)


def _is_run(run_dir: Path) -> bool:
    if not run_dir.is_dir() or not all(NAME_RE.fullmatch(x) for x in run_dir.parts[-2:]):
        return False
    return any((run_dir / m).is_file() for m in RUN_MARKERS) or any(run_dir.glob("*.pt"))


def run_dirs(settings: ServeSettings) -> dict[tuple[str, str], Path]:
    """{(source, run): directory} of every run, sorted; the first root wins."""
    found: dict[tuple[str, str], Path] = {}
    for root in model_roots(settings):
        for run_dir in sorted(root.glob("*/*")):
            key = (run_dir.parent.name, run_dir.name)
            if key not in found and _is_run(run_dir):
                found[key] = run_dir
    return dict(sorted(found.items()))


def metrics_lines(run_dir: Path) -> tuple[list[dict], int]:
    """Every JSON-object line of metrics.jsonl in file order, and the number of other lines."""
    try:
        text = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], 0
    except OSError as exc:
        log.warning("unreadable %s/metrics.jsonl [METRICS_UNREADABLE]: %s", run_dir, exc)
        return [], 0
    lines, bad = [], 0
    for line in filter(str.strip, text.splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            lines.append(doc)
        else:
            bad += 1
    if bad:
        log.warning("%d bad lines in %s/metrics.jsonl [METRICS_LINE_UNREADABLE]", bad, run_dir)
    return lines, bad


def _newest(run_dir: Path) -> int:
    times = [p.stat().st_mtime_ns for p in run_dir.iterdir() if p.is_file()]
    return max(times, default=run_dir.stat().st_mtime_ns)


def run_summary(source: str, run_dir: Path, lines: list[dict]) -> dict[str, Any]:
    """`MlRunOut` fields of one run directory."""
    facts = run_facts(run_dir)
    header = (read_json(run_dir / "run.json") or {}).get("provenance")
    config = header.get("config") if isinstance(header, dict) else None
    scored = [m for m in lines if isinstance(m.get("val_top1"), int | float)]
    return {
        "source": source,
        "run": run_dir.name,
        "path": str(run_dir),
        "encoder": facts.get("encoder"),
        "model": facts.get("model") or {},
        "clean": facts.get("clean"),
        "reason": facts.get("reason"),
        "built_at": facts.get("built_at"),
        "smoke": bool(facts.get("smoke")),
        "config": config if isinstance(config, dict) else None,
        "n_epochs": len(scored),
        "best": epoch_metrics(scored, "best"),
        "last": epoch_metrics(scored, "last"),
        "checkpoints": sorted(p.stem for p in run_dir.glob("*.pt")),
        "reports": sorted(p.name for p in run_dir.glob(RUN_REPORTS)),
        "logs": [name for name in RUN_LOGS if (run_dir / name).is_file()],
        "modified_at": _iso(_newest(run_dir)),
    }


def list_runs(settings: ServeSettings) -> list[dict[str, Any]]:
    """`MlRunOut` fields of every run (`run_dirs` order); an unreadable run is logged and
    skipped with its reason code."""
    out = []
    for (source, _), run_dir in run_dirs(settings).items():
        try:
            out.append(run_summary(source, run_dir, metrics_lines(run_dir)[0]))
        except OSError as exc:
            log.warning("skipping %s [RUN_UNREADABLE]: %s", run_dir, exc)
    return out


def report_out(source: str, path: Path) -> dict[str, Any]:
    """`MlReportOut` fields of one report file."""
    st = path.stat()
    report, reason = None, None
    if st.st_size > MAX_REPORT_BYTES:
        reason = f"REPORT_TOO_LARGE: {st.st_size} bytes > {MAX_REPORT_BYTES}"
    else:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            report = doc if isinstance(doc, dict) else {"value": doc}
        except (OSError, ValueError) as exc:
            reason = f"REPORT_UNREADABLE: {exc}"
    if reason:
        log.warning("report %s: %s", path, reason)
    return {
        "source": source,
        "name": path.name,
        "kind": path.name.split("_report_")[0],
        "path": str(path),
        "size_bytes": int(st.st_size),
        "modified_at": _iso(st.st_mtime_ns),
        "report": report,
        "reason": reason,
    }


def run_detail(settings: ServeSettings, source: str, run: str) -> dict[str, Any]:
    """`MlRunDetailOut` fields; RUN_NOT_FOUND (404) for an unknown run."""
    run_dir = run_dirs(settings).get((source, run))
    if run_dir is None:
        where = ", ".join(str(r) for r in model_roots(settings))
        raise ServeError(Code.RUN_NOT_FOUND, f"no run {source}/{run} under {where}", status=404)
    lines, bad = metrics_lines(run_dir)
    return {
        "run": run_summary(source, run_dir, lines),
        "run_json": read_json(run_dir / "run.json"),
        "epochs": lines,
        "bad_lines": bad,
        "reports": [report_out(source, p) for p in sorted(run_dir.glob(RUN_REPORTS))],
    }


def report_roots(settings: ServeSettings) -> list[Path]:
    """`data/processed` of the shared root, then of the data root (once when they coincide)."""
    out: list[Path] = []
    for r in (settings.shared_root / "data" / "processed", settings.processed):
        if r.resolve() not in [o.resolve() for o in out]:
            out.append(r)
    return out


def list_reports(settings: ServeSettings) -> list[dict[str, Any]]:
    """`MlReportOut` fields of every physics / eval report, by (source, name); the first root
    wins for a file present in both."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for root in report_roots(settings):
        for pattern in PROCESSED_REPORTS:
            for path in root.glob(f"*/{pattern}"):
                key = (path.parent.name, path.name)
                if key in found or not NAME_RE.fullmatch(key[0]):
                    continue
                try:
                    found[key] = report_out(key[0], path)
                except OSError as exc:
                    log.warning("skipping %s [REPORT_UNREADABLE]: %s", path, exc)
    return [found[k] for k in sorted(found)]
