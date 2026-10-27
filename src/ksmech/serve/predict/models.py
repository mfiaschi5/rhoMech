"""The local spec-v9 checkpoints (`ksmech-train fit` runs), listed on the API side without torch.

A checkpoint is `data/models/<source>/<run>/<file>.pt` under the shared root
(`KSMECH_SHARED_CACHE_ROOT`) or the data root. The listing reads only small files beside it:

- `run.json` (the run's provenance header): the encoder and the model size, and whether the run
  was trained from a clean tree. `train.loop.load_model` refuses a checkpoint whose own header
  is dirty (STALE_ARTIFACT); the listing reports that as a reason, never as a crash, and the
  worker checks the checkpoint's own header again when it loads it.
- `metrics.jsonl` (one line per epoch): the epoch and the validation metrics of `last.pt` (the
  last line) and of `best.pt` (the line the trainer kept: the largest val top-1, ties broken by
  the smaller val loss, `train.loop._end_epoch`). The exact epoch is in the checkpoint itself;
  the trace page shows that one.
- A **smoke** run is a `ksmech-train fit --limit N` run (`config.train.limit > 0` in run.json)
  or a run named `smoke*`: listed and usable, the default only when nothing else is usable
  (`default_checkpoint`).

`checkpoint_sha256` hashes a checkpoint once per file version (mtime, size).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ksmech.serve.core.contracts import LRU
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.core.versions import file_sha256

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
METRIC_KEYS = ("val_top1", "val_top3", "val_top5", "val_r_exact", "val_loss", "val_anchor_nll")
MODULES = ("torch", "ksmech.model.net", "ksmech.train.loop", "ksmech.eval.embedding")
_SHA: LRU[tuple[str, int, int], str] = LRU(16)


@dataclass
class Checkpoint:
    """One local checkpoint and what its run directory says about it."""

    source: str
    run: str
    file: str  # the stem: "best", "last", ...
    path: Path
    size_bytes: int
    mtime_ns: int
    encoder: str | None = None
    model: dict[str, Any] = field(default_factory=dict)  # d, layers, heads, d_pair
    epoch: int | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    clean: bool | None = None  # None: no run.json, checked when the worker loads it
    reason: str | None = None  # why it is not usable (a dirty run)
    built_at: str | None = None
    smoke: bool = False  # a --limit run or a run named smoke* (is_smoke)

    @property
    def usable(self) -> bool:
        return self.clean is not False

    @property
    def signature(self) -> tuple[int, int]:
        return self.mtime_ns, self.size_bytes

    def to_out(self) -> dict:
        """`CheckpointOut` fields."""
        d = asdict(self)
        d["path"] = str(self.path)
        d["usable"] = self.usable
        d["modified_at"] = datetime.fromtimestamp(self.mtime_ns / 1e9, UTC).isoformat()
        del d["mtime_ns"]
        return d


def model_roots(settings: ServeSettings) -> list[Path]:
    """`data/models` of the shared root, then of the data root (once when they coincide)."""
    roots = [settings.shared_root / "data" / "models", settings.data_root / "data" / "models"]
    out: list[Path] = []
    for r in roots:
        if r.resolve() not in [o.resolve() for o in out]:
            out.append(r)
    return out


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("unreadable %s [RUN_JSON_UNREADABLE]: %s", path, exc)
        return None


def _metrics_lines(run_dir: Path) -> list[dict]:
    try:
        text = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            log.warning("skipping a bad line of %s [METRICS_LINE_UNREADABLE]", run_dir)
    return [m for m in out if isinstance(m, dict) and "val_top1" in m]


def epoch_metrics(lines: list[dict], file: str) -> dict | None:
    """The metrics line of `best` (largest val top-1, then smallest val loss: the trainer's
    rule) or of `last` (the last line); None for another file or without lines."""
    if not lines or file not in ("best", "last"):
        return None
    if file == "last":
        return lines[-1]
    return max(lines, key=lambda m: (m["val_top1"], -float(m.get("val_loss") or 0.0)))


def is_smoke(run: str, train: dict) -> bool:
    """A smoke run: trained on a `--limit` subset, or named `smoke*`."""
    try:
        limited = int(train.get("limit") or 0) > 0
    except (TypeError, ValueError):
        limited = False
    return limited or run.lower().startswith("smoke")


def _run_facts(run_dir: Path) -> dict:
    """encoder, model, clean, reason, built_at, smoke from `run.json`."""
    doc = _read_json(run_dir / "run.json")
    header = (doc or {}).get("provenance")
    if not isinstance(header, dict):
        return {"reason": None, "clean": None, "smoke": is_smoke(run_dir.name, {})}
    cfg = (header.get("config") or {}).get("model") or {}
    train = (header.get("config") or {}).get("train") or {}
    dirty = bool(header.get("code_dirty")) or str(header.get("code_git_hash", "")).endswith(
        "+dirty"
    )
    head = str(header.get("code_git_hash", ""))[:12]
    return {
        "encoder": cfg.get("encoder"),
        "model": {k: cfg[k] for k in ("d", "d_pair", "layers", "heads") if k in cfg},
        "clean": not dirty,
        "reason": f"STALE_ARTIFACT: trained from a dirty tree (HEAD {head})" if dirty else None,
        "built_at": header.get("built_at"),
        "smoke": is_smoke(run_dir.name, train if isinstance(train, dict) else {}),
    }


def _checkpoint(source: str, run_dir: Path, pt: Path, lines: list[dict], facts: dict):
    st = pt.stat()
    m = epoch_metrics(lines, pt.stem)
    metrics = {k: m.get(k) for k in METRIC_KEYS if k in m} if m else {}
    return Checkpoint(
        source=source,
        run=run_dir.name,
        file=pt.stem,
        path=pt,
        size_bytes=int(st.st_size),
        mtime_ns=int(st.st_mtime_ns),
        epoch=None if m is None else m.get("epoch"),
        metrics=metrics,
        **facts,
    )


def list_checkpoints(settings: ServeSettings) -> list[Checkpoint]:
    """Every `<root>/<source>/<run>/<file>.pt` of `model_roots`, sorted by (source, run,
    file); the first root wins for a (source, run, file) present in both."""
    found: dict[tuple[str, str, str], Checkpoint] = {}
    for root in model_roots(settings):
        for pt in sorted(root.glob("*/*/*.pt")):
            source, run_dir = pt.parent.parent.name, pt.parent
            key = (source, run_dir.name, pt.stem)
            if key in found or not all(NAME_RE.fullmatch(x) for x in key):
                continue
            try:
                found[key] = _checkpoint(
                    source, run_dir, pt, _metrics_lines(run_dir), _run_facts(run_dir)
                )
            except OSError as exc:
                log.warning("skipping %s [CHECKPOINT_UNREADABLE]: %s", pt, exc)
    return [found[k] for k in sorted(found)]


def find_checkpoint(
    settings: ServeSettings, run: str, file: str = "best", source: str | None = None
) -> Checkpoint:
    """The checkpoint of `run` (and `source` when given); CHECKPOINT_NOT_FOUND (404) when
    there is none, REQUEST_INVALID (422) for a malformed name or an ambiguous run."""
    for name, value in (("run", run), ("file", file), ("source", source)):
        if value is not None and not NAME_RE.fullmatch(value):
            raise ServeError(Code.REQUEST_INVALID, f"malformed {name} {value!r}", status=422)
    hits = [
        c
        for c in list_checkpoints(settings)
        if c.run == run and c.file == file and (source is None or c.source == source)
    ]
    if not hits:
        where = ", ".join(str(r) for r in model_roots(settings))
        msg = f"no checkpoint {source or '*'}/{run}/{file}.pt under {where}"
        raise ServeError(Code.CHECKPOINT_NOT_FOUND, msg, status=404)
    if len(hits) > 1:
        sources = sorted(c.source for c in hits)
        msg = f"run {run!r} exists for several sources {sources}: give the source"
        raise ServeError(Code.REQUEST_INVALID, msg, status=422)
    return hits[0]


def _rank_key(c: Checkpoint) -> tuple:
    top1 = c.metrics.get("val_top1")
    top1 = float(top1) if isinstance(top1, int | float) else -1.0
    return (-top1, c.file != "best", c.source, c.run, c.file)


def default_checkpoint(ckpts: list[Checkpoint]) -> Checkpoint | None:
    """The checkpoint a client gets when it names none: the usable non-smoke one with the best
    validation top-1 (best.pt before last.pt on a tie, then by name); a usable smoke run only
    when nothing else is usable; None when no checkpoint is usable."""
    usable = [c for c in ckpts if c.usable]
    real = [c for c in usable if not c.smoke]
    pool = real or usable
    return min(pool, key=_rank_key) if pool else None


def checkpoint_sha256(c: Checkpoint) -> str:
    """sha256 of the checkpoint file, once per (path, mtime, size)."""
    key = (str(c.path), c.mtime_ns, c.size_bytes)
    sha = _SHA.get(key)
    if sha is None:
        sha = file_sha256(c.path)
        _SHA.put(key, sha)
    return sha


def missing_modules() -> list[str]:
    """The modules of `MODULES` that cannot be found (none is imported)."""
    out = []
    for name in MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def embedding_status(settings: ServeSettings) -> tuple[bool, str | None]:
    """The `embedding` capability: torch and the model code importable and one usable local
    checkpoint; else (False, why)."""
    missing = missing_modules()
    if missing:
        return False, f"not importable here: {', '.join(missing)}"
    ckpts = list_checkpoints(settings)
    if not ckpts:
        where = " or ".join(str(r) for r in model_roots(settings))
        return False, f"no checkpoint <source>/<run>/<file>.pt under {where}"
    if not any(c.usable for c in ckpts):
        return False, "every local checkpoint is refused: " + "; ".join(
            f"{c.run}/{c.file}: {c.reason}" for c in ckpts
        )
    return True, None
