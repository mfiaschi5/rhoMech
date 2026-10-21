"""The PMechDB browse index (design 10): species and pool keys of every PMechDB step.

Built ONLY on request: `POST /api/browse/index` (the Browse page's button), `--index-on-start`,
or an export that needs the PMechDB pairs. `run_index` is the `index` job handler (the
JobRunner's background thread, API process): it reads `data/processed/pmechdb/steps.parquet`
(resonance steps included and flagged), sends the steps to the worker in `index_chunk` tasks
of 500 at BACKGROUND priority, then one `index_chunk` task in its `split_badges` form
(`arrows.steps.split_ids` imports torch, so it runs in the worker), and writes
`data/serve/index/pmechdb-<sha12>.parquet` plus a `.json` sidecar. The sha12 is over the
source parquet's bytes and the index and pool-key versions, so a content change makes the
index stale; older index files are removed after a successful build.

A step whose species keys cannot be computed (bare metal records) is skipped with
`SPECIES_KEY_FAILED`, counted and reported through the job context (no silent failures).

`JOBS` tracks the index job of each data root in this process (queued, running, the last
failure), so a second build request returns the running job, and `/browse/status` can say
"building". It lives in memory: a server restart forgets it, as the JobRunner marks every
unfinished job failed(SERVER_RESTART).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ksmech.serve.contracts import JobContext, Runtime
from ksmech.serve.errors import Code, ServeError
from ksmech.serve.keys import POOL_KEY_VERSION
from ksmech.serve.settings import ServeSettings, assert_writable

log = logging.getLogger(__name__)

INDEX_VERSION = 1
CHUNK_STEPS = 500
CHUNK_TIMEOUT_S = 600
BUILD_HINT = "POST /api/browse/index"
QUEUED_STALE_S = 600.0  # a queued index job that never started is forgotten after this
SIDES = ("reactants", "products")
LIST_COLUMNS = (
    "reactant_species",
    "product_species",
    "reactant_token_keys",
    "product_token_keys",
    "flags",
)
STR_COLUMNS = (
    "step_id",
    "reactants_smiles",
    "products_smiles",
    "reactant_pool_key",
    "product_pool_key",
    "arrows_json",
)
NULLABLE_COLUMNS = ("pmechdb_code", "split_random", "split_scaffold")


def index_schema():
    import pyarrow as pa

    cols = [(c, pa.string()) for c in STR_COLUMNS[:3]]
    cols += [("pmechdb_code", pa.string()), ("n_arrows", pa.int32())]
    cols += [("resonance_step", pa.bool_())]
    cols += [(c, pa.string()) for c in STR_COLUMNS[3:]]
    cols += [(c, pa.list_(pa.string())) for c in LIST_COLUMNS]
    cols += [("split_random", pa.string()), ("split_scaffold", pa.string())]
    return pa.schema(cols)


# ---- paths and source digests

_DIGESTS: dict[tuple[str, int, int], str] = {}
_DIGEST_LOCK = threading.Lock()


def source_path(s: ServeSettings) -> Path:
    """`<data_root>/data/processed/pmechdb/steps.parquet` (read-only)."""
    return s.processed / "pmechdb" / "steps.parquet"


def _source_sha12(steps_path: Path) -> str:
    st = Path(steps_path).stat()
    key = (str(Path(steps_path).resolve()), st.st_size, st.st_mtime_ns)
    with _DIGEST_LOCK:
        if key in _DIGESTS:
            return _DIGESTS[key]
    h = hashlib.sha256(f"pmindex-v{INDEX_VERSION}|pool-key-v{POOL_KEY_VERSION}|".encode())
    with open(steps_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    sha = h.hexdigest()[:12]
    with _DIGEST_LOCK:
        _DIGESTS[key] = sha
    return sha


def index_path(s: ServeSettings, steps_path: Path) -> Path:
    """`data/serve/index/pmechdb-<sha12>.parquet` of the given source parquet."""
    return s.index / f"pmechdb-{_source_sha12(steps_path)}.parquet"


def index_key(s: ServeSettings) -> str:
    return str(s.index.resolve())


# ---- the in-process registry of index jobs


@dataclass
class _Entry:
    job_id: str
    state: str  # queued | running
    created: dict  # JobCreatedOut fields
    since: float = field(default_factory=time.monotonic)
    message: str = ""


class IndexJobs:
    """Queued / running index jobs and the last failure, per index directory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, _Entry] = {}
        self._seen: set[str] = set()
        self._failures: dict[str, dict] = {}

    def note_queued(self, key: str, created: dict) -> None:
        """A submitted job (ignored when it already started, e.g. in inline mode)."""
        job_id = str(created["job_id"])
        with self._lock:
            if job_id not in self._seen and key not in self._active:
                self._active[key] = _Entry(job_id, "queued", dict(created))

    def started(self, key: str, job_id: str, kind: str = "index") -> None:
        created = {"job_id": job_id, "kind": kind, "status": "running"}
        created |= {"status_url": f"/api/jobs/{job_id}", "parent_job_id": None}
        with self._lock:
            self._seen.add(job_id)
            old = self._active.get(key)
            if old is not None and old.job_id == job_id:
                created = old.created | {"status": "running"}
            self._active[key] = _Entry(job_id, "running", created)

    def progress(self, key: str, job_id: str, message: str) -> None:
        with self._lock:
            entry = self._active.get(key)
            if entry is not None and entry.job_id == job_id:
                entry.message = message

    def finished(self, key: str, job_id: str, failure: dict | None = None) -> None:
        with self._lock:
            entry = self._active.get(key)
            if entry is not None and entry.job_id == job_id:
                del self._active[key]
            if failure is None:
                self._failures.pop(key, None)
            else:
                self._failures[key] = {"job_id": job_id, **failure}

    def active(self, key: str) -> _Entry | None:
        with self._lock:
            entry = self._active.get(key)
            stale = entry is not None and entry.state == "queued"
            if stale and time.monotonic() - entry.since > QUEUED_STALE_S:
                log.warning("index job %s never started; forgotten", entry.job_id)
                del self._active[key]
                return None
            return entry

    def last_failure(self, key: str) -> dict | None:
        with self._lock:
            return self._failures.get(key)


JOBS = IndexJobs()


# ---- the loaded index


@dataclass
class PMIndex:
    """A loaded index: rows sorted by step_id and the inverted maps the search uses."""

    path: Path
    rows: list[dict]
    meta: dict
    by_step: dict[str, int] = field(default_factory=dict)
    by_species: dict[str, dict[str, set[int]]] = field(default_factory=dict)  # side -> key
    by_token: dict[str, dict[str, set[int]]] = field(default_factory=dict)
    forms: dict[str, set[str]] = field(default_factory=dict)  # species key -> token keys

    @property
    def n_steps(self) -> int:
        return len(self.rows)

    @property
    def built_at(self) -> str | None:
        return self.meta.get("built_at")

    def pairs(self) -> dict[tuple[str, str], str]:
        """(reactant pool key, product pool key) -> the first PMechDB step_id with them."""
        out: dict[tuple[str, str], str] = {}
        for row in self.rows:
            out.setdefault((row["reactant_pool_key"], row["product_pool_key"]), row["step_id"])
        return out


def side_columns(side: str) -> tuple[str, str]:
    """(species column, token-key column) of a side."""
    prefix = "reactant" if side == "reactants" else "product"
    return f"{prefix}_species", f"{prefix}_token_keys"


def _build_maps(index: PMIndex) -> PMIndex:
    by_species: dict[str, dict[str, set[int]]] = {side: defaultdict(set) for side in SIDES}
    by_token: dict[str, dict[str, set[int]]] = {side: defaultdict(set) for side in SIDES}
    forms: dict[str, set[str]] = defaultdict(set)
    for pos, row in enumerate(index.rows):
        index.by_step[row["step_id"]] = pos
        for side in SIDES:
            sp_col, tok_col = side_columns(side)
            for species, token in zip(row[sp_col], row[tok_col], strict=True):
                by_species[side][species].add(pos)
                by_token[side][token].add(pos)
                forms[species].add(token)
    index.by_species, index.by_token, index.forms = by_species, by_token, dict(forms)
    return index


def load_index(path: Path) -> PMIndex | None:
    """The index at `path`, or None when it is missing or unreadable (logged)."""
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_file():
        return None
    try:
        rows = pq.read_table(path).to_pylist()
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("unreadable PMechDB index %s: %s", path, exc)
        return None
    rows.sort(key=lambda r: r["step_id"])
    return _build_maps(PMIndex(path, rows, meta))


# ---- status


def _meta(path: Path) -> dict:
    try:
        return json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _failure_note(s: ServeSettings) -> str:
    failure = JOBS.last_failure(index_key(s))
    if failure is None:
        return ""
    return f"; the last build ({failure['job_id']}) failed: {failure.get('code')}"


def source_status(s: ServeSettings) -> dict:
    """`SourceIndexStatusOut` fields of the PMechDB index."""
    src = source_path(s)
    entry = JOBS.active(index_key(s))
    job_id = None if entry is None else entry.job_id
    out = {"state": "unavailable", "n_steps": 0, "built_at": None, "job_id": job_id}
    if not src.is_file():
        return out | {"reason": f"no PMechDB steps at {src}"}
    path = index_path(s, src)
    if path.is_file():
        meta = _meta(path)
        ready = {"state": "ready", "n_steps": int(meta.get("n_indexed", 0)), "reason": None}
        return out | ready | {"built_at": meta.get("built_at")}
    if entry is not None:
        msg = entry.message or f"index job {entry.job_id} is {entry.state}"
        return out | {"state": "building", "reason": msg}
    older = sorted(s.index.glob("pmechdb-*.parquet")) if s.index.is_dir() else []
    if older:
        meta = _meta(older[-1])
        reason = "the PMechDB steps changed since the index was built; rebuild with " + BUILD_HINT
        return out | {"state": "stale", "built_at": meta.get("built_at"), "reason": reason}
    reason = f"the index is built only on request: {BUILD_HINT}" + _failure_note(s)
    return out | {"state": "missing", "reason": reason}


def not_ready(status: dict) -> ServeError:
    """INDEX_NOT_READY (409) with detail {state, job_id, build}."""
    msg = f"the PMechDB index is {status['state']}: {status.get('reason') or ''}".rstrip(": ")
    detail = {"state": status["state"], "job_id": status.get("job_id"), "build": BUILD_HINT}
    return ServeError(Code.INDEX_NOT_READY, msg, status=409, detail=detail)


# ---- worker task


def index_row(raw: dict) -> dict:
    """One index row from a step dict {step_id, reactants_smiles, products_smiles, arrows,
    resonance_step}. Raises ServeError(SPECIES_KEY_FAILED)."""
    from ksmech.data.pmechdb import format_arrow_code
    from ksmech.data.schema import Arrow
    from ksmech.serve.keys import side_keys

    r, p = side_keys(raw["reactants_smiles"]), side_keys(raw["products_smiles"])
    flags = sorted((set(r.flags) | set(p.flags)) - {"OK"})
    try:
        code = format_arrow_code([Arrow.from_dict(a) for a in raw["arrows"]])
    except (ValueError, KeyError, TypeError):
        code = None
        flags.append("NO_ARROW_CODE")
    return {
        "step_id": str(raw["step_id"]),
        "reactants_smiles": raw["reactants_smiles"],
        "products_smiles": raw["products_smiles"],
        "pmechdb_code": code,
        "n_arrows": len(raw["arrows"]),
        "resonance_step": bool(raw.get("resonance_step", False)),
        "reactant_pool_key": r.pool_key,
        "product_pool_key": p.pool_key,
        "arrows_json": json.dumps(raw["arrows"], sort_keys=True),
        "reactant_species": r.species_keys,
        "product_species": p.species_keys,
        "reactant_token_keys": r.token_keys,
        "product_token_keys": p.token_keys,
        "flags": flags,
    }


def split_badges(processed_dir: Path) -> dict[str, dict[str, str]]:
    """step_id -> {"random": part, "scaffold": part} from `arrows.steps.split_ids` over the
    training steps (seed 0); resonance steps read "resonance". Worker only (torch)."""
    from ksmech.arrows.steps import split_ids
    from ksmech.data.io import read_steps, read_training_steps, steps_path

    path = steps_path("pmechdb", Path(processed_dir))
    badges = {
        s.step_id: {"random": "resonance", "scaffold": "resonance"}
        for s in read_steps(path)
        if s.resonance_step
    }
    training = read_training_steps(path)
    for strategy in ("random", "scaffold"):
        for split, ids in split_ids(training, strategy).items():
            part = split.rsplit("_", 1)[1]
            for step_id in ids:
                badges.setdefault(step_id, {})[strategy] = part
    return badges


def t_index_chunk(rt: Runtime, payload: dict) -> dict:
    """{"rows": [step dicts]} -> {"rows", "skipped": {code: n}, "skipped_rows"}; or
    {"split_badges": {"processed_dir"}} -> {"badges": {step_id: {"random", "scaffold"}}}."""
    if "split_badges" in payload:
        return {"badges": split_badges(Path(payload["split_badges"]["processed_dir"]))}
    rows, skipped_rows = [], []
    for raw in payload.get("rows", []):
        try:
            rows.append(index_row(raw))
        except ServeError as exc:
            skip = {"step_id": raw.get("step_id"), "code": str(exc.code), "detail": exc.message}
            skipped_rows.append(skip)
    skipped = Counter(r["code"] for r in skipped_rows)
    return {"rows": rows, "skipped": dict(skipped), "skipped_rows": skipped_rows}


# ---- the job handler


def _check_cancel(ctx: JobContext) -> None:
    if not ctx.cancelled():
        return
    try:
        from ksmech.serve.jobs import Cancelled
    except ImportError:
        raise ServeError(Code.CANCELLED, f"index job {ctx.job_id} cancelled") from None
    raise Cancelled(ctx.job_id)


def _source_rows(src: Path) -> list[dict]:
    from ksmech.data.io import read_steps

    steps = sorted(read_steps(src), key=lambda st: st.step_id)
    return [
        {
            "step_id": st.step_id,
            "reactants_smiles": st.reactants_smiles,
            "products_smiles": st.products_smiles,
            "arrows": [a.to_dict() for a in st.arrows],
            "resonance_step": bool(st.resonance_step),
        }
        for st in steps
    ]


def _run_chunks(ctx: JobContext, rows: list[dict], key: str) -> tuple[list[dict], Counter]:
    from ksmech.serve.worker import Priority

    out: list[dict] = []
    skipped: Counter = Counter()
    n = len(rows)
    for start in range(0, n, CHUNK_STEPS):
        _check_cancel(ctx)
        chunk = rows[start : start + CHUNK_STEPS]
        label = f"PMechDB index: steps {start + 1}-{start + len(chunk)} of {n}"
        res = ctx.scheduler.run(
            "index_chunk",
            {"rows": chunk},
            priority=Priority.BACKGROUND,
            timeout_s=CHUNK_TIMEOUT_S,
            owner=ctx.job_id,
            label=label,
        )
        if not res.ok:
            raise ServeError.from_task(res)
        out += res.result["rows"]
        for item in res.result.get("skipped_rows", []):
            skipped[item["code"]] += 1
            ctx.skip(item["code"], item["detail"], step_id=item["step_id"], stage="index")
        done = start + len(chunk)
        message = f"indexing {done}/{n}"
        JOBS.progress(key, ctx.job_id, message)
        ctx.progress(phase="index", done=done, total=n, message=message)
    return out, skipped


def _badges(ctx: JobContext) -> dict | None:
    from ksmech.serve.worker import Priority

    _check_cancel(ctx)
    payload = {"split_badges": {"processed_dir": str(ctx.settings.processed)}}
    res = ctx.scheduler.run(
        "index_chunk",
        payload,
        priority=Priority.BACKGROUND,
        timeout_s=CHUNK_TIMEOUT_S,
        owner=ctx.job_id,
        label="PMechDB index: split badges",
    )
    if not res.ok:
        detail = f"split badges unavailable: {res.detail}"
        ctx.skip(res.code or str(Code.UNEXPECTED), detail, stage="split_badges")
        return None
    return res.result["badges"]


def _write(s: ServeSettings, path: Path, rows: list[dict], meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ksmech.serve.pools import atomic_write_text

    assert_writable(path, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=index_schema()), tmp)
    os.replace(tmp, path)
    atomic_write_text(path.with_suffix(".json"), json.dumps(meta, indent=1, sort_keys=True), s)
    for old in path.parent.glob("pmechdb-*.parquet"):
        if old != path:
            for stale in (old, old.with_suffix(".json")):
                stale.unlink(missing_ok=True)


def _build(ctx: JobContext, src: Path, path: Path, key: str) -> dict:
    rows_in = _source_rows(src)
    rows, skipped = _run_chunks(ctx, rows_in, key)
    badges = _badges(ctx)
    for row in rows:
        badge = (badges or {}).get(row["step_id"], {})
        row["split_random"], row["split_scaffold"] = badge.get("random"), badge.get("scaffold")
    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "source": str(src),
        "source_sha12": path.stem.split("-", 1)[1],
        "n_steps": len(rows_in),
        "n_indexed": len(rows),
        "skipped": dict(skipped),
        "split_badges": badges is not None,
        "index_version": INDEX_VERSION,
        "pool_key_version": POOL_KEY_VERSION,
        "job_id": ctx.job_id,
    }
    _write(ctx.settings, path, rows, meta)
    log.info("PMechDB index %s: %d steps, skipped %s", path.name, len(rows), dict(skipped))
    return meta


def run_index(ctx: JobContext, request: dict, *, engine: Any = None) -> dict:
    """The `index` job handler: build the PMechDB index unless it is current (or
    `request["force"]`); returns `IndexStatusOut` fields. `curated_steps` is counted only when
    a database `engine` is given (GET /browse/status always counts it)."""
    s = ctx.settings
    src = source_path(s)
    if not src.is_file():
        raise not_ready(source_status(s))
    path, key = index_path(s, src), index_key(s)
    JOBS.started(key, ctx.job_id, str(request.get("job_kind", "index")))
    try:
        if not path.is_file() or request.get("force"):
            _build(ctx, src, path, key)
        else:
            ctx.progress(phase="index", done=1, total=1, message="the PMechDB index is current")
    except BaseException as exc:
        failure = {"code": str(getattr(exc, "code", Code.UNEXPECTED)), "detail": str(exc)}
        JOBS.finished(key, ctx.job_id, failure)
        raise
    JOBS.finished(key, ctx.job_id)
    return {"pmechdb": source_status(s), "curated_steps": _curated_count(engine)}


def _curated_count(engine: Any) -> int:
    if engine is None:
        return 0
    from sqlalchemy import func
    from sqlmodel import select

    from ksmech.serve import db

    with db.session(engine) as ses:
        return int(ses.exec(select(func.count()).select_from(db.CuratedStep)).one())
