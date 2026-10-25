"""Settings of the serve app: every path absolute, resolved once from the environment.

`ServeSettings` holds only paths and primitives, so it pickles into the spawned compute worker
(`TaskEnv` is what every task receives). The shared caches (`data/cache/qm`, `data/cache/tokens`
under `shared_cache_root`, default the data root) are read-only: serve writes only under
`data/serve/` and `data/curation/`, and `assert_writable` refuses any write that resolves into a
shared cache or the live PMechDB data.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from ksmech.serve.contracts import Limits
from ksmech.serve.versions import oriented_fingerprint

log = logging.getLogger(__name__)

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CHEMISTRIES = ("real", "cached", "synthetic")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")
_FAILURE_LOCK = threading.Lock()


def _repo_root() -> Path:
    import ksmech

    return Path(ksmech.__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServeSettings:
    data_root: Path  # KSMECH_DATA_ROOT, default Path.cwd(), resolved
    repo_root: Path  # Path(ksmech.__file__).resolve().parents[2]
    chemistry: str = "real"  # real | cached | synthetic (KSMECH_SERVE_CHEMISTRY)
    workers: int = 1  # KSMECH_SERVE_WORKERS, clamped [1, 2]
    gbdt_path: Path | None = None  # KSMECH_SERVE_GBDT; default processed/pmechdb/baseline_...
    enable_stub_predictor: bool = False  # KSMECH_SERVE_STUB_PREDICTOR=1
    shared_cache_root: Path | None = None  # KSMECH_SHARED_CACHE_ROOT; default data_root
    auto_index: bool = False  # KSMECH_SERVE_AUTO_INDEX=1 or --index-on-start
    interactive_wait_s: float = 15.0
    qm_timeout_s: int = 900
    max_species: int = 6
    max_count: int = 4
    max_atoms: int = 150
    max_beam: int = 10
    max_steps: int = 12
    max_nodes: int = 200
    frontend_dist: Path | None = None  # KSMECH_SERVE_DIST; default repo/apps/ui/dist
    dev: bool = False

    # ---- shared, read-only
    @property
    def shared_root(self) -> Path:
        return self.shared_cache_root or self.data_root

    @property
    def shared_qm(self) -> Path:
        return self.shared_root / "data" / "cache" / "qm"

    @property
    def shared_tokens(self) -> Path:
        return self.shared_root / "data" / "cache" / "tokens"

    @property
    def processed(self) -> Path:
        return self.data_root / "data" / "processed"

    # ---- serve-owned
    @property
    def serve(self) -> Path:
        return self.data_root / "data" / "serve"

    @property
    def serve_qm(self) -> Path:
        return self.serve / "qm"

    @property
    def serve_tokens(self) -> Path:
        return self.serve / "tokens"

    @property
    def oriented_dir(self) -> Path:
        """Oriented-block cache, namespaced by the oriented fingerprint (never data/cache)."""
        return self.serve / "oriented" / oriented_fingerprint()

    @property
    def pools(self) -> Path:
        return self.serve / "pools"

    @property
    def contact(self) -> Path:
        return self.serve / "contact"

    @property
    def cubes(self) -> Path:
        return self.serve / "cubes"

    @property
    def jobs(self) -> Path:
        return self.serve / "jobs"

    @property
    def index(self) -> Path:
        return self.serve / "index"

    @property
    def logs(self) -> Path:
        return self.serve / "logs"

    @property
    def run(self) -> Path:
        return self.serve / "run"

    @property
    def failures_log(self) -> Path:
        return self.logs / "failures.jsonl"

    # ---- curation, models, frontend, spec
    @property
    def curation_db(self) -> Path:
        return self.data_root / "data" / "curation" / "curation.sqlite"

    @property
    def export_raw(self) -> Path:
        return self.data_root / "data" / "curation" / "export_raw" / "steps.parquet"

    @property
    def gbdt_file(self) -> Path:
        return self.gbdt_path or self.processed / "pmechdb" / "baseline_gbdt_random.npz"

    @property
    def dist(self) -> Path:
        return self.frontend_dist or self.repo_root / "apps" / "ui" / "dist"

    @property
    def spec_path(self) -> Path:
        return self.repo_root / "docs" / "ksmech_architecture.tex"

    @property
    def limits(self) -> Limits:
        return Limits(
            self.max_species,
            self.max_count,
            self.max_atoms,
            self.max_beam,
            self.max_steps,
            self.max_nodes,
        )


@dataclass(frozen=True)
class TaskEnv:
    """What every worker task receives (picklable)."""

    settings: ServeSettings
    enable_diag_tasks: bool = False


def parse_bool(value: str, name: str = "value") -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name}: expected 1/0 or true/false, got {value!r}")


_ENV = {
    "KSMECH_DATA_ROOT": ("data_root", Path),
    "KSMECH_SERVE_CHEMISTRY": ("chemistry", str),
    "KSMECH_SERVE_WORKERS": ("workers", int),
    "KSMECH_SERVE_GBDT": ("gbdt_path", Path),
    "KSMECH_SERVE_STUB_PREDICTOR": ("enable_stub_predictor", bool),
    "KSMECH_SHARED_CACHE_ROOT": ("shared_cache_root", Path),
    "KSMECH_SERVE_AUTO_INDEX": ("auto_index", bool),
    "KSMECH_SERVE_DIST": ("frontend_dist", Path),
}
_PATH_FIELDS = ("data_root", "repo_root", "gbdt_path", "shared_cache_root", "frontend_dist")


def _from_env(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for var, (name, kind) in _ENV.items():
        raw = env.get(var)
        if raw is None or (kind is not bool and not raw.strip()):
            continue
        if kind is bool:
            out[name] = parse_bool(raw, var)
        elif kind is int:
            try:
                out[name] = int(raw)
            except ValueError as exc:
                raise ValueError(f"{var}: expected an integer, got {raw!r}") from exc
        else:
            out[name] = kind(raw)
    return out


def load_settings(env: Mapping[str, str] = os.environ, **overrides: Any) -> ServeSettings:
    """Settings from the environment variables above, then keyword overrides (tests call
    `load_settings({}, data_root=tmp, chemistry="synthetic", enable_stub_predictor=True)`);
    every path resolved to an absolute path."""
    known = {f.name for f in fields(ServeSettings)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"unknown ServeSettings fields: {sorted(unknown)}")
    values: dict[str, Any] = {"data_root": Path.cwd(), "repo_root": _repo_root()}
    values |= _from_env(env)
    values |= overrides
    for name in _PATH_FIELDS:
        if values.get(name) is not None:
            values[name] = Path(values[name]).expanduser().resolve()
    if values.get("chemistry", "real") not in CHEMISTRIES:
        raise ValueError(f"chemistry must be one of {CHEMISTRIES}, got {values['chemistry']!r}")
    values["workers"] = min(2, max(1, int(values.get("workers", 1))))
    if values.get("shared_cache_root") is None:
        values["shared_cache_root"] = values["data_root"]
    s = ServeSettings(**values)
    return replace(s, gbdt_path=s.gbdt_file, frontend_dist=s.dist)


def pin_threads() -> None:
    """One BLAS / OpenMP thread (overwrites any value), and one torch thread once torch is
    imported. Idempotent; call it before numpy-heavy work and after lazy torch imports."""
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def _forbidden_roots(s: ServeSettings) -> list[Path]:
    roots = []
    for base in {s.data_root, s.shared_root}:
        roots += [base / "data" / "cache", base / "data" / "processed" / "pmechdb"]
        roots += [base / "data" / "raw"]
    return [r.resolve() for r in roots]


def assert_writable(path: Path, s: ServeSettings) -> None:
    """Raise ValueError if `path`, after resolving symlinks, lies under a shared cache
    (`data/cache`), the live PMechDB directory (`data/processed/pmechdb`) or `data/raw`, of
    either the data root or the shared cache root."""
    target = Path(path).expanduser().resolve()
    for root in _forbidden_roots(s):
        if target == root or target.is_relative_to(root):
            raise ValueError(f"refusing to write {target}: it lies under the read-only {root}")


def append_failure(s: ServeSettings, code: str, detail: str = "", **where: Any) -> None:
    """Append one reason-coded record to `data/serve/logs/failures.jsonl` (no silent failures).
    Never raises: a log that cannot be written is reported through logging."""
    record = {"time": time.time(), "code": str(code), "detail": detail, **where}
    try:
        assert_writable(s.failures_log, s)
        s.logs.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with _FAILURE_LOCK, open(s.failures_log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError) as exc:
        log.error("cannot append to %s: %s", s.failures_log, exc)
