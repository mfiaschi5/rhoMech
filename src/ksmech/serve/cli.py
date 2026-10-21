"""`ksmech-serve`: the local web app (design 12).

    ksmech-serve [run] --port 8000 [--chemistry real|cached|synthetic] [--dev] [--no-ui]
                       [--index-on-start]
    ksmech-serve recheck [--stale-only] [--predictor P] [--mechanism ID ...]
    ksmech-serve export
    ksmech-serve schema-ts [--out apps/ui/src/api/types.gen.ts]
    ksmech-serve probe-cache --smiles S [--smiles S ...]

The server binds only to 127.0.0.1 (no `--host`; there is no authentication: reach it through
`ssh -L 8000:localhost:8000 <server>`). A first token that is not a subcommand gets `run`
inserted, so `ksmech-serve --port 8000` works. `run`, `recheck` and `export` hold
`data/serve/run/server.lock` (flock), so at most one serve worker pool exists per data root;
`recheck` and `export` refuse with exit 2 while a server runs.

This module imports only the standard library at module level: `main` pins the BLAS / OpenMP
threads to 1 before numpy (or anything that imports it) is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ksmech.serve.contracts import LocalJobContext
    from ksmech.serve.settings import ServeSettings

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
COMMANDS = ("run", "recheck", "export", "schema-ts", "probe-cache")
CHEMISTRY_CHOICES = ("real", "cached", "synthetic")
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
LOCK_HELD = {
    "run": "another ksmech-serve already runs on this data root ({path})",
    "recheck": "server running: use the web UI or POST /api/curate/recheck ({path})",
    "export": "server running: use the web UI or POST /api/curate/export ({path})",
}
_HELD_LOCKS: list[Any] = []  # open lock files, kept for the life of the process


def _pin_env() -> None:
    """One BLAS / OpenMP thread, set before numpy loads (overwrites any value)."""
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))


# ---- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-serve", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="serve the web app on 127.0.0.1 (the default command)")
    run.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port on 127.0.0.1")
    run.add_argument(
        "--chemistry",
        choices=CHEMISTRY_CHOICES,
        default=None,
        help="real (xTB on a cache miss), cached (caches only) or synthetic (EHT)",
    )
    run.add_argument("--dev", action="store_true", help="no static files: use Vite on :5173")
    run.add_argument("--no-ui", action="store_true", help="serve the API only")
    run.add_argument(
        "--index-on-start",
        action="store_true",
        help="build the PMechDB browse index at startup (about 10 min of worker)",
    )
    rc = sub.add_parser("recheck", help="re-check curated mechanisms (server must be stopped)")
    rc.add_argument("--stale-only", action="store_true", help="only stale mechanisms")
    rc.add_argument("--predictor", default=None, help="predictor (default: each mechanism's)")
    rc.add_argument("--mechanism", type=int, nargs="+", default=None, metavar="ID")
    sub.add_parser("export", help="export curated steps to data/processed/user_curated")
    ts = sub.add_parser("schema-ts", help="write the TypeScript mirror of the API models")
    ts.add_argument("--out", type=Path, default=None, help="default apps/ui/src/api/types.gen.ts")
    pc = sub.add_parser(
        "probe-cache",
        help="exit 0 when every species loads from the shared "
        "token cache (KSMECH_SHARED_CACHE_ROOT, real method)",
    )
    pc.add_argument("--smiles", action="append", required=True, help="a SMILES (repeatable)")
    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    """Insert `run` when the first token is not a subcommand (`-h` keeps the top help)."""
    if argv and (argv[0] in COMMANDS or argv[0] in ("-h", "--help")):
        return list(argv)
    return ["run", *argv]


# ---- the server lock


def lock_path(settings: ServeSettings) -> Path:
    return settings.run / "server.lock"


def acquire_server_lock(settings: ServeSettings):
    """The open, flock-ed (LOCK_EX | LOCK_NB) `server.lock`, or None when another process
    holds it. The caller keeps the returned file open for as long as it needs the lock."""
    from ksmech.serve.settings import assert_writable

    path = lock_path(settings)
    assert_writable(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return None
        raise
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def _take_lock(settings: ServeSettings, command: str) -> bool:
    fh = acquire_server_lock(settings)
    if fh is None:
        print(LOCK_HELD[command].format(path=lock_path(settings)), file=sys.stderr)
        return False
    _HELD_LOCKS.append(fh)
    return True


def release_locks() -> None:
    while _HELD_LOCKS:
        fh = _HELD_LOCKS.pop()
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()


# ---- run


def port_in_use(port: int, host: str = HOST) -> bool:
    """True when `host:port` cannot be bound (uvicorn binds with SO_REUSEADDR too)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, int(port)))
        except OSError:
            return True
    return False


class SettingsError(Exception):
    """The environment or the options give invalid settings (exit 2)."""


def _settings(**overrides: Any) -> ServeSettings:
    from ksmech.serve.settings import load_settings

    try:
        return load_settings(**{k: v for k, v in overrides.items() if v is not None})
    except ValueError as exc:
        raise SettingsError(str(exc)) from exc


def cmd_run(args: argparse.Namespace) -> int:
    import uvicorn

    from ksmech.serve.api import create_app

    if not 0 < args.port < 65536:
        print(f"--port must be in 1..65535, got {args.port}", file=sys.stderr)
        return 2
    settings = _settings(
        chemistry=args.chemistry, dev=args.dev or None, auto_index=args.index_on_start or None
    )
    if not _take_lock(settings, "run"):
        return 2
    if port_in_use(args.port):
        print(
            f"port {args.port} is in use on {HOST}: stop the other server or pass --port",
            file=sys.stderr,
        )
        return 2
    app = create_app(settings, serve_ui=not args.no_ui)
    print(
        f"ksmech-serve on http://{HOST}:{args.port} (chemistry {settings.chemistry}, data root "
        f"{settings.data_root}); from your PC: ssh -L {args.port}:localhost:{args.port} "
        "<server>",
        flush=True,
    )
    try:
        uvicorn.run(app, host=HOST, port=args.port, workers=1, reload=False, log_level="info")
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"uvicorn could not serve {HOST}:{args.port} (exit {exc.code})", file=sys.stderr)
            return 2
    return 0


# ---- recheck and export (offline batch jobs through a one-worker ProcessScheduler)


def _job_context(settings: ServeSettings, scheduler, name: str) -> LocalJobContext:
    from ksmech.serve.contracts import LocalJobContext
    from ksmech.serve.settings import append_failure

    class CliJobContext(LocalJobContext):
        """LocalJobContext that also appends every skip to failures.jsonl."""

        def skip(self, code: str, detail: str = "", **where: Any) -> None:
            super().skip(code, detail, **where)
            append_failure(self.settings, code, detail, job_id=self.job_id, **where)

        def progress(self, **fields: Any) -> None:
            super().progress(**fields)
            if fields.get("message"):
                log.info("%s: %s", fields.get("phase", self.job_id), fields["message"])

    return CliJobContext(f"cli-{name}-{int(time.time())}", settings, scheduler)


def _offline(settings: ServeSettings, name: str, body) -> int:
    """Run `body(ctx, job_ctx, engine)` under the server lock with a one-worker
    ProcessScheduler; print the result JSON and the skip counts."""
    from ksmech.serve import db
    from ksmech.serve.context import build_context
    from ksmech.serve.procsched import ProcessScheduler
    from ksmech.serve.settings import TaskEnv
    from ksmech.serve.worker import Scheduler

    if not _take_lock(settings, name):
        return 2
    scheduler: Scheduler = ProcessScheduler(TaskEnv(settings), 1)
    try:
        ctx = build_context(settings, scheduler=scheduler)
        engine = db.make_engine(settings.curation_db)
        db.init_db(engine)
        job = _job_context(settings, scheduler, name)
        result = body(ctx, job, engine)
    finally:
        scheduler.close()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if job.skipped:
        print(f"skipped: {dict(sorted(job.skipped.items()))}", file=sys.stderr)
    return 0


def cmd_recheck(args: argparse.Namespace) -> int:
    from ksmech.serve.recheck import run_recheck

    request = {
        "mechanism_ids": args.mechanism,
        "predictor": args.predictor,
        "stale_only": bool(args.stale_only),
    }

    def body(ctx, job, engine) -> dict:
        return run_recheck(job, request, engine=engine, registry=ctx.registry)

    return _offline(_settings(), "recheck", body)


def _no_submit(kind: str, request: dict) -> dict:
    from ksmech.serve.errors import Code, ServeError

    raise ServeError(Code.REQUEST_INVALID, f"no job runner in the CLI (job {kind})")


def cmd_export(args: argparse.Namespace) -> int:
    from ksmech.serve.browse import BrowseService
    from ksmech.serve.export import run_export

    def body(ctx, job, engine) -> dict:
        browse = BrowseService(ctx, engine, submit=_no_submit)
        return run_export(job, {}, engine=engine, pairs=browse.pmechdb_pairs)

    return _offline(_settings(), "export", body)


# ---- schema-ts and probe-cache


def cmd_schema_ts(args: argparse.Namespace) -> int:
    from ksmech.serve.schemas import write_typescript

    path = write_typescript(args.out)
    print(f"wrote {path}")
    return 0


def probe_keys(smiles: list[str]) -> tuple[list[str], list[str]]:
    """(every fragment token key, the invalid inputs) of the SMILES list."""
    from ksmech.serve.errors import ServeError
    from ksmech.serve.keys import side_keys

    keys, invalid = [], []
    for s in smiles:
        try:
            keys += side_keys(s).token_keys
        except ServeError as exc:
            invalid.append(f"{s}: {exc.message}")
    return list(dict.fromkeys(keys)), invalid


def cmd_probe_cache(args: argparse.Namespace) -> int:
    """File reads only: no lock, no worker, nothing written."""
    from ksmech.qm.xtb_calc import QMConfig
    from ksmech.serve.loader import LayeredTokenLoader

    settings = _settings()
    keys, invalid = probe_keys(list(args.smiles))
    loader = LayeredTokenLoader([settings.shared_tokens], QMConfig().method)
    missing = [k for k in keys if loader(k) is None]
    for line in invalid:
        print(f"invalid: {line}")
    for key in missing:
        print(f"missing: {key}")
    found = len(keys) - len(missing)
    print(f"probe-cache: {found}/{len(keys)} token keys load from {settings.shared_tokens}")
    return 0 if keys and not missing and not invalid else 1


HANDLERS = {
    "run": cmd_run,
    "recheck": cmd_recheck,
    "export": cmd_export,
    "schema-ts": cmd_schema_ts,
    "probe-cache": cmd_probe_cache,
}


def main(argv: list[str] | None = None) -> int:
    _pin_env()
    from ksmech.serve.settings import pin_threads

    pin_threads()
    args = build_parser().parse_args(normalize_argv(sys.argv[1:] if argv is None else argv))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return HANDLERS[args.command](args)
    except SettingsError as exc:
        print(f"ksmech-serve {args.command}: {exc}", file=sys.stderr)
        return 2
    finally:
        release_locks()
