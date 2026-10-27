"""`ksmech-serve`: the local web app (design 12).

    ksmech-serve [run] --port 8000 [--host 127.0.0.1|0.0.0.0|<address>]
                       [--chemistry real|cached|synthetic] [--dev] [--no-ui] [--index-on-start]
    ksmech-serve recheck [--stale-only] [--predictor P] [--mechanism ID ...]
    ksmech-serve export --out FILE [--status validated corrected]
    ksmech-serve schema-ts [--out apps/ui/src/api/types.gen.ts]
    ksmech-serve probe-cache --smiles S [--smiles S ...]
    ksmech-serve gallery [--out DIR] [--only NAME ...] [--derive-templates]

There is no authentication. By default the server binds 127.0.0.1 (reach it through
`ssh -L 8000:localhost:8000 <server>`); `--host 0.0.0.0` (or one of this machine's addresses)
opens it to the local network, and the Host / Origin checks then also accept this machine's
addresses and host names (`lan_hosts`). A first token that is not a subcommand gets `run`
inserted, so `ksmech-serve --port 8000` works. `run` and `recheck` hold
`data/serve/run/server.lock` (flock), so at most one serve worker pool exists per data root;
`recheck` refuses with exit 2 while a server runs. `export` only reads the curation database
and writes the PMechDB-format file (a download, never training data).

This module imports only the standard library at module level: `main` pins the BLAS / OpenMP
threads to 1 before numpy (or anything that imports it) is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import ipaddress
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ksmech.serve.core.contracts import LocalJobContext
    from ksmech.serve.core.settings import ServeSettings

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
COMMANDS = ("run", "recheck", "export", "schema-ts", "probe-cache", "gallery")
CHEMISTRY_CHOICES = ("real", "cached", "synthetic")
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
LOCK_HELD = {
    "run": "another ksmech-serve already runs on this data root ({path})",
    "recheck": "server running: use the web UI or POST /api/curate/recheck ({path})",
    "export": "server running: use the web UI or GET /api/curate/export/pmechdb ({path})",
}
_HELD_LOCKS: list[Any] = []  # open lock files, kept for the life of the process


def _pin_env() -> None:
    """One BLAS / OpenMP thread, set before numpy loads (overwrites any value)."""
    os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))


# ---- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksmech-serve", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="serve the web app (the default command)")
    run.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    run.add_argument(
        "--host",
        default=HOST,
        help="IPv4 address to bind: 127.0.0.1 (default; SSH tunnel) or 0.0.0.0 / a LAN address "
        "for the local network (no authentication)",
    )
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
    ex = sub.add_parser("export", help="write the curated steps as PMechDB raw records (a file)")
    ex.add_argument("--out", type=Path, required=True, help="the CSV to write")
    ex.add_argument(
        "--status", nargs="+", default=None, help="mechanism statuses (validated corrected)"
    )
    ts = sub.add_parser("schema-ts", help="write the TypeScript mirror of the API models")
    ts.add_argument("--out", type=Path, default=None, help="default apps/ui/src/api/types.gen.ts")
    pc = sub.add_parser(
        "probe-cache",
        help="exit 0 when every species loads from the shared "
        "token cache (KSMECH_SHARED_CACHE_ROOT, real method)",
    )
    pc.add_argument("--smiles", action="append", required=True, help="a SMILES (repeatable)")
    gal = sub.add_parser(
        "gallery", help="draw the 2D depiction gallery of hard molecules (PNG + SVG per method)"
    )
    gal.add_argument("--out", type=Path, default=Path("depiction-gallery"), help="output dir")
    gal.add_argument("--only", nargs="+", default=None, metavar="NAME", help="gallery names")
    gal.add_argument(
        "--derive-templates",
        action="store_true",
        help="also write ring_templates.smi/.json re-derived from their derivations into OUT",
    )
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
    from ksmech.serve.core.settings import assert_writable

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
    from ksmech.serve.core.settings import load_settings

    try:
        return load_settings(**{k: v for k, v in overrides.items() if v is not None})
    except ValueError as exc:
        raise SettingsError(str(exc)) from exc


def is_loopback(host: str) -> bool:
    return ipaddress.IPv4Address(host).is_loopback  # ValueError unless an IPv4 address


def _loopback_address(name: str) -> bool:
    """True for a loopback IPv4 address such as 127.0.1.1 (the host name's /etc/hosts alias)."""
    try:
        return ipaddress.IPv4Address(name).is_loopback
    except ValueError:
        return False


def primary_ipv4() -> str | None:
    """The address of the interface that carries the default route (a UDP connect sends no
    packet), or None without a network."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1: never answered, only routed
            return s.getsockname()[0]
        except OSError:
            return None


def lan_hosts(host: str) -> list[str]:
    """Host / Origin names a browser on the local network may use for this server: this
    machine's host names and IPv4 addresses (all of them for 0.0.0.0, else the bound one).
    Empty when the server binds a loopback address."""
    if is_loopback(host):
        return []
    name = socket.gethostname()
    names = {name, f"{name}.local", socket.getfqdn()}
    if host == "0.0.0.0":
        with contextlib.suppress(OSError):
            names |= {ai[4][0] for ai in socket.getaddrinfo(name, None, socket.AF_INET)}
        names.add(primary_ipv4() or "")
    else:
        names.add(host)
    return sorted(n for n in names if n and n != "localhost" and not _loopback_address(n))


def _banner(host: str, port: int, settings: ServeSettings, names: list[str]) -> None:
    where = f"chemistry {settings.chemistry}, data root {settings.data_root}"
    if not names:
        print(
            f"ksmech-serve on http://{host}:{port} ({where}); from your PC: "
            f"ssh -L {port}:localhost:{port} <server>",
            flush=True,
        )
        return
    urls = ", ".join(f"http://{n}:{port}" for n in names if n[0].isdigit()) or host
    print(f"ksmech-serve on {host}:{port} ({where}); on the local network: {urls}", flush=True)
    print(
        f"WARNING: listening on {host}:{port} without authentication: anyone on the local "
        "network can run predictions, edit the curation database and start exports",
        file=sys.stderr,
        flush=True,
    )


def cmd_run(args: argparse.Namespace) -> int:
    import uvicorn

    from ksmech.serve.app.api import create_app

    if not 0 < args.port < 65536:
        print(f"--port must be in 1..65535, got {args.port}", file=sys.stderr)
        return 2
    try:
        names = lan_hosts(args.host)
    except ValueError:
        print(f"--host must be an IPv4 address, got {args.host!r}", file=sys.stderr)
        return 2
    settings = _settings(
        chemistry=args.chemistry, dev=args.dev or None, auto_index=args.index_on_start or None
    )
    if not _take_lock(settings, "run"):
        return 2
    if port_in_use(args.port, args.host):
        print(
            f"port {args.port} is in use on {args.host}: stop the other server or pass --port",
            file=sys.stderr,
        )
        return 2
    app = create_app(settings, serve_ui=not args.no_ui, lan_hosts=names)
    _banner(args.host, args.port, settings, names)
    try:
        uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False, log_level="info")
    except SystemExit as exc:
        if exc.code not in (0, None):
            msg = f"uvicorn could not serve {args.host}:{args.port} (exit {exc.code})"
            print(msg, file=sys.stderr)
            return 2
    return 0


# ---- recheck and export (offline batch jobs through a one-worker ProcessScheduler)


def _job_context(settings: ServeSettings, scheduler, name: str) -> LocalJobContext:
    from ksmech.serve.core.contracts import LocalJobContext
    from ksmech.serve.core.settings import append_failure

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
    from ksmech.serve.compute.procsched import ProcessScheduler
    from ksmech.serve.compute.worker import Scheduler
    from ksmech.serve.core import db
    from ksmech.serve.core.settings import TaskEnv
    from ksmech.serve.runtime.context import build_context

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


def _disabled(capability: str) -> int | None:
    """Exit 2 with the reason when `capability` is off, else None."""
    from ksmech.serve.core.capabilities import CAPABILITIES

    on, reason = CAPABILITIES[capability]
    if on:
        return None
    print(f"FEATURE_DISABLED: {capability} is disabled: {reason}", file=sys.stderr)
    return 2


def cmd_recheck(args: argparse.Namespace) -> int:
    from ksmech.serve.curate.recheck import run_recheck

    if (rc := _disabled("curate_recheck")) is not None:
        return rc

    request = {
        "mechanism_ids": args.mechanism,
        "predictor": args.predictor,
        "stale_only": bool(args.stale_only),
    }

    def body(ctx, job, engine) -> dict:
        return run_recheck(job, request, engine=engine, registry=ctx.registry)

    return _offline(_settings(), "recheck", body)


def cmd_export(args: argparse.Namespace) -> int:
    """The PMechDB-format file of the curated steps (`serve.curate.export`; never training data)."""
    from ksmech.serve.core import db
    from ksmech.serve.curate.export import DEFAULT_STATUSES, pmechdb_export

    if (rc := _disabled("export")) is not None:
        return rc
    settings = _settings()
    engine = db.make_engine(settings.curation_db)
    db.init_db(engine)
    text, report = pmechdb_export(engine, args.status or DEFAULT_STATUSES)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps(report | {"path": str(args.out)}, indent=2, sort_keys=True))
    return 0


# ---- schema-ts and probe-cache


def cmd_schema_ts(args: argparse.Namespace) -> int:
    from ksmech.serve.core.schemas import write_typescript

    path = write_typescript(args.out)
    print(f"wrote {path}")
    return 0


def probe_keys(smiles: list[str]) -> tuple[list[str], list[str]]:
    """(every fragment token key, the invalid inputs) of the SMILES list."""
    from ksmech.serve.core.errors import ServeError
    from ksmech.serve.core.keys import side_keys

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
    from ksmech.serve.chem.loader import LayeredTokenLoader

    settings = _settings()
    keys, invalid = probe_keys(list(args.smiles))
    loader = LayeredTokenLoader(
        [settings.shared_tokens], QMConfig().method, roots=[settings.shared_root]
    )
    missing = [k for k in keys if loader(k) is None]
    for line in invalid:
        print(f"invalid: {line}")
    for key in missing:
        print(f"missing: {key}")
    found = len(keys) - len(missing)
    print(f"probe-cache: {found}/{len(keys)} token keys load from {settings.shared_tokens}")
    return 0 if keys and not missing and not invalid else 1


def cmd_gallery(args: argparse.Namespace) -> int:
    """RDKit only: no settings, no lock, no worker."""
    from ksmech.serve.depiction.depiction_gallery import GALLERY, derive_all, run_gallery

    unknown = sorted(set(args.only or ()) - {name for name, _ in GALLERY})
    if unknown:
        print(f"ksmech-serve gallery: unknown names {unknown}", file=sys.stderr)
        return 2
    summary = run_gallery(args.out, args.only)
    if args.derive_templates:
        derive_all(args.out / "derived-templates")
    poor = [n for n, ms in summary.items() if any(m["stereo"] == "mismatch" for m in ms.values())]
    print(f"gallery: {len(summary)} molecules in {args.out}; stereo mismatches: {poor or 'none'}")
    return 0


HANDLERS = {
    "run": cmd_run,
    "recheck": cmd_recheck,
    "export": cmd_export,
    "schema-ts": cmd_schema_ts,
    "probe-cache": cmd_probe_cache,
    "gallery": cmd_gallery,
}


def main(argv: list[str] | None = None) -> int:
    _pin_env()
    from ksmech.serve.core.settings import pin_threads

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
