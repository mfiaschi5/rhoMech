"""`ksmech-serve` (design 12): the parser (a --port and no --host), the default `run`
insertion, uvicorn on 127.0.0.1 with one worker, --index-on-start, the server lock (exit 2),
a port in use (exit 2), the static frontend (missing-dist page, the SPA mount with the wasm
type, the dev banner), schema-ts, probe-cache exit codes, and offline export / recheck."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket

import pytest

from ksmech.serve import cli
from ksmech.serve.capabilities import CAPABILITIES
from ksmech.serve.testing import requires
from tests.test_serve_fixtures_api import (  # noqa: F401 (fixtures)
    api_client,
    no_xtb,
    serve_settings,
    synthetic_root,
)

pytestmark = pytest.mark.serve


@pytest.fixture
def env_root(tmp_path, monkeypatch):
    """KSMECH_DATA_ROOT on a tmp root (synthetic chemistry); the thread variables set to 4,
    so the CLI must pin them back to 1."""
    for var in list(os.environ):
        if var.startswith("KSMECH_"):
            monkeypatch.delenv(var)
    monkeypatch.setenv("KSMECH_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("KSMECH_SERVE_CHEMISTRY", "synthetic")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(var, "4")
    yield tmp_path
    cli.release_locks()


@pytest.fixture
def fake_server(monkeypatch):
    """Monkeypatched create_app and uvicorn.run; records their arguments."""
    import uvicorn

    from ksmech.serve import api

    calls: dict = {}

    def create_app(settings, **kwargs):
        calls["settings"], calls["app_kwargs"] = settings, kwargs
        return "app"

    def run(app, **kwargs):
        calls["app"], calls["run"] = app, kwargs

    monkeypatch.setattr(api, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", run)
    return calls


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- parser


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def test_parser_host_defaults_to_loopback():
    parser = cli.build_parser()
    args = parser.parse_args(["run", "--port", "8123"])
    assert args.port == 8123 and args.chemistry is None and not args.dev
    assert args.host == "127.0.0.1"
    assert parser.parse_args(["run", "--host", "0.0.0.0"]).host == "0.0.0.0"
    subs = _subcommands(parser)
    assert set(subs) == set(cli.COMMANDS)
    for name, sub in subs.items():
        options = {o for a in sub._actions for o in a.option_strings}
        assert ("--host" in options) == (name == "run"), name
    assert "--port" in {o for a in subs["run"]._actions for o in a.option_strings}


def test_lan_hosts(monkeypatch):
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "labhost")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: "labhost.example.org")
    monkeypatch.setattr(
        cli.socket,
        "getaddrinfo",
        lambda *a: [(2, 1, 6, "", ("10.0.0.7", 0)), (2, 1, 6, "", ("127.0.1.1", 0))],
    )
    monkeypatch.setattr(cli, "primary_ipv4", lambda: "192.168.1.50")
    assert cli.lan_hosts("127.0.0.1") == [] and cli.lan_hosts("127.0.1.1") == []
    every = {"labhost", "labhost.local", "labhost.example.org", "10.0.0.7", "192.168.1.50"}
    assert set(cli.lan_hosts("0.0.0.0")) == every
    bound = cli.lan_hosts("192.168.1.50")
    assert set(bound) == {"labhost", "labhost.local", "labhost.example.org", "192.168.1.50"}
    for bad in ("localhost", "::1", "not-an-address"):
        with pytest.raises(ValueError):
            cli.lan_hosts(bad)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["run"]),
        (["--port", "8001"], ["run", "--port", "8001"]),
        (["export"], ["export"]),
        (["probe-cache", "--smiles", "C"], ["probe-cache", "--smiles", "C"]),
        (["-h"], ["-h"]),
    ],
)
def test_run_is_inserted(argv, expected):
    assert cli.normalize_argv(argv) == expected


# ---- run


def test_run_binds_loopback_with_one_worker(env_root, fake_server):
    port = _free_port()
    assert cli.main(["--port", str(port)]) == 0
    assert fake_server["app"] == "app"
    kw = fake_server["run"]
    assert kw == {
        "host": "127.0.0.1",
        "port": port,
        "workers": 1,
        "reload": False,
        "log_level": "info",
    }
    s = fake_server["settings"]
    assert s.data_root == env_root.resolve() and s.chemistry == "synthetic"
    assert s.auto_index is False and s.dev is False
    assert fake_server["app_kwargs"] == {"serve_ui": True, "lan_hosts": []}
    assert all(
        os.environ[v] == "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )


def test_run_on_the_local_network(env_root, fake_server, monkeypatch, capsys):
    monkeypatch.setattr(cli, "lan_hosts", lambda host: ["192.168.1.50", "labhost"])
    port = _free_port()
    assert cli.main(["--host", "0.0.0.0", "--port", str(port)]) == 0
    assert fake_server["run"]["host"] == "0.0.0.0" and fake_server["run"]["port"] == port
    assert fake_server["app_kwargs"]["lan_hosts"] == ["192.168.1.50", "labhost"]
    out, err = capsys.readouterr()
    assert f"http://192.168.1.50:{port}" in out
    assert "without authentication" in err


def test_run_refuses_a_host_that_is_not_ipv4(env_root, fake_server, capsys):
    assert cli.main(["--host", "localhost"]) == 2
    assert "IPv4" in capsys.readouterr().err and "run" not in fake_server


def test_run_options(env_root, fake_server):
    port = _free_port()
    argv = [
        "run",
        "--port",
        str(port),
        "--index-on-start",
        "--no-ui",
        "--dev",
        "--chemistry",
        "cached",
    ]
    assert cli.main(argv) == 0
    s = fake_server["settings"]
    assert s.auto_index is True and s.dev is True and s.chemistry == "cached"
    assert fake_server["app_kwargs"] == {"serve_ui": False, "lan_hosts": []}


def test_run_refuses_a_held_lock(env_root, fake_server, capsys):
    from ksmech.serve.settings import load_settings

    held = cli.acquire_server_lock(load_settings())
    assert held is not None
    try:
        assert cli.main(["--port", str(_free_port())]) == 2
        assert "run" not in fake_server
        assert cli.main(["export"]) == 2
        err = capsys.readouterr().err
        if CAPABILITIES["export"][0]:
            assert "server running: use the web UI or POST /api/curate/export" in err
        else:
            assert "FEATURE_DISABLED: export is disabled" in err
        assert cli.main(["recheck", "--stale-only"]) == 2
        err = capsys.readouterr().err
        if CAPABILITIES["curate_recheck"][0]:
            assert "POST /api/curate/recheck" in err
        else:
            assert "FEATURE_DISABLED: curate_recheck is disabled" in err
    finally:
        held.close()
    assert cli.main(["--port", str(_free_port())]) == 0
    assert "run" in fake_server


def test_run_refuses_a_port_in_use(env_root, fake_server, capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert cli.port_in_use(port)
        assert cli.main(["--port", str(port)]) == 2
    assert f"port {port} is in use" in capsys.readouterr().err
    assert "run" not in fake_server


def test_bad_environment_is_exit_2(env_root, monkeypatch, capsys):
    monkeypatch.setenv("KSMECH_SERVE_WORKERS", "many")
    assert cli.main(["probe-cache", "--smiles", "C"]) == 2
    assert "KSMECH_SERVE_WORKERS" in capsys.readouterr().err


# ---- static frontend


def test_missing_dist_page(serve_settings, tmp_path):  # noqa: F811
    s = serve_settings(chemistry="synthetic", frontend_dist=tmp_path / "no-dist")
    with api_client(s) as client:
        page = client.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "frontend not built" in page.text and "npm --prefix apps/ui run build" in page.text
        assert client.get("/api/meta").json()["frontend_built"] is False


def test_dist_is_served_with_the_wasm_type(serve_settings, tmp_path):  # noqa: F811
    dist = tmp_path / "dist"
    (dist / "vendor").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ksmech</title><div id=root></div>")
    (dist / "vendor" / "RDKit_minimal.wasm").write_bytes(b"\0asm\1\0\0\0" + b"\0" * 2048)
    s = serve_settings(chemistry="synthetic", frontend_dist=dist)
    with api_client(s) as client:
        assert "<div id=root>" in client.get("/").text
        wasm = client.get("/vendor/RDKit_minimal.wasm")
        assert wasm.status_code == 200
        assert wasm.headers["content-type"] == "application/wasm"
        assert client.get("/api/meta").json()["frontend_built"] is True
        assert client.get("/api/health").status_code == 200


def test_spa_fallback_for_app_routes_only(serve_settings, tmp_path):  # noqa: F811
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    index = "<!doctype html><title>ksmech</title><div id=root></div>"
    (dist / "index.html").write_text(index)
    (dist / "assets" / "a.js").write_text("console.log(1);")
    s = serve_settings(chemistry="synthetic", frontend_dist=dist)
    with api_client(s) as client:
        for path in ("/predict", "/curate/12", "/browse"):
            page = client.get(path)
            assert page.status_code == 200, path
            assert page.headers["content-type"].startswith("text/html") and page.text == index
        assert client.get("/assets/a.js").text == "console.log(1);"
        for path in ("/api/nope", "/assets/missing.js", "/api"):
            r = client.get(path)
            assert r.status_code == 404 and r.headers["content-type"] == "application/json", path
            assert r.json()["error"]["message"] == f"no route or file {path}"


def test_dev_and_no_ui_pages(serve_settings, tmp_path):  # noqa: F811
    s = serve_settings(chemistry="synthetic", dev=True)
    with api_client(s) as client:
        assert "dev: use Vite on :5173" in client.get("/").text
        meta = client.get("/api/meta").json()
        assert meta["dev"] is True and meta["frontend_built"] is False
    s = serve_settings(chemistry="synthetic")
    with api_client(s, serve_ui=False) as client:
        assert "--no-ui" in client.get("/").text


# ---- schema-ts and probe-cache


def test_schema_ts_writes_the_file(env_root):
    from ksmech.serve.schemas import typescript

    out = env_root / "types.gen.ts"
    assert cli.main(["schema-ts", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == typescript()


def _real_token_copy(root, shared, key: str) -> None:
    """Place the synthetic token file of `key` (from the data root `root`) where the
    real-method loader of the shared root `shared` looks for it."""
    from ksmech.orbitals.build import tokens_path
    from ksmech.qm.xtb_calc import QMConfig
    from ksmech.serve.synthetic import SYNTHETIC_METHOD

    src = tokens_path(root / "data" / "serve" / "tokens", key, 0, SYNTHETIC_METHOD)
    dst = tokens_path(shared / "data" / "cache" / "tokens", key, 0, QMConfig().method)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def test_probe_cache_exit_codes(env_root, synthetic_root, monkeypatch, capsys):  # noqa: F811
    shared = env_root / "shared"
    monkeypatch.setenv("KSMECH_SHARED_CACHE_ROOT", str(shared))
    _real_token_copy(synthetic_root, shared, "C[O-]")
    before = sorted(p.name for p in shared.rglob("*"))
    assert cli.main(["probe-cache", "--smiles", "C[O-]"]) == 0
    assert "1/1 token keys load" in capsys.readouterr().out
    assert cli.main(["probe-cache", "--smiles", "C[O-]", "--smiles", "CCl"]) == 1
    assert "missing: CCl" in capsys.readouterr().out
    assert cli.main(["probe-cache", "--smiles", "C1CC"]) == 1
    assert "invalid: C1CC" in capsys.readouterr().out
    assert sorted(p.name for p in shared.rglob("*")) == before
    assert not (env_root / "data" / "serve").exists()


# ---- offline export and recheck (empty database: no worker task runs)


def test_disabled_commands_exit_2(env_root, capsys):
    """`export` and `recheck` refuse with FEATURE_DISABLED (spec v8 port) and touch nothing."""
    for cmd, cap in ((["export"], "export"), (["recheck", "--stale-only"], "curate_recheck")):
        if CAPABILITIES[cap][0]:
            continue
        assert cli.main(cmd) == 2
        assert f"FEATURE_DISABLED: {cap} is disabled: {CAPABILITIES[cap][1]}" in (
            capsys.readouterr().err
        )
    assert not (env_root / "data" / "serve").exists()
    assert not (env_root / "data" / "processed" / "user_curated").exists()


@requires("export")
@requires("curate_recheck")
def test_offline_export_and_recheck(env_root, capsys):
    assert cli.main(["export"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["n_steps_written"] == 0 and report["n_mechanisms"] == 0
    assert (env_root / "data" / "processed" / "user_curated" / "steps.parquet").is_file()
    assert cli.main(["recheck", "--stale-only"]) == 0
    assert json.loads(capsys.readouterr().out)["n_steps"] == 0
    assert (env_root / "data" / "serve" / "run" / "server.lock").is_file()
