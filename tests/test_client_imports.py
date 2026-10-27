"""Import hygiene of the HTTP client: in a fresh interpreter, `ksmech.client` (and its CLI and
summaries) loads httpx, pydantic and `ksmech.serve.core.schemas` and nothing of the web server,
the database, the chemistry or the model."""

from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN = ("fastapi", "starlette", "uvicorn", "sqlmodel", "sqlalchemy", "torch", "rdkit", "mcp")
ALLOWED_KSMECH = {"ksmech", "ksmech.serve", "ksmech.serve.core", "ksmech.serve.core.schemas"}

PROBE = """
import json, sys
for name in sys.argv[1:]:
    __import__(name)
print(json.dumps(sorted(sys.modules)))
"""


def loaded_modules(*names: str) -> list[str]:
    """`sys.modules` of a fresh interpreter after importing `names`."""
    out = subprocess.run(
        [sys.executable, "-c", PROBE, *names], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def test_client_imports_no_server_package():
    mods = loaded_modules("ksmech.client", "ksmech.client.cli", "ksmech.client.summary")
    roots = {m.split(".")[0] for m in mods}
    assert not roots & set(FORBIDDEN), sorted(roots & set(FORBIDDEN))
    assert {"httpx", "pydantic"} <= roots
    ksmech = {m for m in mods if m.split(".")[0] == "ksmech"}
    assert {m for m in ksmech if not m.startswith("ksmech.client")} == ALLOWED_KSMECH


def test_schemas_module_alone_pulls_no_serve_module():
    mods = loaded_modules("ksmech.serve.core.schemas")
    assert {m for m in mods if m.split(".")[0] == "ksmech"} == ALLOWED_KSMECH
    assert not {m.split(".")[0] for m in mods} & set(FORBIDDEN)


def test_mcp_server_imports_the_client_only():
    """The MCP SDK brings starlette and uvicorn (its HTTP transport); nothing of ksmech-serve,
    the database, the chemistry or the model comes with it."""
    mods = loaded_modules("ksmech.mcp.server")
    roots = {m.split(".")[0] for m in mods}
    assert not roots & {"fastapi", "sqlmodel", "sqlalchemy", "torch", "rdkit", "numpy"}
    ksmech = {m for m in mods if m.split(".")[0] == "ksmech"}
    others = {m for m in ksmech if not m.startswith(("ksmech.client", "ksmech.mcp"))}
    assert others == ALLOWED_KSMECH
