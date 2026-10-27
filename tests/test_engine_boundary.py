"""The web app and its clients are add-ons: no engine module imports `ksmech.serve`,
`ksmech.client` or `ksmech.mcp`, the server imports neither of its clients, and the engine's
dependencies carry no web or MCP package (they are the `serve`, `client` and `mcp` dependency
groups)."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "ksmech"
WEB = ("fastapi", "uvicorn", "sqlmodel", "sqlalchemy", "starlette", "pydantic")
ADD_ONS = ("serve", "client", "mcp")  # sub-packages of ksmech that are not the engine


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    return names


def _offenders(parts: tuple[str, ...], targets: tuple[str, ...]) -> list[str]:
    """Imports of `ksmech.<target>` in the modules whose first part under ksmech is in
    `parts` (an empty `parts`: the engine, every module outside the add-ons)."""
    out = []
    for path in sorted(ENGINE.rglob("*.py")):
        first = path.relative_to(ENGINE).parts[:1]
        mine = first[0] in parts if parts else not set(first) & set(ADD_ONS)
        if not mine:
            continue
        for name in _imports(path):
            if any(name == f"ksmech.{t}" or name.startswith(f"ksmech.{t}.") for t in targets):
                out.append(f"{path.relative_to(ROOT)}: {name}")
    return out


def _names(specs: list[str]) -> set[str]:
    return {d.split("=")[0].split("<")[0].split(">")[0].strip() for d in specs}


def test_no_engine_module_imports_an_add_on():
    assert _offenders((), ADD_ONS) == []


def test_the_server_imports_neither_of_its_clients():
    assert _offenders(("serve",), ("client", "mcp")) == []


def test_the_clients_import_only_the_schemas_of_the_server():
    used = {o.split(": ")[1] for o in _offenders(("client", "mcp"), ("serve",))}
    assert used <= {"ksmech.serve.core", "ksmech.serve.core.schemas"}, used


def test_engine_dependencies_carry_no_web_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = _names(project["project"]["dependencies"])
    assert not deps & set(WEB)
    assert set(WEB) <= _names(project["dependency-groups"]["serve"])


def test_engine_dependencies_carry_no_client_or_mcp_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = project["dependency-groups"]
    deps = _names(project["project"]["dependencies"])
    assert not deps & {"mcp", "httpx"}
    assert _names(groups["client"]) == {"httpx"} and _names(groups["mcp"]) == {"mcp"}
    pins = [d for g in ("client", "mcp") for d in groups[g]]
    assert all("==" in d for d in pins), pins  # exact pins
    assert {"client", "mcp"} <= set(project["tool"]["uv"]["default-groups"])
