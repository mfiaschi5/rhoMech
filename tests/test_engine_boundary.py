"""The web app is an add-on: no engine module imports `ksmech.serve`, and the engine's
dependencies carry no web package (they are the `serve` dependency group)."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "ksmech"
WEB = ("fastapi", "uvicorn", "sqlmodel", "sqlalchemy", "starlette", "pydantic")


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    return names


def test_no_engine_module_imports_serve():
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in sorted(ENGINE.rglob("*.py"))
        if "serve" not in path.relative_to(ENGINE).parts[:1]
        for name in _imports(path)
        if name == "ksmech.serve" or name.startswith("ksmech.serve.")
    ]
    assert offenders == []


def test_engine_dependencies_carry_no_web_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = [
        d.split("=")[0].split("<")[0].split(">")[0].strip()
        for d in project["project"]["dependencies"]
    ]
    assert not set(deps) & set(WEB)
    serve = {d.split("=")[0].split("<")[0].strip() for d in project["dependency-groups"]["serve"]}
    assert set(WEB) <= serve
