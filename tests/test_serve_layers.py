"""The sub-packages of `ksmech.serve` keep their layers: a module-level import from another
sub-package must be one `ALLOWED` lists. Imports inside a function (deferred, to keep torch and
the heavy chemistry out of the API process or to break a cycle) and under `TYPE_CHECKING` are
not module-level and are not checked.

The foundation is one chain, each layer free to import every layer before it: core, compute,
chem, science, predict, runtime. `depiction` stands on core alone. The feature packages (pool,
workbench, embedding, browse, curate, ml) stand on the foundation and on depiction and never
import one another, with one exception: `ml` stands on core alone, so that predict, runtime and
the other features may read its checkpoint listing (`ml.checkpoints`). `accounts` (users,
sessions, quotas) also stands on core alone: core reaches it only through what the app injects.
`app` wires them all."""

import ast
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "src" / "ksmech" / "serve"
FOUNDATION = ("core", "compute", "chem", "science", "predict", "runtime")
FEATURES = ("pool", "workbench", "embedding", "browse", "curate", "ml", "accounts")
ALLOWED: dict[str, set[str]] = {
    layer: set(FOUNDATION[:k]) for k, layer in enumerate(FOUNDATION)
} | {"depiction": {"core"}}
ALLOWED |= {f: {*FOUNDATION, "depiction", "ml"} for f in FEATURES}
ALLOWED["ml"] = {"core"}  # the checkpoint listing: readable by the foundation above core
ALLOWED["accounts"] = {"core"}
for _layer in ("predict", "runtime"):
    ALLOWED[_layer].add("ml")
ALLOWED["app"] = {*FOUNDATION, "depiction", *FEATURES}


def _module_level_targets(path: Path) -> set[str]:
    """The serve sub-packages a module imports at module level, outside TYPE_CHECKING."""
    out = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        out |= {n.split(".")[2] for n in names if n.startswith("ksmech.serve.")}
    return out


def test_every_serve_module_lives_in_a_known_sub_package():
    loose = [p.name for p in SERVE.glob("*.py") if p.name != "__init__.py"]
    assert loose == []
    packages = {p.name for p in SERVE.iterdir() if (p / "__init__.py").is_file()}
    assert packages == set(ALLOWED)


def test_module_level_imports_follow_the_layers():
    offenders = []
    for path in sorted(SERVE.rglob("*.py")):
        rel = path.relative_to(SERVE).parts
        if len(rel) < 2:
            continue
        pkg = rel[0]
        for target in sorted(_module_level_targets(path) - {pkg} - ALLOWED[pkg]):
            offenders.append(f"{path.relative_to(SERVE)} imports serve.{target}")
    assert offenders == []


def test_the_allowed_graph_has_no_cycle():
    seen: dict[str, int] = {}

    def visit(node: str) -> None:
        seen[node] = 1
        for nxt in ALLOWED[node]:
            assert seen.get(nxt) != 1, f"cycle through {node} -> {nxt}"
            if nxt not in seen:
                visit(nxt)
        seen[node] = 2

    for node in ALLOWED:
        if node not in seen:
            visit(node)
