"""Version strings and code fingerprints of the serve app.

The spec carries no version string, so `spec_version` is "v9+" + the first 12 hex of
sha256(docs/ksmech_architecture.tex) plus the artifact versions: "v9+<sha12>;tok9;smp15;qm1"
(the constants are read from the upstream sources, so the string follows them).

Two fingerprints (first 12 hex of a sha256 over files hashed by sorted relative path; a missing
file contributes "<missing>", so a fingerprint can be computed before every module exists):

- `token_fingerprint()` namespaces the caches that depend on the tokens and the Gaussian basis
  (orbital cubes, workbench orbital sets, densities, properties). It covers the token and QM
  schema versions and the bytes of the token, QM and basis sources (`serve/chem/gto.py` with its
  STO-nG table), never other serve code.
- `physics_fingerprint()` keys analysis files, predictor versions and curated-step staleness:
  the token inputs plus the geometry-free pair physics (`physics/*.py`), SAMPLE_VERSION, the QM
  method string, the arrows sources and the serve modules that decide scores, validity,
  products and readings.

Upstream constants are read from the source files with `ast` (`source_constant`), not by
importing them: `arrows/steps.py` and `physics/cli.py` import heavy modules, which the API
process must not load at import time.
"""

from __future__ import annotations

import ast
import hashlib
from functools import cache
from pathlib import Path

SPEC_LABEL = "v9"
SRC_ROOT = Path(__file__).resolve().parents[2]  # src/ksmech

TOKEN_FILES = (
    "serve/chem/gto.py",
    "serve/chem/data/sto_ng.json",
    "orbitals/tokens.py",
    "orbitals/descriptors.py",
    "orbitals/localize.py",
    "qm/ao.py",
    "qm/records.py",
    "qm/bond_index.py",
)
PHYSICS_EXTRA_FILES = (
    "physics/*.py",
    "orbitals/classify.py",
    "orbitals/frontier.py",
    "arrows/reactants.py",
    "arrows/steps.py",
    "arrows/apply_predicted.py",
    "arrows/apply.py",
    "arrows/align.py",
    "arrows/delta_be.py",
    "data/resonance.py",
    "qm/xtb_calc.py",
    "serve/science/analysis.py",
    "serve/science/gatom.py",
    "serve/predict/predictors.py",
    "serve/predict/expansion.py",
    "serve/science/stepapply.py",
    "serve/science/reading.py",
    "serve/chem/synthetic.py",
)
# (constant name in the fingerprint, source file, variable)
TOKEN_CONSTANTS = (
    ("tokens_schema", "orbitals/tokens.py", "SCHEMA_VERSION"),
    ("qm_schema", "qm/records.py", "SCHEMA_VERSION"),
)
PHYSICS_CONSTANTS = (("sample_version", "arrows/steps.py", "SAMPLE_VERSION"),)
MISSING = "<missing>"


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_constant(rel: str, name: str, src_root: Path = SRC_ROOT):
    """The literal value assigned to the top-level `name` in `src_root/rel`, read with `ast`
    (the module is never imported). Raises LookupError when the file or the name is absent."""
    path = Path(src_root) / rel
    if not path.is_file():
        raise LookupError(f"{path} does not exist")
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value:
            return ast.literal_eval(node.value)
    raise LookupError(f"{rel} assigns no literal {name}")


def fingerprint_files(kind: str, src_root: Path = SRC_ROOT) -> list[str]:
    """Sorted relative paths a fingerprint covers (glob patterns expanded; a pattern with no
    match is kept as written and hashed as missing)."""
    if kind not in ("token", "physics"):
        raise ValueError(f"unknown fingerprint {kind!r}")
    patterns = TOKEN_FILES + (PHYSICS_EXTRA_FILES if kind == "physics" else ())
    out: set[str] = set()
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            hits = sorted(p.relative_to(src_root).as_posix() for p in Path(src_root).glob(pattern))
            out.update(hits or [pattern])
        else:
            out.add(pattern)
    return sorted(out)


def _constants(kind: str, src_root: Path, qm_method: str | None) -> dict[str, str]:
    table = TOKEN_CONSTANTS + (PHYSICS_CONSTANTS if kind == "physics" else ())
    out = {}
    for label, rel, name in table:
        try:
            out[label] = str(source_constant(rel, name, src_root))
        except LookupError:
            out[label] = MISSING
    if kind == "physics":
        out["qm_method"] = qm_method if qm_method is not None else _qm_method()
    return out


def compute_fingerprint(kind: str, src_root: Path = SRC_ROOT, qm_method: str | None = None) -> str:
    """12 hex of sha256 over the covered files (by sorted relative path) and the constants."""
    h = hashlib.sha256()
    for rel in fingerprint_files(kind, src_root):
        path = Path(src_root) / rel
        digest = file_sha256(path) if path.is_file() else MISSING
        h.update(f"{rel}\0{digest}\n".encode())
    for label, value in sorted(_constants(kind, src_root, qm_method).items()):
        h.update(f"{label}={value}\n".encode())
    return h.hexdigest()[:12]


@cache
def token_fingerprint() -> str:
    return compute_fingerprint("token")


@cache
def physics_fingerprint() -> str:
    return compute_fingerprint("physics")


def _qm_method() -> str:
    from ksmech.qm.xtb_calc import QMConfig

    return QMConfig().method


def spec_sha12(spec_path: Path) -> str | None:
    path = Path(spec_path)
    return file_sha256(path)[:12] if path.is_file() else None


@cache
def artifact_versions() -> dict[str, int | str]:
    """Token, sample and QM schema versions, and the QM method string."""
    return {
        "tokens": source_constant("orbitals/tokens.py", "SCHEMA_VERSION"),
        "samples": source_constant("arrows/steps.py", "SAMPLE_VERSION"),
        "qm": source_constant("qm/records.py", "SCHEMA_VERSION"),
        "qm_method": _qm_method(),
    }


def spec_version(spec_path: Path) -> str:
    """ "v9+<sha12>;tok<T>;smp<S>;qm<Q>" ("v9+unknown;..." when the spec is missing)."""
    v = artifact_versions()
    sha = spec_sha12(spec_path) or "unknown"
    return f"{SPEC_LABEL}+{sha};tok{v['tokens']};smp{v['samples']};qm{v['qm']}"


def floor_version(fp: str) -> str:
    """`floor/g-atom/<fp>`: the pooled-gap floor on the physics fingerprint."""
    return f"floor/g-atom/{fp}"
