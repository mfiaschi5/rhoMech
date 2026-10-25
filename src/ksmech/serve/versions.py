"""Version strings and code fingerprints of the serve app.

The spec carries no version string, so `spec_version` is "v8+" + the first 12 hex of
sha256(docs/ksmech_architecture.tex) plus the artifact versions: "v8+<sha12>;tok9;smp15;ori6;qm1"
(the constants are read from the upstream sources, so the string follows them). The oriented
module moved to `physics/contact/legacy_fixed23.py` (`serve.upstream.ORIENTED_SOURCE`).

Two fingerprints (first 12 hex of a sha256 over files hashed by sorted relative path; a missing
file contributes "<missing>", so a fingerprint can be computed before every module exists):

- `oriented_fingerprint()` namespaces the oriented-block cache. It covers the token, oriented
  and QM schema versions and the bytes of the upstream sources the blocks depend on, never
  serve code.
- `physics_fingerprint()` keys analysis files, predictor versions and curated-step staleness:
  the oriented inputs plus SAMPLE_VERSION, the QM method string, the arrows and remaining
  orbitals sources and the serve modules that decide scores, validity and products.

Upstream constants are read from the source files with `ast` (`source_constant`), not by
importing them: `arrows/steps.py` and `physics/cli.py` import torch, which the API process
must not load at import time.
"""

from __future__ import annotations

import ast
import hashlib
from functools import cache
from pathlib import Path

from ksmech.serve.upstream import ORIENTED_SOURCE

SPEC_LABEL = "v8"
SRC_ROOT = Path(__file__).resolve().parents[1]  # src/ksmech

ORIENTED_FILES = (
    "physics/*.py",
    "physics/contact/*.py",
    "physics/data/sto_ng.json",
    "orbitals/tokens.py",
    "orbitals/descriptors.py",
    "orbitals/localize.py",
    "qm/ao.py",
    "qm/records.py",
    "qm/bond_index.py",
)
PHYSICS_EXTRA_FILES = (
    "orbitals/classify.py",
    "orbitals/frontier.py",
    "arrows/reactants.py",
    "arrows/steps.py",
    "arrows/apply_predicted.py",
    "arrows/apply.py",
    "arrows/align.py",
    "qm/xtb_calc.py",
    "serve/analysis.py",
    "serve/candidates.py",
    "serve/predictors.py",
    "serve/expansion.py",
    "serve/stepapply.py",
    "serve/synthetic.py",
)
# (constant name in the fingerprint, source file, variable)
ORIENTED_CONSTANTS = (
    ("tokens_schema", "orbitals/tokens.py", "SCHEMA_VERSION"),
    ("oriented_version", ORIENTED_SOURCE, "ORIENTED_VERSION"),
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
    if kind not in ("oriented", "physics"):
        raise ValueError(f"unknown fingerprint {kind!r}")
    patterns = ORIENTED_FILES + (PHYSICS_EXTRA_FILES if kind == "physics" else ())
    out: set[str] = set()
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            hits = sorted(p.relative_to(src_root).as_posix() for p in Path(src_root).glob(pattern))
            out.update(hits or [pattern])
        else:
            out.add(pattern)
    return sorted(out)


def _constants(kind: str, src_root: Path, qm_method: str | None) -> dict[str, str]:
    table = ORIENTED_CONSTANTS + (PHYSICS_CONSTANTS if kind == "physics" else ())
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
def oriented_fingerprint() -> str:
    return compute_fingerprint("oriented")


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
    """Token, sample, oriented and QM schema versions, and the QM method string."""
    return {
        "tokens": source_constant("orbitals/tokens.py", "SCHEMA_VERSION"),
        "samples": source_constant("arrows/steps.py", "SAMPLE_VERSION"),
        "oriented": source_constant(ORIENTED_SOURCE, "ORIENTED_VERSION"),
        "qm": source_constant("qm/records.py", "SCHEMA_VERSION"),
        "qm_method": _qm_method(),
    }


def spec_version(spec_path: Path) -> str:
    """ "v8+<sha12>;tok<T>;smp<S>;ori<O>;qm<Q>" ("v8+unknown;..." when the spec is missing)."""
    v = artifact_versions()
    sha = spec_sha12(spec_path) or "unknown"
    return f"{SPEC_LABEL}+{sha};tok{v['tokens']};smp{v['samples']};ori{v['oriented']};qm{v['qm']}"


def floor_version(fp: str) -> str:
    """`floor/<UNTRAINED_SCORE>/d<PRIMARY_DISTANCE>/<fp>`, e.g. floor/ks3+ks2/d2.3/<fp>."""
    score = source_constant("physics/cli.py", "UNTRAINED_SCORE")
    distance = source_constant("physics/pairs.py", "PRIMARY_DISTANCE")
    return f"floor/{score}/d{distance}/{fp}"
