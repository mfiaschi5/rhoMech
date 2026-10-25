"""Provenance of every cached or derived artifact: what it was built from, with which code, spec
and configuration, and the hash of its own content (docs/decisions.md, 2026-10-21).

Every writer calls `stamp()` and stores the result in the artifact's own header:
- `.npz`: the member `__provenance__`, a JSON string (`np.load` reads one member lazily);
- `.json`: the top-level key `provenance`;
- framed binary files (sample files): `MAGIC`, an 8-byte header length, the JSON header, then
  the payload.

Every loader calls `verify()`, which raises `StaleArtifact` (reason STALE_ARTIFACT) naming what
changed. An input is either `file:<path>`, whose current hash `verify` reads from that file's
own header (recomputing it from the bytes only when the file has no header), or a named value
(`smiles`, `tokens:<key>`) that the caller passes as `expect`. `schema_version` and `config` are
compared; `code_git_hash`, `spec_sha256` and `built_at` are recorded for tracing only, so a new
commit does not by itself make an artifact stale.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import zipfile
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

STALE_ARTIFACT = "STALE_ARTIFACT"
HEADER_KEY = "__provenance__"
MAGIC = b"KSMECH-PROVENANCE-1\n"
REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs" / "ksmech_architecture.tex"
_CHUNK = 1 << 20


class CorruptArtifact(ValueError):
    """A file that cannot be read at all (truncated, not the format its suffix says)."""

    reason = "CORRUPT_ARTIFACT"


class StaleArtifact(Exception):
    """An artifact whose header does not match its inputs, schema or configuration."""

    reason = STALE_ARTIFACT

    def __init__(self, path: Path | str, changes: list[str]):
        self.path = Path(path)
        self.changes = list(changes)
        super().__init__(f"{STALE_ARTIFACT}: {path}: " + "; ".join(self.changes))


@dataclass(frozen=True)
class Provenance:
    kind: str  # qm_record | tokens | contact_block | samples | split | baseline
    schema_version: int
    code_git_hash: str
    spec_sha256: str
    inputs: dict[str, str]  # name -> sha256; `file:<path>` names are re-hashed by `verify`
    config: dict[str, Any]
    built_at: str
    content_sha256: str = ""  # hash of the artifact's own payload (everything but the header)
    extra: dict[str, Any] = field(default_factory=dict)  # tracing notes, never compared
    code_dirty: bool = False  # built from a tree with uncommitted changes to tracked files
    code_diff_sha256: str = ""  # sha256 of `git diff HEAD` at build time ("" when clean)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        kw = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        head = str(kw.get("code_git_hash", ""))
        if head.endswith("+dirty"):  # headers written before the flag existed
            kw["code_git_hash"], kw["code_dirty"] = head.removesuffix("+dirty"), True
        return cls(**kw)

    @classmethod
    def from_json(cls, text: str) -> Provenance:
        return cls.from_dict(json.loads(text))


# ---- hashes


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def array_digest(arrays: Mapping[str, Any]) -> str:
    """sha256 over the named arrays (name, dtype, shape, bytes), in sorted name order."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[name]))
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def git_env() -> dict[str, str]:
    """The environment without git's own variables. A hook of `git commit --only` (or `git
    commit <paths>`) exports GIT_INDEX_FILE, the absolute path of the temporary index being
    committed; a git run inside the hook would read and refresh it as this checkout's index."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
        env=git_env(),
    ).stdout


@cache
def code_state() -> tuple[str, bool, str]:
    """(HEAD, dirty, sha256 of `git diff HEAD`): dirty when tracked files differ from HEAD or
    `src/` holds untracked files (a module the build may import); the hash covers the diff and
    those files' contents, so it identifies the uncommitted change an artifact was built with."""
    try:
        head = _git("rev-parse", "HEAD").strip()
        diff = _git("diff", "HEAD")
        untracked = _git("ls-files", "--others", "--exclude-standard", "--", "src").split()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True, ""
    for name in sorted(untracked):
        diff += f"\n--- untracked {name}\n" + (REPO_ROOT / name).read_text(errors="replace")
    return head, bool(diff), sha256_text(diff) if diff else ""


def code_git_hash() -> str:
    head, dirty, _ = code_state()
    return head + ("+dirty" if dirty else "")


@cache
def spec_sha256() -> str:
    return file_sha256(SPEC_PATH) if SPEC_PATH.is_file() else ""


def _normalized(obj: Any) -> Any:
    """`obj` as it reads back from JSON, so stored and expected configs compare equal."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def stamp(
    kind: str,
    schema_version: int,
    inputs: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    content_sha256: str = "",
    extra: Mapping[str, Any] | None = None,
) -> Provenance:
    """The provenance header of an artifact about to be written."""
    head, dirty, diff_sha = code_state()
    return Provenance(
        kind=kind,
        schema_version=int(schema_version),
        code_git_hash=head,
        code_dirty=dirty,
        code_diff_sha256=diff_sha,
        spec_sha256=spec_sha256(),
        inputs=dict(sorted((inputs or {}).items())),
        config=_normalized(dict(config or {})),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        content_sha256=content_sha256,
        extra=_normalized(dict(extra or {})),
    )


# ---- file inputs


def file_input_name(path: Path | str) -> str:
    """`file:<path>`, relative to the repository when the file lies inside it."""
    p = Path(path).resolve()
    try:
        return f"file:{p.relative_to(REPO_ROOT)}"
    except ValueError:
        return f"file:{p}"


def input_path(name: str) -> Path:
    p = Path(name.removeprefix("file:"))
    return p if p.is_absolute() else REPO_ROOT / p


def file_input(path: Path | str) -> dict[str, str]:
    """{`file:<path>`: its current hash}, the form every writer records a file input in."""
    return {file_input_name(path): current_hash(path)}


_HASH_MEMO: dict[tuple[str, int, int], str] = {}


def current_hash(path: Path | str) -> str:
    """The content hash in the file's own header, or the sha256 of its bytes when it has none.
    Memoized per process on (path, mtime, size)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _HASH_MEMO:
        header = read_header(p)
        _HASH_MEMO[key] = (
            header.content_sha256 if header and header.content_sha256 else file_sha256(p)
        )
    return _HASH_MEMO[key]


# ---- headers


def _npz_header(path: Path) -> Provenance | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            return Provenance.from_json(str(z[HEADER_KEY])) if HEADER_KEY in z.files else None
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def _json_header(path: Path) -> Provenance | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    prov = d.get("provenance") if isinstance(d, dict) else None
    return Provenance.from_dict(prov) if isinstance(prov, dict) else None


def _framed_header(path: Path) -> Provenance | None:
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            return None
        n = int.from_bytes(fh.read(8), "little")
        return Provenance.from_json(fh.read(n).decode())


def read_header(path: Path | str) -> Provenance | None:
    """The provenance header of an artifact, or None when it has none (or is unreadable)."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    if p.suffix == ".npz":
        return _npz_header(p)
    if p.suffix == ".json":
        return _json_header(p)
    return _framed_header(p)


# ---- verify


def _config_changes(header: Provenance, config: Mapping[str, Any]) -> list[str]:
    missing = object()
    out = []
    for key, now in _normalized(dict(config)).items():
        was = header.config.get(key, missing)
        if was != now:
            out.append(f"config {key}: {'absent' if was is missing else repr(was)} -> {now!r}")
    return out


def _input_changes(header: Provenance, expect: Mapping[str, str]) -> list[str]:
    out = []
    for name, was in header.inputs.items():
        if name.startswith("file:"):
            p = input_path(name)
            if not p.is_file():
                out.append(f"input missing: {p}")
            elif current_hash(p) != was:
                out.append(f"input changed: {p}")
        elif name in expect and expect[name] != was:
            out.append(f"input changed: {name}")
    out += [f"input not recorded: {name}" for name in expect if name not in header.inputs]
    return out


def verify(
    path: Path | str,
    kind: str | None = None,
    schema_version: int | None = None,
    config: Mapping[str, Any] | None = None,
    expect: Mapping[str, str] | None = None,
) -> Provenance:
    """The artifact's header, after checking kind, schema version, the given config keys and
    every input. Raises StaleArtifact naming each difference."""
    header = read_header(path)
    if header is None:
        if Path(path).suffix == ".npz" and not zipfile.is_zipfile(path):
            raise CorruptArtifact(f"{path}: not a readable .npz")
        raise StaleArtifact(path, ["no provenance header"])
    changes = []
    if kind is not None and header.kind != kind:
        changes.append(f"kind {header.kind!r}, expected {kind!r}")
    if schema_version is not None and header.schema_version != schema_version:
        changes.append(f"schema_version {header.schema_version} -> {schema_version}")
    changes += _config_changes(header, config or {})
    changes += _input_changes(header, expect or {})
    if changes:
        raise StaleArtifact(path, changes)
    return header


def require_clean(header: Provenance, path: Path | str) -> None:
    """Training and evaluation read only artifacts built from committed code: StaleArtifact
    (STALE_ARTIFACT) for a dirty build, naming its diff."""
    if header.code_dirty:
        raise StaleArtifact(
            path,
            [
                f"built from a dirty tree (HEAD {header.code_git_hash[:12]}, "
                f"diff sha256 {header.code_diff_sha256[:12] or 'unknown'})"
            ],
        )


# ---- writers


def atomic_write(path: Path | str, data: bytes) -> Path:
    """tmp file, fsync, rename: a crash never leaves a partial file under the final name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def save_npz(
    path: Path | str, arrays: Mapping[str, Any], prov: Provenance, compressed: bool = True
) -> Path:
    """Write `arrays` plus the header member, atomically. `prov.content_sha256` should be
    `array_digest(arrays)`."""
    buf = io.BytesIO()
    payload = {**arrays, HEADER_KEY: np.array(prov.to_json())}
    (np.savez_compressed if compressed else np.savez)(buf, **payload)
    return atomic_write(path, buf.getvalue())


def write_framed(path: Path | str, payload: bytes, prov: Provenance) -> Path:
    header = prov.to_json().encode()
    return atomic_write(path, MAGIC + len(header).to_bytes(8, "little") + header + payload)


def read_framed(path: Path | str) -> tuple[Provenance | None, bytes]:
    """(header, payload). A file without the frame is all payload."""
    data = Path(path).read_bytes()
    if not data.startswith(MAGIC):
        return None, data
    n = int.from_bytes(data[len(MAGIC) : len(MAGIC) + 8], "little")
    start = len(MAGIC) + 8
    return Provenance.from_json(data[start : start + n].decode()), data[start + n :]


def write_json(path: Path | str, obj: Mapping[str, Any], prov: Provenance) -> Path:
    text = json.dumps({"provenance": asdict(prov), **obj}, indent=1, sort_keys=False)
    return atomic_write(path, (text + "\n").encode())


# ---- collecting the inputs a build step touches


_COLLECTORS: ContextVar[tuple[dict[str, str], ...]] = ContextVar("collectors", default=())


@contextlib.contextmanager
def collecting() -> Iterator[dict[str, str]]:
    """Gather every input `note_input` reports inside the block (nested blocks see theirs and
    pass them up)."""
    used: dict[str, str] = {}
    token = _COLLECTORS.set((*_COLLECTORS.get(), used))
    try:
        yield used
    finally:
        _COLLECTORS.reset(token)


def note_input(name: str, sha: str) -> None:
    for used in _COLLECTORS.get():
        used[name] = sha


# ---- the chain


def _status(header: Provenance) -> list[str]:
    return _input_changes(header, {})


def describe_chain(
    path: Path | str, depth: int = 0, max_depth: int = 4, max_inputs: int = 6
) -> list[str]:
    """Lines describing the artifact and, recursively, its file inputs."""
    pad = "  " * depth
    header = read_header(path)
    if header is None:
        return [f"{pad}{path}: no provenance header (sha256 {file_sha256(path)[:12]})"]
    stale = _status(header)
    lines = [
        f"{pad}{path}: {header.kind} schema {header.schema_version}, built {header.built_at}, "
        f"code {header.code_git_hash[:12]}{'+dirty' if header.code_dirty else ''}, "
        f"spec {header.spec_sha256[:12]}, "
        f"{'STALE' if stale else 'ok'}",
        f"{pad}  config {json.dumps(header.config, sort_keys=True)}",
    ]
    lines += [f"{pad}  ! {c}" for c in stale]
    files = [n for n in header.inputs if n.startswith("file:")]
    named = [n for n in header.inputs if not n.startswith("file:")]
    if named:
        lines.append(f"{pad}  named inputs: {', '.join(named[:max_inputs])}")
    for name in files[:max_inputs]:
        p = input_path(name)
        if depth + 1 >= max_depth or not p.is_file():
            lines.append(f"{pad}  {p}{'' if p.is_file() else ' (missing)'}")
        else:
            lines += describe_chain(p, depth + 1, max_depth, max_inputs)
    if len(files) > max_inputs:
        lines.append(f"{pad}  ... and {len(files) - max_inputs} more file inputs")
    return lines
