"""A layered, thread-safe token loader for a long-running server (adapter A3).

`arrows.reactants.TokenCache` reads one directory, memoizes misses (a molecule built while the
server runs stays "missing") and is not thread-safe. `LayeredTokenLoader` looks in each
directory in order (the shared cache first, then the serve-owned one), skips a file whose
provenance header does not match (`token_header_problems`: no header, another kind or schema,
another token config, or a QM record it was built from that is missing or changed; counted as
`TOKENS_STALE`, logged once per key), memoizes only hits,
and guards everything with a lock. Duplicate species get the same `MoleculeTokens` object, as
with `TokenCache`, while their file is unchanged: a memo entry keeps the file's path,
`st_mtime_ns` and `st_size` (taken before the read), and a hit is dropped and reloaded when
that file changed or vanished, or when an earlier layer gained a usable file (a token rebuild
while the server runs is never served from the memo).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ksmech.orbitals.tokens import MoleculeTokens

log = logging.getLogger(__name__)

Signature = tuple[int, int]  # (st_mtime_ns, st_size)


def _signature(path: Path) -> Signature | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _resolve_input(name: str, root: Path | None) -> Path:
    """The file behind a `file:<path>` provenance input. Upstream writes paths relative to the
    checkout that built the file (`provenance.REPO_ROOT`), so a relative path is resolved
    against the layer's own root (the shared cache root for the shared layer), not against the
    checkout serve runs from."""
    from ksmech.data import provenance as prov

    rel = Path(name.removeprefix("file:"))
    return rel if rel.is_absolute() else Path(root or prov.REPO_ROOT) / rel


def token_header_problems(path: Path, root: Path | None = None) -> list[str]:
    """What makes the token file at `path` stale (empty when it is current): the checks of
    `orbitals.tokens.load_tokens` (kind, schema, `token_config()`, every `file:` input re-hashed),
    with file inputs resolved against `root` (`_resolve_input`)."""
    from ksmech.data import provenance as prov
    from ksmech.orbitals.tokens import SCHEMA_VERSION, token_config

    header = prov.read_header(path)
    if header is None:
        return ["no provenance header"]
    out = []
    if header.kind != "tokens":
        out.append(f"kind {header.kind!r}, expected 'tokens'")
    if header.schema_version != SCHEMA_VERSION:
        out.append(f"schema_version {header.schema_version} -> {SCHEMA_VERSION}")
    want = json.loads(json.dumps(token_config(), sort_keys=True, default=str))
    out += [
        f"config {k}: {header.config.get(k)!r} -> {v!r}"
        for k, v in want.items()
        if header.config.get(k) != v
    ]
    for name, was in header.inputs.items():
        if not name.startswith("file:"):
            continue
        p = _resolve_input(name, root)
        if not p.is_file():
            out.append(f"input missing: {p}")
        elif prov.current_hash(p) != was:
            out.append(f"input changed: {p}")
    return out


class LayeredTokenLoader:
    def __init__(
        self,
        dirs: Sequence[Path],
        method: str,
        maxsize: int = 256,
        roots: Sequence[Path | None] | None = None,
        bond_dirs: Sequence[Path | None] | None = None,
    ):
        """`roots[i]`: the checkout root the token files of `dirs[i]` were built in (their
        relative `file:` inputs resolve against it); None = `provenance.REPO_ROOT`.
        `bond_dirs[i]`: the Wiberg bond-index sidecars of that layer (`qm.bond_index`), attached
        to the tokens on load as `arrows.reactants.TokenCache` does; None: no sidecars."""
        self.dirs = [Path(d) for d in dirs]
        self.bond_dirs = list(bond_dirs) if bond_dirs is not None else [None] * len(self.dirs)
        self.roots = list(roots) if roots is not None else [None] * len(self.dirs)
        if len(self.roots) != len(self.dirs):
            raise ValueError("one root per token directory")
        self.method = method
        self.maxsize = int(maxsize)
        # key -> (path, signature before the read, tokens)
        self._memo: OrderedDict[str, tuple[str, Signature, MoleculeTokens]] = OrderedDict()
        self._digests: dict[tuple[str, int, int], str] = {}
        self._lock = threading.RLock()
        self.stale: dict[str, str] = {}  # key -> stale file skipped (TOKENS_STALE)
        self._events: list[dict] = []

    def _candidates(self, key: str) -> list[Path]:
        from ksmech.orbitals.build import tokens_path

        return [tokens_path(d, key, 0, self.method) for d in self.dirs]

    def _root_of(self, path: Path) -> Path | None:
        for d, root in zip(self.dirs, self.roots, strict=True):
            if Path(path).parent == d:
                return root
        return None

    def _usable(self, key: str, path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return False
            problems = token_header_problems(path, self._root_of(path))
            if not problems:
                return True
            reason = "stale (" + "; ".join(problems) + ")"
        except (OSError, ValueError, KeyError) as exc:
            reason = f"unreadable ({type(exc).__name__}: {exc})"
        with self._lock:
            if self.stale.get(key) != str(path):
                self.stale[key] = str(path)
                event = {"code": "TOKENS_STALE", "key": key, "path": str(path), "detail": reason}
                self._events.append(event)
                log.warning("TOKENS_STALE %s: %s is %s; skipped", key, path, reason)
        return False

    def path_of(self, key: str) -> Path | None:
        """The first layer's token file for `key` that exists and has the current schema."""
        for path in self._candidates(key):
            if self._usable(key, path):
                return path
        return None

    def has(self, key: str) -> bool:
        with self._lock:
            if self._memo_hit(key) is not None:
                return True
        return self.path_of(key) is not None

    def _memo_hit(self, key: str) -> MoleculeTokens | None:
        """The memoized tokens of `key` while their file keeps its signature and no earlier
        layer holds a usable file; otherwise the entry is dropped and None returned."""
        entry = self._memo.get(key)
        if entry is None:
            return None
        path, sig, tokens = entry
        earlier = []
        for p in self._candidates(key):
            if str(p) == path:
                break
            earlier.append(p)
        if _signature(Path(path)) == sig and not any(
            p.is_file() and self._usable(key, p) for p in earlier
        ):
            self._memo.move_to_end(key)
            return tokens
        del self._memo[key]
        log.warning("TOKENS_RELOADED %s: %s changed on disk (or a layer above it did)", key, path)
        return None

    def __call__(self, key: str) -> MoleculeTokens | None:
        """The tokens of `key`, or None (never memoized: a later build becomes visible). A
        memoized entry is served only while its file is unchanged (see `_memo_hit`)."""
        from ksmech.data.provenance import StaleArtifact
        from ksmech.orbitals.tokens import load_tokens

        with self._lock:
            hit = self._memo_hit(key)
            if hit is not None:
                return hit
            for path in self._candidates(key):
                if not self._usable(key, path):
                    continue
                sig = _signature(path)  # before the read: a later rewrite always differs
                try:
                    tokens = load_tokens(path, verify=False)  # `_usable` checked the header
                except (StaleArtifact, OSError, ValueError, KeyError) as exc:
                    log.warning("TOKENS_STALE %s: %s failed to load (%s)", key, path, exc)
                    continue
                self._attach_bond_index(key, path, tokens)
                if sig is not None:
                    self._memo[key] = (str(path), sig, tokens)
                    if len(self._memo) > self.maxsize:
                        self._memo.popitem(last=False)
                return tokens
        return None

    def _attach_bond_index(self, key: str, path: Path, tokens: MoleculeTokens) -> None:
        """The layer's Wiberg sidecar on `tokens.bond_index` (left None when there is none:
        the pair features then fail with the upstream reason, never silently)."""
        from ksmech.qm.bond_index import bond_index_path, load_bond_indices

        if tokens.bond_index is not None:
            return
        for d, bond_dir in zip(self.dirs, self.bond_dirs, strict=True):
            if bond_dir is None or Path(path).parent != d:
                continue
            side = bond_index_path(Path(bond_dir), key, tokens.conformer_id, tokens.method)
            if side.is_file():
                tokens.bond_index = load_bond_indices(side, verify=False)
            else:
                log.warning("NO_BOND_INDEX %s: no Wiberg sidecar %s", key, side)

    def digest(self, key: str) -> str | None:
        """sha256[:16] of the bytes of the token file `key` loads from; None when missing. The
        bytes, not the header's content hash: a rebuilt file (new header) is a new digest."""
        path = self.path_of(key)
        if path is None:
            return None
        st = path.stat()
        memo_key = (str(path), st.st_mtime_ns, st.st_size)
        with self._lock:
            if memo_key in self._digests:
                return self._digests[memo_key]
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        with self._lock:
            self._digests[memo_key] = h
        return h

    def forget(self, key: str | None = None) -> None:
        """Drop one key (or every key) from the memo, e.g. after its file was rebuilt."""
        with self._lock:
            if key is None:
                self._memo.clear()
            else:
                self._memo.pop(key, None)

    def drain_events(self) -> list[dict]:
        """Reason-coded events (TOKENS_STALE) recorded since the last call."""
        with self._lock:
            out, self._events = self._events, []
        return out
