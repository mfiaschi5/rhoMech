"""A layered, thread-safe token loader for a long-running server (adapter A3).

`arrows.reactants.TokenCache` reads one directory, memoizes misses (a molecule built while the
server runs stays "missing") and is not thread-safe. `LayeredTokenLoader` looks in each
directory in order (the shared cache first, then the serve-owned one), skips a file written by
another token schema (counted as `TOKENS_STALE`, logged once per key), memoizes only hits,
and guards everything with a lock. Duplicate species get the same `MoleculeTokens` object, as
with `TokenCache`, while their file is unchanged: a memo entry keeps the file's path,
`st_mtime_ns` and `st_size` (taken before the read), and a hit is dropped and reloaded when
that file changed or vanished, or when an earlier layer gained a usable file (a token rebuild
while the server runs is never served from the memo).
"""

from __future__ import annotations

import hashlib
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


def _schema_ok(path: Path) -> bool:
    """The token file's schema version is the current one (reads one small member)."""
    import numpy as np

    from ksmech.orbitals.tokens import SCHEMA_VERSION

    with np.load(path, allow_pickle=False) as z:
        return int(z["schema_version"].item()) == SCHEMA_VERSION


class LayeredTokenLoader:
    def __init__(self, dirs: Sequence[Path], method: str, maxsize: int = 256):
        self.dirs = [Path(d) for d in dirs]
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

    def _usable(self, key: str, path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return False
            if _schema_ok(path):
                return True
            reason = "another token schema"
        except (OSError, ValueError, KeyError) as exc:
            reason = f"unreadable ({type(exc).__name__}: {exc})"
        with self._lock:
            if self.stale.get(key) != str(path):
                self.stale[key] = str(path)
                self._events.append({"code": "TOKENS_STALE", "key": key, "path": str(path)})
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
        from ksmech.orbitals.tokens import TokenSchemaError, load_tokens

        with self._lock:
            hit = self._memo_hit(key)
            if hit is not None:
                return hit
            for path in self._candidates(key):
                if not self._usable(key, path):
                    continue
                sig = _signature(path)  # before the read: a later rewrite always differs
                try:
                    tokens = load_tokens(path)
                except (TokenSchemaError, OSError, ValueError, KeyError) as exc:
                    log.warning("TOKENS_STALE %s: %s failed to load (%s)", key, path, exc)
                    continue
                if sig is not None:
                    self._memo[key] = (str(path), sig, tokens)
                    if len(self._memo) > self.maxsize:
                        self._memo.popitem(last=False)
                return tokens
        return None

    def digest(self, key: str) -> str | None:
        """sha256[:16] of the bytes of the token file `key` loads from; None when missing."""
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
