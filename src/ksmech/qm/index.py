"""The cache index: one SQLite file per cache root (`<root>/index.sqlite`) that lists every QM
record, failure tombstone, token file and Wiberg sidecar under that root, read from the files'
own provenance headers (docs/decisions.md, 2026-10-27).

The files stay the source of truth; the index is a cache of their headers. `refresh` brings it
up to date incrementally (a file whose size and mtime have not changed is not read again), and
`refresh(rebuild=True)` starts from an empty table, so an index is never copied between hosts:
each checkout rebuilds it from the files it has. A file whose header cannot be read is listed
with its problem (NO_HEADER, CORRUPT_ARTIFACT, KIND_MISMATCH) and counted, never skipped.

A root is a directory holding `qm/`, `tokens/` and `bond_index/`: the engine's `data/cache`,
the web app's `data/serve`. Every root has the same schema; `stats` reads several at once. The
key of a row is the cache key (`records.cache_key`, the file stem), which joins a record with
its tombstone, its tokens and its sidecar (the view `molecule`). Nothing on the hot path reads
the index: the batch and the web app look files up by path, as before.

Version 2 (open-shell plan P4) adds the spin of each record and tombstone: `unpaired` (the
record's own; a tombstone's from its method's spin tag), `spin_source` ('rdkit' for an untagged
method, else the source `data.spin` gives the species: 'species' or 'metal') and
`spin_rule_version` (the tag's version, '+spin-v1-u<u>' -> 'spin-v1'; NULL untagged).

Version 3 (reaction-network plan, item B) adds the thermochemistry sidecars of `qm.thermo_cache`
(`thermo/<key>.npz`, kind `thermo`, with the identity columns of a record) and their tombstones
(`thermo/<key>.failed.json`, kind `thermo_failure`, the columns of a QM tombstone), and the view's
`has_thermo` and `thermo_failure_reason`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ksmech.data import provenance as prov
from ksmech.data import spin
from ksmech.qm import failures

INDEX_NAME = "index.sqlite"
INDEX_VERSION = 3  # PRAGMA user_version; another value drops and recreates the tables
BUSY_TIMEOUT_S = 30.0
TAG = re.compile(r"\+(?P<version>spin-v\d+)-u(?P<u>\d+)")  # `qm.spin`'s tag in a method
COMMIT_EVERY = 1000
# (kind, directory under the root, file suffix)
LAYOUT = (
    ("qm_record", "qm", ".npz"),
    ("qm_failure", "qm", failures.SUFFIX),
    ("tokens", "tokens", ".npz"),
    ("bond_index", "bond_index", ".npz"),
    ("thermo", "thermo", ".npz"),
    ("thermo_failure", "thermo", failures.SUFFIX),
)
RECORD_LIKE = ("qm_record", "thermo")  # npz kinds whose arrays carry a record's identity
FAILURE_KINDS = ("qm_failure", "thermo_failure")
COLUMNS = (
    "kind", "key", "path", "size", "mtime_ns", "problem", "schema_version", "content_sha256",
    "built_at", "code_git_hash", "code_dirty", "inputs", "config", "smiles", "conformer_id",
    "method", "n_atoms", "warnings", "reason", "detail", "seconds", "params", "unpaired",
    "spin_source", "spin_rule_version",
)  # fmt: skip
SCHEMA = """
CREATE TABLE artifact (
    kind TEXT NOT NULL,          -- qm_record | qm_failure | tokens | bond_index | thermo |
                                 -- thermo_failure
    key TEXT NOT NULL,           -- the cache key: sha256(smiles|conformer|method), the file stem
    path TEXT NOT NULL,          -- relative to the root
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    problem TEXT,                -- NO_HEADER | CORRUPT_ARTIFACT | KIND_MISMATCH; NULL when read
    schema_version INTEGER,
    content_sha256 TEXT,
    built_at TEXT,
    code_git_hash TEXT,
    code_dirty INTEGER,
    inputs TEXT,                 -- JSON, the header's inputs
    config TEXT,                 -- JSON, the header's config
    smiles TEXT,                 -- records, thermo sidecars, tombstones: map-free SMILES
    conformer_id INTEGER,
    method TEXT,
    n_atoms INTEGER,
    warnings TEXT,
    reason TEXT,                 -- tombstones: the reason code, detail, seconds and settings
    detail TEXT,
    seconds REAL,
    params TEXT,
    unpaired INTEGER,            -- records and tombstones: the unpaired count computed (P4)
    spin_source TEXT,            -- rdkit | species | metal (`data.spin`; rdkit when untagged)
    spin_rule_version TEXT,      -- the spin tag's version (spin-v1), NULL when untagged
    PRIMARY KEY (kind, key)
);
CREATE INDEX artifact_smiles ON artifact (smiles);
CREATE TABLE meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE VIEW molecule AS
SELECT k.key AS key,
       COALESCE(r.smiles, f.smiles) AS smiles,
       COALESCE(r.method, f.method) AS method,
       r.key IS NOT NULL AS has_record,
       f.reason AS failure_reason,
       t.key IS NOT NULL AS has_tokens,
       b.key IS NOT NULL AS has_bond_index,
       h.key IS NOT NULL AS has_thermo,
       hf.reason AS thermo_failure_reason
FROM (SELECT DISTINCT key FROM artifact WHERE kind IN ('qm_record', 'qm_failure')) AS k
LEFT JOIN artifact AS r ON r.kind = 'qm_record' AND r.key = k.key
LEFT JOIN artifact AS f ON f.kind = 'qm_failure' AND f.key = k.key
LEFT JOIN artifact AS t ON t.kind = 'tokens' AND t.key = k.key
LEFT JOIN artifact AS b ON b.kind = 'bond_index' AND b.key = k.key
LEFT JOIN artifact AS h ON h.kind = 'thermo' AND h.key = k.key
LEFT JOIN artifact AS hf ON hf.kind = 'thermo_failure' AND hf.key = k.key;
"""


def index_path(root: Path) -> Path:
    return Path(root) / INDEX_NAME


def _ensure_schema(con: sqlite3.Connection) -> None:
    if con.execute("PRAGMA user_version").fetchone()[0] == INDEX_VERSION:
        return
    con.executescript(
        "DROP VIEW IF EXISTS molecule; DROP TABLE IF EXISTS artifact; DROP TABLE IF EXISTS meta;"
    )
    con.executescript(SCHEMA)
    con.execute(f"PRAGMA user_version = {INDEX_VERSION}")


@contextmanager
def connect(root: Path) -> Iterator[sqlite3.Connection]:
    """The index of `root` (created when missing): WAL, a 30 s busy timeout, the current
    schema. Commits on normal exit, rolls back on an exception."""
    Path(root).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(index_path(root), timeout=BUSY_TIMEOUT_S)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(con)
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def _scan(directory: Path, suffix: str) -> dict[str, tuple[int, int]]:
    """key -> (size, mtime_ns) of the files of one kind. Hidden files (the temporaries of
    `provenance.atomic_write`) are not listed."""
    out: dict[str, tuple[int, int]] = {}
    try:
        entries = os.scandir(directory)
    except FileNotFoundError:
        return out
    with entries:
        for e in entries:
            if e.name.startswith(".") or not e.name.endswith(suffix) or not e.is_file():
                continue
            st = e.stat()
            out[e.name.removesuffix(suffix)] = (st.st_size, st.st_mtime_ns)
    return out


def _header_columns(header: prov.Provenance | None, kind: str) -> dict[str, Any]:
    if header is None:
        return {"problem": "NO_HEADER"}
    return dict(
        problem=None if header.kind == kind else "KIND_MISMATCH",
        schema_version=header.schema_version,
        content_sha256=header.content_sha256,
        built_at=header.built_at,
        code_git_hash=header.code_git_hash,
        code_dirty=int(header.code_dirty),
        inputs=json.dumps(header.inputs, sort_keys=True),
        config=json.dumps(header.config, sort_keys=True),
        method=header.config.get("method"),
        conformer_id=header.config.get("conformer_id"),
    )


def spin_columns(smiles: str, method: str, unpaired: int | None = None) -> dict[str, Any]:
    """The spin columns of a record or tombstone (module docstring)."""
    found = TAG.search(method or "")
    if found is None:
        return dict(unpaired=unpaired, spin_source=spin.RDKIT, spin_rule_version=None)
    try:
        source = spin.assign_smiles(smiles).source
    except ValueError:
        source = None
    tag_u = int(found.group("u"))
    return dict(
        unpaired=tag_u if unpaired is None else unpaired,
        spin_source=source,
        spin_rule_version=found.group("version"),
    )


def _record_identity(z: Any) -> dict[str, Any]:
    smiles, method = str(z["smiles"]), str(z["method"])
    return dict(
        smiles=smiles,
        conformer_id=int(z["conformer_id"]),
        method=method,
        n_atoms=len(z["atomic_numbers"]),
        warnings=str(z["warnings"]),
        **spin_columns(smiles, method, int(z["unpaired"])),
    )


def _read_npz(path: Path, kind: str) -> dict[str, Any]:
    """The columns of one npz artifact: its header and, for a QM record, its identity."""
    try:
        with np.load(path, allow_pickle=False) as z:
            text = str(z[prov.HEADER_KEY]) if prov.HEADER_KEY in z.files else None
            ident = _record_identity(z) if kind in RECORD_LIKE else {}
        header = None if text is None else prov.Provenance.from_json(text)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return {"problem": "CORRUPT_ARTIFACT"}
    return {**_header_columns(header, kind), **ident}


def _read_tombstone(path: Path, kind: str) -> dict[str, Any]:
    hit = failures.read(path)
    if hit is None:
        return {"problem": "CORRUPT_ARTIFACT"}
    return {
        **_header_columns(prov.read_header(path), kind),
        "smiles": hit.smiles,
        "conformer_id": hit.conformer_id,
        "method": hit.method,
        "reason": hit.reason,
        "detail": hit.detail,
        "seconds": hit.seconds,
        "params": json.dumps(hit.params, sort_keys=True),
        **spin_columns(hit.smiles, hit.method),
    }


def read_artifact(path: Path, kind: str) -> dict[str, Any]:
    """The index columns of one file (its location and stat apart)."""
    return _read_tombstone(path, kind) if kind in FAILURE_KINDS else _read_npz(path, kind)


def _upsert(con: sqlite3.Connection, row: dict[str, Any]) -> None:
    values = [row.get(c) for c in COLUMNS]
    marks = ", ".join("?" for _ in COLUMNS)
    con.execute(f"INSERT OR REPLACE INTO artifact ({', '.join(COLUMNS)}) VALUES ({marks})", values)


def _refresh_kind(con, root: Path, layout: tuple[str, str, str], counts: Counter) -> None:
    kind, subdir, suffix = layout
    files = _scan(Path(root) / subdir, suffix)
    known = {
        key: (size, mtime)
        for key, size, mtime in con.execute(
            "SELECT key, size, mtime_ns FROM artifact WHERE kind = ?", (kind,)
        )
    }
    gone = sorted(known.keys() - files.keys())
    con.executemany("DELETE FROM artifact WHERE kind = ? AND key = ?", [(kind, k) for k in gone])
    counts[f"{kind}:removed"] += len(gone)
    for n, (key, (size, mtime)) in enumerate(sorted(files.items()), start=1):
        if known.get(key) == (size, mtime):
            counts[f"{kind}:unchanged"] += 1
            continue
        rel = f"{subdir}/{key}{suffix}"
        cols = read_artifact(Path(root) / rel, kind)
        _upsert(
            con, {**cols, "kind": kind, "key": key, "path": rel, "size": size, "mtime_ns": mtime}
        )
        counts[f"{kind}:{'updated' if key in known else 'added'}"] += 1
        if cols.get("problem"):
            counts[f"{kind}:problem:{cols['problem']}"] += 1
        if n % COMMIT_EVERY == 0:
            con.commit()


def refresh(root: Path, rebuild: bool = False) -> Counter:
    """Bring `<root>/index.sqlite` up to date with the files under `root`; `rebuild` empties it
    first. Counts by outcome: `<kind>:added|updated|removed|unchanged`, and
    `<kind>:problem:<code>` for the files read in this call whose header is unusable."""
    counts: Counter = Counter()
    with connect(root) as con:
        if rebuild:
            con.execute("DELETE FROM artifact")
        for layout in LAYOUT:
            _refresh_kind(con, Path(root), layout, counts)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        con.execute("INSERT OR REPLACE INTO meta VALUES ('refreshed_at', ?)", (now,))
    return counts


def _rows(con: sqlite3.Connection, sql: str) -> list[tuple]:
    return [tuple(r) for r in con.execute(sql)]


def _unmatched(con: sqlite3.Connection, kind: str, other: str) -> int:
    """Rows of `kind` with no row of `other` under the same key in this root."""
    sql = (
        "SELECT COUNT(*) FROM artifact AS a WHERE a.kind = ? AND NOT EXISTS"
        " (SELECT 1 FROM artifact AS b WHERE b.kind = ? AND b.key = a.key)"
    )
    return int(con.execute(sql, (kind, other)).fetchone()[0])


def stats(root: Path) -> dict[str, Any]:
    """What the index of `root` holds: files by kind (count, bytes, problems, dirty builds),
    records and tombstones by method, failures by reason with an example, and the gaps between
    the layers (a token file may stand on a record of another root: serve's on the shared)."""
    with connect(root) as con:
        refreshed = con.execute("SELECT value FROM meta WHERE name = 'refreshed_at'").fetchone()
        n_failures = con.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'qm_failure'"
        ).fetchone()[0]
        return dict(
            refreshed_at=refreshed[0] if refreshed else None,
            kinds=_rows(
                con,
                "SELECT kind, COUNT(*), SUM(size), COUNT(problem), SUM(code_dirty)"
                " FROM artifact GROUP BY kind ORDER BY kind",
            ),
            methods=_rows(
                con,
                "SELECT kind, method, COUNT(*) FROM artifact WHERE kind IN"
                " ('qm_record', 'qm_failure', 'thermo', 'thermo_failure')"
                " GROUP BY kind, method ORDER BY kind, method",
            ),
            spin=_rows(
                con,
                "SELECT kind, spin_source, spin_rule_version, unpaired, COUNT(*) FROM artifact"
                " WHERE kind IN ('qm_record', 'qm_failure') AND spin_rule_version IS NOT NULL"
                " GROUP BY kind, spin_source, spin_rule_version, unpaired"
                " ORDER BY kind, spin_source, unpaired",
            ),
            reasons=_rows(
                con,
                "SELECT reason, COUNT(*), MIN(smiles) FROM artifact WHERE kind = 'qm_failure'"
                " GROUP BY reason ORDER BY COUNT(*) DESC, reason",
            ),
            problems=_rows(
                con,
                "SELECT kind, problem, COUNT(*) FROM artifact WHERE problem IS NOT NULL"
                " GROUP BY kind, problem ORDER BY kind, problem",
            ),
            records_without_tokens=_unmatched(con, "qm_record", "tokens"),
            tokens_without_bond_index=_unmatched(con, "tokens", "bond_index"),
            failures_with_a_record=int(n_failures) - _unmatched(con, "qm_failure", "qm_record"),
            **_thermo_stats(con),
        )


def _thermo_stats(con: sqlite3.Connection) -> dict[str, Any]:
    """The thermo sidecars' failures by reason, their warnings, and the records without one."""
    warnings: Counter = Counter()
    for (text,) in con.execute("SELECT warnings FROM artifact WHERE kind = 'thermo'"):
        warnings.update(w for w in (text or "").split(",") if w)
    return dict(
        thermo_reasons=_rows(
            con,
            "SELECT reason, COUNT(*), MIN(smiles) FROM artifact WHERE kind = 'thermo_failure'"
            " GROUP BY reason ORDER BY COUNT(*) DESC, reason",
        ),
        thermo_warnings=sorted(warnings.items(), key=lambda kv: (-kv[1], kv[0])),
        records_without_thermo=_unmatched(con, "qm_record", "thermo"),
    )
