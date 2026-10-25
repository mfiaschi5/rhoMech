"""Molecule workbench, API side (RDKit only): input, identity, registry, sketch mapping, status.

- **Input** (`parse_input`): exactly one of a V2000/V3000 molblock (read with explicit H kept,
  so stereo comes from the wedges) or a SMILES. One connected molecule (`MULTIPLE_FRAGMENTS`),
  no query atoms or bonds and nothing RDKit cannot read (`INVALID_MOLFILE` / `INVALID_SMILES`).
- **Key**: `data.chem.unmapped_canonical(mol)`, the QM / token cache key, so a drawn molecule
  shares its QM record and tokens with the pools.
- **Checks**: `featurizer.preflight(key)` (radicals, spin parity, unsupported elements) and the
  atom limit (`settings.max_atoms`, H included).
- **Id**: `"m" + sha256("molecule-v1|" + key)[:20]`; the registry entry
  `data/serve/workbench/molecules/<id>.json` holds the key, the chemistry method, the last
  featurize failure and the first / last time it was posted.
- **Sketch mapping** (`sketch_mapping`): `prepare_mol(key)` (the record's atom order) matched
  against `AddHs(drawn)` with chirality, then without. `sketch_to_record[i]` is the record atom
  of sketch atom i; `record_to_sketch[r]` the sketch atom of record atom r (None for an
  implicit H); `record_parent_sketch[r]` the same with an implicit H sent to its heavy atom. No
  match is `mapping = None` with `mapping_error = MAPPING_FAILED`.
- **Status** (`status_of`): ready (a usable token file), running (a featurize task with this
  key is queued or running), qm_only (a QM record but no tokens, e.g. cached chemistry), failed
  (the last featurize under this method failed), missing.

`create_molecule` submits the existing `featurize` task (QM + tokens) and waits at most
`interactive_wait_s`; it never runs the pair analysis. Display only: `WORKBENCH_VERSION` keys
the workbench caches and never enters the physics fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase

from ksmech.serve import schemas as S
from ksmech.serve.errors import Code, Pending, ServeError, error_body, jsonable
from ksmech.serve.featurize import featurize_label, qm_npz_path
from ksmech.serve.pools import atomic_write_text
from ksmech.serve.worker import task_key

if TYPE_CHECKING:
    from ksmech.serve.context import ApiContext
    from ksmech.serve.settings import ServeSettings

log = logging.getLogger(__name__)

WORKBENCH_VERSION = 1
ID_SALT = "molecule-v1|"
FEATURIZE_EXTRA_S = 120
_DIGEST_LOCK = threading.Lock()
_DIGESTS: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class ParsedInput:
    mol: Chem.Mol  # the drawn molecule, sketch atom order (explicit H atoms of a molblock kept)
    key: str
    source: str  # "molblock" | "smiles"


def molecule_id(key: str) -> str:
    return "m" + hashlib.sha256((ID_SALT + key).encode()).hexdigest()[:20]


def cache_sha(parts: list) -> str:
    """sha256[:32] of the JSON of a cache key (the file name of a workbench artifact)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def file_digest(path: Path) -> str:
    """sha256[:16] of a file's bytes, memoized on (path, mtime, size): the QM content sha the
    workbench caches are keyed by (a rebuilt record is a new digest)."""
    st = Path(path).stat()
    memo = (str(path), st.st_mtime_ns, st.st_size)
    with _DIGEST_LOCK:
        hit = _DIGESTS.get(memo)
    if hit is None:
        hit = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        with _DIGEST_LOCK:
            _DIGESTS[memo] = hit
    return hit


# ---- input


def _exactly_one(molblock: str | None, smiles: str | None) -> tuple[str, str]:
    given = [(k, v) for k, v in (("molblock", molblock), ("smiles", smiles)) if v is not None]
    if len(given) != 1:
        msg = "give exactly one of molblock and smiles"
        raise ServeError(Code.REQUEST_INVALID, msg, detail={"given": [k for k, _ in given]})
    kind, text = given[0]
    if not text.strip():
        raise ServeError(Code.EMPTY_INPUT, f"the {kind} is empty", detail={"field": kind})
    return kind, text


def _read(kind: str, text: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        if kind == "smiles":
            mol = Chem.MolFromSmiles(text.strip())
        else:
            mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
    if mol is None:
        code = Code.INVALID_SMILES if kind == "smiles" else Code.INVALID_MOLFILE
        raise ServeError(code, f"RDKit cannot read the {kind}", detail={"field": kind})
    return mol


def _check_structure(mol: Chem.Mol, kind: str) -> None:
    queries = [a.GetIdx() for a in mol.GetAtoms() if a.HasQuery() or a.GetAtomicNum() == 0]
    if queries or any(b.HasQuery() for b in mol.GetBonds()):
        msg = f"query atoms or bonds are not molecules (atoms {queries})"
        raise ServeError(Code.INVALID_MOLFILE, msg, detail={"field": kind, "atoms": queries})
    if mol.GetNumAtoms() == 0:
        raise ServeError(Code.EMPTY_INPUT, f"the {kind} has no atoms", detail={"field": kind})
    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        msg = f"{len(frags)} fragments: draw one molecule"
        detail = {"field": kind, "fragments": [list(f) for f in frags]}
        raise ServeError(Code.MULTIPLE_FRAGMENTS, msg, detail=detail)


def parse_input(molblock: str | None = None, smiles: str | None = None) -> ParsedInput:
    """The drawn molecule and its cache key (see the module docstring for the errors)."""
    from ksmech.data import chem

    kind, text = _exactly_one(molblock, smiles)
    mol = _read(kind, text)
    _check_structure(mol, kind)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        key = chem.unmapped_canonical(mol)
    except Exception as exc:  # noqa: BLE001 - chem.SanitizeError and RDKit errors alike
        code = Code.INVALID_SMILES if kind == "smiles" else Code.INVALID_MOLFILE
        raise ServeError(code, f"no canonical SMILES: {exc}", detail={"field": kind}) from exc
    return ParsedInput(mol, key, kind)


# ---- sketch <-> record mapping


def _match(record: Chem.Mol, query: Chem.Mol) -> tuple[tuple[int, ...], bool]:
    n = record.GetNumAtoms()
    for chiral in (True, False):
        with rdBase.BlockLogs():
            m = record.GetSubstructMatch(query, useChirality=chiral)
        if len(m) == n == query.GetNumAtoms():
            return tuple(int(x) for x in m), chiral
    return (), False


def sketch_mapping(drawn: Chem.Mol, key: str) -> dict | None:
    """`SketchMappingOut` fields, or None when the drawing does not match the record."""
    from ksmech.qm.embed import prepare_mol

    record = prepare_mol(key)
    query = Chem.AddHs(drawn)
    match, chiral = _match(record, query)
    if not match:
        return None
    n_sketch, n_record = drawn.GetNumAtoms(), record.GetNumAtoms()
    to_sketch: list[int | None] = [None] * n_record
    parent: list[int | None] = [None] * n_record
    for q, r in enumerate(match):
        if q < n_sketch:
            to_sketch[r] = parent[r] = q
        else:  # an H that AddHs appended: its one neighbour is a drawn atom
            nbrs = [nb.GetIdx() for nb in query.GetAtomWithIdx(q).GetNeighbors()]
            parent[r] = nbrs[0] if nbrs and nbrs[0] < n_sketch else None
    return {
        "n_sketch_atoms": n_sketch,
        "n_record_atoms": n_record,
        "sketch_to_record": list(match[:n_sketch]),
        "record_to_sketch": to_sketch,
        "record_parent_sketch": parent,
        "chirality_matched": chiral,
    }


# ---- registry


def registry_path(s: ServeSettings, mid: str) -> Path:
    return s.workbench_molecules / f"{mid}.json"


def load_entry(s: ServeSettings, mid: str) -> dict:
    """The registry entry of a molecule id; MOLECULE_NOT_FOUND (404) when it was never posted."""
    path = registry_path(s, mid)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ServeError(Code.MOLECULE_NOT_FOUND, f"no molecule {mid}", status=404) from None
    except (OSError, ValueError) as exc:
        msg = f"unreadable molecule entry {path}: {exc}"
        raise ServeError(Code.CONSISTENCY_FAILED, msg) from exc


def save_entry(s: ServeSettings, entry: dict) -> None:
    text = json.dumps(jsonable(entry), sort_keys=True, indent=1) + "\n"
    atomic_write_text(registry_path(s, entry["molecule_id"]), text, s)


def register(s: ServeSettings, key: str, method: str) -> dict:
    """The entry of `key` (created on first post); `last_posted` is refreshed."""
    mid = molecule_id(key)
    try:
        entry = load_entry(s, mid)
    except ServeError as exc:
        if exc.code != Code.MOLECULE_NOT_FOUND:
            raise
        entry = {"molecule_id": mid, "key": key, "first_posted": time.time(), "failure": None}
    if entry.get("key") != key:
        raise ServeError(Code.CONSISTENCY_FAILED, f"molecule {mid} holds {entry.get('key')!r}")
    entry |= {"method": method, "last_posted": time.time(), "version": WORKBENCH_VERSION}
    save_entry(s, entry)
    return entry


def record_failure(s: ServeSettings, entry: dict, exc: ServeError, method: str) -> None:
    from ksmech.serve.settings import append_failure

    entry["failure"] = error_body(exc.code, exc.message, upstream=exc.upstream, detail=exc.detail)
    entry["failure"] |= {"method": method, "time": time.time()}
    save_entry(s, entry)
    append_failure(s, str(exc.code), exc.message, key=entry["key"], where="workbench")


# ---- status


def featurize_payload(key: str) -> dict:
    return {"key": key}


def current_failure(ctx: ApiContext, entry: dict) -> dict | None:
    f = entry.get("failure")
    return f if f and f.get("method") == ctx.featurizer.method else None


def status_of(ctx: ApiContext, entry: dict) -> str:
    key = entry["key"]
    if ctx.loader.has(key):
        return "ready"
    if ctx.scheduler.find(task_key("featurize", featurize_payload(key))) is not None:
        return "running"
    if qm_npz_path(ctx.settings, key, ctx.featurizer.method) is not None:
        return "qm_only"
    return "failed" if current_failure(ctx, entry) else "missing"


def run_featurize(ctx: ApiContext, entry: dict) -> None:
    """The `featurize` task of the entry's key (`Pending` while it runs); a failure is recorded
    in the entry and raised again."""
    key = entry["key"]
    label = featurize_label(ctx.featurizer, key)
    timeout = ctx.settings.qm_timeout_s + FEATURIZE_EXTRA_S
    try:
        ctx.interactive("featurize", featurize_payload(key), label=label, timeout_s=timeout)
    except Pending:
        raise
    except ServeError as exc:
        record_failure(ctx.settings, entry, exc, ctx.featurizer.method)
        raise
    if entry.get("failure"):
        entry["failure"] = None
        save_entry(ctx.settings, entry)


# ---- services


def _preflight(ctx: ApiContext, key: str):
    pf = ctx.featurizer.preflight(key)
    if pf.errors:
        code, msg = pf.errors[0]
        detail = {"key": key, "errors": [{"code": c, "message": m} for c, m in pf.errors]}
        raise ServeError(code, msg, status=422, detail=detail)
    limit = ctx.settings.max_atoms
    if pf.n_atoms is not None and pf.n_atoms > limit:
        msg = f"{key} has {pf.n_atoms} atoms (H included); the limit is {limit}"
        detail = {"key": key, "n_atoms": pf.n_atoms, "max_atoms": limit}
        raise ServeError(Code.LIMIT_EXCEEDED, msg, status=422, detail=detail)
    return pf


def create_molecule(ctx: ApiContext, body: S.MoleculeIn) -> S.MoleculeOut:
    """POST /molecule: parse, check, register, map the sketch, and submit `featurize` when the
    tokens are missing (a recorded failure is retried only with `retry`). Returns the molecule
    with its status: ready when the task finished within the interactive wait, else running."""
    parsed = parse_input(body.molblock, body.smiles)
    _preflight(ctx, parsed.key)
    entry = register(ctx.settings, parsed.key, ctx.featurizer.method)
    mapping = sketch_mapping(parsed.mol, parsed.key)
    if _should_featurize(ctx, entry, status_of(ctx, entry), body.retry):
        try:
            run_featurize(ctx, entry)
        except Pending:
            pass
        except ServeError as exc:
            log.info("featurize %s failed [%s] %s", parsed.key, exc.code, exc.message)
    return molecule_out(ctx, entry["molecule_id"], mapping=mapping, posted=True)


def _error_out(failure: dict) -> S.ErrorBody:
    body = {k: failure.get(k) for k in ("code", "message", "upstream", "detail")}
    return S.ErrorBody(**body, request_id="")


def _should_featurize(ctx: ApiContext, entry: dict, status: str, retry: bool) -> bool:
    """Missing tokens are featurized once; a recorded failure only again with `retry`."""
    if status in ("ready", "running"):
        return False
    return retry or current_failure(ctx, entry) is None


def molecule_out(
    ctx: ApiContext, mid: str, *, mapping: dict | None = None, posted: bool = False
) -> S.MoleculeOut:
    """GET /molecule/{id}: the entry and its status (never computes). `mapping` is the sketch
    mapping of the drawing just posted (None on a GET: the drawing is the client's)."""
    entry = load_entry(ctx.settings, mid)
    key = entry["key"]
    pf = ctx.featurizer.preflight(key)
    status = status_of(ctx, entry)
    failure = current_failure(ctx, entry) if status in ("failed", "qm_only") else None
    warnings = [f"TOKENS_STALE: {key}"] if key in ctx.loader.stale else []
    return S.MoleculeOut(
        molecule_id=mid,
        key=key,
        status=status,
        error=None if failure is None else _error_out(failure),
        n_atoms=pf.n_atoms,
        charge=pf.charge,
        multiplicity=pf.multiplicity,
        chemistry=ctx.chemistry,
        method=ctx.featurizer.method,
        qm_ready=qm_npz_path(ctx.settings, key, ctx.featurizer.method) is not None,
        tokens_ready=status == "ready",
        mapping=None if mapping is None else S.SketchMappingOut(**mapping),
        mapping_error=str(Code.MAPPING_FAILED) if posted and mapping is None else None,
        warnings=warnings,
    )
