"""Pool states: build them from user input, key them, persist them (adapter A2).

A pool state is a Kekule, explicit-H SMILES with a map number on every atom
(`chem.canonical_mapped_smiles`). `apply_predicted_arrows` cannot use implicit H as sites or as
leaving atoms, and map numbers are the only atom identity that survives a step. Every state the
app returns is saved under `data/serve/pools/<state_id>/state.json` (content-addressed, atomic,
idempotent) before it becomes a network node.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase

from ksmech.serve import keys
from ksmech.serve.contracts import Limits, PoolState, SpeciesCount
from ksmech.serve.errors import Code, ServeError, error_body
from ksmech.serve.settings import ServeSettings, assert_writable

if TYPE_CHECKING:
    from ksmech.serve.featurize import Featurizer

log = logging.getLogger(__name__)

STATE_ID_RE = re.compile(r"^s[0-9a-f]{20}$")


@dataclass(frozen=True)
class SpeciesInput:
    smiles: str
    count: int


@dataclass
class Preflight:
    species: list[dict]  # SpeciesPreflightOut fields, one per fragment of each input field
    state: PoolState | None
    error: ServeError | None
    notes: list[str] = field(default_factory=list)


def _err(code: Code, message: str, i: int | None = None) -> ServeError:
    return ServeError(code, message, status=422, detail=None if i is None else {"field": i})


def _parse_input(smiles: str, i: int) -> Chem.Mol:
    """The user's SMILES of field i: parsed, radical-free, atom maps dropped."""
    if not smiles or not smiles.strip():
        raise _err(Code.EMPTY_INPUT, f"field {i} is empty", i)
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise _err(Code.INVALID_SMILES, f"RDKit cannot parse {smiles!r}", i)
    if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
        raise _err(Code.RADICAL_NOT_SUPPORTED, f"{smiles!r} has radical electrons", i)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def _record_mol(frag: Chem.Mol, i: int) -> tuple[str, Chem.Mol]:
    """(token key, `prepare_mol(key)`) of one input fragment; checks the spin parity."""
    from ksmech.data import chem
    from ksmech.qm.embed import prepare_mol
    from ksmech.qm.records import QMError
    from ksmech.qm.xtb_calc import resolve_charge_spin

    try:
        key = chem.unmapped_canonical(frag)
        rec = prepare_mol(key)
    except (chem.SanitizeError, ValueError, QMError) as exc:
        raise _err(Code.INVALID_SMILES, str(exc), i) from exc
    try:
        _, unpaired = resolve_charge_spin(rec)
    except QMError as exc:
        raise _err(Code.SPIN_PARITY, f"{key}: {exc.detail}", i) from exc
    if unpaired:
        raise _err(Code.RADICAL_NOT_SUPPORTED, f"{key} has {unpaired} unpaired electrons", i)
    return key, rec


def _check_counts(inputs: Sequence[SpeciesInput], limits: Limits) -> None:
    if not inputs:
        raise _err(Code.EMPTY_INPUT, "no reactant given")
    if len(inputs) > limits.max_species:
        raise _err(Code.LIMIT_EXCEEDED, f"{len(inputs)} species > {limits.max_species}")
    for i, inp in enumerate(inputs):
        if not 1 <= int(inp.count) <= limits.max_count:
            msg = f"count {inp.count} outside 1..{limits.max_count}"
            raise _err(Code.LIMIT_EXCEEDED, msg, i)


def initial_state(inputs: list[SpeciesInput], limits: Limits) -> PoolState:
    """The root state: each fragment of each input in its record atom order (`prepare_mol`,
    all H explicit), repeated `count` times, combined, kekulized, every atom mapped 1..N."""
    from ksmech.data import chem

    _check_counts(inputs, limits)
    records: list[Chem.Mol] = []
    for i, inp in enumerate(inputs):
        for frag in chem.fragments(_parse_input(inp.smiles, i)):
            _, rec = _record_mol(frag, i)
            records += [Chem.Mol(rec) for _ in range(int(inp.count))]
    mol = records[0]
    for rec in records[1:]:
        mol = Chem.CombineMols(mol, rec)
    n = mol.GetNumAtoms()
    if n > limits.max_atoms:
        raise _err(Code.LIMIT_EXCEEDED, f"{n} atoms > {limits.max_atoms}")
    mol = Chem.RWMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return state_from_mapped(chem.canonical_mapped_smiles(mol))


def _check_mapped(mol: Chem.Mol, smiles: str) -> None:
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if 0 in maps or len(set(maps)) != len(maps):
        raise ServeError(Code.INVALID_SMILES, f"not every atom has its own map number: {smiles}")
    implicit = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs(includeNeighbors=False)]
    if implicit:
        msg = f"atoms {implicit} carry implicit H (every H must be a mapped atom): {smiles}"
        raise ServeError(Code.IMPLICIT_H_IN_POOL, msg)


def _species(frags: list[keys.FragmentKey]) -> tuple[SpeciesCount, ...]:
    groups: dict[str, list[keys.FragmentKey]] = defaultdict(list)
    for f in frags:
        groups[f.species_key].append(f)
    out = []
    for key in sorted(groups):
        members = groups[key]
        flags = sorted({f.flag for f in members} - {"OK"}) or ["OK"]
        token_keys = tuple(sorted({f.token_key for f in members}))
        out.append(SpeciesCount(key, token_keys, len(members), members[0].charge, flags[0]))
    return tuple(out)


def state_from_mapped(mapped_smiles: str) -> PoolState:
    """The state of a fully mapped, explicit-H SMILES (re-canonicalized). Fragment order is
    `GetMolFrags` of `parse_mapped(state.mapped_smiles)`, the order of `ReactantTokens`."""
    from ksmech.data import chem

    try:
        canonical = chem.canonical_mapped_smiles(chem.parse_mapped(mapped_smiles))
        mol = chem.parse_mapped(canonical)
    except (chem.SmilesParseError, chem.SanitizeError) as exc:
        raise ServeError(Code.INVALID_SMILES, str(exc)) from exc
    _check_mapped(mol, canonical)
    frags = keys.fragment_keys(mol)
    pool_key = keys.pool_key_of(f.species_key for f in frags)
    return PoolState(
        state_id=keys.state_id(canonical),
        mapped_smiles=canonical,
        pool_key=pool_key,
        pool_id=keys.pool_id(pool_key),
        species=_species(frags),
        fragment_token_keys=tuple(f.token_key for f in frags),
        n_atoms=mol.GetNumAtoms(),
    )


def state_from_mol(mol: Chem.Mol) -> PoolState:
    """The state of a mapped mol (a step product keeps its parent's maps)."""
    from ksmech.data import chem

    return state_from_mapped(chem.canonical_mapped_smiles(mol))


def state_path(state_id: str, s: ServeSettings) -> Path:
    """`data/serve/pools/<state_id>/state.json`. A malformed id is POOL_NOT_FOUND (404), never
    a file access."""
    if not STATE_ID_RE.match(str(state_id)):
        raise ServeError(Code.POOL_NOT_FOUND, f"malformed state id {state_id!r}", status=404)
    return s.pools / state_id / "state.json"


def tmp_sibling(path: Path) -> Path:
    """A tmp file next to `path`, unique per process, thread and call (two writers of one
    file never share a tmp name)."""
    tag = f"{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}"
    return path.with_name(f".{path.name}.{tag}.tmp")


def atomic_write_text(path: Path, text: str, s: ServeSettings) -> Path:
    """Write `text` atomically (own tmp file in the same directory, fsync, os.replace; the tmp
    file is removed when the write fails)."""
    path = Path(path)
    assert_writable(path, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_sibling(path)
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    return path


def save_state(state: PoolState, s: ServeSettings) -> Path:
    """Persist a state (atomic; a no-op when the same content is already there)."""
    path = state_path(state.state_id, s)
    text = json.dumps(state.to_dict(), sort_keys=True, indent=1) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    return atomic_write_text(path, text, s)


def load_state(state_id: str, s: ServeSettings) -> PoolState:
    """A saved state. Raises ServeError(POOL_NOT_FOUND, 404)."""
    path = state_path(state_id, s)
    try:
        state = PoolState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ServeError(Code.POOL_NOT_FOUND, f"no pool state {state_id}", status=404) from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        msg = f"unreadable pool state {state_id}: {exc}"
        raise ServeError(Code.POOL_NOT_FOUND, msg, status=404) from exc
    if state.state_id != state_id:
        raise ServeError(Code.CONSISTENCY_FAILED, f"{path} holds state {state.state_id}")
    return state


def _species_row(i: int, smiles: str, count: int, key: str | None, featurizer) -> dict:
    row = {
        "field": i,
        "input": smiles,
        "token_key": key,
        "species_key": None,
        "count": int(count),
        "charge": None,
        "multiplicity": None,
        "n_atoms": None,
        "tokens_cached": False,
        "qm_cached": False,
        "errors": [],
    }
    if key is None:
        return row
    try:
        row["species_key"] = keys.species_key_of(key)[0]
    except ServeError as exc:
        row["errors"].append(error_body(exc.code, exc.message, detail={"field": i}))
    sp = featurizer.preflight(key)
    row.update(charge=sp.charge, multiplicity=sp.multiplicity, n_atoms=sp.n_atoms)
    row.update(tokens_cached=sp.tokens_cached, qm_cached=sp.qm_cached)
    row["errors"] += [error_body(c, m, detail={"field": i}) for c, m in sp.errors]
    return row


def _field_rows(i: int, inp: SpeciesInput, featurizer, notes: list[str]) -> list[dict]:
    from ksmech.data import chem

    try:
        with rdBase.BlockLogs():
            raw = Chem.MolFromSmiles(inp.smiles.strip()) if inp.smiles else None
        if raw is not None and any(a.GetAtomMapNum() for a in raw.GetAtoms()):
            notes.append(f"field {i}: atom maps in the input were dropped")
        mol = _parse_input(inp.smiles, i)
    except ServeError as exc:
        row = _species_row(i, inp.smiles, inp.count, None, featurizer)
        row["errors"].append(exc.body())
        return [row]
    rows = []
    for frag in chem.fragments(mol):
        try:
            key = chem.unmapped_canonical(frag)
        except (chem.SanitizeError, ValueError) as exc:
            row = _species_row(i, inp.smiles, inp.count, None, featurizer)
            row["errors"].append(error_body(Code.INVALID_SMILES, str(exc), detail={"field": i}))
            rows.append(row)
            continue
        rows.append(_species_row(i, inp.smiles, inp.count, key, featurizer))
    return rows


def preflight_inputs(
    inputs: Sequence[SpeciesInput], featurizer: Featurizer, limits: Limits
) -> Preflight:
    """The shared body of /normalize, /pools and Chemistry.root: one row per fragment of each
    input field, the first error, and the root state when there is none. Never computes."""
    notes: list[str] = []
    try:
        _check_counts(inputs, limits)
    except ServeError as exc:
        if not inputs or len(inputs) > limits.max_species:
            return Preflight([], None, exc, notes)
        first_count_error: ServeError | None = exc
    else:
        first_count_error = None
    species = [r for i, inp in enumerate(inputs) for r in _field_rows(i, inp, featurizer, notes)]
    error = first_count_error
    for row in species:
        if error is None and row["errors"]:
            e = row["errors"][0]
            error = ServeError(e["code"], e["message"], status=422, detail={"field": row["field"]})
    state = None
    if error is None:
        try:
            state = initial_state(list(inputs), limits)
        except ServeError as exc:
            error = exc
    return Preflight(species, state, error, notes)
