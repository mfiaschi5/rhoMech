"""Orbital sets of the molecule workbench (worker task `wb_orbital_set`), display only.

Sets: `canonical` (the QM record's MOs) and every `orbitals.localize.METHODS` entry (IBO, the
model's localization, and Pipek-Mezey/Mulliken for comparison), each from every
`localize.STARTS` start. No Foster-Boys: tblite exposes no dipole integrals.

- **Model set** (`PMConfig()` defaults, `ibo/cholesky`): the existing token file when there is
  one (the tokens the model reads); else it is built like the other sets.
- **Other sets**: `build_molecule_tokens(rec, pm_config=PMConfig(method=..., start=...))`,
  saved with upstream `save_tokens` (provenance: the QM record as input, `token_config(pm)`)
  under `data/serve/workbench/orbitals/<sha>.npz`, keyed by the key, the QM method and content
  sha, the set, the start, `ORBITAL_SET_VERSION` and the oriented fingerprint (which covers the
  token schema, localize.py, descriptors.py and tokens.py).
- **Rows**: canonical rows from `rec.C`, energies and occupations, the frontier blocks of
  `orbitals.frontier.frontier_blocks` and Mulliken centres (`localize.mulliken_populations`);
  LMO rows from the tokens: the model-input descriptors (`token_rows.orbital_model_input`),
  centres, centroid, E_r, the spectral composition over the canonical MOs and the classifier's
  type (diagnostic). Stats: localization convergence, max |C^T S C - I| and the occupied
  projector error against the canonical occupied block.

The rows are written as `<sha>.json` next to the set tokens (the model set's key adds the token
file's digest). QM records are loaded only here, in the worker (`qm.ao` imports tblite): the
synthetic method is read without the tblite AO-layout probe of `load_record`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ksmech.serve.core.contracts import LRU, Runtime
from ksmech.serve.core.errors import Code, ServeError, jsonable
from ksmech.serve.workbench.molecule import WORKBENCH_VERSION, cache_sha, file_digest

if TYPE_CHECKING:
    from ksmech.orbitals.tokens import MoleculeTokens
    from ksmech.qm.records import QMRecord
    from ksmech.serve.core.settings import ServeSettings

log = logging.getLogger(__name__)

ORBITAL_SET_VERSION = 1
CANONICAL = "canonical"
CENTRE_MIN = 0.05  # canonical rows list atoms with at least this Mulliken population
MAX_CENTRES = 12
DEGENERACY_EV = 0.05  # orbitals.frontier.DEGENERACY_EV
NOTE_SETS = (
    "ibo/cholesky is the model's localization; pm_mulliken is the Pipek-Mezey comparison; no "
    "Foster-Boys (tblite exposes no dipole integrals)"
)
_RECORDS: LRU[tuple, QMRecord] = LRU(8)
_TOKENS: LRU[tuple, MoleculeTokens] = LRU(8)


# ---- set names (API side too: localize imports numpy and qm.records only)


def localize_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    from ksmech.orbitals.localize import METHODS, STARTS

    return tuple(METHODS), tuple(STARTS)


def model_set() -> tuple[str, str]:
    from ksmech.orbitals.localize import PMConfig

    pm = PMConfig()
    return pm.method, pm.start


def set_names() -> list[str]:
    return [CANONICAL, *localize_names()[0]]


def check_set(set_: str | None, start: str | None) -> tuple[str, str | None]:
    """(set, start) with the defaults filled in (canonical; the model start for an LMO set);
    REQUEST_INVALID for an unknown name."""
    methods, starts = localize_names()
    set_ = set_ or CANONICAL
    if set_ == CANONICAL:
        return CANONICAL, None
    if set_ not in methods:
        msg = f"set must be one of {set_names()}, got {set_!r}"
        raise ServeError(Code.REQUEST_INVALID, msg, detail={"sets": set_names()})
    start = start or model_set()[1]
    if start not in starts:
        msg = f"start must be one of {list(starts)}, got {start!r}"
        raise ServeError(Code.REQUEST_INVALID, msg, detail={"starts": list(starts)})
    return set_, start


def is_model_set(set_: str, start: str | None) -> bool:
    return (set_, start) == model_set()


def versions() -> dict[str, int | str]:
    from ksmech.serve.core.versions import source_constant, token_fingerprint
    from ksmech.serve.workbench.densities import DENSITY_VERSION

    return {
        "workbench": WORKBENCH_VERSION,
        "orbital_set": ORBITAL_SET_VERSION,
        "density": DENSITY_VERSION,
        "tokens_schema": int(source_constant("orbitals/tokens.py", "SCHEMA_VERSION")),
        "token_fingerprint": token_fingerprint(),
    }


def set_tokens_path(s: ServeSettings, key: str, method: str, qm_digest: str, set_, start) -> Path:
    from ksmech.serve.core.versions import token_fingerprint

    parts = ["orbital_set", key, method, qm_digest, set_, start, ORBITAL_SET_VERSION]
    return s.workbench_orbitals / f"{cache_sha([*parts, token_fingerprint()])}.npz"


def rows_path(
    s: ServeSettings, key: str, method: str, qm_digest: str, set_, start, token_digest
) -> Path:
    """The rows JSON of one set; `token_digest` is the model token file's (None otherwise)."""
    from ksmech.serve.core.versions import token_fingerprint

    parts = ["orbital_rows", key, method, qm_digest, set_, start, token_digest]
    parts += [ORBITAL_SET_VERSION, WORKBENCH_VERSION, token_fingerprint()]
    return s.workbench_orbitals / f"{cache_sha(parts)}.json"


# ---- worker: QM records and set tokens


def _read_synthetic(path: Path, key: str, method: str) -> QMRecord:
    """A synthetic record without `load_record`'s tblite AO-layout probe (the synthetic
    featurizer wrote it with the current layout); the header is verified as upstream does."""
    from ksmech.qm.records import QMRecord, verify_record

    verify_record(path, key, 0, method)
    cast = {"int": int, "float": float, "str": str}
    kwargs = {}
    with np.load(path, allow_pickle=False) as z:
        for f in fields(QMRecord):
            arr = z[f.name]
            kwargs[f.name] = arr if arr.ndim else cast.get(str(f.type), lambda x: x)(arr.item())
    return QMRecord(**kwargs)


def load_qm(rt: Runtime, key: str) -> tuple[QMRecord, Path, str]:
    """(record, path, content digest) of the key's QM record; MOLECULE_NOT_READY (409) when
    there is none or it is stale. Worker only."""
    from ksmech.data.provenance import StaleArtifact
    from ksmech.qm.records import load_record
    from ksmech.serve.chem.featurize import qm_npz_path
    from ksmech.serve.chem.metal_geometry import method_for
    from ksmech.serve.chem.synthetic import SYNTHETIC_METHOD

    method = method_for(key, rt.featurizer.method)  # a metal key's record: +metal-geom-v1
    path = qm_npz_path(rt.settings, key, method)
    if path is None:
        raise ServeError(Code.MOLECULE_NOT_READY, f"no QM record for {key}", status=409)
    digest = file_digest(path)
    memo = (str(path), digest)
    rec = _RECORDS.get(memo)
    if rec is None:
        try:
            if method == SYNTHETIC_METHOD:
                rec = _read_synthetic(path, key, method)
            else:
                rec = load_record(path, (key, 0, method))
        except StaleArtifact as exc:
            msg = f"the QM record of {key} is stale: {exc}"
            raise ServeError(Code.MOLECULE_NOT_READY, msg, upstream="STALE_ARTIFACT") from exc
        _RECORDS.put(memo, rec)
    return rec, path, digest


def _build_set(rt: Runtime, rec, qm_path: Path, out: Path, method: str, start: str):
    from ksmech.data import provenance as prov
    from ksmech.orbitals.localize import LocalizationError, PMConfig
    from ksmech.orbitals.tokens import build_molecule_tokens, save_tokens, token_config
    from ksmech.serve.core.settings import assert_writable

    pm = PMConfig(method=method, start=start)
    try:
        t = build_molecule_tokens(rec, pm_config=pm)
    except LocalizationError as exc:
        msg = f"{method}/{start} localization of {rec.smiles} failed: {exc}"
        raise ServeError(Code.LOCALIZATION_FAILED, msg) from exc
    assert_writable(out, rt.settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tokens(t, out, prov.file_input(qm_path), token_config(pm))
    return t


def set_tokens(rt: Runtime, key: str, set_: str, start: str) -> tuple[MoleculeTokens, str, str]:
    """(tokens, source "token_file" | "workbench", token digest or "") of an LMO set."""
    from ksmech.data.provenance import StaleArtifact
    from ksmech.orbitals.localize import PMConfig
    from ksmech.orbitals.tokens import load_tokens, token_config

    if is_model_set(set_, start):
        t = rt.loader(key)
        if t is not None:
            return t, "token_file", rt.loader.digest(key) or ""
    rec, qm_path, qm_digest = load_qm(rt, key)
    out = set_tokens_path(rt.settings, key, rt.featurizer.method, qm_digest, set_, start)
    memo = (str(out), qm_digest)
    t = _TOKENS.get(memo)
    if t is None and out.is_file():
        try:
            t = load_tokens(out, token_config(PMConfig(method=set_, start=start)))
        except StaleArtifact as exc:
            log.info("rebuilding the workbench set %s: %s", out, exc)
    if t is None:
        t = _build_set(rt, rec, qm_path, out, set_, start)
    _TOKENS.put(memo, t)
    return t, "workbench", ""


# ---- rows


def _centres(pop: np.ndarray, z: np.ndarray) -> list[dict]:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    order = [int(a) for a in np.argsort(-pop, kind="stable") if pop[a] >= CENTRE_MIN]
    return [
        {"atom": a, "element": pt.GetElementSymbol(int(z[a])), "weight": float(pop[a])}
        for a in order[:MAX_CENTRES]
    ]


def degenerate_levels(eps: np.ndarray, tol: float = DEGENERACY_EV) -> np.ndarray:
    """Level id per MO: ascending energy, a new level when an MO lies more than `tol` above
    the first MO of the current level."""
    out = np.zeros(len(eps), dtype=int)
    level, first = -1, None
    for k in np.argsort(eps, kind="stable"):
        if first is None or eps[k] - first > tol:
            level, first = level + 1, eps[k]
        out[k] = level
    return out


def canonical_labels(eps: np.ndarray, occ: np.ndarray) -> list[str]:
    """HOMO, HOMO-1, ... down the occupied MOs and LUMO, LUMO+1, ... up the virtual ones."""
    from ksmech.qm.fukui import OCC_THRESHOLD

    labels = [""] * len(eps)
    o = np.flatnonzero(occ >= OCC_THRESHOLD)
    v = np.flatnonzero(occ < OCC_THRESHOLD)
    for r, k in enumerate(o[np.argsort(-eps[o], kind="stable")]):
        labels[k] = "HOMO" if r == 0 else f"HOMO-{r}"
    for r, k in enumerate(v[np.argsort(eps[v], kind="stable")]):
        labels[k] = "LUMO" if r == 0 else f"LUMO+{r}"
    return labels


def canonical_rows(rec: QMRecord) -> list[dict]:
    from ksmech.orbitals.frontier import frontier_blocks
    from ksmech.orbitals.localize import mulliken_populations
    from ksmech.qm.fukui import OCC_THRESHOLD

    eps, occ = np.asarray(rec.orbital_energies_ev), np.asarray(rec.occupations)
    member = {
        int(k): level for level, ks in frontier_blocks(eps, occ).items() for k in np.ravel(ks)
    }
    pops = mulliken_populations(rec.C, rec.S, rec.ao_atom, rec.n_atoms)
    levels, labels = degenerate_levels(eps), canonical_labels(eps, occ)
    rows = []
    for k in range(len(eps)):
        w = np.clip(pops[:, k], 0.0, None)
        centroid = (w @ rec.coords_ang) / w.sum() if w.sum() > 0 else None
        rows.append(
            {
                "index": k,
                "kind": "occupied" if occ[k] >= OCC_THRESHOLD else "virtual",
                "occupation": float(occ[k]),
                "energy_ev": float(eps[k]),
                "label": labels[k],
                "frontier": member.get(k),
                "level": int(levels[k]),
                "centres": _centres(pops[:, k], rec.atomic_numbers),
                "centroid": None if centroid is None else [float(x) for x in centroid],
                **dict.fromkeys(("model_input", "spectral", "retained_fraction", "diagnostic")),
            }
        )
    return rows


def lmo_rows(t: MoleculeTokens) -> list[dict]:
    from ksmech.serve.chem.token_rows import orbital_diagnostic, orbital_model_input, spectral_rows

    rows = []
    for k, o in enumerate(t.orbitals):
        d = o.descriptors
        centres = [
            {"atom": int(a), "element": t.symbol(int(a)), "weight": float(w)}
            for a, w in zip(d.all_center_atoms, d.all_center_weights, strict=True)
        ]
        rows.append(
            {
                "index": k,
                "kind": str(o.kind),
                "occupation": float(d.occupation),
                "energy_ev": float(d.energy_ev),
                **dict.fromkeys(("label", "frontier", "level")),
                "centres": centres,
                "centroid": [float(x) for x in np.asarray(d.centroid).ravel()],
                "model_input": orbital_model_input(o).model_dump(mode="json"),
                "spectral": spectral_rows(d),
                "retained_fraction": float(d.retained_fraction),
                "diagnostic": orbital_diagnostic(t, o),
            }
        )
    return rows


def orthonormality_error(C: np.ndarray, S: np.ndarray) -> float:
    M = C.T @ S @ C
    return float(np.max(np.abs(M - np.eye(M.shape[0])), initial=0.0))


def projector_error(A: np.ndarray, B: np.ndarray) -> float:
    """max |A A^T - B B^T|: whether two coefficient blocks span the same space."""
    return float(np.max(np.abs(A @ A.T - B @ B.T), initial=0.0))


def set_stats(rec: QMRecord, t: MoleculeTokens | None) -> dict:
    from ksmech.qm.fukui import OCC_THRESHOLD

    occ_idx = np.flatnonzero(np.asarray(rec.occupations) >= OCC_THRESHOLD)
    C_occ = rec.C[:, occ_idx]
    if t is None:
        n_occ, n_virt = len(occ_idx), rec.C.shape[1] - len(occ_idx)
        stats = {"converged": None, "sweeps_occ": None, "sweeps_virt": None}
        stats |= {"orthonormality_error": orthonormality_error(rec.C, rec.S)}
        return stats | {"occupied_projector_error": 0.0, "n_occupied": n_occ, "n_virtual": n_virt}
    occ = [k for k, o in enumerate(t.orbitals) if str(o.kind) == "occupied"]
    C = t.C_loc
    return {
        "n_occupied": len(occ),
        "n_virtual": len(t.orbitals) - len(occ),
        "converged": bool(t.pm_converged),
        "sweeps_occ": int(t.pm_sweeps_occ),
        "sweeps_virt": int(t.pm_sweeps_virt),
        "orthonormality_error": orthonormality_error(C, rec.S),
        "occupied_projector_error": projector_error(C[:, occ], C_occ),
    }


def provenance_of(rt: Runtime, key: str, qm_digest: str, token_digest: str | None) -> dict:
    from ksmech.serve.runtime.context import NOTE_SYNTHETIC

    notes = [NOTE_SYNTHETIC] if rt.settings.chemistry == "synthetic" else []
    return {
        "chemistry": rt.settings.chemistry,
        "method": rt.featurizer.method,
        "key": key,
        "qm_digest": qm_digest,
        "token_digest": token_digest or None,
        "versions": versions(),
        "notes": notes,
    }


def orbital_set_body(rt: Runtime, key: str, set_: str, start: str | None) -> dict:
    """`OrbitalSetOut` fields except `molecule_id`."""
    from ksmech.qm.fukui import homo_index

    rec, _, qm_digest = load_qm(rt, key)
    if set_ == CANONICAL:
        t, source, token_digest = None, "canonical", ""
        rows, homo = canonical_rows(rec), homo_index(np.asarray(rec.occupations))
    else:
        t, source, token_digest = set_tokens(rt, key, set_, start)
        rows, homo = lmo_rows(t), None
    methods, starts = localize_names()
    return {
        "key": key,
        "set": set_,
        "start": start,
        "is_model_set": is_model_set(set_, start),
        "source": source,
        "homo_index": homo,
        "orbitals": rows,
        "stats": set_stats(rec, t),
        "sets": [CANONICAL, *methods],
        "starts": list(starts),
        "model_set": list(model_set()),
        "notes": [NOTE_SETS],
        "provenance": provenance_of(rt, key, qm_digest, token_digest),
    }


def t_wb_orbital_set(rt: Runtime, payload: dict) -> dict:
    """{"key", "set", "start"} -> {"path"}: the rows JSON of one orbital set (cached)."""
    from ksmech.serve.chem.pools import atomic_write_text

    key = str(payload["key"])
    set_, start = check_set(payload.get("set"), payload.get("start"))
    _, _, qm_digest = load_qm(rt, key)
    token_digest = rt.loader.digest(key) if is_model_set(set_, start) else None
    path = rows_path(rt.settings, key, rt.featurizer.method, qm_digest, set_, start, token_digest)
    if not path.is_file():
        body = orbital_set_body(rt, key, set_, start)
        atomic_write_text(path, json.dumps(jsonable(body), sort_keys=True) + "\n", rt.settings)
    return {"path": str(path)}
