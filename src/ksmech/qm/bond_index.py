"""Wiberg bond indices of the Lewis bonds of a QM record (docs/decisions.md 2026-10-23).

WBI_ab = sum over mu on a, nu on b of (S^1/2 P S^1/2)_{mu nu}^2, with P = C diag(n) C^T the
record's total density (fractional occupations included) in the Loewdin-orthogonalized AO
basis; for H2 in a minimal basis it is 1. It says how much of a drawn bond the wavefunction
still holds: a leaving group the record has already half released (an alkoxide next to C-Cl,
a bromonium C-Br+) has a low index, even where no localized orbital spans the bond.

The indices are kept per molecule in a sidecar next to the QM cache
(`data/cache/bond_index/<cache key>.npz`: the record's Lewis bonds as (k, 2) record atom
indices and their WBI), stamped with the QM record as input. They are not part of the token
files, whose content hash keys the contact-block cache.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ksmech.data import provenance as prov
from ksmech.qm.embed import prepare_mol
from ksmech.qm.records import QMRecord, cache_key, load_record

log = logging.getLogger(__name__)

BOND_INDEX_VERSION = 1
DEFAULT_BOND_INDEX_DIR = Path("data/cache/bond_index")


def wiberg_matrix(S: np.ndarray, P: np.ndarray, ao_atom: np.ndarray, n_atoms: int) -> np.ndarray:
    """(n_atoms, n_atoms) Wiberg indices from the AO overlap S and total density P."""
    w, U = np.linalg.eigh(S)
    half = (U * np.sqrt(np.clip(w, 0.0, None))) @ U.T
    Q = half @ P @ half
    onehot = np.zeros((len(ao_atom), n_atoms))
    onehot[np.arange(len(ao_atom)), ao_atom] = 1.0
    W = onehot.T @ (Q * Q) @ onehot
    np.fill_diagonal(W, 0.0)
    return W


def lewis_bond_indices(record: QMRecord) -> tuple[np.ndarray, np.ndarray]:
    """((k, 2) record atom indices of every Lewis bond of the record, (k,) their WBI)."""
    mol = prepare_mol(record.atom_order_smiles)  # the record's atom order, as the tokens'
    if mol.GetNumAtoms() != record.n_atoms:
        raise ValueError(f"{record.smiles}: Lewis structure has {mol.GetNumAtoms()} atoms")
    P = (record.C * np.asarray(record.occupations)) @ record.C.T
    W = wiberg_matrix(record.S, P, np.asarray(record.ao_atom), record.n_atoms)
    pairs = np.array(
        sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()),
        dtype=np.int32,
    ).reshape(-1, 2)
    return pairs, W[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.zeros(0)


def bond_index_path(out_dir: Path, smiles: str, conformer_id: int, method: str) -> Path:
    return Path(out_dir) / f"{cache_key(smiles, conformer_id, method)}.npz"


def as_dict(pairs: np.ndarray, wbi: np.ndarray) -> dict[tuple[int, int], float]:
    return {(int(i), int(j)): float(w) for (i, j), w in zip(pairs, wbi, strict=True)}


def save_bond_indices(record: QMRecord, record_path: Path, out_dir: Path) -> Path:
    pairs, wbi = lewis_bond_indices(record)
    arrays = {"pairs": pairs, "wbi": wbi}
    header = prov.stamp(
        "bond_index",
        BOND_INDEX_VERSION,
        prov.file_input(record_path),
        {"method": record.method, "conformer_id": int(record.conformer_id)},
        prov.array_digest(arrays),
    )
    path = bond_index_path(out_dir, record.smiles, record.conformer_id, record.method)
    path.parent.mkdir(parents=True, exist_ok=True)
    return prov.save_npz(path, arrays, header)


def load_bond_indices(path: Path, verify: bool = True) -> dict[tuple[int, int], float]:
    """Lewis bond (i < j, record atom indices) -> WBI. StaleArtifact when its record changed."""
    if verify:
        prov.verify(path, "bond_index", BOND_INDEX_VERSION)
    with np.load(path, allow_pickle=False) as z:
        return as_dict(z["pairs"], z["wbi"])


def build_bond_indices(qm_dir: Path, out_dir: Path = DEFAULT_BOND_INDEX_DIR) -> Counter:
    """A sidecar for every QM record under `qm_dir` that has none or a stale one; counts by
    outcome (every failure logged with its reason)."""
    counts: Counter = Counter()
    for path in sorted(Path(qm_dir).glob("*.npz")):
        try:
            record = load_record(path)
            out = bond_index_path(out_dir, record.smiles, record.conformer_id, record.method)
            if out.exists():
                try:
                    prov.verify(out, "bond_index", BOND_INDEX_VERSION)
                    counts["up_to_date"] += 1
                    continue
                except prov.StaleArtifact:
                    pass
            save_bond_indices(record, path, out_dir)
            counts["built"] += 1
        except (prov.StaleArtifact, ValueError, KeyError) as exc:
            reason = type(exc).__name__
            log.warning("%s: no bond indices (%s: %s)", path.name, reason, exc)
            counts[f"failed_{reason}"] += 1
    return counts
