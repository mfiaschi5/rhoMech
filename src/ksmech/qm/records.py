"""`QMRecord`: everything one GFN2-xTB calculation produces for one molecule and conformer.

Units: coordinates in Angstrom, orbital energies in eV (suffix `_ev`), total energy and the
H / Fock matrices in Hartree. Atom order is the order of `Chem.AddHs(Chem.MolFromSmiles(
atom_order_smiles))`; AO indices run atom by atom (see `ksmech.qm.ao`).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/cache/qm")


class ReasonCode(StrEnum):
    """Why a molecule was not featurized (failures) or what degraded (warnings)."""

    CANONICALIZE_FAILED = "CANONICALIZE_FAILED"
    EMBED_FAILED = "EMBED_FAILED"
    SPIN_PARITY = "SPIN_PARITY"
    OPT_NOT_CONVERGED = "OPT_NOT_CONVERGED"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    CONNECTIVITY_CHANGED = "CONNECTIVITY_CHANGED"
    XTB_NOT_FOUND = "XTB_NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    WORKER_CRASH = "WORKER_CRASH"
    UNSUPPORTED_ELEMENT = "UNSUPPORTED_ELEMENT"
    UNEXPECTED = "UNEXPECTED"
    # warnings: the record is still written
    FUKUI_SCF_FAIL = "FUKUI_SCF_FAIL"
    VBUR_FAIL = "VBUR_FAIL"
    NO_FRONTIER = "NO_FRONTIER"
    RESTRAINED_GEOM = "RESTRAINED_GEOM"
    EMBED_UFF_FALLBACK = "EMBED_UFF_FALLBACK"
    EMBED_RANDOM_COORDS = "EMBED_RANDOM_COORDS"


class QMError(Exception):
    """A calculation step failed with a known reason code."""

    def __init__(self, reason: ReasonCode, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else str(reason))
        self.reason = reason
        self.detail = detail


@dataclass
class QMRecord:
    # identity
    smiles: str  # map-free canonical SMILES (the cache identity)
    atom_order_smiles: str  # SMILES whose AddHs atom order this record uses
    conformer_id: int
    method: str
    charge: int
    unpaired: int
    # structure
    atomic_numbers: np.ndarray  # (n,) int
    coords_ang: np.ndarray  # (n, 3)
    # energies and MO data
    energy_ha: float
    orbital_energies_ev: np.ndarray  # (nmo,)
    occupations: np.ndarray  # (nmo,)
    C: np.ndarray  # (nao, nmo), S-orthonormal
    S: np.ndarray  # (nao, nao)
    H_core: np.ndarray  # (nao, nao), tblite core Hamiltonian
    fock: np.ndarray  # (nao, nao), S C diag(eps) C^T S
    charges: np.ndarray  # (n,) Mulliken
    ao_atom: np.ndarray  # (nao,) atom index of each AO
    ao_l: np.ndarray  # (nao,) angular momentum of each AO: 0 s, 1 p, 2 d
    # reactivity descriptors (NaN-filled when unavailable)
    fukui_plus: np.ndarray = field(default_factory=lambda: np.empty(0))
    fukui_minus: np.ndarray = field(default_factory=lambda: np.empty(0))
    fukui_zero: np.ndarray = field(default_factory=lambda: np.empty(0))
    fukui_dual: np.ndarray = field(default_factory=lambda: np.empty(0))
    fukui_fmo_plus: np.ndarray = field(default_factory=lambda: np.empty(0))
    fukui_fmo_minus: np.ndarray = field(default_factory=lambda: np.empty(0))
    homo_ev: float = float("nan")
    lumo_ev: float = float("nan")
    chemical_potential_ev: float = float("nan")
    hardness_ev: float = float("nan")
    electrophilicity_ev: float = float("nan")
    # sterics
    buried_volume: np.ndarray = field(default_factory=lambda: np.empty(0))  # (n,) %Vbur
    warnings: str = ""  # comma-separated ReasonCode warning values
    schema_version: int = SCHEMA_VERSION

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    def add_warning(self, code: ReasonCode) -> None:
        parts = [w for w in self.warnings.split(",") if w]
        if str(code) not in parts:
            parts.append(str(code))
        self.warnings = ",".join(parts)


_SCALAR_TYPES = {"int": int, "float": float, "str": str}


def cache_key(smiles: str, conformer_id: int, method: str) -> str:
    """sha256 of the map-free canonical SMILES, the conformer id and the method string."""
    return hashlib.sha256(f"{smiles}|{conformer_id}|{method}".encode()).hexdigest()


def cache_path(cache_dir: Path, smiles: str, conformer_id: int, method: str) -> Path:
    return Path(cache_dir) / f"{cache_key(smiles, conformer_id, method)}.npz"


def save_record(record: QMRecord, path: Path) -> Path:
    """Write the record as .npz (no pickle). The write is atomic: tmp file, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f.name: np.asarray(getattr(record, f.name)) for f in fields(record)}
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(buf.getvalue())
        fh.flush()
        os.fsync(fh.fileno())  # a power loss after the rename must not leave an empty file
    os.replace(tmp, path)
    return path


def is_valid_cache_file(path: Path) -> bool:
    """A non-empty file is the minimum for a cache hit (zero-length files come from crashes)."""
    try:
        return Path(path).stat().st_size > 0
    except OSError:
        return False


def load_record(path: Path) -> QMRecord:
    with np.load(Path(path), allow_pickle=False) as npz:
        kwargs = {}
        for f in fields(QMRecord):
            arr = npz[f.name]
            kwargs[f.name] = (
                arr if arr.ndim else _SCALAR_TYPES.get(str(f.type), lambda x: x)(arr.item())
            )
    return _current_ao_layout(QMRecord(**kwargs), Path(path))


def _current_ao_layout(rec: QMRecord, path: Path) -> QMRecord:
    """The AO map is a function of the atomic numbers: recompute it with the current layout.
    Records written before 2026-10-20 carry an s, p, d order on transition-metal atoms that
    tblite does not use (`ksmech.qm.ao`); the matrices themselves were always right. A record
    whose AO count does not match the layout (a synthetic test record) is left as stored."""
    from ksmech.qm.ao import ao_layout

    try:
        ao_atom, ao_l = ao_layout(rec.atomic_numbers)
    except QMError:
        return rec
    if len(ao_l) != len(rec.ao_l):
        return rec
    if not (np.array_equal(ao_l, rec.ao_l) and np.array_equal(ao_atom, rec.ao_atom)):
        log.warning("%s: stored AO layout differs from tblite's shell order; recomputed", path)
        rec.ao_atom, rec.ao_l = ao_atom, ao_l
    return rec
