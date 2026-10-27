"""`QMRecord`: everything one GFN2-xTB calculation produces for one molecule and conformer.

Units: coordinates in Angstrom, orbital energies in eV (suffix `_ev`), total energy and the
H / Fock matrices in Hartree. Atom order is the order of `Chem.AddHs(Chem.MolFromSmiles(
atom_order_smiles))`; AO indices run atom by atom (see `ksmech.qm.ao`).
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from ksmech.data import provenance as prov

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
    SDF_UNREADABLE = "SDF_UNREADABLE"  # a dataset's molfile RDKit cannot read (QMugs)
    RECORD_ORDER = "RECORD_ORDER"  # a dataset's atoms do not map onto the record's (QMugs)
    # warnings: the record is still written
    FUKUI_SCF_FAIL = "FUKUI_SCF_FAIL"
    VBUR_FAIL = "VBUR_FAIL"
    NO_FRONTIER = "NO_FRONTIER"
    RESTRAINED_GEOM = "RESTRAINED_GEOM"
    EMBED_UFF_FALLBACK = "EMBED_UFF_FALLBACK"
    EMBED_RANDOM_COORDS = "EMBED_RANDOM_COORDS"
    EMBED_RELAXED_BOUNDS = "EMBED_RELAXED_BOUNDS"


class QMError(Exception):
    """A calculation step failed with a known reason code."""

    def __init__(self, reason: ReasonCode, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else str(reason))
        self.reason = reason
        self.detail = detail

    def __reduce__(self):  # a worker's QMError keeps its reason code, not the message
        return type(self), (self.reason, self.detail)


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


def record_config(method: str, conformer_id: int) -> dict[str, Any]:
    """The configuration a QM record is stamped with and checked against: the method string,
    its solvation part and the conformer id."""
    parts = method.split("/")
    return {
        "method": method,
        "solvation": parts[1] if len(parts) > 1 else "",
        "conformer_id": int(conformer_id),
    }


def record_inputs(smiles: str) -> dict[str, str]:
    return {"smiles": prov.sha256_text(smiles)}


def _record_arrays(record: QMRecord) -> dict[str, np.ndarray]:
    return {f.name: np.asarray(getattr(record, f.name)) for f in fields(record)}


def save_record(record: QMRecord, path: Path) -> Path:
    """Write the record as .npz (no pickle) with its provenance header, atomically."""
    arrays = _record_arrays(record)
    header = prov.stamp(
        "qm_record",
        record.schema_version,
        record_inputs(record.smiles),
        record_config(record.method, record.conformer_id),
        prov.array_digest(arrays),
    )
    return prov.save_npz(path, arrays, header)


def is_valid_cache_file(path: Path) -> bool:
    """A non-empty file is the minimum for a cache hit (zero-length files come from crashes)."""
    try:
        return Path(path).stat().st_size > 0
    except OSError:
        return False


def verify_record(path: Path, smiles: str, conformer_id: int, method: str) -> prov.Provenance:
    """Raise StaleArtifact unless the record at `path` was written for this molecule, method
    and conformer at the current schema."""
    return prov.verify(
        path,
        "qm_record",
        SCHEMA_VERSION,
        record_config(method, conformer_id),
        record_inputs(smiles),
    )


def load_record(path: Path, identity: tuple[str, int, str] | None = None) -> QMRecord:
    """Read a record. Its header is always verified (kind, schema); with `identity` =
    (smiles, conformer_id, method) also the molecule and configuration it was built for."""
    if identity is None:
        prov.verify(path, "qm_record", SCHEMA_VERSION)
    else:
        verify_record(Path(path), *identity)
    with np.load(Path(path), allow_pickle=False) as npz:
        kwargs = {}
        for f in fields(QMRecord):
            arr = npz[f.name]
            kwargs[f.name] = (
                arr if arr.ndim else _SCALAR_TYPES.get(str(f.type), lambda x: x)(arr.item())
            )
    return _current_ao_layout(QMRecord(**kwargs), Path(path))


def backfill_header(path: Path) -> str:
    """Add the provenance header to a record written before headers existed, leaving every
    array unchanged. Returns `stamped`, `already` or `unreadable`."""
    if prov.read_header(path) is not None:
        return "already"
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError) as exc:
        log.warning("backfill: %s unreadable: %s", path, exc)
        return "unreadable"
    smiles, method = str(arrays["smiles"]), str(arrays["method"])
    header = prov.stamp(
        "qm_record",
        int(arrays["schema_version"]),
        record_inputs(smiles),
        record_config(method, int(arrays["conformer_id"])),
        prov.array_digest(arrays),
        extra={"backfilled": True},
    )
    prov.save_npz(path, arrays, header)
    return "stamped"


def backfill_headers(cache_dir: Path) -> Counter:
    """`backfill_header` on every record in `cache_dir`; counts by outcome."""
    return Counter(backfill_header(p) for p in sorted(Path(cache_dir).glob("*.npz")))


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
