"""Row helpers shared by the token views: the pool's `/tokens` (pool_service) and the molecule
workbench's orbital sets (orbital_sets).

`orbital_model_input` serializes exactly the model-input descriptors of an orbital token;
`orbital_diagnostic` the classifier's labels (display only, never model input). The other
helpers turn numpy values into JSON-able Python (NaN to None). Only numpy and `schemas` are
imported at the top; upstream modules inside the functions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ksmech.serve.core import schemas as S

if TYPE_CHECKING:
    from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken


def overlap_sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def spectral_set(d) -> list[tuple[float, float]]:
    """(E_k eV, weight) of every spectral entry with weight > 0."""
    w = np.asarray(d.spectral_weights, dtype=float)
    e = np.asarray(d.spectral_energies, dtype=float)
    return [(float(e[k]), float(w[k])) for k in np.flatnonzero(w > 0)]


def spectral_rows(d) -> list[tuple[int, float, float]]:
    """(canonical MO index, E_k eV, weight) of every spectral entry with weight > 0."""
    w = np.asarray(d.spectral_weights, dtype=float)
    e = np.asarray(d.spectral_energies, dtype=float)
    idx = np.asarray(d.spectral_indices, dtype=int)
    return [(int(idx[k]), float(e[k]), float(w[k])) for k in np.flatnonzero(w > 0)]


def orbital_model_input(o: OrbitalToken) -> S.OrbitalModelInputOut:
    """Exactly the model-input descriptors (`orbitals/descriptors.py`, `model_input=True`);
    the spectral set as (E_k, weight) rows with weight > 0, the overlap sign alongside."""
    d = o.descriptors
    return S.OrbitalModelInputOut(
        occupation=float(d.occupation),
        n_centers=int(d.n_centers),
        center_weights=[float(x) for x in np.asarray(d.center_weights).ravel()],
        center_spd=np.asarray(d.center_spd, dtype=float).tolist(),
        cos_a=float(d.cos_a),
        cos_b=float(d.cos_b),
        overlap_population=float(d.overlap_population),
        overlap_sign=overlap_sign(float(d.overlap_population)),
        distance_ab=float(d.distance_ab),
        radius_gyration=float(d.radius_gyration),
        spectral_set=spectral_set(d),
    )


def diag_float(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def jsonish(value: Any) -> Any:
    """numpy scalars and arrays as Python, NaN / inf as None."""
    if isinstance(value, np.ndarray):
        return [jsonish(v) for v in value.tolist()]
    if isinstance(value, list | tuple):
        return [jsonish(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    return value


def fg_label(fg_id: int) -> str:
    """The functional-group label of a vocabulary id (diagnostic)."""
    from ksmech.orbitals.fgroups import default_vocab

    labels = default_vocab().labels
    return labels[fg_id] if 0 <= fg_id < len(labels) else f"fg{fg_id}"


def parents_text(t: MoleculeTokens, o: OrbitalToken) -> str:
    """The classifier's parent atoms as "C0:0.52,O2:0.41" (record indices)."""
    return ",".join(
        f"{t.symbol(int(a))}{int(a)}:{float(w):.2f}"
        for a, w in zip(o.parent_atoms, o.parent_weights, strict=False)
    )


def orbital_diagnostic(t: MoleculeTokens, o: OrbitalToken) -> dict:
    """`OrbitalRowDiagnosticOut` fields: the classifier's labels, display only."""
    return {
        "diagnostic": True,
        "type": str(o.type),
        "parents": parents_text(t, o),
        "s_share": diag_float(o.s_share),
        "p_share": diag_float(o.p_share),
        "d_share": diag_float(o.d_share),
        "classifier_overlap_population": diag_float(o.overlap_population),
        "metal_bound": bool(o.metal_bound),
    }
