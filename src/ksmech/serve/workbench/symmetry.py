"""The workbench's point-group panel: request limits, the looser-tolerance hint and the
`PointGroupOut` fields, over the engine's detector (`ksmech.qm.point_group`, which holds the
algorithm and its tolerance semantics). Display only.

A hint repeats the search at the looser tolerances of HINT_TOLERANCES and reports the highest
group found ("would be Td at 0.3 Å").
"""

from __future__ import annotations

import math

import numpy as np

from ksmech.qm.point_group import DEFAULT_TOLERANCE, Element, expected_order, search

__all__ = ["DEFAULT_TOLERANCE", "SymmetryError", "expected_order", "point_group"]

MIN_TOLERANCE = 0.01
MAX_TOLERANCE = 0.5
MAX_ATOMS = 1000
HINT_MAX_ATOMS = 300
HINT_TOLERANCES = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


class SymmetryError(ValueError):
    """`code` is a serve reason code (REQUEST_INVALID, LIMIT_EXCEEDED)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---- the public entry point


def check_input(z, xyz, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    if not (MIN_TOLERANCE <= float(tolerance) <= MAX_TOLERANCE):
        msg = f"tolerance {tolerance} outside {MIN_TOLERANCE}..{MAX_TOLERANCE} Angstrom"
        raise SymmetryError("REQUEST_INVALID", msg)
    z = np.asarray(z, dtype=int).reshape(-1)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(z) == 0 or len(z) != len(xyz):
        raise SymmetryError("REQUEST_INVALID", f"{len(z)} atomic numbers for {len(xyz)} positions")
    if len(z) > MAX_ATOMS:
        msg = f"{len(z)} atoms: the point-group search takes at most {MAX_ATOMS}"
        raise SymmetryError("LIMIT_EXCEEDED", msg)
    if not np.isfinite(xyz).all():
        raise SymmetryError("REQUEST_INVALID", "non-finite coordinates")
    return z, xyz


def _rank(order: int | None) -> float:
    return math.inf if order is None else float(order)


def looser_hint(z, xyz, tol: float, order: int | None) -> dict | None:
    """The highest group at a looser tolerance of HINT_TOLERANCES (its smallest such
    tolerance), when it is higher than `order`."""
    best: dict | None = None
    for t in HINT_TOLERANCES:
        if t <= tol + 1e-9:
            continue
        s = search(z, xyz, t)
        if _rank(s.order) > _rank(best["order"] if best else order):
            best = {"point_group": s.name, "order": s.order, "tolerance": t}
    return best


def _signed(v: np.ndarray) -> list[float]:
    k = int(np.argmax(np.abs(v)))
    return [float(c) for c in (v if v[k] >= 0 else -v)]


def _element_out(e: Element, k: int, com: np.ndarray) -> dict:
    return {
        "id": f"e{k}",
        "kind": e.kind,
        "label": e.label,
        "order": e.order,
        "vector": None if e.vector is None else _signed(e.vector),
        "point": [float(c) for c in com],
        "atoms": list(e.atoms),
        "deviation_ang": max((o.dev for o in e.ops), default=0.0),
        "principal": e.principal,
        "n_operations": len(e.ops),
    }


def _symbol(z: int) -> str:
    from rdkit import Chem

    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


def point_group(z, xyz, tolerance: float = DEFAULT_TOLERANCE, *, hint: bool = True) -> dict:
    """The point group of a structure: `PointGroupOut`'s fields other than molecule_id and key
    (coordinates and vectors in the input frame). Raises `SymmetryError`."""
    z, xyz = check_input(z, xyz, tolerance)
    s = search(z, xyz, float(tolerance))
    f = s.frame
    disp = np.linalg.norm(f.x - s.symmetrized, axis=1)
    notes = list(s.notes)
    looser = None
    if hint and len(z) <= HINT_MAX_ATOMS:
        looser = looser_hint(z, xyz, float(tolerance), s.order)
    elif hint:
        notes.append(f"no looser-tolerance hint above {HINT_MAX_ATOMS} atoms")
    return {
        "tolerance": float(tolerance),
        "point_group": s.name,
        "order": s.order,
        "top": s.top,
        "moments": [float(m) for m in s.moments],
        "n_atoms": len(z),
        "centre": [float(c) for c in f.com],
        "elements": [_element_out(e, k, f.com) for k, e in enumerate(s.elements)],
        "classes": [{"element": _symbol(z[c[0]]), "atoms": c} for c in s.classes],
        "rms_ang": float(np.sqrt((disp**2).mean())),
        "max_deviation_ang": float(disp.max()),
        "symmetrized_xyz": [[float(c) for c in p] for p in s.symmetrized + f.com],
        "looser": looser,
        "notes": notes,
    }
