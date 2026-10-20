"""AO-to-atom map for tblite's GFN2-xTB basis.

tblite does not report which AO belongs to which atom. GFN2 uses a minimal valence basis per
element, listed atom by atom and, within an atom, shell by shell in the order of tblite's own
GFN2 parametrization (`gfn2_shells`, exported from the library): s, p for C, N, O; s, p, d for
Si to I; but d, s, p for the transition metals (Ti: 3d, 4s, 4p). Within a p shell the order is
(py, pz, px); within a d shell (dxy, dyz, dz2, dxz, dx2-y2). The AO count is also probed from
a single-atom calculation and must agree with the shell list.

Until 2026-10-20 the layout assumed s, p, d on every atom, which mislabelled the AOs of
transition-metal atoms (docs/decisions.md); `records.load_record` now recomputes the layout.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from tblite.interface import Calculator

from ksmech.qm.records import QMError, ReasonCode

_ANGULAR = "spdfg"


@cache
def gfn2_shells() -> dict[int, tuple[tuple[int, int, float, int], ...]]:
    """Atomic number -> ((n, l, Slater exponent, primitives), ...) per shell, in tblite's AO
    order, read from the GFN2-xTB parametrization tblite itself exports."""
    from rdkit import Chem
    from tblite import library

    param = library.new_param()
    library.export_gfn2_param(param)
    table = library.new_table()
    library.dump_param(param, table)
    elements = library.table_to_dict(table)["element"]
    pt = Chem.GetPeriodicTable()
    out = {}
    for symbol, rec in elements.items():
        shells = zip(rec["shells"], rec["slater"], rec["ngauss"], strict=True)
        out[pt.GetAtomicNumber(symbol)] = tuple(
            (int(name[:-1]), _ANGULAR.index(name[-1]), float(zeta), int(ng))
            for name, zeta, ng in shells
        )
    return out


def element_ao_l(z: int) -> tuple[int, ...]:
    """Angular momentum of every AO of element `z`, in tblite's order."""
    return tuple(ell for _, ell, _, _ in gfn2_shells()[int(z)] for _ in range(2 * ell + 1))


@cache
def aos_per_element(z: int) -> int:
    """Number of AOs tblite assigns to element `z`."""
    for uhf in (0, 1):  # odd-electron atoms need one unpaired electron
        try:
            calc = Calculator("GFN2-xTB", np.array([z]), np.zeros((1, 3)), charge=0.0, uhf=uhf)
            calc.set("verbosity", 0)
            return int(calc.singlepoint().get("norbitals"))
        except Exception:  # noqa: BLE001 - tblite raises its own errors; retry then give up
            continue
    raise QMError(ReasonCode.UNSUPPORTED_ELEMENT, f"tblite cannot handle Z={z}")


def ao_layout(atomic_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ao_atom, ao_l): atom index and angular momentum for every AO."""
    atom_idx: list[int] = []
    ang: list[int] = []
    for i, z in enumerate(atomic_numbers):
        if int(z) not in gfn2_shells():
            raise QMError(ReasonCode.UNSUPPORTED_ELEMENT, f"Z={z} has no GFN2 parameters")
        ls = element_ao_l(int(z))
        if len(ls) != aos_per_element(int(z)):
            raise QMError(ReasonCode.UNSUPPORTED_ELEMENT, f"Z={z}: shell list != AO count")
        atom_idx.extend([i] * len(ls))
        ang.extend(ls)
    return np.array(atom_idx, dtype=np.int32), np.array(ang, dtype=np.int32)


def per_atom_sum(ao_values: np.ndarray, ao_atom: np.ndarray, n_atoms: int) -> np.ndarray:
    """Sum an AO-resolved quantity over the AOs of each atom."""
    return np.bincount(ao_atom, weights=ao_values, minlength=n_atoms)
