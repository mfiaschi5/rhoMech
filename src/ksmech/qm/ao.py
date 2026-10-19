"""AO-to-atom map for tblite's GFN2-xTB basis.

tblite does not report which AO belongs to which atom. GFN2 uses a minimal valence basis per
element, listed atom by atom: s (1 AO); s, p (4 AOs); or s, p, d (9 AOs). The AO count of an
element is probed from tblite itself (single-atom calculation), so the table cannot go stale
when the parametrization changes. Within an atom the order is s, p x3, d x5.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from tblite.interface import Calculator

from ksmech.qm.records import QMError, ReasonCode

# angular momentum per AO, by number of AOs on the atom
_LAYOUTS: dict[int, tuple[int, ...]] = {
    1: (0,),
    4: (0, 1, 1, 1),
    9: (0, 1, 1, 1, 2, 2, 2, 2, 2),
}


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
        n_ao = aos_per_element(int(z))
        if n_ao not in _LAYOUTS:
            raise QMError(ReasonCode.UNSUPPORTED_ELEMENT, f"Z={z} has {n_ao} AOs")
        atom_idx.extend([i] * n_ao)
        ang.extend(_LAYOUTS[n_ao])
    return np.array(atom_idx, dtype=np.int32), np.array(ang, dtype=np.int32)


def per_atom_sum(ao_values: np.ndarray, ao_atom: np.ndarray, n_atoms: int) -> np.ndarray:
    """Sum an AO-resolved quantity over the AOs of each atom."""
    return np.bincount(ao_atom, weights=ao_values, minlength=n_atoms)
