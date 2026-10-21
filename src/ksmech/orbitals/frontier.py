"""Per-atom frozen-orbital weights of the frontier levels HOMO-1, HOMO, LUMO, LUMO+1.

Each level is a degenerate block: every canonical MO within `DEGENERACY_EV` of the level's
energy, with the Mulliken populations summed over the block. HOMO-1 is the next distinct level
below the HOMO block, LUMO+1 the next above the LUMO block. This generalizes the frozen-orbital
f+ / f- (LUMO / HOMO populations) to four levels and handles degenerate pairs (benzene e1g).
Rows are NaN when a level does not exist (zero-electron species, no virtuals).
"""

from __future__ import annotations

import numpy as np

from ksmech.qm.ao import per_atom_sum
from ksmech.qm.fukui import OCC_THRESHOLD, homo_index
from ksmech.qm.records import QMRecord

DEGENERACY_EV = 0.05
LEVELS = ("homo_minus_1", "homo", "lumo", "lumo_plus_1")


def _block(energies: np.ndarray, members: np.ndarray, anchor: int, tol: float) -> np.ndarray:
    """MO indices among `members` within `tol` of the energy of `anchor`."""
    return members[np.abs(energies[members] - energies[anchor]) <= tol]


def frontier_blocks(
    energies_ev: np.ndarray, occupations: np.ndarray, tol: float = DEGENERACY_EV
) -> dict[str, np.ndarray]:
    """{level: canonical MO indices of its degenerate block}; missing levels are absent."""
    out: dict[str, np.ndarray] = {}
    occ = np.flatnonzero(occupations >= OCC_THRESHOLD)
    virt = np.flatnonzero(occupations < OCC_THRESHOLD)
    homo = homo_index(occupations)
    if homo is not None:
        out["homo"] = _block(energies_ev, occ, homo, tol)
        below = occ[energies_ev[occ] < energies_ev[homo] - tol]
        if len(below):
            out["homo_minus_1"] = _block(
                energies_ev, below, int(below[np.argmax(energies_ev[below])]), tol
            )
    if len(virt):
        lumo = int(virt[np.argmin(energies_ev[virt])])
        out["lumo"] = _block(energies_ev, virt, lumo, tol)
        above = virt[energies_ev[virt] > energies_ev[lumo] + tol]
        if len(above):
            out["lumo_plus_1"] = _block(
                energies_ev, above, int(above[np.argmin(energies_ev[above])]), tol
            )
    return out


def frontier_weights(rec: QMRecord, tol: float = DEGENERACY_EV) -> np.ndarray:
    """(n_atoms, 4) Mulliken population of each frontier block on each atom, NaN if absent."""
    out = np.full((rec.n_atoms, len(LEVELS)), np.nan)
    blocks = frontier_blocks(rec.orbital_energies_ev, rec.occupations, tol)
    for j, level in enumerate(LEVELS):
        if level not in blocks:
            continue
        total = np.zeros(rec.n_atoms)
        for k in blocks[level]:
            c = rec.C[:, k]
            total += per_atom_sum(c * (rec.S @ c), rec.ao_atom, rec.n_atoms)
        out[:, j] = total
    return out


def frozen_orbital_fukui_blocks(
    rec: QMRecord, tol: float = DEGENERACY_EV
) -> tuple[np.ndarray, np.ndarray]:
    """(f+, f-) per atom: the LUMO / HOMO block populations of `frontier_weights` divided by the
    block size, so each sums to 1 over the atoms like a Fukui function and symmetry-equivalent
    atoms get equal values (one MO of a degenerate pair gave benzene's carbons 0.27 / 0.22 /
    0.00). NaN when the level does not exist."""
    fmo = frontier_weights(rec, tol)
    blocks = frontier_blocks(rec.orbital_energies_ev, rec.occupations, tol)
    f_plus = fmo[:, LEVELS.index("lumo")] / len(blocks["lumo"]) if "lumo" in blocks else fmo[:, 2]
    f_minus = fmo[:, LEVELS.index("homo")] / len(blocks["homo"]) if "homo" in blocks else fmo[:, 1]
    return f_plus, f_minus
