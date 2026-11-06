"""Per-atom frozen-orbital weights of the frontier levels HOMO-1, HOMO, LUMO, LUMO+1.

Each level is a degenerate block: every canonical MO within `DEGENERACY_EV` of the level's
energy, with the Mulliken populations summed over the block. HOMO-1 is the next distinct level
below the HOMO block, LUMO+1 the next above the LUMO block. This generalizes the frozen-orbital
f+ / f- (LUMO / HOMO populations) to four levels and handles degenerate pairs (benzene e1g).
Rows are NaN when a level does not exist (zero-electron species, no virtuals).

Token schema 10 stores a second set beside these: the same blocks with `EXACT_EV` = 1e-4 eV, so
only numerically degenerate levels are summed (open-shell plan P0a, item 3); a model reads one
set or the other (`ModelConfig.frontier_blocks`).

An open shell (u > 0) follows the radical convention (U16, open-shell plan P2b): HOMO and LUMO
are the electron-removal and electron-addition levels, so both are the open group, every MO
whose plateau reference (`localize.reference_occupations`) lies strictly between 0 and 2,
read from the occupations and not from the window. HOMO-1 is the highest level with reference 2
and LUMO+1 the lowest with reference 0, each summed over the window as for a closed shell, so a
column keeps one meaning across molecules. Frozen-orbital f+ = f- = the open group's population
per MO.
"""

from __future__ import annotations

import numpy as np

from ksmech.orbitals.localize import reference_occupations
from ksmech.qm.ao import per_atom_sum
from ksmech.qm.fukui import OCC_THRESHOLD, homo_index
from ksmech.qm.records import QMRecord

DEGENERACY_EV = 0.05
EXACT_EV = 1e-4  # numerical degeneracy: the tolerance of the exact-degeneracy frontier columns
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


def open_shell_frontier_blocks(
    energies_ev: np.ndarray, reference: np.ndarray, tol: float = DEGENERACY_EV
) -> dict[str, np.ndarray]:
    """{level: canonical MO indices} of an open shell from its plateau reference occupations:
    HOMO = LUMO = the open group (0 < reference < 2); HOMO-1 the `tol` block of the highest MO
    with reference 2, LUMO+1 that of the lowest with reference 0; missing levels are absent."""
    out: dict[str, np.ndarray] = {}
    open_group = np.flatnonzero((reference > 0) & (reference < 2))
    if len(open_group):
        out["homo"] = out["lumo"] = open_group
    double, empty = np.flatnonzero(reference == 2), np.flatnonzero(reference == 0)
    if len(double):
        top = int(double[np.argmax(energies_ev[double])])
        out["homo_minus_1"] = _block(energies_ev, double, top, tol)
    if len(empty):
        bottom = int(empty[np.argmin(energies_ev[empty])])
        out["lumo_plus_1"] = _block(energies_ev, empty, bottom, tol)
    return out


def record_frontier_blocks(rec: QMRecord, tol: float = DEGENERACY_EV) -> dict[str, np.ndarray]:
    """The frontier blocks of a record: `frontier_blocks` on the occupations for a closed shell,
    `open_shell_frontier_blocks` on the plateau reference for an open shell (u > 0)."""
    if rec.unpaired > 0:
        return open_shell_frontier_blocks(rec.orbital_energies_ev, reference_occupations(rec), tol)
    return frontier_blocks(rec.orbital_energies_ev, rec.occupations, tol)


def frontier_weights(rec: QMRecord, tol: float = DEGENERACY_EV) -> np.ndarray:
    """(n_atoms, 4) Mulliken population of each frontier block on each atom, NaN if absent."""
    out = np.full((rec.n_atoms, len(LEVELS)), np.nan)
    blocks = record_frontier_blocks(rec, tol)
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
    0.00). NaN when the level does not exist. For an open shell both are the open group's
    population per MO (`record_frontier_blocks`), so f+ = f-."""
    fmo = frontier_weights(rec, tol)
    blocks = record_frontier_blocks(rec, tol)
    f_plus = fmo[:, LEVELS.index("lumo")] / len(blocks["lumo"]) if "lumo" in blocks else fmo[:, 2]
    f_minus = fmo[:, LEVELS.index("homo")] / len(blocks["homo"]) if "homo" in blocks else fmo[:, 1]
    return f_plus, f_minus
