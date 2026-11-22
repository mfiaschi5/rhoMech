"""The donor-acceptor energy of every (occupied, virtual) LMO pair of one molecule: the target
of the phase-3 pretraining (docs/decisions.md, 2026-11-22, "the orbital-pair pretraining
target").

It is NBO's second-order estimate, taken on the localized orbitals of the token file. Each
LMO is restricted to its own centres (the atoms carrying at least tau_atom of its population,
`ks_terms.centre_restricted`), the whole restricted set (occupied and virtual) is
orthonormalized symmetrically under S (Lowdin), and the Fock matrix is read in that basis as
F'. For an occupied r and a virtual s:

    E2(r, s) = n_r |F'_rs|^2 / (F'_ss - F'_rr)   (eV), n_r = 2.

The exact LMOs have no occupied-virtual Fock element, because the two blocks are localized
apart (Brillouin). Restricted to their centres they do have one, and F'_rs measures how much
of the delocalization tail of r points along s. That covers hyperconjugation, conjugation,
n -> sigma* and n -> pi* within the molecule: the intramolecular face of the Klopman-Salem
covalent term. E2 is invariant to the phase of every LMO, and it is defined for closed shells
only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.ks_terms import centre_restricted
from ksmech.qm.fukui import HARTREE_TO_EV

CLOSED_TOL = 0.05  # |n - 2| or |n| above this is not a closed-shell token (`is_fermi_smeared`)
MIN_DENOMINATOR_EV = 0.1  # F'_ss - F'_rr at or below this leaves E2 undefined
MIN_OVERLAP_EIG = 1e-6  # T^T S T closer to singular than this: the restricted set is degenerate
OPEN_SHELL = "OPEN_SHELL"  # reason codes of `ClosedShellOnly` and of a degenerate set
DEGENERATE_BASIS = "DEGENERATE_BASIS"


class ClosedShellOnly(ValueError):
    """The molecule has a token that is neither doubly occupied nor empty."""


class DegenerateBasis(ValueError):
    """The centre-restricted LMOs are linearly dependent under S."""


@dataclass(frozen=True)
class DonorAcceptor:
    e2: np.ndarray  # (n_occ, n_virt) eV, 0 where not `defined`
    coupling: np.ndarray  # (n_occ, n_virt) |F'_rs| eV
    denominator: np.ndarray  # (n_occ, n_virt) F'_ss - F'_rr eV
    defined: np.ndarray  # (n_occ, n_virt) bool: denominator > MIN_DENOMINATOR_EV
    min_overlap_eig: float  # the smallest eigenvalue of T^T S T (1 when nothing overlaps)


def check_closed_shell(t: MoleculeTokens) -> None:
    """Raise `ClosedShellOnly` unless every occupied token has n ~ 2 and every virtual n ~ 0."""
    occ = np.array([o.descriptors.occupation for o in t.occupied()])
    virt = np.array([o.descriptors.occupation for o in t.virtual()])
    if np.any(np.abs(occ - 2.0) > CLOSED_TOL) or np.any(np.abs(virt) > CLOSED_TOL):
        raise ClosedShellOnly(f"{t.smiles}: a token is neither doubly occupied nor empty")


def lowdin(T: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, float]:
    """(T (T^T S T)^-1/2, its smallest eigenvalue): the orthonormal set closest to the columns
    of T under the metric S. Raises `DegenerateBasis` when T^T S T is near singular."""
    M = T.T @ S @ T
    w, V = np.linalg.eigh((M + M.T) / 2)
    if w.min() < MIN_OVERLAP_EIG:
        raise DegenerateBasis(f"T^T S T has an eigenvalue {w.min():.2e}")
    return T @ (V * w**-0.5) @ V.T, float(w.min())


def fock_in(t: MoleculeTokens, orbitals: list[OrbitalToken], restrict: bool = True):
    """(F' in eV, the smallest overlap eigenvalue): the Fock matrix in the Lowdin-orthonormalized
    basis of `orbitals`, each restricted to its own centres (`restrict=False`: the exact LMOs,
    whose occupied-virtual block vanishes; a check of the machinery)."""
    if restrict:
        T = centre_restricted(orbitals, t.ao_atom)
    else:
        T = np.column_stack([o.coeff_vector for o in orbitals]).reshape(len(t.ao_atom), -1)
    Tp, w = lowdin(T, t.S)
    return Tp.T @ t.F @ Tp * HARTREE_TO_EV, w


def donor_acceptor(t: MoleculeTokens, restrict: bool = True) -> DonorAcceptor:
    """E2 of every (occupied, virtual) pair of one closed-shell molecule, rows and columns in the
    order of `t.occupied()` and `t.virtual()` (the order of the sample's pair block)."""
    check_closed_shell(t)
    occ, virt = t.occupied(), t.virtual()
    F, w = fock_in(t, [*occ, *virt], restrict)
    n_occ = len(occ)
    eps = np.diag(F)
    denominator = eps[None, n_occ:] - eps[:n_occ, None]
    coupling = np.abs(F[:n_occ, n_occ:])
    defined = denominator > MIN_DENOMINATOR_EV
    safe = np.where(defined, denominator, 1.0)
    e2 = np.where(defined, 2.0 * coupling**2 / safe, 0.0)
    return DonorAcceptor(e2, coupling, denominator, defined, w)
