"""Klopman-Salem term 1: the closed-shell (four-electron) repulsion between occupied orbitals.

    KS1(r, r') = - sum_{mu, nu} (N_a(mu) + N_a(nu)) beta_mu,nu S_mu,nu l_mu,r l_nu,r'

over AO pairs on different atoms, with beta the Wolfsberg-Helmholz resonance integral.
N_a is the electron population of the atom carrying the AO, valence electrons minus the
Mulliken charge: the spec's Eq. (ks) writes term 1 with lowercase q (populations, as in
Klopman's and Salem's papers) and term 2 with Q (charges), and a partial charge would make the
sign of a "repulsion" follow the charges. For a donor-acceptor pair (r, s) the steric feature
is the repulsion between r and the occupied orbitals around the acceptor:

    KS1(r, s) = sum_{r' in N(s), r' != r} KS1(r, r'),   N(s): occupied tokens whose main centre
                                                        lies within 3.5 Angstrom of m(s)

Between fragments the AO sum has no geometry. Its contact form keeps the populations and
replaces beta S l l by the contact resonance integral times the contact amplitude of the two
occupied tokens, scaled by the buried volumes of the two main centres (as fractions):

    KS1(r, r') = - (N_m(r) + N_m(r')) beta_c(r, r') a(r, r') V_m(r) V_m(r') / 100^2
"""

from __future__ import annotations

import numpy as np

from ksmech.orbitals.localize import valence_electrons
from ksmech.orbitals.tokens import MoleculeTokens, OrbitalToken
from ksmech.physics.contact_kernel import ContactKernel
from ksmech.physics.params import KSParams, wolfsberg_helmholz

NEIGHBOURHOOD_ANG = 3.5


def atom_populations(t: MoleculeTokens) -> np.ndarray:
    """Electron population per atom: GFN2 valence electrons minus the Mulliken charge."""
    return np.array([valence_electrons(a.atomic_number) - a.mulliken_charge for a in t.atoms])


def repulsion_matrix(t: MoleculeTokens, params: KSParams | None = None) -> np.ndarray:
    """(n_occ, n_occ) KS1(r, r') within one molecule, zero diagonal."""
    occ = t.occupied()
    if not occ:
        return np.zeros((0, 0))
    B = wolfsberg_helmholz(t.S, t.F, t.ao_atom, (params or KSParams()).K)  # 0 within an atom
    n = atom_populations(t)[t.ao_atom]
    Q = (n[:, None] + n[None, :]) * B * t.S
    L = np.column_stack([o.coeff_vector for o in occ])
    M = -(L.T @ Q @ L)
    np.fill_diagonal(M, 0.0)
    return M


def neighbourhood(occ: list[OrbitalToken], t_occ: MoleculeTokens, sites: np.ndarray) -> np.ndarray:
    """(n_sites, n_occ) True where occupied token r' has its main centre within 3.5 Angstrom of
    the site (an atom index of `t_occ`)."""
    pos = np.array([a.position for a in t_occ.atoms]).reshape(-1, 3)
    main = np.array([int(o.descriptors.center_atoms[0]) for o in occ], dtype=int)
    d = np.linalg.norm(pos[sites][:, None, :] - pos[main][None, :, :], axis=-1)
    return d <= NEIGHBOURHOOD_ANG


def contact_repulsion_matrix(
    rows: list[OrbitalToken],
    t_r: MoleculeTokens,
    cols: list[OrbitalToken],
    t_s: MoleculeTokens,
    kernel: ContactKernel,
    params: KSParams | None = None,
) -> np.ndarray:
    """(len(rows), len(cols)) contact KS1 between occupied tokens of two fragments."""
    n_r, n_s = atom_populations(t_r), atom_populations(t_s)
    out = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        a = int(r.descriptors.center_atoms[0])
        for j, s in enumerate(cols):
            b = int(s.descriptors.center_atoms[0])
            beta = kernel.beta(r, t_r, s, t_s, params)
            amp = kernel.amplitude(r, t_r, s, t_s)
            vbur = t_r.atoms[a].vbur * t_s.atoms[b].vbur / 1e4
            out[i, j] = -(n_r[a] + n_s[b]) * beta * amp * vbur
    return out
