"""The physics pair vector z_ij of the spec (section 5.6), for every ordered token pair.

    z_ij = [KS3, (l^T B l)^2, G, KS2, KS1, 1[phi(i) != phi(j)], R_ij]

Tokens of all fragments form one sequence: fragment by fragment, atoms then orbitals, in each
fragment's own token order. An entry is defined only where its term exists, and is 0
elsewhere (`defined` says which):

- KS3 and G: occupied i, virtual j (donor first), within one fragment or across two; KS3 is the
  contact term for every pair.
- (l^T B l)^2: occupied i, virtual j of one fragment: the Wolfsberg-Helmholz cross-centre
  coupling, the intramolecular extra.
- KS2: any two tokens on different sites (an atom, or an orbital's main centre).
- KS1: occupied i with occupied j (the pair repulsion), or with virtual j (the repulsion with
  the occupied tokens around j's main centre).
- the fragment flag and R_ij: every pair. R_ij is the distance between token positions (atom
  position, orbital centroid) within a fragment and the van der Waals contact distance of the
  two sites across fragments.

Entries are standardized with statistics of defined entries computed at the initial K and eps
(`fit_pair_stats`, `standardize`); a non-finite value is counted and left undefined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.orbitals.tokens import MoleculeTokens, OrbitalKind
from ksmech.physics.contact_kernel import ContactKernel
from ksmech.physics.contact_table import vdw_sum
from ksmech.physics.coulomb import coulomb_matrix
from ksmech.physics.ks_terms import contact_beta_matrix, gap_kernel_matrix, wh_matrix
from ksmech.physics.params import KSParams
from ksmech.physics.repulsion import contact_repulsion_matrix, neighbourhood, repulsion_matrix

PAIR_FEATURES = ("ks3", "coupling_sq", "gap_kernel", "ks2", "ks1", "inter", "distance")
_F = {name: k for k, name in enumerate(PAIR_FEATURES)}


@dataclass
class TokenIndex:
    """Where every token of every fragment sits in the joint sequence."""

    fragment: np.ndarray  # (N,) fragment index
    site: np.ndarray  # (N,) atom index (in its fragment) of the token's site
    position: np.ndarray  # (N, 3) atom position or orbital centroid, Angstrom
    site_position: np.ndarray  # (N, 3) position of the site atom
    charge: np.ndarray  # (N,) Mulliken charge of the site
    z: np.ndarray  # (N,) atomic number of the site
    occ: dict[int, np.ndarray]  # fragment -> joint indices of its occupied tokens, in order
    virt: dict[int, np.ndarray]


def token_index(fragments: list[MoleculeTokens]) -> TokenIndex:
    frag: list[int] = []
    site: list[int] = []
    pos: list[np.ndarray] = []
    occ, virt = {}, {}
    for f, t in enumerate(fragments):
        start = len(frag) + len(t.atoms)
        frag += [f] * (len(t.atoms) + len(t.orbitals))
        site += [a.atom_index for a in t.atoms]
        site += [int(o.descriptors.center_atoms[0]) for o in t.orbitals]
        pos += [a.position for a in t.atoms] + [o.descriptors.centroid for o in t.orbitals]
        kinds = np.array([str(o.kind) for o in t.orbitals])
        occ[f] = start + np.flatnonzero(kinds == OrbitalKind.OCCUPIED)
        virt[f] = start + np.flatnonzero(kinds == OrbitalKind.VIRTUAL)
    atoms = [fragments[f].atoms[s] for f, s in zip(frag, site, strict=True)]
    return TokenIndex(
        fragment=np.array(frag, dtype=int),
        site=np.array(site, dtype=int),
        position=np.array(pos, dtype=float).reshape(-1, 3),
        site_position=np.array([a.position for a in atoms], dtype=float).reshape(-1, 3),
        charge=np.array([a.mulliken_charge for a in atoms], dtype=float),
        z=np.array([a.atomic_number for a in atoms], dtype=int),
        occ=occ,
        virt=virt,
    )


@dataclass
class PairFeatures:
    z: np.ndarray  # (N, N, 7), PAIR_FEATURES order, 0 where undefined
    defined: np.ndarray  # (N, N, 7) bool
    index: TokenIndex
    n_non_finite: int = 0


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _set(pf: PairFeatures, name: str, rows: np.ndarray, cols: np.ndarray, values) -> None:
    pf.z[np.ix_(rows, cols, [_F[name]])] = np.asarray(values)[..., None]
    pf.defined[np.ix_(rows, cols, [_F[name]])] = True


def _token_level(pf: PairFeatures, params: KSParams) -> None:
    """Fragment flag, R_ij and KS2: defined for (almost) every pair."""
    ix = pf.index
    inter = ix.fragment[:, None] != ix.fragment[None, :]
    vdw = np.array([[vdw_sum(a, b) for b in ix.z] for a in ix.z]).reshape(inter.shape)
    distance = np.where(inter, vdw, _pairwise(ix.position, ix.position))
    site_distance = np.where(inter, vdw, _pairwise(ix.site_position, ix.site_position))
    everything = np.arange(len(ix.fragment))
    _set(pf, "inter", everything, everything, inter.astype(float))
    _set(pf, "distance", everything, everything, distance)
    ks2_defined = inter | (site_distance > 0)
    pf.z[..., _F["ks2"]] = coulomb_matrix(ix.charge, site_distance, ks2_defined, params)
    pf.defined[..., _F["ks2"]] = ks2_defined


def _virtual_sites(t: MoleculeTokens) -> np.ndarray:
    return np.array([int(o.descriptors.center_atoms[0]) for o in t.virtual()], dtype=int)


def _intra(pf, t: MoleculeTokens, f: int, params: KSParams) -> None:
    occ, virt = t.occupied(), t.virtual()
    rows, cols = pf.index.occ[f], pf.index.virt[f]
    if occ and virt:
        L_o = np.column_stack([o.coeff_vector for o in occ])
        L_v = np.column_stack([o.coeff_vector for o in virt])
        _set(pf, "coupling_sq", rows, cols, (L_o.T @ wh_matrix(t, params) @ L_v) ** 2)
    if occ:
        M = repulsion_matrix(t, params)
        _set(pf, "ks1", rows, rows, M)
        if virt:
            _set(pf, "ks1", rows, cols, M @ neighbourhood(occ, t, _virtual_sites(t)).T)


def _inter(pf, fragments, f: int, g: int, kernel: ContactKernel, params: KSParams) -> None:
    t_f, t_g = fragments[f], fragments[g]
    occ_f, occ_g, virt_g = t_f.occupied(), t_g.occupied(), t_g.virtual()
    if not occ_f or not occ_g:
        return
    C = contact_repulsion_matrix(occ_f, t_f, occ_g, t_g, kernel, params)
    _set(pf, "ks1", pf.index.occ[f], pf.index.occ[g], C)
    if virt_g:
        N = neighbourhood(occ_g, t_g, _virtual_sites(t_g))
        _set(pf, "ks1", pf.index.occ[f], pf.index.virt[g], C @ N.T)


def _orbital_term(pf, fragments, f: int, g: int, kernel, params) -> None:
    """KS3 and G for the occupied tokens of f against the virtual tokens of g."""
    t_f, t_g = fragments[f], fragments[g]
    occ, virt = t_f.occupied(), t_g.virtual()
    if not occ or not virt:
        return
    G = gap_kernel_matrix(occ, virt)
    beta = contact_beta_matrix(occ, t_f, virt, t_g, kernel, params)
    _set(pf, "gap_kernel", pf.index.occ[f], pf.index.virt[g], G)
    _set(pf, "ks3", pf.index.occ[f], pf.index.virt[g], beta**2 * G)


def pair_features(
    fragments: list[MoleculeTokens], kernel: ContactKernel, params: KSParams | None = None
) -> PairFeatures:
    """z_ij for every ordered pair of tokens of the reactant set `fragments`."""
    params = params or KSParams()
    ix = token_index(fragments)
    n = len(ix.fragment)
    pf = PairFeatures(np.zeros((n, n, len(PAIR_FEATURES))), np.zeros((n, n, 7), bool), ix)
    _token_level(pf, params)
    for f, t_f in enumerate(fragments):
        for g in range(len(fragments)):
            _orbital_term(pf, fragments, f, g, kernel, params)
            if f == g:
                _intra(pf, t_f, f, params)
            else:
                _inter(pf, fragments, f, g, kernel, params)
    diag = np.arange(n)
    pf.defined[diag, diag, _F["ks1"]] = False  # an orbital does not repel itself
    bad = pf.defined & ~np.isfinite(pf.z)
    pf.n_non_finite = int(bad.sum())
    pf.z[bad], pf.defined[bad] = 0.0, False
    return pf


@dataclass(frozen=True)
class PairStats:
    """Mean and standard deviation of each z entry over the defined entries of a training set,
    computed at the initial K and eps and then frozen."""

    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


def fit_pair_stats(features) -> PairStats:
    """Streaming mean / std over the defined entries of many `PairFeatures`."""
    k = len(PAIR_FEATURES)
    n, s1, s2 = np.zeros(k), np.zeros(k), np.zeros(k)
    for pf in features:
        for j in range(k):
            v = pf.z[..., j][pf.defined[..., j]]
            n[j], s1[j], s2[j] = n[j] + v.size, s1[j] + v.sum(), s2[j] + (v * v).sum()
    mean = np.divide(s1, n, out=np.zeros(k), where=n > 0)
    var = np.divide(s2, n, out=np.zeros(k), where=n > 0) - mean**2
    return PairStats(mean, np.sqrt(np.clip(var, 0.0, None)), n)


def standardize(pf: PairFeatures, stats: PairStats) -> np.ndarray:
    """(z - mean) / std on defined entries, 0 elsewhere; a constant entry is only centred."""
    std = np.where(stats.std > 0, stats.std, 1.0)
    return np.where(pf.defined, (pf.z - stats.mean) / std, 0.0)
