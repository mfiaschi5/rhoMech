"""Oriented contact overlap of two localized orbitals (`ContactConfig(kernel="oriented")`).

The two LMOs are evaluated as if one approached the other along their lobes:

1. **Lobe at a site** (`site_candidates`). For a token r and one of its centres a (the site),
   the |amplitude| of the site's part of the LMO (its AO coefficients on a) is evaluated in 200
   directions on a 1.5 Angstrom sphere around a, restricted to the hemisphere pointing away
   from every other centre c of r (n . (x_c - x_a) <= 0); a one-centre token uses the
   hemisphere away from a's bonded neighbours, weighted by the token's population on each. The
   lobe is the best local maximum; a second local maximum within 5% is a tied lobe (the two
   faces of a pi). A site whose AO part has no direction (an s function) points away from the
   other centres (or the weighted neighbours): sigma(C-H) at H points out beyond H, sigma* at C
   to the backside, a pi or pi* perpendicular. The whole LMO is not used on the sphere: a
   neighbour inside it would win through the LMO's tail on it (docs/decisions.md).
2. **Placement and clash guard**: only the acceptor's local cap is placed (`cap_atoms`: its
   centre atoms and their first bonded neighbours, hydrogens included), for every pair, intra-
   and intermolecular. The cap is moved rigidly so that the acceptor's lobe at site b is
   antiparallel to the donor lobe at site a and b sits at distance d from a along it; the donor
   fragment stays whole. Tied lobes of both tokens are all tried; a placement that clashes is
   also tried at the other `N_ROLLS` - 1 roll angles about the approach axis (which keeps the
   lobes antiparallel and b fixed). The largest |S| among the placements whose cap atoms all
   stay 1.5 Angstrom from the donor fragment's atoms is kept; if none is clash-free, the
   opposite faces are tried the same way; if those clash too, the value is the smallest |S|
   over the unrolled placements of both tiers and the pair is flagged CLASH. An
   intramolecular pair is placed the same way (the molecule itself is not re-posed): before
   the cap, a whole copy of the molecule clashed with itself on most pairs.
3. **Overlap**: S = l_r^T S(placement) l_s over the AO pairs of the donor fragment and the
   placed cap, with the acceptor's LMO truncated to the cap atoms' AOs (the overlap at contact
   is local to the lobes), with tblite's own basis (`ksmech.physics.gto`), each atom pair in its
   local frame (sigma / pi / delta radial functions tabulated from the analytic integrals on a
   0.005 bohr grid, splined; checked to 1e-9); atom pairs whose coefficient-weighted bound is
   below `tol` are skipped.

`site_contact(r, a, s, b)` is the value where the sites are known; the pair-track value of a
token pair (`block_contacts`) is the maximum over the donor's centres x the acceptor's centres,
a clash-free site pair always preferred, with the argmax sites and the CLASH flag recorded.
Blocks are cached per (molecule, molecule, block, d) in memory and, given a directory, on disk.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from ksmech.data import provenance as prov
from ksmech.orbitals.tokens import (
    CandidateConfig,
    MoleculeTokens,
    OrbitalKind,
    OrbitalToken,
    bonded_matrix,
    site_pairs_excluded,
    token_digest,
)
from ksmech.physics.gto import SPH, atom_ao_slices, element_shells, shell_overlap
from ksmech.qm.fukui import HARTREE_TO_EV
from ksmech.qm.geometry import BOHR_ANG

log = logging.getLogger(__name__)

ORIENTED_VERSION = 5  # bump when the lobe rules, placement, integrals or stored fields change
DEFAULT_CACHE_DIR = Path("data/cache/oriented")
SPHERE_RADIUS_ANG = 1.5
N_DIRECTIONS = 200
GRID_STEP = 0.005  # bohr
GRID_MAX = 30.0  # bohr; overlaps beyond are 0
# local-frame class of each spherical component (tblite order): 0 sigma, 1 pi-y, 2 pi-x,
# 3 delta-xy, 4 delta-x2y2; components of two shells couple only within one class
_CLASS = {0: [0], 1: [1, 0, 2], 2: [3, 1, 0, 2, 4]}
_TB_FROM_XYZ = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)  # (y, z, x) <- xyz


@dataclass(frozen=True)
class OrientedConfig:
    distance: float = 2.3  # Angstrom between the site atoms (best KS3 of the d sweep)
    tol: float = 1e-6  # skip atom pairs whose bound on |contribution| is below this
    clash: float = 1.5  # Angstrom: a placement with two atoms of the fragments closer clashes
    cache_dir: Path | None = None  # disk cache (the CLIs pass DEFAULT_CACHE_DIR); memory always
    site_weight: float = CandidateConfig.site_weight  # intramolecular site rule (`block_contacts`)


# ---- local radial tables


@dataclass(frozen=True)
class PairTable:
    """Splines of the local sigma / pi / delta overlaps of every shell pair of two elements."""

    splines: dict[tuple[int, int], CubicSpline]  # (shell i, shell j) -> components by class
    classes: dict[tuple[int, int], tuple[int, ...]]  # local class of each spline component
    envelope: np.ndarray  # max |S| over all components at R' >= R, on the grid


def _local_values(sa, sb, R: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """(n_classes, len(R)) local overlaps with B at +z, one value per shared class."""
    block = shell_overlap(sa, sb, np.column_stack([np.zeros_like(R), np.zeros_like(R), R]))
    cls_a, cls_b = _CLASS[sa.l], _CLASS[sb.l]
    shared = sorted(set(cls_a) & set(cls_b))
    rows = [block[:, cls_a.index(c), cls_b.index(c)] for c in shared]
    return np.array(rows), tuple(shared)


@cache
def pair_table(za: int, zb: int) -> PairTable:
    grid = np.arange(0.0, GRID_MAX + GRID_STEP, GRID_STEP)
    splines, classes, env = {}, {}, np.zeros_like(grid)
    for i, sa in enumerate(element_shells(za)):
        for j, sb in enumerate(element_shells(zb)):
            values, shared = _local_values(sa, sb, grid)
            splines[i, j] = CubicSpline(grid, values, axis=1)
            classes[i, j] = shared
            env = np.maximum(env, np.abs(values).max(axis=0))
    envelope = np.maximum.accumulate(env[::-1])[::-1]
    return PairTable(splines, classes, envelope)


def atom_pair_blocks_local(za: int, zb: int, V: np.ndarray) -> np.ndarray:
    """(N, nao_a, nao_b) lab-frame blocks assembled from the local tables and rotations: the
    same quantity as `gto.atom_pair_blocks`, for checking the tables (V in bohr, B - A)."""
    V = np.asarray(V, dtype=float).reshape(-1, 3)
    R = np.linalg.norm(V, axis=1)
    Q = local_frames(V / R[:, None])
    table = pair_table(int(za), int(zb))
    sh_a, sh_b = element_shells(za), element_shells(zb)
    off_a = np.cumsum([0] + [s.n_sph for s in sh_a])
    off_b = np.cumsum([0] + [s.n_sph for s in sh_b])
    out = np.zeros((len(V), off_a[-1], off_b[-1]))
    for (i, j), spline in table.splines.items():
        values = spline(R)
        loc = np.zeros((len(V), sh_a[i].n_sph, sh_b[j].n_sph))
        for k, cls in enumerate(table.classes[i, j]):
            loc[:, _CLASS[sh_a[i].l].index(cls), _CLASS[sh_b[j].l].index(cls)] = values[k]
        Da, Db = shell_rotation(sh_a[i].l, Q), shell_rotation(sh_b[j].l, Q)
        block = np.einsum("nki,nkq,nqj->nij", Da, loc, Db)
        out[:, off_a[i] : off_a[i + 1], off_b[j] : off_b[j + 1]] = block
    return out


def envelope_at(table: PairTable, R: np.ndarray) -> np.ndarray:
    k = np.clip(np.floor(R / GRID_STEP).astype(int), 0, len(table.envelope) - 1)
    return np.where(R > GRID_MAX, 0.0, table.envelope[k])


# ---- rotations of shells


def _A_d() -> np.ndarray:
    """(5, 3, 3) symmetric matrices of the five d functions: d_k(x) ~ x^T A_k x."""
    out = np.zeros((5, 3, 3))
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]  # xx yy zz xy xz yz
    for k in range(5):
        for c, (i, j) in enumerate(pairs):
            w = SPH[2][k, c] if i == j else 0.5 * SPH[2][k, c]
            out[k, i, j] += w
            if i != j:
                out[k, j, i] += w
    return out


A_D = _A_d()


def shell_rotation(l: int, M: np.ndarray) -> np.ndarray:  # noqa: E741
    """(..., 2l+1, 2l+1) D such that the coefficients of a shell expressed in coordinates
    x' = M x are D c (M orthogonal, tblite component order)."""
    M = np.asarray(M)
    if l == 0:
        return np.ones((*M.shape[:-2], 1, 1))
    if l == 1:
        return _TB_FROM_XYZ @ M @ _TB_FROM_XYZ.T
    lead = M.shape[:-2]
    M = M.reshape(-1, 1, 3, 3)
    B = M @ A_D[None] @ np.swapaxes(M, -1, -2)  # (T, 5, 3, 3): M A_q M^T
    D = B.reshape(len(B), 5, 9) @ A_D.reshape(5, 9).T / 1.5  # [t, q, k] = <A_k, B_q> / 1.5
    return np.swapaxes(D, -1, -2).reshape(*lead, 5, 5)


def rotation_to(u_from: np.ndarray, u_to: np.ndarray) -> np.ndarray:
    """(N, 3, 3) smallest rotations taking unit vectors u_from to u_to (Rodrigues); a half
    turn about an axis perpendicular to u_from when the two are antiparallel."""
    a, b = np.atleast_2d(u_from), np.atleast_2d(u_to)
    v = np.cross(a, b)
    c = np.sum(a * b, axis=1)
    vx = np.zeros((len(a), 3, 3))
    vx[:, 0, 1], vx[:, 0, 2], vx[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    vx[:, 1, 0], vx[:, 2, 0], vx[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    flip = c < -1 + 1e-9
    denom = np.where(flip, 1.0, 1.0 + c)[:, None, None]
    out = np.eye(3)[None] + vx + vx @ vx / denom
    if flip.any():
        ref = np.where(np.abs(a[flip, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        perp = np.cross(a[flip], ref)
        perp /= np.linalg.norm(perp, axis=1, keepdims=True)
        out[flip] = 2 * perp[:, :, None] * perp[:, None, :] - np.eye(3)[None]
    return out


def local_frames(u: np.ndarray) -> np.ndarray:
    """(N, 3, 3) orthogonal Q with rows (e1, e2, u): x' = Q x puts u on the local z axis."""
    ref = np.where(np.abs(u[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = ref - np.sum(ref * u, axis=1, keepdims=True) * u
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    return np.stack([e1, e2, u], axis=1)


# ---- lobe candidates at a site (rule 1)

TIE = 0.95  # a second local maximum within 5% of the best is a tied lobe (two faces of a pi)
ISOTROPIC = 1.05  # max / min amplitude below this: the site's AO part has no direction (s)
N_NEIGHBOURS = 6


@cache
def sphere_directions(n: int = N_DIRECTIONS) -> np.ndarray:
    """(n, 3) quasi-uniform unit vectors (Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = k * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


@cache
def _direction_neighbours(k: int = N_NEIGHBOURS) -> np.ndarray:
    """(n, k) indices of the k nearest sphere directions of each direction."""
    dots = sphere_directions() @ sphere_directions().T
    np.fill_diagonal(dots, -np.inf)
    return np.argsort(-dots, axis=1)[:, :k]


def _positions(t: MoleculeTokens) -> np.ndarray:
    return np.array([a.position for a in t.atoms], dtype=float).reshape(-1, 3)


def bonded_neighbours(t: MoleculeTokens, atom: int) -> np.ndarray:
    """Atoms within 1.3x the covalent-radii sum of `atom` (`tokens.bonded_matrix`)."""
    return np.flatnonzero(bonded_matrix(t)[atom])


def token_populations(o: OrbitalToken, t: MoleculeTokens) -> np.ndarray:
    """Mulliken population of the token on every atom."""
    c = o.coeff_vector
    return np.bincount(t.ao_atom, weights=c * (t.S @ c), minlength=len(t.atoms))


def _other_centres(o: OrbitalToken, site: int) -> tuple[np.ndarray, np.ndarray]:
    d = o.descriptors
    keep = np.asarray(d.all_center_atoms) != site
    return np.asarray(d.all_center_atoms)[keep].astype(int), np.asarray(d.all_center_weights)[keep]


def away_direction(o: OrbitalToken, t: MoleculeTokens, site: int) -> np.ndarray | None:
    """Unit vector from the site away from the token's other centres (weighted by their
    populations); for a one-centre token, away from the site's bonded neighbours weighted by
    the token's Mulliken population on each. None when there is nothing to point away from."""
    atoms, w = _other_centres(o, site)
    if not len(atoms):
        atoms = bonded_neighbours(t, site)
        if not len(atoms):
            return None
        w = np.clip(token_populations(o, t)[atoms], 0.0, None)
        w = w if w.sum() > 1e-6 else np.ones(len(atoms))
    pos = _positions(t)
    bonds = pos[atoms] - pos[site]
    v = -(w[:, None] * bonds / np.linalg.norm(bonds, axis=1, keepdims=True)).sum(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else None


def admissible(o: OrbitalToken, t: MoleculeTokens, site: int) -> np.ndarray:
    """(n_dir,) the hemisphere rule: n . (x_c - x_site) <= 0 for every other centre c; for a
    one-centre token, the hemisphere away from the population-weighted bonded neighbours.
    When no direction satisfies every centre, the least-violating ones (within 0.05)."""
    dirs = sphere_directions()
    atoms, _ = _other_centres(o, site)
    if len(atoms):
        pos = _positions(t)
        bonds = pos[atoms] - pos[site]
        worst = (dirs @ (bonds / np.linalg.norm(bonds, axis=1, keepdims=True)).T).max(axis=1)
        ok = worst <= 1e-9
        return ok if ok.any() else worst <= worst.min() + 0.05
    away = away_direction(o, t, site)
    return dirs @ away >= 0 if away is not None else np.ones(len(dirs), dtype=bool)


def site_candidates(o, t, site: int, sphere_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tied lobes (m, 3), m = 1 or 2; the opposite face of the first) of token `o` at `site`.
    `sphere_values`: (n_dir, nao_site) the site's AO values on the 1.5 Angstrom sphere."""
    dirs = sphere_directions()
    amp = np.abs(sphere_values @ o.coeff_vector[t.ao_atom == site])
    if amp.max() <= ISOTROPIC * max(amp.min(), 1e-300):  # an s function: no lobe of its own
        away = away_direction(o, t, site)
        if away is not None:
            return away[None], -away
    masked = np.where(admissible(o, t, site), amp, -np.inf)
    local = np.isfinite(masked) & (masked >= masked[_direction_neighbours()].max(axis=1))
    order = np.flatnonzero(local)[np.argsort(-masked[local])]
    best = int(order[0]) if len(order) else int(np.argmax(masked))
    cands = [dirs[best]]
    if len(order) > 1 and masked[order[1]] >= TIE * masked[best]:
        cands.append(dirs[order[1]])
    return np.array(cands), -dirs[best]


def molecule_candidates(t: MoleculeTokens) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """(token position in t.orbitals, site atom) -> `site_candidates`, for every centre of every
    token; the site's AO values on its sphere computed once per atom."""
    from ksmech.physics.gto import ao_values

    values: dict[int, np.ndarray] = {}
    out = {}
    for k, o in enumerate(t.orbitals):
        for site in np.asarray(o.descriptors.all_center_atoms, dtype=int):
            site = int(site)
            if site not in values:
                z = np.array([t.atoms[site].atomic_number])
                values[site] = ao_values(
                    z, np.zeros((1, 3)), SPHERE_RADIUS_ANG * sphere_directions()
                )
            out[k, site] = site_candidates(o, t, site, values[site])
    return out


# ---- placements, clash guard, overlaps

TIER1 = ((0, 0), (0, 1), (1, 0), (1, 1))  # direction slots: the tied lobes of both tokens
TIER2 = ((2, 0), (2, 1), (0, 2), (1, 2), (2, 2))  # slot 2: the opposite face


@dataclass
class _Side:
    """Bohr geometry and coefficient rows of the tokens of one fragment."""

    z: np.ndarray
    xyz: np.ndarray  # (n_atoms, 3) bohr
    starts: np.ndarray  # first AO of each atom
    C: np.ndarray  # (n_tokens, nao)
    norms: np.ndarray  # (n_tokens, n_atoms) coefficient norm per atom
    CF: np.ndarray  # (n_tokens, nao) C times the AO Fock diagonal F_mu,mu (eV)

    @classmethod
    def of(cls, t: MoleculeTokens, index: list[int]) -> _Side:
        z = np.array([a.atomic_number for a in t.atoms])
        C = np.array([t.orbitals[k].coeff_vector for k in index]).reshape(len(index), -1)
        norms = np.sqrt(np.stack([np.bincount(t.ao_atom, w, len(z)) for w in C**2]))
        starts = np.array([s.start for s in atom_ao_slices(z)])
        CF = C * ao_energies(t)[None, :]
        return cls(z, _positions(t) / BOHR_ANG, starts, C, norms.reshape(len(index), -1), CF)

    @classmethod
    def cap(cls, t: MoleculeTokens, k: int) -> _Side:
        """Token k alone, on its local cap: the cap atoms, and its LMO truncated to their AOs."""
        atoms = cap_atoms(t.orbitals[k], t)
        aos = [np.flatnonzero(t.ao_atom == a) for a in atoms]
        C = t.orbitals[k].coeff_vector[np.concatenate(aos)][None, :]
        local = np.repeat(np.arange(len(atoms)), [len(x) for x in aos])
        norms = np.sqrt(np.bincount(local, C[0] ** 2, len(atoms)))[None, :]
        starts = np.cumsum([0] + [len(x) for x in aos[:-1]])
        z = np.array([t.atoms[a].atomic_number for a in atoms])
        CF = C * ao_energies(t)[np.concatenate(aos)][None, :]
        return cls(z, _positions(t)[atoms] / BOHR_ANG, starts, C, norms, CF)


def ao_energies(t: MoleculeTokens) -> np.ndarray:
    """(nao,) the AO energies of the Wolfsberg-Helmholz resonance integral (eV): the molecule's
    own GFN2 Fock diagonal, averaged over the 2l + 1 components of each shell. GFN2's
    anisotropic terms split the lab-frame diagonal of a p or d shell (0.06-0.2 eV on C and O p,
    0.4 eV on Br d) in a way that depends on the record's orientation; the shell average is
    rotation-invariant, one valence-state energy per shell."""
    f = np.diag(t.F) * HARTREE_TO_EV
    z = np.array([a.atomic_number for a in t.atoms])
    out = f.copy()
    for atom, sl in enumerate(atom_ao_slices(z)):
        k = sl.start
        for sh in element_shells(int(z[atom])):
            out[k : k + sh.n_sph] = f[k : k + sh.n_sph].mean()
            k += sh.n_sph
    return out


def cap_atoms(o: OrbitalToken, t: MoleculeTokens) -> np.ndarray:
    """The token's local cap: its centre atoms and their first bonded neighbours (hydrogens
    included), sorted. Only the cap of the acceptor is placed (rule 2)."""
    centres = {int(a) for a in o.descriptors.all_center_atoms}
    cap = set(centres)
    for a in centres:
        cap.update(int(n) for n in bonded_neighbours(t, a))
    return np.array(sorted(cap))


@dataclass
class _Entries:
    """One row per (token, site): the token's row in its side, the site atom, three direction
    slots (two tied lobes, the opposite face of the first) and which slots exist."""

    tok: np.ndarray
    site: np.ndarray
    dirs: np.ndarray  # (E, 3, 3)
    valid: np.ndarray  # (E, 3)


def _entries(t: MoleculeTokens, index: list[int], sites=None) -> _Entries:
    cands = _candidates(t)
    tok, site, dirs, valid = [], [], [], []
    for i, k in enumerate(index):
        centres = sites[i] if sites is not None else t.orbitals[k].descriptors.all_center_atoms
        for a in np.asarray(centres, dtype=int):
            lobes, opposite = cands[k, int(a)]
            slots = np.zeros((3, 3))
            slots[: len(lobes)], slots[2] = lobes, opposite
            tok.append(i), site.append(int(a)), dirs.append(slots)
            valid.append([True, len(lobes) > 1, True])
    return _Entries(
        np.array(tok), np.array(site), np.array(dirs).reshape(-1, 3, 3), np.array(valid)
    )


@dataclass
class _Placements:
    sp: np.ndarray  # site-pair id: entry of F * n_entries(G) + entry of G
    rot: np.ndarray  # (P, 3, 3) rotation of fragment G
    anchor: np.ndarray  # (P, 3) bohr: G's site, the rotation centre
    shift: np.ndarray  # (P, 3) bohr: where G's site goes (F's site + d u)
    tok_r: np.ndarray
    tok_s: np.ndarray
    axis: np.ndarray  # (P, 3) the donor lobe u, the approach axis
    roll: np.ndarray  # (P,) roll step about the axis (0: the smallest rotation)


N_ROLLS = 6  # roll angles tried about the approach axis when a placement clashes


def _axis_rotations(u: np.ndarray, theta: float) -> np.ndarray:
    """(N, 3, 3) rotations by theta about the unit axes u (Rodrigues)."""
    ux = np.zeros((len(u), 3, 3))
    ux[:, 0, 1], ux[:, 0, 2], ux[:, 1, 2] = -u[:, 2], u[:, 1], -u[:, 0]
    ux[:, 1, 0], ux[:, 2, 0], ux[:, 2, 1] = u[:, 2], -u[:, 1], u[:, 0]
    outer = u[:, :, None] * u[:, None, :]
    return np.cos(theta) * np.eye(3)[None] + np.sin(theta) * ux + (1 - np.cos(theta)) * outer


def _rolled(pl: _Placements, which: np.ndarray) -> _Placements:
    """The placements `which` turned about their approach axes by every other roll angle; the
    acceptor lobe stays antiparallel to the donor lobe and its site stays at distance d."""
    parts = []
    for k in range(1, N_ROLLS):
        R = _axis_rotations(pl.axis[which], 2 * np.pi * k / N_ROLLS)
        parts.append((R @ pl.rot[which], np.full(len(which), k)))
    fields = {f: np.concatenate([getattr(pl, f)[which]] * (N_ROLLS - 1)) for f in _PL_FIELDS}
    fields["rot"] = np.concatenate([r for r, _ in parts]) if parts else pl.rot[which]
    fields["roll"] = np.concatenate([k for _, k in parts]) if parts else pl.roll[which]
    return _Placements(**fields)


def _placements(F, G, EF, EG, sp: np.ndarray, combos, d_bohr: float) -> _Placements:
    ef, eg = sp // len(EG.tok), sp % len(EG.tok)
    parts = []
    for p, q in combos:
        keep = EF.valid[ef, p] & EG.valid[eg, q]
        parts.append((sp[keep], ef[keep], eg[keep], np.full(keep.sum(), p), np.full(keep.sum(), q)))
    sp, ef, eg, p, q = (np.concatenate(x).astype(int) for x in zip(*parts, strict=True))
    u, v = EF.dirs[ef, p], EG.dirs[eg, q]
    rot = rotation_to(v, -u)
    shift = F.xyz[EF.site[ef]] + d_bohr * u
    zero = np.zeros(len(sp), dtype=int)
    return _Placements(sp, rot, G.xyz[EG.site[eg]], shift, EF.tok[ef], EG.tok[eg], u, zero)


def _placed(G: _Side, pl: _Placements, sel) -> np.ndarray:
    rel = G.xyz[None] - pl.anchor[sel][:, None]
    return np.einsum("pij,pbj->pbi", pl.rot[sel], rel) + pl.shift[sel][:, None]


def _by_acceptor(pl: _Placements, which: np.ndarray):
    """(acceptor token, its placements among `which`, a view with the token as row 0)."""
    for s in np.unique(pl.tok_s[which]):
        sub = which[pl.tok_s[which] == s]
        fields = {f: getattr(pl, f)[sub] for f in _PL_FIELDS}
        view = _Placements(**(fields | {"tok_s": 0 * sub}))
        yield int(s), sub, view


def _clashes(F: _Side, caps: dict, pl: _Placements, limit_bohr: float) -> np.ndarray:
    """(P,) True where an atom of the placed acceptor cap lies closer than the limit to an atom
    of the donor fragment F."""
    out = np.zeros(len(pl.sp), dtype=bool)
    for s, sub, view in _by_acceptor(pl, np.arange(len(pl.sp))):
        G = caps[s]
        step = max(1, int(2e6 // (len(F.z) * len(G.z))))
        for k in range(0, len(sub), step):
            sel = slice(k, k + step)
            diff = _placed(G, view, sel)[:, None, :, :] - F.xyz[None, :, None, :]
            hit = (np.einsum("pabi,pabi->pab", diff, diff) < limit_bohr**2).any(axis=(1, 2))
            out[sub[sel]] = hit
    return out


def _gather(C: np.ndarray, tok: np.ndarray, start: np.ndarray, n: int) -> np.ndarray:
    return C[tok[:, None], start[:, None] + np.arange(n)[None, :]]


_XYZ_FROM_TB, _TB_FROM_XYZ_IDX = [2, 0, 1], [1, 2, 0]  # p shells: (y, z, x) <-> (x, y, z)


def rotate_coeffs(l: int, c: np.ndarray, M: np.ndarray) -> np.ndarray:  # noqa: E741
    """Coefficients (T, 2l+1) of a shell expressed in coordinates x' = M x (M: (T, 3, 3)); the
    same as `shell_rotation(l, M) @ c` without building the rotation matrices."""
    if l == 0:
        return c
    if l == 1:
        return np.einsum("tij,tj->ti", M, c[:, _XYZ_FROM_TB])[:, _TB_FROM_XYZ_IDX]
    C = (c @ A_D.reshape(5, 9)).reshape(-1, 3, 3)  # sum_k c_k A_k
    return (M @ C @ np.swapaxes(M, -1, -2)).reshape(-1, 9) @ A_D.reshape(5, 9).T / 1.5


def _rotate_shells(z: int, c: np.ndarray, M: np.ndarray) -> list[np.ndarray]:
    """Per shell of element z: coefficients (T, 2l+1) expressed in the frames M (T, 3, 3)."""
    out, k = [], 0
    for sh in element_shells(z):
        out.append(rotate_coeffs(sh.l, c[:, k : k + sh.n_sph], M))
        k += sh.n_sph
    return out


def _group_overlap(F, G, za, zb, p, a, b, V, R, rot, tok_r, tok_s) -> np.ndarray:
    """(3, T) contributions of every kept triple (placement p, atom a of F, atom b of G) to
    l_r^T S l_s, (f l_r)^T S l_s and l_r^T S (f l_s), f the AO Fock diagonals: the overlap and
    the two halves of the AO-level resonance sum."""
    u = np.where(R[:, None] > 1e-10, V / np.maximum(R, 1e-10)[:, None], [[0.0, 0.0, 1.0]])
    Q = local_frames(u)
    n_a = sum(s.n_sph for s in element_shells(za))
    n_b = sum(s.n_sph for s in element_shells(zb))
    c_r = [_rotate_shells(za, _gather(X, tok_r[p], F.starts[a], n_a), Q) for X in (F.C, F.CF)]
    M = Q @ rot[p]
    c_s = [_rotate_shells(zb, _gather(X, tok_s[p], G.starts[b], n_b), M) for X in (G.C, G.CF)]
    table = pair_table(int(za), int(zb))
    out = np.zeros((3, len(p)))
    for (i, j), spline in table.splines.items():
        S_loc = spline(R)  # (n_classes, T)
        ls_a, ls_b = element_shells(za)[i].l, element_shells(zb)[j].l
        for k, cls in enumerate(table.classes[i, j]):
            ia, ib = _CLASS[ls_a].index(cls), _CLASS[ls_b].index(cls)
            r0, r1 = c_r[0][i][:, ia], c_r[1][i][:, ia]
            s0, s1 = c_s[0][j][:, ib] * S_loc[k], c_s[1][j][:, ib] * S_loc[k]
            out += (r0 * s0, r1 * s0, r0 * s1)
    return out


def _chunk_overlaps(F: _Side, G: _Side, pl: _Placements, sel, tol: float) -> np.ndarray:
    """(3, len(sel)) signed S_rs, (f l_r)^T S l_s and l_r^T S (f l_s) of the placements `sel`."""
    rot, tok_r, tok_s = pl.rot[sel], pl.tok_r[sel], pl.tok_s[sel]
    V = _placed(G, pl, sel)[:, None, :, :] - F.xyz[None, :, None, :]  # (P, n_F, n_G, 3)
    R = np.linalg.norm(V, axis=-1)
    out = np.zeros((3, len(tok_r)))
    for za in np.unique(F.z):
        for zb in np.unique(G.z):
            ia, ib = np.flatnonzero(F.z == za), np.flatnonzero(G.z == zb)
            bound = F.norms[tok_r][:, ia, None] * G.norms[tok_s][:, None, ib]
            bound = bound * envelope_at(pair_table(int(za), int(zb)), R[:, ia][:, :, ib])
            p, a, b = np.nonzero(bound >= tol)
            if not len(p):
                continue
            a, b = ia[a], ib[b]
            vals = _group_overlap(F, G, za, zb, p, a, b, V[p, a, b], R[p, a, b], rot, tok_r, tok_s)
            out += [np.bincount(p, weights=v, minlength=len(tok_r)) for v in vals]
    return out


def _overlaps(F: _Side, caps: dict, pl: _Placements, which: np.ndarray, tol: float):
    """(|S|, beta) for the placements `which` (indices): the donor's LMO against the acceptor's
    LMO on its placed cap. beta = sum_mu,nu l_mu,r l_nu,s S_mu,nu (F_mu,mu + F_nu,nu) / 2 (eV,
    K = 1), times the sign of S: the AO-level resonance integral in the phase that makes S
    positive, so it no longer depends on the arbitrary signs of the two LMOs."""
    value, beta = np.zeros(len(pl.sp)), np.zeros(len(pl.sp))
    for s, sub, view in _by_acceptor(pl, which):
        G = caps[s]
        step = max(1, int(4e5 // (len(F.z) * len(G.z))))
        for k in range(0, len(sub), step):
            sel = np.arange(k, min(k + step, len(sub)))
            S, half_r, half_s = _chunk_overlaps(F, G, view, sel, tol)
            value[sub[sel]] = np.abs(S)
            beta[sub[sel]] = 0.5 * (half_r + half_s) * np.sign(S)
    return value[which], beta[which]


# ---- the block of one molecule pair


@dataclass
class Contacts:
    """Pair-track contact overlaps of a block: |S| maximized over the donor's centres x the
    acceptor's centres, the argmax site pair (atom indices within each molecule) and the CLASH
    flag (every placement of that site pair clashed)."""

    S: np.ndarray  # (n_rows, n_cols)
    sites: np.ndarray  # (n_rows, n_cols, 2)
    clash: np.ndarray  # (n_rows, n_cols) bool
    beta: np.ndarray  # (n_rows, n_cols) AO-level resonance integral at K = 1 (eV), S's phase


def _site_pair_values(n_sp, pl, clash, values, betas, evaluated):
    """Per site pair: |S| and beta of the evaluated clash-free placement with the largest |S|,
    or, when every placement clashed, of the one with the smallest |S| (flagged)."""
    ok = np.zeros(n_sp, dtype=bool)
    ok[pl.sp[~clash]] = True
    sp, hit = pl.sp[evaluated], clash[evaluated]
    key = np.where(ok[sp], values, -values)  # the largest |S|, or the smallest when failed
    key = np.where(hit & ok[sp], -np.inf, key)  # a clashing placement of a clean site pair
    order = np.lexsort((key, sp))
    last = order[np.r_[sp[order][1:] != sp[order][:-1], True]]
    value, beta = np.zeros(n_sp), np.zeros(n_sp)
    value[sp[last]], beta[sp[last]] = values[last], betas[last]
    return value, beta, ~ok


def _concat(a: _Placements, b: _Placements) -> _Placements:
    return _Placements(**{f: np.concatenate([getattr(a, f), getattr(b, f)]) for f in _PL_FIELDS})


def _tier(F, G, caps, EF, EG, sp, combos, d_bohr: float, limit: float):
    """The placements of one tier and their clash flags; a placement that clashes at the
    smallest rotation is also tried at the other roll angles about its approach axis."""
    base = _placements(F, G, EF, EG, sp, combos, d_bohr)
    clash = _clashes(F, caps, base, limit)
    extra = _rolled(base, np.flatnonzero(clash))
    return _concat(base, extra), np.concatenate([clash, _clashes(F, caps, extra, limit)])


def _evaluate_block(F, G, caps, EF, EG, config: OrientedConfig):
    """|S|, beta and CLASH flag of every site pair: tier 1 (tied lobes), then tier 2 (opposite
    faces) for the site pairs whose tier-1 placements all clash at every roll angle. A failed
    site pair takes the placement with the smallest |S| among its roll-0 ones."""
    n_sp = len(EF.tok) * len(EG.tok)
    d_bohr, limit = config.distance / BOHR_ANG, config.clash / BOHR_ANG
    pl1, c1 = _tier(F, G, caps, EF, EG, np.arange(n_sp), TIER1, d_bohr, limit)
    ok1 = np.zeros(n_sp, dtype=bool)
    ok1[pl1.sp[~c1]] = True
    pl2, c2 = _tier(F, G, caps, EF, EG, np.flatnonzero(~ok1), TIER2, d_bohr, limit)
    ok2 = np.zeros(n_sp, dtype=bool)
    ok2[pl2.sp[~c2]] = True
    failed = ~ok1 & ~ok2
    pl, clash = _concat(pl1, pl2), np.concatenate([c1, c2])
    in_tier1 = np.arange(len(clash)) < len(c1)
    evaluated = (~clash & (in_tier1 | ~ok1[pl.sp])) | (failed[pl.sp] & (pl.roll == 0))
    values, betas = _overlaps(F, caps, pl, np.flatnonzero(evaluated), config.tol)
    return _site_pair_values(n_sp, pl, clash, values, betas, evaluated)


_PL_FIELDS = ("sp", "rot", "anchor", "shift", "tok_r", "tok_s", "axis", "roll")


def block_contacts(
    t_F,
    rows: list[int],
    t_G,
    cols: list[int],
    config: OrientedConfig,
    sites_F=None,
    sites_G=None,
    intra: bool = False,
) -> Contacts:
    """Contacts of tokens `rows` of t_F (positions in t_F.orbitals) with tokens `cols` of t_G.
    `sites_F` / `sites_G` restrict each token's sites (default: all its centres). `intra`: t_F
    and t_G are one molecule, so a site pair that is a bond interacting with itself
    (`tokens.site_pairs_excluded`, `config.site_weight`) only counts when the token pair has no
    other site pair (it is then not an arrow candidate either)."""
    shape = (len(rows), len(cols))
    S, beta, clash = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=bool)
    sites = np.full((*shape, 2), -1, dtype=int)
    if not rows or not cols:
        return Contacts(S, sites, clash, beta)
    F, G = _Side.of(t_F, rows), _Side.of(t_G, cols)
    caps = {i: _Side.cap(t_G, k) for i, k in enumerate(cols)}
    EF, EG = _entries(t_F, rows, sites_F), _entries(t_G, cols, sites_G)
    value, b_sp, flagged = _evaluate_block(F, G, caps, EF, EG, config)
    ef, eg = np.divmod(np.arange(len(value)), len(EG.tok))
    pair = EF.tok[ef] * len(cols) + EG.tok[eg]
    key = np.where(flagged, value - 10.0, value)  # a clash-free site pair always wins
    if intra:
        donors = [t_F.orbitals[rows[i]] for i in EF.tok[ef]]
        acceptors = [t_G.orbitals[cols[j]] for j in EG.tok[eg]]
        w = config.site_weight
        own = site_pairs_excluded(t_F, donors, EF.site[ef], acceptors, EG.site[eg], w)
        key = np.where(own, key - 100.0, key)  # then any admissible site pair wins
    order = np.lexsort((key, pair))
    last = order[np.r_[pair[order][1:] != pair[order][:-1], True]]
    i, j = np.divmod(pair[last], len(cols))
    S[i, j], beta[i, j], clash[i, j] = value[last], b_sp[last], flagged[last]
    sites[i, j, 0], sites[i, j, 1] = EF.site[ef[last]], EG.site[eg[last]]
    return Contacts(S, sites, clash, beta)


def site_contact(t_F, r: int, a: int, t_G, s: int, b: int, config: OrientedConfig):
    """(|S|, CLASH) for donor token r at site a and acceptor token s at site b, the form used
    where the site is known (labels, the decoder's acceptor step, applying an arrow)."""
    c = block_contacts(t_F, [r], t_G, [s], config, sites_F=[[a]], sites_G=[[b]])
    return float(c.S[0, 0]), bool(c.clash[0, 0])


# ---- caches


def _mol_key(t: MoleculeTokens) -> str:
    return f"{t.smiles}|{t.conformer_id}|{t.method}"


_CANDIDATES: dict[str, dict] = {}
_BLOCKS: dict[str, tuple[Contacts, str]] = {}
_MEMORY_LIMIT = 4096


def _candidates(t: MoleculeTokens) -> dict:
    key = _mol_key(t)
    if key not in _CANDIDATES:
        if len(_CANDIDATES) > _MEMORY_LIMIT:
            _CANDIDATES.clear()
        _CANDIDATES[key] = molecule_candidates(t)
    return _CANDIDATES[key]


def _block_config(block: str, config: OrientedConfig, intra: bool) -> dict:
    """The configuration a contact block is stamped with (contact mode, direction count,
    distance, filter threshold)."""
    return {
        "contact_mode": f"oriented_v{ORIENTED_VERSION}",
        "block": block,
        "n_directions": N_DIRECTIONS,
        "n_rolls": N_ROLLS,
        "distance": config.distance,
        "tol": config.tol,
        "clash": config.clash,
        "site_weight": config.site_weight if intra else None,
    }


def _block_key(t_F, t_G, block: str, config: OrientedConfig, intra: bool = False) -> str:
    raw = "|".join(
        [f"v{ORIENTED_VERSION}", _mol_key(t_F), _mol_key(t_G), block, str(config.distance)]
        + [str(config.tol), str(config.clash)]
        + ([f"intra{config.site_weight}"] if intra else [])
    )
    return hashlib.sha256(raw.encode()).hexdigest()


STALE_BLOCKS: Counter = Counter()  # STALE_ARTIFACT recomputations, by the first change


def _arrays(value: Contacts) -> dict[str, np.ndarray]:
    return {"S": value.S, "sites": value.sites, "clash": value.clash, "beta": value.beta}


def _read_block(path: Path, cfg: dict, inputs: dict) -> Contacts | None:
    """The cached block, or None when absent or stale (counted in STALE_BLOCKS)."""
    if not (path.is_file() and path.stat().st_size > 0):
        return None
    try:
        prov.verify(path, "contact_block", ORIENTED_VERSION, cfg, inputs)
    except prov.StaleArtifact as exc:
        STALE_BLOCKS[exc.changes[0]] += 1
        log.info("recomputing %s", exc)
        return None
    with np.load(path, allow_pickle=False) as z:
        return Contacts(z["S"], z["sites"], z["clash"], z["beta"])


def _load_or_compute(key: str, config: OrientedConfig, compute, cfg: dict, inputs: dict):
    """The block from memory, then disk (verified), else computed; written with its header
    whenever the disk copy is missing. The file used is reported to `provenance.note_input`."""
    path = Path(config.cache_dir) / f"{key}.npz" if config.cache_dir else None
    if key in _BLOCKS:
        value, digest = _BLOCKS[key]
        on_disk = path is None or path.is_file()
    else:
        value = _read_block(path, cfg, inputs) if path is not None else None
        on_disk = value is not None or path is None
        if value is None:
            value = compute()
        digest = prov.array_digest(_arrays(value))
        if len(_BLOCKS) > _MEMORY_LIMIT:
            _BLOCKS.clear()
        _BLOCKS[key] = (value, digest)
    if path is not None:
        if not on_disk:
            header = prov.stamp("contact_block", ORIENTED_VERSION, inputs, cfg, digest)
            prov.save_npz(path, _arrays(value), header, compressed=False)
        prov.note_input(prov.file_input_name(path), digest)
    return value


def oriented_block(t_F, t_G, block: str, config: OrientedConfig, intra: bool = False) -> Contacts:
    """Cached contacts of every occupied token of t_F with every virtual ("occ_virt") or
    occupied ("occ_occ") token of t_G; `intra`: the two are one molecule (two fragments that are
    copies of one molecule are not), see `block_contacts`."""
    occ_F = [k for k, o in enumerate(t_F.orbitals) if o.kind == OrbitalKind.OCCUPIED]
    want = OrbitalKind.OCCUPIED if block == "occ_occ" else OrbitalKind.VIRTUAL
    cols = [k for k, o in enumerate(t_G.orbitals) if o.kind == want]
    key = _block_key(t_F, t_G, block, config, intra)
    inputs = {"tokens:F": token_digest(t_F), "tokens:G": token_digest(t_G)}
    return _load_or_compute(
        key,
        config,
        lambda: block_contacts(t_F, occ_F, t_G, cols, config, intra=intra),
        _block_config(block, config, intra),
        inputs,
    )
