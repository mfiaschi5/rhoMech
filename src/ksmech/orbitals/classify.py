"""Classify a localized orbital from its Mulliken populations and AO composition.

Decision order for one orbital (occupied or virtual):

1. One atom carries at least `one_atom_frac` of the population: `lone_pair` (occupied); for a
   virtual, `empty_p` / `empty_s` / `empty_d` by the dominant angular momentum.
2. The top two atoms carry at least `two_atom_frac`: a two-centre orbital.
   - Occupied, and exactly one of the two atoms is an ionic metal (`ionic_metals`: Li, Na, Mg,
     Al, K, Zn): `lone_pair` on the other atom with `metal_bound = True` and `metal_atom` set.
     A C-Mg or C-Li bond is the carbanion-like donor of Grignard and organolithium additions.
   - Otherwise the bond-axis test projects each atom's p-coefficient vector on the unit vector
     between the two atoms; |cos| > `sigma_cos` is sigma-like. At least one sigma-like atom (or
     no p on either atom): `sigma` / `sigma_star`; both pi-like: `pi` / `pi_star`. The Mulliken
     overlap population between the two atoms must be positive for an occupied orbital and
     negative for a virtual one (the sign change across the bond); otherwise `unknown`.
3. Otherwise the orbital is spread over three or more atoms: `delocalized_pi` /
   `delocalized_pi_star` when its s-AO share is below `max_s_share_pi`, else `unknown`. The
   parents are every atom carrying at least `member_min_pop`.

`classify_block` then ranks the orbitals of one block that share the same two parents (a
double or triple bond: 2 or 3 LMOs with the right bonding sign) by mean |cos|: the largest is
the sigma, the others pi, regardless of `sigma_cos`. PM tilts alkyne orbitals slightly towards
banana bonds, which otherwise puts the sigma right at the threshold.

Parents are listed by descending population, so `parent_atoms[0]` is the dominant atom.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from ksmech.orbitals.localize import P_AXIS_ORDER


class OrbitalType(StrEnum):
    LONE_PAIR = "lone_pair"
    SIGMA = "sigma"
    PI = "pi"
    DELOCALIZED_PI = "delocalized_pi"
    SIGMA_STAR = "sigma_star"
    PI_STAR = "pi_star"
    EMPTY_P = "empty_p"
    EMPTY_S = "empty_s"
    EMPTY_D = "empty_d"
    DELOCALIZED_PI_STAR = "delocalized_pi_star"
    UNKNOWN = "unknown"


OCCUPIED_TYPES = frozenset(
    {
        OrbitalType.LONE_PAIR,
        OrbitalType.SIGMA,
        OrbitalType.PI,
        OrbitalType.DELOCALIZED_PI,
        OrbitalType.UNKNOWN,
    }
)
VIRTUAL_TYPES = frozenset(
    {
        OrbitalType.SIGMA_STAR,
        OrbitalType.PI_STAR,
        OrbitalType.EMPTY_P,
        OrbitalType.EMPTY_S,
        OrbitalType.EMPTY_D,
        OrbitalType.DELOCALIZED_PI_STAR,
        OrbitalType.UNKNOWN,
    }
)
DELOCALIZED_TYPES = frozenset({OrbitalType.DELOCALIZED_PI, OrbitalType.DELOCALIZED_PI_STAR})
# "not cleanly classified": what the build summary reports as delocalized-or-unknown
UNRESOLVED_TYPES = DELOCALIZED_TYPES | {OrbitalType.UNKNOWN}
PI_TYPES = frozenset(
    {
        OrbitalType.PI,
        OrbitalType.PI_STAR,
        OrbitalType.DELOCALIZED_PI,
        OrbitalType.DELOCALIZED_PI_STAR,
    }
)


@dataclass(frozen=True)
class ClassifyConfig:
    one_atom_frac: float = 0.85  # lone pair / empty orbital: population on the top atom
    two_atom_frac: float = 0.85  # bond: population on the top two atoms
    sigma_cos: float = 0.7  # |cos(p vector, bond axis)| above this = sigma-like
    min_p_norm: float = 0.05  # an atom with a smaller p-coefficient norm has no p character
    max_s_share_pi: float = 0.25  # a delocalized orbital is pi-type below this s-AO share
    member_min_pop: float = 0.05  # parents of a delocalized orbital carry at least this
    # Li, Na, Mg, Al, K, Zn: a bond from one of these is a metal-bound lone pair on the partner
    ionic_metals: frozenset[int] = frozenset({3, 11, 12, 13, 19, 30})
    rank_bond_orbitals: bool = True  # 2 or 3 LMOs on one pair: largest |cos| = sigma
    metal_min_pop: float = 0.10  # a one-atom lone pair is metal-bound above this metal share


@dataclass(frozen=True)
class Classification:
    type: OrbitalType
    occupied: bool
    parent_atoms: tuple[int, ...]  # record atom indices, by descending population
    parent_weights: tuple[float, ...]  # Mulliken population on each parent
    populations: np.ndarray  # (n_atoms,) Mulliken population on every atom
    s_share: float
    p_share: float
    d_share: float
    overlap_population: float  # between the two top atoms of a bond orbital, else NaN
    axis_cosines: tuple[float, ...]  # |cos| per parent with p character (bond orbitals only)
    metal_bound: bool = False  # lone pair that is really a bond to an ionic metal
    metal_atom: int = -1  # that metal's atom index, else -1

    @property
    def bonding_sign_ok(self) -> bool:
        """Two-centre orbital whose overlap population has the sign its block requires."""
        if len(self.parent_atoms) != 2 or np.isnan(self.overlap_population):
            return False
        return (self.overlap_population > 0) == self.occupied


def atom_populations(c: np.ndarray, S: np.ndarray, ao_atom: np.ndarray, n_atoms: int):
    """Mulliken gross population of orbital `c` on every atom (sums to 1 for a normalized c)."""
    return np.bincount(ao_atom, weights=c * (S @ c), minlength=n_atoms)


def ao_shares(c: np.ndarray, S: np.ndarray, ao_l: np.ndarray) -> tuple[float, float, float]:
    """Fraction of the orbital's population on s, p and d AOs."""
    gross = c * (S @ c)
    total = float(gross.sum())
    if total <= 0:
        return 0.0, 0.0, 0.0
    s, p, d = (float(gross[ao_l == ell].sum()) / total for ell in (0, 1, 2))
    return s, p, d


def p_vector(c: np.ndarray, ao_atom: np.ndarray, ao_l: np.ndarray, atom: int) -> np.ndarray | None:
    """Cartesian (x, y, z) p coefficients of `c` on `atom`; None if the atom has no p shell."""
    idx = np.flatnonzero((ao_atom == atom) & (ao_l == 1))
    if len(idx) != 3:
        return None
    vec = np.zeros(3)
    vec[np.array(P_AXIS_ORDER)] = c[idx]
    return vec


def axis_cosines(
    c: np.ndarray,
    ao_atom: np.ndarray,
    ao_l: np.ndarray,
    coords: np.ndarray,
    a: int,
    b: int,
    config: ClassifyConfig,
) -> tuple[float, ...]:
    """|cos| between each atom's p vector and the a->b axis, for atoms with p character."""
    axis = coords[b] - coords[a]
    axis = axis / np.linalg.norm(axis)
    out = []
    for atom in (a, b):
        vec = p_vector(c, ao_atom, ao_l, atom)
        if vec is None:
            continue
        norm = float(np.linalg.norm(vec))
        if norm < config.min_p_norm:
            continue
        out.append(abs(float(vec @ axis)) / norm)
    return tuple(out)


def overlap_population(c: np.ndarray, S: np.ndarray, ao_atom: np.ndarray, a: int, b: int):
    """Mulliken overlap population between atoms a and b: 2 sum_{mu in a, nu in b} c S c."""
    ia, ib = ao_atom == a, ao_atom == b
    return float(2.0 * c[ia] @ S[np.ix_(ia, ib)] @ c[ib])


def _one_atom_type(shares: tuple[float, float, float], occupied: bool) -> OrbitalType:
    if occupied:
        return OrbitalType.LONE_PAIR
    s, p, d = shares
    if p >= s and p >= d:
        return OrbitalType.EMPTY_P
    return OrbitalType.EMPTY_D if d >= s else OrbitalType.EMPTY_S


def bond_type(
    cosines: tuple[float, ...], overlap_pop: float, occupied: bool, config: ClassifyConfig
) -> OrbitalType:
    """sigma / pi (occupied) or sigma_star / pi_star (virtual) from the bond-axis test."""
    if (overlap_pop > 0) != occupied:  # bonding must be occupied, antibonding virtual
        return OrbitalType.UNKNOWN
    sigma_like = [x > config.sigma_cos for x in cosines]
    if not cosines or any(sigma_like):  # s-s bond, X-H, or at least one atom along the axis
        return OrbitalType.SIGMA if occupied else OrbitalType.SIGMA_STAR
    if len(cosines) == 2:
        return OrbitalType.PI if occupied else OrbitalType.PI_STAR
    return OrbitalType.UNKNOWN  # one atom with p character, pointing off the axis


def _delocalized_type(s_share: float, occupied: bool, config: ClassifyConfig) -> OrbitalType:
    if s_share >= config.max_s_share_pi:
        return OrbitalType.UNKNOWN
    return OrbitalType.DELOCALIZED_PI if occupied else OrbitalType.DELOCALIZED_PI_STAR


def metal_partner(
    a: int, b: int, atomic_numbers: np.ndarray | None, config: ClassifyConfig
) -> tuple[int, int] | None:
    """(non-metal atom, metal atom) when exactly one of a and b is an ionic metal, else None."""
    if atomic_numbers is None:
        return None
    metal_a = int(atomic_numbers[a]) in config.ionic_metals
    metal_b = int(atomic_numbers[b]) in config.ionic_metals
    if metal_a == metal_b:
        return None
    return (b, a) if metal_a else (a, b)


def classify_orbital(
    c: np.ndarray,
    S: np.ndarray,
    ao_atom: np.ndarray,
    ao_l: np.ndarray,
    coords: np.ndarray,
    occupied: bool,
    config: ClassifyConfig | None = None,
    atomic_numbers: np.ndarray | None = None,
) -> Classification:
    """Classify one S-normalized AO coefficient vector (see the module docstring).

    Without `atomic_numbers` the ionic-metal rule is skipped.
    """
    config = config or ClassifyConfig()
    n_atoms = len(coords)
    pops = atom_populations(c, S, ao_atom, n_atoms)
    shares = ao_shares(c, S, ao_l)
    order = [int(i) for i in np.argsort(-pops, kind="stable")]
    a = order[0]
    b = order[1] if n_atoms > 1 else None
    nan = float("nan")
    if b is None or pops[a] >= config.one_atom_frac:
        kind = _one_atom_type(shares, occupied)
        metal = None
        if occupied and b is not None and pops[b] >= config.metal_min_pop:
            metal = metal_partner(a, b, atomic_numbers, config)
        bound = metal is not None and metal[0] == a
        flags = (True, metal[1]) if bound else (False, -1)
        return Classification(
            kind, occupied, (a,), (float(pops[a]),), pops, *shares, nan, (), *flags
        )
    if pops[a] + pops[b] >= config.two_atom_frac:
        op = overlap_population(c, S, ao_atom, a, b)
        metal = metal_partner(a, b, atomic_numbers, config) if occupied else None
        if metal is not None:
            site, metal_atom = metal
            weights = (float(pops[site]),)
            lp = OrbitalType.LONE_PAIR
            return Classification(
                lp, occupied, (site,), weights, pops, *shares, op, (), True, metal_atom
            )
        cos = axis_cosines(c, ao_atom, ao_l, coords, a, b, config)
        kind = bond_type(cos, op, occupied, config)
        weights = (float(pops[a]), float(pops[b]))
        return Classification(kind, occupied, (a, b), weights, pops, *shares, op, cos)
    members = tuple(i for i in order if pops[i] >= config.member_min_pop) or (a,)
    weights = tuple(float(pops[i]) for i in members)
    kind = _delocalized_type(shares[0], occupied, config)
    return Classification(kind, occupied, members, weights, pops, *shares, nan, ())


def _mean_cos(cls: Classification) -> float:
    return float(np.mean(cls.axis_cosines)) if cls.axis_cosines else 1.0


def rank_bond_orbitals(classes: list[Classification]) -> list[Classification]:
    """Pairs carrying 2 or 3 bonding-sign-correct LMOs: largest mean |cos| is sigma, rest pi."""
    groups: dict[tuple[int, int], list[int]] = {}
    for k, cls in enumerate(classes):
        if cls.bonding_sign_ok and not cls.metal_bound:
            pair = (min(cls.parent_atoms), max(cls.parent_atoms))
            groups.setdefault(pair, []).append(k)
    out = list(classes)
    for members in groups.values():
        if len(members) not in (2, 3):
            continue
        ranked = sorted(members, key=lambda k: -_mean_cos(classes[k]))
        for rank, k in enumerate(ranked):
            occ = classes[k].occupied
            if rank == 0:
                kind = OrbitalType.SIGMA if occ else OrbitalType.SIGMA_STAR
            else:
                kind = OrbitalType.PI if occ else OrbitalType.PI_STAR
            out[k] = replace(classes[k], type=kind)
    return out


def classify_block(
    C_loc: np.ndarray,
    S: np.ndarray,
    ao_atom: np.ndarray,
    ao_l: np.ndarray,
    coords: np.ndarray,
    occupied: bool,
    config: ClassifyConfig | None = None,
    atomic_numbers: np.ndarray | None = None,
) -> list[Classification]:
    """Classify every column of a localized block, then rank multi-LMO bonds."""
    config = config or ClassifyConfig()
    classes = [
        classify_orbital(C_loc[:, k], S, ao_atom, ao_l, coords, occupied, config, atomic_numbers)
        for k in range(C_loc.shape[1])
    ]
    return rank_bond_orbitals(classes) if config.rank_bond_orbitals else classes
