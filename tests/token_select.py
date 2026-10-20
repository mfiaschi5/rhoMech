"""Pick orbital tokens by their continuous descriptors, never by the classifier's label.

Tests that check physics (energies, couplings, the gap kernel) select their tokens here, so they
do not depend on `ksmech.orbitals.classify`. A token is one-centre when its main centre carries
at least `ONE_CENTRE` of the population, two-centre when its two leading centres together do.
Antibonding means a negative overlap population between the two leading centres.
"""

from __future__ import annotations

ONE_CENTRE = 0.85
TWO_CENTRE = 0.85
AXIAL = 0.7  # cos-character above this: p vectors along the axis (sigma-like)
PERPENDICULAR = 0.3  # below this: p vectors across the axis (pi-like)


def leading(o, k: int = 2) -> tuple[int, ...]:
    """The k leading centre atoms of a token, by descending population."""
    return tuple(int(a) for a in o.descriptors.center_atoms[:k] if a >= 0)


def overlap(o) -> float:
    """Signed overlap population between the two leading centres (0 for one centre)."""
    return float(o.descriptors.overlap_population)


def one_centre(o) -> bool:
    return o.descriptors.center_weights[0] >= ONE_CENTRE


def two_centre(o) -> bool:
    w = o.descriptors.center_weights
    return not one_centre(o) and len(leading(o)) == 2 and w[0] + w[1] >= TWO_CENTRE


def lone_pairs(t, symbol: str | None = None) -> list:
    """One-centre occupied tokens, optionally on atoms of one element."""
    return [
        o
        for o in t.occupied()
        if one_centre(o) and (symbol is None or t.symbol(leading(o, 1)[0]) == symbol)
    ]


def highest_lone_pair(t, symbol: str):
    return max(lone_pairs(t, symbol), key=lambda o: o.energy_ev)


def bond_orbitals(t, block: list, symbols: set[str] | None = None) -> list:
    """Two-centre tokens of `block` whose two leading centres are of these elements."""
    return [
        o
        for o in block
        if two_centre(o) and (symbols is None or {t.symbol(a) for a in leading(o)} == set(symbols))
    ]


def sigma_stars(t, symbols: set[str] | None = None) -> list:
    """Antibonding, axis-aligned two-centre virtual tokens."""
    return [
        o
        for o in bond_orbitals(t, t.virtual(), symbols)
        if overlap(o) < 0 and o.descriptors.cos_character() > AXIAL
    ]


def sigma_star(t, symbols: set[str]):
    found = sigma_stars(t, symbols)
    assert len(found) == 1, (t.smiles, symbols, [leading(o) for o in found])
    return found[0]
