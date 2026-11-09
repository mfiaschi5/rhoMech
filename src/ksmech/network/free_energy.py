"""Free energies of pool states for the network's Markov chain (reaction-network plan, items E
and F). Units: kcal/mol, entropies in kcal/(mol K), temperatures in kelvin.

The chain reads thermodynamics through two callables, so the per-molecule thermochemistry of
item B plugs in without the network knowing how it is computed:

- `FreeEnergy`: (state, T) -> the state's free energy at T, as a float (G), a (G, H, S) triple
  or a `Thermo`; None when the state has none (counted NO_FREE_ENERGY). A pool's G holds every
  species it carries, reservoir copies included, each at its chemical potential;
- `ChemicalPotential`: (species key, T) -> mu of a reservoir species at T (solvent, reagent in
  excess), the same forms; it prices the copies a step takes from the reservoir or gives back
  (`network.reservoir`).

`ThermoModel` bundles them. `load_thermo_model` is the adapter point for item B: its
G(pool) = sum_i G_i(T) + RT ln(c_i / c0) (`qm.thermo.pool_free_energy`, in Hartree, with the
reservoir's fixed potentials) is wired there as 'qm' (`network.thermo_provider`); with 'none'
the chain falls back to the model's own probabilities.

`inversion(dh, ds)`: the inversion temperature T* = dH / dS and on which side of it the change
is favoured, from dG(T) = dH - T dS linearized at the T where H and S were evaluated:
an association (dH < 0, dS < 0) is favoured below T* and loses with heat, a release of a small
molecule (dH > 0, dS > 0) is favoured above T*.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

R_KCAL = 8.314462618 / 4184.0  # kcal / (mol K)
EXP_CLIP = 700.0  # |exponent| bound of exp(), so K(T) and the weights stay finite
NO_FREE_ENERGY = "NO_FREE_ENERGY"


@dataclass(frozen=True)
class Thermo:
    """G, and H and S when known, of a state or a change (kcal/mol, kcal/(mol K))."""

    g: float
    h: float | None = None
    s: float | None = None


def as_thermo(value: Any) -> Thermo | None:
    """A `FreeEnergy`'s answer as a `Thermo`: a float is G, a triple (G, H, S); None, or a
    non-finite G, is no free energy."""
    if value is None:
        return None
    if isinstance(value, Thermo):
        out = value
    elif isinstance(value, int | float):
        out = Thermo(float(value))
    elif len(value) == 3:
        g, h, s = value
        out = Thermo(float(g), None if h is None else float(h), None if s is None else float(s))
    else:
        raise TypeError(f"a free energy is G, (G, H, S) or a Thermo, got {value!r}")
    return out if math.isfinite(out.g) else None


def combine(base: Thermo, terms: Sequence[tuple[Thermo, float]]) -> Thermo:
    """base + sum of coefficient x term, each of G, H and S apart (None when a part misses)."""
    items = list(terms)
    g = base.g + sum(c * t.g for t, c in items)
    parts = []
    for name in ("h", "s"):
        values = [getattr(base, name)] + [getattr(t, name) for t, _ in items]
        coefs = [1.0] + [c for _, c in items]
        ok = all(v is not None for v in values)
        parts.append(sum(c * v for c, v in zip(coefs, values, strict=True)) if ok else None)
    return Thermo(g, *parts)


def difference(after: Thermo, before: Thermo) -> Thermo:
    """after - before."""
    return combine(after, [(before, -1.0)])


class FreeEnergy(Protocol):
    def __call__(self, state: Any, temperature_k: float) -> Any: ...


class ChemicalPotential(Protocol):
    def __call__(self, species_key: str, temperature_k: float) -> Any: ...


@dataclass(frozen=True)
class ThermoModel:
    free_energy: FreeEnergy
    chemical_potential: ChemicalPotential | None = None


def load_thermo_model(kind: str = "none", **options) -> ThermoModel | None:
    """ADAPTER POINT (reaction-network plan, item B): the free energies a network run reads.
    'none' gives None (every state NO_FREE_ENERGY: the chain runs on the model's own
    probabilities and no sink can be declared). 'qm' gives item B's provider
    (`network.thermo_provider.qm_thermo_model`, `options` its arguments): G(pool) from the
    thermochemistry sidecars in kcal/mol, the reservoir species at their concentration."""
    if kind == "none":
        return None
    if kind == "qm":  # item B: the thermochemistry sidecars (`network.thermo_provider`)
        from ksmech.network.thermo_provider import qm_thermo_model

        return qm_thermo_model(**options)
    raise ValueError(f"unknown free-energy provider {kind!r}: 'none' or 'qm'")


def rt(temperature_k: float) -> float:
    """RT in kcal/mol."""
    return R_KCAL * float(temperature_k)


def clipped_exp(x: float) -> float:
    return math.exp(max(-EXP_CLIP, min(EXP_CLIP, x)))


def equilibrium_constant(dg: float, temperature_k: float) -> float:
    """K(T) = exp(-dG / RT), the exponent clipped at +-700."""
    return clipped_exp(-dg / rt(temperature_k))


def climb_threshold(temperature_k: float, epsilon: float, margin_kcal: float) -> float:
    """delta(T) = RT ln(1 / epsilon) + margin: a step climbing more holds less than epsilon of
    its pair's equilibrium population, with the margin for the free energies' error."""
    return rt(temperature_k) * math.log(1.0 / epsilon) + margin_kcal


@dataclass(frozen=True)
class Inversion:
    t_star: float | None  # K; None when no positive T* exists
    favoured: str  # "above" | "below" T*, or "always" | "never" | "neutral"


def inversion(dh: float | None, ds: float | None) -> Inversion | None:
    """T* = dH / dS and on which side of it dG < 0 (module docstring); None without dH, dS."""
    if dh is None or ds is None:
        return None
    if abs(ds) < 1e-12:
        return Inversion(None, "always" if dh < 0 else "never" if dh > 0 else "neutral")
    t_star = dh / ds
    if t_star <= 0:  # dH and dS of opposite signs (or dH = 0): one sign at every T
        return Inversion(None, "always" if ds > 0 else "never")
    return Inversion(t_star, "above" if ds > 0 else "below")
