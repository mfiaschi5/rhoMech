"""Thermochemistry of one molecule and the free energy of a pool: plain numpy, no I/O, no tblite
(reaction-network plan, item B; docs/decisions.md, 2026-11-09).

Per molecule (`MolecularThermo`, read from the sidecar `qm.thermo_cache` writes), at any T:

- translation (ideal gas at p0 = 1 atm, Sackur-Tetrode), rotation (rigid rotor; symmetry number
  from `qm.point_group`; none for an atom), vibration (harmonic, with Grimme's quasi-RRHO for the
  entropy: S = w S_HO + (1 - w) S_free-rotor, w = 1 / (1 + (nu0 / nu)^4), nu0 = 50 cm^-1, the
  free rotor's moment mu' = mu B_av / (mu + B_av) with B_av the mean principal moment of the
  molecule), and the electronic entropy R ln(2S + 1);
- the frequencies are those of the species' thermo minimum: the record's geometry, re-optimized
  along its imaginary modes below -10 cm^-1 when it has any (`qm.thermo_reopt`); they are used as
  xtb's thermo uses them: |nu| <= 1 cm^-1 is dropped, an imaginary mode above -20 cm^-1 is
  inverted (made real), one at or below -20 cm^-1 is dropped and counted (a mode the
  re-optimization could not remove, IMAGINARY_PERSISTENT);
- the ZPE and the vibrational enthalpy are harmonic over every kept mode (xtb interpolates only
  the entropy and the heat capacity);
- the electronic energy is the thermo minimum's (the record's when nothing was re-optimized):
  GFN2 with ALPB, whose solvation free energy is the
  'gsolv' state (1 L of gas to 1 L of solution, no shift). The standard state of a G returned
  here is 1 M in solution: G = E_el + ZPE + H_thermal - T S(gas, 1 atm) + RT ln(V_m / 1 L),
  with V_m = RT / p0 (24.465 L/mol at 298.15 K, so +1.894 kcal/mol).

Energies are in Hartree, entropies in Hartree / K, temperatures in K, concentrations in mol/L.

The pool (`pool_free_energy`): G(pool) = sum over species i of n_i mu_i, with
mu_i = G_i(T) + RT ln(c_i / c0) (c0 = 1 M, the standard state of G_i) and, for a reservoir
species (solvent, a reagent in excess), a fixed chemical potential instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

# CODATA 2018
H_PLANCK = 6.62607015e-34  # J s
K_B = 1.380649e-23  # J / K
N_A = 6.02214076e23  # 1 / mol
R_GAS = K_B * N_A  # J / (mol K)
C_CM = 2.99792458e10  # speed of light, cm / s
AMU_KG = 1.66053906660e-27
ANG_M = 1e-10
HARTREE_J = 4.3597447222071e-18
HARTREE_J_MOL = HARTREE_J * N_A
HARTREE_KCAL = 627.5094740631  # kcal / mol per Hartree
R_HARTREE = K_B / HARTREE_J  # Hartree / K: RT in Hartree is R_HARTREE * T
ATM_PA = 101325.0
LITRE_M3 = 1e-3
T_STANDARD = 298.15
STATES = ("1M", "1atm")  # the standard state of a returned G: 1 M solution, or the 1 atm gas


@dataclass(frozen=True)
class ThermoSettings:
    """How frequencies become thermochemistry (xtb's defaults where it has one)."""

    rotor_cutoff_cm: float = 50.0  # nu0 of the qRRHO weight (xtb `sthr`)
    rotor_exponent: int = 4
    imag_cutoff_cm: float = -20.0  # imaginary modes above it are inverted (xtb `imagthr`)
    zero_cutoff_cm: float = 1.0  # |nu| at or below it is dropped (xtb's `vibthr`)
    pressure_pa: float = ATM_PA  # the gas-phase standard pressure of S_trans
    standard_state: str = "1M"
    c0_mol_l: float = 1.0  # the solution standard state

    def __post_init__(self) -> None:
        if self.standard_state not in STATES:
            raise ValueError(f"standard_state must be one of {STATES}, got {self.standard_state!r}")


DEFAULTS = ThermoSettings()


@dataclass(frozen=True)
class MolecularThermo:
    """Everything G(T) needs, without the Hessian."""

    energy_ha: float  # electronic energy at the thermo minimum, ALPB included ('gsolv' state)
    frequencies_cm: np.ndarray  # (3N-6,) or (3N-5,) or (0,): signed, negative = imaginary
    moments_amu_ang2: np.ndarray  # (3,) principal moments, ascending
    mass_amu: float
    symmetry_number: int
    multiplicity: int  # 2S + 1 = unpaired + 1
    linear: bool
    n_atoms: int


@dataclass(frozen=True)
class ModeUse:
    """The frequencies the thermochemistry uses (cm^-1, all > 0) and what was done to the rest."""

    used_cm: np.ndarray
    n_inverted: int  # imaginary modes above the cutoff, made real
    n_dropped_imaginary: int  # imaginary modes at or below the cutoff, left out
    n_dropped_zero: int  # |nu| at or below the zero cutoff, left out


@dataclass(frozen=True)
class ThermoResult:
    """One molecule at one temperature. H and G are absolute (they include E_el)."""

    temperature: float
    energy_ha: float
    zpe: float
    h_thermal: float  # H - E_el - ZPE: translation, rotation, vibration (pV = RT included)
    entropy: float  # S of the ideal gas at p0, every part, Hartree / K
    entropy_parts: dict[str, float] = field(default_factory=dict)  # trans, rot, vib, el
    state_shift: float = 0.0  # RT ln(V_m / V0) into the 1 M state, 0 for '1atm'
    modes: ModeUse | None = None

    @property
    def enthalpy(self) -> float:
        return self.energy_ha + self.zpe + self.h_thermal

    @property
    def free_energy(self) -> float:
        return self.enthalpy - self.temperature * self.entropy + self.state_shift

    @property
    def g_correction(self) -> float:
        """G - E_el."""
        return self.free_energy - self.energy_ha


def usable_modes(frequencies_cm: np.ndarray, settings: ThermoSettings = DEFAULTS) -> ModeUse:
    """xtb's treatment of the frequencies before its thermo (`print_thermo`): drop |nu| at or
    below the zero cutoff, invert the imaginary modes above the imaginary cutoff, drop the
    others (counted)."""
    nu = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    zero = np.abs(nu) <= settings.zero_cutoff_cm
    nu = nu[~zero]
    soft = (nu < 0) & (nu > settings.imag_cutoff_cm)
    nu = np.where(soft, -nu, nu)
    return ModeUse(nu[nu > 0], int(soft.sum()), int((nu < 0).sum()), int(zero.sum()))


# ---- the parts, in J / mol and J / (mol K)


def translational(mass_amu: float, temperature: float, pressure_pa: float) -> tuple[float, float]:
    """(H, S) of the ideal-gas translation: H = 5/2 RT (pV included), Sackur-Tetrode at p."""
    m = mass_amu * AMU_KG
    kt = K_B * temperature
    q_v = (2.0 * math.pi * m * kt / H_PLANCK**2) ** 1.5 * kt / pressure_pa
    return 2.5 * R_GAS * temperature, R_GAS * (math.log(q_v) + 2.5)


def rotational_temperatures(moments_amu_ang2: np.ndarray) -> np.ndarray:
    """theta_i = h^2 / (8 pi^2 I_i k) in K, one per moment (inf for a zero moment)."""
    inertia = np.asarray(moments_amu_ang2, dtype=float) * AMU_KG * ANG_M**2
    with np.errstate(divide="ignore"):
        return H_PLANCK**2 / (8.0 * math.pi**2 * inertia * K_B)


def rotational(
    moments_amu_ang2: np.ndarray, symmetry_number: int, linear: bool, temperature: float
) -> tuple[float, float]:
    """(H, S) of the rigid rotor: nonlinear 3/2 RT, R (ln q + 3/2); linear RT, R (ln q + 1)
    with the mean of the two largest moments."""
    moments = np.sort(np.asarray(moments_amu_ang2, dtype=float))
    if linear:
        theta = rotational_temperatures(np.array([moments[1:].mean()]))[0]
        q = temperature / (symmetry_number * theta)
        return R_GAS * temperature, R_GAS * (math.log(q) + 1.0)
    theta = rotational_temperatures(moments)
    q = math.sqrt(math.pi) / symmetry_number * math.sqrt(np.prod(temperature / theta))
    return 1.5 * R_GAS * temperature, R_GAS * (math.log(q) + 1.5)


def harmonic_entropy(nu_cm: np.ndarray, temperature: float) -> np.ndarray:
    """S_HO / R of each mode."""
    x = H_PLANCK * C_CM * np.asarray(nu_cm, dtype=float) / (K_B * temperature)
    return x / np.expm1(x) - np.log(-np.expm1(-x))


def free_rotor_entropy(nu_cm: np.ndarray, temperature: float, b_av_kgm2: float) -> np.ndarray:
    """S_R / R of each mode as a free rotor (Grimme 2012): mu = h / (8 pi^2 nu), limited by the
    average moment, mu' = mu B_av / (mu + B_av); S_R = R (1/2 + ln sqrt(8 pi^3 mu' k T / h^2))."""
    mu = H_PLANCK / (8.0 * math.pi**2 * C_CM * np.asarray(nu_cm, dtype=float))
    mu_eff = mu * b_av_kgm2 / (mu + b_av_kgm2)
    return 0.5 + 0.5 * np.log(8.0 * math.pi**3 * mu_eff * K_B * temperature / H_PLANCK**2)


def qrrho_weight(nu_cm: np.ndarray, settings: ThermoSettings = DEFAULTS) -> np.ndarray:
    """The harmonic share w = 1 / (1 + (nu0 / nu)^alpha) of each mode's entropy."""
    ratio = settings.rotor_cutoff_cm / np.asarray(nu_cm, dtype=float)
    return 1.0 / (1.0 + ratio**settings.rotor_exponent)


def vibrational(
    nu_cm: np.ndarray, temperature: float, b_av_kgm2: float, settings: ThermoSettings
) -> tuple[float, float, float]:
    """(ZPE, H_thermal, S) of the vibrations: the ZPE and the thermal energy harmonic over every
    mode, the entropy the qRRHO interpolation."""
    nu = np.asarray(nu_cm, dtype=float)
    if not len(nu):
        return 0.0, 0.0, 0.0
    theta = H_PLANCK * C_CM * nu / K_B
    zpe = 0.5 * R_GAS * theta.sum()
    h_thermal = R_GAS * (theta / np.expm1(theta / temperature)).sum()
    w = qrrho_weight(nu, settings)
    s_ho = harmonic_entropy(nu, temperature)
    s_fr = free_rotor_entropy(nu, temperature, b_av_kgm2) if b_av_kgm2 > 0 else s_ho
    return zpe, h_thermal, R_GAS * float((w * s_ho + (1.0 - w) * s_fr).sum())


def electronic_entropy(multiplicity: int) -> float:
    """R ln(2S + 1), J / (mol K): the spin degeneracy only (no orbital degeneracy)."""
    return R_GAS * math.log(multiplicity)


def standard_state_shift(
    temperature: float, pressure_pa: float = ATM_PA, c0_mol_l: float = 1.0
) -> float:
    """RT ln(V_m / V0), Hartree: the ideal gas at p (V_m = RT / p) to the concentration c0.
    At 298.15 K, 1 atm and 1 M: RT ln(24.465) = 1.894 kcal/mol."""
    v_m = R_GAS * temperature / pressure_pa  # m^3 / mol
    v_0 = LITRE_M3 / c0_mol_l
    return R_HARTREE * temperature * math.log(v_m / v_0)


def average_moment_kgm2(moments_amu_ang2: np.ndarray) -> float:
    """B_av of the free rotor: the mean of the three principal moments (xtb's `avmom`)."""
    return float(np.mean(moments_amu_ang2)) * AMU_KG * ANG_M**2


# ---- one molecule


def thermo_at(
    data: MolecularThermo, temperature: float, settings: ThermoSettings = DEFAULTS
) -> ThermoResult:
    """The thermochemistry of one molecule at `temperature` (module docstring)."""
    t = float(temperature)
    if t <= 0:
        raise ValueError(f"temperature must be positive, got {t}")
    h_tr, s_tr = translational(data.mass_amu, t, settings.pressure_pa)
    h_rot, s_rot, modes = 0.0, 0.0, usable_modes(np.zeros(0), settings)
    zpe = h_vib = s_vib = 0.0
    if data.n_atoms > 1:
        h_rot, s_rot = rotational(data.moments_amu_ang2, data.symmetry_number, data.linear, t)
        modes = usable_modes(data.frequencies_cm, settings)
        b_av = average_moment_kgm2(data.moments_amu_ang2)
        zpe, h_vib, s_vib = vibrational(modes.used_cm, t, b_av, settings)
    s_el = electronic_entropy(data.multiplicity)
    parts = {"trans": s_tr, "rot": s_rot, "vib": s_vib, "el": s_el}
    shift = 0.0
    if settings.standard_state == "1M":
        shift = standard_state_shift(t, settings.pressure_pa, settings.c0_mol_l)
    return ThermoResult(
        temperature=t,
        energy_ha=float(data.energy_ha),
        zpe=zpe / HARTREE_J_MOL,
        h_thermal=(h_tr + h_rot + h_vib) / HARTREE_J_MOL,
        entropy=sum(parts.values()) / HARTREE_J_MOL,
        entropy_parts={k: v / HARTREE_J_MOL for k, v in parts.items()},
        state_shift=shift,
        modes=modes,
    )


def thermo_table(
    data: MolecularThermo,
    temperatures: Iterable[float],
    settings: ThermoSettings = DEFAULTS,
) -> list[ThermoResult]:
    return [thermo_at(data, t, settings) for t in temperatures]


def standard_free_energy(
    data: MolecularThermo, temperature: float, settings: ThermoSettings = DEFAULTS
) -> float:
    """G_i(T) in Hartree, in the standard state of `settings` (1 M solution by default)."""
    return thermo_at(data, temperature, settings).free_energy


# ---- the pool


def rt_hartree(temperature: float) -> float:
    return R_HARTREE * float(temperature)


def chemical_potential(
    g_standard: float, temperature: float, concentration: float, c0: float = 1.0
) -> float:
    """mu = G + RT ln(c / c0), Hartree; `g_standard` in the standard state c0 (mol/L). Also the
    chemical potential of a reservoir species held at `concentration` (water as solvent:
    55.3 M at 298 K)."""
    if concentration <= 0 or c0 <= 0:
        raise ValueError(f"concentrations must be positive, got c={concentration}, c0={c0}")
    return float(g_standard) + rt_hartree(temperature) * math.log(concentration / c0)


def chemical_potentials(
    species: Iterable[str],
    g_standard: Mapping[str, float],
    temperature: float,
    concentrations: Mapping[str, float] | None = None,
    reservoir: Mapping[str, float] | None = None,
    c0: float = 1.0,
) -> dict[str, float]:
    """mu of each species (Hartree): a reservoir species' fixed potential, else
    G + RT ln(c / c0) with c from `concentrations` (c0 when absent). KeyError names a species
    with neither a reservoir potential nor a G."""
    concentrations, reservoir = concentrations or {}, reservoir or {}
    out = {}
    for key in species:
        if key in reservoir:
            out[key] = float(reservoir[key])
        elif key in g_standard:
            c = concentrations.get(key, c0)
            out[key] = chemical_potential(g_standard[key], temperature, c, c0)
        else:
            raise KeyError(f"species {key!r} has no G and is not a reservoir species")
    return out


def pool_free_energy(
    pool: Mapping[str, int],
    g_standard: Mapping[str, float],
    temperature: float,
    concentrations: Mapping[str, float] | None = None,
    reservoir: Mapping[str, float] | None = None,
    c0: float = 1.0,
) -> float:
    """G(pool) = sum_i n_i mu_i in Hartree.

    pool: species key -> count (the multiset of a pool state; any hashable naming, e.g. the
        identity key of `data.resonance`). A negative count is a species owed to a reservoir:
        two pools compare when they hold the same atoms once the reservoir species exchanged
        with the reservoir are counted (pool A + H2O(reservoir) -> pool B: give A the count
        H2O: +1, or B the count H2O: -1).
    g_standard: species key -> G_i(T) in the standard state c0 (`standard_free_energy`, 1 M).
    temperature: K, the one G_i was evaluated at.
    concentrations: species key -> mol/L; c0 when absent (RT ln(c / c0) = 0).
    reservoir: species key -> fixed chemical potential mu (Hartree), used for every copy of
        that species instead of G + RT ln(c / c0): solvent, reagents in excess
        (`chemical_potential(G, T, c)` gives one at a concentration).
    c0: the standard concentration of `g_standard`, mol/L.
    """
    mu = chemical_potentials(pool, g_standard, temperature, concentrations, reservoir, c0)
    return float(sum(int(n) * mu[key] for key, n in pool.items()))


def reaction_free_energy(
    before: Mapping[str, int],
    after: Mapping[str, int],
    g_standard: Mapping[str, float],
    temperature: float,
    concentrations: Mapping[str, float] | None = None,
    reservoir: Mapping[str, float] | None = None,
    c0: float = 1.0,
) -> float:
    """G(after) - G(before), Hartree (`pool_free_energy` for the arguments)."""
    args = (g_standard, temperature, concentrations, reservoir, c0)
    return pool_free_energy(after, *args) - pool_free_energy(before, *args)


def equilibrium_constant(delta_g_ha: float, temperature: float) -> float:
    """K = exp(-dG / RT) for dG in Hartree."""
    return math.exp(-float(delta_g_ha) / rt_hartree(temperature))
