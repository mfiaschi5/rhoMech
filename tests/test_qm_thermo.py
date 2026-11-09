"""`qm.thermo`: ideal-gas thermochemistry with Grimme's qRRHO, the 1 M standard state, and the
free energy of a pool. Pure numpy: reference values from textbook molecules, no QM."""

import math

import numpy as np
import pytest

from ksmech.qm import thermo as th
from ksmech.qm.hessian import atomic_masses, principal_moments

J_PER_HARTREE_MOL = th.HARTREE_J_MOL
CAL = 4.184


def _j_mol_k(s_hartree_per_k: float) -> float:
    return s_hartree_per_k * J_PER_HARTREE_MOL


def _water(multiplicity: int = 1) -> th.MolecularThermo:
    """Water at its experimental geometry (0.9572 A, 104.52 deg) with its experimental
    fundamentals."""
    half = math.radians(104.52 / 2)
    xyz = np.array(
        [[0.0, 0.0, 0.0], [0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half)],
         [-0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half)]]
    )  # fmt: skip
    masses = atomic_masses(np.array([8, 1, 1]))
    return th.MolecularThermo(
        energy_ha=-76.0,
        frequencies_cm=np.array([1594.6, 3657.1, 3755.9]),
        moments_amu_ang2=principal_moments(xyz, masses),
        mass_amu=float(masses.sum()),
        symmetry_number=2,
        multiplicity=multiplicity,
        linear=False,
        n_atoms=3,
    )


def test_the_atomic_unit_frequency_factor():
    from ksmech.qm.hessian import AU_TO_CM

    bohr_m = 0.529177210903e-10
    omega = math.sqrt(th.HARTREE_J / (bohr_m**2 * th.AMU_KG))
    assert pytest.approx(omega / (2 * math.pi * th.C_CM), rel=1e-8) == AU_TO_CM


def test_modes_are_used_as_xtb_uses_them():
    use = th.usable_modes(np.array([-150.0, -20.0, -19.0, -0.5, 0.8, 30.0, 1200.0]))
    assert sorted(use.used_cm) == [19.0, 30.0, 1200.0]  # -19 inverted, -20 and -150 dropped
    assert (use.n_inverted, use.n_dropped_imaginary, use.n_dropped_zero) == (1, 2, 2)


def test_sackur_tetrode_for_argon():
    """S(Ar, 298.15 K, 1 atm) = 154.85 J/(mol K) at 1 bar - R ln(1.01325)."""
    h, s = th.translational(39.948, 298.15, th.ATM_PA)
    assert s == pytest.approx(154.846 - th.R_GAS * math.log(1.01325), abs=0.02)
    assert h == pytest.approx(2.5 * th.R_GAS * 298.15)


def test_rigid_rotor_of_n2_and_the_symmetry_number():
    """N2 (r = 1.0977 A): theta_rot = 2.874 K, S_rot(298.15 K) = 41.1 J/(mol K)."""
    moment = 14.0067 / 2 * 1.0977**2
    theta = th.rotational_temperatures(np.array([moment]))[0]
    assert theta == pytest.approx(2.874, abs=0.002)
    moments = np.array([0.0, moment, moment])
    h, s2 = th.rotational(moments, 2, True, 298.15)
    _, s1 = th.rotational(moments, 1, True, 298.15)
    assert h == pytest.approx(th.R_GAS * 298.15)
    assert s2 == pytest.approx(41.14, abs=0.03)
    assert s1 - s2 == pytest.approx(th.R_GAS * math.log(2))


def test_water_against_its_standard_entropy_and_enthalpy():
    """S(H2O, g, 298.15 K, 1 bar) = 188.83 J/(mol K); H(298.15) - H(0) = 9.904 kJ/mol."""
    r = th.thermo_at(_water(), 298.15)
    assert _j_mol_k(r.entropy) == pytest.approx(188.83 - th.R_GAS * math.log(1.01325), abs=0.3)
    assert r.h_thermal * J_PER_HARTREE_MOL / 1e3 == pytest.approx(9.904, abs=0.03)
    zpe = 0.5 * (1594.6 + 3657.1 + 3755.9)  # cm^-1
    assert r.zpe * 219474.6313632 == pytest.approx(zpe, rel=1e-6)
    assert r.modes.n_dropped_imaginary == 0 and len(r.modes.used_cm) == 3


def test_qrrho_is_rrho_for_stiff_modes_and_a_free_rotor_for_soft_ones():
    b_av = 3e-45
    stiff, soft = np.array([2000.0]), np.array([2.0])
    w_stiff, w_soft = th.qrrho_weight(stiff)[0], th.qrrho_weight(soft)[0]
    assert w_stiff > 1 - 1e-6 and w_soft < 1e-5
    assert th.qrrho_weight(np.array([50.0]))[0] == pytest.approx(0.5)
    _, _, s = th.vibrational(stiff, 298.15, b_av, th.DEFAULTS)
    assert s == pytest.approx(th.R_GAS * th.harmonic_entropy(stiff, 298.15)[0], abs=1e-5)
    _, _, s = th.vibrational(soft, 298.15, b_av, th.DEFAULTS)
    assert s == pytest.approx(th.R_GAS * th.free_rotor_entropy(soft, 298.15, b_av)[0], rel=1e-4)
    # the harmonic entropy diverges as nu -> 0, the interpolated one stays bounded
    nus = np.array([0.01, 1.0, 10.0])
    s_ho = th.harmonic_entropy(nus, 298.15)
    s_q = [th.vibrational(np.array([n]), 298.15, b_av, th.DEFAULTS)[2] / th.R_GAS for n in nus]
    assert s_ho[0] > 10.0 and max(s_q) < 6.0


def test_free_rotor_limit_matches_xtb_atomic_units():
    """xtb: mu = 1 / (2 omega) in atomic units, S_R / R = 1/2 + ln sqrt(2 pi mu' / beta)."""
    nu, temp, b_av = 30.0, 298.15, 2e-45
    omega = nu / 219474.6313632  # Hartree
    me_a0 = 9.1093837015e-31 * 0.529177210903e-10**2  # kg m^2 per atomic unit
    mu = 0.5 / omega
    b = b_av / me_a0
    mu_eff = mu * b / (mu + b)
    beta = 1.0 / (th.R_HARTREE * temp)
    xtb = 0.5 + math.log(math.sqrt(2 * math.pi * mu_eff / beta))
    assert th.free_rotor_entropy(np.array([nu]), temp, b_av)[0] == pytest.approx(xtb, rel=1e-6)


def test_the_1m_standard_state_shift():
    """RT ln(24.465) = 1.894 kcal/mol at 298.15 K; the molar volume grows with T."""
    shift = th.standard_state_shift(298.15)
    assert shift * th.HARTREE_KCAL == pytest.approx(1.8943, abs=2e-4)
    hot = th.standard_state_shift(373.15)
    rt = th.R_HARTREE * 373.15
    assert hot == pytest.approx(rt * math.log(24.4654 * 373.15 / 298.15), rel=1e-4)
    r1m = th.thermo_at(_water(), 298.15)
    gas = th.thermo_at(_water(), 298.15, th.ThermoSettings(standard_state="1atm"))
    assert r1m.free_energy - gas.free_energy == pytest.approx(shift)
    assert r1m.enthalpy == pytest.approx(gas.enthalpy) and r1m.entropy == gas.entropy


def test_triplet_electronic_entropy():
    singlet, triplet = th.thermo_at(_water(1), 298.15), th.thermo_at(_water(3), 298.15)
    assert _j_mol_k(triplet.entropy_parts["el"]) == pytest.approx(th.R_GAS * math.log(3))
    assert _j_mol_k(triplet.entropy_parts["el"]) == pytest.approx(9.134, abs=1e-3)
    assert singlet.entropy_parts["el"] == 0.0
    dg = triplet.free_energy - singlet.free_energy
    assert dg == pytest.approx(-298.15 * th.R_HARTREE * math.log(3))


def test_an_atom_has_translation_and_spin_only():
    atom = th.MolecularThermo(-13.0, np.zeros(0), np.zeros(3), 79.904, 1, 2, False, 1)
    r = th.thermo_at(atom, 298.15)
    assert r.zpe == 0.0 and r.entropy_parts["rot"] == 0.0 and r.entropy_parts["vib"] == 0.0
    assert r.h_thermal == pytest.approx(2.5 * th.R_HARTREE * 298.15)


def test_a_table_is_thermo_at_each_temperature():
    temps = [273.15, 298.15, 373.15]
    rows = th.thermo_table(_water(), temps)
    assert [r.temperature for r in rows] == temps
    assert rows[1].free_energy == th.standard_free_energy(_water(), 298.15)
    assert rows[0].free_energy > rows[1].free_energy > rows[2].free_energy
    with pytest.raises(ValueError):
        th.thermo_at(_water(), 0.0)


def test_pool_free_energy():
    temp, rt = 298.15, th.R_HARTREE * 298.15
    g = {"A": -1.0, "B": -2.0, "W": -76.0}
    assert th.pool_free_energy({"A": 2, "B": 1}, g, temp) == pytest.approx(-4.0)
    dilute = th.pool_free_energy({"A": 1}, g, temp, concentrations={"A": 0.01})
    assert dilute == pytest.approx(-1.0 + rt * math.log(0.01))
    mu_w = th.chemical_potential(g["W"], temp, 55.3)
    assert mu_w == pytest.approx(-76.0 + rt * math.log(55.3))
    # A + W(reservoir) -> B: the water drawn from the reservoir counts at its potential
    before, after = {"A": 1, "W": 1}, {"B": 1}
    dg = th.reaction_free_energy(before, after, g, temp, reservoir={"W": mu_w})
    assert dg == pytest.approx(-2.0 - (-1.0 + mu_w))
    owed = th.pool_free_energy({"B": 1, "W": -1}, g, temp, reservoir={"W": mu_w})
    assert owed - th.pool_free_energy({"A": 1}, g, temp) == pytest.approx(dg)
    assert th.equilibrium_constant(-rt * math.log(10.0), temp) == pytest.approx(10.0)
    with pytest.raises(KeyError):
        th.pool_free_energy({"X": 1}, g, temp)
    with pytest.raises(ValueError):
        th.chemical_potential(-1.0, temp, 0.0)


def test_settings_reject_an_unknown_standard_state():
    with pytest.raises(ValueError):
        th.ThermoSettings(standard_state="1bar")
