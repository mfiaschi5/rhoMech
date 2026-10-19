import pytest

from ksmech.data.stats import NO_CLASS, compute_stats, distribution


def test_stats_on_mini_fixture(mini_steps):
    stats = compute_stats(list(mini_steps.values()))
    assert stats["n_steps"] == 5
    assert stats["n_pathways"] == 0 and stats["n_steps_without_pathway"] == 5
    assert stats["steps_per_class"] == {NO_CLASS: 5}
    assert stats["arrows_per_step"] == {"2": 4, "3": 1}
    # resonance is the only intramolecular step
    assert stats["intermolecular_fraction"] == pytest.approx(4 / 5)
    # OH-, CH3Cl, AcOH, CN-, MeCHO, EtO-, iPrBr, enolate
    assert stats["reactant_molecules"]["unique"] == 8
    assert stats["reactant_molecules"]["total_fragments"] == 9
    assert stats["charged"]["unique_reactants_charged"] == pytest.approx(4 / 8)
    assert stats["charged"]["steps_with_charged_reactant"] == pytest.approx(1.0)
    assert set(stats["elements"]) == {"Br", "C", "Cl", "H", "N", "O"}
    assert stats["elements"]["Cl"] == 1
    assert stats["heavy_atoms_per_step"]["max"] == 7  # E2: EtO- (3) + iPrBr (4)


def test_distribution():
    d = distribution([1, 2, 3, 12])
    assert d["n"] == 4 and d["min"] == 1 and d["max"] == 12
    assert d["hist"] == {"0-9": 3, "10-19": 1}
    assert distribution([]) == {"n": 0}
