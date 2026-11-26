"""The donor-acceptor energy E2 of the phase-3 pretraining target (`physics.donor_acceptor`), on
the fixture molecules (gas-phase single points at the force-field geometry)."""

from dataclasses import replace

import numpy as np
import pytest

from ksmech.physics.donor_acceptor import (
    ClosedShellOnly,
    DegenerateBasis,
    donor_acceptor,
    lowdin,
)
from ksmech.qm.embed import prepare_mol
from tests.fixtures.molecules import MOLECULES
from tests.token_select import highest_lone_pair, leading, lone_pairs, sigma_stars

KCAL_PER_EV = 23.0605


@pytest.fixture(scope="module")
def tokens(token_loader):
    return {name: token_loader(smi) for name, smi in MOLECULES.items()}


def _cell(t, da, donor, acceptor) -> float:
    return float(da.e2[t.occupied().index(donor), t.virtual().index(acceptor)])


def _top(t, da):
    i, j = np.unravel_index(np.argmax(da.e2), da.e2.shape)
    return t.occupied()[i], t.virtual()[j]


@pytest.mark.parametrize("name", list(MOLECULES))
def test_layout_signs_and_conditioning(tokens, name):
    t = tokens[name]
    da = donor_acceptor(t)
    shape = (len(t.occupied()), len(t.virtual()))
    for a in (da.e2, da.coupling, da.denominator, da.defined):
        assert a.shape == shape
    assert np.isfinite(da.e2).all() and (da.e2 >= 0).all()
    assert da.defined.all(), "a closed-shell fixture has every LMO denominator above 0.1 eV"
    assert 0.3 < da.min_overlap_eig <= 1.0 + 1e-9


@pytest.mark.parametrize("name", ["acetone", "benzene", "methyl_acetate"])
def test_exact_lmos_have_no_occupied_virtual_coupling(tokens, name):
    """Unrestricted, the basis is the exact LMOs: their occupied-virtual Fock block vanishes
    (Brillouin), so any coupling the target reports comes from the restriction alone."""
    da = donor_acceptor(tokens[name], restrict=False)
    assert da.coupling.max() < 1e-5
    assert donor_acceptor(tokens[name]).coupling.max() > 1.0


def test_phase_invariance(tokens):
    t = tokens["acrolein"]
    flipped = [
        replace(o, coeff_vector=-o.coeff_vector) if k % 3 == 0 else o
        for k, o in enumerate(t.orbitals)
    ]
    np.testing.assert_allclose(
        donor_acceptor(replace(t, orbitals=flipped)).e2, donor_acceptor(t).e2, atol=1e-10
    )


def test_ester_resonance_is_the_largest_term(tokens):
    """Methyl acetate: the ether oxygen's p lone pair into the carbonyl pi* (the ester
    resonance, ~45-50 kcal/mol in NBO) is the strongest donor-acceptor pair."""
    t = tokens["methyl_acetate"]
    da = donor_acceptor(t)
    donor, acceptor = _top(t, da)
    ether = [a for a in range(len(t.atoms)) if t.symbol(a) == "O" and _heavy_degree(t, a) == 2]
    assert len(ether) == 1 and leading(donor, 1) == (ether[0],)
    assert {t.symbol(a) for a in leading(acceptor)} == {"C", "O"}
    assert acceptor.descriptors.cos_character() < 0.5, "pi*, not sigma*: lobes off the axis"
    assert 30 < da.e2.max() * KCAL_PER_EV < 80


def test_carbonyl_lone_pair_into_vicinal_sigma_star(tokens):
    """Acetone: the oxygen's p lone pair gives the two C-C sigma* the largest, equal terms."""
    t = tokens["acetone"]
    da = donor_acceptor(t)
    n_o = highest_lone_pair(t, "O")
    cc = sigma_stars(t, {"C"})
    assert len(cc) == 2
    values = [_cell(t, da, n_o, s) for s in cc]
    assert values[0] == pytest.approx(values[1], rel=0.05)
    assert min(values) == pytest.approx(da.e2.max(), rel=0.05)


def test_ethane_antiperiplanar_wins(tokens, sp_record):
    """Ethane: each C-H bond gives most to the sigma*(C-H) antiperiplanar to it."""
    t = tokens["ethane"]
    xyz = sp_record("CC").coords_ang
    da = donor_acceptor(t)
    ch_star = sigma_stars(t, {"C", "H"})
    for donor in t.occupied():
        if {t.symbol(a) for a in leading(donor)} != {"C", "H"}:
            continue
        c, h = sorted(leading(donor), key=lambda a: t.symbol(a) == "H")
        others = [s for s in ch_star if c not in leading(s)]
        best = max(others, key=lambda s: _cell(t, da, donor, s))
        c2, h2 = sorted(leading(best), key=lambda a: t.symbol(a) == "H")
        assert _dihedral(xyz[h], xyz[c], xyz[c2], xyz[h2]) > 150


def test_lone_pairs_of_a_closed_shell_are_donors(tokens):
    t = tokens["methylamine"]
    da = donor_acceptor(t)
    n_n = lone_pairs(t, "N")[0]
    assert da.e2[t.occupied().index(n_n)].max() * KCAL_PER_EV > 4


@pytest.mark.parametrize("smiles", ["[CH3]", "C=C[CH2]"])
def test_open_shell_refused(token_loader, smiles):
    with pytest.raises(ClosedShellOnly):
        donor_acceptor(token_loader(smiles))


def test_degenerate_set_refused():
    T = np.array([[1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(DegenerateBasis):
        lowdin(T, np.eye(2))


def _heavy_degree(t, atom: int) -> int:
    """Heavy neighbours of a record atom, on the record's own molecule (`prepare_mol`)."""
    a = prepare_mol(t.smiles).GetAtomWithIdx(atom)
    return sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors())


def _dihedral(a, b, c, d) -> float:
    b0, b1, b2 = b - a, c - b, d - c
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    cos = n1 @ n2 / (np.linalg.norm(n1) * np.linalg.norm(n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
