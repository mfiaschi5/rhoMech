"""Pair-level helpers: the candidate filter, the stored atom-atom Coulomb matrix and the
standardization of the pair features."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import build_molecule_tokens
from ksmech.physics.coulomb import coulomb, vdw_sum
from ksmech.physics.pairs import (
    PairStats,
    atom_coulomb_matrix,
    fit_pair_stats,
    standardize,
    valid_pairs,
)
from ksmech.physics.params import COULOMB_EV_ANG, REDUCED, KSParams
from ksmech.physics.stage_a import STAGE_A_FEATURES, stage_a_block
from tests.token_select import highest_lone_pair, leading, sigma_star


@pytest.fixture(scope="module")
def pair(sp_record):
    methanol = build_molecule_tokens(sp_record("CO"))
    chloromethane = build_molecule_tokens(sp_record("CCl"))
    return methanol, chloromethane, stage_a_block([methanol, chloromethane])


def test_coulomb_term():
    assert coulomb(0.5, -0.5, 2.0, KSParams(eps=4.0)) == COULOMB_EV_ANG * -0.25 / 8.0
    assert coulomb(1.0, 1.0, 0.0) == 0.0


def test_standardization(pair):
    *_, block = pair
    stats = fit_pair_stats([(block.z, block.defined)], STAGE_A_FEATURES)
    zhat = standardize(block.z, block.defined, stats)
    for k in range(len(STAGE_A_FEATURES)):
        v = zhat[..., k][block.defined[..., k]]
        assert abs(v.mean()) < 1e-9 and (stats.std[k] == 0 or abs(v.std() - 1.0) < 1e-9)
    assert (zhat[~block.defined] == 0).all()
    back = PairStats.from_json(stats.to_json(), STAGE_A_FEATURES)
    assert np.allclose(back.mean, stats.mean) and np.allclose(back.std, stats.std)
    with pytest.raises(ValueError):
        PairStats.from_json(stats.to_json(), STAGE_A_FEATURES[::-1])


def _at(seq: list, x) -> int:
    """Position of `x` in `seq` by identity (tokens of two molecules may compare unequal-ly)."""
    return next(k for k, y in enumerate(seq) if y is x)


def test_valid_pairs_filter(pair):
    methanol, chloromethane, _ = pair
    frags = [methanol, chloromethane]
    valid = valid_pairs(frags)
    occ = [o for t in frags for o in t.occupied()]
    virt = [o for t in frags for o in t.virtual()]
    assert valid.shape == (len(occ), len(virt))
    d_type = [_at(virt, o) for o in chloromethane.virtual() if chloromethane.is_d_type(o)]
    assert d_type and not valid[:, d_type].any()  # chlorine d functions are never acceptors
    star = sigma_star(chloromethane, {"C", "Cl"})
    bond = next(o for o in chloromethane.occupied() if set(leading(o)) == set(leading(star)))
    assert bond.descriptors.overlap_population > 0 > star.descriptors.overlap_population
    assert not valid[_at(occ, bond), _at(virt, star)]  # the donor's own antibond
    lone = highest_lone_pair(methanol, "O")
    assert valid[_at(occ, lone), _at(virt, star)]
    assert valid.sum() == sum(len(t.donor_candidates()) for t in frags) * sum(
        len(t.acceptor_candidates()) for t in frags
    ) - sum(
        len(t.acceptor_candidates()) - len(t.acceptor_candidates(donor=d))
        for t in frags
        for d in t.donor_candidates()
    )


def test_atom_coulomb_matrix(pair):
    methanol, chloromethane, _ = pair
    M = atom_coulomb_matrix([methanol, chloromethane])
    atoms = methanol.atoms + chloromethane.atoms
    assert M.shape == (len(atoms),) * 2 and np.allclose(M, M.T) and not np.diag(M).any()
    a, b = 0, 1  # two atoms of methanol: the geometric distance
    r = np.linalg.norm(atoms[a].position - atoms[b].position)
    q = atoms[a].mulliken_charge * atoms[b].mulliken_charge
    assert M[a, b] == pytest.approx(COULOMB_EV_ANG * q / (4.0 * r))
    c = len(methanol.atoms)  # first atom of chloromethane: the van der Waals contact
    q = atoms[a].mulliken_charge * atoms[c].mulliken_charge
    r = vdw_sum(atoms[a].atomic_number, atoms[c].atomic_number)
    assert M[a, c] == pytest.approx(COULOMB_EV_ANG * q / (4.0 * r))
    assert np.allclose(atom_coulomb_matrix([methanol, chloromethane], REDUCED), 4.0 * M)
