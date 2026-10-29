"""Pair-level helpers: the candidate filter, the stored atom-atom Coulomb matrix and the
standardization of the pair features."""

import numpy as np
import pytest

from ksmech.orbitals.tokens import CandidateConfig, build_molecule_tokens
from ksmech.physics.coulomb import coulomb, vdw_sum
from ksmech.physics.pairs import (
    PAIR_FLAGS,
    PAIR_VALID,
    PairStats,
    atom_coulomb_matrix,
    cell_columns,
    fit_pair_stats,
    pair_flags,
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


@pytest.fixture(scope="module")
def fixed_tokens(sp_record, smiles):
    return {name: build_molecule_tokens(sp_record(smi)) for name, smi in smiles.items()}


def _ids(tokens: list) -> set[int]:
    return {id(o) for o in tokens}


def test_pair_valid_is_donor_acceptor_and_no_exclusion_flag(fixed_tokens):
    """On every fixed molecule the legacy mask is a donor x an empty acceptor (d functions
    allowed) without any of the three stored exclusion flags (open-shell plan P0a, item 1)."""
    with_d = CandidateConfig(allow_d_acceptors=True)
    for name, t in fixed_tokens.items():
        flags = pair_flags([t])
        assert tuple(flags) == PAIR_FLAGS
        donors, acceptors = _ids(t.donor_candidates()), _ids(t.acceptor_candidates(with_d))
        ok_d = np.array([id(o) in donors for o in t.occupied()])
        ok_a = np.array([id(o) in acceptors for o in t.virtual()])
        excluded = flags["d_acceptor"] | flags["own_antibond"] | flags["site_self"]
        expected = ok_d[:, None] & ok_a[None, :] & ~excluded
        assert np.array_equal(valid_pairs([t]), expected), name


def test_d_acceptor_marks_the_chlorine_d_functions(fixed_tokens):
    t = fixed_tokens["chloromethane"]
    d_cols = [k for k, o in enumerate(t.virtual()) if t.is_d_type(o)]
    flag = pair_flags([t])["d_acceptor"]
    assert d_cols and flag[:, d_cols].all()
    assert not np.delete(flag, d_cols, axis=1).any()
    assert not pair_flags([fixed_tokens["ethane"]])["d_acceptor"].any()


@pytest.mark.parametrize("name", ["acetone", "acrolein"])
def test_own_antibond_marks_the_carbonyl_pi_to_pi_star_cell(fixed_tokens, name):
    t = fixed_tokens[name]
    own = pair_flags([t])["own_antibond"]
    occ, virt = t.occupied(), t.virtual()

    def carbonyl_pi(o) -> bool:
        return {t.symbol(a) for a in leading(o)} == {
            "C",
            "O",
        } and o.descriptors.cos_character() < 0.5

    cells = [
        (r, s)
        for r, s in zip(*np.nonzero(own), strict=True)
        if carbonyl_pi(occ[r]) and carbonyl_pi(virt[s])
    ]
    assert cells, name
    assert all(occ[r].descriptors.overlap_population > 0 for r, _ in cells)


def test_exclusions_stay_within_a_fragment(pair):
    methanol, chloromethane, _ = pair
    flags = pair_flags([methanol, chloromethane])
    n_occ, n_virt = len(methanol.occupied()), len(methanol.virtual())
    for name in ("own_antibond", "site_self"):
        assert not flags[name][:n_occ, n_virt:].any() and not flags[name][n_occ:, :n_virt].any()
    assert flags["own_antibond"][:n_occ, :n_virt].any()


def test_cell_columns_select_by_name_and_mask_the_legacy_cells():
    stored = ("g", "inter", "d_acceptor")
    z = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    defined = np.ones((2, 3, 3), dtype=bool)
    valid = np.array([[True, False, True], [False, False, True]])
    zv, dv = cell_columns(z, defined, valid, stored, ("inter", "g"))
    assert np.array_equal(zv, z[..., [1, 0]]) and np.array_equal(dv, np.stack([valid] * 2, -1))
    za, da = cell_columns(z, defined, valid, stored, (*stored, PAIR_VALID), "all")
    assert da.all() and np.array_equal(za[..., 3], valid.astype(float))
    with pytest.raises(ValueError, match="not stored"):
        cell_columns(z, defined, valid, stored, ("wbi",))
    with pytest.raises(ValueError, match="pair cells"):
        cell_columns(z, defined, valid, stored, ("g",), "some")
