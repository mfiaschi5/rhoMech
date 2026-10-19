"""Orbital classification rules on small single-point molecules and on synthetic inputs."""

from collections import Counter

import numpy as np

from ksmech.orbitals.classify import (
    Classification,
    ClassifyConfig,
    OrbitalType,
    ao_shares,
    atom_populations,
    bond_type,
    classify_block,
    rank_bond_orbitals,
)
from ksmech.orbitals.localize import localize_record

T = OrbitalType


def classify(rec, occupied: bool, config=None):
    loc = localize_record(rec)
    C = loc.C_occ if occupied else loc.C_virt
    return classify_block(
        C, rec.S, rec.ao_atom, rec.ao_l, rec.coords_ang, occupied, config, rec.atomic_numbers
    )


def types(classes) -> Counter:
    return Counter(c.type for c in classes)


def test_bond_type_rules():
    cfg = ClassifyConfig()
    assert bond_type((0.95, 0.90), +0.5, True, cfg) == T.SIGMA
    assert bond_type((0.95, 0.90), -0.5, False, cfg) == T.SIGMA_STAR
    assert bond_type((0.10, 0.05), +0.3, True, cfg) == T.PI
    assert bond_type((0.10, 0.05), -0.3, False, cfg) == T.PI_STAR
    assert bond_type((), +0.5, True, cfg) == T.SIGMA  # H-H: no p on either atom
    assert bond_type((0.95,), +0.5, True, cfg) == T.SIGMA  # C-H: p on one atom only
    assert bond_type((0.95, 0.10), +0.5, True, cfg) == T.SIGMA  # one atom along the axis
    assert bond_type((0.10,), +0.3, True, cfg) == T.UNKNOWN  # pi needs p on both atoms
    assert bond_type((0.95, 0.90), -0.5, True, cfg) == T.UNKNOWN  # occupied but antibonding
    assert bond_type((0.95, 0.90), +0.5, False, cfg) == T.UNKNOWN  # virtual but bonding


def test_water(sp_record):
    rec = sp_record("O")
    occ = classify(rec, True)
    assert types(occ) == {T.LONE_PAIR: 2, T.SIGMA: 2}
    oxygen = int(np.flatnonzero(rec.atomic_numbers == 8)[0])
    assert all(c.parent_atoms == (oxygen,) for c in occ if c.type == T.LONE_PAIR)
    assert all(
        oxygen in c.parent_atoms and len(c.parent_atoms) == 2 for c in occ if c.type == T.SIGMA
    )
    assert types(classify(rec, False)) == {T.SIGMA_STAR: 2}


def test_ethylene_pi_and_pi_star(sp_record):
    rec = sp_record("C=C")
    occ = classify(rec, True)
    assert types(occ) == {T.SIGMA: 5, T.PI: 1}
    pi = next(c for c in occ if c.type == T.PI)
    assert sorted(pi.parent_atoms) == [0, 1] and pi.s_share < 0.05 and pi.overlap_population > 0
    virt = classify(rec, False)
    assert types(virt) == {T.SIGMA_STAR: 5, T.PI_STAR: 1}
    pi_star = next(c for c in virt if c.type == T.PI_STAR)
    assert sorted(pi_star.parent_atoms) == [0, 1] and pi_star.overlap_population < 0


def test_methyl_cation_has_an_empty_p(sp_record):
    rec = sp_record("[CH3+]")
    virt = classify(rec, False)
    assert types(virt) == {T.EMPTY_P: 1, T.SIGMA_STAR: 3}
    empty = next(c for c in virt if c.type == T.EMPTY_P)
    assert empty.parent_atoms == (0,) and empty.p_share > 0.9


def test_proton_and_chloride(sp_record):
    proton = sp_record("[H+]")
    assert classify(proton, True) == []
    assert types(classify(proton, False)) == {T.EMPTY_S: 1}
    chloride = sp_record("[Cl-]")
    assert types(classify(chloride, True)) == {T.LONE_PAIR: 4}
    assert types(classify(chloride, False)) == {T.EMPTY_D: 5}


def test_parents_are_ordered_by_population(sp_record):
    rec = sp_record("CO")
    for c in classify(rec, True) + classify(rec, False):
        assert list(c.parent_weights) == sorted(c.parent_weights, reverse=True)
        assert np.isclose(c.populations.sum(), 1.0, atol=1e-8)
        assert abs(c.s_share + c.p_share + c.d_share - 1.0) < 1e-8


def test_thresholds_come_from_the_config(sp_record):
    rec = sp_record("O")
    strict = ClassifyConfig(one_atom_frac=1.5, two_atom_frac=1.5)  # nothing can pass
    occ = classify(rec, True, strict)
    assert T.LONE_PAIR not in types(occ) and T.SIGMA not in types(occ)
    assert set(types(occ)) <= {T.DELOCALIZED_PI, T.UNKNOWN}


def test_population_helpers(sp_record):
    rec = sp_record("O")
    c = rec.C[:, 0]
    pops = atom_populations(c, rec.S, rec.ao_atom, rec.n_atoms)
    assert pops.shape == (3,) and np.isclose(pops.sum(), 1.0)
    s, p, d = ao_shares(c, rec.S, rec.ao_l)
    assert np.isclose(s + p + d, 1.0) and d == 0.0
    assert ao_shares(np.zeros_like(c), rec.S, rec.ao_l) == (0.0, 0.0, 0.0)


def test_methylmagnesium_bromide_carbanion_lone_pair(sp_record):
    rec = sp_record("C[Mg]Br")
    carbon = int(np.flatnonzero(rec.atomic_numbers == 6)[0])
    magnesium = int(np.flatnonzero(rec.atomic_numbers == 12)[0])
    bromine = int(np.flatnonzero(rec.atomic_numbers == 35)[0])
    occ = classify(rec, True)
    on_c = [c for c in occ if c.type == T.LONE_PAIR and c.parent_atoms == (carbon,)]
    assert len(on_c) == 1 and on_c[0].metal_bound and on_c[0].metal_atom == magnesium
    assert on_c[0].overlap_population > 0 and 0.4 < on_c[0].parent_weights[0] < 0.8
    assert T.UNKNOWN not in types(occ) and T.SIGMA not in {
        c.type for c in occ if magnesium in c.parent_atoms
    }
    # the Br-Mg bond is a metal-bound lone pair on Br too; the free Br lone pairs are not bound
    br = [c for c in occ if c.type == T.LONE_PAIR and c.parent_atoms == (bromine,)]
    assert sum(c.metal_bound for c in br) == 1 and len(br) == 4
    # the rule needs atomic numbers: without them the C-Mg orbital is an ordinary bond test
    plain = classify_block(
        localize_record(rec).C_occ, rec.S, rec.ao_atom, rec.ao_l, rec.coords_ang, True
    )
    assert not any(c.metal_bound for c in plain)


def test_metal_rule_is_configurable(sp_record):
    rec = sp_record("C[Li]")
    no_metals = ClassifyConfig(ionic_metals=frozenset())
    assert not any(c.metal_bound for c in classify(rec, True, no_metals))
    bound = [c for c in classify(rec, True) if c.metal_bound]
    assert len(bound) == 1 and rec.atomic_numbers[bound[0].parent_atoms[0]] == 6


def test_propyne_triple_bond_is_one_sigma_two_pi(sp_record):
    from rdkit import Chem

    rec = sp_record("CC#C")
    mol = Chem.MolFromSmiles(rec.atom_order_smiles)
    triple = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)
    sp_carbons = {triple.GetBeginAtomIdx(), triple.GetEndAtomIdx()}
    occ = [c for c in classify(rec, True) if set(c.parent_atoms) == sp_carbons]
    assert types(occ) == {T.SIGMA: 1, T.PI: 2}
    virt = [c for c in classify(rec, False) if set(c.parent_atoms) == sp_carbons]
    assert types(virt) == {T.SIGMA_STAR: 1, T.PI_STAR: 2}


def _two_centre(kind, occupied, cos, op=0.3):
    pops = np.array([0.5, 0.5, 0.0])
    return Classification(kind, occupied, (0, 1), (0.5, 0.5), pops, 0.3, 0.7, 0.0, op, cos)


def test_rank_bond_orbitals_overrides_the_cosine_threshold():
    tilted = [  # an alkyne PM tilted towards banana bonds: sigma at the 0.7 threshold
        _two_centre(T.UNKNOWN, True, (0.68, 0.70)),
        _two_centre(T.PI, True, (0.23, 0.23)),
        _two_centre(T.PI, True, (0.15, 0.16)),
        _two_centre(T.SIGMA, True, (1.0,), op=0.4),  # a C-H bond on another pair: untouched
    ]
    tilted[3] = Classification(
        T.SIGMA, True, (1, 2), (0.5, 0.5), tilted[3].populations, 0.6, 0.4, 0.0, 0.4, (1.0,)
    )
    ranked = rank_bond_orbitals(tilted)
    assert [c.type for c in ranked] == [T.SIGMA, T.PI, T.PI, T.SIGMA]
    # a double bond with two banana orbitals: one sigma, one pi
    bananas = [
        _two_centre(T.PI, False, (0.6, 0.6), op=-0.3),
        _two_centre(T.PI, False, (0.5, 0.55), op=-0.3),
    ]
    assert [c.type for c in rank_bond_orbitals(bananas)] == [T.SIGMA_STAR, T.PI_STAR]
    # wrong bonding sign or a single LMO: left alone
    wrong = [_two_centre(T.UNKNOWN, True, (0.9, 0.9), op=-0.2), _two_centre(T.PI, True, (0.1, 0.1))]
    assert [c.type for c in rank_bond_orbitals(wrong)] == [T.UNKNOWN, T.PI]
    assert ClassifyConfig(rank_bond_orbitals=False).rank_bond_orbitals is False
