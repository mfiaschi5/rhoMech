"""The arrow reading of a reaction matrix (`serve.reading`): every move type, the directed
pooled-gap choice between decompositions, and the PMechDB code of a reading."""

from __future__ import annotations

import pytest

from ksmech.arrows.delta_be import ReactionMatrix
from ksmech.serve.reading import (
    BOND_TO_BOND,
    BOND_TO_LONE,
    LONE_TO_BOND,
    Move,
    decompose,
    pmech_code,
)


def R(bonds, lone=()) -> ReactionMatrix:
    return ReactionMatrix(tuple(sorted(bonds)), tuple(sorted(lone)))


def kinds(moves) -> list[tuple]:
    return sorted((m.kind, m.source, m.target) for m in moves)


def test_sn2_lone_pair_attack_and_leaving_group():
    # Nu(0) lone pair -> C(1); C(1)-X(2) breaks onto X
    moves = decompose(R([(0, 1, 1), (1, 2, -1)], [(0, -2), (2, 2)]))
    assert kinds(moves) == [
        (BOND_TO_LONE, (1, 2), (2,)),
        (LONE_TO_BOND, (0,), (0, 1)),
    ]


def test_e2_relay_reads_three_arrows():
    # base(0) takes H(1) from C_beta(2); C_beta=C_alpha(3) forms; C_alpha-X(4) leaves
    r = R([(0, 1, 1), (1, 2, -1), (2, 3, 1), (3, 4, -1)], [(0, -2), (4, 2)])
    assert kinds(decompose(r)) == [
        (BOND_TO_BOND, (2, 1), (2, 3)),
        (BOND_TO_LONE, (3, 4), (4,)),
        (LONE_TO_BOND, (0,), (0, 1)),
    ]


def test_carbonyl_addition_pi_bond_to_oxygen():
    # Nu(0) -> C(1) of C(1)=O(2): the pi pair goes to O
    moves = decompose(R([(0, 1, 1), (1, 2, -1)], [(0, -2), (2, 2)]))
    assert Move(BOND_TO_LONE, (1, 2), (2,)) in moves
    assert Move(LONE_TO_BOND, (0,), (0, 1)) in moves


def test_proton_transfer():
    # B(5) lone pair -> H(3); H(3)-A(7) breaks onto A
    moves = decompose(R([(3, 5, 1), (3, 7, -1)], [(5, -2), (7, 2)]))
    assert kinds(moves) == [
        (BOND_TO_LONE, (3, 7), (7,)),
        (LONE_TO_BOND, (5,), (5, 3)),
    ]


def test_heterolysis_is_one_bond_to_lone_pair_move():
    assert decompose(R([(0, 1, -1)], [(1, 2)])) == [Move(BOND_TO_LONE, (0, 1), (1,))]
    assert decompose(R([(0, 1, -1)], [(0, 2)])) == [Move(BOND_TO_LONE, (1, 0), (0,))]


def test_pi_bond_formation_on_a_bonded_pair():
    # an amine N(0) closes an iminium on C+(1): the N-C bond gains its pi order
    assert decompose(R([(0, 1, 1)], [(0, -2)])) == [Move(LONE_TO_BOND, (0,), (0, 1))]


def test_two_decompositions_follow_the_directed_gap():
    # metathesis-like: 0-2 and 1-3 break, 0-1 and 2-3 form: two readings
    r = R([(0, 1, 1), (0, 2, -1), (1, 3, -1), (2, 3, 1)])
    first = decompose(r)
    assert kinds(first) == [(BOND_TO_BOND, (0, 2), (0, 1)), (BOND_TO_BOND, (3, 1), (3, 2))]
    favour = {(1, 0): 5.0, (2, 3): 5.0}
    other = decompose(r, lambda a, b: favour.get((a, b), 0.1))
    assert kinds(other) == [(BOND_TO_BOND, (1, 3), (1, 0)), (BOND_TO_BOND, (2, 0), (2, 3))]


@pytest.mark.parametrize(
    "r",
    [
        R([(0, 1, 1)], [(2, -2)]),  # the pair comes from an atom outside the bond
        R([(0, 1, 1)], [(0, -1), (1, -1)]),  # odd lone-electron changes
        R([(0, 1, -1)], [(2, 2)]),  # the pair lands on an atom outside the bond
    ],
)
def test_matrices_without_a_reading(r):
    assert decompose(r) is None


def test_pmech_code_of_each_move_kind():
    maps = {0: 7, 1: 3, 2: 9, 3: 4}.__getitem__
    moves = [
        Move(LONE_TO_BOND, (0,), (0, 1)),
        Move(BOND_TO_BOND, (2, 3), (2, 1)),
        Move(BOND_TO_LONE, (1, 3), (3,)),
    ]
    assert pmech_code(moves, maps) == "7=3;9,4=9,3;3,4=4"


def test_pmech_code_parses_back_and_reproduces_the_edit():
    """The code of an SN2 reading, parsed with the PMechDB reader and applied as one edit,
    gives the product of R (sigma key)."""
    from ksmech.arrows.apply import apply_arrows, apply_edit, sanitize_product
    from ksmech.data import chem
    from ksmech.data.pmechdb import parse_arrow_code
    from ksmech.data.resonance import sigma_key

    mol = chem.parse_mapped("[CH3:1][O-:2].[CH3:3][Cl:4]")
    o, c, cl = (chem.map_to_index(mol)[m] for m in (2, 3, 4))
    r = R([(*sorted((o, c)), 1), (*sorted((c, cl)), -1)], [(o, -2), (cl, 2)])
    code = pmech_code(decompose(r), lambda a: mol.GetAtomWithIdx(a).GetAtomMapNum())
    assert code == "2=3;3,4=4"
    via_code = apply_arrows(mol, parse_arrow_code(code))
    direct = sanitize_product(apply_edit(mol, r.to_edit()))
    assert via_code.ok and via_code.key == sigma_key(direct)
