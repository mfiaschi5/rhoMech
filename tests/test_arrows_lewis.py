"""The reactant's Lewis structure on the token atoms and the check of an edit label against it
(`ksmech.arrows.lewis`)."""

import numpy as np
import pytest
from rdkit import Chem

from ksmech.arrows.delta_be import EditLabel, ReactionMatrix
from ksmech.arrows.lewis import check_edit, graph_distances, reactant_lewis
from ksmech.arrows.reactants import ReactantTokens
from tests.fixtures.molecules import MOLECULES


def lewis(smiles: str, loader):
    rt = ReactantTokens.from_smiles(smiles, loader)
    return (rt, *reactant_lewis(rt))


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_every_token_atom_has_its_valence_electrons(name, token_loader):
    """On every token atom, implicit H included: 2 x lone pairs + bond orders + formal charge =
    valence electrons; every H has one single bond; every Wiberg index is positive."""
    rt, bonds, lone, wiberg = lewis(MOLECULES[name], token_loader)
    order = np.zeros(rt.n_atoms, dtype=int)
    for i, j, o in bonds:
        order[i] += o
        order[j] += o
    table = Chem.GetPeriodicTable()
    for f, frag in enumerate(rt.fragments):
        for a in frag.record.GetAtoms():
            g = int(rt.atom_offsets[f]) + a.GetIdx()
            valence = table.GetNOuterElecs(a.GetAtomicNum())
            assert 2 * lone[g] + order[g] + a.GetFormalCharge() == valence, (name, g)
            if a.GetAtomicNum() == 1:
                assert lone[g] == 0 and order[g] == 1
    assert len(bonds) == sum(frag.record.GetNumBonds() for frag in rt.fragments)
    assert (bonds[:, 0] < bonds[:, 1]).all() and np.isfinite(wiberg).all() and (wiberg > 0).all()


def test_benzene_keeps_the_drawn_kekule_structure(token_loader):
    """The bond orders are the drawing's (alternating 1 and 2); the Wiberg indices are the
    wavefunction's, the same on all six ring bonds (about 1.4)."""
    rt, bonds, _, wiberg = lewis("C1=CC=CC=C1", token_loader)
    carbons = set(rt.global_atom.values())
    ring = [k for k, (i, j, _) in enumerate(bonds) if int(i) in carbons and int(j) in carbons]
    assert sorted(int(bonds[k, 2]) for k in ring) == [1, 1, 1, 2, 2, 2]
    cc = wiberg[ring].astype(float)
    assert cc.min() > 1.2 and cc.max() < 1.7 and cc.max() - cc.min() < 0.05


def test_graph_distances_within_and_between_molecules(token_loader):
    rt, bonds, *_ = lewis("CCO.C", token_loader)
    dist = graph_distances(bonds, rt.n_atoms)
    heavy = {rt.mol.GetAtomWithIdx(m).GetSymbol() + str(m): g for m, g in rt.global_atom.items()}
    c0, c1, o2, c3 = heavy["C0"], heavy["C1"], heavy["O2"], heavy["C3"]
    assert dist[c0, c1] == 1 and dist[c0, o2] == 2 and dist[o2, c0] == 2
    assert dist[c0, c3] == -1 and (np.diag(dist) == 0).all()


def test_check_edit_names_what_an_edit_breaks(token_loader):
    """CH3-Cl: C-Cl heterolysis is valid; a bond order changed by 3 (no elementary step), a bond
    broken where there is none, a lone pair taken from an atom without one, unbalanced
    electrons and a breaking anchor on an unbonded pair are not."""
    rt, bonds, lone, _ = lewis("CCl", token_loader)
    c = rt.global_atom[0]
    cl = rt.global_atom[1]
    h = next(int(j) for i, j, _ in bonds if int(i) == c and int(j) != cl)

    def label(bond_entries, lone_entries, anchors):
        r = ReactionMatrix(tuple(bond_entries), tuple(lone_entries))
        return EditLabel((r,), (tuple(anchors),), True, 1, False)

    pair = (min(c, cl), max(c, cl))
    assert check_edit(label([(*pair, -1)], [(cl, 2)], [(*pair, "break")]), bonds, lone) is None
    assert check_edit(label([(*pair, 3)], [(cl, -6)], [(*pair, "form")]), bonds, lone) == (
        "NOT_ELEMENTARY"
    )
    hcl = (min(h, cl), max(h, cl))
    assert check_edit(label([(*hcl, -1)], [(cl, 2)], [(*hcl, "break")]), bonds, lone) == (
        "BOND_ORDER"
    )
    assert check_edit(label([(*pair, 1)], [(c, -2)], [(*pair, "form")]), bonds, lone) == (
        "LONE_PAIRS"
    )
    assert check_edit(label([(*pair, -1)], [], [(*pair, "break")]), bonds, lone) == "CHARGE"
    assert check_edit(label([], [], [(*hcl, "break")]), bonds, lone) == "ANCHOR_MASK"
