"""`network.lift` on the fixture molecules (reaction-network plan, item J): the canonical local
atom numbering of a species, pools built from species drawings, a pool step split into a
species step, a hyperpath compared with a pool-graph pathway, and a species graph built from a
real pool with the tiny model of the pool tests."""

import random
from collections import Counter

import pytest
import torch
from rdkit import Chem

from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key
from ksmech.model.config import ModelConfig
from ksmech.model.decode import DecodeConfig
from ksmech.network.flux import atom_balance, masses
from ksmech.network.hyperpaths import (
    hypergraph_of,
    k_shortest_hyperpaths,
    linearize,
    shortest_hyperpath,
    species_class,
    unfold,
)
from ksmech.network.lift import (
    NO_STEP_INFO,
    ModelPoolReactor,
    compare,
    local_ranks,
    mapped_fragments,
    pool_of,
    species_of,
    split_proposal,
    start_numbering,
    start_of,
    target_slots,
)
from ksmech.network.pools import ModelExpander, apply_on_mol, pool_state, signature
from ksmech.network.search import Edge, Proposal
from ksmech.network.species import ExpansionParams, SpeciesGraph, expand_species
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL

ALL = [*MOLECULES.values(), *(s for s, _ in OPEN_SHELL.values())]
MECL_OH = "[CH3:1][Cl:2].[OH-:3]"


def _labelled(mol):
    """(element by local index, bonds on local indices): the species' labelled graph."""
    ranks = local_ranks(mol)
    elements = [0] * len(ranks)
    for atom in mol.GetAtoms():
        elements[ranks[atom.GetIdx()]] = atom.GetAtomicNum()
    bonds = sorted(tuple(sorted((ranks[b.GetBeginAtomIdx()], ranks[b.GetEndAtomIdx()])))
                   for b in mol.GetBonds())  # fmt: skip
    return tuple(elements), tuple(bonds)


@pytest.mark.parametrize("smiles", ALL)
def test_local_numbering_ignores_the_atom_order(smiles):
    mol = parse_mapped(pool_state(smiles).smiles)
    rng = random.Random(0)
    want = _labelled(mol)
    for _ in range(10):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        assert _labelled(Chem.RenumberAtoms(mol, order)) == want
    assert sorted(local_ranks(mol)) == list(range(mol.GetNumAtoms()))


@pytest.mark.parametrize(
    "forms", [("C=C[O-]", "[CH2-]C=O"), ("C1=CC=CC=C1", "C=1C=CC=CC=1"), ("C=C[CH2]", "[CH2]C=C")]
)
def test_resonance_forms_number_their_atoms_alike(forms):
    a, b = (parse_mapped(pool_state(s).smiles) for s in forms)
    assert identity_key(a) == identity_key(b) and _labelled(a) == _labelled(b)


@pytest.mark.parametrize("smiles", ALL)
def test_a_species_drawing_carries_its_local_numbers(smiles):
    state = pool_state(smiles)
    for maps, frag in mapped_fragments(parse_mapped(state.smiles)):
        sp, ranks = species_of(frag)
        drawing = parse_mapped(sp.smiles)
        assert sorted(a.GetAtomMapNum() for a in drawing.GetAtoms()) == list(
            range(1, sp.n_atoms + 1)
        )
        assert identity_key(drawing) == sp.key and sp.n_atoms == len(maps) == len(ranks)
        assert all(sp.elements[a.GetAtomMapNum() - 1] == a.GetAtomicNum()
                   for a in drawing.GetAtoms())  # fmt: skip
        again = species_of(drawing)[0]  # its own drawing, up to the graph's automorphisms
        assert (again.key, again.n_atoms, again.skeleton, again.elements) == (
            sp.key, sp.n_atoms, sp.skeleton, sp.elements)  # fmt: skip
        assert _labelled(drawing) == _labelled(frag)
    records, amounts = start_of(state)
    assert pool_of([records[k] for k in amounts.elements()]).key == state.key


def test_a_pool_of_species_numbers_each_molecule_apart():
    records, _ = start_of(pool_state("CO"))
    (sp,) = records.values()
    pool = pool_of([sp, sp])
    assert Counter(pool.species) == {sp.key: 2}
    numbering = start_numbering(pool)
    assert len(numbering[sp.key]) == 2
    assert sorted(m for t in numbering[sp.key] for m in t) == list(range(1, 2 * sp.n_atoms + 1))


def _sn2(spectator: str = ""):
    """The SN2 step of chloromethane and hydroxide on a pool state, as a proposal."""
    state = pool_state(MECL_OH + spectator)
    mol = parse_mapped(state.smiles)
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    c, cl, o = idx[1], idx[2], idx[3]
    child = apply_on_mol(mol, [(c, o, 1), (c, cl, -1)], [(o, -2), (cl, 2)])
    maps = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    text, touched = signature([(c, o, 1), (c, cl, -1)], [(o, -2), (cl, 2)], maps)
    info = {"bonds": ((1, 2, -1), (1, 3, 1)), "lone": ((2, 2), (3, -2))}
    return state, Proposal(child, -0.4, text, touched, info)


def test_a_pool_step_splits_into_a_species_step():
    state, prop = _sn2(".[CH4:9]")  # methane looks on
    step, records, why = split_proposal(state, prop)
    keys = lambda *s: tuple(sorted(identity_key(Chem.MolFromSmiles(x)) for x in s))  # noqa: E731
    assert why is None and step.reactants == keys("CCl", "[OH-]")
    assert step.products == keys("CO", "[Cl-]") and set(records) == set(step.products)
    sizes = [5, 2]  # CH3Cl, OH- by their sorted keys
    got = sorted(x for slot in step.atom_map for x in slot)
    assert got == sorted((i, a) for i, n in enumerate(sizes) for a in range(n))  # a bijection
    assert len(step.bonds) == 2 and len(step.lone) == 2 and step.logp == -0.4
    assert split_proposal(state, Proposal(prop.child, -0.4))[2] == NO_STEP_INFO


def test_a_hyperpath_matches_the_pool_pathway_it_linearizes_to():
    state, prop = _sn2()
    step, records, _ = split_proposal(state, prop)
    start, amounts = start_of(state)
    g = SpeciesGraph({**start, **records}, dict(amounts))
    g.add_step(step)
    hp = shortest_hyperpath(hypergraph_of(g, target_slots("CO.[Cl-]")))
    edge = Edge(state.key, prop.child.key, prop)
    lin = linearize(hp, numbering=start_numbering(state))
    assert [e.proposal.signature for e in lin] == [prop.signature]
    assert lin[0].source == state.key and lin[0].target == prop.child.key
    res = compare(hp, [edge], state)
    assert res.same_species_class and res.same_mechanism_class and res.failure is None


TINY = ModelConfig(
    d=32, d_pair=16, layers=1, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)


@pytest.fixture(scope="module")
def expander(token_loader):
    from ksmech.arrows.lewis import reactant_lewis
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.arrows.steps import pair_stats, reactant_sample, with_lewis
    from ksmech.model.inputs import fit_feature_stats
    from ksmech.model.net import KSMech

    rt = ReactantTokens.from_smiles(MECL_OH, token_loader)
    sample = with_lewis(reactant_sample("t", "t", rt), reactant_lewis(rt))
    torch.manual_seed(0)
    model = KSMech(TINY, fit_feature_stats([sample]), pair_stats([sample])).eval()
    cfg = DecodeConfig(anchors=10, per_anchor=3, max_applied=16)
    return ModelExpander(model, token_loader, cfg)


@pytest.fixture(scope="module")
def tiny_graph(expander):
    start = pool_state(MECL_OH)
    records, amounts = start_of(start)
    params = ExpansionParams(tolerance=0.5, max_species=2, max_pools=10)
    g = expand_species(ModelPoolReactor(expander), records, amounts, params=params)
    return start, records, amounts, g


def test_a_species_graph_from_a_real_pool_with_the_tiny_model(expander, tiny_graph):
    start, records, amounts, g = tiny_graph
    assert g.core == sorted(amounts) and g.pools == 2 + 3 and g.steps
    assert not any(k.startswith(("SPLIT:", "EXPAND_FAILED")) for k in g.skips)
    for step in g.steps.values():
        whole = sorted((i, a) for i, r in enumerate(step.reactants)
                       for a in range(g.species[r].n_atoms))  # fmt: skip
        assert sorted(x for slot in step.atom_map for x in slot) == whole
        for p, slot in zip(step.products, step.atom_map, strict=True):
            assert g.species[p].n_atoms == len(slot)
            assert identity_key(parse_mapped(g.species[p].smiles)) == p
    mass, _ = masses(g)
    for node, (inflow, outflow) in atom_balance(g, mass).items():
        assert inflow == pytest.approx(outflow, abs=1e-9), node
    same_pool = pool_of([records[k] for k in amounts.elements()])  # the start pool, as drawn
    assert [p.logp for p in expander.expand(same_pool).proposals] == pytest.approx(
        [p.logp for p in expander.expand(start).proposals], abs=1e-4)  # fmt: skip
    shared = []
    for p in expander.expand(start).proposals:
        step, _, why = split_proposal(start, p)
        if why is None and step.key in g.steps:
            shared.append((p, step))
    assert shared  # the pool graph's first steps are on the species graph
    p, step = shared[0]
    hg = hypergraph_of(g, [(k, g.species[k].skeleton) for k in step.products], ("EXACT",))
    found, _ = k_shortest_hyperpaths(hg, 50, distinct=lambda h: unfold(h).species_class)
    same = [h for h in found if unfold(h).species_class == species_class([step])]
    assert same and same[0].cost == pytest.approx(g.steps[step.key].cost * len(set(step.products)))
    assert compare(same[0], [Edge(start.key, p.child.key, p)], start).same_species_class
    lin = linearize(same[0], numbering=start_numbering(start))
    assert [e.proposal.touched <= set(range(1, 8)) for e in lin] == [True]


def test_the_species_report_of_the_tiny_graph(tiny_graph):
    import argparse
    import json

    from ksmech.network.species_cli import add_parser, species_report

    _, _, _, g = tiny_graph
    step = max(g.steps.values(), key=lambda s: s.logp)
    slots = [(k, g.species[k].skeleton) for k in step.products]
    body = species_report(g, slots, ("EXACT",), k=2)
    json.dumps(body)
    assert body["pools"] == 5 and {r["role"] for r in body["species"]} >= {"start", "frontier"}
    assert sum(r["end_share"] for r in body["species"]) == pytest.approx(1.0)
    assert body["hyperpaths"] and body["hyperpaths"][0]["n_events"] >= 1
    assert all(f["formed"] and 0 < f["share"] <= 1 for f in body["atom_flux"])
    assert "hyperpaths" not in species_report(g)
    sub = argparse.ArgumentParser().add_subparsers(dest="command")
    add_parser(sub)
    args = sub.choices["species"].parse_args(["--run", "r", "--start", "CO", "--k", "2"])
    assert args.k == 2 and args.levels == ["EXACT", "SKELETON"] and args.target is None
