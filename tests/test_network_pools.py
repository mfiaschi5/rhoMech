"""`network.pools`, `network.target` and `network.towards` on the fixture molecules
(reaction-network plan, items C and D): mapped explicit-H pool states, B + R on them, the
arrival levels, the remaining-change bound, and a search towards a product with a tiny model."""

from collections import Counter

import pytest
import torch
from rdkit import Chem

from ksmech.data.chem import parse_mapped
from ksmech.data.resonance import identity_key
from ksmech.model.config import ModelConfig
from ksmech.model.decode import DecodeConfig
from ksmech.network.keys import pool_key_of, split_pool_key
from ksmech.network.pools import ModelExpander, apply_on_mol, pool_state, signature
from ksmech.network.search import ImplicitGraph, SearchParams
from ksmech.network.target import (
    EXACT,
    MAPPING_FAILED,
    SKELETON,
    arrival,
    skeleton_key,
    steps_lower_bound,
    target_from_smiles,
)
from ksmech.network.towards import pathway_rows, search_towards

MECL_OH = "[CH3:1][Cl:2].[OH-:3]"


def _idx(state, map_number: int) -> int:
    mol = parse_mapped(state.smiles)
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == map_number)


def test_a_pool_state_is_explicit_h_with_every_atom_mapped():
    s = pool_state(MECL_OH)
    mol = parse_mapped(s.smiles)
    assert all(a.GetAtomMapNum() for a in mol.GetAtoms())
    assert sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()) == 4
    assert {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1} == {1, 2, 3}
    want = sorted(identity_key(Chem.MolFromSmiles(x)) for x in ("CCl", "[OH-]"))
    assert list(s.species) == want and s.key == pool_key_of(want)
    assert split_pool_key(s.key) == Counter(want)
    assert pool_state(s.smiles) == s  # a state's own SMILES gives it back


def test_b_plus_r_on_the_mapped_pool_writes_the_sn2_products():
    s = pool_state(MECL_OH)
    c, cl, o = (_idx(s, m) for m in (1, 2, 3))
    child = apply_on_mol(parse_mapped(s.smiles), [(c, o, 1), (c, cl, -1)], [(o, -2), (cl, 2)])
    want = sorted(identity_key(Chem.MolFromSmiles(x)) for x in ("CO", "[Cl-]"))
    assert list(child.species) == want
    assert child.orders == {(1, 3): 1} and child.nonbonding[2] == 8 and child.nonbonding[3] == 4
    text, touched = signature([(c, o, 1), (c, cl, -1)], [(o, -2)], {c: 1, cl: 2, o: 3})
    assert text == "1-2:-1;1-3:+1|3:-2" and touched == frozenset({1, 2, 3})


def test_a_homolysis_leaves_two_radicals():
    s = pool_state("[Cl:1][Cl:2]")
    a, b = _idx(s, 1), _idx(s, 2)
    child = apply_on_mol(parse_mapped(s.smiles), [(a, b, -1)], [(a, 1), (b, 1)])
    mol = parse_mapped(child.smiles)
    assert [x.GetNumRadicalElectrons() for x in mol.GetAtoms()] == [1, 1]
    assert child.species[0] == child.species[1] and child.species[0].count("|") >= 2  # u in key


def test_arrival_exact_and_up_to_protons():
    product = pool_state("CC(=O)[O-].[Na+]")
    acid = target_from_smiles("CC(=O)O")
    assert arrival(acid, product.species, product.skeletons) == SKELETON
    assert arrival(acid, product.species, product.skeletons, levels=(EXACT,)) is None
    salt = target_from_smiles("CC(=O)[O-]")
    assert arrival(salt, product.species, product.skeletons) == EXACT
    assert skeleton_key(Chem.MolFromSmiles("C=CO")) == skeleton_key(Chem.MolFromSmiles("CC=O"))
    two = target_from_smiles("CO.CO")  # multiplicity counts
    assert arrival(two, pool_state("CO").species, pool_state("CO").skeletons) is None


def test_the_remaining_change_bounds_the_steps_left():
    start = pool_state(MECL_OH)
    target = target_from_smiles("[CH3:1][OH:3].[Cl-:2]")
    assert target.orders == {(1, 3): 1}
    assert steps_lower_bound(target, start.orders, start.nonbonding, 8, 2) == 1
    done = pool_state("[CH3:1][OH:3].[Cl-:2]")
    assert steps_lower_bound(target, done.orders, done.nonbonding, 8, 2) == 0
    assert target_from_smiles("CO").orders is None  # no maps: no bound
    other = target_from_smiles("[CH3:1][OH:9]")
    assert steps_lower_bound(other, start.orders, start.nonbonding, 8, 2) is None


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
    cfg = DecodeConfig(anchors=10, per_anchor=3, max_applied=16)  # a random model: many drops
    return ModelExpander(model, token_loader, cfg)


def test_the_model_expander_proposes_distinct_mapped_children(expander):
    start = pool_state(MECL_OH)
    ex = expander.expand(start)
    assert ex.failure is None and ex.proposals
    keys = [expander.key(p.child) for p in ex.proposals]
    assert len(set(keys)) == len(keys)  # merged by the child's pool key
    assert [p.logp for p in ex.proposals] == sorted((p.logp for p in ex.proposals), reverse=True)
    for p in ex.proposals:
        mol = parse_mapped(p.child.smiles)
        assert {a.GetAtomMapNum() for a in mol.GetAtoms()} == set(range(1, 8))
        assert p.touched and p.touched <= set(range(1, 8))


def test_a_pool_without_tokens_fails_its_node(expander):
    blind = ModelExpander(expander.model, lambda key: None, expander.cfg)  # a cache with nothing
    ex = blind.expand(pool_state(MECL_OH))
    assert ex.proposals == [] and ex.failure == "MISSING_TOKENS"


def test_search_towards_reaches_a_proposed_child(expander):
    start = pool_state(MECL_OH)
    first = expander.expand(start).proposals[0].child
    target = target_from_smiles(first.smiles)
    res = search_towards(expander, start, target, SearchParams(k=1, max_expansions=1))
    assert res.search.reason == "K_FOUND" and res.search.pathways[0].arrival == EXACT
    assert res.graph.states[res.search.pathways[0].nodes[-1]].key == first.key
    rows = pathway_rows(res)
    assert rows[0]["pools"][0] == start.smiles and rows[0]["mechanism_class"] == 0
    unmapped = target_from_smiles(Chem.MolToSmiles(Chem.MolFromSmiles("CO")))
    res2 = search_towards(expander, start, unmapped, SearchParams(k=1, max_expansions=1,
                                                                  step_cost=0.5))  # fmt: skip
    assert res2.skips[MAPPING_FAILED] == 1
    graph = ImplicitGraph(expander, start)
    assert graph.edges(graph.root) and graph.expanded == 1


def test_a_step_and_its_reverse_are_scored_on_map_numbers(expander):
    start = pool_state(MECL_OH)
    proposal = expander.expand(start).proposals[0]
    forward = expander.step_logp(start, proposal.info["bonds"], proposal.info["lone"])
    assert forward.per_anchor[proposal.info["anchor"]] == pytest.approx(proposal.logp, abs=1e-6)
    assert forward.best >= proposal.logp - 1e-6
    back = expander.reverse_logp(proposal)
    assert not isinstance(back, str) and back.best <= 0.0
    assert expander.step_logp(start, [(1, 99, 1)], []) == "STEP_OUTSIDE_POOL"
