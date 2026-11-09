"""Prediction without a product on real pool states (reaction-network plan, items E, F, I4): the
reservoir's copies kept in every pool, and the tiny model of `test_network_pools` scoring the
reverse edges of an exploration end to end."""

from collections import Counter

import pytest
import torch

from ksmech.data.chem import parse_mapped
from ksmech.model.decode import DecodeConfig
from ksmech.network.chain import stationary
from ksmech.network.explore import UNEXPLORED, ExploreParams, explore, pathways_to_ends
from ksmech.network.free_energy import ThermoModel
from ksmech.network.pools import ModelExpander, pool_state
from ksmech.network.reservoir import (
    KEPT_TOUCHED,
    RESERVOIR_DRAW,
    ReservoirExpander,
    normalize_pool,
    reservoir_from_smiles,
)
from ksmech.network.search import Expansion, Proposal
from ksmech.network.stop import ModelRescan
from tests.test_network_pools import MECL_OH, TINY

WATER = reservoir_from_smiles(["O"])


def _count(state, smiles):
    return Counter(state.species)[pool_state(smiles).species[0]]


def test_the_reservoir_is_named_by_identity_keys_and_kept_in_every_pool():
    (key,) = WATER.keys
    assert key == pool_state("O").species[0] and WATER.smiles[key] == "O"
    start, drawn = normalize_pool(pool_state(MECL_OH), WATER)
    assert _count(start, "O") == 1 and drawn == Counter({key: 1})
    assert normalize_pool(start, WATER) == (start, Counter())  # already normal
    two, _ = normalize_pool(pool_state("CCl.O.O"), WATER)
    assert _count(two, "O") == 1 and _count(two, "CCl") == 1
    with pytest.raises(ValueError):
        reservoir_from_smiles(["O.O"])


def test_a_released_copy_the_step_touches_is_never_removed():
    pool = pool_state("[CH3:1][Cl:2].[OH2:3].[OH2:4]")
    skips = Counter()
    kept, drawn = normalize_pool(pool, WATER, keep=frozenset({3}), skips=skips)
    maps = {a.GetAtomMapNum() for a in parse_mapped(kept.smiles).GetAtoms()}
    assert 3 in maps and 4 not in maps and drawn == Counter({next(iter(WATER.keys)): -1})
    both, _ = normalize_pool(pool, WATER, keep=frozenset({3, 4}), skips=skips)
    assert _count(both, "O") == 2 and skips[KEPT_TOUCHED] == 1


class _Consumes:
    """Proposes CH3Cl + H2O -> CH3OH2+ + Cl- on map numbers: the reservoir's water is used."""

    def key(self, state):
        return state.key

    def expand(self, state):
        child = pool_state("[CH3:1][OH2+:4].[Cl-:2].[OH-:3]")
        return Expansion([Proposal(child, -0.5, "s", frozenset({1, 2, 4}))])


def test_the_reservoir_expander_replenishes_and_records_the_draw():
    exp = ReservoirExpander(_Consumes(), WATER)
    start = exp.start(pool_state("[CH3:1][Cl:2].[OH-:3]"))
    (p,) = exp.expand(start).proposals
    assert _count(p.child, "O") == 1 and p.info[RESERVOIR_DRAW] == {next(iter(WATER.keys)): 1}
    assert exp.rescan is None and exp.reverse_logp is None  # the inner expander has neither


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
    cfg = DecodeConfig(anchors=4, per_anchor=2, max_applied=16)
    return ModelExpander(model, token_loader, cfg)


def test_the_tiny_model_scores_every_reverse_edge_end_to_end(expander):
    synthetic = ThermoModel(lambda state, t: -1.0 * len(state.species))  # kcal/mol, synthetic
    params = ExploreParams(max_expansions=1, mass_tol=0.0, margin_kcal=0.0)
    res = explore(ModelRescan(expander), pool_state(MECL_OH), synthetic, params)
    chain, root = res.chain, res.graph.root
    assert res.reason == "EXPANSION_BUDGET" and res.graph.expanded == 1
    assert chain.edges and all(e.source == root for e in chain.edges)
    missing = sum(n for k, n in chain.counts.items() if k.startswith("REVERSE_MISSING"))
    scored = [e for e in chain.edges if e.reverse_logp is not None]
    assert len(scored) + missing == len(chain.edges) and scored
    edge = next(e for e in res.graph.out[root] if e.target == scored[0].target)
    assert scored[0].reverse_logp == pytest.approx(expander.reverse_logp(edge.proposal).best)
    for e in scored:  # two-way links with the thermodynamic weights
        i, j = chain.index[e.source], chain.index[e.target]
        assert chain.rates[i, j] > 0 and chain.rates[j, i] > 0 and e.dg is not None
    assert all(res.absorbing[k] == UNEXPLORED for k in chain.keys if k != root)
    assert res.unexplored_mass == pytest.approx(1.0) and res.ends == []
    balanced = [root] + [e.target for e in scored]
    pi = stationary(chain, balanced)
    want = {k: v for k, v in res.equilibrium.items() if k in balanced}
    z = sum(want.values())
    assert pi == pytest.approx({k: v / z for k, v in want.items()}, rel=1e-6)
    for _, tp in pathways_to_ends(res):
        assert tp is None or abs(sum(p.share for p in tp.pathways) - 1.0) < 1e-6


def test_the_model_rescan_widens_the_decode_and_restores_it(expander):
    wide = ModelRescan(expander)
    start = pool_state(MECL_OH)
    narrow = {p.child.key for p in wide.expand(start).proposals}
    cfg = expander.cfg
    wider = {p.child.key for p in wide.rescan(start, 3).proposals}
    assert expander.cfg == cfg and narrow <= wider and len(wider) > len(narrow)


def test_the_predict_command_parses():
    from ksmech.network.cli import build_parser

    argv = ["predict", "--run", "r", "--start", "CC(C)=O.C[O-]", "--temperature-k", "383.15",
            "--reservoir", "CO"]  # fmt: skip
    args = build_parser().parse_args(argv)
    assert args.temperature_k == 383.15 and args.reservoir == ["CO"] and args.mass_tol == 0.01
    assert args.free_energy == "none" and args.max_climb == 20.0
