"""`model.decode.matrix_logp`: the log P of a given reaction matrix at a state (reaction-network
plan, item C) is the decoder's own log P of the completion that writes it, anchor by anchor."""

from collections import Counter

import numpy as np
import pytest
import torch

from ksmech.arrows.lewis import reactant_lewis
from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.steps import pair_stats, reactant_sample, with_lewis
from ksmech.model.config import ModelConfig
from ksmech.model.decode import (
    BREAK,
    FORM,
    DecodeConfig,
    EntryTable,
    complete_anchor,
    decode_pool,
    matrix_anchors,
    matrix_logp,
    table_logp,
)
from ksmech.model.inputs import fit_feature_stats
from ksmech.model.net import KSMech

TINY = ModelConfig(
    d=32, d_pair=16, layers=1, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)


def _random_table(n: int, anchor, seed: int = 0, favour=((), ())) -> EntryTable:
    """Random entry log-probabilities around 0 on every entry, `favour`'s entries likelier."""
    rng = np.random.default_rng(seed)
    pair = rng.normal(size=(n, n, 3)) + np.array([0.0, 2.0, 0.0])
    atom = rng.normal(size=(n, 3)) + np.array([0.0, 2.0, 0.0])
    for i, j, d in favour[0]:
        pair[i, j, d + 1] += 4.0
    for i, d in favour[1]:
        atom[i, d // 2 + 1] += 4.0
    pair -= np.log(np.exp(pair).sum(-1, keepdims=True))
    atom -= np.log(np.exp(atom).sum(-1, keepdims=True))
    return EntryTable(anchor, -0.7, pair, atom)


@pytest.mark.parametrize(
    ("anchor", "favour"),
    [((0, 1, FORM), (((0, 1, 1), (1, 2, -1)), ())), ((1, 3, BREAK), (((1, 3, -1),), ((1, 2),)))],
)
def test_table_logp_equals_the_decoders_exact_logp(anchor, favour):
    t = _random_table(6, anchor, favour=favour)
    apply = lambda bonds, lone: (repr((bonds, lone)), None)  # noqa: E731
    found = complete_anchor(t, apply, DecodeConfig(per_anchor=6, max_applied=20), Counter())
    assert found
    for c in found.values():
        assert table_logp(t, c.bonds, c.lone) == pytest.approx(c.logp, abs=1e-9)


def test_a_matrix_the_head_cannot_write_scores_minus_infinity():
    t = _random_table(5, (0, 1, FORM))
    assert table_logp(t, ((0, 1, 1),), ((2, 1),)) == float("-inf")  # odd entry, 3-class head
    assert table_logp(t, ((0, 1, -1),), ()) == float("-inf")  # the anchor's sign disagrees
    br = _random_table(5, (0, 1, BREAK))
    assert table_logp(br, ((0, 1, -1), (2, 3, 1)), ()) == float("-inf")


def test_the_anchors_are_the_forming_bonds_or_else_the_breaking_ones():
    assert matrix_anchors(((3, 1, 1), (0, 2, -1), (0, 4, 1))) == [(0, 4, FORM), (1, 3, FORM)]
    assert matrix_anchors(((2, 0, -1),)) == [(0, 2, BREAK)]
    assert matrix_anchors(()) == []


@pytest.fixture(scope="module")
def model_and_sample(token_loader):
    rt = ReactantTokens.from_smiles("[CH3:1][Cl:2].[OH-:3]", token_loader)
    sample = with_lewis(reactant_sample("t", "t", rt), reactant_lewis(rt))
    torch.manual_seed(0)
    model = KSMech(TINY, fit_feature_stats([sample]), pair_stats([sample])).eval()
    return model, sample


def test_matrix_logp_reproduces_every_decoded_completion(model_and_sample):
    model, sample = model_and_sample
    apply = lambda bonds, lone: (repr((bonds, lone)), None)  # noqa: E731
    cfg = DecodeConfig(anchors=3, per_anchor=3, max_applied=8)
    found = decode_pool(model, sample, cfg, apply, "all", Counter())
    assert found
    for c in found:
        score = matrix_logp(model, sample, c.bonds, c.lone)
        assert score.per_anchor[c.anchor] == pytest.approx(c.logp, abs=1e-6)
        assert score.best >= c.logp - 1e-6 and score.marginal >= score.best
    assert matrix_logp(model, sample, (), ()).best == float("-inf")
