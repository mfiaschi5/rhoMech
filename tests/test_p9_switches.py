"""Open-shell plan P9, step 4 and the end-to-end path: every ablation is a batch-time selection
out of the v19 superset, each switch changes only its named columns, and the three encoders read
the same switches; an open-shell model encodes and decodes a radical pool."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.lewis import reactant_lewis
from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.steps import pair_stats, reactant_sample, with_lewis
from ksmech.eval.embedding_page import token_rows
from ksmech.eval.floor import intra_open_cells
from ksmech.model.config import PRESETS, ModelConfig
from ksmech.model.decode import DecodeConfig, decode_pool
from ksmech.model.inputs import (
    fit_feature_stats,
    open_cells,
    real_cells,
    token_columns,
    without_open_cells,
)
from ksmech.model.net import KSMech
from ksmech.physics.stage_a import (
    G2_FEATURES,
    OPEN_SHELL_BLOCK,
    OPEN_SHELL_PAIR_FEATURES,
    OWN_PAIR_FEATURES,
    cell_features,
)
from ksmech.physics.textbook import network_scorer

TINY = ModelConfig(
    d=32, d_pair=16, layers=1, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)
JOINT = replace(TINY, **PRESETS["open_shell"])
POOLS = ("[CH3:1].[CH3:2]", "[CH3:1][CH2:2]", "[CH3:1][CH3:2]")


def _sample(smiles: str, loader):
    rt = ReactantTokens.from_smiles(smiles, loader)
    return with_lewis(reactant_sample("t", "t", rt), reactant_lewis(rt))


@pytest.fixture(scope="module")
def samples(token_loader):
    return [_sample(x, token_loader) for x in POOLS]


def _model(samples, cfg: ModelConfig = JOINT) -> KSMech:
    torch.manual_seed(0)
    stats = pair_stats(samples, cells=cfg.pair_cells, extra=OPEN_SHELL_BLOCK)
    return KSMech(cfg, fit_feature_stats(samples, *token_columns(cfg)), stats).eval()


def _prepared(samples, cfg: ModelConfig = JOINT):
    model = _model(samples, cfg)
    inp, z, defined, y = model.prepare(collate_samples(samples))
    return model.pair_stats.names, inp, z, defined, y


def _changed(a: torch.Tensor, b: torch.Tensor, names) -> set[str]:
    """The feature names whose column differs (last axis)."""
    return {n for k, n in enumerate(names) if not a[..., k].equal(b[..., k])}


def test_the_open_shell_pair_statistics_extend_the_cell_features(samples):
    stats = pair_stats(samples, cells="all", extra=OPEN_SHELL_BLOCK)
    assert stats.names == (*cell_features("all"), *OPEN_SHELL_BLOCK)
    assert stats.count[stats.names.index("somo_pair")] > 0
    assert pair_stats(samples, cells="all").names == cell_features("all")


def test_the_own_pair_switch_changes_only_its_columns(samples):
    names, _, z, defined, _ = _prepared(samples)
    _, _, z2, d2, _ = _prepared(samples, replace(JOINT, own_pair_channel=False))
    assert _changed(defined, d2, names) == set(OWN_PAIR_FEATURES)
    assert _changed(z, z2, names) == set(OWN_PAIR_FEATURES)
    assert not d2[..., [names.index(n) for n in OWN_PAIR_FEATURES]].any()


def test_a_g_variant_is_a_name_selection(samples):
    names, _, z, defined, _ = _prepared(samples)
    for dropped in (("g", "log_abs_g"), G2_FEATURES):
        cfg = replace(JOINT, drop_pair_features=dropped)
        _, _, z2, d2, _ = _prepared(samples, cfg)
        assert _changed(defined, d2, names) == set(dropped)
        assert _changed(z, z2, names) == set(dropped)


def test_somo_channels_off_clears_the_open_rows_and_hole_columns(samples):
    names, _, z, defined, _ = _prepared(samples)
    _, _, z2, d2, y2 = _prepared(samples, replace(JOINT, somo_channels=False))
    batch = collate_samples(samples)
    off = open_cells(batch)
    assert off.any() and not off[2].any()
    assert not d2[off].any()
    assert not d2[..., [names.index(n) for n in OPEN_SHELL_PAIR_FEATURES]].any()
    rest = real_cells(batch) & ~off
    keep = [k for k, n in enumerate(names) if n not in OPEN_SHELL_PAIR_FEATURES]
    assert z[rest][:, keep].equal(z2[rest][:, keep])
    assert defined[rest][:, keep].equal(d2[rest][:, keep])
    assert y2[2].equal(_prepared(samples)[4][2]), "a closed pool keeps its physics"


def test_without_open_cells_clears_pair_valid_and_leaves_a_closed_batch(samples):
    batch = collate_samples(samples)
    out = without_open_cells(batch)
    off = open_cells(batch)
    assert not out["pair_valid"][off].any()
    assert out["pair_valid"][~off].equal(batch["pair_valid"][~off])
    assert "cells_off" not in batch, "the input batch is not changed"
    closed = collate_samples(samples[2:])
    assert without_open_cells(closed)["pair_defined"].equal(closed["pair_defined"])


def test_the_floor_reads_the_same_open_cells(samples):
    batch = collate_samples(samples)
    assert intra_open_cells(batch).sum() <= open_cells(batch).sum()
    assert not (intra_open_cells(batch) & ~open_cells(batch)).any()


@pytest.mark.parametrize("spin", ["none", "categorical_u", "somo_population"])
def test_the_atom_spin_input_changes_only_its_input(samples, spin):
    cfg = replace(JOINT, atom_spin_input=spin)
    atoms = token_columns(cfg)[0]
    assert ("somo_population" in atoms) == (spin == "somo_population")
    model = _model(samples, cfg)
    assert hasattr(model.encoder.embed, "spin") == (spin == "categorical_u")
    base = token_columns(replace(JOINT, atom_spin_input="none"))[0]
    a = _prepared(samples, replace(JOINT, atom_spin_input="none"))[1]["atom_x"]
    b = _prepared(samples, cfg)[1]["atom_x"]
    assert b[..., [atoms.index(n) for n in base]].equal(a[..., : len(base)])


def test_the_fd_frontier_variant_changes_only_its_columns(samples):
    cfg = replace(JOINT, atom_fukui="fd")
    atoms, base = token_columns(cfg)[0], token_columns(JOINT)[0]
    assert {n for n, m in zip(atoms, base, strict=True) if n != m} == {"f_plus", "f_minus"}


@pytest.mark.parametrize("encoder", ["dense", "flex", "sparse"])
def test_every_encoder_reads_the_same_switches(samples, encoder):
    cfg = replace(JOINT, somo_channels=False, own_pair_channel=False)
    dense = _prepared(samples, cfg)
    other = _prepared(samples, replace(cfg, encoder=encoder, flex_kernel=False))
    for a, b in zip(dense[2:], other[2:], strict=True):
        assert a.equal(b)
    model = _model(samples, replace(cfg, encoder=encoder, flex_kernel=False))
    with torch.no_grad():
        h, *_ = model.encode(collate_samples(samples))
    assert torch.isfinite(h).all()


def test_an_open_shell_model_decodes_a_radical_pool(samples):
    model = _model(samples)
    assert model.decoder.atom_values == (-2, -1, 0, 1, 2)
    skips: Counter = Counter()
    cfg = DecodeConfig(anchors=2, per_anchor=2, max_applied=4)
    found = decode_pool(model, samples[0], cfg, lambda b, n: (repr((b, n)), None), "all", skips)
    assert found and all(c.logp <= 0 for c in found)


def test_a_three_class_model_on_the_same_batch_reads_no_open_shell_block(samples):
    cfg = replace(TINY, pair_cells="all")
    model = KSMech(cfg, fit_feature_stats(samples), pair_stats(samples, cells="all")).eval()
    assert not set(OPEN_SHELL_BLOCK) & set(model.pair_stats.names)
    assert model.decoder.atom_values == (-2, 0, 2)
    assert model.prepare(collate_samples(samples))[0]["atom_x"].isfinite().all()


def test_a_trained_model_scores_every_cell_of_the_radical_suite(samples):
    sample = samples[0]
    scores = network_scorer(_model(samples))(sample, None)
    assert scores.shape == sample.pair_valid.shape
    assert not np.isnan(scores).any()


def test_the_embedding_page_names_a_somo_and_its_hole(samples):
    names = [row["name"] for row in token_rows(samples[0], None)]
    assert any(n.startswith("SOMO ") for n in names)
    assert any(n.startswith("hole ") for n in names)
    closed = [row["name"] for row in token_rows(samples[2], None)]
    assert not any(n.startswith(("SOMO ", "hole ")) for n in closed)
