"""Open-shell plan P9, steps 1-3: the configuration, the 5-class atom head with its bookkeeping
masks, the loss and the beam decoder on the radical shapes, and the closed-shell decode screen
with the optional -1 filter."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from ksmech.arrows.delta_be import ReactionMatrix
from ksmech.eval.decode import decode_report
from ksmech.model import decode as nd
from ksmech.model.config import PRESETS, ModelConfig
from ksmech.model.decoder import ATOM_VALUES, BREAK, FORM, OPEN_SHELL_ATOM_VALUES, EditDecoder
from ksmech.model.loss import atom_classes, greedy_edit, step_loss

CFG = ModelConfig(d=16, d_pair=8, layers=1, heads=2, decoder_hidden=8, physics_hidden=4)
NEW_FIELDS = ("open_shell", "somo_channels", "own_pair_channel")
# name -> (bond entries, atom entries, anchor); atoms 0-5, every R conserves charge
SHAPES = {
    "homolysis": (((0, 1, -1),), ((0, 1), (1, 1)), (0, 1, BREAK)),
    "recombination": (((0, 1, 1),), ((0, -1), (1, -1)), (0, 1, FORM)),
    "addition": (((0, 1, 1), (1, 2, -1)), ((0, -1), (2, 1)), (0, 1, FORM)),
    "abstraction": (((0, 1, 1), (1, 3, -1)), ((0, -1), (3, 1)), (0, 1, FORM)),
    "beta_scission": (((0, 1, -1), (1, 2, 1)), ((0, 1), (2, -1)), (1, 2, FORM)),
    "disproportionation": (((0, 1, 1), (1, 2, -1), (2, 3, 1)), ((0, -1), (3, -1)), (0, 1, FORM)),
    "fragmentation": (((0, 1, -1), (1, 2, 1), (2, 3, -1)), ((0, 1), (3, 1)), (1, 2, FORM)),
    "cyclization": (((0, 4, 1), (4, 5, -1)), ((0, -1), (5, 1)), (0, 4, FORM)),
}


def test_the_new_fields_default_to_todays_behaviour():
    assert ModelConfig().open_shell is False
    assert ModelConfig().somo_channels and ModelConfig().own_pair_channel
    assert {f.name for f in fields(ModelConfig)} >= set(NEW_FIELDS)


def test_a_run_json_without_the_new_keys_reads_the_three_class_head():
    old = {k: v for k, v in CFG.to_dict().items() if k not in NEW_FIELDS}
    cfg = ModelConfig.from_dict(old)
    assert cfg == CFG
    fresh, again = EditDecoder(CFG, 5), EditDecoder(cfg, 5)
    assert again.load_state_dict(fresh.state_dict(), strict=True)
    assert again.atom_values == ATOM_VALUES and again.n_out.out_features == 3


def test_a_five_class_configuration_round_trips():
    cfg = replace(CFG, **PRESETS["open_shell"])
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    dec = EditDecoder(ModelConfig.from_dict(cfg.to_dict()), 5)
    assert dec.atom_values == OPEN_SHELL_ATOM_VALUES and dec.n_out.out_features == 5


def _entry_batch(nonbonding: list[int]) -> dict:
    n = len(nonbonding)
    nb = torch.tensor([nonbonding], dtype=torch.int8)
    return {
        "bond_order": torch.zeros(1, n, n, dtype=torch.int8),
        "graph_distance": torch.ones(1, n, n, dtype=torch.int16),
        "can_lose_pair": nb >= 2,
        "can_lose_electron": nb >= 1,
    }


def _atom_logits(nonbonding: list[int]) -> torch.Tensor:
    cfg = replace(CFG, open_shell=True)
    dec = EditDecoder(cfg, 5)
    n = len(nonbonding)
    h, p = torch.randn(1, n + 1, cfg.d), torch.randn(1, n, n, cfg.d_pair)
    rows = torch.tensor([[0, 0, 1, FORM]])
    return dec.entry_logits(h, p, _entry_batch(nonbonding), rows)[1][0]


def test_the_atom_masks_are_the_non_negative_n_laws():
    """-2 needs N >= 2, -1 needs N >= 1; +1 and +2 are never masked."""
    atom = _atom_logits([0, 1, 2, 3])
    assert atom[:, 0].isinf().tolist() == [True, True, False, False]
    assert atom[:, 1].isinf().tolist() == [True, False, False, False]
    assert atom[:, 2:].isfinite().all()


def test_no_pool_level_rule_masks_a_minus_one_on_a_closed_pool():
    """Every N even (u = 0): a -1 entry keeps a finite log-probability, so its loss is finite."""
    atom = _atom_logits([2, 2, 4])
    assert atom[:, 1].isfinite().all()
    assert torch.log_softmax(atom, -1)[:, 1].isfinite().all()


class _Uniform:
    """Six atoms, one candidate anchor for every (i, j, sign), every class equally likely."""

    atom_values = OPEN_SHELL_ATOM_VALUES

    def stop_logits(self, h):
        return torch.zeros(1)

    def anchor_logits(self, h, p, y, batch):
        return torch.zeros(1, 6, 6, 2)

    def entry_logits(self, h, p, batch, anchors):
        return torch.zeros(len(anchors), 6, 6, 3), torch.zeros(len(anchors), 6, 5)


def _label_batch(bonds, lone, anchor) -> dict:
    n = 6
    b = {
        "r_pairs": torch.zeros(1, 1, n, n, dtype=torch.int8),
        "r_atoms": torch.zeros(1, 1, n, dtype=torch.int8),
        "r_anchors": torch.zeros(1, 1, n, n, 2, dtype=torch.bool),
        "variant_mask": torch.ones(1, 1, dtype=torch.bool),
        "is_atom": torch.ones(1, n, dtype=torch.bool),
        "bond_order": torch.zeros(1, n, n, dtype=torch.int8),
    }
    for i, j, d in bonds:
        b["r_pairs"][0, 0, i, j] = b["r_pairs"][0, 0, j, i] = d
    for i, d in lone:
        b["r_atoms"][0, 0, i] = d
    i, j, sign = anchor
    b["r_anchors"][0, 0, i, j, sign] = b["r_anchors"][0, 0, j, i, sign] = True
    return b


@pytest.mark.parametrize("shape", SHAPES)
def test_the_loss_takes_each_radical_shape(shape):
    """-log P(anchor) - sum of every other entry's class: 14 pairs at 1/3 and 6 atoms at 1/5."""
    bonds, lone, anchor = SHAPES[shape]
    loss, _ = step_loss(_Uniform(), None, None, None, _label_batch(bonds, lone, anchor))
    expect = math.log(72) + 14 * math.log(3) + 6 * math.log(5)
    assert loss.item() == pytest.approx(expect, rel=1e-6)


def test_the_three_class_head_still_refuses_an_odd_entry():
    with pytest.raises(ValueError, match="outside the head's alphabet"):
        atom_classes(torch.tensor([[1]]), ATOM_VALUES, torch.tensor([[True]]))
    real = torch.ones(1, 5, dtype=torch.bool)
    classes = atom_classes(torch.tensor([[-2, -1, 0, 1, 2]]), OPEN_SHELL_ATOM_VALUES, real)
    assert classes.tolist() == [[0, 1, 2, 3, 4]]


def _table(bonds, lone, anchor, n: int = 6, closed: bool = False) -> nd.EntryTable:
    """0.9 on each entry of the shape, 0.98 on 0 everywhere else, over the 5-class alphabet."""
    pair = np.log(np.tile([0.01, 0.98, 0.01], (n, n, 1)))
    atom = np.log(np.tile([0.005, 0.005, 0.98, 0.005, 0.005], (n, 1)))
    for i, j, d in bonds:
        pair[i, j] = np.log([0.9 if d == -1 else 0.05, 0.05, 0.9 if d == 1 else 0.05])
    for i, d in lone:
        row = np.full(5, 0.025)
        row[OPEN_SHELL_ATOM_VALUES.index(d)] = 0.9
        atom[i] = np.log(row)
    return nd.EntryTable(anchor, -0.1, pair, atom, OPEN_SHELL_ATOM_VALUES, closed)


def _apply(bonds, lone):
    return repr((bonds, lone)), None


@pytest.mark.parametrize("shape", SHAPES)
def test_the_beam_writes_each_radical_shape_first(shape):
    bonds, lone, anchor = SHAPES[shape]
    cfg = nd.DecodeConfig(max_atoms=4)
    found = list(nd.complete_anchor(_table(bonds, lone, anchor), _apply, cfg, Counter()).values())
    assert found[0].bonds == tuple(sorted(bonds)) and found[0].lone == tuple(sorted(lone))


@pytest.mark.parametrize("shape", SHAPES)
def test_a_completions_log_p_is_the_sum_over_its_entries(shape):
    """The separate pair and atom arrays add up as one entry list: the search's score is the
    exact sum, background included."""
    bonds, lone, anchor = SHAPES[shape]
    table = _table(bonds, lone, anchor)
    pb = nd.problem(table, 48)
    score, chosen = next(nd.ranked_completions(pb, 7, 4, 20_000))
    assert score == pytest.approx(nd.exact_logp(pb, nd.classes(pb, chosen), table.anchor_logp))


class _Greedy(_Uniform):
    """The class of each entry of one shape wins."""

    def __init__(self, bonds, lone):
        self.bonds, self.lone = bonds, lone

    def entry_logits(self, h, p, batch, anchors):
        pair, atom = torch.zeros(1, 6, 6, 3), torch.zeros(1, 6, 5)
        pair[..., 1], atom[..., 2] = 1.0, 1.0
        for i, j, d in self.bonds:
            pair[0, i, j, d + 1] = 5.0
        for i, d in self.lone:
            atom[0, i, OPEN_SHELL_ATOM_VALUES.index(d)] = 5.0
        return pair, atom


def test_greedy_edit_reads_the_five_class_values():
    bonds, lone, (a, b, sign) = SHAPES["homolysis"]
    lp = torch.full((1, 6, 6, 2), -9.0)
    lp[0, a, b, sign] = 0.0
    batch = {"is_atom": torch.ones(1, 6, dtype=torch.bool), "bond_order": torch.zeros(1, 6, 6)}
    pairs, atoms = greedy_edit(_Greedy(bonds, lone), None, None, lp, batch)
    assert pairs[0, 0, 1] == -1 and atoms[0].tolist() == [1, 1, 0, 0, 0, 0]


HOMOLYSIS = (((0, 1, -1),), ((0, 1), (1, 1)))
AZOMETHANE = (((0, 1, -1), (1, 2, 1), (2, 3, -1)), ((0, 1), (3, 1)))  # CN=NC -> .CH3 + N2 + .CH3
CL2_RH = (((0, 1, -1), (1, 2, 1), (2, 3, -1)), ((0, 1), (3, 1)))  # Cl-Cl + H-C -> Cl. + HCl + C.
OFF_BOND = (((0, 1, 1), (2, 3, -1)), ((4, 1), (5, -1)))  # a +1 on no broken bond
MINUS_ONE = (((0, 1, 1), (2, 3, -1)), ((0, -1), (2, 1)))  # X. + H-Y drawn on a closed pool
FORM_ANCHOR, BREAK_ANCHOR = (0, 1, FORM), (0, 1, BREAK)


def _screen(step, anchor, closed=True, **kw):
    return nd.screen(*step, anchor, nd.DecodeConfig(max_atoms=4, **kw), closed)


@pytest.mark.parametrize("step", [HOMOLYSIS, AZOMETHANE, CL2_RH])
def test_a_closed_pool_keeps_the_steps_that_turn_n_odd_on_a_broken_bond(step):
    assert _screen(step, BREAK_ANCHOR if step is HOMOLYSIS else (1, 2, FORM)) is None


def test_a_lone_plus_one_off_every_broken_bond_is_dropped():
    assert _screen(OFF_BOND, FORM_ANCHOR) == nd.ODD_OFF_BROKEN_BOND


def test_the_minus_one_part_is_only_the_optional_filter():
    assert _screen(MINUS_ONE, FORM_ANCHOR) is None
    assert _screen(MINUS_ONE, FORM_ANCHOR, closed_minus1_filter=True) == nd.CLOSED_MINUS1


def test_with_both_on_each_odd_entry_is_plus_one_on_a_broken_bond():
    assert _screen(HOMOLYSIS, BREAK_ANCHOR, closed_minus1_filter=True) is None
    assert _screen(OFF_BOND, FORM_ANCHOR, closed_minus1_filter=True) == nd.CLOSED_MINUS1


def test_the_screen_does_not_act_on_an_open_pool():
    for step in (OFF_BOND, MINUS_ONE):
        assert _screen(step, FORM_ANCHOR, closed=False, closed_minus1_filter=True) is None


def test_a_screened_completion_is_counted_and_never_applied():
    applied = []

    def apply(bonds, lone):
        applied.append((bonds, lone))
        return "key", None

    skips: Counter = Counter()
    table = _table(*OFF_BOND, FORM_ANCHOR, closed=True)
    nd.complete_anchor(table, apply, nd.DecodeConfig(max_atoms=2), skips)
    assert skips[f"DECODE_DROPPED:{nd.ODD_OFF_BROKEN_BOND}"] >= 1
    assert OFF_BOND not in applied


def _sample(step_id: str, label: ReactionMatrix):
    from types import SimpleNamespace

    from ksmech.arrows.delta_be import EditLabel

    return SimpleNamespace(
        step_id=step_id, edit_label=EditLabel((label,), (label.anchors(),), True, 1, False)
    )


def test_the_report_and_its_hit_rows_carry_the_screen():
    """The hit rows reproduce the aggregate, name what the screen dropped per sample and say
    whether the best completion is odd."""
    good = ReactionMatrix(*HOMOLYSIS)
    polar = ReactionMatrix(((0, 1, -1),), ((0, 2),))

    def decode(sample, skips):
        if sample.step_id == "miss":
            skips[f"DECODE_DROPPED:{nd.ODD_OFF_BROKEN_BOND}"] += 2
            return [nd.Completion(-1.0, polar.bonds, polar.lone, (0, 1, BREAK))]
        return [nd.Completion(-1.0, good.bonds, good.lone, (0, 1, BREAK))]

    report = decode_report([_sample("hit", good), _sample("miss", good)], decode)
    rows = {r["step_id"]: r for r in report["per_sample"]}
    assert report["top1"] == sum(r["rank"] == 0 for r in rows.values()) / len(rows) == 0.5
    assert rows["hit"]["odd_top1"] and not rows["miss"]["odd_top1"]
    assert report["odd_top1"] == 0.5
    assert rows["miss"]["screened"] == {nd.ODD_OFF_BROKEN_BOND: 2}
    assert "screened" not in rows["hit"]
    assert report["skips"][f"DECODE_DROPPED:{nd.ODD_OFF_BROKEN_BOND}"] == 2
