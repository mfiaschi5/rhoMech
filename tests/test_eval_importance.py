"""Permutation importance of the token inputs (`eval.importance`) on samples of the mini
PMechDB steps (the fixture molecules): the groups follow the model's columns, a permutation
moves only its group's entries and only between real tokens of its kind, a pass is
reproducible, and the report has one row per group with the baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.steps import MechanismSample, pair_stats, sample_for_step
from ksmech.eval.importance import (
    NOT_PERMUTED,
    SPECTRAL,
    Group,
    Perturbed,
    groups,
    importance,
    permute_group,
)
from ksmech.model.config import ModelConfig
from ksmech.model.inputs import fit_feature_stats, token_columns
from ksmech.model.net import KSMech

TINY = ModelConfig(
    d=32, d_pair=16, layers=2, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)


@pytest.fixture(scope="module")
def samples(mini_steps, token_loader) -> list[MechanismSample]:
    out = [sample_for_step(mini_steps[k], token_loader) for k in mini_steps]
    out = [s for s in out if isinstance(s, MechanismSample)]
    assert len(out) >= 3
    return out


def _model(samples, **kw) -> KSMech:
    torch.manual_seed(0)
    cfg = replace(TINY, **kw)
    stats = fit_feature_stats(samples, *token_columns(cfg))
    return KSMech(cfg, stats, pair_stats(samples)).eval()


def _real(batch, kind):
    a = batch["atom_categorical"].shape[1]
    return batch["is_atom"][:, :a].bool() if kind == "atom" else batch["is_orbital"][:, a:].bool()


def test_groups_follow_the_model_columns(samples):
    atoms, orbitals = _model(samples, orbital_extra=("energy_ev",)).feature_stats.names()
    found = {g.name: g for g in groups(atoms, orbitals)}
    assert set(atoms) | {"element", "formal_charge", SPECTRAL} <= set(found)
    assert "energy_ev" in found and "retained_fraction" not in found
    assert found["atom:frontier"].columns == (
        "fmo_homo_minus_1", "fmo_homo", "fmo_lumo", "fmo_lumo_plus_1"
    )  # fmt: skip
    assert "orbital:energy" not in found  # one of its two columns: no block
    assert not set(NOT_PERMUTED) & set(found)
    assert {g.name for g in groups(atoms, orbitals, "blocks")} <= {n for n in found if ":" in n}
    with pytest.raises(ValueError):
        groups(atoms, orbitals, "nope")


def test_an_atom_column_moves_with_its_missing_flag_between_real_atoms(samples):
    batch = collate_samples(samples)
    cols = tuple(batch["atom_columns"])
    k = cols.index("mulliken_charge")
    g = Group("mulliken_charge", "atom", ("mulliken_charge",))
    out = permute_group(batch, g, torch.Generator().manual_seed(1))
    real = _real(batch, "atom")
    before, after = batch["atom_features"][real], out["atom_features"][real]
    assert not torch.equal(before[:, k], after[:, k])
    assert Counter(before[:, k].tolist()) == Counter(after[:, k].tolist())
    others = [j for j in range(len(cols)) if j != k]
    assert torch.equal(before[:, others], after[:, others])
    assert torch.equal(batch["atom_features"][~real], out["atom_features"][~real])

    def pairs(b):
        x, m = b["atom_features"][real][:, k], b["atom_missing"][real][:, k]
        return Counter(zip(x.tolist(), m.tolist(), strict=True))

    assert pairs(batch) == pairs(out)
    assert out["orbital_features"] is batch["orbital_features"]


def test_orbital_spectral_and_categorical_groups_touch_only_their_inputs(samples):
    batch = collate_samples(samples)
    gen = torch.Generator().manual_seed(2)
    real = _real(batch, "orbital")
    spec = permute_group(batch, Group(SPECTRAL, "spectral"), gen)
    assert Counter(map(tuple, spec["orbital_spectra"][real].flatten(1).tolist())) == Counter(
        map(tuple, batch["orbital_spectra"][real].flatten(1).tolist())
    )
    assert spec["orbital_features"] is batch["orbital_features"]
    el = permute_group(batch, Group("element", "atom_categorical", (0,)), gen)
    assert torch.equal(el["atom_categorical"][..., 1:], batch["atom_categorical"][..., 1:])
    occ = tuple(batch["orbital_columns"]).index("occupation")
    o = permute_group(batch, Group("occupation", "orbital", ("occupation",)), gen)
    assert Counter(o["orbital_features"][real][:, occ].tolist()) == Counter(
        batch["orbital_features"][real][:, occ].tolist()
    )


def test_a_pass_is_reproducible(samples):
    loader = [collate_samples(samples[:2]), collate_samples(samples[2:])]
    g = Group("mulliken_charge", "atom", ("mulliken_charge",))
    fn = lambda b, gen: permute_group(b, g, gen)  # noqa: E731
    a, b = list(Perturbed(loader, fn, 7)), list(Perturbed(loader, fn, 7))
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x["atom_features"], y["atom_features"])


def test_report_has_every_group_and_the_baseline(samples):
    model = _model(samples)
    loader = [collate_samples(samples)]
    found = groups(*model.feature_stats.names(), "blocks")[:2]
    found.append(Group("element", "atom_categorical", (0,)))
    report = importance(model, loader, found, 2, 0, torch.device("cpu"), None)
    assert report["baseline"]["n"] == len(samples)
    assert {r["group"] for r in report["groups"]} == {g.name for g in found}
    nll = [r["d_anchor_nll"] for r in report["groups"]]
    assert nll == sorted(nll, reverse=True)
    assert all(r["d_anchor_nll_sd"] >= 0 for r in report["groups"])
