"""The orbital-token ablations of docs/todo.md (2026-11-09): `ModelConfig.drop_token_features`
removes token input columns by name ('orbital': every orbital column), `spectral_set` the set
encoder over the spectral entries, `inherit_atoms` the centre-weighted atom embeddings; each
switch removes exactly its term, on the fixture molecules, in every encoder."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.lewis import reactant_lewis
from ksmech.arrows.reactants import ReactantTokens
from ksmech.arrows.steps import pair_stats, reactant_sample, with_lewis
from ksmech.eval.embedding import embedding_parts
from ksmech.model.config import PRESETS, ModelConfig
from ksmech.model.inputs import fit_feature_stats, to_device, token_columns
from ksmech.model.net import KSMech
from ksmech.orbitals.tokens import feature_names

TINY = ModelConfig(
    d=32, d_pair=16, layers=1, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)
GENERAL = replace(TINY, **PRESETS["general"])
POOLS = ("CC(=O)C.[OH-]", "CCl.[OH-]", "C=CC=O")  # acetone, chloromethane, acrolein (fixtures)
ARMS = {
    "b": {"drop_token_features": ("orbital",)},
    "c": {"drop_token_features": ("orbital",), "spectral_set": False},
    "d": {"drop_token_features": ("orbital",), "spectral_set": False, "inherit_atoms": False},
}


@pytest.fixture(scope="module")
def samples(token_loader):
    out = []
    for smiles in POOLS:
        rt = ReactantTokens.from_smiles(smiles, token_loader)
        out.append(with_lewis(reactant_sample("t", "t", rt), reactant_lewis(rt)))
    return out


def _model(samples, cfg: ModelConfig) -> KSMech:
    torch.manual_seed(0)
    stats = pair_stats(samples, cells=cfg.pair_cells)
    return KSMech(cfg, fit_feature_stats(samples, *token_columns(cfg)), stats).eval()


def _h0(model, batch) -> torch.Tensor:
    with torch.no_grad():
        inp = model.prepare(batch)[0]
        return model.encoder.embed(batch, inp)


def test_token_columns_drop_by_name_and_the_orbital_group():
    atoms, orbitals = token_columns(GENERAL)
    assert orbitals == (*feature_names(), "retained_fraction", "energy_ev")
    one = replace(GENERAL, drop_token_features=("occupation", "mulliken_charge"))
    a1, o1 = token_columns(one)
    assert "occupation" not in o1 and len(o1) == len(orbitals) - 1
    assert "mulliken_charge" not in a1 and len(a1) == len(atoms) - 1
    assert token_columns(replace(GENERAL, **ARMS["b"])) == (atoms, ())
    with pytest.raises(ValueError, match="not input columns"):
        token_columns(replace(GENERAL, drop_token_features=("no_such_column",)))
    with pytest.raises(ValueError, match="no atom column"):
        token_columns(replace(GENERAL, drop_token_features=atoms))


def test_the_config_round_trips():
    cfg = replace(GENERAL, **ARMS["d"])
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    old = {k: v for k, v in GENERAL.to_dict().items() if k not in ARMS["d"]}  # a pre-ablation run
    back = ModelConfig.from_dict(old)
    assert back.drop_token_features == () and back.spectral_set and back.inherit_atoms


def _perturbed(batch, key, scale=3.0):
    out = dict(batch)
    out[key] = batch[key] + scale * torch.randn_like(batch[key].float()).to(batch[key].dtype)
    return out


@pytest.mark.parametrize("arm", ["a", "b", "c", "d"])
def test_each_switch_removes_exactly_its_term(samples, arm):
    cfg = replace(GENERAL, **ARMS.get(arm, {}))
    model = _model(samples, cfg)
    batch = to_device(collate_samples(samples), torch.device("cpu"))
    base = _h0(model, batch)
    n_atoms = batch["atom_categorical"].shape[1]
    orb = slice(1 + n_atoms, None)  # orbital tokens (MECH at 0, then the atoms)
    torch.manual_seed(1)
    changed = {
        "orbital": not torch.allclose(base, _h0(model, _perturbed(batch, "orbital_features"))),
        "spectral": not torch.allclose(base, _h0(model, _perturbed(batch, "orbital_spectra"))),
        "atoms": not torch.allclose(base[:, orb],
                                    _h0(model, _perturbed(batch, "atom_features"))[:, orb]),
    }  # fmt: skip
    want = {
        "a": {"orbital": True, "spectral": True, "atoms": True},
        "b": {"orbital": False, "spectral": True, "atoms": True},
        "c": {"orbital": False, "spectral": False, "atoms": True},
        "d": {"orbital": False, "spectral": False, "atoms": False},
    }[arm]
    assert changed == want
    parts = embedding_parts(model.encoder.embed, batch, model.prepare(batch)[0])
    with torch.no_grad():
        assert torch.allclose(sum(parts.values()), base, atol=1e-5)


@pytest.mark.parametrize("encoder", ["dense", "flex", "sparse"])
def test_every_encoder_reads_the_switches(samples, encoder):
    cfg = replace(GENERAL, **ARMS["d"], encoder=encoder, flex_kernel=False)
    model = _model(samples, cfg)
    assert model.encoder.embed.orbital is None and model.encoder.embed.spectral is None
    with torch.no_grad():
        h, *_ = model.encode(collate_samples(samples))
    assert torch.isfinite(h).all()


def test_a_run_that_predates_the_ablation_fields_still_resumes():
    """`train.loop.comparable_config`: a header written before `drop_token_features`,
    `spectral_set` and `inherit_atoms` (or `direction`) existed compares equal to today's header
    of the same run, and a run with another value of one of them does not."""
    from dataclasses import asdict

    from ksmech.data import provenance as prov
    from ksmech.train.loop import TrainConfig, comparable_config, train_config

    train = TrainConfig(seed=1, epochs=3)
    now = prov.normalized({"train": train_config(train), "model": GENERAL.to_dict(),
                           "sample_version": 19})  # fmt: skip
    model_old = {k: v for k, v in now["model"].items() if k not in ARMS["d"]}
    train_old = {k: v for k, v in asdict(train).items() if k != "direction"}
    old = prov.normalized({"train": train_old, "model": model_old, "sample_version": 19})
    assert old != now and comparable_config(old) == comparable_config(now)
    ablated = prov.normalized(now | {"model": replace(GENERAL, **ARMS["b"]).to_dict()})
    assert comparable_config(ablated) != comparable_config(now)
