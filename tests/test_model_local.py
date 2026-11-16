"""The local selector (docs/decisions.md, 2026-11-16; preset `local`) on the fixture samples:
with `attention_scope='molecule'` a token vector depends on its own molecule only, and the
logit of an atom pair on the molecules of its two atoms; the orbital-pair anchor head starts at
the physics floor, reads no atom token, and its terms sum to the logit. The defaults are the
model as it was."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.eval import ood
from ksmech.eval.embedding import anchor_terms, embedding_parts, term_names, trace
from ksmech.eval.floor import _ranked, floor_scores
from ksmech.model.config import PRESETS, ModelConfig, check_local
from ksmech.model.decoder import BREAK, FORM, ORBITAL_TERMS
from ksmech.model.encoder import PairVectors, attention_keys
from ksmech.model.inputs import _pool, cell_mask, centre_weights
from ksmech.model.loss import anchor_log_probs, anchor_ranks, greedy_edit, step_loss
from ksmech.train.cli import build_parser
from ksmech.train.loop import TrainConfig, fit, make_loader
from tests.test_model import TINY, _two_source_stores, make, samples  # noqa: F401

LOCAL = PRESETS["local"]


def local(samples, **kw):  # noqa: F811
    """The local selector at the fixture size, its learned terms away from zero: the anchor
    head's output layer and its charge coefficients are random, not at their start."""
    model = make(samples, **(LOCAL | kw))
    head = model.decoder.orbital
    if head is not None:
        torch.manual_seed(1)
        for layer in (head.score[-1], head.donor, head.acceptor):
            torch.nn.init.normal_(layer.weight, std=0.5)
        if head.charge is not None:
            torch.nn.init.normal_(head.charge)
    return model


def _logits(model, batch):
    with torch.no_grad():
        h, p, y, _ = model.encode(batch)
        return h, p, y, model.decoder.anchor_logits(h, p, y, batch)


def _weights(batch: dict, x: torch.Tensor):
    w_occ, w_virt = centre_weights(batch)
    return w_occ, x[..., None], w_virt


def _two_molecules(samples) -> dict:  # noqa: F811
    several = [s for s in samples if len(set(s.atom_fragment.tolist())) > 1]
    assert several, "the fixture needs a pool of several molecules"
    return collate_samples(several)


def _moved(batch: dict, fragment: int) -> dict:
    """The batch with every token input of the molecule `fragment` changed."""
    a = batch["atom_features"].shape[1]
    frag = batch["fragment"]
    out = dict(batch)
    atoms = (frag[:, :a] == fragment) & batch["is_atom"][:, :a]
    orbitals = (frag[:, a:] == fragment) & batch["is_orbital"][:, a:]
    out["atom_features"] = batch["atom_features"] + 0.7 * atoms[..., None]
    out["orbital_features"] = batch["orbital_features"] + 0.7 * orbitals[..., None]
    assert atoms.any() and orbitals.any()
    return out


def test_the_defaults_are_the_model_as_it_was(samples):  # noqa: F811
    old = {k: v for k, v in TINY.to_dict().items() if k not in LOCAL or k == "pair_cells"}
    old.pop("anchor_charge_term")
    cfg = ModelConfig.from_dict(old)  # a checkpoint written before the local selector
    assert (cfg.attention_scope, cfg.anchor_head, cfg.charge_embedding) == ("pool", "pointer", True)
    keys = set(make(samples).state_dict())
    assert {"decoder.u_sign.weight", "encoder.embed.charge.weight"} <= keys
    assert not any(k.startswith("decoder.orbital") for k in keys)
    new = set(make(samples, **LOCAL).state_dict())
    gone = {"u_sign", "w_sign", "step_query", "physics"}
    assert {k.split(".")[1] for k in keys - new if k.startswith("decoder.")} == gone
    assert "encoder.embed.charge.weight" not in new and "encoder.embed.fragment.weight" not in new
    head = ("kernel_scale", "kernel_bias", "off_level", "charge")
    layers = ("score.0", "score.2", "donor", "acceptor")
    score = (f"{n}.{w}" for n in layers for w in ("weight", "bias"))
    assert new - keys == {f"decoder.orbital.{k}" for k in (*head, *score)}


def test_the_options_are_checked(samples):  # noqa: F811
    with pytest.raises(ValueError, match="attention_scope"):
        check_local(replace(TINY, attention_scope="fragment"))
    for encoder in ("flex", "sparse"):
        with pytest.raises(ValueError, match="need encoder dense"):
            make(samples, encoder, attention_scope="molecule")
        with pytest.raises(ValueError, match="need encoder dense"):
            make(samples, encoder, anchor_head="orbital_pairs")
    args = build_parser("local").parse_args(["fit", "--run", "x", "--preset", "local"])
    assert {k: getattr(args, k) for k in LOCAL} == {
        k: list(v) if isinstance(v, tuple) else v for k, v in LOCAL.items()
    }


def test_a_token_reads_its_own_molecule_only(samples):  # noqa: F811
    batch = _two_molecules(samples)
    keys = attention_keys(batch, "molecule")
    real = batch["token_mask"]
    off = ~torch.eye(real.shape[1], dtype=torch.bool)
    assert torch.equal(keys[:, 1:, 1:] & off, batch["same_fragment"] & off)
    assert torch.equal(keys[:, 0, 1:], real) and keys[:, 0, 0].all()  # MECH reads every token
    assert not keys[:, 1:, 0].any()  # and no token reads MECH
    assert keys.any(-1).all()  # a padded query has a key: no empty softmax
    assert attention_keys(batch, "pool").shape == (real.shape[0], real.shape[1] + 1)


@pytest.mark.parametrize("fragment", [0, 1])
def test_a_logit_depends_on_the_molecules_of_its_two_atoms_only(samples, fragment):  # noqa: F811
    batch = _two_molecules(samples)
    moved = _moved(batch, fragment)
    a = batch["atom_features"].shape[1]
    there = (batch["fragment"][:, :a] == fragment) & batch["is_atom"][:, :a]
    away = batch["is_atom"][:, :a] & ~there
    apart = away[:, :, None] & away[:, None, :]  # pairs with no atom in the moved molecule
    cand = torch.stack([batch["can_form"], batch["can_break"]], -1)
    model = local(samples)
    h0, p0, _, l0 = _logits(model, batch)
    h1, p1, _, l1 = _logits(model, moved)
    untouched = cand & apart[..., None]
    touched = cand & (there[:, :, None] ^ there[:, None, :])[..., None]
    assert untouched.any() and touched.any()
    torch.testing.assert_close(l1[untouched], l0[untouched], atol=1e-5, rtol=0)
    assert not torch.allclose(l1[touched], l0[touched], atol=1e-3)
    torch.testing.assert_close(h1[:, 1 : a + 1][away], h0[:, 1 : a + 1][away], atol=1e-5, rtol=0)
    torch.testing.assert_close(p1.atoms[apart], p0.atoms[apart], atol=1e-5, rtol=0)
    assert not torch.allclose(h1[:, 0], h0[:, 0], atol=1e-4)  # MECH reads the whole pool
    pool = local(samples, attention_scope="pool")
    ha, hb = _logits(pool, batch)[0], _logits(pool, moved)[0]
    assert not torch.allclose(hb[:, 1 : a + 1][away], ha[:, 1 : a + 1][away], atol=1e-4)


def test_the_entries_given_an_anchor_stay_within_its_molecules(samples):  # noqa: F811
    batch = _two_molecules(samples)
    a = batch["atom_features"].shape[1]
    last = int(batch["fragment"][0, :a][batch["is_atom"][0, :a]].max())
    moved = _moved(batch, last)
    away = batch["is_atom"][:, :a] & (batch["fragment"][:, :a] != last)
    i, j = (away[0, :, None] & away[0, None, :]).triu(1).nonzero()[0].tolist()
    rows = torch.tensor([[0, i, j, FORM]])
    model = local(samples)
    out = []
    for b in (batch, moved):
        h, p, _, _ = _logits(model, b)
        with torch.no_grad():
            out.append(model.decoder.entry_logits(h, p, b, rows))
    keep = away[0, :, None] & away[0, None, :]
    torch.testing.assert_close(out[1][0][0][keep], out[0][0][0][keep], atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1][1][0][away[0]], out[0][1][0][away[0]], atol=1e-5, rtol=0)


def test_the_reactant_order_and_the_batch_change_nothing(samples):  # noqa: F811
    batch = collate_samples(samples)
    model = local(samples)
    _, _, _, logits = _logits(model, batch)
    relabelled = ood.permute_fragments(batch, torch.Generator().manual_seed(0))
    assert not torch.equal(relabelled["fragment"], batch["fragment"])
    assert torch.equal(_logits(model, relabelled)[3], logits)
    for k, s in enumerate(samples):
        alone = _logits(model, collate_samples([s]))[3][0]
        n = alone.shape[0]
        real = torch.isfinite(alone)
        torch.testing.assert_close(alone[real], logits[k, :n, :n][real], atol=1e-4, rtol=0)
    assert all(torch.isfinite(x).all() for x in _logits(model, batch)[:1])


def test_an_untrained_head_is_the_physics_floor(samples):  # noqa: F811
    batch = collate_samples(samples)
    model = make(samples, **LOCAL)  # as initialized: f = 0, c = 0
    head = model.decoder.orbital
    with torch.no_grad():  # a = 1, b = 0, no level off the kernel: the floor itself
        head.kernel_scale.fill_(1.0)
        head.kernel_bias.fill_(0.0)
        head.off_level.fill_(float("-inf"))
    h, p, y, logits = _logits(model, batch)
    assert isinstance(p, PairVectors) and p.cells.shape[:3] == batch["pair_valid"].shape
    assert p.kernel_on[..., FORM].any() and p.kernel_on[..., BREAK].any()
    assert not (p.kernel_on[..., FORM] & p.kernel_on[..., BREAK]).any()
    names = tuple(batch["pair_features"])
    g = batch["pair_z"][..., names.index("g")]
    valid = cell_mask(batch, "valid")
    g_abs = _pool(*_weights(batch, torch.where(valid, g.abs(), 0.0)))[..., 0]
    _, b_own = floor_scores(batch, "valid", "g", False)
    cand = torch.stack([batch["can_form"], batch["can_break"]], -1)
    cand &= torch.ones_like(cand[0, ..., 0]).triu(1)[None, :, :, None]
    floor = torch.stack([g_abs, b_own], -1)  # sum of |g|: a pair with g > 0 does not cancel
    some = cand & (floor > 1e-6)
    assert some[..., FORM].any() and some[..., BREAK].any()
    torch.testing.assert_close(logits[some].exp(), floor[some], atol=1e-6, rtol=1e-4)
    assert torch.isinf(logits[~cand]).all() and torch.isfinite(logits[cand]).all()
    # the ranking of `eval.floor`'s `own_break`, on the sum of |g| in place of |G^atom|
    admissible = (batch["r_anchors"] & batch["variant_mask"][:, :, None, None, None]).any(1)
    rank, reachable = _ranked(floor, cand, admissible & cand)
    assert reachable.all()
    assert torch.equal(anchor_ranks(anchor_log_probs(logits), batch), rank)


def test_the_terms_sum_to_the_logit_and_no_orbital_pair_means_no_logit(samples):  # noqa: F811
    batch = collate_samples(samples)
    model = local(samples)
    h, p, y, logits = _logits(model, batch)
    assert term_names(model) == ORBITAL_TERMS and term_names(make(samples)) != ORBITAL_TERMS
    with torch.no_grad():
        terms = anchor_terms(model, h, p, y, batch)
    ok = torch.isfinite(logits)
    torch.testing.assert_close(terms.sum(-1)[ok], logits[ok], atol=1e-4, rtol=0)
    learned = terms[..., ORBITAL_TERMS.index("orbital_pairs")]
    assert learned[ok].abs().max() > 1e-3 and terms[..., 2][ok].abs().max() > 1e-4
    a = batch["bond_order"].shape[1]
    bare = dict(batch) | {"incidence": batch["incidence"].clone()}
    bare["incidence"][:, a:, 0] = 0.0  # no orbital has a weight on atom 0
    with torch.no_grad():
        none = model.decoder.orbital.pooled(model.decoder.orbital.kernel(p), p, bare)
        kept = model.decoder.orbital.pooled(model.decoder.orbital.kernel(p), p, batch)
    assert (none[:, 0, 1:] < -60).all() and (kept[:, 0, 1:] > -30).any()  # TINY: out of reach
    torch.testing.assert_close(none[:, 1:, 1:], kept[:, 1:, 1:], atol=1e-5, rtol=0)
    plain = local(samples, anchor_charge_term=False)
    assert plain.decoder.orbital.charge is None
    with torch.no_grad():
        assert not anchor_terms(plain, *_logits(plain, batch)[:3], batch)[..., 2].any()


def test_the_head_trains_and_the_loss_reads_it(samples):  # noqa: F811
    batch = collate_samples(samples)
    model = make(samples, **LOCAL).train()
    h, p, y, _ = model.encode(batch)
    loss, parts = step_loss(model.decoder, h, p, y, batch)
    loss.backward()
    head = model.decoder.orbital
    params = (head.score[-1].weight, head.kernel_scale, head.kernel_bias, head.off_level)
    params += (head.donor.weight, head.acceptor.weight)
    for param in (*params, head.charge):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert model.encoder.layers[-1].u_out.weight.grad.abs().sum() > 0  # the entry heads' path
    with torch.no_grad():
        r_pairs, r_atoms = greedy_edit(model.decoder, h, p, parts["lp_anchor"], batch)
    assert r_pairs.shape == batch["bond_order"].shape and r_atoms.shape == r_pairs.shape[:2]


def test_the_evaluation_tools_read_the_local_model(samples):  # noqa: F811
    model = local(samples)
    batch = collate_samples(samples)
    inp = model.prepare(batch)[0]
    with torch.no_grad():
        parts = embedding_parts(model.encoder.embed, batch, inp)
        total = model.encoder.embed(batch, inp)
    torch.testing.assert_close(sum(parts.values()), total, atol=1e-5, rtol=0)
    assert not parts["charge"].any() and not parts["fragment"].any()
    tr = trace(model, samples[0])
    assert max(tr["errors"]["layers"]) < 1e-4
    assert tr["readout"]["term_names"] == list(ORBITAL_TERMS)
    assert set(ORBITAL_TERMS) <= set(tr["readout"]["top"][0])
    loader = make_loader(samples, TrainConfig(workers=0, cell_budget=10**7), shuffle=False)
    ranks = ood.term_ranks(model, loader, torch.device("cpu"), None)
    assert set(ranks) == {f"{pre}{t}" for t in ORBITAL_TERMS for pre in ("", "no_")}
    report, _ = ood.ood_report(model, loader, torch.device("cpu"), None)
    assert report["symmetry"]["fragment_order"]["top_changed"] == 0.0


def test_a_local_run_fits_and_loads(samples, tmp_path):  # noqa: F811
    from ksmech.train.loop import load_model

    _two_source_stores(samples, tmp_path)
    cfg = TrainConfig(source="pmechdb+reactmech", split="x", epochs=2, workers=0,
                      cell_budget=10**7, warmup=1)  # fmt: skip
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = fit(cfg, replace(TINY, **LOCAL), run_dir, torch.device("cpu"), tmp_path)
    assert best["val_n"] == 2 * len(samples) and torch.isfinite(torch.tensor(best["val_loss"]))
    header = json.loads((run_dir / "run.json").read_text())["provenance"]
    assert header["config"]["model"]["anchor_head"] == "orbital_pairs"
    assert header["config"]["model"]["attention_scope"] == "molecule"
    model, _ = load_model(run_dir / "best.pt")
    assert model.decoder.orbital is not None and model.cfg.attention_scope == "molecule"
