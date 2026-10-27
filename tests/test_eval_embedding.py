"""The embedding trace (`eval.embedding`) and its page (`eval.embedding_page`) on samples of
the mini PMechDB steps, for the three encoders: the parts of h^(0) sum to the token embedding,
every rebuilt layer equals the model's own, the anchor terms sum to the logits, the readout
agrees with the loss, and the page holds every token and the trace."""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.steps import MechanismSample, pair_stats, sample_for_step
from ksmech.eval.embedding import (
    anchor_terms,
    embedding_parts,
    ensure_pool,
    r_changes,
    sample_from_smiles,
    trace,
)
from ksmech.eval.embedding_page import (
    TYPE_SYMBOL,
    layout,
    orbital_centres,
    orbital_types,
    page_payload,
    pca_path,
    write_page,
)
from ksmech.model.config import ModelConfig
from ksmech.model.inputs import fit_feature_stats
from ksmech.model.loss import anchor_log_probs, anchor_ranks, step_loss
from ksmech.model.net import KSMech
from ksmech.physics.textbook import fragment_keys

TINY = ModelConfig(
    d=32, d_pair=16, layers=2, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)
ENCODERS = ("dense", "flex", "sparse")


@pytest.fixture(scope="module")
def samples(mini_steps, token_loader) -> list[MechanismSample]:
    out = [sample_for_step(mini_steps[k], token_loader) for k in mini_steps]
    out = [s for s in out if isinstance(s, MechanismSample)]
    assert len(out) >= 3
    return out


def make(samples, encoder: str = "dense", **kw) -> KSMech:
    torch.manual_seed(0)
    cfg = replace(TINY, encoder=encoder, **kw)
    return KSMech(cfg, fit_feature_stats(samples), pair_stats(samples)).eval()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_the_parts_of_h0_sum_to_the_token_embedding(samples, encoder):
    """On a padded batch: the sum is h^(0), each part acts only on its own tokens, padding is
    zero."""
    model = make(samples, encoder)
    batch = collate_samples(samples)
    inp = model.prepare(batch)[0]
    with torch.no_grad():
        parts = embedding_parts(model.encoder.embed, batch, inp)
        h0 = model.encoder.embed(batch, inp)
    assert torch.allclose(sum(parts.values()), h0, atol=1e-5)
    a = batch["atom_categorical"].shape[1]
    for k in ("element", "charge", "atom_inputs"):
        assert parts[k][:, a + 1 :].abs().max() == 0
    for k in ("orbital_inputs", "spectral", "inherited"):
        assert parts[k][:, 1 : a + 1].abs().max() == 0
    assert parts["mech"][:, 1:].abs().max() == 0
    pad = ~batch["token_mask"]
    assert pad.any()
    for v in parts.values():
        assert v[:, 1:][pad].abs().max() == 0


@pytest.mark.parametrize("encoder", ENCODERS)
def test_every_rebuilt_layer_equals_the_model(samples, encoder):
    """The attention rebuilt from the recorded layer inputs gives the layer's own output; the
    sparse layer attends only to MECH, itself and its top-k keys."""
    k = 4
    model = make(samples, encoder, **({"topk": k} if encoder == "sparse" else {}))
    for s in samples:
        tr = trace(model, s)
        assert tr["errors"]["embedding"] < 1e-5
        assert max(tr["errors"]["layers"]) < 1e-4
        n = s.n_atoms + s.n_orbitals + 1
        assert tr["h"].shape == (TINY.layers + 1, n, TINY.d)
        assert tr["attn"].shape == (TINY.layers, TINY.heads, n, n)
        assert np.allclose(tr["attn"].sum(-1), 1.0, atol=1e-5)
        if encoder == "sparse":
            assert ((tr["attn"] > 0).sum(-1) <= k + 2).all()


def test_the_anchor_terms_sum_to_the_logits(samples):
    model = make(samples)
    batch = collate_samples(samples)
    with torch.no_grad():
        h, p, y, _ = model.encode(batch)
        logits = model.decoder.anchor_logits(h, p, y, batch)
        terms = anchor_terms(model, h, p, y)
    ok = torch.isfinite(logits)
    assert torch.allclose(terms.sum(-1)[ok], logits[ok], atol=1e-5)


def test_the_readout_agrees_with_the_loss(samples):
    """The label's rank and probability are the loss's; the anchors come by probability; each
    anchor's probability is counted on both its atoms."""
    model = make(samples)
    for s in samples:
        r = trace(model, s)["readout"]
        batch = collate_samples([s])
        with torch.no_grad():
            h, p, y, _ = model.encode(batch)
            lp = anchor_log_probs(model.decoder.anchor_logits(h, p, y, batch))
            _, parts = step_loss(model.decoder, h, p, y, batch)
        assert r["label_rank"] == int(anchor_ranks(lp, batch)[0])
        assert r["label_probability"] == pytest.approx(math.exp(-parts["anchor_nll"][0]), rel=1e-4)
        probs = [x["p"] for x in r["top"]]
        assert probs == sorted(probs, reverse=True)
        assert sum(r["atom_probability"]) == pytest.approx(2.0, abs=1e-4)


def test_r_changes_lists_the_bonds_then_the_lone_pairs():
    r = np.zeros((3, 3), dtype=int)
    r[0, 1] = r[1, 0] = 1
    r[1, 2] = r[2, 1] = -1
    atoms = np.array([0, -2, 2])
    assert r_changes(r, atoms) == [
        ["bond", 0, 1, 1],
        ["bond", 1, 2, -1],
        ["lone", 1, -2],
        ["lone", 2, 2],
    ]


def test_orbitals_are_drawn_beside_their_centres(samples):
    for s in samples:
        centres = orbital_centres(s)
        xy = layout(s, centres)
        bonds = [np.linalg.norm(xy[i] - xy[j]) for i, j, _ in s.bonds]
        for j, c in enumerate(centres):
            near = min(np.linalg.norm(xy[s.n_atoms + j] - xy[a]) for a, _ in c)
            assert near <= 1.2 * np.median(bonds)


def test_pca_path_keeps_the_signs_of_the_layer_before():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 6)) * np.array([5.0, 3.0, 1.0, 0.5, 0.2, 0.1])
    coords, shares = pca_path(np.stack([x, -x]))
    assert np.allclose(coords[1], coords[0], atol=1e-8)
    assert 0.9 < shares[0] <= 1.0


def _payload_of(html: str) -> dict:
    return json.loads(re.search(r"const D = (\{.*?\});\nconst T = ", html, re.S).group(1))


def test_the_page_holds_every_token_and_the_trace(samples, token_loader, tmp_path):
    model = make(samples, "flex")
    s = samples[0]
    tr = trace(model, s)
    types = orbital_types(s.tokens(token_loader))
    assert len(types) == s.n_orbitals
    meta = {"sample": s.step_id, "heads": TINY.heads}
    html = write_page(page_payload(tr, s, types, meta), tmp_path / "page.html").read_text()
    assert "/*PAYLOAD*/" not in html
    data = _payload_of(html)
    n = s.n_atoms + s.n_orbitals + 1
    rows = data["tokens"]
    assert len(rows) == n and rows[0]["kind"] == "mech"
    assert all("x" in t and "y" in t for t in rows[1:])
    for row, t in zip(rows[s.n_atoms + 1 :], types, strict=True):
        assert row["name"].startswith(TYPE_SYMBOL[t])
    h = np.frombuffer(base64.b64decode(data["layers"]["h"]["data"]), dtype=np.float32)
    assert np.allclose(h.reshape(tr["h"].shape), tr["h"])
    att = np.frombuffer(base64.b64decode(data["layers"]["attention"]["data"]), dtype=np.uint8)
    assert np.allclose((att.reshape(tr["attn"].shape) / 255.0) ** 2, tr["attn"], atol=0.01)
    assert data["readout"]["label"]


def test_a_pool_from_smiles_has_the_inputs_of_its_stored_sample(
    mini_steps, mini_token_cache, samples, qm_config
):
    """A pool built from SMILES matches the stored sample's model inputs, without a label;
    a pool whose files all exist builds nothing."""
    step = next(iter(mini_steps.values()))
    assert ensure_pool(fragment_keys(step.reactants_smiles), mini_token_cache, qm_config) == []
    s = sample_from_smiles(step.reactants_smiles, mini_token_cache)
    stored = sample_for_step(step, mini_token_cache)
    assert s.edit_label is None
    for field in ("atom_categorical", "bonds", "lone_pairs", "orbital_features", "pair_z"):
        np.testing.assert_array_equal(getattr(s, field), getattr(stored, field))
    readout = trace(make(samples), s)["readout"]
    assert "label_rank" not in readout and readout["top"]
