"""The model (spec v9) on samples of the mini PMechDB steps, for the three encoders (dense,
flex, sparse): input preparation, invariances, the decoder's masks, the step loss against an
explicit sum and a case computed by hand, the sparse indexer, the cell-budget batches, the
physics floor and one training step."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.arrows.delta_be import ReactionMatrix
from ksmech.arrows.steps import MechanismSample, pair_stats, sample_for_step
from ksmech.eval.floor import floor_ranks
from ksmech.model.config import ModelConfig
from ksmech.model.decoder import BREAK, FORM
from ksmech.model.inputs import (
    fit_feature_stats,
    pair_channels,
    pair_channels_at,
    pair_context,
    pooled_gap,
    standardized_pair_z,
)
from ksmech.model.loss import (
    admissible_rows,
    anchor_log_probs,
    anchor_ranks,
    edit_matches,
    greedy_edit,
    step_loss,
)
from ksmech.model.net import KSMech
from ksmech.orbitals.tokens import atom_feature_names, feature_names
from ksmech.train.batching import CellBudgetSampler, cells
from ksmech.train.loop import (
    TrainConfig,
    checkpoint_payload,
    evaluate,
    load_model,
    stop_weight,
    train_epoch,
)

TINY = ModelConfig(
    d=32, d_pair=16, layers=2, heads=4, decoder_hidden=16, physics_hidden=8, checkpoint=False
)
ENCODERS = ("dense", "flex", "sparse")
ATOM_FIELDS = (
    "atom_categorical",
    "atom_features",
    "atom_missing",
    "atom_fragment",
    "atom_map",
    "atom_positions",
    "lone_pairs",
)
ORBITAL_FIELDS = (
    "orbital_features",
    "orbital_spectra",
    "orbital_fragment",
    "orbital_occupied",
    "donor_mask",
    "acceptor_mask",
    "orbital_centroids",
)
BLOCK_FIELDS = ("pair_z", "pair_defined", "pair_gap_sign", "pair_valid")


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


@pytest.fixture(scope="module")
def model(samples) -> KSMech:
    return make(samples)


def _real(b: dict) -> torch.Tensor:
    """(B, N + 1) real tokens on the MECH layout."""
    return torch.cat([torch.ones_like(b["token_mask"][:, :1]), b["token_mask"]], 1)


def test_model_inputs_are_descriptors_only(samples, model):
    """No categorical orbital type, hybridization, aromaticity or group id is a model input:
    the token inputs are the continuous descriptor lists (and the missing flags)."""
    inp = model.prepare(collate_samples(samples))[0]
    assert inp["atom_x"].shape[-1] == 2 * len(atom_feature_names())
    assert inp["orbital_x"].shape[-1] == len(feature_names())
    taxonomy = ("type", "kind", "hybrid", "aromatic", "group", "fg_")
    assert not [n for n in feature_names() + atom_feature_names() if n.startswith(taxonomy)]


def test_pooled_gap_is_the_weighted_sum_over_valid_pairs(samples):
    b = collate_samples(samples)
    g_atom = pooled_gap(b).numpy()
    g_col = samples[0].pair_features.index("g")
    for k, s in enumerate(samples):
        w = np.zeros((s.n_orbitals, s.n_atoms))
        rows = s.incidence
        w[rows[:, 0].astype(int), rows[:, 1].astype(int)] = rows[:, 2]
        occ, virt = np.flatnonzero(s.orbital_occupied), np.flatnonzero(~s.orbital_occupied)
        expect = np.zeros((s.n_atoms, s.n_atoms))
        for r, o in enumerate(occ):
            for c, v in enumerate(virt):
                if s.pair_valid[r, c]:
                    expect += s.pair_z[r, c, g_col] * (np.outer(w[o], w[v]) + np.outer(w[v], w[o]))
        np.testing.assert_allclose(g_atom[k, : s.n_atoms, : s.n_atoms], expect, atol=1e-4)


def _channels(samples, model):
    b = collate_samples(samples)
    z_hat, defined = standardized_pair_z(b, model.pair_stats, (), TINY.K)
    return b, z_hat, defined, pair_channels(b, z_hat, defined, TINY.rbf, TINY.rbf_max)


def test_pair_channels_put_z_on_occupied_virtual_cells(samples, model):
    b, z_hat, defined, ch = _channels(samples, model)
    f = z_hat.shape[-1]
    zc = slice(TINY.rbf, TINY.rbf + f)
    orient = TINY.rbf + 2 * f
    i, j = int(b["pair_row_pos"][0, 0]) + 1, int(b["pair_col_pos"][0, 0]) + 1
    torch.testing.assert_close(ch[0, i, j, zc], z_hat[0, 0, 0])
    torch.testing.assert_close(ch[0, j, i, zc], z_hat[0, 0, 0])
    assert ch[0, i, j, orient] == 1 and ch[0, j, i, orient] == -1
    assert (ch[0, 1, 1:, orient] == 0).all()  # an atom row carries no orbital-pair features
    assert (ch[:, 0, :, -1] == 1).all() and (ch[:, 1:, 1:, -1] == 0).all()  # MECH kind
    across = ~b["same_fragment"]
    assert (ch[:, 1:, 1:, : TINY.rbf][across] == 0).all()  # no distance between molecules


def test_pair_channels_at_equal_the_dense_grid(samples, model):
    b, z_hat, defined, ch = _channels(samples, model)
    ctx = pair_context(b, z_hat, defined)
    grid = torch.meshgrid(*(torch.arange(n) for n in ch.shape[:3]), indexing="ij")
    bi, i, j = (x.flatten() for x in grid)
    at = pair_channels_at(ctx, b, bi, i, j, TINY.rbf, TINY.rbf_max)
    torch.testing.assert_close(at, ch.reshape(-1, ch.shape[-1]), atol=1e-5, rtol=1e-5)


def _remap_label(label, ia: np.ndarray):
    def pair(i, j):
        return tuple(sorted((int(ia[i]), int(ia[j]))))

    def matrix(v: ReactionMatrix) -> ReactionMatrix:
        return ReactionMatrix(
            tuple((*pair(i, j), d) for i, j, d in v.bonds),
            tuple((int(ia[i]), d) for i, d in v.lone),
        )

    anchors = tuple(tuple((*pair(i, j), kind) for i, j, kind in a) for a in label.anchors)
    return replace(label, variants=tuple(map(matrix, label.variants)), anchors=anchors)


def _permuted(s: MechanismSample, rng):
    """`s` with its atoms and orbitals listed in random orders: (sample, atom order, orbital
    order), new position k holding old atom (orbital) order[k]."""
    pa, po = rng.permutation(s.n_atoms), rng.permutation(s.n_orbitals)
    ia, io = np.argsort(pa), np.argsort(po)
    row_of = {o: r for r, o in enumerate(np.flatnonzero(s.orbital_occupied))}
    col_of = {o: c for c, o in enumerate(np.flatnonzero(~s.orbital_occupied))}
    rows = [row_of[o] for o in po if o in row_of]
    cols = [col_of[o] for o in po if o in col_of]
    bonds = s.bonds.copy()
    bonds[:, :2] = np.sort(ia[s.bonds[:, :2]], axis=1)
    inc = s.incidence.copy()
    inc[:, 0], inc[:, 1] = io[s.incidence[:, 0].astype(int)], ia[s.incidence[:, 1].astype(int)]
    fields = {k: getattr(s, k)[pa] for k in ATOM_FIELDS}
    fields |= {k: getattr(s, k)[po] for k in ORBITAL_FIELDS}
    fields |= {k: getattr(s, k)[rows][:, cols] for k in BLOCK_FIELDS}
    label = _remap_label(s.edit_label, ia)
    return replace(s, **fields, bonds=bonds, incidence=inc, edit_label=label, labels=[]), pa, po


@pytest.mark.parametrize("encoder", ENCODERS)
def test_permuting_tokens_permutes_the_outputs(samples, encoder):
    """Equivariance over atoms and orbitals: token vectors and atom-pair vectors follow the
    permutation, the loss does not change. The sparse encoder is checked with k above N: with
    fewer keys, tied index scores are broken by position."""
    model = make(samples, encoder, topk=10_000)
    rng = np.random.default_rng(0)
    base = [replace(s, labels=[]) for s in samples[:2]]
    permuted = [_permuted(s, rng) for s in base]
    moved, orders = [m for m, _, _ in permuted], [(pa, po) for _, pa, po in permuted]
    with torch.no_grad():
        b0, b1 = collate_samples(base), collate_samples(list(moved))
        h0, p0, y0, _ = model.encode(b0)
        h1, p1, y1, _ = model.encode(b1)
        l0 = step_loss(model.decoder, h0, p0, y0, b0)[1]["loss"]
        l1 = step_loss(model.decoder, h1, p1, y1, b1)[1]["loss"]
    n_atoms = b0["bond_order"].shape[1]
    tol = {"atol": 1e-4, "rtol": 1e-4}
    for k, (pa, po) in enumerate(orders):
        torch.testing.assert_close(h1[k, 0], h0[k, 0], **tol)  # MECH
        torch.testing.assert_close(h1[k, 1 : len(pa) + 1], h0[k, 1 + pa], **tol)
        o1 = slice(1 + n_atoms, 1 + n_atoms + len(po))
        torch.testing.assert_close(h1[k, o1], h0[k, 1 + n_atoms + po], **tol)
        idx = torch.as_tensor(pa)
        torch.testing.assert_close(p1[k, : len(pa), : len(pa)], p0[k][idx][:, idx], **tol)
    torch.testing.assert_close(l1, l0, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_rotation_and_translation_leave_the_outputs_unchanged(samples, encoder):
    model = make(samples, encoder)
    b = collate_samples(samples)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))
    moved = dict(b)
    for key in ("atom_positions", "orbital_centroids"):
        moved[key] = b[key] @ q.T + torch.tensor([3.0, -2.0, 7.0])
    with torch.no_grad():
        h, p, _, _ = model.encode(b)
        h2, p2, _, _ = model.encode(moved)
    torch.testing.assert_close(h, h2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(p, p2, atol=1e-4, rtol=1e-4)


def test_flex_kernel_equals_its_dense_reference(samples):
    """FlexAttention (GPU only: no backward on the CPU) against the same bias through SDPA."""
    from ksmech.train.loop import rocm_override

    rocm_override()
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    model = make(samples, "flex", d=64).cuda()  # FlexAttention needs heads of width >= 16
    batch = collate_samples(samples)
    b = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    real = _real(b)
    outputs, grads = [], []
    for kernel in (True, False):
        model.encoder.cfg = replace(model.cfg, flex_kernel=kernel)
        model.zero_grad()
        h, p = model.encode(b)[:2]
        (h[real].square().sum() + p.square().sum()).backward()
        outputs.append((h[real].detach(), p.detach()))
        grads.append({n: w.grad.clone() for n, w in model.encoder.named_parameters()})
    for got, want in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-4)
    for name, g in grads[1].items():  # the bias tables and z included
        torch.testing.assert_close(grads[0][name], g, atol=1e-3, rtol=1e-3, msg=name)


def test_sparse_with_every_key_equals_its_dense_path(samples):
    model = make(samples, "sparse", topk=10_000)
    b = collate_samples(samples)
    with torch.no_grad():
        h_sparse, p_sparse, _, aux_s = model.encode(b, dense_warmup=False)
        h_dense, p_dense, _, aux_d = model.encode(b, dense_warmup=True)
    real = _real(b)
    torch.testing.assert_close(h_sparse[real], h_dense[real], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(p_sparse, p_dense, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(aux_s["indexer_kl"], aux_d["indexer_kl"], atol=1e-4, rtol=1e-4)


def test_sparse_selects_mech_self_and_top_k():
    from ksmech.model.sparse import select_keys

    scores = torch.randn(1, 6, 6, generator=torch.Generator().manual_seed(0))
    keys = torch.tensor([[True, True, True, True, True, False]])
    idx, ok = select_keys(scores, keys, 2)
    assert idx.shape == (1, 6, 4) and (idx[..., 0] == 0).all()
    assert (idx[0, :, 1] == torch.arange(6)).all()
    assert not ok[0, 0, 1]  # MECH's own query lists MECH once
    assert not ok[0, 5, 1]  # a padding query's self is no key
    for q in range(6):
        chosen = idx[0, q, 2:][ok[0, q, 2:]].tolist()
        others = [j for j in range(1, 5) if j != q]
        best = sorted(others, key=lambda j: -float(scores[0, q, j]))[:2]
        assert sorted(chosen) == sorted(best)


def test_indexer_kl_trains_the_indexer_only(samples):
    model = make(samples, "sparse", topk=8).train()
    _, _, _, aux = model.encode(collate_samples(samples))
    aux["indexer_kl"].backward()
    for name, param in model.named_parameters():
        touched = param.grad is not None and bool(param.grad.abs().sum() > 0)
        assert touched == (".indexer." in name), name


def test_decoder_masks(samples, model):
    b = collate_samples(samples)
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        logits = model.decoder.anchor_logits(h, p, y, b)
        n = logits.shape[1]
        lower = ~torch.ones(n, n, dtype=torch.bool).triu(1)
        assert torch.isinf(logits[:, lower]).all()
        assert torch.isinf(logits[..., FORM][~b["can_form"]]).all()
        assert torch.isinf(logits[..., BREAK][~b["can_break"]]).all()
        rows = admissible_rows(b)[:, [0, 2, 3, 4]]
        pair, atom = model.decoder.entry_logits(h, p, b, rows)
    order = b["bond_order"][rows[:, 0]]
    assert torch.isinf(pair[..., 0][order == 0]).all()
    assert torch.isfinite(pair[..., 0][order > 0]).all()
    assert torch.isinf(pair[..., 2][order == 3]).all()
    assert torch.isinf(atom[..., 0][b["lone_pairs"][rows[:, 0]] == 0]).all()


def _explicit_loss(model, b, k: int) -> float:
    """-log sum over sample k's (variant, anchor) rows, every entry summed in a Python loop."""
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        lp_anchor = anchor_log_probs(model.decoder.anchor_logits(h, p, y, b))
        rows = admissible_rows(b)
        rows = rows[rows[:, 0] == k]
        pair, atom = model.decoder.entry_logits(h, p, b, rows[:, [0, 2, 3, 4]])
    pair, atom = torch.log_softmax(pair, -1), torch.log_softmax(atom, -1)
    n = int(b["is_atom"][k].sum())
    terms = []
    for x, (_, v, a, c, sign) in enumerate(rows.tolist()):
        t = float(lp_anchor[k, a, c, sign])
        for i in range(n):
            t += float(atom[x, i, int(b["r_atoms"][k, v, i]) // 2 + 1])
            for j in range(i + 1, n):
                if (i, j) != (a, c):
                    t += float(pair[x, i, j, int(b["r_pairs"][k, v, i, j]) + 1])
        terms.append(t)
    top = max(terms)
    return -(top + math.log(sum(math.exp(t - top) for t in terms)))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_step_loss_is_the_marginal_over_class_and_anchors(samples, encoder):
    model = make(samples, encoder)
    b = collate_samples(samples)
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        mean, parts = step_loss(model.decoder, h, p, y, b)
    for k in range(len(samples)):
        assert parts["loss"][k].item() == pytest.approx(_explicit_loss(model, b, k), rel=1e-4)
    assert mean.item() == pytest.approx(parts["loss"].mean().item())
    assert (parts["anchor_nll"] <= parts["loss"] + 1e-5).all()


class _StubDecoder:
    """Fixed logits on three atoms: anchors (0,1) form, (1,2) form, (0,2) break with
    probabilities 1/5, 3/5, 1/5; given (0,1) every entry class has probability 1/3, given
    (1,2) the zero class has 1/2 and the others 1/4; P(END_MECH) is 3/4."""

    def stop_logits(self, h):
        return torch.tensor([math.log(3)])

    def anchor_logits(self, h, p, y, batch):
        out = torch.full((1, 3, 3, 2), float("-inf"))
        out[0, 0, 1, FORM], out[0, 1, 2, FORM], out[0, 0, 2, BREAK] = 0.0, math.log(3), 0.0
        return out

    def entry_logits(self, h, p, batch, anchors):
        k = len(anchors)
        pair, atom = torch.zeros(k, 3, 3, 3), torch.zeros(k, 3, 3)
        late = (anchors[:, 1] == 1) & (anchors[:, 2] == 2)
        pair[late, ..., 1], atom[late, ..., 1] = math.log(2), math.log(2)
        return pair, atom


def _hand_batch(variants: list) -> dict:
    """variants: (pair entries {(i, j): d}, atom entries {i: d}, anchors [(i, j, sign)])."""
    v = len(variants)
    b = {
        "r_pairs": torch.zeros(1, v, 3, 3, dtype=torch.int8),
        "r_atoms": torch.zeros(1, v, 3, dtype=torch.int8),
        "r_anchors": torch.zeros(1, v, 3, 3, 2, dtype=torch.bool),
        "variant_mask": torch.ones(1, v, dtype=torch.bool),
        "is_atom": torch.ones(1, 3, dtype=torch.bool),
        "bond_order": torch.zeros(1, 3, 3, dtype=torch.int8),
    }
    for x, (pairs, atoms, anchors) in enumerate(variants):
        for (i, j), d in pairs.items():
            b["r_pairs"][0, x, i, j] = b["r_pairs"][0, x, j, i] = d
        for i, d in atoms.items():
            b["r_atoms"][0, x, i] = d
        for i, j, sign in anchors:
            b["r_anchors"][0, x, i, j, sign] = b["r_anchors"][0, x, j, i, sign] = True
    return b


def test_step_loss_by_hand():
    third, half, quarter = math.log(1 / 3), math.log(1 / 2), math.log(1 / 4)
    one = ({(0, 1): 1}, {}, [(0, 1, FORM)])  # anchor (0,1); 2 pairs + 3 atoms at 1/3
    t1 = math.log(1 / 5) + 5 * third
    two = ({(1, 2): 1}, {0: -2}, [(1, 2, FORM)])  # anchor (1,2); atom 0 at 1/4, rest 1/2
    t2 = math.log(3 / 5) + 2 * half + quarter + 2 * half
    both = ({(0, 1): 1, (1, 2): 1}, {}, [(0, 1, FORM), (1, 2, FORM)])  # two anchors
    t3a = math.log(1 / 5) + 5 * third
    t3b = math.log(3 / 5) + quarter + half + 3 * half
    cases = [
        ([one], -t1),
        ([one, two], -math.log(math.exp(t1) + math.exp(t2))),
        ([both], -math.log(math.exp(t3a) + math.exp(t3b))),
    ]
    for variants, expect in cases:
        loss, _ = step_loss(_StubDecoder(), None, None, None, _hand_batch(variants))
        assert loss.item() == pytest.approx(expect, rel=1e-6)
    for label, bce in ((1, -math.log(3 / 4)), (0, -math.log(1 / 4))):
        batch = _hand_batch([one]) | {"stop_label": torch.tensor([label])}
        loss, parts = step_loss(_StubDecoder(), None, None, None, batch, 0.5)
        assert loss.item() == pytest.approx(-t1 + 0.5 * bce, rel=1e-6)
        assert parts["p_stop"].item() == pytest.approx(3 / 4)


def test_anchor_nll_counts_an_anchor_shared_by_variants_once():
    """Two variants with the same anchor (0, 1): the anchor NLL is -log 1/5, not -log 2/5,
    while the loss sums the two paths (every entry at 1/3 given that anchor)."""
    first = ({(0, 1): 1}, {}, [(0, 1, FORM)])
    second = ({(0, 1): 1}, {0: -2}, [(0, 1, FORM)])
    loss, parts = step_loss(_StubDecoder(), None, None, None, _hand_batch([first, second]))
    assert parts["anchor_nll"].item() == pytest.approx(-math.log(1 / 5), rel=1e-6)
    path = math.log(1 / 5) + 5 * math.log(1 / 3)
    assert loss.item() == pytest.approx(-math.log(2 * math.exp(path)), rel=1e-6)


def test_stop_head_gives_one_probability_per_sample(samples, model):
    b = collate_samples(samples)
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        _, parts = step_loss(model.decoder, h, p, y, b)
    assert model.decoder.stop_logits(h).shape == (len(samples),)
    assert parts["p_stop"].shape == (len(samples),)
    assert ((parts["p_stop"] > 0) & (parts["p_stop"] < 1)).all()


def test_stop_loss_is_masked_per_sample(samples, model):
    """Sample 0 ends the mechanism, sample 1 goes on, the others have no stop label."""
    b = collate_samples(samples)
    labels = torch.full((len(samples),), -1)
    labels[0], labels[1] = 1, 0
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        _, base = step_loss(model.decoder, h, p, y, b)
        mean, parts = step_loss(model.decoder, h, p, y, b | {"stop_label": labels}, 0.5)
    q = base["p_stop"]
    bce = torch.zeros_like(q)
    bce[0], bce[1] = -torch.log(q[0]), -torch.log(1 - q[1])
    torch.testing.assert_close(parts["loss"], base["loss"] + 0.5 * bce)
    assert mean.item() == pytest.approx((base["loss"] + 0.5 * bce).mean().item(), rel=1e-6)


def test_without_stop_labels_the_loss_is_the_step_loss(samples, model):
    b = collate_samples(samples)
    unknown, ends = torch.full((len(samples),), -1), torch.ones(len(samples))
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        base, _ = step_loss(model.decoder, h, p, y, b)
        for batch, weight in (
            (b, 0.5),
            (b | {"stop_label": unknown}, 0.5),
            (b | {"stop_label": ends}, 0.0),
        ):
            assert torch.equal(step_loss(model.decoder, h, p, y, batch, weight)[0], base)
    assert stop_weight(TrainConfig()) == 0.0  # PMechDB: isolated steps
    assert stop_weight(TrainConfig(source="flower")) == 0.5
    assert stop_weight(TrainConfig(stop_weight=0.2)) == 0.2


def test_ranks_and_greedy_edit(samples, model):
    b = collate_samples(samples)
    with torch.no_grad():
        h, p, y, _ = model.encode(b)
        lp = anchor_log_probs(model.decoder.anchor_logits(h, p, y, b))
        ranks = anchor_ranks(lp, b)
        r_pairs, r_atoms = greedy_edit(model.decoder, h, p, lp, b)
    assert (ranks >= 0).all()
    assert r_pairs.equal(r_pairs.transpose(1, 2))
    # the label itself matches; a changed entry does not
    truth = b["r_pairs"][:, 0].long(), b["r_atoms"][:, 0].long()
    assert edit_matches(*truth, b).all()
    wrong = truth[0].clone()
    wrong[:, 0, 1] = wrong[:, 1, 0] = 2
    assert not edit_matches(wrong, truth[1], b).any()


def test_cell_budget_batches():
    sizes = np.random.default_rng(0).integers(10, 300, size=500)
    sampler = CellBudgetSampler(sizes, cell_budget=200_000, max_batch=16, seed=3)
    first = sampler.batches()
    assert sorted(i for batch in first for i in batch) == list(range(500))
    for batch in first:
        assert len(batch) <= 16
        assert len(batch) * cells(int(sizes[batch].max())) <= 200_000
    assert sampler.batches() == first
    sampler.set_epoch(1)
    assert sampler.batches() != first
    with pytest.raises(ValueError, match="over the budget"):
        CellBudgetSampler([1000], cell_budget=10_000)


def test_each_epoch_draws_every_sample_in_one_of_its_drawings():
    sizes = np.random.default_rng(0).integers(10, 300, size=500)
    drawings = np.random.default_rng(1).integers(1, 4, size=500)
    sampler = CellBudgetSampler(sizes, 200_000, max_batch=16, seed=3, drawings=drawings)
    plain = CellBudgetSampler(sizes, 200_000, max_batch=16, seed=3)
    first = sampler.batches()
    assert [[i for i, _ in b] for b in first] == plain.batches()  # the same batches, keyed
    assert all(0 <= k < drawings[i] for b in first for i, k in b)
    assert {k for b in first for i, k in b if drawings[i] == 3} == {0, 1, 2}
    assert all(k == 0 for b in first for i, k in b if drawings[i] == 1)
    assert sampler.batches() == first
    sampler.set_epoch(1)
    assert sampler.batches() != first
    unshuffled = CellBudgetSampler(sizes, 200_000, shuffle=False, drawings=drawings)
    assert all(isinstance(i, int) for b in unshuffled.batches() for i in b)  # evaluation


def test_floor_ranks_the_largest_pooled_gap_first(samples):
    b = collate_samples(samples)
    ranks, reachable = floor_ranks(b)
    assert (ranks >= 0).all()
    score = pooled_gap(b).abs()
    n = score.shape[1]
    cand = (b["can_form"] | b["can_break"]) & torch.ones(n, n, dtype=torch.bool).triu(1)
    flat = score.masked_fill(~cand, -1).flatten(1).argmax(1)
    i, j = flat // n, flat % n
    k = torch.arange(len(samples))
    b["r_anchors"][:] = False
    sign = torch.where(b["can_form"][k, i, j], FORM, BREAK)
    b["r_anchors"][k, 0, i, j, sign] = True
    ranks, reachable = floor_ranks(b)
    assert (ranks == 0).all() and reachable.all()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_a_training_step_lowers_the_loss_and_the_checkpoint_reloads(samples, tmp_path, encoder):
    from ksmech.data import provenance as prov

    model = make(samples, encoder, topk=8)
    cfg = TrainConfig(workers=0, cell_budget=10**7, log_every=1, lr=1e-2, warmup=1)
    loader = [collate_samples(samples)]
    before = evaluate(model, loader, torch.device("cpu"), None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    state = {"step": 0, "total": 20, "epoch": 0}
    for _ in range(20):
        train_epoch(model, loader, opt, sched, scaler, torch.device("cpu"), None, cfg, state)
    after = evaluate(model, loader, torch.device("cpu"), None)
    assert after["loss"] < before["loss"]
    assert 0 <= after["top1"] <= after["top3"] <= after["top5"] <= 1
    header = prov.stamp("train_run", 1, {}, {})
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint_payload(model, header), path)
    loaded, _ = load_model(path)
    b = collate_samples(samples)
    with torch.no_grad():
        torch.testing.assert_close(loaded.eval().encode(b)[0], model.eval().encode(b)[0])


def test_rocm_override_only_for_rdna2_cards_without_kernels(tmp_path, monkeypatch):
    from ksmech.train import loop

    monkeypatch.setattr(loop.torch.version, "hip", "7.2")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    for node, version in (("0", 0), ("1", 100302)):
        (tmp_path / node).mkdir()
        (tmp_path / node / "properties").write_text(
            f"cpu_cores_count 0\ngfx_target_version {version}\n"
        )
    assert loop.rocm_override(tmp_path) == "10.3.0"
    assert loop.os.environ["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"
    assert loop.rocm_override(tmp_path) is None  # already set: left alone
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION")
    (tmp_path / "1" / "properties").write_text("gfx_target_version 100300\n")  # gfx1030
    assert loop.rocm_override(tmp_path) is None
    monkeypatch.setattr(loop.torch.version, "hip", None)
    assert loop.rocm_override(tmp_path) is None
