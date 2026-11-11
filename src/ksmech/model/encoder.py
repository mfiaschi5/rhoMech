"""The two-track encoder (spec v9, "Model architecture"): token embeddings, the pair track
initialized from the geometry-free pair features, and L layers of attention whose logits carry
a learned bias from the pair track, each followed by the O(N^2) pair update. Every head is
learned; no categorical orbital type is embedded."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ksmech.model.config import ModelConfig
from ksmech.model.inputs import (
    CHARGE_MAX,
    KINDS,
    MAX_Z,
    SPIN_MAX,
    PairInitParts,
    frozen,
    pair_channel_count,
    pair_channels,
    pair_init_parts,
)


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


class TokenEmbedding(nn.Module):
    """h^(0) of atom tokens (E_Z + E_charge + MLP_A) and orbital tokens (MLP_O + set encoder over
    the spectral entries + the centre-weighted atom embeddings), the fragment embedding, and the
    learned MECH token prepended at position 0. The ablations of the orbital token
    (`ModelConfig.drop_token_features`, `spectral_set`, `inherit_atoms`) remove a term: with no
    orbital column MLP_O is a learned vector (the MLP of an empty input), without the set encoder
    or the inherited embeddings their term is absent (`orbital_terms`)."""

    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int):
        super().__init__()
        d = cfg.d
        self.z = nn.Embedding(MAX_Z + 1, d)
        self.charge = nn.Embedding(2 * CHARGE_MAX + 1, d)
        self.atom = mlp(n_atom_x, d, d)
        self.orbital = mlp(n_orbital_x, d, d) if n_orbital_x else None
        if self.orbital is None:
            self.orbital_base = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.spectral = mlp(2, d, d) if getattr(cfg, "spectral_set", True) else None
        self.inherit = bool(getattr(cfg, "inherit_atoms", True))
        self.fragment = nn.Embedding(cfg.fragments, d)
        if getattr(cfg, "atom_spin_input", "none") == "categorical_u":  # P7, P9's ablation arm
            self.spin = nn.Embedding(SPIN_MAX + 1, d)
        self.mech = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.n_fragments = cfg.fragments

    def forward(self, batch: dict, inp: dict) -> torch.Tensor:
        cat = batch["atom_categorical"]
        n_atoms = cat.shape[1]
        charge = cat[..., 1].clamp(-CHARGE_MAX, CHARGE_MAX) + CHARGE_MAX
        h_atom = self.z(cat[..., 0].clamp(0, MAX_Z)) + self.charge(charge)
        if hasattr(self, "spin"):
            h_atom = h_atom + self.spin(cat[..., 2].clamp(0, SPIN_MAX))
        h_atom = (h_atom + self.atom(inp["atom_x"])) * batch["is_atom"][:, :n_atoms, None]
        h_orb = sum(self.orbital_terms(batch, inp, h_atom).values())
        h = torch.cat([h_atom, h_orb], 1)
        h = h + self.fragment(batch["fragment"].clamp(min=0) % self.n_fragments)
        h = h * batch["token_mask"][..., None]
        return torch.cat([self.mech.expand(h.shape[0], -1, -1).to(h.dtype), h], 1)

    def orbital_terms(self, batch: dict, inp: dict, h_atom: torch.Tensor) -> dict:
        """The additive terms of the orbital tokens' h^(0), each (B, O, d): `orbital_inputs`
        (MLP_O), `spectral` (the set encoder) and `inherited` (the centre-weighted atom
        embeddings), zero where an ablation removes them."""
        n_atoms = h_atom.shape[1]
        incidence = batch["incidence"][:, n_atoms:, :].to(h_atom.dtype)
        zero = torch.zeros(*incidence.shape[:2], h_atom.shape[-1], dtype=h_atom.dtype,
                           device=h_atom.device)  # fmt: skip
        if self.orbital is not None:
            inputs = self.orbital(inp["orbital_x"])
        else:
            inputs = zero + self.orbital_base.to(h_atom.dtype)
        spectral = zero
        if self.spectral is not None:
            spectral = (self.spectral(inp["spectra"]) * inp["spectra_mask"][..., None]).sum(-2)
        inherited = incidence @ h_atom if self.inherit else zero
        return {"orbital_inputs": inputs, "spectral": spectral, "inherited": inherited}


class EncoderLayer(nn.Module):
    """Pre-norm attention with the pair bias w_h^T p_ij and the padding mask, a GELU
    feed-forward block of width 4d, then p_ij <- p_ij + MLP_U([p_ij, h_i, h_j]) (its first
    linear layer split over the three inputs, which is the same map). Without `update`
    (`ModelConfig.pair_update_every`) the layer reads p for its bias and leaves it as it is.
    `half_pair_norm` runs the LayerNorm of p in the autocast dtype (`_norm_p`)."""

    def __init__(self, cfg: ModelConfig, update: bool = True):
        super().__init__()
        d, dp, self.heads = cfg.d, cfg.d_pair, cfg.heads
        self.half_norm, self.update = bool(getattr(cfg, "half_pair_norm", False)), update
        self.norm_h, self.norm_p, self.norm_f = nn.LayerNorm(d), nn.LayerNorm(dp), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.bias = nn.Linear(dp, cfg.heads)
        self.out = nn.Linear(d, d)
        self.ffn = mlp(d, 4 * d, d)
        if update:
            self.norm_u = nn.LayerNorm(d)
            self.u_p, self.u_i, self.u_j = nn.Linear(dp, dp), nn.Linear(d, dp), nn.Linear(d, dp)
            self.u_out = nn.Linear(dp, dp)

    def _norm_p(self, p: torch.Tensor) -> torch.Tensor:
        """LayerNorm of the pair track. Autocast runs layer_norm in float32 and returns float32,
        so the largest tensor of the model is cast up and back down in every layer; with
        `half_norm` it is normalized in the autocast dtype instead (the kernel accumulates in
        float32 all the same). Outside autocast, as before."""
        kind = p.device.type
        if not (self.half_norm and torch.is_autocast_enabled(kind)):
            return self.norm_p(p)
        dtype = torch.get_autocast_dtype(kind)
        norm = self.norm_p
        with torch.autocast(kind, enabled=False):
            weight, bias = norm.weight.to(dtype), norm.bias.to(dtype)
            return nn.functional.layer_norm(
                p.to(dtype), norm.normalized_shape, weight, bias, norm.eps
            )

    def forward(self, h: torch.Tensor, p: torch.Tensor, key_mask: torch.Tensor):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm_h(h)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        pn = self._norm_p(p)
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        logits = logits.float() + self.bias(pn).permute(0, 3, 1, 2).float()
        logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, -1).to(v.dtype)
        h = h + self.out((attn @ v).transpose(1, 2).reshape(b, n, d))
        h = h + self.ffn(self.norm_f(h))
        if not self.update:
            return h, p
        hn = self.norm_u(h)
        u = self.u_p(pn) + self.u_i(hn)[:, :, None, :] + self.u_j(hn)[:, None, :, :]
        return h, p + self.u_out(nn.functional.gelu(u))


def split_pair_init(lin: nn.Linear, parts: PairInitParts, n_rbf: int) -> torch.Tensor:
    """`lin(pair_channels(...))` from `inputs.pair_init_parts`: each part times its columns of
    the weight, summed on the cells where the part is non-zero (`ModelConfig.split_pair_init`).
    The same linear map; only the summation order, hence the rounding, differs."""
    w = lin.weight
    f2 = parts.z.shape[-1]
    c = n_rbf + f2 + 1  # same fragment, then bonded, order, distance, centre weight, kinds
    inner = nn.functional.linear(parts.basis, w[:, :n_rbf])  # (B, N, N, d_p), autocast dtype
    dt = inner.dtype
    kinds = torch.cat([w[:, c + 5 : c + 5 + KINDS].T, torch.zeros_like(w[:, :1].T)])  # (6, d_p)
    table = (kinds[:, None, :] + torch.stack([0 * w[:, c], w[:, c]])[None]).reshape(-1, w.shape[0])
    pre = (table.to(dt)[parts.kind] + lin.bias.to(dt)).contiguous()
    pre[:, 1:, 1:] += inner
    a = parts.n_atoms
    pre[:, 1 : a + 1, 1 : a + 1] += nn.functional.linear(parts.atom, w[:, c + 1 : c + 4]).to(dt)
    w_c = w[:, c + 4].to(dt)
    pre[:, a + 1 :, 1 : a + 1] += parts.centre.to(dt)[..., None] * w_c
    pre[:, 1 : a + 1, a + 1 :] += parts.centre.transpose(1, 2).to(dt)[..., None] * w_c
    zc = nn.functional.linear(parts.z, w[:, n_rbf : n_rbf + f2]).to(dt)
    orient = w[:, n_rbf + f2].to(dt)
    b, i, j = parts.cell
    pre.index_put_((b, i, j), zc + orient, accumulate=True)
    pre.index_put_((b, j, i), zc - orient, accumulate=True)
    return pre


def key_mask(batch: dict) -> torch.Tensor:
    """(B, N + 1) real keys on the MECH layout: MECH and the batch's real tokens."""
    mech = torch.ones_like(batch["token_mask"][:, :1])
    return torch.cat([mech, batch["token_mask"]], 1)


class Recompute:
    """Run a block directly, or recompute it in the backward pass (activation checkpointing)
    when `on`."""

    def __init__(self, on: bool):
        self.on = on

    def __call__(self, fn, *args):
        return checkpoint(fn, *args, use_reentrant=False) if self.on else fn(*args)


class DenseEncoder(nn.Module):
    """Spec v9: the pair track (B, N + 1, N + 1, d_p) from MLP_P, a pair bias in every head,
    the pair update in every layer (every `pair_update_every`-th with that ablation). The
    decoder reads the symmetrized final pair vectors of the atom pairs."""

    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg, n_atom_x, n_orbital_x)
        channels = pair_channel_count(cfg.rbf, n_features)
        self.pair_init = mlp(channels, cfg.d_pair, cfg.d_pair)
        every = int(getattr(cfg, "pair_update_every", 1))
        if every < 1:
            raise ValueError(f"pair_update_every {every}: at least 1")
        self.layers = nn.ModuleList(
            EncoderLayer(cfg, update=(k + 1) % every == 0) for k in range(cfg.layers)
        )

    def _init_split(self, parts: PairInitParts) -> torch.Tensor:
        pre = split_pair_init(self.pair_init[0], parts, self.cfg.rbf)
        return self.pair_init[2](self.pair_init[1](pre))

    def forward(self, batch: dict, inp: dict, z_hat, defined, dense_warmup: bool = False):
        """(h, p_atoms, aux): token vectors (B, N + 1, d), MECH at 0; symmetrized atom-pair
        vectors (B, A, A, d_p); no auxiliary loss."""
        run = Recompute(self.cfg.checkpoint and self.training)
        split = getattr(self.cfg, "split_pair_init", False)
        with frozen(z_hat.device.type):
            make = pair_init_parts if split else pair_channels
            channels = make(batch, z_hat, defined, self.cfg.rbf, self.cfg.rbf_max)
        h = self.embed(batch, inp)
        p = run(self._init_split, channels) if split else run(self.pair_init, channels)
        keys = key_mask(batch)
        for layer in self.layers:
            h, p = run(layer, h, p, keys)
        a = batch["bond_order"].shape[1]
        pa = p[:, 1 : a + 1, 1 : a + 1]
        return h, (pa + pa.transpose(1, 2)) / 2, {}


class AtomPairHead(nn.Module):
    """Atom-pair vectors for the decoder of an encoder without a pair track, on A x A only:
    an MLP of the static atom-pair inputs (`inputs.atom_pair_static`), h_a * h_b and
    h_a + h_b (its first layer split over the three), symmetric in (a, b)."""

    def __init__(self, cfg: ModelConfig, n_static: int):
        super().__init__()
        hid = cfg.decoder_hidden
        self.static = nn.Linear(n_static, hid)
        self.prod = nn.Linear(cfg.d, hid, bias=False)
        self.sum = nn.Linear(cfg.d, hid, bias=False)
        self.out = nn.Linear(hid, cfg.d_pair)

    def forward(self, h: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        a = static.shape[1]
        ha = h[:, 1 : a + 1]
        s = self.sum(ha)
        pre = self.static(static.to(ha.dtype)) + self.prod(ha[:, :, None] * ha[:, None])
        return self.out(nn.functional.gelu(pre + s[:, :, None] + s[:, None]))
