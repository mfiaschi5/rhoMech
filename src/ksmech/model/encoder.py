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
from ksmech.model.inputs import CHARGE_MAX, MAX_Z, frozen, pair_channel_count, pair_channels


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


class TokenEmbedding(nn.Module):
    """h^(0) of atom tokens (E_Z + E_charge + MLP_A) and orbital tokens (MLP_O + set encoder over
    the spectral entries + the centre-weighted atom embeddings), the fragment embedding, and the
    learned MECH token prepended at position 0."""

    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int):
        super().__init__()
        d = cfg.d
        self.z = nn.Embedding(MAX_Z + 1, d)
        self.charge = nn.Embedding(2 * CHARGE_MAX + 1, d)
        self.atom = mlp(n_atom_x, d, d)
        self.orbital = mlp(n_orbital_x, d, d)
        self.spectral = mlp(2, d, d)
        self.fragment = nn.Embedding(cfg.fragments, d)
        self.mech = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.n_fragments = cfg.fragments

    def forward(self, batch: dict, inp: dict) -> torch.Tensor:
        cat = batch["atom_categorical"]
        n_atoms = cat.shape[1]
        charge = cat[..., 1].clamp(-CHARGE_MAX, CHARGE_MAX) + CHARGE_MAX
        h_atom = self.z(cat[..., 0].clamp(0, MAX_Z)) + self.charge(charge)
        h_atom = (h_atom + self.atom(inp["atom_x"])) * batch["is_atom"][:, :n_atoms, None]
        spectral = self.spectral(inp["spectra"]) * inp["spectra_mask"][..., None]
        inherited = batch["incidence"][:, n_atoms:, :].to(h_atom.dtype) @ h_atom
        h_orb = self.orbital(inp["orbital_x"]) + spectral.sum(-2) + inherited
        h = torch.cat([h_atom, h_orb], 1)
        h = h + self.fragment(batch["fragment"].clamp(min=0) % self.n_fragments)
        h = h * batch["token_mask"][..., None]
        return torch.cat([self.mech.expand(h.shape[0], -1, -1).to(h.dtype), h], 1)


class EncoderLayer(nn.Module):
    """Pre-norm attention with the pair bias w_h^T p_ij and the padding mask, a GELU
    feed-forward block of width 4d, then p_ij <- p_ij + MLP_U([p_ij, h_i, h_j]) (its first
    linear layer split over the three inputs, which is the same map)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, dp, self.heads = cfg.d, cfg.d_pair, cfg.heads
        self.norm_h, self.norm_p, self.norm_f = nn.LayerNorm(d), nn.LayerNorm(dp), nn.LayerNorm(d)
        self.norm_u = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.bias = nn.Linear(dp, cfg.heads)
        self.out = nn.Linear(d, d)
        self.ffn = mlp(d, 4 * d, d)
        self.u_p, self.u_i, self.u_j = nn.Linear(dp, dp), nn.Linear(d, dp), nn.Linear(d, dp)
        self.u_out = nn.Linear(dp, dp)

    def forward(self, h: torch.Tensor, p: torch.Tensor, key_mask: torch.Tensor):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm_h(h)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        pn = self.norm_p(p)
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        logits = logits.float() + self.bias(pn).permute(0, 3, 1, 2).float()
        logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, -1).to(v.dtype)
        h = h + self.out((attn @ v).transpose(1, 2).reshape(b, n, d))
        h = h + self.ffn(self.norm_f(h))
        hn = self.norm_u(h)
        u = self.u_p(pn) + self.u_i(hn)[:, :, None, :] + self.u_j(hn)[:, None, :, :]
        return h, p + self.u_out(nn.functional.gelu(u))


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
    the pair update in every layer. The decoder reads the symmetrized final pair vectors of
    the atom pairs."""

    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg, n_atom_x, n_orbital_x)
        channels = pair_channel_count(cfg.rbf, n_features)
        self.pair_init = mlp(channels, cfg.d_pair, cfg.d_pair)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.layers))

    def forward(self, batch: dict, inp: dict, z_hat, defined, dense_warmup: bool = False):
        """(h, p_atoms, aux): token vectors (B, N + 1, d), MECH at 0; symmetrized atom-pair
        vectors (B, A, A, d_p); no auxiliary loss."""
        run = Recompute(self.cfg.checkpoint and self.training)
        with frozen(z_hat.device.type):
            channels = pair_channels(batch, z_hat, defined, self.cfg.rbf, self.cfg.rbf_max)
        h = self.embed(batch, inp)
        p = run(self.pair_init, channels)
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
