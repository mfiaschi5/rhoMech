"""The sparse encoder: learned sparsity in the style of DeepSeek Sparse Attention (V3.2),
selecting single tokens. The tokens are a set with no order, so the contiguous blocks of NSA
or MoBA would be an arbitrary choice.

In each layer a lightweight indexer scores every key for every query,
I_ij = sum_h w_ih ReLU(q_ih . k_j), with a few small heads. Each query attends to its top-k
keys, plus MECH and itself. The pair bias is MLP_P of `inputs.pair_channels_at`, evaluated on
the selected cells only and projected per head.

The indexer reads the token vectors detached. It is trained only by the KL divergence from
the main attention (head-averaged, detached) to its own softmax over the same keys, returned
as the auxiliary loss `indexer_kl`. With `dense_warmup` (the trainer's first steps, the LR
warm-up) the main attention runs over every key with the bias on the full grid, and the
indexer learns from it; after that, top-k. With k >= N the two paths agree.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ksmech.model.config import ModelConfig
from ksmech.model.encoder import AtomPairHead, Recompute, TokenEmbedding, key_mask, mlp
from ksmech.model.inputs import (
    PairContext,
    atom_pair_static,
    frozen,
    pair_channel_count,
    pair_channels,
    pair_channels_at,
    pair_context,
)

NEG = float("-inf")


class Indexer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.heads, self.dim = cfg.d, cfg.indexer_heads, cfg.indexer_dim
        self.norm = nn.LayerNorm(d)
        self.q, self.k, self.w = (
            nn.Linear(d, self.heads * self.dim),
            nn.Linear(d, self.dim),
            nn.Linear(d, self.heads),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, N, N) index scores of every (query, key)."""
        x = self.norm(h)
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.heads, self.dim)
        s = torch.relu(torch.einsum("bnhd,bmd->bnhm", q, self.k(x)))
        return torch.einsum("bnh,bnhm->bnm", self.w(x), s)


@torch.no_grad()
def select_keys(scores: torch.Tensor, keys: torch.Tensor, k: int):
    """(idx, ok), both (B, N, k + 2): MECH, the query itself, then its top-k other real keys
    by index score; `ok` marks the real entries (MECH's own query lists MECH once)."""
    b, n, _ = scores.shape
    s = scores.float().masked_fill(~keys[:, None, :], NEG)
    eye = torch.arange(n, device=s.device)
    s[:, eye, eye] = NEG
    s[:, :, 0] = NEG
    top = s.topk(min(k, n), -1)
    idx = torch.cat(
        [torch.zeros_like(top.indices[..., :1]), eye.expand(b, n)[..., None], top.indices], -1
    )
    self_ok = (keys & (eye != 0)[None])[..., None]
    ok = torch.cat([torch.ones_like(self_ok), self_ok, top.values > NEG], -1)
    return idx, ok


def _indexer_kl(scores, attn, idx, ok, queries) -> torch.Tensor:
    """KL(main || indexer) over each query's attended keys, mean over the real queries. attn
    (B, H, N, S) is the main attention over keys `idx` (B, N, S); scores (B, N, N)."""
    p = attn.detach().float().mean(1)
    sel = scores.float().gather(-1, idx).masked_fill(~ok, NEG)
    log_q = torch.log_softmax(sel, -1)
    terms = torch.where(ok & (p > 0), p * (torch.log(p.clamp(min=1e-30)) - log_q), 0.0)
    kl = terms.sum(-1)
    return (kl * queries).sum() / queries.sum()


class SparseLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, n_channels: int):
        super().__init__()
        d, self.heads, self.k = cfg.d, cfg.heads, cfg.topk
        self.norm_h, self.norm_f = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out, self.ffn = nn.Linear(d, 3 * d), nn.Linear(d, d), mlp(d, 4 * d, d)
        self.indexer = Indexer(cfg)
        self.pair, self.bias = (
            mlp(n_channels, cfg.d_pair, cfg.d_pair),
            nn.Linear(cfg.d_pair, cfg.heads),
        )
        self.n_rbf, self.r_max = cfg.rbf, cfg.rbf_max

    def _bias(self, channels: torch.Tensor) -> torch.Tensor:
        return self.bias(self.pair(channels.to(self.out.weight.dtype))).movedim(-1, 1).float()

    def _sparse(self, q, k, v, idx, ok, ctx: PairContext, batch: dict):
        b_sz = q.shape[0]
        bi = torch.arange(b_sz, device=q.device)[:, None, None].expand_as(idx)
        qi = torch.arange(idx.shape[1], device=q.device)[None, :, None].expand_as(idx)
        with frozen(q.device.type):
            channels = pair_channels_at(ctx, batch, bi, qi, idx, self.n_rbf, self.r_max)
        ks, vs = (x.transpose(1, 2)[bi, idx] for x in (k, v))  # (B, N, S, H, d_h)
        logits = torch.einsum("bhnd,bnshd->bhns", q, ks).float() / math.sqrt(q.shape[-1])
        logits = (logits + self._bias(channels)).masked_fill(~ok[:, None], NEG)
        attn = torch.softmax(logits, -1)
        return attn, torch.einsum("bhns,bnshd->bhnd", attn.to(vs.dtype), vs)

    def _dense(self, q, k, v, keys, channels):
        logits = (q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1]) + self._bias(channels)
        attn = torch.softmax(logits.masked_fill(~keys[:, None, None, :], NEG), -1)
        return attn, attn.to(v.dtype) @ v

    def forward(self, h, ctx: PairContext, batch: dict, keys, channels, dense: bool):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm_h(h)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        scores = self.indexer(h.detach())
        if dense:
            attn, o = self._dense(q, k, v, keys, channels)
            idx = torch.arange(n, device=h.device).expand(b, n, n)
            ok = keys[:, None, :].expand(b, n, n)
        else:
            idx, ok = select_keys(scores, keys, self.k)
            attn, o = self._sparse(q, k, v, idx, ok, ctx, batch)
        kl = _indexer_kl(scores, attn, idx, ok, keys.float())
        h = h + self.out(o.transpose(1, 2).reshape(b, n, d))
        return h + self.ffn(self.norm_f(h)), kl


class SparseEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg, n_atom_x, n_orbital_x)
        channels = pair_channel_count(cfg.rbf, n_features)
        self.layers = nn.ModuleList(SparseLayer(cfg, channels) for _ in range(cfg.layers))
        self.pairs = AtomPairHead(cfg, cfg.rbf + 5)

    def forward(self, batch: dict, inp: dict, z_hat, defined, dense_warmup: bool = False):
        """(h, p_atoms, {"indexer_kl": mean over layers}) as `encoder.DenseEncoder`."""
        with frozen(z_hat.device.type):
            ctx = pair_context(batch, z_hat, defined)
            static = atom_pair_static(batch, self.cfg.rbf, self.cfg.rbf_max)
            channels = None
            if dense_warmup:
                channels = pair_channels(batch, z_hat, defined, self.cfg.rbf, self.cfg.rbf_max)
        run = Recompute(self.cfg.checkpoint and self.training)
        keys = key_mask(batch)
        h = self.embed(batch, inp)
        kls = []
        for layer in self.layers:
            h, kl = run(layer, h, ctx, batch, keys, channels, dense_warmup)
            kls.append(kl)
        return h, self.pairs(h, static), {"indexer_kl": torch.stack(kls).mean()}
