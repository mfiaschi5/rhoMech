"""The flex encoder: no pair state. Each layer and head adds a learned relative bias, linear in
the pair features (T5 / ALiBi style), inside FlexAttention's score_mod:

- RBF(distance) from the O(N) positions, within a fragment, with per-head weights. Each
  head's radial function sum_k w_hk RBF_k(d) is tabulated 50 times per A (from the weights,
  so it stays differentiable) and interpolated linearly; the dense reference reads the same
  table;
- on atom pairs, learned tables over bond order (0..3) and capped graph distance (0..6, 7 for
  another molecule), and a Wiberg weight;
- a same-fragment weight;
- z on the (occupied row, virtual column) cells, projected per head: the only materialized
  block, (B, R, C, H).

Padding is a block mask. FlexAttention has no backward on the CPU, so the kernel runs on the
GPU; `FlexBias.dense` builds the same bias as a (B, H, N + 1, N + 1) tensor for SDPA, the
forward-only reference the kernel is tested against. The decoder's atom-pair vectors come from
`encoder.AtomPairHead`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ksmech.model.config import ModelConfig
from ksmech.model.encoder import AtomPairHead, Recompute, TokenEmbedding, key_mask, mlp
from ksmech.model.inputs import (
    DIST_BUCKETS,
    GRAPH_CAP,
    PairContext,
    atom_pair_static,
    distances,
    frozen,
    pair_context,
    rbf,
)

# Bins of the interpolated radial function per Angstrom. An int: under dynamic shapes dynamo
# traces a float constant as a symbolic float, which Inductor cannot lower in a score_mod.
GRID_PER_A = 50
GRID_MAX = 10  # Angstrom; the Gaussians centred up to 8 A have decayed, farther pairs read the end

_flex = None


def grid_bins(d: torch.Tensor, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(bin, fraction) of distances on the radial grid, for linear interpolation; works on a
    score_mod scalar and on a tensor alike."""
    t = d * GRID_PER_A
    i = t.floor().clamp(0, n_bins - 1)
    return i.to(torch.int64), (t - i).clamp(0, 1)


def compiled_flex():
    """FlexAttention compiled with dynamic shapes. A batch of one and fewer than 128 queries
    (the decoding kernel) each compile a variant of their own (60-100 s each on the card), with
    and without gradient: 8 variants, dynamo's default limit, past which it would fall back to
    the slow eager path. The limit is raised to 16."""
    global _flex
    if _flex is None:
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)
        _flex = torch.compile(flex_attention, dynamic=True)
    return _flex


@dataclass
class FlexInputs:
    ctx: PairContext
    order: torch.Tensor  # (B, A, A) long bond order
    dist: torch.Tensor  # (B, A, A) long graph-distance bucket
    wiberg: torch.Tensor  # (B, A, A)
    keys: torch.Tensor  # (B, N + 1) real keys
    block_mask: object = None  # FlexAttention block mask (GPU path only)


def flex_inputs(batch: dict, ctx: PairContext, kernel: bool) -> FlexInputs:
    dist = batch["graph_distance"].long()
    bucket = torch.where(dist < 0, DIST_BUCKETS - 1, dist.clamp(max=GRAPH_CAP))
    keys = key_mask(batch)
    block = None
    if kernel:
        b_sz, n = keys.shape
        block = create_block_mask(
            lambda b, h, q, kv: keys[b, kv], b_sz, None, n, n, device=keys.device
        )
    order = batch["bond_order"].long()
    return FlexInputs(ctx, order, bucket, batch["wiberg"].float(), keys, block)


class FlexBias(nn.Module):
    def __init__(self, cfg: ModelConfig, n_z: int):
        super().__init__()
        h = cfg.heads
        self.n_rbf, self.r_max = cfg.rbf, cfg.rbf_max
        self.w_rbf = nn.Parameter(0.02 * torch.randn(h, cfg.rbf))
        self.order = nn.Parameter(0.02 * torch.randn(h, 4))
        self.dist = nn.Parameter(0.02 * torch.randn(h, DIST_BUCKETS))
        self.w_wiberg = nn.Parameter(torch.zeros(h))
        self.w_frag = nn.Parameter(torch.zeros(h))
        self.z = nn.Linear(n_z, h)

    def radial_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-head radial function sum_k w_hk RBF_k(d) on a grid of GRID_PER_A points per
        A up to GRID_MAX, as two (H, G - 1) tables: the lower and the upper neighbour of each bin.
        FlexAttention cannot index one tensor that requires gradients twice in a score_mod
        (the 32-term sum did), so the interpolation reads two tensors."""
        grid = torch.arange(GRID_MAX * GRID_PER_A + 1, device=self.w_rbf.device) / GRID_PER_A
        table = self.w_rbf @ rbf(grid, self.n_rbf, self.r_max).T
        return table[:, :-1].contiguous(), table[:, 1:].contiguous()

    def score_mod(self, x: FlexInputs, zproj: torch.Tensor):
        """The bias of one cell, for FlexAttention: the same sum as `dense`. The small per-head
        tables are read from one copy per (sample, head, query): read directly, every cell of a
        head adds its gradient to the same few addresses, and the atomic adds serialize (2.3 s
        of a 2.5 s backward on the RX 6600 XT; 0.3 s with the copies)."""
        ctx, a = x.ctx, x.order.shape[1]
        lo, hi = self.radial_tables()
        b_sz, n = x.keys.shape
        order, dist, w_wbi, w_frag = (
            t[None, :, None].expand(b_sz, t.shape[0], n, *t.shape[1:]).contiguous()
            for t in (self.order, self.dist, self.w_wiberg, self.w_frag)
        )

        def mod(score, b, h, q, kv):
            d2 = sum((ctx.pos[b, q, c] - ctx.pos[b, kv, c]) ** 2 for c in range(3))
            i, frac = grid_bins(torch.sqrt(d2), lo.shape[1])
            radial = lo[h, i] * (1 - frac) + hi[h, i] * frac
            same = (ctx.frag[b, q] == ctx.frag[b, kv]) & (ctx.frag[b, q] >= 0)
            qa, ka = (q - 1).clamp(0, a - 1), (kv - 1).clamp(0, a - 1)
            aa = (q >= 1) & (q <= a) & (kv >= 1) & (kv <= a)
            lewis = order[b, h, q, x.order[b, qa, ka]] + dist[b, h, q, x.dist[b, qa, ka]]
            lewis = lewis + w_wbi[b, h, q] * x.wiberg[b, qa, ka]
            r, c = ctx.occ_row[b, q], ctx.virt_col[b, kv]
            zz = zproj[b, r.clamp(min=0), c.clamp(min=0), h]
            out = score + torch.where(same, radial + w_frag[b, h, q], 0.0)
            out = out + torch.where(aa, lewis, 0.0)
            return out + torch.where((r >= 0) & (c >= 0), zz, 0.0)

        return mod

    def dense(self, x: FlexInputs, zproj: torch.Tensor) -> torch.Tensor:
        """(B, H, N + 1, N + 1) the same bias, materialized (the reference path)."""
        ctx, a = x.ctx, x.order.shape[1]
        same = (ctx.frag[:, :, None] == ctx.frag[:, None, :]) & (ctx.frag[:, :, None] >= 0)
        lo, hi = self.radial_tables()
        i, frac = grid_bins(distances(ctx.pos), lo.shape[1])
        radial = (lo[:, i] * (1 - frac) + hi[:, i] * frac).transpose(0, 1)
        bias = torch.where(same[:, None], radial + self.w_frag[None, :, None, None], 0.0)
        lewis = self.order[:, x.order] + self.dist[:, x.dist]
        lewis = lewis + self.w_wiberg[:, None, None, None] * x.wiberg
        n1 = bias.shape[-1]
        bias = bias + nn.functional.pad(lewis.transpose(0, 1), (1, n1 - a - 1, 1, n1 - a - 1))
        cells = (ctx.occ_row[:, :, None] >= 0) & (ctx.virt_col[:, None, :] >= 0)
        b, r, c = cells.nonzero(as_tuple=True)
        zcell = torch.zeros_like(bias)
        zcell[b, :, r, c] = zproj[b, ctx.occ_row[b, r], ctx.virt_col[b, c]].to(zcell.dtype)
        return bias + zcell


class FlexLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, n_z: int):
        super().__init__()
        d, self.heads = cfg.d, cfg.heads
        self.norm_h, self.norm_f = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out, self.ffn = nn.Linear(d, 3 * d), nn.Linear(d, d), mlp(d, 4 * d, d)
        self.bias = FlexBias(cfg, n_z)

    def attend(self, h: torch.Tensor, x: FlexInputs, kernel: bool) -> torch.Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm_h(h)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        zproj = self.bias.z(x.ctx.z)
        if kernel:
            o = compiled_flex()(
                q, k, v, score_mod=self.bias.score_mod(x, zproj), block_mask=x.block_mask
            )
        else:
            bias = self.bias.dense(x, zproj).masked_fill(~x.keys[:, None, None, :], float("-inf"))
            o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return o.transpose(1, 2).reshape(b, n, d)

    def forward(self, h: torch.Tensor, x: FlexInputs, kernel: bool) -> torch.Tensor:
        h = h + self.out(self.attend(h, x, kernel))
        return h + self.ffn(self.norm_f(h))


class FlexEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg, n_atom_x, n_orbital_x)
        self.layers = nn.ModuleList(FlexLayer(cfg, 2 * n_features) for _ in range(cfg.layers))
        self.pairs = AtomPairHead(cfg, cfg.rbf + 5)

    def forward(self, batch: dict, inp: dict, z_hat, defined, dense_warmup: bool = False):
        """(h, p_atoms, aux) as `encoder.DenseEncoder`; the kernel runs on the GPU when
        `cfg.flex_kernel`, the dense reference elsewhere."""
        kernel = self.cfg.flex_kernel and z_hat.is_cuda
        with frozen(z_hat.device.type):
            x = flex_inputs(batch, pair_context(batch, z_hat, defined), kernel)
            static = atom_pair_static(batch, self.cfg.rbf, self.cfg.rbf_max)
        run = Recompute(self.cfg.checkpoint and self.training)
        h = self.embed(batch, inp)
        for layer in self.layers:
            h = run(layer, h, x, kernel)
        return h, self.pairs(h, static), {}
