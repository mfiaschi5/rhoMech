"""Linear layers of the pair track with a split-K weight gradient (ModelConfig.pair_linear_splitk):
the same forward as nn.Linear under autocast (F.linear on the operands autocast would cast, bit
for bit); the weight gradient, a sum over every pair cell, computed as SPLIT_K batched products
summed in float32, because one GEMM with a 64 x 64 output over ~5e5 rows keeps only a few compute
units busy (docs/model_impl.md, item 21)."""

from __future__ import annotations

import torch
from torch import nn

# measured on the RX 6600 XT, R = 550000 rows, fp16: 32 splits give 2.93x for a 64 x 64 weight
# gradient, 2.85x for 64 x 8, against the stock GEMM
SPLIT_K = 32


def _accumulate_dtype(dt: torch.dtype) -> torch.dtype:
    return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt


def weight_grad(g2: torch.Tensor, x2: torch.Tensor, splits: int = SPLIT_K) -> torch.Tensor:
    """g2^T x2 (O, I) for g2 (R, O), x2 (R, I): the row sum in `splits` batched products plus
    the remainder rows, in float32 (float64 stays float64). No branch on R, so dynamic shapes
    add no guard."""
    r, o, i = g2.shape[0], g2.shape[1], x2.shape[1]
    per = r // splits
    main = per * splits
    part = torch.bmm(
        g2[:main].reshape(splits, per, o).transpose(1, 2), x2[:main].reshape(splits, per, i)
    )
    acc = _accumulate_dtype(g2.dtype)
    return part.to(acc).sum(0) + (g2[main:].t() @ x2[main:]).to(acc)


def _compute_dtype(x: torch.Tensor) -> torch.dtype | None:
    """The dtype autocast gives F.linear's operands, None outside autocast."""
    kind = x.device.type
    return torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else None


def _cast(x, w, b, dt):
    return x.to(dt), w.to(dt), (None if b is None else b.to(dt))


@torch.library.custom_op("ksmech::pair_linear", mutates_args=())
def _pair_linear_op(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, dt: torch.dtype | None
) -> torch.Tensor:
    """F.linear on the operands autocast would cast (`dt`), autocast off inside."""
    if dt is not None:
        x, w, b = _cast(x, w, b, dt)
    with torch.autocast(x.device.type, enabled=False):
        return nn.functional.linear(x, w, b)


@_pair_linear_op.register_fake
def _(x, w, b, dt):
    out_dt = dt if dt is not None else torch.promote_types(x.dtype, w.dtype)
    return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=out_dt)


def _pair_linear_setup(ctx, inputs, output):
    x, w, b, dt = inputs
    ctx.save_for_backward(x, w)
    ctx.has_bias, ctx.dt = b is not None, dt


def _pair_linear_backward(ctx, g):
    x, w = ctx.saved_tensors
    xc, wc = (x, w) if ctx.dt is None else (x.to(ctx.dt), w.to(ctx.dt))
    g2 = g.reshape(-1, g.shape[-1])
    gx = gw = gb = None
    if ctx.needs_input_grad[0]:
        gx = (g2 @ wc).reshape(*g.shape[:-1], wc.shape[1]).to(x.dtype)
    if ctx.needs_input_grad[1]:
        gw = weight_grad(g2, xc.reshape(-1, xc.shape[-1])).to(w.dtype)
    if ctx.has_bias and ctx.needs_input_grad[2]:
        gb = g2.sum(0, dtype=_accumulate_dtype(g2.dtype)).to(w.dtype)
    return gx, gw, gb, None


_pair_linear_op.register_autograd(_pair_linear_backward, setup_context=_pair_linear_setup)


class PairLinear:
    """`F.linear(x, w, b)` (under autocast, on the operands autocast casts) with `weight_grad`
    as the gradient of w. A custom op, not an autograd.Function: a compiled layer under
    activation checkpointing recomputes its forward on another compiled graph after a
    recompile, and the saved tensors of an inlined Function differ between the two."""

    @staticmethod
    def apply(x, w, b):
        return _pair_linear_op(x, w, b, _compute_dtype(x))


def pair_linear(lin: nn.Linear, x: torch.Tensor, on: bool) -> torch.Tensor:
    """`lin(x)`, through PairLinear when `on` (the module itself otherwise, so the default path
    is unchanged)."""
    return PairLinear.apply(x, lin.weight, lin.bias) if on else lin(x)
