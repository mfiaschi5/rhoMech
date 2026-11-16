"""The two-track encoder (spec v9, "Model architecture"): token embeddings, the pair track
initialized from the geometry-free pair features, and L layers of attention whose logits carry
a learned bias from the pair track, each followed by the O(N^2) pair update. Every head is
learned; no categorical orbital type is embedded."""

from __future__ import annotations

import math
from typing import NamedTuple

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
    floor_kernels,
    frozen,
    pair_channel_count,
    pair_channels,
    pair_init_parts,
)
from ksmech.model.pair_linear import PairLinear, pair_linear


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


class TokenEmbedding(nn.Module):
    """h^(0) of atom tokens (E_Z + E_charge + MLP_A) and orbital tokens (MLP_O + set encoder over
    the spectral entries + the centre-weighted atom embeddings), the fragment embedding (absent
    with `ModelConfig.fragment_embedding=False`), and the learned MECH token prepended at
    position 0. `ModelConfig.element_dropout` masks E_Z (`elements`), and without
    `charge_embedding` E_charge is absent. The ablations of the
    orbital token (`drop_token_features`, `spectral_set`, `inherit_atoms`) remove a term: with no
    orbital column MLP_O is a learned vector (the MLP of an empty input), without the set encoder
    or the inherited embeddings their term is absent (`orbital_terms`)."""

    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int):
        super().__init__()
        d = cfg.d
        self.z = nn.Embedding(MAX_Z + 1, d)
        self.charge = None
        if getattr(cfg, "charge_embedding", True):
            self.charge = nn.Embedding(2 * CHARGE_MAX + 1, d)
        self.atom = mlp(n_atom_x, d, d)
        self.orbital = mlp(n_orbital_x, d, d) if n_orbital_x else None
        if self.orbital is None:
            self.orbital_base = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.spectral = mlp(2, d, d) if getattr(cfg, "spectral_set", True) else None
        self.inherit = bool(getattr(cfg, "inherit_atoms", True))
        self.fragment = None
        if getattr(cfg, "fragment_embedding", True):
            self.fragment = nn.Embedding(cfg.fragments, d)
        self.element_dropout = float(getattr(cfg, "element_dropout", 0.0))
        if not 0.0 <= self.element_dropout <= 1.0:
            raise ValueError(f"element_dropout {self.element_dropout}: a probability")
        if getattr(cfg, "atom_spin_input", "none") == "categorical_u":  # P7, P9's ablation arm
            self.spin = nn.Embedding(SPIN_MAX + 1, d)
        self.mech = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.n_fragments = cfg.fragments

    def forward(self, batch: dict, inp: dict) -> torch.Tensor:
        cat = batch["atom_categorical"]
        n_atoms = cat.shape[1]
        charge = cat[..., 1].clamp(-CHARGE_MAX, CHARGE_MAX) + CHARGE_MAX
        h_atom = self.z(self.elements(cat[..., 0]))
        if self.charge is not None:
            h_atom = h_atom + self.charge(charge)
        if hasattr(self, "spin"):
            h_atom = h_atom + self.spin(cat[..., 2].clamp(0, SPIN_MAX))
        h_atom = (h_atom + self.atom(inp["atom_x"])) * batch["is_atom"][:, :n_atoms, None]
        h_orb = sum(self.orbital_terms(batch, inp, h_atom).values())
        h = torch.cat([h_atom, h_orb], 1)
        if self.fragment is not None:
            h = h + self.fragment(batch["fragment"].clamp(min=0) % self.n_fragments)
        h = h * batch["token_mask"][..., None]
        return torch.cat([self.mech.expand(h.shape[0], -1, -1).to(h.dtype), h], 1)

    def elements(self, z: torch.Tensor) -> torch.Tensor:
        """The rows of E_Z the atoms read: their atomic numbers, or row 0 (no element has it:
        the unknown element) for the atoms `element_dropout` masks: each with that probability
        in training, none in evaluation, and every atom when it is 1."""
        z = z.clamp(0, MAX_Z)
        p = self.element_dropout
        if p >= 1.0:
            return torch.zeros_like(z)
        if p > 0.0 and self.training:
            return torch.where(torch.rand(z.shape, device=z.device) < p, 0, z)
        return z

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
    `half_pair_norm` runs the LayerNorm of p in the autocast dtype (`_norm_p`). With `atoms`
    (a tensor (B, A, A) of the batch, only its shape is read; the last layer under
    `ModelConfig.last_update_atoms`) the update runs on the atom block p[:, 1:A+1, 1:A+1] alone
    and returns that block: the one the decoder reads, the same values as the full update.
    A tensor, not an int, so that a compiled layer with dynamic shapes never specializes on A.
    `key_mask` is (B, N + 1), the keys of every query, or (B, N + 1, N + 1), the keys of each
    query (`attention_keys`)."""

    def __init__(self, cfg: ModelConfig, update: bool = True):
        super().__init__()
        d, dp, self.heads = cfg.d, cfg.d_pair, cfg.heads
        self.half_norm, self.update = bool(getattr(cfg, "half_pair_norm", False)), update
        self.splitk = bool(getattr(cfg, "pair_linear_splitk", False))
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

    def forward(self, h, p, key_mask, atoms: torch.Tensor | None = None):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm_h(h)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        pn = self._norm_p(p)
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        logits = (
            logits.float() + pair_linear(self.bias, pn, self.splitk).permute(0, 3, 1, 2).float()
        )
        keys = key_mask[:, None, None, :] if key_mask.dim() == 2 else key_mask[:, None]
        logits = logits.masked_fill(~keys, float("-inf"))
        attn = torch.softmax(logits, -1).to(v.dtype)
        h = h + self.out((attn @ v).transpose(1, 2).reshape(b, n, d))
        h = h + self.ffn(self.norm_f(h))
        if not self.update:
            return h, p
        hn = self.norm_u(h)
        pn_u = pn
        if atoms is not None:  # last layer, `last_update_atoms`: only the atom block is read
            a = atoms.shape[1]
            pn_u, p, hn = pn[:, 1 : a + 1, 1 : a + 1], p[:, 1 : a + 1, 1 : a + 1], hn[:, 1 : a + 1]
        u = (
            pair_linear(self.u_p, pn_u, self.splitk)
            + self.u_i(hn)[:, :, None, :]
            + self.u_j(hn)[:, None, :, :]
        )
        return h, p + pair_linear(self.u_out, nn.functional.gelu(u), self.splitk)


def split_pair_init(
    lin: nn.Linear, parts: PairInitParts, n_rbf: int, splitk: bool = False
) -> torch.Tensor:
    """`lin(pair_channels(...))` from `inputs.pair_init_parts`: each part times its columns of
    the weight, summed on the cells where the part is non-zero (`ModelConfig.split_pair_init`).
    The same linear map; only the summation order, hence the rounding, differs. The kind table
    is gathered with F.embedding for its backward."""
    w = lin.weight
    f2 = parts.z.shape[-1]
    c = n_rbf + f2 + 1  # same fragment, then bonded, order, distance, centre weight, kinds
    # (B, N, N, d_p), autocast dtype
    inner = (
        PairLinear.apply(parts.basis, w[:, :n_rbf], None)
        if splitk
        else nn.functional.linear(parts.basis, w[:, :n_rbf])
    )
    dt = inner.dtype
    kinds = torch.cat([w[:, c + 5 : c + 5 + KINDS].T, torch.zeros_like(w[:, :1].T)])  # (6, d_p)
    table = (kinds[:, None, :] + torch.stack([0 * w[:, c], w[:, c]])[None]).reshape(-1, w.shape[0])
    # F.embedding, not table[kind]: the same gather (a copy, bit for bit), but its backward is a
    # sorted segment sum; index_put's accumulate serialized the 12 rows' collisions (50 ms/step)
    pre = (nn.functional.embedding(parts.kind, table.to(dt)) + lin.bias.to(dt)).contiguous()
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


def attention_keys(batch: dict, scope: str = "pool") -> torch.Tensor:
    """The keys of the attention. `pool`: `key_mask`, (B, N + 1), every query reads every real
    token. `molecule` (`ModelConfig.attention_scope`): (B, N + 1, N + 1), a token reads the
    real tokens of its own molecule; MECH reads every real token and no token reads MECH, so
    nothing of another molecule reaches a token; every position reads itself, so that a padded
    query has a key."""
    keys = key_mask(batch)
    if scope == "pool":
        return keys
    same = batch["same_fragment"]
    b, n = same.shape[:2]
    out = torch.zeros(b, n + 1, n + 1, dtype=torch.bool, device=same.device)
    out[:, 1:, 1:] = same
    out[:, 0, :] = keys
    every = torch.arange(n + 1, device=same.device)
    out[:, every, every] = True
    return out


class PairVectors(NamedTuple):
    """What the decoder of `anchor_head='orbital_pairs'` reads of the pair track: `atoms`
    (B, A, A, d_p), the symmetrized atom-pair vectors (the entry heads); `cells`
    (B, R, C, d_p), the vectors of the (occupied, virtual) orbital pairs in the layout of the
    pair block, the mean of the two orientations; `read` (B, R, C), the cells the model reads
    (`ModelConfig.pair_cells`, without the cells an ablation takes away); `log_kernel` and
    `kernel_on` (B, R, C, 2), `inputs.floor_kernels` on the cells read."""

    atoms: torch.Tensor
    cells: torch.Tensor
    read: torch.Tensor
    log_kernel: torch.Tensor
    kernel_on: torch.Tensor


def atom_pairs(p) -> torch.Tensor:
    """(B, A, A, d_p) the atom-pair vectors of an encoder's pair output."""
    return p.atoms if isinstance(p, PairVectors) else p


def orbital_pair_cells(p: torch.Tensor, batch: dict) -> torch.Tensor:
    """(B, R, C, d_p) the pair track (B, N + 1, N + 1, d_p) on the (occupied, virtual) cells of
    the pair block, the mean of the cell and its transpose; a padded row or column reads MECH's
    cells, which no reader keeps (`PairVectors.read`)."""
    i = (batch["pair_row_pos"] + 1).clamp(min=0)[:, :, None]
    j = (batch["pair_col_pos"] + 1).clamp(min=0)[:, None, :]
    b = torch.arange(p.shape[0], device=p.device)[:, None, None]
    return (p[b, i, j] + p[b, j, i]) / 2


def pair_vectors(pa: torch.Tensor, p: torch.Tensor, batch: dict, defined) -> PairVectors:
    """`PairVectors` from the symmetrized atom block `pa`, the pair track `p` and the defined
    flags of the pair features (a cell is read when any of them is)."""
    read = defined.any(-1)
    with frozen(p.device.type):
        log_kernel, on = floor_kernels(batch)
    return PairVectors(pa, orbital_pair_cells(p, batch), read, log_kernel, on & read[..., None])


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
    decoder reads the symmetrized final pair vectors of the atom pairs, and with
    `anchor_head='orbital_pairs'` those of the (occupied, virtual) orbital pairs too
    (`PairVectors`; `last_update_atoms` then has no effect: the last update is needed on the
    orbital cells). `attention_scope='molecule'` restricts every layer's keys
    (`attention_keys`)."""

    def __init__(self, cfg: ModelConfig, n_atom_x: int, n_orbital_x: int, n_features: int):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg, n_atom_x, n_orbital_x)
        channels = pair_channel_count(cfg.rbf, n_features)
        self.pair_init = mlp(channels, cfg.d_pair, cfg.d_pair)
        self.splitk = bool(getattr(cfg, "pair_linear_splitk", False))
        self.scope = getattr(cfg, "attention_scope", "pool")
        self.orbital_pairs = getattr(cfg, "anchor_head", "pointer") == "orbital_pairs"
        every = int(getattr(cfg, "pair_update_every", 1))
        if every < 1:
            raise ValueError(f"pair_update_every {every}: at least 1")
        self.layers = nn.ModuleList(
            EncoderLayer(cfg, update=(k + 1) % every == 0) for k in range(cfg.layers)
        )

    def _init_split(self, parts: PairInitParts) -> torch.Tensor:
        pre = split_pair_init(self.pair_init[0], parts, self.cfg.rbf, self.splitk)
        return pair_linear(self.pair_init[2], self.pair_init[1](pre), self.splitk)

    def _init_dense(self, channels: torch.Tensor) -> torch.Tensor:
        lin1, act, lin2 = self.pair_init
        return pair_linear(lin2, act(pair_linear(lin1, channels, True)), True)

    def forward(self, batch: dict, inp: dict, z_hat, defined, dense_warmup: bool = False):
        """(h, p_atoms, aux): token vectors (B, N + 1, d), MECH at 0; symmetrized atom-pair
        vectors (B, A, A, d_p), or `PairVectors` for the orbital-pair anchor head; no
        auxiliary loss."""
        run = Recompute(self.cfg.checkpoint and self.training)
        split = getattr(self.cfg, "split_pair_init", False)
        with frozen(z_hat.device.type):
            make = pair_init_parts if split else pair_channels
            channels = make(batch, z_hat, defined, self.cfg.rbf, self.cfg.rbf_max)
        h = self.embed(batch, inp)
        if split:
            p = run(self._init_split, channels)
        else:
            p = run(self._init_dense if self.splitk else self.pair_init, channels)
        keys = attention_keys(batch, self.scope)
        atoms_only = bool(getattr(self.cfg, "last_update_atoms", False)) and self.layers[-1].update
        atoms_only = atoms_only and not self.orbital_pairs
        last = len(self.layers) - 1
        for k, layer in enumerate(self.layers):
            extra = (batch["bond_order"],) if atoms_only and k == last else ()
            h, p = run(layer, h, p, keys, *extra)
        a = batch["bond_order"].shape[1]
        pa = p if atoms_only else p[:, 1 : a + 1, 1 : a + 1]
        pa = (pa + pa.transpose(1, 2)) / 2
        if self.orbital_pairs:
            return h, pair_vectors(pa, p, batch, defined), {}
        return h, pa, {}


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
