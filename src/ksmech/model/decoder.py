"""The edit decoder (spec v9, "Edit decoder"): a pointer over the candidate atom pairs with a
sign (form, break) chooses the anchor; given it, every other pair entry is classified over
{-1, 0, +1} and every atom entry over {-2, 0, +2}. A Bernoulli head on h_MECH gives
P(END_MECH), whether the mechanism ends after the step (spec v9, "Stop head").

Atom tokens sit at positions 1..A of the encoder output (MECH at 0). `p` is the encoder's
atom-pair block (B, A, A, d_p), symmetric in (i, j) since an entry of R is: the symmetrized
pair track of the dense encoder, or `encoder.AtomPairHead` of the others. The first linear layer of
MLP_R and MLP_N is split over its inputs (the same map), so the parts that do not depend on
the anchor are computed once per sample.
"""

from __future__ import annotations

import torch
from torch import nn

from ksmech.arrows.batches import ANCHOR_SIDE
from ksmech.model.config import ModelConfig
from ksmech.model.encoder import mlp
from ksmech.model.inputs import DIST_BUCKETS, GRAPH_CAP

NEG = float("-inf")
FORM, BREAK = ANCHOR_SIDE["form"], ANCHOR_SIDE["break"]


def anchor_buckets(dist: torch.Tensor, b, a, c) -> torch.Tensor:
    """(K, A) graph distance of every atom to the nearer anchor atom, capped at 6; 7 for an
    atom of another molecule than both."""
    far = 1 << 14
    da, dc = (dist[b, :, x].long() for x in (a, c))
    d = torch.minimum(torch.where(da < 0, far, da), torch.where(dc < 0, far, dc))
    return torch.where(d >= far, DIST_BUCKETS - 1, d.clamp(max=GRAPH_CAP))


class EditDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_physics: int):
        super().__init__()
        d, dp, hid = cfg.d, cfg.d_pair, cfg.decoder_hidden
        self.step_query = mlp(d, d, d)  # MLP_S on h_MECH
        self.w_sign = nn.Linear(d, 2 * d, bias=False)  # W_form^T s, W_break^T s
        self.u_sign = nn.Linear(dp, 2)
        self.physics = mlp(n_physics, cfg.physics_hidden, 2)  # v_sigma^T MLP_Y(y_ij)
        self.r_pair, self.r_tok = nn.Linear(dp, hid), nn.Linear(d, hid)
        self.r_anchor = nn.Linear(d, hid, bias=False)
        self.r_join = nn.Linear(dp, hid, bias=False)
        self.r_dist = nn.Embedding(DIST_BUCKETS * DIST_BUCKETS, hid)
        self.r_out = nn.Linear(hid, 3)
        self.n_tok, self.n_anchor = nn.Linear(d, hid), nn.Linear(d, hid, bias=False)
        self.n_join = nn.Linear(dp, hid, bias=False)
        self.n_dist = nn.Embedding(DIST_BUCKETS, hid)
        self.n_out = nn.Linear(hid, 3)
        self.stop = mlp(d, hid, 1)  # END_MECH

    def stop_logits(self, h: torch.Tensor) -> torch.Tensor:
        """(B,) logit of END_MECH from h_MECH: on the re-featurized pool after a step, whether
        the mechanism ends there."""
        return self.stop(h[:, 0])[:, 0].float()

    def anchor_logits(self, h, p, y: torch.Tensor, batch: dict) -> torch.Tensor:
        """(B, A, A, 2) anchor logits (last axis ANCHOR_SIDE), -inf off the candidates: pairs
        i < j that may form (order stays <= 3) or break (bonded)."""
        n_atoms = y.shape[1]
        h_a, p_a = h[:, 1 : n_atoms + 1], p
        u = self.w_sign(self.step_query(h[:, 0])).view(h.shape[0], 2, -1)  # (B, 2, d)
        site = torch.einsum("bad,bsd->bas", h_a, u)  # (B, A, 2)
        logits = site[:, :, None, :] + site[:, None, :, :] + self.u_sign(p_a)
        logits = (logits + self.physics(y.to(logits.dtype))).float()
        upper = torch.ones(n_atoms, n_atoms, dtype=torch.bool, device=y.device).triu(1)
        ok = torch.stack([batch["can_form"], batch["can_break"]], -1) & upper[None, :, :, None]
        return logits.masked_fill(~ok, NEG)

    def entry_logits(self, h, p, batch: dict, anchors: torch.Tensor):
        """Entry logits given K anchors (rows: sample, atom a, atom c, sign): pairs (K, A, A, 3)
        over {-1, 0, +1} and atoms (K, A, 3) over {-2, 0, +2}, -inf on the masked classes (-1
        without a bond, +1 on a triple bond, -2 without a lone pair)."""
        b, a, c = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        n_atoms = batch["bond_order"].shape[1]
        h_a, p_a = h[:, 1 : n_atoms + 1], p
        h_anchor = h_a[b, a] + h_a[b, c]  # (K, d)
        join = p_a[b, :, a] + p_a[b, :, c]  # (K, A, d_p): pair vectors to the anchor atoms
        dist = anchor_buckets(batch["graph_distance"], b, a, c)  # (K, A)
        lo = torch.minimum(dist[:, :, None], dist[:, None])
        hi = torch.maximum(dist[:, :, None], dist[:, None])
        tok = self.r_tok(h_a)[b]  # (K, A, hid)
        jn = self.r_join(join)
        pre = self.r_pair(p_a)[b] + tok[:, :, None] + tok[:, None] + jn[:, :, None] + jn[:, None]
        pre = pre + self.r_anchor(h_anchor)[:, None, None] + self.r_dist(lo * DIST_BUCKETS + hi)
        pair = self.r_out(nn.functional.gelu(pre)).float()
        pre_n = self.n_tok(h_a)[b] + self.n_anchor(h_anchor)[:, None] + self.n_join(join)
        atom = self.n_out(nn.functional.gelu(pre_n + self.n_dist(dist))).float()
        order = batch["bond_order"][b]
        pair[..., 0] = pair[..., 0].masked_fill(order == 0, NEG)
        pair[..., 2] = pair[..., 2].masked_fill(order >= 3, NEG)
        atom[..., 0] = atom[..., 0].masked_fill(batch["lone_pairs"][b] == 0, NEG)
        return pair, atom
