"""The edit decoder (spec v9, "Edit decoder"): a pointer over the candidate atom pairs with a
sign (form, break) chooses the anchor; given it, every other pair entry is classified over
{-1, 0, +1} and every atom entry over the head's alphabet: {-2, 0, +2}, or with
`ModelConfig.open_shell` {-2, -1, 0, +1, +2} (spec v10). The atom masks are electron bookkeeping
on the non-bonding count N: -1 needs N >= 1, -2 needs N >= 2; no pool-level rule enters the head
or the loss. A Bernoulli head on h_MECH gives P(END_MECH), whether the mechanism ends after the
step (spec v9, "Stop head").

With `ModelConfig.anchor_head='orbital_pairs'` the anchor is `OrbitalPairAnchor` in the pointer's
place; the entry heads and the stop head are the same.

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
from ksmech.model.encoder import PairVectors, atom_pairs, mlp
from ksmech.model.inputs import DIST_BUCKETS, GRAPH_CAP, _pool, centre_weights

NEG = float("-inf")
FORM, BREAK = ANCHOR_SIDE["form"], ANCHOR_SIDE["break"]
# The value of each class of the atom head, in class order. An odd entry (a one-electron
# step) is outside the 3-class alphabet: the loss raises on it instead of mapping it to a class.
ATOM_VALUES = (-2, 0, 2)
OPEN_SHELL_ATOM_VALUES = (-2, -1, 0, 1, 2)
ATOM_MASKS = {-2: "can_lose_pair", -1: "can_lose_electron"}  # a class -> the batch mask it needs


def anchor_buckets(dist: torch.Tensor, b, a, c) -> torch.Tensor:
    """(K, A) graph distance of every atom to the nearer anchor atom, capped at 6; 7 for an
    atom of another molecule than both."""
    far = 1 << 14
    da, dc = (dist[b, :, x].long() for x in (a, c))
    d = torch.minimum(torch.where(da < 0, far, da), torch.where(dc < 0, far, dc))
    return torch.where(d >= far, DIST_BUCKETS - 1, d.clamp(max=GRAPH_CAP))


# The orbital-pair anchor head at its start (ANCHOR_SIDE order: form, break), learned from there:
# the exponent and the level of each sign's kernel, and the level of an orbital pair the kernel
# of that sign is not defined on. Fitted once on the anchor NLL of the untrained head over
# PMechDB's training part (docs/decisions.md, 2026-11-16).
KERNEL_SCALE, KERNEL_BIAS, OFF_LEVEL = (1.75, 3.5), (0.0, -1.4), (-4.2, -4.1)
TINY = 1e-30  # an atom pair with no orbital pair keeps this much of a sample's largest term
ORBITAL_TERMS = ("floor", "orbital_pairs", "charge")  # `OrbitalPairAnchor.terms`


class OrbitalPairAnchor(nn.Module):
    """The anchor logits of `ModelConfig.anchor_head='orbital_pairs'`:

        l_sigma(i, j) = log sum_rs (w_i^r w_j^s + w_j^r w_i^s) exp(phi_sigma(r, s))
                        + c_sigma q_i q_j
        phi_sigma(r, s) = kappa_sigma(r, s) + f_sigma(p_rs) + d_sigma . h_r + e_sigma . h_s

    The sum runs over the (occupied r, virtual s) orbital pairs the model reads, with the
    centre weights of G^atom, so before the candidate mask the probability of an atom pair is
    a mixture over the pool's orbital pairs: an orbital pair is chosen, and its centres say
    which atoms. kappa is the physics floor: b_sigma + a_sigma log k_sigma(r, s) on the cells
    of that sign's kernel (`inputs.floor_kernels`: |g| on the valid cells for form, |own_g| for
    break) and a learned level u_sigma elsewhere; f is an MLP of the pair track's final vector
    of the orbital pair, d and e are linear in the final vectors of the occupied and of the
    virtual orbital token (how good a donor, how good an acceptor), all three zero at the
    start. With a = 1, b = 0 and u = -inf an untrained head gives the log of the pooled |g| and
    of |B^own|, the ranking of `eval.floor`'s `own_break`. c (`anchor_charge_term`) scales the
    Klopman-Salem charge term. No term reads an atom token, the atom-pair vector or h_MECH; an
    atom pair without an orbital pair has no logit to speak of (TINY)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.score = mlp(cfg.d_pair, cfg.decoder_hidden, 2)
        self.donor, self.acceptor = nn.Linear(cfg.d, 2), nn.Linear(cfg.d, 2)
        for layer in (self.score[-1], self.donor, self.acceptor):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.kernel_scale = nn.Parameter(torch.tensor(KERNEL_SCALE))
        self.kernel_bias = nn.Parameter(torch.tensor(KERNEL_BIAS))
        self.off_level = nn.Parameter(torch.tensor(OFF_LEVEL))
        self.charge = nn.Parameter(torch.zeros(2)) if cfg.anchor_charge_term else None

    def kernel(self, p: PairVectors) -> torch.Tensor:
        """(B, R, C, 2) kappa: the floor of every orbital pair, per sign."""
        on_kernel = self.kernel_bias.float() + self.kernel_scale.float() * p.log_kernel
        return torch.where(p.kernel_on, on_kernel, self.off_level.float())

    def pooled(self, phi: torch.Tensor, p: PairVectors, batch: dict) -> torch.Tensor:
        """(B, A, A, 2) log sum_rs W_rs(i, j) exp(phi_rs) over the cells read, in float32."""
        phi = phi.float().masked_fill(~p.read[..., None], NEG)
        top = phi.detach().flatten(1, 2).amax(1)  # (B, 2)
        top = torch.where(torch.isfinite(top), top, 0.0)[:, None, None, :]
        w_occ, w_virt = centre_weights(batch)
        mass = _pool(w_occ, torch.exp(phi - top), w_virt)
        return torch.log(mass.clamp(min=TINY)) + top

    def learned(self, h: torch.Tensor, p: PairVectors, batch: dict) -> torch.Tensor:
        """(B, R, C, 2) f(p_rs) + d . h_r + e . h_s, in float32."""

        def tokens(pos: torch.Tensor) -> torch.Tensor:  # the orbital tokens of a block axis
            at = (pos + 1).clamp(min=0)[..., None].expand(-1, -1, h.shape[-1])
            return torch.gather(h, 1, at)

        donor = self.donor(tokens(batch["pair_row_pos"])).float()[:, :, None, :]
        acceptor = self.acceptor(tokens(batch["pair_col_pos"])).float()[:, None, :, :]
        return self.score(p.cells).float() + donor + acceptor

    def _charge(self, y: torch.Tensor) -> torch.Tensor:
        if self.charge is None:
            return torch.zeros(*y.shape[:3], 2, device=y.device)
        return y.float() * self.charge.float()

    def forward(self, h, p: PairVectors, y: torch.Tensor, batch: dict) -> torch.Tensor:
        """(B, A, A, 2) the logits, before the candidate mask."""
        phi = self.kernel(p) + self.learned(h, p, batch)
        return self.pooled(phi, p, batch) + self._charge(y)

    def terms(self, h, p: PairVectors, y: torch.Tensor, batch: dict) -> torch.Tensor:
        """(B, A, A, 2, 3), ORBITAL_TERMS: the floor (the logit of kappa alone), what the
        learned scores add to it, and the charge term; their sum is the logit."""
        kappa = self.kernel(p)
        floor = self.pooled(kappa, p, batch)
        full = self.pooled(kappa + self.learned(h, p, batch), p, batch)
        return torch.stack([floor, full - floor, self._charge(y)], -1)


class EditDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_physics: int):
        super().__init__()
        d, dp, hid = cfg.d, cfg.d_pair, cfg.decoder_hidden
        self.orbital = None
        if getattr(cfg, "anchor_head", "pointer") == "orbital_pairs":
            self.orbital = OrbitalPairAnchor(cfg)
        else:
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
        self.atom_values = OPEN_SHELL_ATOM_VALUES if cfg.open_shell else ATOM_VALUES
        self.n_out = nn.Linear(hid, len(self.atom_values))
        self.stop = mlp(d, hid, 1)  # END_MECH

    def stop_logits(self, h: torch.Tensor) -> torch.Tensor:
        """(B,) logit of END_MECH from h_MECH: on the re-featurized pool after a step, whether
        the mechanism ends there."""
        return self.stop(h[:, 0])[:, 0].float()

    def anchor_logits(self, h, p, y: torch.Tensor, batch: dict) -> torch.Tensor:
        """(B, A, A, 2) anchor logits (last axis ANCHOR_SIDE), -inf off the candidates: pairs
        i < j that may form (order stays <= 3) or break (bonded). The pointer of spec v9, or
        `OrbitalPairAnchor`."""
        n_atoms = y.shape[1]
        if self.orbital is not None:
            logits = self.orbital(h, p, y, batch)
        else:
            logits = self._pointer_logits(h, p, y)
        upper = torch.ones(n_atoms, n_atoms, dtype=torch.bool, device=y.device).triu(1)
        ok = torch.stack([batch["can_form"], batch["can_break"]], -1) & upper[None, :, :, None]
        return logits.masked_fill(~ok, NEG)

    def _pointer_logits(self, h, p, y: torch.Tensor) -> torch.Tensor:
        n_atoms = y.shape[1]
        h_a, p_a = h[:, 1 : n_atoms + 1], p
        u = self.w_sign(self.step_query(h[:, 0])).view(h.shape[0], 2, -1)  # (B, 2, d)
        site = torch.einsum("bad,bsd->bas", h_a, u)  # (B, A, 2)
        logits = site[:, :, None, :] + site[:, None, :, :] + self.u_sign(p_a)
        return (logits + self.physics(y.to(logits.dtype))).float()

    def entry_logits(self, h, p, batch: dict, anchors: torch.Tensor):
        """Entry logits given K anchors (rows: sample, atom a, atom c, sign): pairs (K, A, A, 3)
        over {-1, 0, +1} and atoms (K, A, len(atom_values)), -inf on the masked classes (-1
        without a bond, +1 on a triple bond, -2 with N < 2, -1 on an atom with N < 1)."""
        b, a, c = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        n_atoms = batch["bond_order"].shape[1]
        h_a, p_a = h[:, 1 : n_atoms + 1], atom_pairs(p)
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
        for k, value in enumerate(self.atom_values):
            if value in ATOM_MASKS:
                atom[..., k] = atom[..., k].masked_fill(~batch[ATOM_MASKS[value]][b], NEG)
        return pair, atom
