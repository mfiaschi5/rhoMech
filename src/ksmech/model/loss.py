"""The step loss and the edit-level metrics (spec v9, "Training objective", "Evaluation
protocol").

    L_step = -log sum_{R in class} sum_{e in E(R)} P(e) prod_{x != e} P(R_x | e)
    L = L_step + lambda_stop BCE(P(END_MECH), stop label)

E(R) is the anchors the label stores with each variant (`r_anchors`): its +1 entries, or its
-1 entries when it has none. Every pair entry i < j of the sample's atoms other than the anchor
and every atom entry is in the product, zeros included. The stop term is masked per sample:
only samples with a stop label have one (PMechDB, isolated steps, has none).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ksmech.model.decoder import FORM, EditDecoder


def _upper(n: int, device) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).triu(1)


def _real_atoms(batch: dict) -> torch.Tensor:
    return batch["is_atom"][:, : batch["bond_order"].shape[1]]


def anchor_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """log P(e) over each sample's candidates: (B, A, A, 2), -inf off them."""
    flat = logits.flatten(1)
    return torch.log_softmax(flat, -1).view_as(logits)


def admissible_rows(batch: dict) -> torch.Tensor:
    """(K, 5) rows (sample, variant, i, j, sign) with i < j: every admissible anchor of every
    real variant of the label class."""
    n = batch["r_anchors"].shape[2]
    keep = batch["r_anchors"] & _upper(n, batch["r_anchors"].device)[None, None, :, :, None]
    keep &= batch["variant_mask"][:, :, None, None, None]
    return keep.nonzero()


def _segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    top = torch.full((n,), float("-inf"), device=x.device).scatter_reduce(0, seg, x, "amax")
    total = torch.zeros(n, device=x.device).scatter_add(0, seg, torch.exp(x - top[seg]))
    return top + torch.log(total)


def _entry_sums(pair_lp, atom_lp, r_pairs, r_atoms, real, i, j) -> torch.Tensor:
    """Sum of the target classes' log-probabilities over the real entries, the anchor (i, j)
    left out. pair_lp (K, A, A, 3), atom_lp (K, A, 3); targets (K, A, A) and (K, A)."""
    n = r_pairs.shape[1]
    lp = pair_lp.gather(-1, (r_pairs.long() + 1)[..., None])[..., 0]
    m = _upper(n, lp.device)[None] & real[:, :, None] & real[:, None, :]
    m[torch.arange(len(i)), i, j] = False
    la = atom_lp.gather(-1, (torch.div(r_atoms.long(), 2, rounding_mode="floor") + 1)[..., None])
    return torch.where(m, lp, 0.0).sum((1, 2)) + torch.where(real, la[..., 0], 0.0).sum(1)


def stop_bce(logits: torch.Tensor, batch: dict) -> torch.Tensor | None:
    """(B,) BCE of END_MECH, 0 on the samples without a stop label; None when the batch has no
    labels. `batch["stop_label"]` (B,), optional: 1 the mechanism ends after this step, 0 it
    goes on, -1 unknown."""
    labels = batch.get("stop_label")
    if labels is None or not (labels >= 0).any():
        return None
    known = labels >= 0
    target = labels.clamp(min=0).to(logits.dtype)  # no NaN gradient from the masked samples
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return torch.where(known, bce, 0.0)


def step_loss(
    dec: EditDecoder, h, p, y, batch: dict, stop_weight: float = 0.0
) -> tuple[torch.Tensor, dict]:
    """(mean L over the batch, parts): the marginal over the label class and its anchors, plus
    `stop_weight` times the masked stop BCE; with no stop label or a zero weight the loss is
    L_step exactly. `parts` holds the per-sample losses, the anchor NLL (-log of the
    probability on any admissible anchor), the anchor log-probabilities and P(END_MECH) (B,),
    detached."""
    lp_anchor = anchor_log_probs(dec.anchor_logits(h, p, y, batch))
    rows = admissible_rows(batch)
    b, v, i, j, sign = rows.unbind(1)
    if torch.unique(b).numel() != lp_anchor.shape[0]:
        raise ValueError("a sample of the batch has no admissible anchor")
    pair, atom = dec.entry_logits(h, p, batch, torch.stack([b, i, j, sign], 1))
    sums = _entry_sums(
        torch.log_softmax(pair, -1),
        torch.log_softmax(atom, -1),
        batch["r_pairs"][b, v],
        batch["r_atoms"][b, v],
        _real_atoms(batch)[b],
        i,
        j,
    )
    terms = lp_anchor[b, i, j, sign] + sums
    per_sample = -_segment_logsumexp(terms, b, lp_anchor.shape[0])
    anchor_nll = -_segment_logsumexp(lp_anchor[b, i, j, sign], b, lp_anchor.shape[0])
    stop = dec.stop_logits(h)
    bce = stop_bce(stop, batch) if stop_weight else None
    if bce is not None:
        per_sample = per_sample + stop_weight * bce
    parts = {
        "loss": per_sample.detach(),
        "anchor_nll": anchor_nll.detach(),
        "lp_anchor": lp_anchor.detach(),
        "p_stop": torch.sigmoid(stop).detach(),
    }
    return per_sample.mean(), parts


def anchor_ranks(lp_anchor: torch.Tensor, batch: dict) -> torch.Tensor:
    """(B,) 0-based rank of the best admissible anchor among the candidates (set
    marginalization: any admissible anchor of any variant counts)."""
    rows = admissible_rows(batch)
    b, _, i, j, sign = rows.unbind(1)
    best = torch.full((lp_anchor.shape[0],), float("-inf"), device=lp_anchor.device)
    best = best.scatter_reduce(0, b, lp_anchor[b, i, j, sign], "amax")
    return (lp_anchor.flatten(1) > best[:, None]).sum(1)


def greedy_edit(dec: EditDecoder, h, p, lp_anchor: torch.Tensor, batch: dict):
    """(R pairs (B, A, A) int, R atoms (B, A) int): the top anchor and, given it, the most
    probable class of every other entry (no beam, no conservation check)."""
    n_b, n = lp_anchor.shape[:2]
    flat = lp_anchor.flatten(1).argmax(1)
    i, rest = flat // (n * 2), flat % (n * 2)
    j, sign = rest // 2, rest % 2
    b = torch.arange(n_b, device=flat.device)
    pair, atom = dec.entry_logits(h, p, batch, torch.stack([b, i, j, sign], 1))
    r_pairs = pair.argmax(-1) - 1
    r_pairs = torch.where(_upper(n, flat.device), r_pairs, 0)
    r_pairs[b, i, j] = torch.where(sign == FORM, 1, -1)
    r_pairs = r_pairs + r_pairs.transpose(1, 2)
    real = _real_atoms(batch)
    r_pairs = r_pairs * (real[:, :, None] & real[:, None, :])
    return r_pairs, (atom.argmax(-1) - 1) * 2 * real


def edit_matches(r_pairs: torch.Tensor, r_atoms: torch.Tensor, batch: dict) -> torch.Tensor:
    """(B,) bool: the predicted R equals a variant of the label class."""
    same = (batch["r_pairs"] == r_pairs[:, None]).flatten(2).all(-1)
    same &= (batch["r_atoms"] == r_atoms[:, None]).all(-1)
    return (same & batch["variant_mask"]).any(1)
