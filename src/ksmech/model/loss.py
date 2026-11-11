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
from ksmech.model.inputs import device_index


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
    real variant of the label class; the collate's `anchor_rows` when the batch has them (the
    same rows, without a host sync)."""
    if "anchor_rows" in batch:
        return batch["anchor_rows"]
    n = batch["r_anchors"].shape[2]
    keep = batch["r_anchors"] & _upper(n, batch["r_anchors"].device)[None, None, :, :, None]
    keep &= batch["variant_mask"][:, :, None, None, None]
    return keep.nonzero()


def _segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    top = torch.full((n,), float("-inf"), device=x.device).scatter_reduce(0, seg, x, "amax")
    total = torch.zeros(n, device=x.device).scatter_add(0, seg, torch.exp(x - top[seg]))
    return top + torch.log(total)


def atom_classes(r_atoms: torch.Tensor, values: tuple[int, ...], real: torch.Tensor, present=None):
    """Class index of every atom entry in the head's alphabet `values` (the decoder's
    `atom_values`). A real entry outside it raises: an odd entry of a one-electron step is not
    a class of the 3-class head, and mapping it to one would train on the wrong target.
    `present`: the values of those real entries, known on the host (the collate's
    `anchor_atom_values`), checked without a host sync."""
    table = device_index(tuple(values), r_atoms.device)
    hit = r_atoms.long()[..., None] == table
    if present is not None:
        odd = sorted(set(present) - set(values))
    else:
        bad = real & ~hit.any(-1)
        odd = sorted({int(x) for x in r_atoms[bad].tolist()}) if bad.any() else []
    if odd:
        raise ValueError(f"atom entries {odd} are outside the head's alphabet {values}")
    return torch.where(hit.any(-1), hit.long().argmax(-1), values.index(0))


def _entry_sums(
    pair_lp, atom_lp, r_pairs, r_atoms, real, i, j, values, present=None
) -> torch.Tensor:
    """Sum of the target classes' log-probabilities over the real entries, the anchor (i, j)
    left out. pair_lp (K, A, A, 3), atom_lp (K, A, len(values)); targets (K, A, A) and (K, A);
    `values` the atom classes' values; `present` as `atom_classes`."""
    n = r_pairs.shape[1]
    lp = pair_lp.gather(-1, (r_pairs.long() + 1)[..., None])[..., 0]
    m = _upper(n, lp.device)[None] & real[:, :, None] & real[:, None, :]
    m[torch.arange(len(i), device=lp.device), i, j] = False
    la = atom_lp.gather(-1, atom_classes(r_atoms, values, real, present)[..., None])
    return torch.where(m, lp, 0.0).sum((1, 2)) + torch.where(real, la[..., 0], 0.0).sum(1)


def _anchor_nll(lp_anchor: torch.Tensor, b, i, j, sign, unique=None) -> torch.Tensor:
    """(B,) -log P(any admissible anchor). An anchor that several variants share is one
    choice of the pointer, so it counts once; the loss keeps one path per variant. `unique`:
    the distinct (b, i, j, sign) rows when known (the collate's `anchor_unique`)."""
    if unique is None:
        unique = torch.unique(torch.stack([b, i, j, sign], 1), dim=0)
    ub, ui, uj, us = unique.unbind(1)
    return -_segment_logsumexp(lp_anchor[ub, ui, uj, us], ub, lp_anchor.shape[0])


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
    missing = batch.get("anchors_missing")
    if missing is None:
        missing = lp_anchor.shape[0] - torch.unique(b).numel()
    if missing:
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
        dec.atom_values,
        batch.get("anchor_atom_values"),
    )
    terms = lp_anchor[b, i, j, sign] + sums
    per_sample = -_segment_logsumexp(terms, b, lp_anchor.shape[0])
    anchor_nll = _anchor_nll(lp_anchor, b, i, j, sign, batch.get("anchor_unique"))
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
    """(B,) 0-based rank of the best admissible anchor: the candidates outside the label that
    score at least as high (set marginalization: any admissible anchor of any variant counts).
    A tie with such a candidate counts against the label, so the rank does not depend on how
    the precision breaks ties: in bfloat16, equivalent atoms tie exactly and a tie used to count
    as a hit (docs/decisions.md, 2026-10-26). A NaN score counts against the label too, and a
    NaN label score ranks last: NaN logits compare false with everything, so a model with NaN
    weights used to score top-1 = 1 (docs/decisions.md, 2026-10-28)."""
    rows = admissible_rows(batch)
    b, _, i, j, sign = rows.unbind(1)
    best = torch.full((lp_anchor.shape[0],), float("-inf"), device=lp_anchor.device)
    best = best.scatter_reduce(0, b, lp_anchor[b, i, j, sign], "amax")
    label = torch.zeros_like(lp_anchor, dtype=torch.bool)
    label[b, i, j, sign] = True
    beats = (lp_anchor >= best[:, None, None, None]) | lp_anchor.isnan()
    rank = (beats & ~label).flatten(1).sum(1)
    return torch.where(best.isnan(), lp_anchor[0].numel(), rank)


def greedy_edit(dec: EditDecoder, h, p, lp_anchor: torch.Tensor, batch: dict):
    """(R pairs (B, A, A) int, R atoms (B, A) int): the top anchor and, given it, the most
    probable class of every other entry (no beam, no conservation check)."""
    n_b, n = lp_anchor.shape[:2]
    sides = lp_anchor.shape[-1]  # len(ANCHOR_SIDE)
    flat = lp_anchor.flatten(1).argmax(1)
    i, rest = flat // (n * sides), flat % (n * sides)
    j, sign = rest // sides, rest % sides
    b = torch.arange(n_b, device=flat.device)
    pair, atom = dec.entry_logits(h, p, batch, torch.stack([b, i, j, sign], 1))
    r_pairs = pair.argmax(-1) - 1
    r_pairs = torch.where(_upper(n, flat.device), r_pairs, 0)
    r_pairs[b, i, j] = torch.where(sign == FORM, 1, -1)
    r_pairs = r_pairs + r_pairs.transpose(1, 2)
    real = _real_atoms(batch)
    r_pairs = r_pairs * (real[:, :, None] & real[:, None, :])
    values = device_index(tuple(dec.atom_values), atom.device)
    return r_pairs, values[atom.argmax(-1)] * real


def edit_matches(r_pairs: torch.Tensor, r_atoms: torch.Tensor, batch: dict) -> torch.Tensor:
    """(B,) bool: the predicted R equals a variant of the label class."""
    same = (batch["r_pairs"] == r_pairs[:, None]).flatten(2).all(-1)
    same &= (batch["r_atoms"] == r_atoms[:, None]).all(-1)
    return (same & batch["variant_mask"]).any(1)
