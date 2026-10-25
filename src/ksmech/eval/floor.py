"""The physics floor (spec v9, "Evaluation protocol"): the anchor ranked by |G^atom_ij|, the
pooled gap factor over the valid orbital pairs (`model.inputs.pooled_gap`), with no learned
parameter.

Each candidate pair (i < j) is proposed once, with the sign form when it may form (its order
stays at most 3) and break otherwise (a triple bond); 94% of the PMechDB anchors form. A
break anchor on a pair that may also form is therefore out of the floor's reach; such steps
are counted as `unreachable` and scored as misses. Top-k uses set marginalization: a hit when
an admissible anchor of any variant of the label class is among the k best.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.model.decoder import BREAK, FORM
from ksmech.model.inputs import pooled_gap


@torch.no_grad()
def floor_ranks(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(rank, reachable), both (B,): the 0-based rank of the best admissible anchor under
    |G^atom|, and whether the floor proposes an admissible anchor at all (the rank is the
    number of candidates, a miss at every k, when it does not)."""
    score = pooled_gap(batch).abs()
    n = score.shape[1]
    upper = torch.ones(n, n, dtype=torch.bool).triu(1)
    form, brk = batch["can_form"] & upper, batch["can_break"] & upper
    cand = form | brk
    score = score.masked_fill(~cand, float("-inf"))
    admissible = (batch["r_anchors"] & batch["variant_mask"][:, :, None, None, None]).any(1)
    hit = torch.where(form, admissible[..., FORM], admissible[..., BREAK]) & cand
    best = score.masked_fill(~hit, float("-inf")).flatten(1).amax(1)
    rank = (score.flatten(1) > best[:, None]).sum(1)
    reachable = hit.flatten(1).any(1)
    return torch.where(reachable, rank, cand.flatten(1).sum(1)), reachable


def floor_report(samples: Iterable, batch_size: int = 16) -> dict:
    """Top-1/3/5, the median rank and the unreachable count over `samples`."""
    samples = list(samples)
    ranks, unreachable = [], 0
    for start in range(0, len(samples), batch_size):
        batch = collate_samples(samples[start : start + batch_size])
        r, reachable = floor_ranks(batch)
        unreachable += int((~reachable).sum())
        ranks.append(r.numpy())
    r = np.concatenate(ranks) if ranks else np.zeros(0, dtype=np.int64)
    out = {f"top{k}": float((r < k).mean()) if len(r) else 0.0 for k in (1, 3, 5)}
    return out | {
        "median_rank": float(np.median(r)) if len(r) else 0.0,
        "n": len(r),
        "unreachable": unreachable,
    }
