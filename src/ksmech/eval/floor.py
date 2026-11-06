"""The physics floor (spec v9, "Evaluation protocol"): the anchor ranked by |G^atom_ij|, the
pooled gap factor over the valid orbital pairs (`model.inputs.pooled_gap`; over every pair with
`cells='all'`, spec v9.1), with no learned parameter.

Each candidate pair (i < j) is proposed once, with the sign form when it may form (its order
stays at most 3) and break otherwise (a triple bond); 94% of the PMechDB anchors form. A
break anchor on a pair that may also form is therefore out of the floor's reach; such steps
are counted as `unreachable` and scored as misses. Top-k uses set marginalization: a hit when
an admissible anchor of any variant of the label class is among the k best.

The second floor, `own_break` (open-shell plan P7), proposes a break anchor on every bonded pair,
scored by |B^own| (the own-pair gap factor, `model.inputs.pooled_own`), beside the form anchors
scored by |G^atom|: one kernel, one unit, one ranking, so a homolysis is reachable. `kernel`
'g2' pools the two-level column (g2, own_g2) in place of g. `drop_intra_open` leaves the
cells of an open-shell row or hole column within one fragment out of both poolings.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from ksmech.arrows.batches import collate_samples
from ksmech.model.decoder import BREAK, FORM
from ksmech.model.inputs import _pool, cell_mask, centre_weights

FLOORS = ("legacy", "own_break")
KERNELS = {"g": ("g", "own_g"), "g2": ("g2", "own_g2")}  # kernel -> (pooled, own) columns


def intra_open_cells(batch: dict) -> torch.Tensor:
    """(B, R, C) the cells of an open-shell row or a hole column within one fragment."""
    cols = tuple(batch["orbital_columns"])
    if "spin_role" not in cols:
        return torch.zeros_like(batch["pair_valid"])
    role = batch["orbital_features"][..., cols.index("spin_role")]
    n_atoms = batch["atom_features"].shape[1]

    def at(pos):
        idx = (pos - n_atoms).clamp(min=0)
        return torch.gather(role, 1, idx) * (pos >= 0)

    rows, cols_ = at(batch["pair_row_pos"]), at(batch["pair_col_pos"])
    inter = batch["pair_z"][..., tuple(batch["pair_features"]).index("inter")] > 0
    return ((rows[:, :, None] != 0) | (cols_[:, None, :] != 0)) & ~inter


def pooled_column(batch: dict, name: str, cells: torch.Tensor) -> torch.Tensor:
    """(B, A, A) the pair column `name` pooled like G^atom over the `cells` where it is
    defined."""
    k = tuple(batch["pair_features"]).index(name)
    x = torch.where(cells & batch["pair_defined"][..., k], batch["pair_z"][..., k].float(), 0.0)
    w_occ, w_virt = centre_weights(batch)
    return _pool(w_occ, x[..., None], w_virt)[..., 0]


def floor_scores(batch: dict, cells: str, kernel: str, drop_intra_open: bool):
    """(|G^atom|, |B^own|) on the atom pairs, both (B, A, A), under `kernel`: the pooled column
    over `cells` and the own column over every cell where it is defined, on bonded pairs."""
    pooled, own = KERNELS[kernel]
    seen = cell_mask(batch, cells)
    every = torch.ones_like(seen)
    if drop_intra_open:
        out = intra_open_cells(batch)
        seen, every = seen & ~out, every & ~out
    g_atom = pooled_column(batch, pooled, seen).abs()
    b_own = pooled_column(batch, own, every).abs() if own in batch["pair_features"] else None
    if b_own is None:
        b_own = torch.zeros_like(g_atom)
    return g_atom, b_own * (batch["bond_order"] > 0)


def _ranked(score, cand, hit) -> tuple[torch.Tensor, torch.Tensor]:
    """(rank, reachable) of the best hit among the candidates (ties count against)."""
    score = score.masked_fill(~cand, float("-inf"))
    best = score.masked_fill(~hit, float("-inf")).flatten(1).amax(1)
    shape = (-1,) + (1,) * (score.dim() - 1)
    rank = ((score >= best.view(shape)) & cand & ~hit).flatten(1).sum(1)
    reachable = hit.flatten(1).any(1)
    return torch.where(reachable, rank, cand.flatten(1).sum(1)), reachable


@torch.no_grad()
def floor_ranks(
    batch: dict,
    cells: str = "valid",
    floor: str = "legacy",
    kernel: str = "g",
    drop_intra_open: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(rank, reachable), both (B,): the 0-based rank of the best admissible anchor, and
    whether the floor proposes an admissible anchor at all (the rank is the number of
    candidates, a miss at every k, when it does not). `legacy`: each pair once, scored by
    |G^atom| pooled over the pairs of `cells`; `own_break`: form anchors by |G^atom| and break
    anchors on every bonded pair by |B^own| (module docstring)."""
    if floor not in FLOORS or kernel not in KERNELS:
        raise ValueError(f"floor {floor!r} / kernel {kernel!r}: {FLOORS} / {tuple(KERNELS)}")
    g_atom, b_own = floor_scores(batch, cells, kernel, drop_intra_open)
    n = g_atom.shape[1]
    upper = torch.ones(n, n, dtype=torch.bool).triu(1)
    form, brk = batch["can_form"] & upper, batch["can_break"] & upper
    admissible = (batch["r_anchors"] & batch["variant_mask"][:, :, None, None, None]).any(1)
    if floor == "legacy":
        cand = form | brk
        hit = torch.where(form, admissible[..., FORM], admissible[..., BREAK]) & cand
        return _ranked(g_atom, cand, hit)
    score = torch.stack([g_atom, b_own], -1)  # last axis FORM, BREAK
    cand = torch.stack([form, brk], -1)
    return _ranked(score, cand, admissible & cand)


def floor_report(samples: Iterable, batch_size: int = 16, cells: str = "valid", **floor) -> dict:
    """Top-1/3/5, the median rank and the unreachable count over `samples`; `per_sample`
    holds one row per sample (step id, 0-based rank, reachable) for paired comparisons.
    `floor`: the options of `floor_ranks` (floor, kernel, drop_intra_open)."""
    samples = list(samples)
    ranks, reach, unreachable = [], [], 0
    for start in range(0, len(samples), batch_size):
        batch = collate_samples(samples[start : start + batch_size])
        r, reachable = floor_ranks(batch, cells, **floor)
        unreachable += int((~reachable).sum())
        ranks.append(r.numpy())
        reach.append(reachable.numpy())
    r = np.concatenate(ranks) if ranks else np.zeros(0, dtype=np.int64)
    ok = np.concatenate(reach) if reach else np.zeros(0, dtype=bool)
    out = {f"top{k}": float((r < k).mean()) if len(r) else 0.0 for k in (1, 3, 5)}
    rows = [
        {"step_id": getattr(s, "step_id", None), "rank": int(k), "reachable": bool(a)}
        for s, k, a in zip(samples, r, ok, strict=True)
    ]
    return out | {
        "median_rank": float(np.median(r)) if len(r) else 0.0,
        "n": len(r),
        "unreachable": unreachable,
        "per_sample": rows,
    }
