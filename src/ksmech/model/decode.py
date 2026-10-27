"""The step decoder of the spec-v9 network (spec v9, "Decoding a step" and "Edit decoder": the
constrained beam search). B + R is checked by the caller's `apply`: `eval.decode.lewis_apply`
on a stored sample's Lewis structure, `serve.science.stepapply.apply_matrix` on a serve pool.

`decode_pool(model, sample, cfg, apply, scope, skips)` on one pool (the sample's global atoms):

1. Encode the pool once (`KSMech.encode`, CPU, no gradient) and take the anchor
   log-probabilities log P(e) over the candidates (`model.loss.anchor_log_probs`); scope
   "inter" keeps the anchors whose atoms lie in two molecules.
2. For each of the `cfg.anchors` most probable anchors e = (a, b, sign), the entry
   log-probabilities given e (`EditDecoder.entry_logits`, log-softmax over {-1, 0, +1} for a
   pair i < j, {-2, 0, +2} for an atom; the anchor's own entry is +1 for form, -1 for break).
3. log P(R) = log P(e) + sum over every other entry x of log P(R_x | e), zeros included:
   the background (every entry 0) plus, per changed entry, its gain log P(v) - log P(0). The
   candidate changes are the `max_candidates` non-zero classes with the largest gains. A
   best-first branch and bound over them (`ranked_completions`: include or exclude each
   candidate in gain order, the bound adding the best positive gains the remaining bond and
   atom capacity still admits) yields the completions with at most `max_bonds` bond (the
   anchor included: the PMechDB maximum) and `max_atoms` atom changes, one class per entry,
   in exact order of decreasing log P(R).
4. Each completion is checked in that order, and each drop is counted
   (`DECODE_DROPPED:<code>`): CHARGE_NOT_CONSERVED (sum R_ii + 2 sum_{i<j} R_ij != 0), then
   B + R by `apply(R)`, its first issue code
   (VALENCE_EXCEEDED, RESONANCE_ONLY_STEP, ...). `screen` also refuses TOO_MANY_CHANGES and
   ANCHOR_NOT_ADMISSIBLE (a breaking anchor with a +1 entry: such an R's anchors are its +1
   entries), which the search never proposes: a breaking anchor's candidates have no +1.
5. The `cfg.per_anchor` best distinct products of each anchor are kept; over all anchors the
   products are merged by their pool key (the sorted sigma keys of the product), the best
   log P kept (`DECODE_MERGED` counts the others); best first. An anchor with no valid
   completion within `max_pops` search nodes or `max_applied` applications is counted as
   `DECODE_ANCHOR_UNCOMPLETED`.

The END_MECH head is not read: on PMechDB it is untrained (lambda_stop = 0).
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

FORM, BREAK = 0, 1  # arrows.batches.ANCHOR_SIDE
NONMETALS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53})  # FORMAL_CHARGE_RANGE
SIGN = {FORM: "form", BREAK: "break"}
PAIR_VALUES = np.array([-1, 0, 1])
ATOM_VALUES = np.array([-2, 0, 2])
ZERO = 1  # the class index of 0 in both


@dataclass(frozen=True)
class DecodeConfig:
    anchors: int = 5  # B
    per_anchor: int = 5  # products kept per anchor
    max_bonds: int = 8  # bond changes of one R, the anchor included (C_max)
    max_atoms: int = 2  # atom changes of one R
    max_candidates: int = 48  # non-zero classes searched per anchor, by gain
    max_pops: int = 20_000  # search nodes per anchor
    max_applied: int = 64  # completions applied (B + R) per anchor


@dataclass(frozen=True)
class EntryTable:
    """One anchor's completion problem on the sample's atoms (float64 log-probabilities)."""

    anchor: tuple[int, int, int]  # (a, b, sign), a < b, sign FORM | BREAK
    anchor_logp: float
    pair_lp: np.ndarray  # (A, A, 3) over (-1, 0, +1); only i < j is read
    atom_lp: np.ndarray  # (A, 3) over (-2, 0, +2)


@dataclass(frozen=True)
class Completion:
    logp: float
    bonds: tuple[tuple[int, int, int], ...]  # (i, j, delta), i < j, the anchor included
    lone: tuple[tuple[int, int], ...]  # (i, delta)
    anchor: tuple[int, int, int]


# ---- the candidate changes of one anchor


@dataclass(frozen=True)
class Problem:
    """The entries of one table (pairs i < j without the anchor, then atoms) and the candidate
    changes, sorted by decreasing gain."""

    pi: np.ndarray  # (P,) first atom of each pair entry
    pj: np.ndarray  # (P,) second atom
    lp: np.ndarray  # (P + A, 3)
    base: float  # log P(e) + sum of every entry's log P(0)
    entry: np.ndarray  # (M,) the candidates' entries
    cls: np.ndarray  # (M,) their classes (0 or 2)
    gain: np.ndarray  # (M,) log P(class) - log P(0), descending
    is_bond: np.ndarray  # (M,) bool


def problem(t: EntryTable, max_candidates: int) -> Problem:
    a, b, _ = t.anchor
    n = t.atom_lp.shape[0]
    pi, pj = np.triu_indices(n, 1)
    keep = ~((pi == a) & (pj == b))
    pi, pj = pi[keep], pj[keep]
    lp = np.concatenate([t.pair_lp[pi, pj], t.atom_lp], 0).astype(np.float64)
    gain = lp - lp[:, ZERO : ZERO + 1]
    gain[:, ZERO] = -np.inf
    if t.anchor[2] == BREAK:  # a breaking anchor's R has no +1 entry (ANCHOR_NOT_ADMISSIBLE)
        gain[: len(pi), 2] = -np.inf
    entry, cls = np.nonzero(np.isfinite(gain))
    g = gain[entry, cls]
    order = np.lexsort((cls, entry, -g))[: max(0, int(max_candidates))]
    base = float(t.anchor_logp + lp[:, ZERO].sum())
    return Problem(pi, pj, lp, base, entry[order], cls[order], g[order], entry[order] < len(pi))


class _Bound:
    """The largest gain the candidates from position t on can still add with nb more bond and
    na more atom changes: the best positive gains of each kind (one class per entry relaxed)."""

    def __init__(self, pb: Problem) -> None:
        self.pos, self.prefix, self.npos = [], [], []
        for kind in (True, False):
            where = np.flatnonzero(pb.is_bond == kind)
            g = pb.gain[where]
            self.pos.append(where)
            self.prefix.append(np.concatenate([[0.0], np.cumsum(np.maximum(g, 0.0))]))
            self.npos.append(int((g > 0).sum()))

    def __call__(self, t: int, nb: int, na: int) -> float:
        out = 0.0
        for k, room in enumerate((nb, na)):
            i = int(np.searchsorted(self.pos[k], t))
            lo, hi = min(i, self.npos[k]), min(i + max(room, 0), self.npos[k])
            out += float(self.prefix[k][hi] - self.prefix[k][lo])
        return out


def ranked_completions(pb: Problem, n_bonds: int, n_atoms: int, max_pops: int) -> Iterator:
    """(log P, candidate positions) of the completions with at most `n_bonds` more bond and
    `n_atoms` atom changes and one class per entry, in decreasing log P: best-first over
    include / exclude of each candidate in gain order, keyed by the score plus `_Bound`
    (an upper bound, exact on a leaf), so leaves pop in exact order."""
    bound, m = _Bound(pb), len(pb.gain)
    heap = [(-(pb.base + bound(0, n_bonds, n_atoms)), 0, 0, pb.base, n_bonds, n_atoms, ())]
    for _ in range(int(max_pops)):
        if not heap:
            return
        _, _, t, score, nb, na, chosen = heapq.heappop(heap)
        if t == m:
            yield score, chosen
            continue
        heapq.heappush(
            heap, (-(score + bound(t + 1, nb, na)), -t - 1, t + 1, score, nb, na, chosen)
        )
        bond = bool(pb.is_bond[t])
        room = nb if bond else na
        used = pb.entry[list(chosen)]
        if room > 0 and pb.entry[t] not in used:
            s2, nb2, na2 = score + float(pb.gain[t]), nb - bond, na - (not bond)
            key = -(s2 + bound(t + 1, nb2, na2))
            heapq.heappush(heap, (key, -t - 1, t + 1, s2, nb2, na2, (*chosen, t)))


# ---- completions


def classes(pb: Problem, chosen: tuple[int, ...]) -> np.ndarray:
    """The class of every entry: 0 but on the chosen candidates."""
    cls = np.full(len(pb.lp), ZERO)
    cls[pb.entry[list(chosen)]] = pb.cls[list(chosen)]
    return cls


def _matrix(pb: Problem, cls: np.ndarray, anchor) -> tuple[tuple, tuple]:
    a, b, sign = anchor
    n_pairs = len(pb.pi)
    bonds = [(a, b, 1 if sign == FORM else -1)]
    lone = []
    for e in np.flatnonzero(cls != ZERO).tolist():
        if e < n_pairs:
            bonds.append((int(pb.pi[e]), int(pb.pj[e]), int(PAIR_VALUES[cls[e]])))
        else:
            lone.append((e - n_pairs, int(ATOM_VALUES[cls[e]])))
    return tuple(sorted(bonds)), tuple(sorted(lone))


def screen(bonds, lone, anchor, cfg: DecodeConfig) -> str | None:
    """The first rule a completion breaks before B + R (module docstring), or None."""
    if len(bonds) > cfg.max_bonds or len(lone) > cfg.max_atoms:
        return "TOO_MANY_CHANGES"
    if anchor[2] == BREAK and any(d > 0 for *_, d in bonds):
        return "ANCHOR_NOT_ADMISSIBLE"
    if sum(d for _, d in lone) + 2 * sum(d for *_, d in bonds) != 0:
        return "CHARGE_NOT_CONSERVED"
    return None


def exact_logp(pb: Problem, cls: np.ndarray, anchor_logp: float) -> float:
    """log P(e) + sum over every other entry of log P(its class), summed afresh."""
    return float(anchor_logp + pb.lp[np.arange(len(cls)), cls].sum())


Apply = Callable[[tuple, tuple], tuple[str | None, str | None]]  # -> (product key, issue)


def complete_anchor(t: EntryTable, apply: Apply, cfg: DecodeConfig, skips: Counter):
    """The best completions of one anchor with distinct products: {product key: Completion},
    in decreasing log P (module docstring, steps 3-5)."""
    pb = problem(t, cfg.max_candidates)
    out: dict[str, Completion] = {}
    applied = 0
    for _, chosen in ranked_completions(pb, cfg.max_bonds - 1, cfg.max_atoms, cfg.max_pops):
        cls = classes(pb, chosen)
        bonds, lone = _matrix(pb, cls, t.anchor)
        why = screen(bonds, lone, t.anchor, cfg)
        if why is None:
            key, why = apply(bonds, lone)
            applied += 1
            if key is not None and key in out:
                skips["DECODE_MERGED"] += 1
            elif key is not None:
                out[key] = Completion(exact_logp(pb, cls, t.anchor_logp), bonds, lone, t.anchor)
        if why is not None:
            skips[f"DECODE_DROPPED:{why}"] += 1
        if len(out) >= cfg.per_anchor or applied >= cfg.max_applied:
            break
    if not out:
        skips["DECODE_ANCHOR_UNCOMPLETED"] += 1
    return out


def merge(per_anchor: list[dict[str, Completion]], skips: Counter) -> list[Completion]:
    """Merged by product key over the anchors (the best log P kept), best first."""
    best: dict[str, Completion] = {}
    for found in per_anchor:
        for key, c in found.items():
            old = best.get(key)
            if old is not None:
                skips["DECODE_MERGED"] += 1
            if old is None or c.logp > old.logp:
                best[key] = c
    return sorted(best.values(), key=lambda c: (-c.logp, c.anchor, c.bonds, c.lone))


# ---- the model


def top_anchors(lp: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple]:
    """(log P, a, b, sign) of the `k` most probable finite anchors of an (A, A, 2) table,
    `allowed` (A, A) restricting the pairs; ties by index."""
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    if allowed is not None:
        lp = np.where(allowed[:, :, None], lp, -np.inf)
    flat = lp.ravel()
    finite = np.flatnonzero(np.isfinite(flat))
    order = finite[np.lexsort((finite, -flat[finite]))][: max(0, int(k))]
    n = lp.shape[1]
    return [(float(flat[f]), int(f // (2 * n)), int((f % (2 * n)) // 2), int(f % 2)) for f in order]


def encode_pool(model, sample):
    """(batch, h, p, log P(anchor) as an (A, A, 2) float64 array) of one sample on the CPU."""
    import torch

    from ksmech.arrows.batches import collate_samples
    from ksmech.model.inputs import to_device
    from ksmech.model.loss import anchor_log_probs

    batch = to_device(collate_samples([sample]), torch.device("cpu"))
    with torch.no_grad():
        h, p, y, _ = model.encode(batch)
        lp = anchor_log_probs(model.decoder.anchor_logits(h, p, y, batch))
    return batch, h, p, lp[0].double().numpy()


def entry_table(model, batch, h, p, anchor: tuple[float, int, int, int]) -> EntryTable:
    """The entry log-probabilities of sample 0 given one anchor (log P, a, b, sign)."""
    import torch

    logp, a, b, sign = anchor
    rows = torch.tensor([[0, a, b, sign]], dtype=torch.long)
    with torch.no_grad():
        pair, atom = model.decoder.entry_logits(h, p, batch, rows)
        pair_lp = torch.log_softmax(pair[0].double(), -1).numpy()
        atom_lp = torch.log_softmax(atom[0].double(), -1).numpy()
    return EntryTable((a, b, sign), float(logp), pair_lp, atom_lp)


def decode_pool(model, sample, cfg: DecodeConfig, apply: Apply, scope: str, skips: Counter):
    """The ranked distinct completions of one pool (module docstring)."""
    model.eval()
    batch, h, p, lp = encode_pool(model, sample)
    allowed = None
    if scope == "inter":
        frag = np.asarray(sample.atom_fragment)
        allowed = frag[:, None] != frag[None, :]
    found = []
    for anchor in top_anchors(lp, cfg.anchors, allowed):
        table = entry_table(model, batch, h, p, anchor)
        found.append(complete_anchor(table, apply, cfg, skips))
    return merge(found, skips)
