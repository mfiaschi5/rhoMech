"""Batches by a cell budget (spec v9, "Complexity"): the pair track of a batch costs
B x (N_max + 1)^2 cells, so a batch holds samples of similar size up to `cell_budget` padded
cells, rather than a fixed number of steps.

Each epoch the samples are shuffled, cut into pools of `pool` samples, sorted by size inside a
pool and packed greedily; the batches are then shuffled. The order depends only on the seed and
the epoch.
"""

from __future__ import annotations

import numpy as np
import torch


def cells(n_tokens: int) -> int:
    """Pair-track cells of one sample: its tokens and the MECH token, squared."""
    return (n_tokens + 1) ** 2


def pack(order: np.ndarray, sizes: np.ndarray, cell_budget: int, max_batch: int) -> list:
    """Greedy batches of `order` (already sorted by size) under the padded-cell budget."""
    batches, current, largest = [], [], 0
    for i in order:
        n = max(largest, int(sizes[i]))
        if current and (len(current) + 1) * cells(n) > cell_budget or len(current) == max_batch:
            batches.append(current)
            current, n = [], int(sizes[i])
        current.append(int(i))
        largest = n
    if current:
        batches.append(current)
    return batches


class CellBudgetSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        sizes,
        cell_budget: int,
        max_batch: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        pool: int = 1024,
    ):
        self.sizes = np.asarray(sizes, dtype=np.int64)
        too_big = int((self.sizes + 1).max(initial=0) ** 2)
        if too_big > cell_budget:
            raise ValueError(f"a sample needs {too_big} cells, over the budget {cell_budget}")
        self.cell_budget, self.max_batch = int(cell_budget), int(max_batch)
        self.shuffle, self.seed, self.pool, self.epoch = shuffle, seed, pool, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        if not self.shuffle:
            order = np.argsort(self.sizes, kind="stable")
            return pack(order, self.sizes, self.cell_budget, self.max_batch)
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.sizes))
        out = []
        for start in range(0, len(order), self.pool):
            chunk = order[start : start + self.pool]
            chunk = chunk[np.argsort(self.sizes[chunk], kind="stable")]
            out += pack(chunk, self.sizes, self.cell_budget, self.max_batch)
        return [out[k] for k in rng.permutation(len(out))]

    def __iter__(self):
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())
