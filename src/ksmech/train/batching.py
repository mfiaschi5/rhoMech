"""Batches by a cell budget (spec v9, "Complexity"): the pair track of a batch costs
B x (N_max + 1)^2 cells, so a batch holds samples of similar size up to `cell_budget` padded
cells, rather than a fixed number of steps.

Each epoch the samples are shuffled, cut into pools of `pool` samples, sorted by size inside a
pool and packed greedily; the batches are then shuffled. With `drawings` (the resonance drawings
of every sample) each sample of a shuffled epoch is also drawn in a random one of them (spec v9,
"Resonance augmentation"): the batches then hold (index, drawing) keys, which `DrawnSamples`
serves. The order and the drawings depend only on the seed and the epoch, and they are chosen
in the main process, so persistent DataLoader workers see every epoch's choice.
"""

from __future__ import annotations

import numpy as np
import torch

from ksmech.arrows.store import dense


def cells(n_tokens: int) -> int:
    """Pair-track cells of one sample: its tokens and the MECH token, squared."""
    return (n_tokens + 1) ** 2


def pack(order: np.ndarray, sizes: np.ndarray, cell_budget: int, max_batch: int) -> list:
    """Greedy batches of `order` (already sorted) under the padded-cell budget. `sizes` is the
    total token count of each sample, or (atoms, orbitals) per sample: the collate pads the
    atoms and the orbitals apart, so a batch then costs len x (max atoms + max orbitals + 1)^2,
    the cells it really holds."""
    sizes = np.asarray(sizes)
    split = sizes.ndim == 2
    batches, current = [], []
    big = np.zeros(2 if split else 1, dtype=np.int64)
    for i in order:
        n = np.maximum(big, sizes[i])
        if current and ((len(current) + 1) * cells(int(n.sum())) > cell_budget
                        or len(current) == max_batch):  # fmt: skip
            batches.append(current)
            current, n = [], np.atleast_1d(sizes[i]).astype(np.int64)
        current.append(int(i))
        big = n
    if current:
        batches.append(current)
    return batches


def size_order(sizes: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """`chunk` sorted by size: the total, or with (atoms, orbitals) by orbitals, then atoms.
    That order wastes the fewest padded cells (`docs/model_impl.md`, item 8: 89-93% of the
    cells real, against 87-91% by the total)."""
    if sizes.ndim == 1:
        return chunk[np.argsort(sizes[chunk], kind="stable")]
    return chunk[np.lexsort((sizes[chunk, 0], sizes[chunk, 1]))]


class DrawnSamples(torch.utils.data.Dataset):
    """Samples by index, or by (index, k) in their k-th resonance drawing
    (`MechanismSample.drawn`). A light sample with compact cells is made whole here, in the
    DataLoader worker (`store.dense`), so only the batch being collated is dense."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, k = key
            return dense(self.samples[i]).drawn(k)
        return dense(self.samples[key])


class CellBudgetSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        sizes,
        cell_budget: int,
        max_batch: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        pool: int = 1024,
        drawings=None,
    ):
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.drawings = None if drawings is None else np.asarray(drawings, dtype=np.int64)
        totals = self.sizes.sum(1) if self.sizes.ndim == 2 else self.sizes
        too_big = int((totals + 1).max(initial=0) ** 2)
        if too_big > cell_budget:
            raise ValueError(f"a sample needs {too_big} cells, over the budget {cell_budget}")
        self.cell_budget, self.max_batch = int(cell_budget), int(max_batch)
        self.shuffle, self.seed, self.pool, self.epoch = shuffle, seed, pool, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        if not self.shuffle:
            order = size_order(self.sizes, np.arange(len(self.sizes)))
            return pack(order, self.sizes, self.cell_budget, self.max_batch)
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.sizes))
        out = []
        for start in range(0, len(order), self.pool):
            chunk = size_order(self.sizes, order[start : start + self.pool])
            out += pack(chunk, self.sizes, self.cell_budget, self.max_batch)
        out = [out[k] for k in rng.permutation(len(out))]
        if self.drawings is None:
            return out
        return [[(i, int(rng.integers(self.drawings[i]))) for i in b] for b in out]

    def __iter__(self):
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())
