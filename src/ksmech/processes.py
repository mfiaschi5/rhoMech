"""Spawned worker processes for the builders and scorers (token build, alignment, samples,
`ksmech-physics`): one BLAS / OpenMP thread each, results in task order.

It imports nothing from ksmech, so every layer can use it. A task's arguments and results
cross the process boundary by pickle: an exception raised in a worker must rebuild from its
pickle (`provenance.StaleArtifact.__reduce__`), or `multiprocessing.Pool` hangs in the parent.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing import get_context

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@contextmanager
def single_threaded_children():
    """Spawned workers start with one BLAS / OpenMP thread each. Without it every worker's
    numpy fans out over all cores (a 3-worker build ran ~15 busy cores and, next to the QM
    batch, hard-rebooted the host; see docs/decisions.md)."""
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def chunked(items: Sequence, size: int) -> list:
    """Consecutive slices of `items`, `size` long (the last one shorter)."""
    return [items[k : k + size] for k in range(0, len(items), size)]


def in_processes(fn: Callable, tasks: Iterable, n_workers: int, chunksize: int = 1) -> Iterator:
    """`fn` over `tasks` in task order: in this process when `n_workers` <= 1, else in that
    many spawned single-threaded processes (`fn` must be importable at module level)."""
    tasks = list(tasks)
    if n_workers <= 1:
        yield from map(fn, tasks)
        return
    with single_threaded_children(), get_context("spawn").Pool(n_workers) as pool:
        yield from pool.imap(fn, tasks, chunksize=chunksize)
