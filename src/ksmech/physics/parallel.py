"""Sample-level work of `ksmech-physics` in spawned processes.

A part (the val view of a split, say) is cut into chunks of consecutive step ids in the order
of its view, and each chunk is a task: the worker loads its own samples and their contact
entries by id, so no sample crosses a process boundary on the way in, and only what the parent
keeps comes back. `pool.imap` returns the chunks in order, so a parallel run gives the parent
the same sequence as a single process, and the reports are identical. Workers start with one
BLAS / OpenMP thread each (`ksmech.processes.in_processes`); a `TokenCache` sent to
a worker starts empty.
"""

from __future__ import annotations

from pathlib import Path

from ksmech.arrows import contacts
from ksmech.arrows.steps import load_samples

CHUNK = 64  # consecutive samples per task


def joined_samples(
    path: Path, view_dir: Path, split: str, ids: list[str], allow_stale: bool
) -> tuple[list, int]:
    """The samples of `ids` that have a legacy contact entry and ArrowLabels, joined with the
    entry (`contacts.with_legacy`), in view order; and how many of `ids` had an entry."""
    legacy = contacts.load_contacts(view_dir, split, allow_stale, ids)
    samples = load_samples(path, allow_stale, ids=legacy)
    joined = [contacts.with_legacy(s, legacy[s.step_id]) for s in samples if s.labels]
    return joined, len(samples)


def scored_chunk(task: tuple) -> tuple[list, dict, int]:
    """ks-rank's work on one chunk: the joined samples with their validity refreshed, the
    components of each under every kernel ({distance: [...]}) and the number with an entry."""
    from ksmech.physics.ranking import pair_components, refresh_validity

    path, view_dir, split, ids, allow_stale, loader, kernels = task
    samples, n_with_entry = joined_samples(path, view_dir, split, ids, allow_stale)
    samples = [refresh_validity(s, loader) for s in samples]
    comps = {d: [pair_components(s, loader, k) for s in samples] for d, k in kernels.items()}
    return samples, comps, n_with_entry
