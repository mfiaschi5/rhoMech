"""The pool key: the sorted species keys of a pool joined by "." with repeats.

A species key is the identity key v2 of one fragment (`data.resonance.identity_key`), which
never holds a '.', so the join is unambiguous; it is not a SMILES, since a key ends in
`|<charge>` (or the open-shell suffix). The web app's pool ids hash this key
(`serve.core.keys.pool_id`), so its format is fixed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def pool_key_of(species_keys: Iterable[str]) -> str:
    """The pool key of a multiset of species keys."""
    return ".".join(sorted(species_keys))


def split_pool_key(pool_key: str) -> Counter[str]:
    """The multiset of species keys of a pool key."""
    return Counter(k for k in pool_key.split(".") if k)
