"""The one seam to upstream modules that moved since the app was written (spec v8 port).

`physics/oriented.py` became `physics/contact/legacy_fixed23.py` (the frozen oriented-contact
baseline). Serve code and its tests reach it only through `oriented_module()`, and the
fingerprints read its constants from `ORIENTED_SOURCE`, so a further move is one edit here.
The module is imported lazily: it pulls in the GTO integrals, which the API process does not
need at import time.
"""

from __future__ import annotations

import importlib
from types import ModuleType

ORIENTED_MODULE = "ksmech.physics.contact.legacy_fixed23"
ORIENTED_SOURCE = "physics/contact/legacy_fixed23.py"  # relative to src/ksmech


def oriented_module() -> ModuleType:
    """The upstream oriented-contact module (`ORIENTED_VERSION`, `cap_atoms`, `block_contacts`,
    `oriented_block`, the private placement helpers and its in-memory caches)."""
    return importlib.import_module(ORIENTED_MODULE)
