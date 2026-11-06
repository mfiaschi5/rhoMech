"""Spin for the QM layer (open-shell plan P4 steps 2 and 3): the tagged method of a species whose
spin `data.spin` reassigns.

`resolve_charge_spin` (RDKit) is unchanged: serve's pre-screens call it and refuse
RADICAL_NOT_SUPPORTED or SPIN_PARITY before any QM. The engine asks `spin_for_qm` instead. Where
`data.spin.assign` gives a count other than the drawing's (a singlet carbene or CO, a Cu(II) or
Fe(III) fragment that RDKit counts 0), the record is computed at the assigned count under the
method tagged `+spin-v1-u<u>` (`spin_tag`); everywhere else the method is untagged and every
existing cache key is unchanged. The tag names the procedure version and the count, so a later
table that changes one species' count re-keys that species alone, and no reader can take an
untagged record (the old triplet Cl[C]Cl) or one of another count for a tagged species.

The tag sits right after the optimizer part of the method and before any later suffix, so serve's
metal path composes '...tblite-lbfgs+spin-v1-u1+metal-geom-v1' (fixed now, for S1).

Every site that builds a cache path (records, tombstones, tokens, Wiberg sidecars) from a
configured method goes through `method_for` (checked by a grep test, tests/test_qm_spin.py).
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import TypeVar

from rdkit import Chem

from ksmech.data import spin

SPIN_TAG_VERSION = "spin-v1"  # tagged records: two starts, the lower optimum without a new bond
TAG_PREFIX = "+spin-"
C = TypeVar("C")


@lru_cache(maxsize=65536)
def spin_tag(smiles: str) -> str | None:
    """'spin-v1-u<u>' when `data.spin` reassigns the species of this map-free canonical SMILES,
    else None (also for a SMILES RDKit cannot parse: its calculation fails on its own reason)."""
    try:
        a = spin.assign_smiles(smiles)
    except ValueError:
        return None
    drawn = spin.rdkit_unpaired(Chem.MolFromSmiles(smiles))
    return f"{SPIN_TAG_VERSION}-u{a.unpaired}" if spin.reassigned(a, drawn) else None


def tagged(base: str, tag: str | None) -> str:
    """`base` with `tag` inserted after its optimizer part (before any '+' suffix)."""
    if tag is None or TAG_PREFIX in base:
        return base
    head, plus, rest = base.partition("+")
    return f"{head}+{tag}{plus}{rest}"


def method_for(smiles: str, base: str) -> str:
    """The method of one species' record and of every artifact keyed by it: `base` with the
    species' spin tag (idempotent: an already tagged `base` is returned as it is)."""
    return tagged(base, spin_tag(smiles))


def config_for(smiles: str, config: C) -> C:
    """`config` (a `QMConfig` or a subclass) with the species' spin tag set."""
    tag = spin_tag(smiles)
    if getattr(config, "spin_tag", None) == tag:
        return config
    return dataclasses.replace(config, spin_tag=tag)


def spin_for_qm(mol: Chem.Mol, smiles: str) -> tuple[int, int, str | None]:
    """(total charge, unpaired electrons, spin tag) of the molecule a calculation runs on:
    the assigned count of a reassigned species, else RDKit's (`resolve_charge_spin`, which
    raises SPIN_PARITY on an impossible pair: a species `data.spin` cannot assign)."""
    from ksmech.qm.records import QMError, ReasonCode
    from ksmech.qm.xtb_calc import resolve_charge_spin

    tag = spin_tag(smiles)
    if tag is None:
        return (*resolve_charge_spin(mol), None)
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    unpaired = int(spin.assign_smiles(smiles).unpaired)
    n_elec = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if n_elec < unpaired or (n_elec - unpaired) % 2:
        raise QMError(ReasonCode.SPIN_PARITY, f"{n_elec} electrons cannot have {unpaired} unpaired")
    return charge, unpaired, tag
