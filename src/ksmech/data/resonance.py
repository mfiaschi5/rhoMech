"""Resonance keys: identify steps whose reactants and products are resonance structures.

`canonical_resonance_key(mol)` is the lexicographically smallest canonical SMILES over RDKit's
`ResonanceMolSupplier` structures of the (sanitized, atom-mapped) molecule, at most
`MAX_STRUCTURES` of them; when the enumeration hits the cap the key falls back to the plain
canonical SMILES and the `RESONANCE_TRUNCATED` flag is set. Charges may sit on any atom and
charge separation is allowed, because PMechDB resonance steps move charge onto carbon
(enolate <-> carbanion, allyl cation shifts, C=O <-> C+-O-).

Resonance structures share the sigma skeleton, so `skeleton_key` (every bond single, charges
zero, maps and explicit H kept) is a cheap necessary condition: two sides with different
skeletons cannot share a resonance key and the enumeration is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from rdkit import Chem, rdBase
from rdkit.Chem import rdchem

from ksmech.data.schema import ElementaryStep

MAX_STRUCTURES = 1000
_FLAGS = (
    rdchem.ResonanceFlags.UNCONSTRAINED_CATIONS
    | rdchem.ResonanceFlags.UNCONSTRAINED_ANIONS
    | rdchem.ResonanceFlags.ALLOW_CHARGE_SEPARATION
)


class ResonanceFlag(StrEnum):
    OK = "OK"
    RESONANCE_TRUNCATED = "RESONANCE_TRUNCATED"  # cap hit: key is the plain canonical SMILES
    RESONANCE_FAILED = "RESONANCE_FAILED"  # sanitization failed: key is the plain canonical SMILES


@dataclass(frozen=True)
class ResonanceKey:
    key: str
    n_structures: int
    flag: ResonanceFlag


def _sanitized_copy(mol: Chem.Mol) -> Chem.Mol | None:
    m = Chem.Mol(mol)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(m)
    except Chem.rdchem.MolSanitizeException:
        return None
    return m


def canonical_resonance_key(mol: Chem.Mol, max_structures: int = MAX_STRUCTURES) -> ResonanceKey:
    """Smallest canonical SMILES over the resonance structures of `mol` (atom maps kept)."""
    plain = Chem.MolToSmiles(mol)
    m = _sanitized_copy(mol)
    if m is None:
        return ResonanceKey(plain, 0, ResonanceFlag.RESONANCE_FAILED)
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(m, int(_FLAGS), max_structures)
        n = len(supplier)
        if n >= max_structures:
            return ResonanceKey(Chem.MolToSmiles(m), n, ResonanceFlag.RESONANCE_TRUNCATED)
        keys = [Chem.MolToSmiles(supplier[i]) for i in range(n) if supplier[i] is not None]
    if not keys:
        return ResonanceKey(Chem.MolToSmiles(m), n, ResonanceFlag.RESONANCE_FAILED)
    return ResonanceKey(min(keys), n, ResonanceFlag.OK)


def skeleton_key(mol: Chem.Mol) -> str:
    """Canonical SMILES with every bond single and every charge zero: the sigma skeleton."""
    m = Chem.RWMol(mol)
    for bond in m.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(True)
    m.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(m, canonical=True)


def is_resonance_step(
    reactants: Chem.Mol, products: Chem.Mol, max_structures: int = MAX_STRUCTURES
) -> tuple[bool, ResonanceFlag]:
    """True when both sides share a resonance key. The flag reports a truncated enumeration."""
    if skeleton_key(reactants) != skeleton_key(products):
        return False, ResonanceFlag.OK
    kr = canonical_resonance_key(reactants, max_structures)
    kp = canonical_resonance_key(products, max_structures)
    worst = ResonanceFlag.OK
    for flag in (kr.flag, kp.flag):
        if flag != ResonanceFlag.OK:
            worst = flag
    return kr.key == kp.key, worst


def training_steps(steps: list[ElementaryStep]) -> list[ElementaryStep]:
    """Drop resonance steps and renumber `step_index` 0.. within each pathway (input order)."""
    kept = [s for s in steps if not s.resonance_step]
    counters: dict[str, int] = {}
    out = []
    for s in kept:
        if s.pathway_id is None or s.step_index is None:
            out.append(s)
            continue
        idx = counters.get(s.pathway_id, 0)
        counters[s.pathway_id] = idx + 1
        out.append(s if s.step_index == idx else replace(s, step_index=idx))
    return out
