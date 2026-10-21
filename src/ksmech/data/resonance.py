"""Resonance keys: identify steps whose reactants and products are resonance structures.

`canonical_resonance_key(mol)` is the lexicographically smallest canonical SMILES over RDKit's
`ResonanceMolSupplier` structures of the molecule with its atom maps stripped, at most
`MAX_STRUCTURES` of them; when the enumeration hits the cap the key falls back to the plain
canonical SMILES and the `RESONANCE_TRUNCATED` flag is set. The enumeration runs on the
molecule re-read from its own map-free canonical SMILES, so the key depends neither on the
input's atom order nor on which equivalent atom carries a charge (RDKit's enumeration does
depend on atom order: phenoxide gave two keys; docs/decisions.md, 2026-10-22). Charges may
sit on any atom and charge separation is allowed, because PMechDB resonance steps move charge
onto carbon (enolate <-> carbanion, allyl cation shifts, C=O <-> C+-O-).

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


def _map_free(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    return m


def _canonical_form(mol: Chem.Mol) -> Chem.Mol | None:
    """The sanitized molecule rebuilt from its map-free canonical SMILES: canonical atom order
    and Kekule structure, whatever order and drawing it came in. Every atom keeps the H count
    it had (a re-read SMILES would let RDKit add implicit H in the enumerated structures: a
    terminal diazo N gained two)."""
    m = _sanitized_copy(_map_free(mol))
    if m is None:
        return None
    with rdBase.BlockLogs():
        out = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    if out is None:
        return None
    for atom in out.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
    return _sanitized_copy(out)


def resonance_structures(
    mol: Chem.Mol, max_structures: int = MAX_STRUCTURES
) -> tuple[set[str], int, ResonanceFlag]:
    """Map-free canonical SMILES of every resonance structure of `mol` (enumerated on its
    canonical form), the enumeration size and the flag. Truncated or failed: the plain SMILES."""
    m = _canonical_form(mol)
    if m is None:
        return {Chem.MolToSmiles(_map_free(mol))}, 0, ResonanceFlag.RESONANCE_FAILED
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(m, int(_FLAGS), max_structures)
        n = len(supplier)
        if n >= max_structures:
            return {Chem.MolToSmiles(m)}, n, ResonanceFlag.RESONANCE_TRUNCATED
        keys = {Chem.MolToSmiles(supplier[i]) for i in range(n) if supplier[i] is not None}
    if not keys:
        return {Chem.MolToSmiles(m)}, n, ResonanceFlag.RESONANCE_FAILED
    return keys, n, ResonanceFlag.OK


def canonical_resonance_key(mol: Chem.Mol, max_structures: int = MAX_STRUCTURES) -> ResonanceKey:
    """Smallest map-free canonical SMILES over the resonance structures of `mol`."""
    keys, n, flag = resonance_structures(mol, max_structures)
    return ResonanceKey(min(keys), n, flag)


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
    """True when the two sides share a resonance structure: their map-free enumerations
    intersect. RDKit does not always enumerate a charge separation from the neutral side
    (C=N-X -> C+-N(-)-X), so comparing the two minima would miss pairs that either side's
    enumeration connects. The flag reports a truncated or failed enumeration."""
    if skeleton_key(reactants) != skeleton_key(products):
        return False, ResonanceFlag.OK
    kr, _, fr = resonance_structures(reactants, max_structures)
    kp, _, fp = resonance_structures(products, max_structures)
    worst = next((f for f in (fr, fp) if f != ResonanceFlag.OK), ResonanceFlag.OK)
    return bool(kr & kp), worst


def training_steps(steps: list[ElementaryStep]) -> list[ElementaryStep]:
    """The training target: drop resonance steps and renumber `step_index` 0.. within each
    pathway (input order), then drop the steps outside the closed-shell domain
    (`domain_exclusion`, `data.domain`) without renumbering, so a pathway that lost one keeps
    the gap in its indices."""
    return [s for s in _renumbered(steps) if s.domain_exclusion is None]


def _renumbered(steps: list[ElementaryStep]) -> list[ElementaryStep]:
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
