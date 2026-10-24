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

Every enumeration runs with the hydrogens frozen (`frozen_hydrogens`) and keeps only the forms
that have the input's H count on every atom, its total charge and its radical electrons
(`resonance_forms`). RDKit re-derives the implicit H of an unbracketed atom in each structure
(a nitro O came out as [OH2+]) and also returns structures with unpaired electrons
(CCN([O+])[O-]); neither is a resonance form.
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
_KEKULE_OPS = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY


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


def frozen_hydrogens(mol: Chem.Mol, aromaticity: bool = False) -> Chem.Mol | None:
    """A copy in which every atom keeps the H count it has (explicit H = total H, no implicit
    H), sanitized again, since the enumeration needs the conjugation sanitization perceives;
    None when it does not sanitize. Aromaticity is perceived only when asked: on a Kekule
    structure it changes which structures RDKit returns, and the edit labels read bond orders."""
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    for atom in m.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
    ops = Chem.SanitizeFlags.SANITIZE_ALL if aromaticity else _KEKULE_OPS
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(m, sanitizeOps=ops)
    except Chem.rdchem.MolSanitizeException:
        return None
    return m


def _electron_state(mol: Chem.Mol) -> tuple[tuple[int, ...], int, int]:
    """(H count of every atom, total charge, radical electrons): what a resonance form keeps."""
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    atoms = list(m.GetAtoms())
    return (
        tuple(a.GetTotalNumHs() for a in atoms),
        sum(a.GetFormalCharge() for a in atoms),
        sum(a.GetNumRadicalElectrons() for a in atoms),
    )


def resonance_forms(
    mol: Chem.Mol, max_structures: int, read_truncated: bool = True
) -> tuple[list[Chem.Mol], int]:
    """The resonance forms of `mol` (its H frozen by the caller: `frozen_hydrogens`) among the
    first `max_structures` structures RDKit enumerates, keeping only those with the electron
    state of `mol`; and the number of structures RDKit counted. When that reaches
    `max_structures` the enumeration is truncated, and with `read_truncated=False` no form is
    read."""
    ref = _electron_state(mol)
    with rdBase.BlockLogs():
        supplier = rdchem.ResonanceMolSupplier(mol, int(_FLAGS), max_structures)
        n = len(supplier)
        if n >= max_structures and not read_truncated:
            return [], n
        raw = [supplier[k] for k in range(min(n, max_structures))]
    return [f for f in raw if f is not None and _electron_state(f) == ref], n


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
    return frozen_hydrogens(out, aromaticity=True)


def resonance_structures(
    mol: Chem.Mol, max_structures: int = MAX_STRUCTURES
) -> tuple[set[str], int, ResonanceFlag]:
    """Map-free canonical SMILES of every resonance form of `mol` (enumerated on its
    canonical form, `resonance_forms`) and of that form itself, the enumeration size and the
    flag. Truncated or failed: the plain SMILES."""
    m = _canonical_form(mol)
    if m is None:
        return {Chem.MolToSmiles(_map_free(mol))}, 0, ResonanceFlag.RESONANCE_FAILED
    forms, n = resonance_forms(m, max_structures, read_truncated=False)
    if n >= max_structures:
        return {Chem.MolToSmiles(m)}, n, ResonanceFlag.RESONANCE_TRUNCATED
    return {Chem.MolToSmiles(f) for f in forms} | {Chem.MolToSmiles(m)}, n, ResonanceFlag.OK


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
