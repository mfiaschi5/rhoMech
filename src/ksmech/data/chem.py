"""RDKit helpers for atom-mapped step SMILES.

Mapped SMILES are parsed with every explicit H kept and **without aromaticity perception**, so
the Kekulé structure written by the curator survives. Arrows refer to specific single and double
bonds, and resonance steps only move pi bonds, so we cannot let RDKit pick another Kekulé form.
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem, rdBase
from rdkit.Chem import rdmolops

_SANITIZE_NO_AROMATICITY = (
    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
)


class SmilesParseError(ValueError):
    """RDKit could not read the SMILES string at all."""


class SanitizeError(ValueError):
    """RDKit read the SMILES but rejected the chemistry (valence, kekulization)."""


def parse_mapped(smiles: str) -> Chem.Mol:
    """Parse a (possibly multi-fragment) atom-mapped SMILES. Keep all H atoms and the Kekulé form.

    Aromatic input is kekulized with RDKit's own choice of Kekulé structure, and the aromatic
    flags are then cleared.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    with rdBase.BlockLogs():  # failures are re-raised with the RDKit message; no stderr spam
        mol = Chem.MolFromSmiles(smiles, params)
        if mol is None or mol.GetNumAtoms() == 0:
            raise SmilesParseError(f"RDKit cannot parse {smiles!r}")
        try:
            Chem.SanitizeMol(mol, _SANITIZE_NO_AROMATICITY)
            if any(a.GetIsAromatic() for a in mol.GetAtoms()):
                Chem.Kekulize(mol, clearAromaticFlags=True)
        except Chem.rdchem.MolSanitizeException as exc:
            raise SanitizeError(f"{type(exc).__name__}: {exc}") from exc
    return mol


def canonical_mapped_smiles(mol: Chem.Mol) -> str:
    """RDKit canonical SMILES with atom maps, explicit H atoms and the Kekulé form preserved."""
    return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)


def canonicalize_mapped(smiles: str) -> str:
    return canonical_mapped_smiles(parse_mapped(smiles))


def map_to_index(mol: Chem.Mol) -> dict[int, int]:
    """Atom-map number -> atom index for mapped atoms. Duplicates keep the last index."""
    return {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}


def map_numbers(mol: Chem.Mol) -> list[int]:
    return [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()]


def heavy_composition(mol: Chem.Mol) -> Counter[str]:
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def hydrogen_count(mol: Chem.Mol) -> int:
    """Explicit H atoms plus implicit/explicit H counts on every atom."""
    return sum((a.GetAtomicNum() == 1) + a.GetTotalNumHs() for a in mol.GetAtoms())


def formal_charge(mol: Chem.Mol) -> int:
    return rdmolops.GetFormalCharge(mol)


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def fragments(mol: Chem.Mol) -> list[Chem.Mol]:
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def map_to_fragment(mol: Chem.Mol) -> dict[int, int]:
    """Atom-map number -> index of the connected fragment that holds it."""
    out = {}
    for frag_idx, atom_ids in enumerate(Chem.GetMolFrags(mol)):
        for i in atom_ids:
            num = mol.GetAtomWithIdx(i).GetAtomMapNum()
            if num:
                out[num] = frag_idx
    return out


def unmapped_canonical(mol: Chem.Mol) -> str:
    """Map-free, H-suppressed, aromatic canonical SMILES. Use it for molecule identity."""
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    smi = Chem.MolToSmiles(m)
    with rdBase.BlockLogs():  # silences "not removing hydrogen atom without neighbors" ([H+])
        std = Chem.MolFromSmiles(smi)
    if std is None:
        raise SanitizeError(f"cannot standardize unmapped SMILES {smi!r}")
    return Chem.MolToSmiles(std)
