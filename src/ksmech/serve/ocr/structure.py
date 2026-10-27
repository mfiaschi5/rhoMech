"""The engine's answer checked by RDKit: the SMILES and molfile `OcrOut` returns.

MolScribe answers a SMILES (its abbreviations, such as Ph or OMe, expanded) and a molfile of
the atoms as drawn (written before the expansion, so an abbreviation is one pseudo-atom). Its
molfile coordinates are fractions of the image width and height with y already pointing up.
In order:

1. the molfile parses and has the SMILES's structure: its drawn layout is kept, the fractions
   scaled back by the image's width and height (the drawing's aspect ratio), then to a median
   bond length of `BOND_LENGTH` (1.5 Å) about the origin;
2. the SMILES parses: that structure, laid out by RDKit, with a warning (the drawing held an
   abbreviation, or its molfile did not parse);
3. only the molfile parses: its structure and layout, with a warning;
4. the molfile reads without sanitization (a valence RDKit refuses): returned as drawn, with a
   warning that names RDKit's reason;
5. nothing parses: OCR_FAILED (422).

The SMILES out is RDKit's canonical SMILES of the returned structure; the molfile is V2000.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.ocr.engine import RawOcr

BOND_LENGTH = 1.5
INVALID_SMILES = "<invalid>"  # MolScribe's SMILES when its graph does not convert
FRACTION_MAX = 1.5  # coordinates within this bound are image fractions, not pixels


@dataclass(frozen=True)
class Structure:
    smiles: str
    molfile: str
    warnings: list[str]


def _quiet():
    """RDKit's log muted until the returned object is dropped (the reasons go into the
    warnings, not the server log)."""
    from rdkit.rdBase import BlockLogs

    return BlockLogs()


def _mol_from_block(block: str | None, sanitize: bool = True):
    from rdkit import Chem

    if not block or not block.strip():
        return None
    try:
        return Chem.MolFromMolBlock(block, sanitize=sanitize, removeHs=False)
    except Exception:  # noqa: BLE001 - RDKit raises several types on malformed blocks
        return None


def _mol_from_smiles(smiles: str | None):
    from rdkit import Chem

    if not smiles or not smiles.strip() or smiles.strip() == INVALID_SMILES:
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:  # noqa: BLE001
        return None


def _canonical(mol) -> str | None:
    from rdkit import Chem

    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
    except Exception:  # noqa: BLE001
        return None


def _bond_median(xy: np.ndarray, mol) -> float | None:
    lengths = [
        float(np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]))
        for b in mol.GetBonds()
    ]
    lengths = [x for x in lengths if x > 1e-9]
    return float(np.median(lengths)) if lengths else None


def drawn_layout(mol, width: int | None, height: int | None) -> None:
    """The drawn coordinates of `mol`'s conformer, in place: image fractions back to the image's
    aspect ratio, then scaled to a median bond of `BOND_LENGTH` and centred on the origin."""
    conf = mol.GetConformer()
    xy = np.asarray(conf.GetPositions(), dtype=float).reshape(-1, 3)[:, :2]
    if xy.size and np.abs(xy).max() <= FRACTION_MAX and width and height:
        xy = xy * np.array([float(width), float(height)])
    median = _bond_median(xy, mol)
    if median:
        xy = xy * (BOND_LENGTH / median)
    if xy.size:
        xy = xy - (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    for i, (x, y) in enumerate(xy.tolist()):
        conf.SetAtomPosition(i, (x, y, 0.0))
    conf.Set3D(False)


def rdkit_layout(mol) -> None:
    """A fresh RDKit 2D layout (RDKit's own bond length is 1.5 Å)."""
    from rdkit.Chem import rdDepictor

    rdDepictor.Compute2DCoords(mol)


def _finish(mol, notes: list[str], sanitized: bool = True) -> Structure:
    from rdkit import Chem

    try:
        block = Chem.MolToMolBlock(mol, kekulize=sanitized)
        smiles = Chem.MolToSmiles(mol) if sanitized else Chem.MolToSmiles(mol, canonical=False)
    except Exception as exc:  # noqa: BLE001
        raise ServeError(Code.OCR_FAILED, f"RDKit cannot write the structure: {exc}") from exc
    return Structure(smiles, block, notes)


def _unsanitized(raw: RawOcr, notes: list[str]) -> Structure | None:
    from rdkit import Chem

    mol = _mol_from_block(raw.molfile, sanitize=False)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol.UpdatePropertyCache(strict=False)
    failed = Chem.SanitizeMol(mol, catchErrors=True)
    notes.append(f"the structure is not valid for RDKit ({failed}): returned as read, unchecked")
    drawn_layout(mol, raw.width, raw.height)
    return _finish(mol, notes, sanitized=False)


def structure_of(raw: RawOcr) -> Structure:
    """What `OcrOut` returns for the engine's answer (module docstring)."""
    muted = _quiet()  # noqa: F841 - held for the duration of the call
    notes: list[str] = []
    drawn, parsed = _mol_from_block(raw.molfile), _mol_from_smiles(raw.smiles)
    if drawn is not None and drawn.GetNumAtoms() == 0:
        drawn = None
    if drawn is not None and (parsed is None or _canonical(drawn) == _canonical(parsed)):
        if parsed is None:
            notes.append(
                "the engine's SMILES does not parse: this one is RDKit's, from its molfile"
            )
        drawn_layout(drawn, raw.width, raw.height)
        return _finish(drawn, notes)
    if parsed is not None:
        why = "its molfile does not parse" if drawn is None else "the drawing holds abbreviations"
        notes.append(f"the 2D layout is RDKit's, not the drawing's: {why}")
        rdkit_layout(parsed)
        return _finish(parsed, notes)
    found = _unsanitized(raw, notes)
    if found is not None:
        return found
    detail = {"smiles": (raw.smiles or "")[:500], "has_molfile": bool(raw.molfile)}
    msg = "no structure could be read from the image (RDKit parses nothing the engine returned)"
    raise ServeError(Code.OCR_FAILED, msg, detail=detail)
