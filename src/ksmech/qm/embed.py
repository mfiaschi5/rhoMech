"""3D embedding: ETKDGv3 with explicit hydrogens, force-field pre-relaxation, RMSD pruning.

Atom order is never changed, so atom-map numbers set on the input survive. `embed` works in place
on the mol (the RDKit idiom) and returns the conformer ids. Fallbacks it took are recorded in the
mol property `ksmech_embed_notes` (comma-separated `ReasonCode` warning values).
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from ksmech.qm.records import QMError, ReasonCode

log = logging.getLogger(__name__)

NOTES_PROP = "ksmech_embed_notes"
PRUNE_RMSD = 0.5  # Angstrom, heavy atoms
_FF_MAX_ITERS = 2000


def _add_note(mol: Chem.Mol, code: ReasonCode) -> None:
    notes = [
        n for n in (mol.GetProp(NOTES_PROP) if mol.HasProp(NOTES_PROP) else "").split(",") if n
    ]
    mol.SetProp(NOTES_PROP, ",".join([*notes, str(code)]))


def prepare_mol(smiles: str) -> Chem.Mol:
    """The molecule for a canonical map-free SMILES, hydrogens explicit.

    This defines the atom order of every `QMRecord`: heavy atoms in SMILES order, then hydrogens
    as `Chem.AddHs` appends them. Parsing is deterministic, so the order is reproducible.
    """
    with rdBase.BlockLogs():  # silences "not removing hydrogen atom without neighbors" ([H+])
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise QMError(ReasonCode.EMBED_FAILED, f"RDKit cannot parse {smiles!r}")
    return Chem.AddHs(mol)


def has_implicit_hydrogens(mol: Chem.Mol) -> bool:
    return any(a.GetNumImplicitHs() > 0 for a in mol.GetAtoms())


def _generate(mol: Chem.Mol, n_confs: int, seed: int) -> list[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = PRUNE_RMSD
    params.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:  # hard cases (metal centres, strained rings): retry from random coordinates
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
        if cids:
            _add_note(mol, ReasonCode.EMBED_RANDOM_COORDS)
    if not cids:  # inconsistent bounds (a metal chelate ring, an aryne): ignore the smoothing
        params.ignoreSmoothingFailures = True
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = False
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
        if cids:
            _add_note(mol, ReasonCode.EMBED_RELAXED_BOUNDS)
    return cids


def _relax(mol: Chem.Mol, cids: list[int]) -> dict[int, float]:
    """Relax every conformer with MMFF (UFF if MMFF lacks parameters). Returns energies."""
    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=_FF_MAX_ITERS, numThreads=1)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        _add_note(mol, ReasonCode.EMBED_UFF_FALLBACK)
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=_FF_MAX_ITERS, numThreads=1)
    else:
        _add_note(mol, ReasonCode.EMBED_UFF_FALLBACK)
        log.info("no force-field parameters; keeping raw ETKDG coordinates")
        return {c: 0.0 for c in cids}
    return {c: float(e) for c, (_, e) in zip(cids, res, strict=True)}


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of two (n, 3) coordinate sets after optimal rigid superposition."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, s, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    s[-1] *= sign
    return float(np.sqrt(max((a**2).sum() + (b**2).sum() - 2 * s.sum(), 0.0) / len(a)))


def _select(mol: Chem.Mol, energies: dict[int, float], n_keep: int) -> list[int]:
    """Lowest-energy conformers, skipping any within PRUNE_RMSD of one already kept."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1] or list(
        range(mol.GetNumAtoms())
    )
    kept: list[int] = []
    for cid in sorted(energies, key=energies.get):
        xyz = mol.GetConformer(cid).GetPositions()[heavy]
        if all(
            kabsch_rmsd(xyz, mol.GetConformer(k).GetPositions()[heavy]) > PRUNE_RMSD for k in kept
        ):
            kept.append(cid)
        if len(kept) == n_keep:
            break
    return kept


def _renumber(mol: Chem.Mol, order: list[int]) -> list[int]:
    """Keep only `order`, as conformer ids 0..k-1 (0 = lowest energy)."""
    coords = [mol.GetConformer(c).GetPositions() for c in order]
    mol.RemoveAllConformers()
    for new_id, xyz in enumerate(coords):
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetId(new_id)
        for i, p in enumerate(xyz):
            conf.SetAtomPosition(i, p.tolist())
        mol.AddConformer(conf, assignId=False)
    return list(range(len(coords)))


def embed(mol: Chem.Mol, n_conformers: int = 3, seed: int = 0) -> list[int]:
    """Embed `mol` in place and return the conformer ids (0 is the lowest MMFF energy).

    `mol` must carry explicit hydrogens (`Chem.AddHs`). Raises `QMError(EMBED_FAILED)`.
    """
    if has_implicit_hydrogens(mol):
        raise ValueError("embed needs explicit hydrogens: call Chem.AddHs first")
    if mol.HasProp(NOTES_PROP):
        mol.ClearProp(NOTES_PROP)
    mol.RemoveAllConformers()
    if mol.GetNumAtoms() == 1:
        return _renumber_single(mol)
    cids = _generate(mol, max(n_conformers * 3, 10), seed)
    if not cids:
        raise QMError(ReasonCode.EMBED_FAILED, "ETKDGv3 produced no conformer")
    energies = _relax(mol, cids)
    return _renumber(mol, _select(mol, energies, n_conformers))


def _renumber_single(mol: Chem.Mol) -> list[int]:
    conf = Chem.Conformer(1)
    conf.SetId(0)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conf, assignId=False)
    return [0]
