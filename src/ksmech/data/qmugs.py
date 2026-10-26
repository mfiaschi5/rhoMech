"""QMugs (Isert et al., Sci. Data 9, 273 (2022); CC BY-SA 4.0; docs/data_schema.md): about 665k
ChEMBL molecules with three conformers each, optimized with GFN2-xTB, and GFN2 and DFT
(wB97X-D/def2-SVP) properties. ksmech uses it for encoder pretraining (spec, "Data sources")
and takes its geometries as they are: GFN2-xTB single points, no re-optimization
(`QMConfig.given_geometry`, `qm.given`).

`iter_conformers` streams `structures.tar.gz` (members `structures/<CHEMBL id>/conf_<nn>.sdf`,
one V2000 molfile each, every H explicit, Kekule bonds, the properties as SD tags) without
extracting it. `record_mol` reads one molfile and puts its atoms in the record order of its
map-free canonical SMILES (`qm.embed.prepare_mol`, the order every QMRecord has), with the
QMugs geometry as the conformer of that id.
"""

from __future__ import annotations

import re
import tarfile
from collections.abc import Collection, Iterator
from pathlib import Path

from rdkit import Chem, rdBase

from ksmech.arrows.reactants import CorrespondenceError, record_correspondence
from ksmech.data import chem
from ksmech.qm.embed import prepare_mol

RAW_DIR = Path("data/raw/qmugs")
DEFAULT_TAR = RAW_DIR / "structures.tar.gz"
PROCESSED_DIR = Path("data/processed/qmugs")
CACHE_DIR = Path("data/cache/qmugs/qm")
MEMBER = re.compile(r"structures/(CHEMBL\d+)/conf_(\d+)\.sdf$")


def iter_conformers(
    tar_path: Path = DEFAULT_TAR, conformers: Collection[int] = (0,)
) -> Iterator[tuple[str, int, str]]:
    """(ChEMBL id, conformer index, molfile text) of every conformer in `conformers`, in the
    archive's order (ChEMBL ids sorted as strings)."""
    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tar:
            m = MEMBER.match(member.name)
            if m is None or int(m[2]) not in conformers:
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                yield m[1], int(m[2]), handle.read().decode()


def record_mol(molblock: str, conf_id: int) -> tuple[str, Chem.Mol]:
    """(map-free canonical SMILES, the molecule in its record order with the molfile's
    geometry as conformer `conf_id`). Raises ValueError when RDKit cannot read the molfile and
    CorrespondenceError when its atoms do not all map onto the record's."""
    with rdBase.BlockLogs():
        sdf = Chem.MolFromMolBlock(molblock, removeHs=False)
    if sdf is None:
        raise ValueError("RDKit cannot read the molfile")
    key = chem.unmapped_canonical(sdf)
    mol = prepare_mol(key)
    match = record_correspondence(sdf, mol, key)
    if sdf.GetNumAtoms() != mol.GetNumAtoms() or len(set(match.values())) != mol.GetNumAtoms():
        raise CorrespondenceError(f"{sdf.GetNumAtoms()} molfile atoms, {mol.GetNumAtoms()} record")
    pos = sdf.GetConformer().GetPositions()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, j in match.items():
        conf.SetAtomPosition(j, pos[i].tolist())
    conf.SetId(conf_id)
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=False)
    return key, mol
