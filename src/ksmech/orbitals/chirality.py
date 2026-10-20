"""Per-atom chirality sign: the signed volume of a stereocentre's three highest-ranked neighbours.

    V = (x_1 - x_c) . [(x_2 - x_c) x (x_3 - x_c)],   chirality = -sign(V)

with neighbours 1, 2, 3 ordered by RDKit's CIP ranks, so the value does not depend on atom
order. +1 means ranks 1 -> 2 -> 3 turn clockwise when viewed with the lowest-ranked neighbour (or
the lone pair) pointing away, the R-like sense; -1 the S-like sense. The ranks are RDKit's
legacy CIP ranks, so "R-like" is exact wherever those match the CIP rules. (RDKit's legacy R/S
*labels* are wrong for three-coordinate centres such as sulfoxides; the ranks and this sign are
not, and agree with `rdCIPLabeler`.) The sign is read from the geometry of the QM record, but
only for stereocentres whose configuration the SMILES specifies: an unspecified centre gets an
arbitrary enantiomer from ETKDG, so its geometric sign carries no information and the feature is
0, as it is for every atom that is not a stereocentre. A geometry whose chiral tag disagrees
with the SMILES (an inverted centre) keeps its geometric sign and is logged; RDKit does not
perceive three-coordinate P centres from 3D, so those are not cross-checked.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem

log = logging.getLogger(__name__)
_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _ranked_neighbours(atom: Chem.Atom) -> list[int]:
    nbrs = sorted(atom.GetNeighbors(), key=lambda n: -int(n.GetProp("_CIPRank")))
    return [n.GetIdx() for n in nbrs[:3]]


def _tags_from_geometry(m: Chem.Mol, coords: np.ndarray) -> list[Chem.ChiralType]:
    g = Chem.Mol(m)
    g.RemoveAllConformers()
    conf = Chem.Conformer(g.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, xyz.tolist())
    g.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(g, replaceExistingTags=True)
    return [a.GetChiralTag() for a in g.GetAtoms()]


def chirality_signs(mol: Chem.Mol, coords: np.ndarray) -> np.ndarray:
    """(n_atoms,) +1 / -1 on stereocentres specified in `mol`, from `coords`; 0 elsewhere."""
    m = Chem.Mol(mol)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)  # ranks; drops non-stereo tags
    out = np.zeros(m.GetNumAtoms())
    specified = [a for a in m.GetAtoms() if a.GetChiralTag() in _TETRAHEDRAL]
    if not specified:
        return out
    from_3d = _tags_from_geometry(m, coords)
    for atom in specified:
        idx = atom.GetIdx()
        v = coords[_ranked_neighbours(atom)] - coords[idx]
        out[idx] = -float(np.sign(np.linalg.det(v)))
        if from_3d[idx] in _TETRAHEDRAL and from_3d[idx] != atom.GetChiralTag():
            log.warning(
                "atom %d of %s: the geometry inverts the stereocentre the SMILES specifies",
                idx,
                Chem.MolToSmiles(Chem.RemoveHs(m)),
            )
    return out
