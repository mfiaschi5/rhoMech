"""GFN2-xTB geometries of d-metal species of the kind the mechanism datasets carry (ReactMech,
FlowER): after the optimization the metal keeps every drawn ligand, at a bond length near
experiment, in its coordination geometry (square-planar d8 Pd(II), tetrahedral d0 Cr(VI)).

RDKit's force fields cannot type these metals (`EMBED_UFF_FALLBACK`), so GFN2 starts from the
raw ETKDG coordinates and has to find the geometry itself. Experimental references: Pd-Cl 2.30
A in [PdCl4]2- (K2PdCl4), Pd-P 2.33 and Pd-Cl 2.30 A in trans-PdCl2(PMe3)2, Cr-O 1.65 A in
chromate.
"""

import itertools

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.qm.embed import prepare_mol

pytestmark = pytest.mark.slow

TETRACHLOROPALLADATE = "Cl[Pd-2](Cl)(Cl)Cl"
PDCL2_PME3 = "C[P+](C)(C)[Pd-2](Cl)(Cl)[P+](C)(C)C"  # the dative P->Pd bonds charge-separated
CHROMATE = "[O-][Cr](=O)(=O)[O-]"
CHROMIC_ACID = "[O]=[Cr](=[O])([OH])[OH]"  # as ReactMech draws it


def _around(qm_record, smiles: str, metal: str):
    """(the record, {ligand atom: (symbol, distance to the metal in A)}, the ligand-metal-ligand
    angles in degrees, sorted) of the optimized geometry."""
    rec = qm_record(smiles)
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smiles)))
    m = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == metal)
    xyz = rec.coords_ang
    ligands = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()]
    dist = {i: (mol.GetAtomWithIdx(i).GetSymbol(), float(np.linalg.norm(xyz[i] - xyz[m])))
            for i in ligands}  # fmt: skip
    angles = []
    for a, b in itertools.combinations(ligands, 2):
        u, v = xyz[a] - xyz[m], xyz[b] - xyz[m]
        cos = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
        angles.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return rec, dist, sorted(angles)


def _square_planar(angles: list[float]) -> bool:
    """Four cis angles near 90 that close the plane, two trans angles near 180."""
    cis, trans = angles[:4], angles[4:]
    return all(80 < x < 100 for x in cis) and abs(sum(cis) - 360) < 8 and min(trans) > 165


def test_tetrachloropalladate_is_square_planar(qm_record):
    rec, dist, angles = _around(qm_record, TETRACHLOROPALLADATE, "Pd")
    assert "EMBED_UFF_FALLBACK" in rec.warnings  # no force field types Pd
    assert len(dist) == 4 and all(2.22 < d < 2.38 for _, d in dist.values())
    assert _square_planar(angles), angles


def test_a_palladium_phosphine_chloride_is_square_planar_and_trans(qm_record):
    _, dist, angles = _around(qm_record, PDCL2_PME3, "Pd")
    assert sorted(s for s, _ in dist.values()) == ["Cl", "Cl", "P", "P"]
    assert all(2.25 < d < 2.40 for s, d in dist.values() if s == "P")
    assert all(2.22 < d < 2.38 for s, d in dist.values() if s == "Cl")
    assert _square_planar(angles), angles


@pytest.mark.parametrize("smiles", [CHROMATE, CHROMIC_ACID])
def test_chromium_vi_oxo_species_are_tetrahedral(qm_record, smiles):
    _, dist, angles = _around(qm_record, smiles, "Cr")
    assert len(dist) == 4 and all(1.50 < d < 1.80 for _, d in dist.values())
    assert all(100 < x < 118 for x in angles), angles


def test_chromate_bonds_are_equal_and_near_experiment(qm_record):
    _, dist, _ = _around(qm_record, CHROMATE, "Cr")
    lengths = [d for _, d in dist.values()]
    assert max(lengths) - min(lengths) < 0.02 and all(1.58 < d < 1.72 for d in lengths)


def test_chromic_acid_keeps_short_oxo_and_long_hydroxo_bonds(qm_record):
    """Cr=O shorter than Cr-OH: the optimization keeps the drawn bond orders apart."""
    _, dist, _ = _around(qm_record, CHROMIC_ACID, "Cr")
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(CHROMIC_ACID)))
    degree = {i: mol.GetAtomWithIdx(i).GetDegree() for i in dist}  # the record's H are atoms
    oxo = [d for i, (_, d) in dist.items() if degree[i] == 1]
    hydroxo = [d for i, (_, d) in dist.items() if degree[i] == 2]
    assert len(oxo) == len(hydroxo) == 2 and max(oxo) + 0.05 < min(hydroxo)
