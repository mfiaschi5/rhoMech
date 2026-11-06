"""Open-shell token fixtures (open-shell plan P2b): restricted gas-phase GFN2 single points at
fixed idealized geometries, so the degenerate open groups are exact (an optimized or embedded
geometry would split them). Atom order is `prepare_mol`'s: the heavy atoms of the SMILES, then
the hydrogens; each hydrogen sits 1.08 A (0.97 A on O) out from its heavy atom, away from the
heavy atom's other neighbours.

Each entry: SMILES, unpaired count u, heavy-atom coordinates (A). Linear CuCl2 is Cu(II), d9,
so u = 1 is given here; RDKit reads the drawing Cl[Cu]Cl with no radical.
"""

from __future__ import annotations

import numpy as np

_HEXAGON = [[1.40 * np.cos(np.pi * k / 3), 1.40 * np.sin(np.pi * k / 3), 0.0] for k in range(6)]

OPEN_SHELL_GEOMETRY: dict[str, tuple[str, int, list]] = {
    "methyl": ("[CH3]", 1, [[0.0, 0.0, 0.0]]),
    "hydroxyl": ("[OH]", 1, [[0.0, 0.0, 0.0]]),
    "nitric_oxide": ("[N]=O", 1, [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]),
    "triplet_o2": ("[O][O]", 2, [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]]),
    "superoxide": ("[O][O-]", 1, [[0.0, 0.0, 0.0], [1.33, 0.0, 0.0]]),
    "bromine_atom": ("[Br]", 1, [[0.0, 0.0, 0.0]]),
    "chlorine_atom": ("[Cl]", 1, [[0.0, 0.0, 0.0]]),
    "benzene_radical_anion": ("[CH-]1C=CC=C[CH]1", 1, _HEXAGON),
    "copper_dichloride": ("Cl[Cu]Cl", 1, [[-2.05, 0.0, 0.0], [0.0, 0.0, 0.0], [2.05, 0.0, 0.0]]),
}


def _hydrogen_positions(mol, heavy: np.ndarray) -> np.ndarray:
    """One position per hydrogen of `mol`, after the heavy atoms; a lone heavy atom's
    hydrogens are spread in the xy plane (planar CH3)."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        p = heavy[parent.GetIdx()]
        others = [n.GetIdx() for n in parent.GetNeighbors() if n.GetAtomicNum() != 1]
        d = 0.97 if parent.GetAtomicNum() == 8 else 1.08
        if others:
            v = p - heavy[others].mean(axis=0)
        else:
            k = sum(1 for q in out if np.allclose(q[1], p))
            n_h = sum(1 for n in parent.GetNeighbors() if n.GetAtomicNum() == 1)
            t = 2 * np.pi * k / n_h
            v = np.array([np.cos(t), np.sin(t), 0.0])
        out.append((atom.GetIdx(), p, p + d * v / np.linalg.norm(v)))
    return np.array([q[2] for q in out]).reshape(-1, 3)


def open_shell_record(name: str):
    """The restricted gas-phase GFN2 record of one fixture at its idealized geometry."""
    from ksmech.data import chem
    from ksmech.qm.embed import prepare_mol
    from ksmech.qm.geometry import BOHR_ANG, new_calculator
    from ksmech.qm.xtb_calc import GFN2Calculator

    smiles, u, heavy = OPEN_SHELL_GEOMETRY[name]
    mol = prepare_mol(smiles)
    heavy = np.asarray(heavy, dtype=float)
    xyz = np.vstack([heavy, _hydrogen_positions(mol, heavy)])
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    charge = chem.formal_charge(mol)
    calc = new_calculator(z, xyz / BOHR_ANG, charge, u, None, save_integrals=True)
    return GFN2Calculator()._assemble(calc.singlepoint(), z, xyz, smiles, 0, charge, u)
