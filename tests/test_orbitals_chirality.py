"""Chirality sign from the signed volume of CIP-ranked neighbours."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from ksmech.orbitals.chirality import chirality_signs
from ksmech.orbitals.tokens import atom_feature_names, build_molecule_tokens
from ksmech.qm.embed import embed, prepare_mol


def signs(smiles: str, seed: int = 0) -> tuple[Chem.Mol, np.ndarray]:
    mol = prepare_mol(smiles)
    embed(mol, n_conformers=1, seed=seed)
    return mol, chirality_signs(mol, mol.GetConformer(0).GetPositions())


def centre_label(mol: Chem.Mol) -> dict[int, str]:
    """CIP labels from RDKit's new labeller (the legacy one mislabels 3-coordinate centres)."""
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@@H](O)CC",
        "C[C@H](O)CC",
        "F[C@H](Cl)Br",
        "N[C@@H](C)C(=O)O",
        "C[S@@](=O)CC",
        "C[P@@](CC)c1ccccc1",
    ],
)
def test_sign_matches_the_cip_label(smiles):
    mol, chi = signs(smiles)
    labels = centre_label(mol)
    assert len(labels) == 1
    ((idx, label),) = labels.items()
    assert chi[idx] == (1.0 if label == "R" else -1.0)
    assert np.count_nonzero(chi) == 1


def test_enantiomers_have_opposite_signs_whatever_the_atom_order():
    _, r = signs("C[C@@H](O)CC")
    _, s = signs("C[C@H](O)CC")
    _, r_reordered = signs("CC[C@@H](C)O")  # the same (R) centre, written from the other end
    assert r[1] == -s[1] != 0
    assert r_reordered[2] == r[1]


def test_unspecified_and_achiral_centres_are_zero(smiles):
    for smi in [*smiles.values(), "CC(O)CC"]:  # the ten fixtures, and 2-butanol without stereo
        for seed in (0, 1):
            _, chi = signs(smi, seed)
            assert not chi.any(), smi


def test_atom_tokens_carry_the_sign(sp_record):
    names = atom_feature_names()
    assert names[-1] == "chirality"
    r = build_molecule_tokens(sp_record("C[C@@H](O)CC"))
    s = build_molecule_tokens(sp_record("C[C@H](O)CC"))
    col_r, col_s = r.atom_features()[:, -1], s.atom_features()[:, -1]
    assert np.count_nonzero(col_r) == 1 and np.array_equal(col_r, -col_s)
    assert r.atoms[int(np.flatnonzero(col_r)[0])].atomic_number == 6
