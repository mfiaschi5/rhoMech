"""embed(): explicit-H ETKDGv3 embedding that keeps atom order."""

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.qm.embed import NOTES_PROP, PRUNE_RMSD, embed, kabsch_rmsd, prepare_mol
from ksmech.qm.records import QMError, ReasonCode
from tests.fixtures.molecules import MOLECULES


def graph(mol):
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    edges = sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds())
    return z, edges


@pytest.mark.parametrize("name", MOLECULES)
def test_embed_keeps_atom_order_and_returns_ids(name, smiles):
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smiles[name])))
    before = graph(mol)
    ids = embed(mol, n_conformers=3, seed=0)
    assert graph(mol) == before
    assert ids == list(range(len(ids))) and 1 <= len(ids) <= 3
    assert mol.GetNumConformers() == len(ids)
    for cid in ids:
        xyz = mol.GetConformer(cid).GetPositions()
        assert xyz.shape == (mol.GetNumAtoms(), 3) and np.isfinite(xyz).all()


def test_atom_map_numbers_survive_embedding():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    for i, atom in enumerate(mol.GetAtoms(), start=10):
        atom.SetAtomMapNum(i)
    embed(mol, n_conformers=2)
    assert [a.GetAtomMapNum() for a in mol.GetAtoms()] == list(range(10, 10 + mol.GetNumAtoms()))


def test_embedding_is_deterministic_for_a_seed():
    a, b = prepare_mol("CCCCO"), prepare_mol("CCCCO")
    embed(a, 3, seed=7)
    embed(b, 3, seed=7)
    assert np.allclose(a.GetConformer(0).GetPositions(), b.GetConformer(0).GetPositions())


def test_returned_conformers_are_pruned_by_rmsd():
    mol = prepare_mol("CCCCCCC")  # flexible: many distinct conformers
    ids = embed(mol, n_conformers=4, seed=0)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    for i in ids:
        for j in ids:
            if i < j:
                xi = mol.GetConformer(i).GetPositions()[heavy]
                xj = mol.GetConformer(j).GetPositions()[heavy]
                assert kabsch_rmsd(xi, xj) > PRUNE_RMSD


def test_single_atom_gets_one_conformer_at_the_origin():
    mol = prepare_mol("[Na+]")
    assert embed(mol, n_conformers=3) == [0]
    assert np.allclose(mol.GetConformer(0).GetPositions(), 0.0)


def test_implicit_hydrogens_are_rejected():
    with pytest.raises(ValueError, match="explicit hydrogens"):
        embed(Chem.MolFromSmiles("CCO"))


def test_bare_proton_embeds():
    assert embed(prepare_mol("[H+]")) == [0]


def test_unparseable_smiles_is_a_qm_error():
    with pytest.raises(QMError):
        prepare_mol("C(C")


def test_kabsch_rmsd_ignores_rigid_motion():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(6, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert kabsch_rmsd(xyz, xyz @ q.T + 5.0) == pytest.approx(0.0, abs=1e-6)
    assert kabsch_rmsd(xyz, xyz + 0.1 * rng.normal(size=xyz.shape)) > 0


def notes(mol) -> list[str]:
    return mol.GetProp(NOTES_PROP).split(",") if mol.HasProp(NOTES_PROP) else []


def test_a_calcium_chelate_embeds_with_relaxed_bounds():
    """ETKDG's triangle smoothing rejects the distance bounds of a Ca chelate ring, from random
    coordinates too; the last attempt ignores the smoothing failure and the knowledge terms, and
    records that it did. A molecule that embeds normally never reaches it."""
    chelate = prepare_mol("O=C([O-])C[NH+]1CC(=O)[O][Ca]1")
    assert embed(chelate, n_conformers=1, seed=0) == [0]
    assert str(ReasonCode.EMBED_RELAXED_BOUNDS) in notes(chelate)
    xyz = chelate.GetConformer(0).GetPositions()
    assert np.isfinite(xyz).all() and xyz.std() > 0.1
    ethanol = prepare_mol("CCO")
    embed(ethanol, n_conformers=1, seed=0)
    assert str(ReasonCode.EMBED_RELAXED_BOUNDS) not in notes(ethanol)
