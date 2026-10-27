"""QMugs single points: the molfile in record order, the streamed archive, the given geometry
and the resumable batch (`data.qmugs`, `qm.given`)."""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ksmech.data import chem, qmugs
from ksmech.qm.embed import prepare_mol
from ksmech.qm.given import featurize_qmugs
from ksmech.qm.records import QMError, ReasonCode
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig
from tests.fixtures.molecules import MOLECULES


def _molblock(smiles: str, seed: int = 0) -> str:
    """A 3D molfile with every H explicit, its atoms shuffled as a dataset might order them."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    order = np.random.default_rng(seed).permutation(mol.GetNumAtoms()).tolist()
    return Chem.MolToMolBlock(Chem.RenumberAtoms(mol, order))


def _distances(mol: Chem.Mol, conf_id: int = -1) -> np.ndarray:
    pos = mol.GetConformer(conf_id).GetPositions()
    return np.sort(np.linalg.norm(pos[:, None] - pos[None], axis=-1).ravel())


def _archive(path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_a_molfile_is_put_in_record_order_with_its_geometry(name):
    block = _molblock(MOLECULES[name])
    key, mol = qmugs.record_mol(block, 2)
    assert key == chem.unmapped_canonical(Chem.MolFromSmiles(MOLECULES[name]))
    ref = prepare_mol(key)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in ref.GetAtoms()]
    sdf = Chem.MolFromMolBlock(block, removeHs=False)
    assert np.allclose(_distances(mol, 2), _distances(sdf), atol=1e-4)
    for b in mol.GetBonds():  # every record bond is a short distance in the geometry
        p = mol.GetConformer(2).GetPositions()
        assert np.linalg.norm(p[b.GetBeginAtomIdx()] - p[b.GetEndAtomIdx()]) < 2.0


def test_the_archive_is_streamed_by_conformer(tmp_path):
    path = tmp_path / "structures.tar.gz"
    blocks = {
        "structures/CHEMBL1/conf_00.sdf": _molblock("CCO"),
        "structures/CHEMBL1/conf_01.sdf": _molblock("CCO", 1),
        "structures/CHEMBL2/conf_00.sdf": _molblock("CN"),
        "structures/README.txt": "not a conformer",
    }
    _archive(path, blocks)
    got = list(qmugs.iter_conformers(path, (0,)))
    assert [(c, n) for c, n, _ in got] == [("CHEMBL1", 0), ("CHEMBL2", 0)]
    assert got[1][2] == blocks["structures/CHEMBL2/conf_00.sdf"]
    assert len(list(qmugs.iter_conformers(path, (0, 1)))) == 3


def test_a_given_geometry_is_kept_and_checked(tmp_path):
    config = QMConfig(given_geometry=True, cache_dir=tmp_path)
    assert config.method == "GFN2-xTB/ALPB-water/given" and QMConfig().method != config.method
    _, mol = qmugs.record_mol(_molblock(MOLECULES["ethanol"]), 0)
    rec = GFN2Calculator(config).run(mol, 0)
    assert np.allclose(rec.coords_ang, mol.GetConformer(0).GetPositions())
    assert np.isfinite(rec.fukui_plus).all() and rec.method == config.method
    conf = mol.GetConformer(0)
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    far = np.asarray(conf.GetPositions()[o]) * 3.0 + 2.0
    conf.SetAtomPosition(o, far.tolist())  # the C-O bond broken
    with pytest.raises(QMError) as exc:
        GFN2Calculator(config).run(mol, 0)
    assert exc.value.reason == ReasonCode.CONNECTIVITY_CHANGED


def test_the_batch_resumes_and_counts_failures(tmp_path):
    path = tmp_path / "structures.tar.gz"
    _archive(
        path,
        {
            "structures/CHEMBL1/conf_00.sdf": _molblock(MOLECULES["ethane"]),
            "structures/CHEMBL2/conf_00.sdf": _molblock(MOLECULES["methylamine"]),
            "structures/CHEMBL3/conf_00.sdf": "not a molfile",
            "structures/CHEMBL4/conf_00.sdf": _molblock(MOLECULES["acetone"]),
        },
    )
    config = QMConfig(given_geometry=True, cache_dir=tmp_path / "qm")
    out = tmp_path / "out"
    first = featurize_qmugs(path, n_workers=1, limit=3, config=config, out_dir=out)
    assert list(first["status"]) == ["ok", "ok", "failed"]
    assert first.loc[2, "reason"] == ReasonCode.SDF_UNREADABLE
    assert len(list((tmp_path / "qm").glob("*.npz"))) == 2
    again = featurize_qmugs(path, n_workers=1, limit=4, config=config, out_dir=out)
    assert list(again["chembl_id"]) == ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"]
    manifest = pd.read_parquet(out / "qm_manifest.parquet")
    assert len(manifest) == 4 and (manifest["status"] == "ok").sum() == 3
