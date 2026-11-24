"""GPU xTB stage 5: `ksmech-qm run --optimizer gpu-anc` (`qm.gpu_run`).

The routing, the finishing of a GPU geometry on tblite (single point, polish, record), the CPU
retry of a GPU failure and the driver run without a device, the GPU's outcomes faked; the tests
marked `gpu` run the real batched optimizer against tblite-anc on the CPU.
"""

from __future__ import annotations

import types
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.qm import failures, gpu_run
from ksmech.qm.batch import MolTask
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import BOHR_ANG
from ksmech.qm.records import ReasonCode, cache_path, is_valid_cache_file, load_record
from ksmech.qm.spin import config_for
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL
from tests.test_qm_gpu import gpu

METHOD = "GFN2-xTB/ALPB-water/gpu-anc"


def _config(tmp_path, optimizer="gpu-anc") -> QMConfig:
    return QMConfig(optimizer=optimizer, cache_dir=tmp_path)


def _canon(smiles: str) -> str:
    """The batch's key of a species: its map-free canonical SMILES."""
    return chem.unmapped_canonical(Chem.MolFromSmiles(smiles))


def _task(smiles: str, config: QMConfig, n: int = 1) -> MolTask:
    key = _canon(smiles)
    return MolTask(key, n, config_for(key, config))


def _cpu_record(smiles: str, tmp_path):
    """tblite-anc's record from the embedded start, as `qm.batch._compute` makes it."""
    mol = prepare_mol(_canon(smiles))
    embed(mol, 1, seed=0)
    return mol, GFN2Calculator(_config(tmp_path / "cpu", "tblite-anc")).run(mol, 0)


def test_gpu_anc_is_its_own_method_and_runs_anc_on_the_cpu(tmp_path, monkeypatch):
    config = _config(tmp_path)
    assert config.method == METHOD
    called = []
    monkeypatch.setattr(
        "ksmech.qm.xtb_calc.optimize_tblite_anc", lambda *a, **k: called.append(k) or a[1]
    )
    mol = prepare_mol(_canon(MOLECULES["ethanol"]))
    embed(mol, 1, seed=0)
    x = mol.GetConformer(0).GetPositions() / BOHR_ANG
    GFN2Calculator(config)._optimize(mol, np.array([6, 6, 8, 1, 1, 1, 1, 1, 1]), x, 0, 0)
    assert len(called) == 1 and called[0]["guard"] is not None  # ANC, the guard on a free run


def test_run_from_a_converged_geometry_is_the_cpu_record(tmp_path):
    mol, ref = _cpu_record(MOLECULES["acetone"], tmp_path)
    rec = GFN2Calculator(_config(tmp_path)).run_from(mol, 0, ref.coords_ang / BOHR_ANG, False)
    assert rec.method == METHOD
    assert ReasonCode.GPU_POLISHED not in rec.warnings
    assert abs(rec.energy_ha - ref.energy_ha) < 1e-10
    assert np.abs(rec.fukui_plus - ref.fukui_plus).max() < 1e-8


def test_run_from_an_unconverged_geometry_is_polished_by_tblite_anc(tmp_path):
    mol, ref = _cpu_record(MOLECULES["acetone"], tmp_path)
    start = mol.GetConformer(0).GetPositions() / BOHR_ANG
    rec = GFN2Calculator(_config(tmp_path)).run_from(mol, 0, start, False)
    assert ReasonCode.GPU_POLISHED in rec.warnings
    assert abs(rec.energy_ha - ref.energy_ha) < 1e-10  # the polish from the start is the run


def test_run_from_refuses_a_tagged_species(tmp_path):
    mol = prepare_mol(_canon(OPEN_SHELL["dichlorocarbene"][0]))
    embed(mol, 1, seed=0)
    with pytest.raises(ValueError, match="tagged"):
        GFN2Calculator(_config(tmp_path)).run_from(mol, 0, np.zeros((3, 3)), False)


def test_prepare_routes_the_species_the_gpu_does_not_take(tmp_path):
    config = _config(tmp_path)
    ethanol = _task(MOLECULES["ethanol"], config)
    assert gpu_run.prepare((ethanol, gpu_run.GPU_MIN_ATOMS)) == ("cpu", "small")
    assert gpu_run.prepare((_task("CCO", config, 2), 1)) == ("cpu", "conformers")
    assert gpu_run.prepare((_task(OPEN_SHELL["methyl"][0], config), 1)) == ("cpu", "open_shell")
    carbene = _task(OPEN_SHELL["dichlorocarbene"][0], config)
    assert gpu_run.prepare((carbene, 1)) == ("cpu", "spin_tag")
    kind, item = gpu_run.prepare((ethanol, 1))
    assert kind == "gpu" and item.z.tolist() == [6, 6, 8, 1, 1, 1, 1, 1, 1]
    assert item.start.shape == (9, 3) and item.charge == 0
    failures.remember(ethanol.config, ethanol.smiles, 0, "OPT_NOT_CONVERGED", "x", 1.0)
    kind, rows = gpu_run.prepare((ethanol, 1))
    assert kind == "rows" and rows[0]["detail"].startswith(failures.KNOWN_PREFIX)


def test_finish_writes_the_record_of_a_gpu_geometry(tmp_path):
    config = _config(tmp_path)
    _, item = gpu_run.prepare((_task(MOLECULES["acetone"], config), 1))
    _, ref = _cpu_record(MOLECULES["acetone"], tmp_path)
    rows = gpu_run.finish(gpu_run.Finish(item, ref.coords_ang / BOHR_ANG, False, "", 12))
    assert [r["status"] for r in rows] == ["ok"]
    assert rows[0]["detail"] == "optimized on the GPU, 12 energies"
    path = cache_path(tmp_path, item.task.smiles, 0, METHOD)
    assert rows[0]["cache_path"] == str(path)
    assert abs(load_record(path).energy_ha - ref.energy_ha) < 1e-10


def test_a_gpu_failure_is_retried_on_the_cpu_and_leaves_no_tombstone(tmp_path):
    config = _config(tmp_path)
    _, item = gpu_run.prepare((_task(MOLECULES["acetone"], config), 1))
    job = gpu_run.Finish(item, None, False, "GPU: OPT_NOT_CONVERGED: 500 energies", 500)
    rows = gpu_run.finish(job)
    assert rows[0]["status"] == "ok" and gpu_run.RETRY_NOTE in rows[0]["detail"]
    assert is_valid_cache_file(cache_path(tmp_path, item.task.smiles, 0, METHOD))
    assert not failures.tombstone_path(tmp_path, item.task.smiles, 0, METHOD).exists()


def _fake_gpu(inbox, config, out):
    """The GPU thread: the CPU's tblite-anc geometry for the first item it gets, a failure for
    the rest (the items as the driver feeds them, `more` read until it closes)."""
    k = 0
    while (got := inbox.more(True)) is not None:
        for task in got:
            calc = GFN2Calculator(config)
            xyz = None if k else calc._relax(task.mol, task.z, task.start, 0, 0)[0]
            reason = None if k == 0 else ReasonCode.OPT_NOT_CONVERGED
            fields = dict(key=task.key, xyz=xyz, restrained=False, reason=reason, detail="x")
            out.put(types.SimpleNamespace(**fields, n_evals=7))
            k += 1
    out.put(gpu_run._DONE)


def test_execute_routes_finishes_and_retries_with_a_fake_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_run, "_gpu_thread", _fake_gpu)
    config = _config(tmp_path)
    memo = _task(MOLECULES["ethane"], config)
    failures.remember(memo.config, memo.smiles, 0, "OPT_NOT_CONVERGED", "x", 1.0)
    names = ("acetone", "methyl_acetate")
    tasks = [_task(MOLECULES[n], config) for n in names]
    tasks += [_task(OPEN_SHELL["methyl"][0], config), memo]
    calls = []
    counts = gpu_run.execute(tasks, 2, calls.append, min_atoms=1)
    assert sorted(r[0]["smiles"] for r in calls) == sorted(t.smiles for t in tasks)
    assert counts == Counter(gpu=2, cpu_open_shell=1, memo=1, gpu_ok=1, gpu_retried_ok=1)
    by = {r[0]["smiles"]: r[0] for r in calls}
    assert all(by[t.smiles]["status"] == "ok" for t in tasks[:3])


def test_execute_refuses_another_solvent(tmp_path):
    config = QMConfig(optimizer="gpu-anc", solvent="dmso", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="water"):
        gpu_run.execute([_task("CCO", config)], 1, print)


@pytest.mark.gpu
@gpu
def test_gpu_anc_records_match_tblite_anc(tmp_path):
    config = _config(tmp_path)
    names = ("acetone", "methyl_acetate", "benzene", "acrolein")
    tasks = [_task(MOLECULES[n], config) for n in names]
    tasks += [_task(OPEN_SHELL[n][0], config) for n in ("methyl", "dichlorocarbene")]
    calls = []
    counts = gpu_run.execute(tasks, 2, calls.append, min_atoms=1)
    assert counts["gpu"] == 4 and counts["cpu_open_shell"] + counts["cpu_spin_tag"] == 2
    assert counts["gpu_ok"] + counts["gpu_polished"] >= 4 and not counts["gpu_retried_ok"]
    for n in names:
        _, ref = _cpu_record(MOLECULES[n], tmp_path)
        rec = load_record(cache_path(tmp_path, ref.smiles, 0, METHOD))
        assert abs(rec.energy_ha - ref.energy_ha) < 1e-5, n


@pytest.mark.gpu
@gpu
def test_relax_many_takes_the_scf_cycle_limit(tmp_path):
    from ksmech.qm.gpu.optimize import RelaxTask, relax_many

    _, item = gpu_run.prepare((_task(MOLECULES["acetone"], _config(tmp_path)), 1))
    task = RelaxTask("a", prepare_mol(item.task.smiles), item.z, item.start, 0, 0)
    seen = []
    out = relax_many([task], QMConfig(optimizer="gpu-anc", scf_maxiter=2), on_result=seen.append)
    assert out[0].reason is ReasonCode.SCF_NOT_CONVERGED and seen == out
