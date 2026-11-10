"""The thermo sidecar (`qm.thermo_cache`), `ksmech-qm thermo` (`qm.thermo_batch`), its failure
memo, its rows in the cache index, and the ALPB solvent as a parameter of the QM run."""

import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from ksmech.data import provenance as prov
from ksmech.qm import index, thermo_batch
from ksmech.qm import thermo as th
from ksmech.qm import thermo_cache as tc
from ksmech.qm.cli import build_parser, main
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.records import QMError, ReasonCode, cache_key, cache_path, save_record
from ksmech.qm.spin import method_for
from ksmech.qm.xtb_calc import ALPB_SOLVENTS, GFN2Calculator, QMConfig, solvent_of


@pytest.fixture
def qm_dir(qm_record, qm_config):
    """The session's QM cache with ethane and chloromethane records in it."""
    for smiles in ("CC", "CCl"):
        qm_record(smiles)
    return qm_config.cache_dir


def _config(qm_dir, tmp_path, **kw) -> tc.ThermoConfig:
    return tc.ThermoConfig(qm_dir=qm_dir, out_dir=tmp_path / "thermo", **kw)


def _task(smiles, config, memo=True):
    return thermo_batch.make_tasks([smiles], QMConfig().method, config, memo=memo)[0]


def test_the_sidecar_round_trips_and_goes_stale_with_its_record(qm_dir, tmp_path, qm_record):
    config = _config(qm_dir, tmp_path)
    rec = qm_record("CC")
    record_path = cache_path(qm_dir, "CC", 0, rec.method)
    arrays = tc.compute_thermo(rec, config)
    path = tc.save_thermo(arrays, record_path, config, seconds=1.5)
    assert path == tc.thermo_path(config.out_dir, "CC", 0, rec.method)
    header = prov.read_header(path)
    assert header.kind == "thermo" and header.extra["seconds"] == 1.5
    assert tc.is_current(path, rec.method, 0, config)
    assert not tc.is_current(path, rec.method, 0, replace(config, fd_step_bohr=0.01))
    loaded = tc.load_thermo(path, (rec.method, 0), config)
    direct = tc.as_thermo(arrays)
    for temp in (273.15, 298.15, 400.0):
        assert th.thermo_at(loaded, temp).free_energy == th.thermo_at(direct, temp).free_energy
    assert tc.free_energy_of("CC", 298.15, config=config) == th.standard_free_energy(direct, 298.15)
    other = tmp_path / "other.npz"
    save_record(replace(rec, energy_ha=rec.energy_ha - 1.0), other)
    stale = tc.save_thermo(arrays, other, replace(config, out_dir=tmp_path / "t2"))
    save_record(rec, other)  # the record changed after its sidecar was written
    with pytest.raises(prov.StaleArtifact):
        tc.load_thermo(stale)
    with pytest.raises(FileNotFoundError):
        tc.thermo_of("CCO", config=config)


def test_the_command_computes_skips_and_answers_from_the_cache(qm_dir, tmp_path, capsys):
    out, report = tmp_path / "thermo", tmp_path / "report.json"
    argv = ["thermo", "--smiles", "CC", "ClC", "CCCCCCN", "--cache-dir", str(qm_dir)]
    argv += ["--out-dir", str(out), "--report", str(report), "--temperatures", "298.15", "350"]
    assert main(argv) == 0
    rows = {r["smiles"]: r for r in json.loads(report.read_text())["rows"]}
    assert rows["CC"]["status"] == "ok" and rows["CCl"]["status"] == "ok"
    assert (rows["CCCCCCN"]["status"], rows["CCCCCCN"]["reason"]) == ("skipped", "NO_RECORD")
    assert [t["T"] for t in rows["CC"]["thermo"]] == [298.15, 350.0]
    assert rows["CC"]["thermo"][0]["g_minus_e_kcal"] == pytest.approx(35.0, abs=1.5)
    printed = capsys.readouterr().out
    assert "NO_RECORD" in printed and "time per Hessian" in printed
    assert main(argv) == 0
    rows = {r["smiles"]: r for r in json.loads(report.read_text())["rows"]}
    assert rows["CC"]["status"] == "up_to_date" and "thermo" in rows["CC"]


def test_a_recurring_failure_leaves_a_tombstone(qm_dir, tmp_path, monkeypatch):
    config = _config(qm_dir, tmp_path)

    def fail(record, cfg):
        raise QMError(ReasonCode.SCF_NOT_CONVERGED, "displaced geometry: 250 cycles")

    monkeypatch.setattr(thermo_batch, "compute_thermo", fail)
    [row] = thermo_batch.process_task(_task("CC", config))
    assert (row["status"], row["reason"]) == ("failed", "SCF_NOT_CONVERGED")
    method = method_for("CC", QMConfig().method)
    assert tc.failure_path(config.out_dir, "CC", 0, method).is_file()
    [row] = thermo_batch.process_task(_task("CC", config))
    assert row["status"] == "known_failure" and "not recomputed" in row["detail"]
    [row] = thermo_batch.process_task(_task("CC", replace(config, scf_maxiter=500)))
    assert row["status"] == "failed"  # a larger budget runs again
    monkeypatch.undo()
    [row] = thermo_batch.process_task(_task("CC", config, memo=False))
    assert row["status"] == "ok"
    assert not tc.failure_path(config.out_dir, "CC", 0, method).exists()


def test_timeouts_are_remembered_and_unexpected_errors_are_not(qm_dir, tmp_path, monkeypatch):
    config = _config(qm_dir, tmp_path)
    method = method_for("CCl", QMConfig().method)
    tomb = tc.failure_path(config.out_dir, "CCl", 0, method)

    def timeout(record, cfg):
        raise thermo_batch.batch._Timeout

    monkeypatch.setattr(thermo_batch, "compute_thermo", timeout)
    [row] = thermo_batch.process_task(_task("CCl", config))
    assert row["reason"] == "TIMEOUT" and tomb.is_file()
    tomb.unlink()
    monkeypatch.setattr(thermo_batch, "compute_thermo", lambda r, c: 1 / 0)
    [row] = thermo_batch.process_task(_task("CCl", config))
    assert row["reason"] == "UNEXPECTED" and not tomb.exists()


def test_a_record_with_no_medium_is_skipped(qm_dir, tmp_path, qm_record):
    rec = replace(qm_record("CC"), method="GFN2-xTB/tblite-lbfgs")
    old = tmp_path / "old"
    save_record(rec, cache_path(old, "CC", 0, rec.method))
    config = replace(_config(qm_dir, tmp_path), qm_dir=old)
    task = thermo_batch.ThermoTask("CC", 0, rec.method, config)
    [row] = thermo_batch.process_task(task)
    assert (row["status"], row["reason"]) == ("skipped", "UNKNOWN_METHOD")


def test_the_index_lists_sidecars_and_their_tombstones(qm_dir, tmp_path, monkeypatch):
    root = tmp_path / "root"
    config = tc.ThermoConfig(qm_dir=qm_dir, out_dir=root / "thermo")
    [ok] = thermo_batch.process_task(_task("CC", config))
    monkeypatch.setattr(
        thermo_batch, "compute_thermo", lambda r, c: (_ for _ in ()).throw(QMError("TIMEOUT"))
    )
    [bad] = thermo_batch.process_task(_task("CCl", config))
    assert (ok["status"], bad["reason"]) == ("ok", "TIMEOUT")
    method = method_for("CC", QMConfig().method)
    save_record_path = cache_path(root / "qm", "CC", 0, method)
    save_record_path.parent.mkdir(parents=True)
    save_record_path.write_bytes(cache_path(qm_dir, "CC", 0, method).read_bytes())
    counts = index.refresh(root)
    assert counts["thermo:added"] == 1 and counts["thermo_failure:added"] == 1
    con = sqlite3.connect(index.index_path(root))
    row = con.execute(
        "SELECT smiles, method, n_atoms, unpaired, problem FROM artifact WHERE kind = 'thermo'"
    ).fetchone()
    assert row == ("CC", method, 8, 0, None)
    tomb = con.execute("SELECT smiles, reason FROM artifact WHERE kind = 'thermo_failure'")
    assert tomb.fetchall() == [("CCl", "TIMEOUT")]
    view = con.execute("SELECT smiles, has_record, has_thermo FROM molecule").fetchall()
    assert view == [("CC", 1, 1)]
    s = index.stats(root)
    assert s["thermo_reasons"] == [("TIMEOUT", 1, "CCl")] and s["records_without_thermo"] == 0
    assert dict((k, n) for k, n, *_ in s["kinds"])["thermo"] == 1
    assert main(["index", "stats", "--root", str(root), "--no-refresh"]) == 0


def test_the_thermo_command_needs_one_source_for_steps(tmp_path, capsys):
    argv = ["thermo", "--source", "pmechdb", "reactmech", "--steps-from", str(tmp_path / "x")]
    assert main(argv) == 2
    assert "exactly one --source" in capsys.readouterr().out


# ---- the solvent as a parameter


def test_the_solvent_is_part_of_the_method_and_the_key():
    water, methanol = QMConfig(), QMConfig(solvent="methanol")
    assert water.method == "GFN2-xTB/ALPB-water/tblite-lbfgs"  # every existing key unchanged
    assert methanol.method == "GFN2-xTB/ALPB-methanol/tblite-lbfgs"
    assert QMConfig(solvent=None).method == "GFN2-xTB/gas/tblite-lbfgs"
    assert cache_key("CCO", 0, water.method) != cache_key("CCO", 0, methanol.method)
    for cfg in (water, methanol, QMConfig(solvent=None)):
        assert solvent_of(cfg.method) == cfg.solvent
    assert solvent_of(method_for("Cl[C]Cl", methanol.method)) == "methanol"
    with pytest.raises(ValueError):
        QMConfig(solvent="h2o")  # tblite's alias of water: one solvent, one method string
    with pytest.raises(ValueError):
        solvent_of("GFN2-xTB/tblite-lbfgs")
    assert {"water", "methanol", "dmso", "thf", "acetonitrile"} <= set(ALPB_SOLVENTS)


def test_the_run_command_takes_a_solvent():
    args = build_parser().parse_args(["run", "--source", "pmechdb", "--solvent", "dmso"])
    assert args.solvent == "dmso"
    assert build_parser().parse_args(["run", "--source", "pmechdb"]).solvent == "water"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--source", "pmechdb", "--solvent", "h2o"])


def test_another_solvent_writes_its_own_record_and_hessian(tmp_path, monkeypatch):
    cfg = QMConfig(solvent="methanol", cache_dir=tmp_path)
    mol = prepare_mol("CO")
    embed(mol, 1)
    rec = GFN2Calculator(cfg).run(mol)
    water = GFN2Calculator(replace(cfg, solvent="water")).run(mol)
    assert rec.method == "GFN2-xTB/ALPB-methanol/tblite-lbfgs" != water.method
    assert abs(rec.energy_ha - water.energy_ha) > 1e-4
    seen = []
    real = tc.fd_hessian
    monkeypatch.setattr(tc, "fd_hessian", lambda *a, **k: seen.append(a[4]) or real(*a, **k))
    tc.compute_thermo(rec)
    assert seen == ["methanol"]


def test_every_alpb_solvent_runs_in_tblite():
    from tblite.interface import Calculator

    z, xyz = np.array([8, 1, 1]), np.array([[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]])
    energies = set()
    for solvent in ALPB_SOLVENTS:
        calc = Calculator("GFN2-xTB", z, xyz.astype(float))
        calc.set("verbosity", 0)
        calc.add("alpb-solvation", solvent)
        energies.add(round(float(calc.singlepoint().get("energy")), 8))
    assert len(energies) == len(ALPB_SOLVENTS)


def test_species_selection_from_smiles(tmp_path):
    from ksmech.qm.cli import thermo_species

    args = build_parser().parse_args(["thermo", "--smiles", "OCC", "CCO", "C(Cl)", "--limit", "1"])
    picked = thermo_species(args)
    assert len(picked) == 1 and picked[0] in {"CCO", "CCl"}
    args = build_parser().parse_args(["thermo", "--smiles", "OCC", "CCO", "C(Cl)"])
    assert thermo_species(args) == ["CCO", "CCl"]
