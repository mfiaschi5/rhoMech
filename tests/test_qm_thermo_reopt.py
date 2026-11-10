"""Re-optimization along imaginary modes before the thermochemistry (`qm.thermo_reopt`, its use
in `qm.thermo_cache.compute_thermo` and `ksmech-qm thermo`): a saddle built on the fixture set
relaxes to a lower minimum with the same bonds, a minimum is left alone, the record file is
never written, a sidecar from before the re-optimization reads stale, and every fallback keeps
xtb's rule with its stop code."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ksmech.data import provenance as prov
from ksmech.qm import thermo as th
from ksmech.qm import thermo_batch
from ksmech.qm import thermo_cache as tc
from ksmech.qm import thermo_reopt as ro
from ksmech.qm.cli import build_parser, main, thermo_config
from ksmech.qm.embed import prepare_mol
from ksmech.qm.geometry import BOHR_ANG, bond_stretch_ratios, new_calculator, optimize_tblite
from ksmech.qm.records import QMError, ReasonCode, cache_path, load_record, save_record
from ksmech.qm.xtb_calc import QMConfig, solvent_of
from tests.fixtures.molecules import MOLECULES

ETHANE = MOLECULES["ethane"]
OLD_LAYOUT = (  # the arrays of a sidecar written before 2026-11-10
    "smiles", "conformer_id", "method", "charge", "unpaired", "multiplicity", "atomic_numbers",
    "energy_ha", "frequencies_cm", "moments_amu_ang2", "mass_amu", "symmetry_number",
    "point_group", "linear", "grad_max", "hessian_asymmetry", "n_gradients", "fresh_starts",
    "warnings",
)  # fmt: skip


@pytest.fixture(scope="module")
def eclipsed_ethane(qm_record):
    """The ethane record with one methyl turned by 60 degrees and optimized again by the
    records' own L-BFGS-B at fmax 3e-4: it stays on the symmetric geometry, the eclipsed saddle
    of the torsion (about -290 cm^-1, 2.6 kcal/mol above the staggered minimum)."""
    rec = qm_record(ETHANE)
    x = rec.coords_ang.copy()
    axis = (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])
    turn = Rotation.from_rotvec(np.pi / 3 * axis).as_matrix()
    x[5:8] = (x[5:8] - x[1]) @ turn.T + x[1]  # the hydrogens of the second carbon
    z, solvent = rec.atomic_numbers, solvent_of(rec.method)
    xb = optimize_tblite(z, x / BOHR_ANG, rec.charge, rec.unpaired, 3e-4, 500, solvent)
    calc = new_calculator(z, xb, rec.charge, rec.unpaired, solvent)
    return replace(rec, coords_ang=xb * BOHR_ANG, energy_ha=float(calc.singlepoint().get("energy")))


@pytest.fixture(scope="module")
def saddle_arrays(eclipsed_ethane):
    return tc.compute_thermo(eclipsed_ethane)


def _ratios(smiles, xyz):
    return np.array([r for *_, r in bond_stretch_ratios(prepare_mol(smiles), xyz)])


def test_a_saddle_relaxes_to_a_lower_minimum_with_the_same_bonds(eclipsed_ethane, saddle_arrays):
    rec, arrays = eclipsed_ethane, saddle_arrays
    first, nu = arrays["record_frequencies_cm"], arrays["frequencies_cm"]
    assert (first < -10).sum() == 1 and first.min() < -200  # the torsion of the saddle
    assert nu.min() > 200  # real again: the staggered torsion, about 300 cm^-1
    assert (str(arrays["reopt_outcome"]), int(arrays["reopt_rounds"])) == ("REOPTIMIZED", 1)
    assert float(arrays["energy_ha"]) < rec.energy_ha == float(arrays["record_energy_ha"])
    drop = float(arrays["energy_drop_ha"]) * th.HARTREE_KCAL
    assert 1.5 < drop < 4.0
    mol, settings = prepare_mol(rec.atom_order_smiles), ro.ReoptSettings()
    same = ro.bond_state(mol, arrays["coords_ang"], settings)
    assert same == ro.bond_state(mol, rec.coords_ang, settings)
    stretch = _ratios(rec.smiles, arrays["coords_ang"]) - _ratios(rec.smiles, rec.coords_ang)
    assert np.abs(stretch).max() < 0.05
    assert (str(arrays["point_group"]), int(arrays["symmetry_number"])) == ("D3d", 6)
    warnings = str(arrays["warnings"]).split(",")
    assert "IMAGINARY_REOPTIMIZED" in warnings and "IMAGINARY_PERSISTENT" not in warnings
    r = th.thermo_at(tc.as_thermo(arrays), 298.15)
    assert r.modes.n_dropped_imaginary == 0 and r.energy_ha == float(arrays["energy_ha"])


def test_the_acetone_record_ends_without_a_mode_below_the_threshold(qm_record):
    """The pilot's acetone record is a methyl saddle (-76 cm^-1), and so is the fixture's."""
    arrays = tc.compute_thermo(qm_record(MOLECULES["acetone"]))
    assert str(arrays["reopt_outcome"]) in ("MINIMUM", "REOPTIMIZED")
    assert arrays["frequencies_cm"].min() > ro.REOPT_THRESHOLD_CM
    assert float(arrays["energy_drop_ha"]) >= 0.0


def test_a_minimum_is_left_untouched(qm_record, monkeypatch):
    monkeypatch.setattr(ro, "relax_direction", lambda *a, **k: pytest.fail("a minimum moved"))
    rec = qm_record(ETHANE)
    arrays = tc.compute_thermo(rec)
    assert (str(arrays["reopt_outcome"]), str(arrays["reopt_stop"])) == ("MINIMUM", "")
    assert int(arrays["reopt_rounds"]) == 0 and str(arrays["reopt_rejected"]) == ""
    assert np.array_equal(arrays["coords_ang"], rec.coords_ang)
    assert float(arrays["energy_ha"]) == rec.energy_ha and float(arrays["energy_drop_ha"]) == 0.0
    assert np.array_equal(arrays["frequencies_cm"], arrays["record_frequencies_cm"])
    assert "IMAGINARY" not in str(arrays["warnings"])


def test_the_command_never_writes_the_record(eclipsed_ethane, tmp_path, capsys):
    rec = eclipsed_ethane
    qm_dir, out, report = tmp_path / "qm", tmp_path / "thermo", tmp_path / "report.json"
    path = save_record(rec, cache_path(qm_dir, rec.smiles, 0, rec.method))
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    argv = ["thermo", "--smiles", ETHANE, "--cache-dir", str(qm_dir), "--out-dir", str(out)]
    assert main([*argv, "--report", str(report)]) == 0
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before
    assert [p.name for p in qm_dir.rglob("*") if p.is_file()] == [path.name]
    again = load_record(path, (rec.smiles, 0, rec.method))
    assert np.array_equal(again.coords_ang, rec.coords_ang) and again.energy_ha == rec.energy_ha
    [row] = json.loads(report.read_text())["rows"]
    assert (row["status"], row["reopt_outcome"], row["reopt_rounds"]) == ("ok", "REOPTIMIZED", 1)
    assert row["energy_drop_kcal"] > 1.5 and row["lowest_record_cm"] < -200 < row["lowest_cm"]
    side = tc.read_arrays(Path(row["path"]))
    assert np.abs(side["coords_ang"] - rec.coords_ang).max() > 0.1
    printed = capsys.readouterr().out
    assert "re-optimization along imaginary modes" in printed and "REOPTIMIZED" in printed


def _old_sidecar(arrays: dict, record_path: Path, config: tc.ThermoConfig) -> Path:
    """A sidecar as the code before 2026-11-10 wrote it: no re-optimization arrays, none of
    its keys in the header's config."""
    old = {k: v for k, v in arrays.items() if k in OLD_LAYOUT}
    method, cid = str(arrays["method"]), int(arrays["conformer_id"])
    full = tc.sidecar_config(method, cid, config)
    keys = {k: v for k, v in full.items() if not k.startswith("reopt_")}
    header = prov.stamp(
        tc.KIND, tc.THERMO_VERSION, prov.file_input(record_path), keys, prov.array_digest(old)
    )
    path = tc.thermo_path(config.out_dir, str(arrays["smiles"]), cid, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    return prov.save_npz(path, old, header)


def test_a_sidecar_from_before_the_reoptimization_reads_stale(qm_record, tmp_path):
    rec = qm_record(ETHANE)
    record_path = save_record(rec, cache_path(tmp_path / "qm", rec.smiles, 0, rec.method))
    config = tc.ThermoConfig(qm_dir=tmp_path / "qm", out_dir=tmp_path / "thermo")
    arrays = tc.compute_thermo(rec, config)
    path = _old_sidecar(arrays, record_path, config)
    assert not tc.is_current(path, rec.method, 0, config)
    with pytest.raises(prov.StaleArtifact, match="config reopt_threshold_cm: absent"):
        tc.load_thermo(path, (rec.method, 0), config)
    with pytest.raises(prov.StaleArtifact):
        tc.thermo_of(rec.smiles, config=config)
    task = thermo_batch.make_tasks([rec.smiles], QMConfig().method, config)[0]
    [row] = thermo_batch.process_task(task)  # recomputed, not up_to_date
    assert row["status"] == "ok" and row["reopt_outcome"] == "MINIMUM"
    assert tc.is_current(path, rec.method, 0, config) and "reopt_outcome" in tc.read_arrays(path)
    assert tc.thermo_of(rec.smiles, config=config).energy_ha == rec.energy_ha
    # other re-optimization settings make it stale again; a budget does not
    assert not tc.is_current(path, rec.method, 0, replace(config, reopt_threshold_cm=-20.0))
    assert not tc.is_current(path, rec.method, 0, replace(config, reopt_rounds=0))
    assert tc.is_current(path, rec.method, 0, replace(config, reopt_maxiter=50, timeout_s=10))
    # a reader of the old layout still reads the new sidecar: the change only adds
    old_keys = {
        k: v for k, v in tc.sidecar_config(rec.method, 0, config).items() if "reopt" not in k
    }
    prov.verify(path, tc.KIND, tc.THERMO_VERSION, old_keys)


def _fail(reason):
    def raise_(*args, **kw):
        raise QMError(reason, "test")

    return raise_


def test_no_kept_direction_falls_back_to_xtbs_rule(eclipsed_ethane, monkeypatch):
    monkeypatch.setattr(ro, "relax_direction", _fail(ReasonCode.CONNECTIVITY_CHANGED))
    arrays = tc.compute_thermo(eclipsed_ethane)
    outcome = (str(arrays["reopt_outcome"]), str(arrays["reopt_stop"]))
    assert outcome == ("PERSISTENT", "REOPT_NO_LOWER_MINIMUM")
    assert str(arrays["reopt_rejected"]) == "CONNECTIVITY_CHANGED,CONNECTIVITY_CHANGED"
    assert int(arrays["reopt_rounds"]) == 0 and float(arrays["energy_drop_ha"]) == 0.0
    assert np.array_equal(arrays["coords_ang"], eclipsed_ethane.coords_ang)
    assert np.array_equal(arrays["frequencies_cm"], arrays["record_frequencies_cm"])
    warnings = set(str(arrays["warnings"]).split(","))
    assert {"IMAGINARY_PERSISTENT", "IMAGINARY_DROPPED"} <= warnings
    assert "IMAGINARY_REOPTIMIZED" not in warnings
    r = th.thermo_at(tc.as_thermo(arrays), 298.15)
    assert r.modes.n_dropped_imaginary == 1  # xtb's rule: the torsion below -20 is left out


def test_a_direction_that_is_not_lower_is_not_kept(eclipsed_ethane, monkeypatch):
    up = (eclipsed_ethane.coords_ang, eclipsed_ethane.energy_ha + 1e-3)
    monkeypatch.setattr(ro, "relax_direction", lambda *a, **k: up)
    arrays = tc.compute_thermo(eclipsed_ethane)
    assert str(arrays["reopt_stop"]) == "REOPT_NO_LOWER_MINIMUM"
    assert str(arrays["reopt_rejected"]) == "NOT_LOWER,NOT_LOWER"


@pytest.mark.parametrize(
    ("change", "stop"),
    [({"reopt_rounds": 0}, "REOPT_OFF"), ({"timeout_s": 0}, "REOPT_BUDGET")],
)
def test_no_round_without_rounds_or_time(eclipsed_ethane, saddle_arrays, change, stop):
    arrays = tc.compute_thermo(eclipsed_ethane, replace(tc.ThermoConfig(), **change))
    assert (str(arrays["reopt_outcome"]), str(arrays["reopt_stop"])) == ("PERSISTENT", stop)
    record_nu = saddle_arrays["record_frequencies_cm"]  # another run: equal to 1e-3 cm^-1
    assert np.allclose(arrays["frequencies_cm"], record_nu, rtol=0, atol=1e-3)
    assert "IMAGINARY_PERSISTENT" in str(arrays["warnings"])


# ---- the rounds and the guard, without a Hessian


def _point(nu, xyz, energy):
    n3 = np.asarray(xyz).size
    modes = np.linalg.qr(np.random.default_rng(0).normal(size=(n3, len(nu))))[0]
    return ro.Point(np.asarray(xyz, dtype=float), energy, np.asarray(nu, float), modes, 0.0)


def test_the_rounds_end_when_a_mode_persists(qm_record, monkeypatch):
    rec = qm_record(ETHANE)
    seen = []

    def evaluate(xyz, energy):
        seen.append(energy)
        return _point([-50.0, 300.0, 900.0], xyz, energy)

    def lower(record, mol, base, point, masses, settings):
        return (point.xyz_ang, point.energy_ha - 1e-4), []

    monkeypatch.setattr(ro, "lower_minimum", lower)
    settings = ro.ReoptSettings(rounds=2)
    res = ro.follow_imaginary(rec, evaluate, np.ones(rec.n_atoms), settings)
    assert (res.outcome, res.stop, res.rounds) == (ro.Outcome.PERSISTENT, "REOPT_ROUNDS", 2)
    assert seen == pytest.approx([rec.energy_ha, rec.energy_ha - 1e-4, rec.energy_ha - 2e-4])


def test_a_failed_hessian_at_a_kept_geometry_keeps_the_last_good_one(qm_record, monkeypatch):
    rec = qm_record(ETHANE)

    def evaluate(xyz, energy):
        if energy < rec.energy_ha:
            raise QMError(ReasonCode.SCF_NOT_CONVERGED, "displaced geometry")
        return _point([-50.0, 300.0], xyz, energy)

    def lower(record, mol, base, point, masses, settings):
        return (point.xyz_ang, point.energy_ha - 1e-4), []

    monkeypatch.setattr(ro, "lower_minimum", lower)
    res = ro.follow_imaginary(rec, evaluate, np.ones(rec.n_atoms), ro.ReoptSettings())
    assert (res.outcome, res.stop, res.rounds) == (ro.Outcome.PERSISTENT, "REOPT_HESSIAN_FAILED", 0)
    assert res.rejected == ("SCF_NOT_CONVERGED",) and res.final.energy_ha == rec.energy_ha


def test_a_record_whose_drawing_is_not_its_atoms_has_no_guard(qm_record):
    rec = replace(qm_record(ETHANE), atom_order_smiles="CO")
    res = ro.follow_imaginary(
        rec, lambda x, e: _point([-50.0], x, e), np.ones(rec.n_atoms), ro.ReoptSettings()
    )
    assert (res.outcome, res.stop) == (ro.Outcome.PERSISTENT, "REOPT_NO_GRAPH")


def test_the_bond_guard_is_relative_to_the_record(qm_record):
    rec = qm_record(ETHANE)
    mol, settings = prepare_mol(rec.atom_order_smiles), ro.ReoptSettings()
    base = ro.bond_state(mol, rec.coords_ang, settings)
    assert base == ro.BondState(frozenset(), frozenset())
    close = rec.coords_ang.copy()
    side = np.cross(close[2] - close[0], close[1] - close[0])
    close[3] = close[2] + 0.3 * side / np.linalg.norm(side)  # two H of one carbon 0.3 A apart
    with pytest.raises(QMError) as hit:
        ro.check_bonds(mol, base, close, settings)
    assert hit.value.reason is ReasonCode.BOND_FORMED
    ro.check_bonds(mol, ro.bond_state(mol, close, settings), close, settings)  # the record's own
    apart = rec.coords_ang.copy()
    apart[5:8] += 3.0 * (apart[1] - apart[0])  # the second carbon's hydrogens pulled away
    with pytest.raises(QMError) as hit:
        ro.check_bonds(mol, base, apart, settings)
    assert hit.value.reason is ReasonCode.CONNECTIVITY_CHANGED


def test_the_displacement_leaves_every_saddle_at_once():
    rng = np.random.default_rng(3)
    masses = rng.uniform(1.0, 35.0, 6)
    modes = np.linalg.qr(rng.normal(size=(18, 3)))[0]
    d = ro.mode_displacement(modes, masses, 0.2)
    assert d.shape == (6, 3)
    assert np.linalg.norm(d, axis=1).max() == pytest.approx(0.2)
    mass_weighted = np.sqrt(np.repeat(masses, 3)) * d.reshape(-1)
    assert (modes.T @ mass_weighted > 0).all()


def test_the_thermo_command_takes_the_reopt_flags(tmp_path, capsys):
    argv = ["thermo", "--smiles", ETHANE, "--reopt-threshold", "-25", "--reopt-rounds", "1"]
    cfg = thermo_config(build_parser().parse_args([*argv, "--reopt-fmax", "2e-4"]))
    assert (cfg.reopt_threshold_cm, cfg.reopt_rounds, cfg.reopt_fmax) == (-25.0, 1, 2e-4)
    default = thermo_config(build_parser().parse_args(["thermo", "--smiles", ETHANE]))
    assert default.reopt_settings() == tc.ThermoConfig().reopt_settings()
    bad = ["thermo", "--smiles", ETHANE, "--reopt-threshold", "-0.5", "--out-dir", str(tmp_path)]
    assert main(bad) == 2
    assert "threshold must be below" in capsys.readouterr().out
    with pytest.raises(ValueError):
        tc.ThermoConfig(reopt_threshold_cm=-1.0)
