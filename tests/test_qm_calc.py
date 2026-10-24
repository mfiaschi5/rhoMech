"""GFN2Calculator: charge/spin resolution, units, matrices, descriptors, failure codes."""

import shutil
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from tblite.interface import Calculator

from ksmech.qm.ao import aos_per_element
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.fukui import HARTREE_TO_EV
from ksmech.qm.records import ReasonCode
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig, resolve_charge_spin


def mol_for(smi: str) -> Chem.Mol:
    from ksmech.data import chem

    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smi)))
    embed(mol, 1)
    return mol


@pytest.mark.parametrize(
    ("smi", "charge", "unpaired"),
    [("CCO", 0, 0), ("C[O-]", -1, 0), ("CC[NH3+]", 1, 0), ("[CH3]", 0, 1), ("[CH2]", 0, 2)],
)
def test_charge_and_unpaired_electrons_come_from_rdkit(smi, charge, unpaired):
    assert resolve_charge_spin(Chem.AddHs(Chem.MolFromSmiles(smi))) == (charge, unpaired)


def test_impossible_spin_state_is_flagged():
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
    mol.GetAtomWithIdx(0).SetNumRadicalElectrons(0)  # 9 electrons, no unpaired one
    from ksmech.qm.records import QMError

    with pytest.raises(QMError) as exc:
        resolve_charge_spin(mol)
    assert exc.value.reason is ReasonCode.SPIN_PARITY


def test_record_shapes_and_units(qm_record):
    rec = qm_record("CC")
    n_ao = 4 * 2 + 1 * 6  # two C (s,p) and six H
    assert rec.n_atoms == 8 and rec.S.shape == (n_ao, n_ao) == rec.C.shape == rec.fock.shape
    assert rec.orbital_energies_ev.shape == rec.occupations.shape == (n_ao,)
    assert list(np.bincount(rec.ao_atom)) == [4, 4, 1, 1, 1, 1, 1, 1]
    assert list(rec.ao_l[:4]) == [0, 1, 1, 1]
    cc = np.linalg.norm(rec.coords_ang[0] - rec.coords_ang[1])
    assert 1.45 < cc < 1.6, "coordinates must be in Angstrom"
    # energies in eV: compare with an independent tblite call at the stored geometry
    calc = Calculator(
        "GFN2-xTB", rec.atomic_numbers, rec.coords_ang / 0.529177210903, charge=0.0, uhf=0
    )
    calc.set("verbosity", 0)
    calc.add("alpb-solvation", "water")
    eps = calc.singlepoint().get("orbital-energies") * HARTREE_TO_EV
    assert np.allclose(rec.orbital_energies_ev, eps, atol=1e-6)
    assert rec.homo_ev < rec.lumo_ev


def test_matrices_are_consistent(qm_record):
    rec = qm_record("CCO")
    assert np.allclose(rec.S, rec.S.T) and np.allclose(rec.fock, rec.fock.T)
    assert np.abs(rec.C.T @ rec.S @ rec.C - np.eye(rec.C.shape[1])).max() < 1e-6
    eps_ha = rec.orbital_energies_ev / HARTREE_TO_EV
    assert np.allclose(rec.fock @ rec.C, rec.S @ rec.C * eps_ha, atol=1e-10)
    assert np.allclose(rec.H_core, rec.H_core.T)
    assert not np.allclose(rec.H_core, rec.fock)  # core H is not the converged Fock


def test_charges_and_fukui_sum_rules(qm_record):
    rec = qm_record("C[O-]")
    assert rec.charge == -1 and rec.charges.sum() == pytest.approx(-1.0, abs=1e-8)
    for arr in (rec.fukui_plus, rec.fukui_minus, rec.fukui_zero):
        assert arr.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(rec.fukui_dual, rec.fukui_plus - rec.fukui_minus)
    assert rec.fukui_fmo_plus.sum() == pytest.approx(1.0, abs=1e-6)
    assert rec.fukui_fmo_minus.sum() == pytest.approx(1.0, abs=1e-6)


def test_radical_is_open_shell_with_fractional_somo(qm_record):
    rec = qm_record("[CH3]")
    assert rec.unpaired == 1
    assert np.isclose(rec.occupations[rec.occupations < 1.5].max(), 1.0, atol=1e-3)
    assert rec.fukui_plus.sum() == pytest.approx(1.0, abs=1e-6)  # N+1 / N-1 used 0 unpaired


def test_bare_cation_has_no_frontier_orbitals(qm_record):
    rec = qm_record("[Na+]")
    assert "NO_FRONTIER" in rec.warnings
    assert np.isnan(rec.homo_ev) and np.isnan(rec.fukui_plus).all()
    assert rec.n_atoms == 1 and rec.buried_volume[0] == 0.0


def test_buried_volume_is_a_percentage_per_atom(qm_record):
    rec = qm_record("CC(=O)C")
    assert rec.buried_volume.shape == (rec.n_atoms,)
    assert np.isfinite(rec.buried_volume).all()
    assert (rec.buried_volume > 5).all() and (rec.buried_volume < 100).all()
    assert rec.buried_volume[1] > rec.buried_volume[0]  # carbonyl C is more crowded than a CH3 C


def test_ao_counts_match_tblite_for_every_element_in_the_fixtures():
    assert [aos_per_element(z) for z in (1, 6, 7, 8, 17)] == [1, 4, 4, 4, 9]


def test_solvation_changes_energy_and_cache_identity(qm_config, qm_record):
    gas = replace(qm_config, solvent=None)
    assert gas.method != qm_config.method and "ALPB-water" in qm_config.method
    assert qm_record("C[O-]", gas).energy_ha > qm_record("C[O-]").energy_ha  # water stabilizes


def test_try_run_never_raises_and_reports_reason_codes(qm_config):
    calc = GFN2Calculator(qm_config)
    out = calc.try_run(mol_for("COC=[NH+][CuH2-]"), 0)  # 63 electrons, 0 unpaired (PMechDB)
    assert out.record is None and out.reason is ReasonCode.SPIN_PARITY
    out = calc.try_run(Chem.AddHs(Chem.MolFromSmiles("CC")), 0)  # no conformer at all
    assert out.record is None and out.reason is ReasonCode.UNEXPECTED


def test_atom_order_must_come_from_prepare_mol(qm_config):
    mol = Chem.AddHs(Chem.MolFromSmiles("OCC"))  # canonical SMILES parses as CCO: other order
    embed(mol, 1)
    with pytest.raises(ValueError, match="prepare_mol"):
        GFN2Calculator(qm_config).run(mol, 0)


def test_invalid_optimizer_is_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        QMConfig(optimizer="orca")


@pytest.mark.skipif(shutil.which("xtb") is not None, reason="xtb binary is installed")
def test_xtb_backend_without_binary_reports_reason_code(qm_config):
    out = GFN2Calculator(replace(qm_config, optimizer="xtb")).try_run(mol_for("CCO"), 0)
    assert out.reason is ReasonCode.XTB_NOT_FOUND


@pytest.mark.skipif(shutil.which("xtb") is None, reason="needs the xtb binary")
def test_xtb_backend_agrees_with_tblite_backend(qm_config, qm_record):
    xtb = GFN2Calculator(replace(qm_config, optimizer="xtb")).run(mol_for("CCO"), 0)
    assert abs(xtb.energy_ha - qm_record("CCO").energy_ha) < 1e-3


def test_convergence_limits_are_not_part_of_the_method():
    """--maxiter and --scf-maxiter relax convergence only: records keep their cache key."""
    base = QMConfig()
    relaxed = replace(base, maxiter=2000, scf_maxiter=1000, timeout_s=3600)
    assert relaxed.method == base.method


def test_the_scf_cycle_limit_reaches_tblite():
    from ksmech.qm.records import QMError
    from ksmech.qm.xtb_calc import _single_point

    z = np.array([8, 1, 1])
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])  # water, Bohr
    assert _single_point(z, xyz, 0, 0, "water", False).get("energy") < 0
    with pytest.raises(QMError) as exc:
        _single_point(z, xyz, 0, 0, "water", False, scf_maxiter=1)
    assert exc.value.reason == ReasonCode.SCF_NOT_CONVERGED


def test_qm_cli_passes_the_convergence_flags(monkeypatch, tmp_path):
    import ksmech.qm.cli as cli

    seen = {}

    def fake(source, **kw):
        seen["config"] = kw["config"]
        import pandas as pd

        return pd.DataFrame({"status": ["ok"]})

    monkeypatch.setattr(cli, "featurize_dataset", fake)
    monkeypatch.setattr(cli, "summarize", lambda m: {"success_fraction": 1.0})
    flags = ["--timeout", "3600", "--maxiter", "2000", "--scf-maxiter", "1000"]
    assert cli.main(["run", "--source", "pmechdb", *flags]) == 0
    cfg = seen["config"]
    assert (cfg.timeout_s, cfg.maxiter, cfg.scf_maxiter) == (3600, 2000, 1000)
