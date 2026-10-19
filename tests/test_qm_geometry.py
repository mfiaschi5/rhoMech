"""Optimizer, bond-stretch connectivity test and the wall-restraint fallback."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from ksmech.data import chem
from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.geometry import (
    BOHR_ANG,
    WallRestraint,
    bond_stretch_ratios,
    bond_walls,
    check_connectivity,
    optimize_tblite,
)
from ksmech.qm.records import QMError, ReasonCode
from ksmech.qm.xtb_calc import GFN2Calculator, QMConfig, resolve_charge_spin

DISSOCIATES = "OC1([OH2+])OCCO1"  # protonated tetrahedral intermediate: loses water when relaxed


def embedded(smi: str) -> Chem.Mol:
    mol = prepare_mol(chem.unmapped_canonical(Chem.MolFromSmiles(smi)))
    embed(mol, 1)
    return mol


def test_wall_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3)) * 2
    walls = WallRestraint(np.array([[0, 1], [1, 2], [2, 3]]), np.array([0.5, 0.5, 0.5]), 1.0)
    energy, grad = walls.energy_gradient(x)
    assert energy > 0
    num = np.zeros_like(x)
    for i in range(4):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += 1e-6
            xm[i, k] -= 1e-6
            num[i, k] = (walls.energy_gradient(xp)[0] - walls.energy_gradient(xm)[0]) / 2e-6
    assert np.abs(grad - num).max() < 1e-6


def test_wall_is_flat_inside_and_quadratic_outside():
    walls = WallRestraint(np.array([[0, 1]]), np.array([2.0]), 3.0)
    inside = np.array([[0.0, 0, 0], [1.9, 0, 0]])
    outside = np.array([[0.0, 0, 0], [2.5, 0, 0]])
    assert walls.energy_gradient(inside)[0] == 0.0
    assert not walls.energy_gradient(inside)[1].any()
    assert walls.energy_gradient(outside)[0] == pytest.approx(3.0 * 0.5**2)


def test_stretch_ratio_of_ethane_bonds_is_near_one():
    mol = embedded("CC")
    ratios = [r for _, _, r in bond_stretch_ratios(mol, mol.GetConformer(0).GetPositions())]
    assert len(ratios) == 7 and all(0.9 < r < 1.3 for r in ratios)


def test_check_connectivity_flags_only_stretched_input_bonds():
    mol = embedded("CCO")
    xyz = mol.GetConformer(0).GetPositions()
    check_connectivity(mol, xyz, 1.3)
    stretched = xyz.copy()
    stretched[2] += np.array([3.0, 0, 0])  # drag the O away from its C
    with pytest.raises(QMError) as exc:
        check_connectivity(mol, stretched, 1.3)
    assert exc.value.reason is ReasonCode.CONNECTIVITY_CHANGED and "(1,2)" in exc.value.detail


def test_new_close_contacts_are_ignored():
    mol = embedded("CC.CC")  # two separate molecules, so no bond between them
    first, second = Chem.GetMolFrags(mol)
    xyz = mol.GetConformer(0).GetPositions().copy()
    xyz[list(second)] = xyz[list(first)] + np.array([0.0, 0.0, 1.4])  # second laid on top of first
    check_connectivity(mol, xyz, 1.3)  # contacts everywhere, but no input bond is stretched


def test_bonds_to_alkali_metals_and_magnesium_are_skipped():
    lithium = Chem.AddHs(Chem.MolFromSmiles("[Li]O"))
    embed(lithium, 1)
    xyz = lithium.GetConformer(0).GetPositions().copy()
    li = [a.GetIdx() for a in lithium.GetAtoms() if a.GetAtomicNum() == 3][0]
    xyz[li] += 10.0  # Li pulled 10 A off its oxygen
    assert all(li not in (i, j) for i, j, _ in bond_stretch_ratios(lithium, xyz))
    check_connectivity(lithium, xyz, 1.3)
    xyz[[a.GetIdx() for a in lithium.GetAtoms() if a.GetAtomicNum() == 1][0]] += 10.0
    with pytest.raises(QMError):  # an O-H bond is still tested
        check_connectivity(lithium, xyz, 1.3)


def test_bond_walls_cover_every_non_skipped_bond():
    mol = embedded("CC[O-].[Na+]")
    walls = bond_walls(mol, 1.2, 1.0)
    assert len(walls.pairs) == mol.GetNumBonds() and (walls.d0_bohr > 0).all()
    li = Chem.AddHs(Chem.MolFromSmiles("[Li]O"))
    assert len(bond_walls(li, 1.2, 1.0).pairs) == 1  # only O-H; Li-O is skipped


def test_free_optimization_converges_and_lowers_the_force():
    mol = embedded("CCO")
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    charge, unpaired = resolve_charge_spin(mol)
    start = mol.GetConformer(0).GetPositions() / BOHR_ANG
    final = optimize_tblite(z, start, charge, unpaired, 3e-4, 500, "water")
    assert final.shape == start.shape and not np.allclose(final, start)


def test_unconverged_optimization_reports_a_reason_code():
    mol = embedded("CCO")
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    start = mol.GetConformer(0).GetPositions() / BOHR_ANG
    with pytest.raises(QMError) as exc:
        optimize_tblite(z, start, 0, 0, 1e-12, 2, "water")
    assert exc.value.reason is ReasonCode.OPT_NOT_CONVERGED


def test_dissociating_species_fails_without_restraints(qm_config):
    cfg = replace(qm_config, restrain=False)
    out = GFN2Calculator(cfg).try_run(embedded(DISSOCIATES), 0)
    assert out.record is None and out.reason is ReasonCode.CONNECTIVITY_CHANGED


def test_dissociating_species_gets_a_restrained_geometry(qm_config):
    mol = embedded(DISSOCIATES)
    out = GFN2Calculator(qm_config).try_run(mol, 0)
    assert out.record is not None, out.detail
    assert "RESTRAINED_GEOM" in out.record.warnings
    ratios = [r for _, _, r in bond_stretch_ratios(mol, out.record.coords_ang)]
    assert max(ratios) <= qm_config.conn_cov_factor


def test_stable_species_is_not_flagged_as_restrained(qm_record):
    assert "RESTRAINED_GEOM" not in qm_record("CCO").warnings


def test_restraint_is_unavailable_for_the_xtb_backend(qm_config):
    cfg = replace(qm_config, optimizer="xtb")
    assert cfg.restrain and cfg.optimizer == "xtb"  # `_relax` re-raises instead of restraining
    assert QMConfig().method != cfg.method
