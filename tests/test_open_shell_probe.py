"""scripts/probes/open_shell_probe.py (open-shell plan P2 step 5): every literature IP/EA value
carries its citation, and the delta-SCF levels it is compared with are gas-phase."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "open_shell_probe", ROOT / "scripts" / "probes" / "open_shell_probe.py"
)
probe = importlib.util.module_from_spec(_spec)
sys.modules["open_shell_probe"] = probe  # dataclasses look their module up while being made
_spec.loader.exec_module(probe)


def test_every_literature_value_carries_its_citation():
    assert set(probe.IPEA_LITERATURE) == set(probe.IPEA_RADICALS)
    for name, (ie, ie_cite, ea, ea_cite) in probe.IPEA_LITERATURE.items():
        assert 0 < ea < ie, name
        assert ie_cite.strip() and ea_cite.strip(), name


def test_the_delta_scf_levels_are_gas_phase(monkeypatch, tmp_path):
    solvents = []

    def record_for(smiles, u, solvent, root, given=None):
        solvents.append(solvent)
        return SimpleNamespace(atomic_numbers=np.array([6]), coords_ang=np.zeros((1, 3)), charge=0)

    def single_point(z, xyz, charge, u, solvent, spin_polarized):
        solvents.append(solvent)
        return {"energy_ha": -1.0 - 0.1 * charge}

    monkeypatch.setattr(probe, "record_for", record_for)
    monkeypatch.setattr(probe, "single_point", single_point)
    monkeypatch.setattr(
        probe, "cached", lambda root, smiles, method, compute: compute() | {"status": "ok"}
    )
    row = probe.ipea_species("CH3", tmp_path)
    assert row["status"] == "ok" and row["sp"]["IE_ev"] is not None
    assert solvents and all(s is None for s in solvents)


def test_ipea_comparison_reads_the_shift_and_the_rank():
    names = list(probe.IPEA_LITERATURE)
    rows = []
    for k, name in enumerate(names):
        ie, _, ea, _ = probe.IPEA_LITERATURE[name]
        level = {"IE_ev": ie + 5.0 + 0.01 * k, "EA_ev": ea + 4.0}
        rows.append({"name": name, "status": "ok", "restricted": level, "sp": level})
    out = probe.ipea_comparison(rows)
    assert abs(out["sp_EA_ev"]["mean_signed_error"] - 4.0) < 1e-9
    assert out["sp_EA_ev"]["mad_shift_removed"] < 1e-9
    assert out["restricted_IE_ev"]["n"] == len(names)


def _bent(smiles: str, angle_deg: float, d: float = 1.5) -> tuple:
    """A three-atom chain 0-1-2 with both bonds `d` Angstrom at `angle_deg`."""
    from rdkit import Chem

    t = np.radians(angle_deg)
    xyz = np.array([[d, 0.0, 0.0], [0.0, 0.0, 0.0], [d * np.cos(t), d * np.sin(t), 0.0]])
    return Chem.MolFromSmiles(smiles), xyz


def test_formed_bonds_reads_a_ring_closure_and_leaves_metals_out():
    mol, closed = _bent("[CH2]C[CH2]", 60.0)
    assert [(i, j) for i, j, _ in probe.formed_bonds(mol, closed)] == [(0, 2)]
    mol, open_ = _bent("[CH2]C[CH2]", 112.0)
    assert probe.formed_bonds(mol, open_) == []
    for smiles in ("[Cu]CO", "[Na]OC"):  # a metal-ligand contact is not a formed bond
        mol, xyz = _bent(smiles, 60.0)
        assert probe.formed_bonds(mol, xyz) == [], smiles


def test_ch3_and_ethane_reproduce(tmp_path):
    """Step 1's reproduction checks at the engine's embedding: [CH3] restricted SOMO -8.911 eV,
    spGFN2 alpha HOMO -9.687 / beta LUMO -8.150 eV; ethane's spGFN2 and restricted energies
    are equal (closed shell, no spin to polarize)."""
    known = {"somo": -8.911, "alpha_homo": -9.687, "beta_lumo": -8.150}
    got = probe.ch3_known(tmp_path)
    assert all(abs(got[k] - v) < 0.0005 for k, v in known.items()), got
    mol = probe.prepare_mol("CC")
    probe.embed(mol, 1, seed=0)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    xyz = mol.GetConformer(0).GetPositions() / probe.BOHR_ANG
    e = [probe.single_point(z, xyz, 0, 0, probe.SOLVENT, sp)["energy_ha"] for sp in (False, True)]
    assert abs(e[1] - e[0]) * probe.HA_EV < 0.005
