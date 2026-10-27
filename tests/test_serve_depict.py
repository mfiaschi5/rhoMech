"""depict.py and geometry.py: the server molblock and atom table, the server SVG with exact
GetDrawCoords, step depictions of the mini PMechDB steps, and per-species geometry."""

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from ksmech.serve.chem import geometry
from ksmech.serve.core.errors import Code, ServeError
from ksmech.serve.depiction import depict
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


@pytest.fixture(scope="module")
def sn2_rt(sn2_state, synthetic_loader):  # noqa: F811
    from ksmech.arrows.reactants import ReactantTokens

    return ReactantTokens.from_smiles(sn2_state.mapped_smiles, synthetic_loader)


def _molblock_mol(dep: dict) -> Chem.Mol:
    mol = Chem.MolFromMolBlock(dep["molblock"], removeHs=False, sanitize=False)
    assert mol is not None
    return mol


def _h_maps(state) -> list[int]:
    from ksmech.data import chem

    mol = chem.parse_mapped(state.mapped_smiles)
    return [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]


def test_molblock_matches_the_atom_table_and_has_no_maps(sn2_state, sn2_rt):  # noqa: F811
    dep = depict.pool_depiction(sn2_state, sn2_rt, set(), server_svg=False)
    mol = _molblock_mol(dep)
    assert mol.GetNumAtoms() == len(dep["atoms"]) == 4  # C, O, C, Cl: every H hidden
    assert all(a.GetAtomMapNum() == 0 for a in mol.GetAtoms())
    assert [a["depiction_idx"] for a in dep["atoms"]] == list(range(4))
    assert [a.GetSymbol() for a in mol.GetAtoms()] == [a["element"] for a in dep["atoms"]]
    conf = mol.GetConformer()
    assert not conf.Is3D() and np.ptp(conf.GetPositions()[:, :2], axis=0).max() > 0.5
    assert dep["ref"] == sn2_state.state_id and dep["server_svg"] is None
    assert sorted(a["hidden_h"] for a in dep["atoms"]) == [0, 0, 3, 3]
    assert {a["element"]: a["hidden_h"] for a in dep["atoms"] if a["element"] != "C"} == {
        "O": 0,
        "Cl": 0,
    }


def test_hydrogens_kept_only_for_keep_maps(sn2_state, sn2_rt):  # noqa: F811
    h_maps = _h_maps(sn2_state)
    keep = {h_maps[0], h_maps[-1]}
    dep = depict.pool_depiction(sn2_state, sn2_rt, keep, server_svg=False)
    hs = [a for a in dep["atoms"] if a["element"] == "H"]
    assert sorted(a["map"] for a in hs) == sorted(keep)
    assert _molblock_mol(dep).GetNumAtoms() == len(dep["atoms"]) == 6
    assert sum(a["hidden_h"] for a in dep["atoms"]) == len(h_maps) - 2
    everything = depict.pool_depiction(sn2_state, sn2_rt, set(h_maps), server_svg=False)
    assert len(everything["atoms"]) == sn2_state.n_atoms
    assert all(a["hidden_h"] == 0 for a in everything["atoms"])


def test_atom_table_agrees_with_global_atoms(sn2_state, sn2_rt):  # noqa: F811
    from ksmech.data import chem

    dep = depict.pool_depiction(sn2_state, sn2_rt, set(_h_maps(sn2_state)[:1]), False)
    by_map = chem.map_to_index(sn2_rt.mol)
    for a in dep["atoms"]:
        mol_idx = by_map[a["map"]]
        assert a["global_site"] == sn2_rt.global_atom[mol_idx]
        assert a["fragment"] == int(sn2_rt.atom_fragment[a["global_site"]])
        assert sn2_rt.mol.GetAtomWithIdx(mol_idx).GetSymbol() == a["element"]
    no_tokens = depict.pool_depiction(sn2_state, None, set(), False)
    assert all(a["global_site"] is None for a in no_tokens["atoms"])
    assert [a["map"] for a in no_tokens["atoms"]] == [
        a["map"] for a in depict.pool_depiction(sn2_state, sn2_rt, set(), False)["atoms"]
    ]


def test_server_svg_gives_draw_coords_of_every_atom(sn2_state, sn2_rt):  # noqa: F811
    dep = depict.pool_depiction(sn2_state, sn2_rt, set(), server_svg=True)
    assert dep["server_svg"].lstrip().startswith("<?xml") and "<svg" in dep["server_svg"]
    coords = dep["server_coords"]
    assert len(coords) == len(dep["atoms"])
    for x, y in coords:
        assert math.isfinite(x) and math.isfinite(y)
        assert 0 <= x <= depict.SVG_WIDTH and 0 <= y <= depict.SVG_HEIGHT
    # the same molblock drawn again gives the same exact positions
    drawer = rdMolDraw2D.MolDraw2DSVG(depict.SVG_WIDTH, depict.SVG_HEIGHT)
    drawer.drawOptions().prepareMolsBeforeDrawing = False
    mol = _molblock_mol(dep)
    mol.UpdatePropertyCache(strict=False)
    drawer.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False, addChiralHs=False))
    drawer.FinishDrawing()
    again = [(drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y) for i in range(len(coords))]
    np.testing.assert_allclose(np.array(coords), np.array(again), atol=1e-6)
    assert len({(round(x), round(y)) for x, y in coords}) == len(coords)


@pytest.mark.parametrize(
    "label", ["sn2", "proton_transfer", "carbonyl_addition", "e2", "resonance"]
)
def test_step_depiction_of_mini_pmechdb_steps(mini_steps, label):
    from ksmech.data import chem

    step = mini_steps[label]
    keep = {a for arrow in step.arrows for a in (*arrow.source_atoms, *arrow.sink_atoms)}
    for side in (step.reactants_smiles, step.products_smiles):
        dep = depict.step_depiction(side, keep, server_svg=True)
        assert dep["ref"].startswith("smiles:")
        assert _molblock_mol(dep).GetNumAtoms() == len(dep["atoms"]) == len(dep["server_coords"])
        assert all(a["global_site"] is None for a in dep["atoms"])
        mol = chem.parse_mapped(side)
        drawn = {a["map"] for a in dep["atoms"] if a["map"]}
        assert drawn == {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()}
        assert keep & drawn == keep & {a.GetAtomMapNum() for a in mol.GetAtoms()}
    explicit = depict.step_depiction(mini_steps["proton_transfer"].reactants_smiles, {20}, False)
    assert [a["element"] for a in explicit["atoms"] if a["map"] == 20] == ["H"]
    hidden = depict.step_depiction(mini_steps["proton_transfer"].reactants_smiles, set(), False)
    assert 20 not in {a["map"] for a in hidden["atoms"]}
    named = depict.step_depiction(mini_steps["sn2"].reactants_smiles, set(), False, ref="x:1")
    assert named["ref"] == "x:1"


def test_step_depiction_rejects_bad_smiles():
    with pytest.raises(ServeError) as err:
        depict.step_depiction("C1CC", set(), False)
    assert err.value.code == Code.INVALID_SMILES


def test_lone_and_bridging_hydrogens_stay():
    dep = depict.step_depiction("[H-].[H][H].[OH-]", set(), False)
    assert [a["element"] for a in dep["atoms"]] == ["H", "H", "H", "O"]
    assert dep["atoms"][-1]["hidden_h"] == 1


def test_species_geometry_xyz_molblock_and_offsets(sn2_rt):
    rows = geometry.species_geometry(sn2_rt)
    assert [r["token_key"] for r in rows] == [f.key for f in sn2_rt.fragments]
    assert {r["species_key"] for r in rows} == {"[CH3][O]|-1", "[CH3][Cl]|0"}
    x_end = None
    for f, row in enumerate(rows):
        t = sn2_rt.fragments[f].tokens
        pos = np.array([a.position for a in t.atoms])
        lines = row["xyz"].splitlines()
        assert int(lines[0]) == len(t.atoms) == len(row["atoms"])
        assert lines[1].startswith(f"{row['token_key']} charge=") and "mult=1" in lines[1]
        assert f"method={t.method}" in lines[1]
        xyz = np.array([[float(v) for v in ln.split()[1:]] for ln in lines[2:]])
        np.testing.assert_allclose(xyz, pos, atol=5e-7)
        mol = Chem.MolFromMolBlock(row["molblock"], removeHs=False)
        assert mol.GetNumAtoms() == len(t.atoms)
        np.testing.assert_allclose(mol.GetConformer().GetPositions(), pos, atol=1e-4)
        assert Chem.GetFormalCharge(mol) == (-1 if row["token_key"] == "C[O-]" else 0)
        assert len(row["bonds"]) == mol.GetNumBonds()
        for a in row["atoms"]:
            assert a["global_site"] == sn2_rt.atom_offsets[f] + a["record_index"]
            assert a["map"] == geometry.map_of(sn2_rt, a["global_site"]) is not None
        x = pos[:, 0] + row["display_offset"][0]
        if x_end is None:
            assert row["display_offset"] == (0.0, 0.0, 0.0)
        else:
            assert x.min() == pytest.approx(x_end + geometry.FRAGMENT_GAP_ANG)
        x_end = x.max()
        assert "display" in row["frame_note"]
    ccl = rows[[r["token_key"] for r in rows].index("CCl")]
    assert (0, 1, 1) in ccl["bonds"]


def test_aromatic_geometry_is_kekulized(synthetic_settings, synthetic_loader):  # noqa: F811
    from ksmech.arrows.reactants import ReactantTokens
    from ksmech.serve.chem.pools import SpeciesInput, initial_state

    state = initial_state([SpeciesInput("c1ccccc1", 1)], synthetic_settings.limits)
    rt = ReactantTokens.from_smiles(state.mapped_smiles, synthetic_loader)
    (row,) = geometry.species_geometry(rt)
    ring = [order for i, j, order in row["bonds"] if row["atoms"][i]["Z"] == row["atoms"][j]["Z"]]
    assert sorted(ring) == [1, 1, 1, 2, 2, 2]


def test_orbital_position_and_subset_molblock(sn2_rt):
    f, k = geometry.orbital_position(sn2_rt, sn2_rt.n_atoms)
    assert (f, k) == (0, 0)
    last = sn2_rt.n_tokens - 1
    assert geometry.orbital_position(sn2_rt, last) == (
        len(sn2_rt.fragments) - 1,
        len(sn2_rt.fragments[-1].tokens.orbitals) - 1,
    )
    for bad in (0, sn2_rt.n_atoms - 1, sn2_rt.n_tokens):
        with pytest.raises(ServeError) as err:
            geometry.orbital_position(sn2_rt, bad)
        assert err.value.code == Code.NOT_AN_ORBITAL_TOKEN
    frag = sn2_rt.fragments[1]
    pos = np.array([a.position for a in frag.tokens.atoms])
    block = geometry.molblock_subset(frag.record, [0, 1], pos[[0, 1]])
    sub = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
    assert sub.GetNumAtoms() == 2 and sub.GetNumBonds() == 1
    with pytest.raises(ValueError):
        geometry.molblock_subset(frag.record, [1, 0], pos[[1, 0]])
