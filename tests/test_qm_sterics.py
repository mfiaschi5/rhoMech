"""%Vbur per atom: the shared-grid implementation gives morfeus' own values."""

import numpy as np
import pytest
from morfeus import BuriedVolume

from ksmech.qm.embed import embed, prepare_mol
from ksmech.qm.sterics import DENSITY, RADIUS_ANG, buried_volume_per_atom, sphere_tree
from tests.fixtures.molecules import MOLECULES, OPEN_SHELL


def morfeus_per_atom(z, xyz, radius=RADIUS_ANG, density=DENSITY):
    """The previous implementation: one morfeus BuriedVolume per centre atom."""
    return np.array(
        [
            100.0
            * BuriedVolume(
                [int(a) for a in z],
                xyz,
                metal_index=i + 1,
                radius=radius,
                include_hs=True,
                density=density,
            ).fraction_buried_volume
            for i in range(len(z))
        ]
    )


def geometry(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    mol = prepare_mol(smiles)
    cid = embed(mol, 1, seed=0)[0]
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return z, mol.GetConformer(cid).GetPositions()


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_equals_morfeus_on_the_fixture_molecules(name):
    z, xyz = geometry(MOLECULES[name])
    assert np.array_equal(buried_volume_per_atom(z, xyz), morfeus_per_atom(z, xyz))


@pytest.mark.parametrize("name", ["tempo", "benzyl", "copper_dichloride", "tetrabromoferrate"])
def test_equals_morfeus_on_open_shell_and_metal_fixtures(name):
    z, xyz = geometry(OPEN_SHELL[name][0])
    assert np.array_equal(buried_volume_per_atom(z, xyz), morfeus_per_atom(z, xyz))


def test_radius_and_density_reach_the_grid():
    z, xyz = geometry(MOLECULES["acetone"])
    got = buried_volume_per_atom(z, xyz, radius=3.0, density=0.01)
    assert np.array_equal(got, morfeus_per_atom(z, xyz, radius=3.0, density=0.01))
    assert not np.array_equal(got, buried_volume_per_atom(z, xyz))


def test_a_lone_atom_and_far_apart_atoms_bury_nothing():
    assert buried_volume_per_atom(np.array([35]), np.zeros((1, 3))).tolist() == [0.0]
    far = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    assert buried_volume_per_atom(np.array([17, 17]), far).tolist() == [0.0, 0.0]


def test_one_grid_serves_every_call():
    assert sphere_tree(RADIUS_ANG, DENSITY) is sphere_tree(RADIUS_ANG, DENSITY)
