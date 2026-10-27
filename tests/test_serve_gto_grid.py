"""serve/chem/gto_grid.py: the separable evaluation on a cube grid equals `gto.ao_values @ C`
(s, p and d shells, several columns, blocked terms and columns), and the square sums equal the
squared columns summed."""

from __future__ import annotations

import numpy as np
import pytest

from ksmech.serve.chem import cubes, gto_grid
from ksmech.serve.chem.gto import ao_values

pytestmark = pytest.mark.serve

# H, C, N, O and the d shells of S, Cl and Br
Z = np.array([6, 1, 1, 8, 7, 16, 17, 35])


def _system(seed: int = 0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(len(Z), 3)) * 1.4
    nao = ao_values(Z, xyz, np.zeros((1, 3))).shape[1]
    g = cubes.grid_around(xyz, 0.37, 1.5)  # a non-cubic box
    return xyz, rng.normal(size=(nao, 5)), g


def _reference(xyz, C, g) -> np.ndarray:
    pts = cubes.grid_points(g, 0, g.n_points)
    return (ao_values(Z, xyz, pts) @ C).T.reshape(-1, *g.n)


def test_columns_equal_ao_values_times_c():
    xyz, C, g = _system()
    assert len(set(g.n)) > 1
    got = gto_grid.columns_on_grid(Z, xyz, C, g.origin_ang, g.n, g.spacing_ang)
    ref = _reference(xyz, C, g)
    assert got.shape == ref.shape == (C.shape[1], *g.n)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_blocks_give_the_same_values(monkeypatch):
    xyz, C, g = _system(1)
    whole = gto_grid.square_sum_on_grid(Z, xyz, C, g.origin_ang, g.n, g.spacing_ang, 0.5)
    monkeypatch.setattr(gto_grid, "BLOCK_BYTES", 8 * g.n[0] * g.n[1] * 3)  # 3 terms, 1 column
    blocked = gto_grid.square_sum_on_grid(Z, xyz, C, g.origin_ang, g.n, g.spacing_ang, 0.5)
    ref = 0.5 * np.sum(_reference(xyz, C, g) ** 2, axis=0)
    np.testing.assert_allclose(whole, ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(blocked, ref, rtol=1e-12, atol=1e-14)


def test_a_single_column_vector():
    xyz, C, g = _system(2)
    got = gto_grid.columns_on_grid(Z, xyz, C[:, 0], g.origin_ang, g.n, g.spacing_ang)
    ref = _reference(xyz, C[:, :1], g)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
