"""cubes.py: the grid on the token's cap, the Gaussian cube in bohr, exact LMO values, the
placed acceptor's back-transform, and the CUBE_TOO_LARGE guard."""

import math
from pathlib import Path

import numpy as np
import pytest

from ksmech.serve import cubes, placement
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_fixtures import (  # noqa: F401 (fixtures)
    no_xtb,
    sn2_state,
    synthetic_loader,
    synthetic_root,
    synthetic_runtime,
    synthetic_settings,
)

pytestmark = pytest.mark.serve


def _bohr() -> float:
    from ksmech.qm.geometry import BOHR_ANG

    return BOHR_ANG


@pytest.fixture(scope="module")
def sn2_rt(synthetic_runtime, sn2_state):  # noqa: F811
    return synthetic_runtime.reactant_tokens(sn2_state)


def _token_id(rt, key: str, kind: str, elements: set[str]) -> tuple[int, int, int]:
    """(global token id, fragment, position) of the first token of `kind` with those
    centre elements in fragment `key`."""
    f = [fr.key for fr in rt.fragments].index(key)
    t = rt.fragments[f].tokens
    for k, o in enumerate(t.orbitals):
        centres = {t.symbol(int(a)) for a in o.descriptors.all_center_atoms}
        if o.kind.value == kind and centres == elements:
            return rt.n_atoms + int(rt.orbital_offsets[f]) + k, f, k
    raise AssertionError((key, kind, elements))


def _expected_n(points: np.ndarray, spacing: float, padding: float) -> tuple[int, ...]:
    ext = points.max(axis=0) - points.min(axis=0) + 2 * padding
    return tuple(int(math.ceil(e / spacing - 1e-9)) + 1 for e in ext)


def _read_cube(text: str) -> dict:
    lines = text.splitlines()
    natoms, *origin = lines[2].split()
    axes = [ln.split() for ln in lines[3:6]]
    atoms = [ln.split() for ln in lines[6 : 6 + int(natoms)]]
    body = lines[6 + int(natoms) :]
    return {
        "comments": lines[:2],
        "natoms": int(natoms),
        "origin": np.array(origin, dtype=float),
        "n": tuple(int(a[0]) for a in axes),
        "steps": np.array([[float(x) for x in a[1:]] for a in axes]),
        "atoms": atoms,
        "body": body,
        "values": np.array([float(v) for ln in body for v in ln.split()]),
    }


def test_orbital_cube_header_grid_and_values(synthetic_runtime, sn2_state, sn2_rt):  # noqa: F811
    from ksmech.physics.gto import ao_values
    from ksmech.physics.oriented import cap_atoms

    token, f, k = _token_id(sn2_rt, "C[O-]", "occupied", {"O"})
    payload = {"state": sn2_state.to_dict(), "token": token, "spacing": 0.2, "padding": 4.0}
    path = Path(cubes.t_orbital_cube(synthetic_runtime, payload)["path"])
    assert path.parent == synthetic_runtime.settings.cubes and path.suffix == ".cube"
    cube = _read_cube(path.read_text())
    first, units = cube["comments"]
    assert first.startswith(f"ksmech-serve cube v1 state={sn2_state.state_id} token={token}")
    assert "kind=occ" in first and "E_r=" in first
    assert units == cubes.UNITS_COMMENT
    t = sn2_rt.fragments[f].tokens
    z, xyz = cubes.fragment_arrays(t)
    cap = xyz[cap_atoms(t.orbitals[k], t)]
    b = _bohr()
    assert cube["natoms"] == len(t.atoms)
    np.testing.assert_allclose(cube["origin"], (cap.min(axis=0) - 4.0) / b, atol=1e-6)
    assert cube["n"] == _expected_n(cap, 0.2, 4.0)
    np.testing.assert_allclose(cube["steps"], np.eye(3) * 0.2 / b, atol=1e-6)
    for row, zi, p in zip(cube["atoms"], z, xyz, strict=True):
        assert int(row[0]) == zi and float(row[1]) == float(zi)
        np.testing.assert_allclose([float(v) for v in row[2:]], p / b, atol=1e-6)
    nx, ny, nz = cube["n"]
    assert len(cube["body"]) == nx * ny * math.ceil(nz / 6)
    assert all(len(ln) == 13 * len(ln.split()) for ln in cube["body"][:50])
    g = cubes.grid_around(cap)
    exact = ao_values(z, xyz, cubes.grid_points(g, 0, g.n_points)) @ t.orbitals[k].coeff_vector
    np.testing.assert_allclose(cubes.lmo_values(t, k, g), exact, rtol=0, atol=1e-13)
    np.testing.assert_allclose(cube["values"], exact, rtol=1e-5, atol=1e-12)
    assert np.abs(exact).max() > 0.1  # the lobe is on the grid


def test_grid_order_and_points():
    g = cubes.grid_around(np.array([[0.0, 0.0, 0.0], [1.0, 0.4, 0.0]]), 0.5, 1.0)
    assert g.n == (7, 6, 5)
    pts = cubes.grid_points(g, 0, g.n_points)
    np.testing.assert_allclose(pts[0], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(pts[1], [-1.0, -1.0, -0.5])  # z innermost
    np.testing.assert_allclose(pts[g.n[2]], [-1.0, -0.5, -1.0])
    np.testing.assert_allclose(pts[-1], [2.0, 1.5, 1.0])
    np.testing.assert_allclose(cubes.grid_points(g, 7, 9), pts[7:9])


def test_virtual_cube_and_cache(synthetic_runtime, sn2_state, sn2_rt):  # noqa: F811
    token, _, _ = _token_id(sn2_rt, "CCl", "virtual", {"C", "Cl"})
    payload = {"state": sn2_state.to_dict(), "token": token, "spacing": 0.3, "padding": 3.0}
    path = Path(cubes.t_orbital_cube(synthetic_runtime, payload)["path"])
    assert "kind=virt" in path.read_text().splitlines()[0]
    mtime = path.stat().st_mtime_ns
    assert cubes.t_orbital_cube(synthetic_runtime, payload)["path"] == str(path)
    assert path.stat().st_mtime_ns == mtime
    other = cubes.t_orbital_cube(synthetic_runtime, payload | {"spacing": 0.25})["path"]
    assert other != str(path)


def test_cube_too_large_and_bad_boxes(synthetic_runtime, sn2_state, sn2_rt):  # noqa: F811
    with pytest.raises(ServeError) as err:
        cubes.grid_around(np.zeros((1, 3)), spacing=0.01, padding=4.0)
    assert err.value.code == Code.CUBE_TOO_LARGE and err.value.status == 422
    assert err.value.detail["points"] > cubes.MAX_POINTS
    token, _, _ = _token_id(sn2_rt, "C[O-]", "occupied", {"O"})
    payload = {"state": sn2_state.to_dict(), "token": token, "spacing": 0.02, "padding": 4.0}
    with pytest.raises(ServeError) as err:
        cubes.t_orbital_cube(synthetic_runtime, payload)
    assert err.value.code == Code.CUBE_TOO_LARGE
    for spacing, padding in ((0.0, 4.0), (-0.2, 4.0), (0.2, -1.0), (float("nan"), 4.0)):
        with pytest.raises(ServeError) as err:
            cubes.grid_around(np.zeros((1, 3)), spacing, padding)
        assert err.value.code == Code.REQUEST_INVALID
    with pytest.raises(ServeError) as err:
        cubes.t_orbital_cube(synthetic_runtime, payload | {"token": 0, "spacing": 0.2})
    assert err.value.code == Code.NOT_AN_ORBITAL_TOKEN


@pytest.mark.parametrize(("spacing", "padding"), [(1e-308, 4.0), (5e-324, 1.0), (0.2, 1e308)])
def test_cube_ratio_overflow_is_too_large(spacing, padding):
    """An extent / spacing that overflows to inf (or a box beyond MAX_POINTS on one axis) is
    CUBE_TOO_LARGE (422) before any int() of it, never an OverflowError (500)."""
    with pytest.raises(ServeError) as err:
        cubes.grid_around(np.zeros((1, 3)), spacing=spacing, padding=padding)
    e = err.value
    assert e.code == Code.CUBE_TOO_LARGE and e.status == 422
    assert e.detail == {"spacing": spacing, "padding": padding, "max_points": cubes.MAX_POINTS}


def test_cube_point_count_is_exact():
    """3,000,001 points per axis: the count is the exact integer, not an int64 product."""
    spacing = 8.0 / 3_000_000
    with pytest.raises(ServeError) as err:
        cubes.grid_around(np.zeros((1, 3)), spacing=spacing, padding=4.0)
    assert err.value.detail["n"] == [3_000_001] * 3
    assert err.value.detail["points"] == 3_000_001**3
    g = cubes.grid_around(np.zeros((1, 3)), spacing=0.2, padding=4.0)  # unchanged
    assert g.n == (41, 41, 41) and g.n_points == 41**3


def test_cube_task_answers_too_large_for_a_subnormal_spacing(synthetic_runtime, sn2_state, sn2_rt):  # noqa: F811
    token, _, _ = _token_id(sn2_rt, "C[O-]", "occupied", {"O"})
    payload = {"state": sn2_state.to_dict(), "token": token, "spacing": 1e-308, "padding": 4.0}
    with pytest.raises(ServeError) as err:
        cubes.t_orbital_cube(synthetic_runtime, payload)
    assert err.value.code == Code.CUBE_TOO_LARGE and err.value.status == 422


def test_placed_acceptor_back_transform(synthetic_runtime, sn2_rt):  # noqa: F811
    """At 50 random points the back-transformed values equal the rotated coefficients on the
    placed atoms (1e-10); the cap of sigma*(C-Cl) carries Cl d shells."""
    from ksmech.physics.gto import ao_values

    cfg = synthetic_runtime.kernels.main.config.oriented
    _, f, r = _token_id(sn2_rt, "C[O-]", "occupied", {"O"})
    _, g, s = _token_id(sn2_rt, "CCl", "virtual", {"C", "Cl"})
    t_F, t_G = sn2_rt.fragments[f].tokens, sn2_rt.fragments[g].tokens
    assert 2 in set(t_G.ao_l.tolist())  # d functions on Cl
    rng = np.random.default_rng(7)
    for b in (int(x) for x in t_G.orbitals[s].descriptors.all_center_atoms):
        pl = placement.reconstruct_contact(t_F, r, 1, t_G, s, b, cfg)
        lo, hi = pl.cap_xyz_ang.min(axis=0) - 2.0, pl.cap_xyz_ang.max(axis=0) + 2.0
        X = lo + rng.random((50, 3)) * (hi - lo)
        back = cubes.placed_cap_values_at(t_G, s, pl, X)
        z_cap = np.array([t_G.atoms[int(i)].atomic_number for i in pl.cap_atoms])
        c_placed = placement.placed_cap_coefficients(t_G, s, pl.rot)
        direct = ao_values(z_cap, pl.cap_xyz_ang, X) @ c_placed
        np.testing.assert_allclose(back, direct, rtol=0, atol=1e-10)
        assert np.abs(direct).max() > 1e-3
        g_box = cubes.grid_around(pl.cap_xyz_ang, 0.5, 1.0)
        on_grid = cubes.placed_cap_values(t_G, s, pl, g_box)
        pts = cubes.grid_points(g_box, 0, g_box.n_points)
        np.testing.assert_allclose(on_grid, cubes.placed_cap_values_at(t_G, s, pl, pts), atol=1e-14)


def test_write_cube_layout():
    g = cubes.Grid(np.array([0.0, 0.0, 0.0]), (1, 2, 7), 0.5)
    text = cubes.write_cube(g, [1], np.zeros((1, 3)), np.arange(14.0), ("a\nb", "c"))
    lines = text.splitlines()
    assert lines[0] == "a b" and lines[1] == "c"
    assert lines[2].split() == ["1", "0.000000", "0.000000", "0.000000"]
    assert lines[5].split()[0] == "7"
    body = lines[7:]
    assert [len(ln.split()) for ln in body] == [6, 1, 6, 1]  # a break after each z column
    assert float(body[2].split()[0]) == 7.0
