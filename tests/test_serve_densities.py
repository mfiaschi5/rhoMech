"""serve/densities.py: the canonical cube is `ao_values @ C[:, k]`; at 0.2 Angstrom with enough
padding |psi|^2 integrates to 1, the density to N_valence, f+ / f- to 1 and the dual to 0
(within 1%); the point-charge ESP is positive around NH4+, negative around acetate and Q/r far
away; quality sets the spacing; requests are normalized; the errors NO_FRONTIER,
ORBITAL_OUT_OF_RANGE, CUBE_TOO_LARGE and REQUEST_INVALID; cubes are cached with metadata."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ksmech.serve import cubes
from ksmech.serve import densities as D
from ksmech.serve import orbital_sets as osets
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_cubes import _read_cube
from tests.test_serve_fixtures_wb import (  # noqa: F401 (fixtures)
    no_xtb,
    synthetic_root,
    wb_key,
    wb_runtime,
    wb_settings,
)

pytestmark = pytest.mark.serve

FINE = 0.2  # Angstrom
PAD = 5.0  # Angstrom: the valence tails are below the grid's resolution beyond this


def _integral(rt, key: str, kind: str, index=None, set_=None, start=None) -> tuple[float, dict]:
    rec, _, _ = osets.load_qm(rt, key)
    req = D.cube_request(kind, set_, start, index, None, FINE, PAD)
    g = cubes.grid_around(rec.coords_ang, FINE, PAD)
    evaluate, info = D.surface(rt, key, rec, req)
    return D.grid_integral(kind, cubes.on_grid(g, evaluate), g), info


def _cube(rt, key: str, **request) -> tuple[dict, dict, Path]:
    req = D.cube_request(**request)
    out = D.t_wb_cube(rt, {"key": key, "request": req.to_payload()})
    path = Path(out["path"])
    return _read_cube(path.read_text()), json.loads(Path(out["meta"]).read_text()), path


def test_canonical_cube_is_ao_values_times_c(wb_runtime):  # noqa: F811
    from ksmech.serve.gto import ao_values

    key = "CCO"
    rec, _, _ = osets.load_qm(wb_runtime, key)
    k = int(np.flatnonzero(np.asarray(rec.occupations) >= 0.5).max())  # the HOMO
    cube, meta, _ = _cube(
        wb_runtime, key, kind="orbital", set_="canonical", index=k, spacing=0.3, padding=3.0
    )
    g = cubes.grid_around(rec.coords_ang, 0.3, 3.0)
    assert cube["n"] == g.n and cube["natoms"] == rec.n_atoms
    pts = cubes.grid_points(g, 0, g.n_points)
    exact = ao_values(rec.atomic_numbers, rec.coords_ang, pts) @ rec.C[:, k]
    np.testing.assert_allclose(cube["values"], exact, rtol=1e-5, atol=1e-10)
    first, second = cube["comments"]
    assert "kind=orbital" in first and f"index={k}" in first and "set=canonical" in first
    assert "values bohr^-3/2" in second and "integral (of psi^2)=" in second
    assert meta["units"] == "bohr^-3/2" and meta["expected"] == 1.0 and meta["index"] == k


def test_lmo_cube_is_the_token_coefficients(wb_runtime):  # noqa: F811
    from ksmech.serve.gto import ao_values

    key = "CCO"
    t = wb_runtime.loader(key)
    cube, meta, _ = _cube(
        wb_runtime, key, kind="orbital", set_="ibo", index=2, spacing=0.35, padding=2.5
    )
    rec, _, _ = osets.load_qm(wb_runtime, key)
    g = cubes.grid_around(rec.coords_ang, 0.35, 2.5)
    exact = ao_values(rec.atomic_numbers, rec.coords_ang, cubes.grid_points(g, 0, g.n_points))
    np.testing.assert_allclose(
        cube["values"], exact @ t.orbitals[2].coeff_vector, rtol=1e-5, atol=1e-10
    )
    assert meta["set"] == "ibo" and meta["start"] == "cholesky"


@pytest.mark.parametrize(
    ("kind", "index", "set_"),
    [("orbital", 0, "canonical"), ("orbital", 4, "canonical"), ("orbital_density", 3, "ibo")],
)
def test_orbital_norm_is_one(wb_runtime, kind, index, set_):  # noqa: F811
    value, info = _integral(wb_runtime, "CO", kind, index, set_)
    assert info["expected"] == 1.0 and abs(value - 1.0) < 0.01


@pytest.mark.parametrize("smiles", ["CO", "CC(=O)[O-]", "[NH4+]"])
def test_density_and_fukui_integrals(wb_runtime, smiles):  # noqa: F811
    key = wb_key(wb_runtime, smiles)
    rec, _, _ = osets.load_qm(wb_runtime, key)
    n_val = float(np.sum(rec.occupations))
    rho, info = _integral(wb_runtime, key, "density")
    assert info["expected"] == n_val and abs(rho - n_val) < 0.01 * n_val
    for kind in ("fukui_minus", "fukui_plus"):
        assert abs(_integral(wb_runtime, key, kind)[0] - 1.0) < 0.01
    assert abs(_integral(wb_runtime, key, "dual")[0]) < 0.01


def test_esp_signs_and_far_field(wb_runtime):  # noqa: F811
    for smiles, sign in (("[NH4+]", 1.0), ("CC(=O)[O-]", -1.0)):
        key = wb_key(wb_runtime, smiles)
        rec, _, _ = osets.load_qm(wb_runtime, key)
        g = cubes.grid_around(rec.coords_ang, 0.4, 4.0)
        pts = cubes.grid_points(g, 0, g.n_points)
        far = np.linalg.norm(pts[:, None] - rec.coords_ang[None], axis=2).min(axis=1) > 2.0
        esp = D.esp_at(rec.coords_ang, rec.charges, pts[far])
        assert np.all(sign * esp > 0), (smiles, esp.min(), esp.max())
        centre = rec.coords_ang.mean(axis=0)
        probe = centre + np.array([[100.0, 0.0, 0.0], [0.0, -80.0, 30.0]])
        r_bohr = np.linalg.norm(probe - centre, axis=1) / cubes.bohr_ang()
        np.testing.assert_allclose(
            D.esp_at(rec.coords_ang, rec.charges, probe) * r_bohr, rec.charge, rtol=0.02
        )
    near = D.esp_at(np.zeros((1, 3)), np.array([1.0]), np.zeros((1, 3)))
    assert near[0] == pytest.approx(1.0 / D.ESP_FLOOR_BOHR)


def test_esp_cube_is_labelled_approximate(wb_runtime):  # noqa: F811
    cube, meta, _ = _cube(wb_runtime, "CO", kind="esp", spacing=0.4, padding=2.0)
    assert meta["integral"] is None and meta["units"] == "hartree/e"
    assert "approximate ESP" in cube["comments"][1] and "integral=none" in cube["comments"][1]
    assert meta["set"] is None and meta["index"] is None


def test_quality_sets_the_spacing(wb_runtime):  # noqa: F811
    rec, _, _ = osets.load_qm(wb_runtime, "CCO")
    for quality, target in D.QUALITY_POINTS.items():
        sp = D.spacing_for(rec.coords_ang, quality, 3.0)
        n = cubes.grid_around(rec.coords_ang, sp, 3.0).n_points
        assert 0.6 * target < n <= 1.2 * target and round(sp, 3) == sp
    _, meta, _ = _cube(wb_runtime, "CCO", kind="density", quality="draft")
    assert meta["quality"] == "draft" and meta["spacing"] is None
    assert meta["spacing_ang"] == D.spacing_for(rec.coords_ang, "draft", D.DEFAULT_PADDING_ANG)
    assert abs(meta["integral"] - meta["expected"]) < 0.05 * meta["expected"]


def test_cube_cache(wb_runtime, wb_settings):  # noqa: F811
    _, _, path = _cube(wb_runtime, "CCO", kind="fukui_plus", spacing=0.4, padding=2.0)
    assert path.parent == wb_settings.workbench_cubes
    mtime = path.stat().st_mtime_ns
    _, _, again = _cube(wb_runtime, "CCO", kind="fukui_plus", spacing=0.4, padding=2.0)
    assert again == path and path.stat().st_mtime_ns == mtime
    _, _, other = _cube(wb_runtime, "CCO", kind="fukui_plus", spacing=0.4, padding=2.5)
    assert other != path


def test_request_normalization():
    a = D.cube_request("density", "ibo", "canonical", 3, "fine", None, None)
    assert (a.set, a.start, a.index, a.quality, a.padding) == (None, None, None, "fine", 3.0)
    b = D.cube_request("orbital", None, None, 2, "draft", 0.25, 4.0)
    assert (b.set, b.start, b.quality, b.spacing) == ("canonical", None, None, 0.25)
    c = D.cube_request("orbital_density", "pm_mulliken", None, 0)
    assert (c.set, c.start, c.quality) == ("pm_mulliken", "cholesky", "normal")
    assert D.CubeRequest.from_payload(json.loads(json.dumps(c.to_payload()))) == c


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "electron_density"},
        {"kind": "orbital"},  # no index
        {"kind": "density", "quality": "ultra"},
        {"kind": "density", "spacing": 0.0},
        {"kind": "density", "padding": -1.0},
        {"kind": "density", "spacing": math.nan},
        {"kind": "orbital", "index": 0, "set_": "boys"},
    ],
)
def test_request_errors(kwargs):
    with pytest.raises(ServeError) as info:
        D.cube_request(**kwargs)
    assert info.value.code == Code.REQUEST_INVALID


def test_orbital_out_of_range(wb_runtime):  # noqa: F811
    rec, _, _ = osets.load_qm(wb_runtime, "CCO")
    for set_ in ("canonical", "pm_mulliken"):
        with pytest.raises(ServeError) as info:
            _cube(wb_runtime, "CCO", kind="orbital", set_=set_, index=rec.C.shape[1])
        assert info.value.code == Code.ORBITAL_OUT_OF_RANGE
        assert info.value.detail == {"index": rec.C.shape[1], "n": rec.C.shape[1]}


def test_no_frontier(wb_runtime):  # noqa: F811
    rec, _, _ = osets.load_qm(wb_runtime, "CCO")
    full = replace(rec, occupations=np.full(len(rec.occupations), 2.0))  # no virtual MO
    empty = replace(rec, occupations=np.zeros(len(rec.occupations)))  # no electron
    for r, kind, level in ((full, "fukui_plus", "lumo"), (empty, "fukui_minus", "homo")):
        for k in (kind, "dual"):
            with pytest.raises(ServeError) as info:
                D.surface(wb_runtime, "CCO", r, D.cube_request(k))
            assert info.value.code == Code.NO_FRONTIER and info.value.status == 422
            assert info.value.detail["level"] == level
    with pytest.raises(ServeError) as info:
        D.frontier_columns(full, "lumo")
    assert info.value.detail["level"] == "lumo"


def test_cube_too_large(wb_runtime):  # noqa: F811
    with pytest.raises(ServeError) as info:
        _cube(wb_runtime, "CCO", kind="density", spacing=0.01, padding=3.0)
    assert info.value.code == Code.CUBE_TOO_LARGE
