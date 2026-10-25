"""The molecule workbench over HTTP (synthetic app, InlineScheduler): POST /api/molecule with a
molblock or a SMILES, the molecule's status, geometry, properties (NaN as null), every orbital
set and every cube kind with its headers; 202 Pending then 200 for featurize and for a cube;
the error codes (404, 422, 409 never reached here), id validation and the capability gate."""

from __future__ import annotations

import pytest

from ksmech.serve import schemas as S
from ksmech.serve.capabilities import CAPABILITIES
from tests.test_serve_api import error_of
from tests.test_serve_fixtures_api import api_client
from tests.test_serve_fixtures_wb import (  # noqa: F401 (fixtures)
    DeferredScheduler,
    no_xtb,
    synthetic_root,
    wb_settings,
)
from tests.test_serve_molecule import molblock

pytestmark = pytest.mark.serve

SETS = [
    ("canonical", None),
    ("ibo", "cholesky"),
    ("ibo", "canonical"),
    ("pm_mulliken", "cholesky"),
    ("pm_mulliken", "canonical"),
]
KINDS = [
    ("orbital", {"index": 4}),
    ("orbital", {"index": 4, "set": "pm_mulliken", "start": "canonical"}),
    ("orbital_density", {"index": 2, "set": "ibo"}),
    ("density", {}),
    ("fukui_minus", {}),
    ("fukui_plus", {}),
    ("dual", {}),
    ("esp", {}),
]
FAST = {"spacing": 0.4, "padding": 2.5}


@pytest.fixture(scope="module")
def app(wb_settings):  # noqa: F811
    with api_client(wb_settings) as client:
        yield client


@pytest.fixture(scope="module")
def methanol(app):
    r = app.post("/api/molecule", json={"molblock": molblock("CO", order=[1, 0])})
    assert r.status_code == 200, r.text
    return r.json()


def test_post_and_get(app, methanol):
    out = S.MoleculeOut.model_validate(methanol)
    assert out.status == "ready" and out.key == "CO" and out.chemistry == "synthetic"
    assert out.mapping is not None and out.mapping.sketch_to_record == [1, 0]
    assert out.mapping.n_record_atoms == out.n_atoms == 6
    same = app.post("/api/molecule", json={"smiles": "OC"}).json()
    assert same["molecule_id"] == out.molecule_id and same["mapping"]["sketch_to_record"] == [1, 0]
    got = app.get(f"/api/molecule/{out.molecule_id}").json()
    assert got["status"] == "ready" and got["mapping"] is None


def test_geometry(app, methanol):
    r = app.get(f"/api/molecule/{methanol['molecule_id']}/geometry")
    assert r.status_code == 200, r.text
    g = S.MoleculeGeometryOut.model_validate(r.json())
    assert [a.symbol for a in g.atoms] == ["C", "O", "H", "H", "H", "H"]
    assert len(g.bonds) == 5 and g.xyz.splitlines()[0] == "6" and "M  END" in g.molblock


def test_properties(app, methanol):
    r = app.get(f"/api/molecule/{methanol['molecule_id']}/properties")
    assert r.status_code == 200, r.text
    body = r.json()
    p = S.MoleculePropertiesOut.model_validate(body)
    gp = p.global_props
    assert gp.n_valence_electrons == 14.0 and gp.multiplicity == 1 and gp.charge == 0
    assert gp.gap_ev == pytest.approx(gp.lumo_ev - gp.homo_ev) and gp.pm_converged
    assert len(p.atoms) == 6 and p.fmo_levels == ["homo_minus_1", "homo", "lumo", "lumo_plus_1"]
    assert body["atoms"][0]["f_plus_fd"] is None  # synthetic chemistry has no FD Fukui: NaN
    assert all(a.diagnostic.diagnostic for a in p.atoms)
    assert sum(a.mulliken_charge for a in p.atoms) == pytest.approx(0.0, abs=1e-6)
    lewis = [b for b in p.bonds if b.bonded]
    assert len(lewis) == 5 and all(b.order == 1 for b in lewis)
    co = next(b for b in lewis if {b.i, b.j} == {0, 1})
    assert 0.7 < co.wiberg < 1.3 and 1.2 < co.length_ang < 1.6
    assert all(b.wiberg >= p.wiberg_threshold for b in p.bonds if not b.bonded)
    assert p.geometry.key == "CO" and p.provenance.qm_digest


@pytest.mark.parametrize(("set_", "start"), SETS)
def test_orbital_sets(app, methanol, set_, start):
    params = {"set": set_} | ({"start": start} if start else {})
    r = app.get(f"/api/molecule/{methanol['molecule_id']}/orbitals", params=params)
    assert r.status_code == 200, r.text
    o = S.OrbitalSetOut.model_validate(r.json())
    assert o.set == set_ and o.start == start and len(o.orbitals) == 12  # one per AO: C 4, O 4, 4 H
    assert o.stats.orthonormality_error < 1e-9 and o.stats.occupied_projector_error < 1e-9
    assert o.is_model_set == ((set_, start) == ("ibo", "cholesky"))
    want = {"canonical": "canonical", "ibo": "token_file" if o.is_model_set else "workbench"}
    assert o.source == want.get(set_, "workbench")


@pytest.mark.parametrize(("kind", "params"), KINDS)
def test_every_cube_kind(app, methanol, kind, params):
    r = app.get(
        f"/api/molecule/{methanol['molecule_id']}/cube", params={"kind": kind} | params | FAST
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["x-cube-kind"] == kind and r.headers["content-disposition"].startswith(
        "inline"
    )
    lines = r.text.splitlines()
    assert lines[0].startswith("ksmech-serve workbench cube v1") and f"kind={kind}" in lines[0]
    assert r.headers["x-cube-units"] in lines[1]
    if kind == "esp":
        assert r.headers["x-cube-integral"] == "none"
    else:
        expected = float(r.headers["x-cube-expected"])
        assert abs(float(r.headers["x-cube-integral"]) - expected) < 0.05 * max(1.0, expected)


def test_download_header(app, methanol):
    params = {"kind": "density", "download": "true"} | FAST
    r = app.get(f"/api/molecule/{methanol['molecule_id']}/cube", params=params)
    assert r.headers["content-disposition"] == (
        f"attachment; filename={methanol['molecule_id']}-density.cube"
    )


def test_pending_then_ready(wb_settings):  # noqa: F811
    from dataclasses import replace

    from ksmech.serve.settings import TaskEnv

    settings = replace(wb_settings, interactive_wait_s=0.01)
    sched = DeferredScheduler(TaskEnv(settings))
    with api_client(settings, scheduler=sched) as client:
        created = client.post("/api/molecule", json={"smiles": "CCCOC"})
        assert created.status_code == 200 and created.json()["status"] == "running"
        mid = created.json()["molecule_id"]
        assert client.get(f"/api/molecule/{mid}/geometry").status_code == 202
        assert sched.drain() == 1 and sched.names == ["featurize"]
        assert client.get(f"/api/molecule/{mid}").json()["status"] == "ready"
        url = f"/api/molecule/{mid}/cube"
        params = {"kind": "density"} | FAST
        r = client.get(url, params=params)
        assert r.status_code == 202, r.text
        pending = S.PendingOut.model_validate(r.json())
        assert "density cube of CCCOC" in pending.label and r.headers["retry-after"]
        assert client.get(url, params=params).json()["task_id"] == pending.task_id
        assert sched.drain() == 1
        done = client.get(url, params=params)
        assert done.status_code == 200 and done.text.startswith("ksmech-serve workbench cube")
        assert client.get(url, params=params).status_code == 200  # cached: no task
        assert sched.names == ["featurize", "wb_cube"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"smiles": "CCO.O"}, "MULTIPLE_FRAGMENTS"),
        ({"molblock": "garbage"}, "INVALID_MOLFILE"),
        ({"smiles": "C1CC"}, "INVALID_SMILES"),
        ({"smiles": "[CH3]"}, "RADICAL_NOT_SUPPORTED"),
        ({"smiles": "CC", "molblock": "x"}, "REQUEST_INVALID"),
        ({"smiles": "CC", "extra": 1}, "REQUEST_INVALID"),
    ],
)
def test_post_errors(app, body, code):
    error_of(app.post("/api/molecule", json=body), 422, code)


def test_get_errors(app, methanol):
    mid = methanol["molecule_id"]
    error_of(app.get("/api/molecule/%2E%2E"), 404, "MOLECULE_NOT_FOUND")
    error_of(app.get("/api/molecule/M0123456789abcdef0123"), 404, "MOLECULE_NOT_FOUND")
    error_of(app.get(f"/api/molecule/m{'0' * 20}/properties"), 404, "MOLECULE_NOT_FOUND")
    cube = f"/api/molecule/{mid}/cube"
    error_of(app.get(cube, params={"kind": "nope"}), 422, "REQUEST_INVALID")
    error_of(app.get(cube, params={"kind": "orbital"}), 422, "REQUEST_INVALID")
    error_of(app.get(cube, params={"kind": "orbital", "index": 12}), 422, "ORBITAL_OUT_OF_RANGE")
    error_of(app.get(cube, params={"kind": "orbital", "index": -1}), 422, "REQUEST_INVALID")
    error_of(app.get(cube, params={"kind": "density", "spacing": 0.005}), 422, "CUBE_TOO_LARGE")
    orbitals = f"/api/molecule/{mid}/orbitals"
    error_of(app.get(orbitals, params={"set": "boys"}), 422, "REQUEST_INVALID")
    error_of(app.get(orbitals, params={"set": "ibo", "start": "x"}), 422, "REQUEST_INVALID")


def test_workbench_capability(app, monkeypatch, methanol):
    from ksmech.serve import capabilities

    meta = app.get("/api/meta").json()
    assert meta["capabilities"]["workbench"] == {"enabled": True, "reason": None}
    assert CAPABILITIES["workbench"] == (True, None)
    monkeypatch.setitem(capabilities.CAPABILITIES, "workbench", (False, "test"))
    mid = methanol["molecule_id"]
    for method, path in (
        ("post", "/api/molecule"),
        ("get", f"/api/molecule/{mid}"),
        ("get", f"/api/molecule/{mid}/geometry"),
        ("get", f"/api/molecule/{mid}/properties"),
        ("get", f"/api/molecule/{mid}/orbitals"),
        ("get", f"/api/molecule/{mid}/cube?kind=density"),
    ):
        kwargs = {"json": {"smiles": "C"}} if method == "post" else {}
        err = error_of(getattr(app, method)(path, **kwargs), 503, "FEATURE_DISABLED")
        assert err["detail"] == {"capability": "workbench", "reason": "test"}
