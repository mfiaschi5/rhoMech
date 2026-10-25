"""The sketcher's 2D depiction (POST /api/depict2d, serve/depiction2d.py) and its perspective
template library (serve/depiction_templates.py, serve/data/ring_templates.*): RDKit only, never
xTB (match3d gets an RDKit structure here, and the synthetic chemistry's record over HTTP).

Checked: every template parses (and RDKit's own loader accepts the file), its `fit` is a real
3D shape seen as the picture, the picture and its slots follow the drawing rules (bond lengths,
angles, atom-bond clearance, crossings) and so do the substituted examples drawn on it; it
matches and is used for its examples; the answer keeps the input atom and bond order and the
stereo; camphor gets RDKit's ring template (textbook) and the norbornane template
(perspective); match3d follows the view; the error codes."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ksmech.serve import depiction2d as D
from ksmech.serve import depiction_metrics as M
from ksmech.serve import depiction_templates as T
from ksmech.serve import molecule
from ksmech.serve.depiction_gallery import principal_frame, slot_angles
from ksmech.serve.errors import Code, ServeError
from tests.test_serve_fixtures_api import api_client
from tests.test_serve_fixtures_wb import (  # noqa: F401 (fixtures)
    no_xtb,
    synthetic_root,
    wb_settings,
)
from tests.test_serve_molecule import molblock

pytestmark = pytest.mark.serve

CAMPHOR = "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"
MENTHOL = "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"
STEREO = [CAMPHOR, MENTHOL, "N[C@@H](C)C(=O)O", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"]


def parsed(smiles: str, **kw) -> tuple[Chem.Mol, Chem.Mol]:
    return D.parse_molblock(molblock(smiles, **kw))


def code_of(fn, *args) -> str:
    with pytest.raises(ServeError) as info:
        fn(*args)
    return str(info.value.code)


# ---- the template library


def test_library_parses_and_rdkit_accepts_the_file():
    lib = T.load_library()
    assert len(lib) >= 9 and [t.size for t in lib] == sorted((t.size for t in lib), reverse=True)
    T.validate_with_rdkit()  # raises on a template RDKit's loader refuses; restores the defaults
    for t in lib:
        assert t.source and t.convention and t.examples, t.name
        assert np.isfinite(t.xyz).all() and t.xyz.shape == (t.size, 3)
        assert np.isfinite(t.fit).all() and t.fit.shape == (t.size, 3)


@pytest.mark.parametrize("t", T.load_library(), ids=lambda t: t.name)
def test_template_fit_is_a_3d_structure_and_depth_its_z(t):
    lengths = [np.linalg.norm(t.fit[a] - t.fit[b]) for a, b in t.bonds]
    assert np.std(lengths) / np.mean(lengths) < 0.03, t.name  # one bond length in 3D
    assert np.allclose(t.xyz[:, 2], t.fit[:, 2], atol=1e-3)  # depth agrees with the picture
    rms = np.sqrt(
        ((t.fit[:, :2] - t.fit[:, :2].mean(0) - t.xyz[:, :2] + t.xyz[:, :2].mean(0)) ** 2)
        .sum(axis=1)
        .mean()
    )
    assert rms < 0.3 * T.BOND, t.name  # the fit is a view of the picture


@pytest.mark.parametrize("t", T.load_library(), ids=lambda t: t.name)
def test_template_picture_follows_the_drawing_rules(t):
    rep = M.rule_report(t.xyz[:, :2], t.bonds, t.foreshortened)
    assert rep.violations() == [], (t.name, rep)
    assert rep.crossings == t.crossings, t.name
    drawn = [np.linalg.norm(t.xyz[a, :2] - t.xyz[b, :2]) for a, b in t.bonds]
    assert 0.9 * T.BOND < np.median(drawn) < 1.1 * T.BOND
    lo, hi = slot_angles(t)  # with a stub on every drawn slot
    assert lo >= M.RULE_ANGLE_MIN and hi <= M.RULE_ANGLE_MAX, (t.name, lo, hi)


def test_chair_slots_are_axial_vertical_and_equatorial_parallel():
    t = next(t for t in T.load_library() if t.name == "cyclohexane-chair")
    lines = [T._unit(t.xyz[b, :2] - t.xyz[a, :2]) for a, b in t.bonds]
    for k in range(t.size):
        axial, eq = sorted(t.slots[k], key=lambda s: -abs(s.d[1]))
        assert abs(axial.d[0]) < 1e-9
        assert max(abs(float(eq.d @ v)) for v in lines) > 1 - 1e-9


SUBSTITUTED = {
    "camphor": CAMPHOR,
    "borneol": "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](O)C2",
    "norcamphor": "O=C1C[C@@H]2CC[C@H]1C2",
    "memantine": "CC12CC3CC(C)(C1)CC(N)(C3)C2",
    "1-adamantanol": "OC12CC3CC(CC(C3)C1)C2",
    "quinuclidine": "C1CN2CCC1CC2",
    "3-quinuclidinol": "O[C@@H]1C[N@]2CC[C@H]1CC2",
    "cocaine": "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C",
    "tropine": "CN1[C@H]2CC[C@@H]1C[C@@H](O)C2",
    "cubane acid": "OC(=O)C12C3C4C1C1C2C3C41",
    "menthol": MENTHOL,
    "trans-decalin": "[C@@H]12CCCC[C@H]1CCCC2",
    "cholesterol": (
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
    ),
}


@pytest.mark.parametrize("smiles", SUBSTITUTED.values(), ids=SUBSTITUTED.keys())
def test_substituted_examples_follow_the_drawing_rules(smiles):
    mol, _ = parsed(smiles)
    lay = T.perspective_layout(mol, T.conformer3d(mol))
    assert lay is not None
    xy, _ = D.normalized(mol, lay.xy)
    rep = M.mol_rule_report(mol, xy, lay.short)
    assert rep.violations() == [], rep
    t = next(t for t in T.load_library() if t.name == lay.template)
    assert rep.crossings <= t.crossings  # substituents add no crossing


def test_symmetric_template_puts_substituents_on_clear_slots():
    mol, _ = parsed("OC12CC3CC(CC(C3)C1)C2")  # 1-adamantanol: the OH on the apex
    lay = T.perspective_layout(mol, T.conformer3d(mol))
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    assert lay.xy[o, 1] == pytest.approx(lay.xy[:, 1].max())


@pytest.mark.parametrize(
    ("name", "smiles"),
    [(t.name, ex) for t in T.load_library() for ex in t.examples],
)
def test_template_matches_and_is_used_for_its_examples(name, smiles):
    mol, raw = parsed(smiles)
    lay = T.perspective_layout(mol, T.conformer3d(mol))
    assert lay is not None and lay.template == name and lay.rmsd < T.FIT_RMSD_MAX
    out = D.clean(mol, raw, "perspective")
    if out["template"] is None:  # glucose: the chair crowds its OH labels
        assert any("more defects" in n for n in out["notes"]), out["notes"]
    else:
        assert out["template"] == name and out["method"] == f"perspective:{name}"
        assert out["depth"] is not None and len(out["depth"]) == mol.GetNumAtoms()


def test_no_template_for_benzene_or_a_boat():
    for smiles in ("c1ccccc1", "C1CC2CCC1CC2C"):  # flat; bicyclo[2.2.2]octane is all boats
        mol, raw = parsed(smiles)
        lay = T.perspective_layout(mol, T.conformer3d(mol))
        assert lay is None or lay.template != "cyclohexane-chair"


def test_chair_puts_axial_and_equatorial_substituents_on_their_slots():
    mol, raw = parsed("CC(C)(C)[C@H]1CC[C@@H](O)CC1")  # cis: OH axial, tBu equatorial
    out = D.clean(mol, raw, "perspective")
    assert out["template"] == "cyclohexane-chair"
    xy = np.array(out["coords"])
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    c = mol.GetAtomWithIdx(o).GetNeighbors()[0].GetIdx()
    v = xy[o] - xy[c]
    assert abs(v[0]) < 1e-6 * np.linalg.norm(v)  # the axial OH is exactly vertical


# ---- clean


def test_atom_and_bond_order_kept_with_explicit_h():
    order = [5, 3, 0, 9, 1, 2, 4, 6, 7, 8, 10]
    text = molblock(CAMPHOR, order=order)
    mol, raw = D.parse_molblock(text)
    for style in ("textbook", "perspective"):
        out = D.clean(mol, raw, style)
        back = Chem.MolFromMolBlock(out["molblock"], removeHs=False)
        assert [a.GetSymbol() for a in back.GetAtoms()] == [a.GetSymbol() for a in mol.GetAtoms()]
        pairs = [{b.GetBeginAtomIdx(), b.GetEndAtomIdx()} for b in back.GetBonds()]
        assert pairs == [{b.GetBeginAtomIdx(), b.GetEndAtomIdx()} for b in raw.GetBonds()]
        assert len(out["coords"]) == mol.GetNumAtoms()
        xy = back.GetConformer().GetPositions()[:, :2]
        assert np.allclose(xy, out["coords"], atol=1e-3)
    mol, raw = D.parse_molblock(molblock("CCO", explicit_h=True))
    out = D.clean(mol, raw, "textbook")
    back = Chem.MolFromMolBlock(out["molblock"], removeHs=False)
    assert back.GetNumAtoms() == 9 and out["smiles"] == Chem.MolToSmiles(mol)


def test_kekule_form_and_charges_are_the_inputs():
    text = molblock("C[N+](C)(C)Cc1ccccc1")
    mol, raw = D.parse_molblock(text)
    out = D.clean(mol, raw, "textbook")
    back = Chem.MolFromMolBlock(out["molblock"], sanitize=False, removeHs=False)
    assert [b.GetBondType() for b in back.GetBonds()] == [b.GetBondType() for b in raw.GetBonds()]
    assert [a.GetFormalCharge() for a in back.GetAtoms()] == [
        a.GetFormalCharge() for a in raw.GetAtoms()
    ]


@pytest.mark.parametrize("smiles", STEREO)
@pytest.mark.parametrize("style", ["textbook", "perspective"])
def test_stereo_kept(smiles, style):
    mol, raw = parsed(smiles)
    out = D.clean(mol, raw, style)
    assert out["stereo"] == "kept" and out["smiles"] == Chem.MolToSmiles(mol)
    centres = {c for c, _ in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)}
    assert {w["narrow"] for w in out["wedges"]} == centres  # one mark per centre


def test_camphor_textbook_uses_rdkit_ring_templates_and_perspective_the_norbornane():
    mol, raw = parsed(CAMPHOR)
    textbook = D.clean(mol, raw, "textbook")
    assert textbook["method"] == "rdkit-templates" and textbook["depth"] is None
    assert {c["method"] for c in textbook["candidates"]} == {"rdkit-templates", "coordgen"}
    persp = D.clean(mol, raw, "perspective")
    assert persp["template"] == "norbornane" and persp["defects"]["score"] < 10


def test_defects_flag_overlaps():
    mol, _ = parsed("CCCC")
    xy = np.array([[0.0, 0.0], [1.5, 0.0], [0.2, 0.1], [1.7, 0.1]])
    d = D.M.defects(mol, xy)
    assert d.close_contacts >= 1 and d.small_angles >= 1 and d.score > 10
    good = D.M.defects(mol, D.rdkit_layout(mol))
    assert good.score < 1


# ---- match3d


def view_quat(axis, deg) -> list[float]:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    s = np.sin(np.deg2rad(deg) / 2)
    return [*(axis * s), float(np.cos(np.deg2rad(deg) / 2))]


def test_view_matrix_is_projection_ts_rotate_by_quat():
    q = view_quat([0.3, -0.5, 0.8], 70)
    v = np.array([0.4, 1.1, -0.7])
    qv, w = np.array(q[:3]), q[3]
    t = 2 * np.cross(qv, v)
    ts = v + w * t + np.cross(qv, t)  # projection.ts rotateByQuat
    assert np.allclose(D.view_matrix(q) @ v, ts)
    assert np.allclose(D.view_matrix(view_quat([0, 0, 1], 90)) @ [1, 0, 0], [0, 1, 0])


def structure(smiles: str) -> tuple[Chem.Mol, Chem.Mol, np.ndarray]:
    mol, raw = parsed(smiles)
    return mol, raw, T.conformer3d(mol)


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    """The in-plane angle (deg) that best turns a onto b (both centred, no mirror)."""
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    cross = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    return float(np.degrees(np.arctan2(cross, float((a * b).sum()))))


@pytest.mark.parametrize(
    ("smiles", "style", "worst"),
    [
        (MENTHOL, "textbook", 0.45),
        ("CCCCCCO", "textbook", 0.3),
        (CAMPHOR, "perspective", 0.7),  # a cage: the stylised template only turns in the plane
    ],
)
def test_match3d_follows_the_view(smiles, style, worst):
    mol, raw, xyz = structure(smiles)
    xyz = principal_frame(xyz)  # x the long axis: the views below look at its broad side
    for q in (view_quat([1, 0, 0], 20), view_quat([0, 0, 1], 90), view_quat([0.3, 1, 0], 35)):
        out = D.match3d(mol, raw, xyz, q, style)
        assert out["stereo"] in ("kept", "none") and out["view_residual"] < worst
        proj, depth = D.view_frame(mol, (xyz - xyz.mean(axis=0)) @ D.view_matrix(q).T)
        persp = out["method"].startswith("match3d-perspective")
        up = D.TEMPLATE_UPRIGHT_ABOVE if persp else None
        _, res = D.align_to(np.array(out["coords"]), proj, mirror=not persp, upright_above=up)
        assert res == pytest.approx(out["view_residual"], abs=2e-3)
        if not persp:  # the depth is the view's (a perspective drawing keeps its template's)
            assert np.allclose(out["depth"], depth, atol=1e-3)


def test_match3d_turns_with_an_in_plane_view_rotation():
    mol, raw, xyz = structure(MENTHOL)
    base = view_quat([1, 0, 0], 1e-6)
    a = np.array(D.match3d(mol, raw, xyz, base, "textbook")["coords"])
    b = np.array(D.match3d(mol, raw, xyz, view_quat([0, 0, 1], 90), "textbook")["coords"])
    turn = _turn(a, b)
    assert abs(turn - 90) < 5 or abs(abs(turn) - 90) < 5


def test_match3d_stereo_is_the_structure_s():
    mol, raw, xyz = structure(MENTHOL)
    flat = Chem.MolFromSmiles("CC(C)C1CCC(C)CC1O")  # drawn without stereo
    AllChem.Compute2DCoords(flat)
    fmol, fraw = D.parse_molblock(Chem.MolToMolBlock(flat))
    match = fmol.GetSubstructMatch(mol)
    ref = np.zeros_like(xyz)
    ref[list(match)] = xyz
    out = D.match3d(fmol, fraw, ref, view_quat([0, 1, 0], 20), "textbook")
    assert out["stereo"] == "kept" and "@" in out["smiles"]
    assert Chem.CanonSmiles(out["smiles"]) == Chem.CanonSmiles(MENTHOL)


def test_align_to_turns_mirrors_and_clamps():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 2))
    t = np.deg2rad(70)
    b = a @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
    assert D.align_to(a, b)[1] < 1e-9 and D.align_to(a, b @ np.diag([-1.0, 1.0]))[1] < 1e-9
    assert D.align_to(a, b @ np.diag([-1.0, 1.0]), mirror=False)[1] > 0.3
    assert D.align_to(a, b, upright_above=1e-6)[1] < 1e-9  # a good turn is kept
    far = D.align_to(a, rng.normal(size=(8, 2)), upright_above=0.0)  # no fit: stays upright
    assert np.allclose(far[0], a - a.mean(axis=0))


def test_match3d_perspective_draws_camphor_with_the_template():
    mol, raw, xyz = structure(CAMPHOR)
    out = D.match3d(mol, raw, xyz, view_quat([1, 1, 0], 25), "perspective")
    assert out["stereo"] == "kept"
    assert any(c["method"] == "match3d-perspective:norbornane" for c in out["candidates"])


def test_cage_stereo_is_not_drawn():
    mol, raw, xyz = structure("OC12CC3CC(CC(C3)C1)C2")  # 1-adamantanol: no stereocentre
    out = D.match3d(mol, raw, xyz, view_quat([0, 1, 0], 10), "textbook")
    assert out["wedges"] == [] and "@" not in out["smiles"]


# ---- errors


def test_input_errors():
    assert code_of(D.parse_molblock, "") == Code.EMPTY_INPUT
    assert code_of(D.parse_molblock, "not a molfile") == Code.INVALID_MOLFILE
    big = Chem.MolFromSmiles("C" * (D.MAX_ATOMS + 1))
    AllChem.Compute2DCoords(big)
    assert code_of(D.parse_molblock, Chem.MolToMolBlock(big)) == Code.LIMIT_EXCEEDED
    assert code_of(D.view_matrix, [0, 0, 0, 0]) == Code.REQUEST_INVALID
    mol, raw = parsed("CCO")
    assert code_of(D.match3d, mol, raw, np.zeros((2, 3)), [0, 0, 0, 1], "textbook") == (
        Code.MAPPING_FAILED
    )


# ---- over HTTP (synthetic chemistry)


@pytest.fixture(scope="module")
def app(wb_settings):  # noqa: F811
    with api_client(wb_settings) as client:
        yield client


def post(app, **body):
    return app.post("/api/depict2d", json=body)


def test_http_clean(app):
    r = post(app, molblock=molblock(CAMPHOR), mode="clean", style="perspective")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["template"] == "norbornane" and body["stereo"] == "kept"
    assert len(body["coords"]) == 11 and len(body["depth"]) == 11 and body["molecule_id"] is None


def test_http_match3d_on_the_record(app):
    mb = molblock("CO", order=[1, 0])
    mol = app.post("/api/molecule", json={"molblock": mb}).json()
    assert mol["status"] == "ready", mol
    view = {"quaternion": view_quat([0, 1, 0], 40), "convention": "3dmol-rotationGroup"}
    r = post(app, molblock=mb, mode="match3d", molecule_id=mol["molecule_id"], view=view)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "match3d" and body["depth"] is not None and len(body["coords"]) == 2
    other = molblock("CCO")
    r = post(app, molblock=other, mode="match3d", molecule_id=mol["molecule_id"], view=view)
    assert r.status_code == 422 and r.json()["error"]["code"] == "MAPPING_FAILED"


def test_http_errors(app, wb_settings):  # noqa: F811
    mb = molblock("CCO")
    view = {"quaternion": [0, 0, 0, 1]}
    r = post(app, molblock=mb, mode="match3d")
    assert r.status_code == 422 and r.json()["error"]["code"] == "REQUEST_INVALID"
    r = post(app, molblock=mb, mode="match3d", molecule_id="m" + "0" * 20, view=view)
    assert r.status_code == 404 and r.json()["error"]["code"] == "MOLECULE_NOT_FOUND"
    r = post(app, molblock=mb, mode="sideways")
    assert r.status_code == 422
    r = post(app, molblock="", mode="clean")
    assert r.json()["error"]["code"] == "EMPTY_INPUT"
    r = post(app, molblock=mb, mode="match3d", molecule_id="nope", view=view)
    assert r.status_code in (404, 422)
    key = molecule.parse_input(molblock=molblock("CCCO")).key
    entry = molecule.register(wb_settings, key, "never-computed-method")
    r = post(app, molblock=molblock("CCCO"), mode="match3d", molecule_id=entry["molecule_id"])
    assert r.json()["error"]["code"] == "REQUEST_INVALID"  # no view
    r = post(
        app, molblock=molblock("CCCO"), mode="match3d", molecule_id=entry["molecule_id"], view=view
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "MOLECULE_NOT_READY"


def test_match3d_view_template_keeps_the_view_of_a_cage():
    mol, raw, xyz = structure(CAMPHOR)
    out = D.match3d(mol, raw, principal_frame(xyz), view_quat([1, 0, 0], 20), "textbook")
    view = next(c for c in out["candidates"] if c["method"] == "match3d-view")
    flat = next(c for c in out["candidates"] if c["method"] == "match3d-rdkit")
    assert view["residual"] < 0.3 < flat["residual"]  # the ring system where the view shows it
