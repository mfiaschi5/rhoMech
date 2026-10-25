"""`ksmech-serve gallery`: a fixed visual gallery of hard molecules for the 2D depiction, and the
derivation of the perspective templates.

The gallery draws every molecule of `GALLERY` with each method of `METHODS` (the old CoordGen
clean-up, the textbook and perspective clean-ups, and match3d from two fixed views of an
RDKit ETKDG + MMFF structure, so no xTB is needed) and writes one grid per method,
`gallery_<method>.png` and `.svg`, plus `sheet_<k>.png` (molecules as rows, methods as columns)
and `gallery.json` (method, template, score, defects and stereo of every drawing). RDKit's
drawer renders them, so the depth-based crossing breaks of the sketcher are not shown.

`derive_template(spec)` turns a template's derivation (`ring_templates.json`) into its .smi line
and depth: the lowest-energy MMFF geometry of the parent (20 srETKDGv3 structures), the
ring-system atoms, an up and a right axis (atom centroids or the ring's mean-plane normal), then a
spin about up, a tilt about the screen x axis (positive shows the top face) and a roll about the
view axis, scaled so the median drawn bond is 1.5. `ksmech-serve gallery --derive-templates`
writes the derived files next to the gallery for review; copying them into `serve/data/` is a
deliberate, reviewed step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

from ksmech.serve import depiction2d as D
from ksmech.serve import depiction_templates as T

GALLERY: tuple[tuple[str, str], ...] = (
    ("camphor", "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"),
    ("borneol", "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](O)C2"),
    ("norbornene", "C1=C[C@H]2CC[C@@H]1C2"),
    ("cocaine", "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C"),
    ("quinuclidine", "C1CN2CCC1CC2"),
    ("1-adamantanol", "OC12CC3CC(CC(C3)C1)C2"),
    ("memantine", "CC12CC3CC(C)(C1)CC(N)(C3)C2"),
    ("cubane-carboxylic acid", "OC(=O)C12C3C4C1C1C2C3C41"),
    ("twistane", "C12CC3CCC2CC3CC1"),
    ("menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"),
    ("cis-4-tBu-cyclohexanol", "CC(C)(C)[C@H]1CC[C@@H](O)CC1"),
    ("beta-D-glucopyranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("trans-decalin", "[C@@H]12CCCC[C@H]1CCCC2"),
    ("cis-decalin", "[C@@H]12CCCC[C@@H]1CCCC2"),
    (
        "cholesterol",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    ),
    ("morphine", "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O"),
    (
        "strychnine",
        "O=C7N2c1ccccc1[C@@]64[C@@H]2[C@@H]3[C@@H](OC/C=C5\\[C@@H]3C[C@@H]6N(CC4)C5)C7",
    ),
    ("artemisinin", "C[C@@H]1CC[C@H]2[C@H](C(=O)O[C@H]3[C@@]24[C@H]1CC[C@](O3)(OO4)C)C"),
    ("1,4-dioxaspiro[4.5]decane", "C1CCC2(CC1)OCCO2"),
    ("18-crown-6", "C1COCCOCCOCCOCCOCCO1"),
    ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
    ("alpha-pinene", "CC1=CC[C@@H]2C[C@H]1C2(C)C"),
)
METHODS = ("coordgen", "textbook", "perspective", "match3d_A", "match3d_B")
VIEWS = {
    "A": (0.0, 0.0, 0.0, 1.0),  # the principal-axes frame (largest spread along x)
    "B": (0.3368, 0.3368, 0.0, 0.8794),  # 56 degrees about (1, 1, 0)
}
CELL = (320, 280)


# ---- inputs


def sketch_molblock(smiles: str) -> str:
    """A drawn molecule as the sketcher would send it: RDKit 2D, wedges, V2000."""
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(m, forceRDKit=True)
    return Chem.MolToMolBlock(m)


def principal_frame(xyz: np.ndarray) -> np.ndarray:
    """xyz centred and turned so x has the largest spread, y the next (a proper rotation)."""
    c = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return c @ vt.T


def depict(name: str, smiles: str, method: str) -> dict:
    mol, raw = D.parse_molblock(sketch_molblock(smiles))
    if method == "coordgen":
        cand = D.Candidate("coordgen", D.coordgen_layout(mol))
        best = D.finish(cand, mol, raw, Chem.MolToSmiles(mol))
        return D._result(best, [best], "clean", "coordgen", [])
    if method in ("textbook", "perspective"):
        return D.clean(mol, raw, method)
    xyz = T.conformer3d(mol)
    if xyz is None:
        raise RuntimeError(f"{name}: no 3D structure")
    view = VIEWS[method.split("_")[1]]
    return D.match3d(mol, raw, principal_frame(xyz), view, "perspective")


# ---- drawing


def _drawable(res: dict) -> Chem.Mol:
    with rdBase.BlockLogs():
        m = Chem.MolFromMolBlock(res["molblock"], sanitize=True, removeHs=False)
    Chem.ReapplyMolBlockWedging(m)
    return rdMolDraw2D.PrepareMolForDrawing(m, kekulize=False, addChiralHs=False, wedgeBonds=False)


def _legend(name: str, res: dict) -> str:
    d = res["defects"] or {}
    tag = res["template"] or res["method"]
    flag = " STEREO!" if res["stereo"] == "mismatch" else ""
    return f"{name} | {tag} | {d.get('score', 0):.1f}{flag}"


def draw_grid(cells: list[tuple[Chem.Mol, str]], path: Path, per_row: int = 4) -> None:
    """`path`.png and `path`.svg of the molecules in a grid with legends."""
    rows = max(1, -(-len(cells) // per_row))
    w, h = CELL
    for kind in ("png", "svg"):
        cls = rdMolDraw2D.MolDraw2DCairo if kind == "png" else rdMolDraw2D.MolDraw2DSVG
        drawer = cls(w * per_row, h * rows, w, h)
        opts = drawer.drawOptions()
        opts.prepareMolsBeforeDrawing = False
        opts.legendFontSize = 15
        opts.fixedBondLength = 30
        drawer.DrawMolecules([c[0] for c in cells], legends=[c[1] for c in cells])
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        out = path.with_suffix(f".{kind}")
        if kind == "png":
            out.write_bytes(data)
        else:
            out.write_text(data, encoding="utf-8")


def run_gallery(out: Path, names: list[str] | None = None) -> dict:
    """Draw the gallery into `out`; returns the summary written to gallery.json."""
    out.mkdir(parents=True, exist_ok=True)
    chosen = [g for g in GALLERY if names is None or g[0] in names]
    summary: dict[str, dict] = {}
    cells: dict[str, list] = {m: [] for m in METHODS}
    for name, smi in chosen:
        summary[name] = {}
        for method in METHODS:
            res = depict(name, smi, method)
            summary[name][method] = {k: res[k] for k in ("method", "template", "stereo")}
            summary[name][method] |= {"defects": res["defects"], "candidates": res["candidates"]}
            cells[method].append((_drawable(res), _legend(name, res)))
    for method in METHODS:
        draw_grid(cells[method], out / f"gallery_{method}")
    for k in range(0, len(chosen), 4):
        sheet = [cells[m][i] for i in range(k, min(k + 4, len(chosen))) for m in METHODS]
        draw_grid(sheet, out / f"sheet_{k // 4 + 1}", per_row=len(METHODS))
    (out / "gallery.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ---- template derivation


def _centroid(xyz: np.ndarray, atoms: list[int]) -> np.ndarray:
    return xyz[atoms].mean(axis=0)


def _axis(spec, xyz: np.ndarray) -> np.ndarray:
    if spec == "normal":
        c = xyz - xyz.mean(axis=0)
        return np.linalg.svd(c, full_matrices=False)[2][2]
    a, b = spec
    v = _centroid(xyz, list(b)) - _centroid(xyz, list(a))
    return v / np.linalg.norm(v)


def _rot(axis: int, deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    i, j = [k for k in range(3) if k != axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def _parent_geometry(spec: dict) -> tuple[Chem.Mol, np.ndarray]:
    """(the ring-system subgraph as a mol, its MMFF coordinates) in parent atom order."""
    parent = Chem.AddHs(Chem.MolFromSmiles(spec["parent"]))
    params = AllChem.srETKDGv3()
    params.randomSeed = int(spec.get("seed", 7))
    ids = list(AllChem.EmbedMultipleConfs(parent, 20, params))
    if not ids:
        raise RuntimeError(f"cannot embed {spec['parent']}")
    energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(parent, maxIters=5000)]
    conf = parent.GetConformer(ids[int(np.argmin(energies))])
    ring = sorted(max(T.ring_systems(parent), key=len))
    sub = Chem.RWMol()
    for _ in ring:
        sub.AddAtom(Chem.Atom(0))
    index = {a: k for k, a in enumerate(ring)}
    for b in parent.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in index and j in index:
            sub.AddBond(index[i], index[j], Chem.BondType.UNSPECIFIED)
    return sub.GetMol(), conf.GetPositions()[ring]


def _view(spec: dict, xyz: np.ndarray) -> np.ndarray:
    c = xyz - xyz.mean(axis=0)
    up = _axis(spec["up"], xyz)
    right = _axis(spec["right"], xyz)
    right = right - up * (right @ up)
    right /= np.linalg.norm(right)
    toward = np.cross(right, up)
    v = c @ np.column_stack([right, up, toward])
    rot = _rot(2, spec.get("roll", 0)) @ _rot(0, spec.get("tilt", 0)) @ _rot(1, spec.get("spin", 0))
    return v @ rot.T


def derive_template(spec: dict) -> tuple[str, list[float]]:
    """(the .smi line without its name, the depth list) of a template derivation."""
    sub, xyz = _parent_geometry(spec)
    v = _view(spec, xyz)
    ends = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in sub.GetBonds()]
    ls = [np.linalg.norm(v[i, :2] - v[j, :2]) for i, j in ends]
    v = v * (T.BOND / float(np.median(ls)))
    v = np.round(v - v.mean(axis=0), 3)
    conf = Chem.Conformer(sub.GetNumAtoms())
    for i, p in enumerate(v):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), 0.0))
    sub.AddConformer(conf)
    cx = Chem.MolToCXSmiles(sub, canonical=False)
    parsed = Chem.MolFromSmarts(cx).GetConformer().GetPositions()[:, :2]
    order = [int(np.argmin(np.abs(v[:, :2] - p).sum(axis=1))) for p in parsed]
    return cx, [float(v[k, 2]) for k in order]


def derive_all(out: Path) -> Path:
    """Write ring_templates.smi / .json derived from the current JSON's derivations into out."""
    meta = json.loads(T.META_PATH.read_text(encoding="utf-8"))
    lines = []
    for name, entry in meta["templates"].items():
        cx, depth = derive_template(entry["derivation"])
        entry["depth"] = depth
        lines.append(f"{cx} {name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ring_templates.smi").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "ring_templates.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return out
