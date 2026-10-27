"""`ksmech-serve gallery`: a fixed visual gallery of hard molecules for the 2D depiction, and the
derivation of the perspective templates.

The gallery draws every molecule of `GALLERY` with each method of `METHODS` (the old CoordGen
clean-up, the textbook and perspective clean-ups, and match3d from two fixed views of an
RDKit ETKDG + MMFF structure, so no xTB is needed) twice: as RDKit's drawer renders it
(`gallery_<method>.png` / `.svg`, `sheet_<k>.png`: molecules as rows, methods as columns) and as
the sketcher draws it (`app_<method>.png` / `.svg`: `app_scene` reimplements apps/ui
sketch/drawing.ts and crossings.ts, so carbons are bare vertices, double bonds go into rings and
the bond behind each crossing is broken by depth); `perspective_rdkit_vs_app.png` puts the two
side by side. `templates.png` shows every perspective template with a stub on each drawn slot,
then its examples, in the sketcher's style; `templates.json` holds their drawing-rule reports
(depiction_metrics.rule_report) and `gallery.json` the method, template, score, defects and
stereo of every drawing.

`derive_template(spec, cx)` derives a template's `fit` and `depth` from its picture (the .smi
line) and its derivation (parent SMILES, seed, front atoms): the parent's lowest-energy MMFF
structure (20 srETKDGv3 embeddings) turned to the orthographic view that best reproduces the
picture. `ksmech-serve gallery --derive-templates` writes the re-derived files next to the
gallery for review; copying them into `serve/depiction/data/` is a deliberate, reviewed step. A new
picture is written with `smi_line` (coordinates in the parent ring system's atom order).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D, Point3D

from ksmech.serve.depiction import depiction2d as D
from ksmech.serve.depiction import depiction_metrics as M
from ksmech.serve.depiction import depiction_templates as T

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
    app: dict[str, list] = {m: [] for m in METHODS}
    for name, smi in chosen:
        summary[name] = {}
        for method in METHODS:
            res = depict(name, smi, method)
            summary[name][method] = {k: res[k] for k in ("method", "template", "stereo")}
            summary[name][method] |= {"defects": res["defects"], "candidates": res["candidates"]}
            cells[method].append((_drawable(res), _legend(name, res)))
            app[method].append((result_scene(res), _legend(name, res)))
    for method in METHODS:
        draw_grid(cells[method], out / f"gallery_{method}")
        draw_app_grid(app[method], out / f"app_{method}")
    for k in range(0, len(chosen), 4):
        sheet = [cells[m][i] for i in range(k, min(k + 4, len(chosen))) for m in METHODS]
        draw_grid(sheet, out / f"sheet_{k // 4 + 1}", per_row=len(METHODS))
    draw_side_by_side(cells["perspective"], app["perspective"], out / "perspective_rdkit_vs_app")
    report = draw_templates(out / "templates")
    (out / "templates.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "gallery.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ---- the sketcher's own drawing (apps/ui sketch/drawing.ts + crossings.ts, reimplemented)

APP_SCALE = 25.0  # px per drawing unit (Sketcher DEFAULT_SCALE)
FONT_PX = 13.0
DOUBLE_GAP_PX = 4.5
INNER_SHORTEN = 0.14
WEDGE_HALF_PX = 3.6
BREAK_GAP_PX = 3.2
STROKE_PX = 1.5
_COLOURS = {"O": (0.85, 0.1, 0.1), "N": (0.1, 0.2, 0.85), "S": (0.75, 0.6, 0.0)}


@dataclass
class AppScene:
    """What the sketcher draws, in px (y down): stroke segments, filled polygons, labels."""

    lines: list = field(default_factory=list)  # (p, q) numpy points
    polys: list = field(default_factory=list)  # [p, ...]
    texts: list = field(default_factory=list)  # (text, p, rgb)
    size: tuple[float, float] = (0.0, 0.0)


def _label(atom: Chem.Atom) -> str | None:
    """drawing.ts atomLabel: carbon is a bare vertex unless charged or isolated."""
    if atom.GetSymbol() == "C" and atom.GetFormalCharge() == 0 and atom.GetDegree() > 0:
        return None
    h = atom.GetTotalNumHs()
    q = atom.GetFormalCharge()
    charge = "" if q == 0 else (f"{abs(q)}" if abs(q) > 1 else "") + ("+" if q > 0 else "-")
    return atom.GetSymbol() + ("H" if h else "") + (str(h) if h > 1 else "") + charge


def _crossing_cuts(mol: Chem.Mol, p: np.ndarray, depth) -> dict[int, list]:
    """crossings.ts findCrossings: {bond behind: [(point, front direction, front half-width)]};
    the front bond has the larger interpolated depth, else it is the later bond."""
    bonds = [(b, b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    cuts: dict[int, list] = {}
    for i, (b1, a1, c1) in enumerate(bonds):
        for b2, a2, c2 in bonds[i + 1 :]:
            if {a1, c1} & {a2, c2}:
                continue
            d1, d2, w = p[c1] - p[a1], p[c2] - p[a2], p[a2] - p[a1]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-12:
                continue
            s = (w[0] * d2[1] - w[1] * d2[0]) / den
            t = (w[0] * d1[1] - w[1] * d1[0]) / den
            if not (0.02 < s < 0.98 and 0.02 < t < 0.98):
                continue
            first = False
            if depth is not None:
                z1 = depth[a1] + (depth[c1] - depth[a1]) * s
                z2 = depth[a2] + (depth[c2] - depth[a2]) * t
                first = abs(z1 - z2) > 1e-9 and z1 > z2
            front, back = (b1, b2) if first else (b2, b1)
            fd = p[front.GetEndAtomIdx()] - p[front.GetBeginAtomIdx()]
            half = DOUBLE_GAP_PX if front.GetBondType() == Chem.BondType.DOUBLE else 0.8
            cuts.setdefault(back.GetIdx(), []).append(
                (p[a1] + d1 * s, fd / np.linalg.norm(fd), half)
            )
    return cuts


def _cut_interval(p: np.ndarray, q: np.ndarray, cut) -> tuple[float, float] | None:
    """drawing.ts cutInterval: the arclength interval of p->q the cut removes."""
    d = q - p
    ln = float(np.linalg.norm(d))
    at, fdir, half = cut
    den = d[0] * fdir[1] - d[1] * fdir[0]
    if ln < 1e-9 or abs(den) < 1e-9 * ln:
        return None
    w = at - p
    along = (w[0] * fdir[1] - w[1] * fdir[0]) / den * ln
    g = (BREAK_GAP_PX + half) / max(abs(den) / ln, 0.35)
    return along - g, along + g


def _cut_segment(seg, cut) -> list:
    p, q = seg
    iv = _cut_interval(p, q, cut)
    ln = float(np.linalg.norm(q - p))
    if iv is None or iv[1] <= 0 or iv[0] >= ln:
        return [seg]
    u = (q - p) / ln
    out = [(p, p + u * iv[0])] if iv[0] > 0.6 else []
    return out + ([(p + u * iv[1], q)] if ln - iv[1] > 0.6 else [])


def _cut_wedge(poly, axis, cut) -> list:
    p, q = axis
    iv = _cut_interval(p, q, cut)
    ln = float(np.linalg.norm(q - p))
    if iv is None or iv[1] <= 0 or iv[0] >= ln:
        return [poly]
    w0, w1, w2, w3 = poly

    def piece(t0: float, t1: float) -> list:
        return [w0 + (w1 - w0) * t0, w0 + (w1 - w0) * t1, w3 + (w2 - w3) * t1, w3 + (w2 - w3) * t0]

    out = [piece(0, iv[0] / ln)] if iv[0] > 0.6 else []
    return out + ([piece(iv[1] / ln, 1)] if ln - iv[1] > 0.6 else [])


def _offset(seg, k: float):
    p, q = seg
    d = q - p
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
    return (p + n * k, q + n * k)


def _shorten(seg, f: float, at_p: bool, at_q: bool):
    p, q = seg
    d = q - p
    return (p + d * f if at_p else p, q - d * f if at_q else q)


def _double_side(mol: Chem.Mol, a: int, b: int, p: np.ndarray) -> int:
    """drawing.ts doubleBondSide: towards the smallest ring's centre, else the substituents."""
    d = p[b] - p[a]

    def side(x) -> int:
        return int(np.sign(d[0] * (x[1] - p[a][1]) - d[1] * (x[0] - p[a][0])))

    rings = [r for r in mol.GetRingInfo().AtomRings() if a in r and b in r]
    if rings:
        return side(p[list(min(rings, key=len))].mean(axis=0)) or 1
    na = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b]
    nb = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a]
    if (not na and not nb) or (not na and len(nb) >= 2) or (not nb and len(na) >= 2):
        return 0
    total = sum(side(p[x]) for x in na + nb)
    return 1 if total >= 0 else -1


def _stereo_shape(main, kind: str) -> tuple[list, list]:
    s, e = main
    u = (e - s) / (np.linalg.norm(e - s) or 1.0)
    n = np.array([-u[1], u[0]])
    if kind == "wedge":
        return [], [[s + n * 0.4, e + n * WEDGE_HALF_PX, e - n * WEDGE_HALF_PX, s - n * 0.4]]
    count = max(4, round(float(np.linalg.norm(e - s)) / 4.5))
    ticks = []
    for i in range(count):
        t = (i + 0.5) / count
        c, w = s + (e - s) * t, 0.6 + (WEDGE_HALF_PX - 0.6) * t
        ticks.append((c + n * w, c - n * w))
    return ticks, []


def _bond_shape(mol: Chem.Mol, a: int, b: int, p, trims, kind) -> tuple[tuple, list, list]:
    """drawing.ts bondShape: (axis, lines, wedge polygons), a the narrow end of a wedge."""
    d = p[b] - p[a]
    ln = float(np.linalg.norm(d)) or 1.0
    u = d / ln
    main = (p[a] + u * min(trims[a], ln / 2), p[b] - u * min(trims[b], ln / 2))
    if kind is not None:
        return main, *_stereo_shape(main, kind)
    btype = mol.GetBondBetweenAtoms(a, b).GetBondType()
    if btype == Chem.BondType.DOUBLE:
        side = _double_side(mol, a, b, p)
        if side == 0:
            return main, [_offset(main, DOUBLE_GAP_PX / 2), _offset(main, -DOUBLE_GAP_PX / 2)], []
        inner = _offset(main, side * DOUBLE_GAP_PX)
        return main, [main, _shorten(inner, INNER_SHORTEN, not trims[a], not trims[b])], []
    if btype == Chem.BondType.TRIPLE:
        side_lines = [_offset(main, k * DOUBLE_GAP_PX) for k in (1, -1)]
        return (
            main,
            [main] + [_shorten(s, 0.08, not trims[a], not trims[b]) for s in side_lines],
            [],
        )
    return main, [main], []


def app_scene(mol: Chem.Mol, xy, depth=None, wedges=None, scale: float = APP_SCALE) -> AppScene:
    """The sketcher's drawing of `mol` (Kekule) at `xy` (drawing units, y up), crossing breaks
    from `depth`, wedges {bond index: (narrow atom, 'wedge' | 'hash')}."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    pad = FONT_PX * 2
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    p = np.column_stack([(xy[:, 0] - lo[0]) * scale + pad, (hi[1] - xy[:, 1]) * scale + pad])
    scene = AppScene(size=tuple(float(v) for v in (hi - lo) * scale + 2 * pad))
    labels = [_label(a) for a in mol.GetAtoms()]
    trims = [
        0.0 if t is None else FONT_PX * (0.72 if len(a.GetSymbol()) > 1 else 0.55)
        for t, a in zip(labels, mol.GetAtoms(), strict=True)
    ]
    cuts = _crossing_cuts(mol, p, depth)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        kind = (wedges or {}).get(bond.GetIdx())
        if kind is not None and kind[0] == b:
            a, b = b, a
        axis, lines, polys = _bond_shape(mol, a, b, p, trims, None if kind is None else kind[1])
        for cut in cuts.get(bond.GetIdx(), []):
            if kind is not None and kind[1] == "hash":
                lines = [t for t in lines if _tick_kept(t, axis, cut)]
            else:
                lines = [s for seg in lines for s in _cut_segment(seg, cut)]
            polys = [w for poly in polys for w in _cut_wedge(poly, axis, cut)]
        scene.lines += lines
        scene.polys += polys
    for k, t in enumerate(labels):
        if t is not None:
            colour = _COLOURS.get(mol.GetAtomWithIdx(k).GetSymbol(), (0.2, 0.2, 0.2))
            scene.texts.append((t, p[k], colour))
    return scene


def _tick_kept(tick, axis, cut) -> bool:
    iv = _cut_interval(axis[0], axis[1], cut)
    along = float(np.linalg.norm((tick[0] + tick[1]) / 2 - axis[0]))
    return iv is None or along < iv[0] or along > iv[1]


def _pt(v) -> Point2D:
    return Point2D(float(v[0]), float(v[1]))


def paint_scene(drawer, scene: AppScene, origin, legend: str = "") -> None:
    """Draw `scene` with its top-left at `origin` (px) on an RDKit drawer (raw pixel coords)."""
    o = np.asarray(origin, dtype=float)
    drawer.SetColour((0.13, 0.13, 0.13))
    drawer.SetLineWidth(int(round(STROKE_PX)))
    for p, q in scene.lines:
        drawer.DrawLine(_pt(p + o), _pt(q + o), True)
    drawer.SetFillPolys(True)
    for poly in scene.polys:
        drawer.DrawPolygon([_pt(v + o) for v in poly], True)
    drawer.SetFontSize(FONT_PX)
    for text, at, colour in scene.texts:
        drawer.SetColour((1.0, 1.0, 1.0))
        half = (0.62 + 0.5 * (len(text) - 1)) * FONT_PX / 2
        drawer.DrawRect(
            _pt(at + o - [half, FONT_PX * 0.55]), _pt(at + o + [half, FONT_PX * 0.55]), True
        )
        drawer.SetColour(colour)
        drawer.DrawString(text, _pt(at + o), 0, True)
    if legend:
        drawer.SetColour((0.1, 0.1, 0.1))
        drawer.DrawString(legend, _pt(o + [CELL[0] / 2, CELL[1] - 12]), 0, True)


def draw_side_by_side(mols: list, scenes: list, path: Path) -> None:
    """`path`.png: each molecule as RDKit draws it, then as the sketcher draws it (two pairs
    per row)."""
    w, h = CELL
    rows = max(1, -(-len(mols) // 2))
    drawer = rdMolDraw2D.MolDraw2DCairo(4 * w, h * rows, w, h)
    opts = drawer.drawOptions()
    opts.prepareMolsBeforeDrawing = False
    opts.legendFontSize = 15
    opts.fixedBondLength = 30
    for i, (mol, legend) in enumerate(mols):
        drawer.SetOffset((i % 2) * 2 * w, (i // 2) * h)
        drawer.DrawMolecule(mol, legend="RDKit: " + legend)
    drawer.SetOffset(0, 0)
    for i, (scene, legend) in enumerate(scenes):
        cx, cy = (i % 2) * 2 * w + w, (i // 2) * h
        paint_scene(
            drawer, scene, (cx + (w - scene.size[0]) / 2, cy + (h - 24 - scene.size[1]) / 2)
        )
        drawer.SetColour((0.1, 0.1, 0.1))
        drawer.SetFontSize(13)
        drawer.DrawString(
            "sketcher: " + legend.split(" | ")[0], _pt((cx + w / 2, cy + h - 12)), 0, True
        )
    drawer.FinishDrawing()
    path.with_suffix(".png").write_bytes(drawer.GetDrawingText())


def result_scene(res: dict, cell=CELL) -> AppScene:
    """The sketcher's drawing of a depiction answer (its molblock, depth and wedges), at the
    sketcher's scale or smaller to fit the cell."""
    with rdBase.BlockLogs():
        m = Chem.MolFromMolBlock(res["molblock"], sanitize=True, removeHs=False)
    Chem.Kekulize(m, clearAromaticFlags=True)
    xy = np.asarray(res["coords"], dtype=float)
    wedges = {w["bond"]: (w["narrow"], w["kind"]) for w in res["wedges"]}
    return app_scene(m, xy, res["depth"], wedges, _fit_scale(xy, cell))


def _fit_scale(xy: np.ndarray, cell) -> float:
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-6)
    room = np.array([cell[0], cell[1] - 30.0]) - 4 * FONT_PX
    return float(min(APP_SCALE, *(room / span)))


def draw_app_grid(cells: list[tuple[AppScene, str]], path: Path, per_row: int = 4) -> None:
    """`path`.png and `path`.svg of sketcher-style scenes in a grid with legends."""
    rows = max(1, -(-len(cells) // per_row))
    w, h = CELL
    for kind in ("png", "svg"):
        cls = rdMolDraw2D.MolDraw2DCairo if kind == "png" else rdMolDraw2D.MolDraw2DSVG
        drawer = cls(w * per_row, h * rows)
        drawer.drawOptions().clearBackground = True
        drawer.ClearDrawing()
        for i, (scene, legend) in enumerate(cells):
            cx, cy = (i % per_row) * w, (i // per_row) * h
            origin = (cx + (w - scene.size[0]) / 2, cy + (h - 24 - scene.size[1]) / 2)
            paint_scene(drawer, scene, origin)
            drawer.SetColour((0.1, 0.1, 0.1))
            drawer.SetFontSize(13)
            drawer.DrawString(legend, _pt((cx + w / 2, cy + h - 12)), 0, True)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        out = path.with_suffix(f".{kind}")
        if kind == "png":
            out.write_bytes(data)
        else:
            out.write_text(data, encoding="utf-8")


# ---- the template sheet


def slot_points(t: T.Template) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """(points, bonds, depth) of a template with a stub on every drawn slot."""
    pts, bonds, depth = [*t.xyz[:, :2]], list(t.bonds), [*t.xyz[:, 2]]
    for k, ss in enumerate(t.slots):
        for slot in ss:
            if slot.drawn:
                pts.append(t.xyz[k, :2] + T.BOND * slot.d)
                depth.append(t.xyz[k, 2] + T.BOND * slot.u[2])
                bonds.append((k, len(pts) - 1))
    return np.array(pts), bonds, np.array(depth)


def template_scene(t: T.Template, with_slots: bool = True) -> AppScene:
    """The sketcher's drawing of a template (generic carbons), with its slots as stubs."""
    pts, bonds, depth = slot_points(t) if with_slots else (t.xyz[:, :2], t.bonds, t.xyz[:, 2])
    rw = Chem.RWMol()
    for _ in pts:
        rw.AddAtom(Chem.Atom(6))
    for a, b in bonds:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    return app_scene(m, pts, depth, None, _fit_scale(pts, CELL))


def slot_angles(t: T.Template) -> tuple[float, float]:
    """(smallest, straightest) angle in degrees between two bonds at one template atom,
    counting its ring bonds and its drawn slots (where a substituent would go)."""
    lo, hi = 180.0, 0.0
    xy = t.xyz[:, :2]
    for k in range(t.size):
        dirs = [T._unit(xy[b if a == k else a] - xy[k]) for a, b in t.bonds if k in (a, b)]
        dirs += [s.d for s in t.slots[k] if s.drawn]
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                ang = float(np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1.0, 1.0))))
                lo, hi = min(lo, ang), max(hi, ang)
    return round(lo, 1), round(hi, 1)


def template_report(t: T.Template) -> dict:
    """The drawing rules on a template: its ring bonds, and the angles with its slots."""
    return {
        "ring": M.rule_report(t.xyz[:, :2], t.bonds, t.foreshortened),
        "slot_angle": slot_angles(t),
    }


def draw_templates(path: Path) -> dict:
    """`path`.png: per template (one row) the bare template with its slots, then its examples
    as the perspective clean-up draws them, in the sketcher's style; returns the rule reports
    (template, examples) for templates.json."""
    rows, report = [], {}
    for t in sorted(T.load_library(), key=lambda t: t.name):
        rep = template_report(t)
        entry = {"ring": asdict(rep["ring"]), "slot_angle": rep["slot_angle"], "examples": {}}
        rows.append((template_scene(t), f"{t.name} ({len(t.bonds)} bonds)"))
        for smi in t.examples[:3]:
            mol, raw = D.parse_molblock(sketch_molblock(smi))
            res = D.clean(mol, raw, "perspective")
            got = M.mol_rule_report(mol, np.array(res["coords"]))
            entry["examples"][smi] = {"template": res["template"], **asdict(got)}
            rows.append((result_scene(res), f"{res['template'] or 'textbook'} {smi[:28]}"))
        rows += [(AppScene(size=(0.0, 0.0)), "")] * (3 - min(3, len(t.examples)))
        report[t.name] = entry
    draw_app_grid(rows, path, per_row=4)
    return report


# ---- template derivation


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


def smi_line(parent: str, xy, seed: int = 7) -> str:
    """The .smi CXSMILES (without the name) of a picture `xy` drawn in the atom order of the
    parent's ring system (as `_parent_geometry` lists it): how a new template is written."""
    sub, _ = _parent_geometry({"parent": parent, "seed": seed})
    xy = np.asarray(xy, dtype=float)
    conf = Chem.Conformer(sub.GetNumAtoms())
    for i, p in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), 0.0))
    sub.AddConformer(conf)
    return Chem.MolToCXSmiles(sub, canonical=False)


def _orient(x3: np.ndarray, xy: np.ndarray, front: list[int], starts: int = 24):
    """(fit, rms): s (x3 R^T) for the proper rotation R and scale s whose orthographic
    projection best matches the picture `xy` (both centred) among the views with the `front`
    atoms in front of the centre (a fixed set of random starts), or None."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    def view(rv: np.ndarray) -> tuple[np.ndarray, float]:
        v = x3 @ Rotation.from_rotvec(rv).as_matrix().T
        a = v[:, :2]
        return v * float((a * xy).sum() / max(float((a * a).sum()), 1e-12)), 0.0

    def cost(rv: np.ndarray) -> float:
        v, _ = view(rv)
        miss = max(0.0, -float(v[front, 2].mean())) if front else 0.0
        return float(((v[:, :2] - xy) ** 2).sum()) + 10.0 * miss**2

    best = None
    for r0 in Rotation.random(starts, random_state=0):
        res = minimize(cost, r0.as_rotvec(), method="BFGS")
        v, _ = view(res.x)
        if front and v[front, 2].mean() <= 0:
            continue
        rms = float(np.sqrt(((v[:, :2] - xy) ** 2).sum(axis=1).mean()))
        if best is None or rms < best[1]:
            best = (v, rms)
    return best


def derive_template(spec: dict, cx: str) -> dict:
    """`fit` and `depth` of a template from its picture (`cx`, the .smi CXSMILES) and its
    derivation: the parent's MMFF structure, matched onto the template atoms in every way and
    turned to the orthographic view that best reproduces the picture, keeping only views with
    the `front` atoms in front of the centre (the picture alone cannot tell a cage from its
    depth-reversed twin), scaled like the picture."""
    query = Chem.MolFromSmarts(cx)
    xy = query.GetConformer().GetPositions()[:, :2]
    xy = xy - xy.mean(axis=0)
    sub, x3 = _parent_geometry(spec)
    front = list(spec.get("front", []))
    best = None
    for match in sub.GetSubstructMatches(query, uniquify=False, maxMatches=5000):
        got = _orient(x3[list(match)] - x3[list(match)].mean(axis=0), xy, front)
        if got is not None and (best is None or got[1] < best[1]):
            best = got
    if best is None:
        raise RuntimeError(f"{spec['parent']}: no view matches the picture")
    fit = best[0]
    return {
        "fit": np.round(fit, 3).tolist(),
        "depth": np.round(fit[:, 2], 3).tolist(),
        "projection_rms": round(best[1] / T.BOND, 3),
    }


def derive_all(out: Path) -> Path:
    """Write ring_templates.smi (unchanged) and .json with `fit` and `depth` re-derived from
    the pictures and derivations into out."""
    meta = json.loads(T.META_PATH.read_text(encoding="utf-8"))
    for cx, name in T._smi_lines(T.SMI_PATH):
        entry = meta["templates"][name]
        got = derive_template(entry["derivation"], cx)
        entry["derivation"]["projection_rms"] = got.pop("projection_rms")
        entry.update(got)
    out.mkdir(parents=True, exist_ok=True)
    smi = T.SMI_PATH.read_text(encoding="utf-8")
    (out / "ring_templates.smi").write_text(smi, encoding="utf-8")
    (out / "ring_templates.json").write_text(dump_meta(meta), encoding="utf-8")
    return out


def dump_meta(meta: dict) -> str:
    """ring_templates.json text: indented, with the coordinate rows on one line each."""
    text = json.dumps(meta, indent=1)
    return re.sub(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", r"[\1, \2, \3]", text) + "\n"
