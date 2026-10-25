"""2D depiction of a sketch (POST /api/depict2d), API side, RDKit only.

The sketcher sends its V2000 molblock; the answer keeps EXACTLY its atom and bond order: the
2D coordinates of atom i, the wedges by bond index (narrow end = the stereocentre), the depth of
each atom when the drawing has one, the canonical isomeric SMILES the drawing means to RDKit
and a defect report (`depiction_metrics`). Bond orders are the input's (its Kekule form kept);
the molblock is written here (V2000, charges as `M  CHG`), not by RDKit, so the wedge's narrow
end is always the centre.

Modes:
- **clean**, style **textbook**: RDKit's depictor with its ring templates
  (`Compute2DCoords(forceRDKit=True, useRingTemplates=True)`) and CoordGen
  (`rdCoordGen.AddCoords`); the one with the lower defect score wins (ties go to RDKit).
- **clean**, style **perspective**: the perspective template library
  (`depiction_templates.perspective_layout`, fitted to an ETKDG + MMFF structure of the
  drawing); without a fitting template, the textbook result, with a note.
- **match3d**: the computed structure (`ref_xyz`, sketch atom order) turned to the 3D viewer's
  orientation (`view_matrix`: q = 3Dmol getView()[4..7] [x, y, z, w], the view frame is R(q) p
  with x right, y up, z towards the viewer: `sketch/projection.ts`). Candidates:
  `GenerateDepictionMatching3DStructure` on the 3D distances, `Compute2DCoordsMimicDistmat` on
  the projected (on-screen) distances, the view itself as a ring template
  (`depiction_templates.view_layout`: ring atoms where the view shows them, RDKit for the rest),
  the textbook layout and, for style perspective, the template layout; each is turned (and,
  unless it is a perspective picture, mirrored when that fits better) onto the projection and
  scored by its defects plus 20 x the residual of that fit (the RMS distance to the projection
  over its RMS spread). The depth is the view-frame z, scaled like the drawing (a template
  keeps its own).

Wedges: the stereo of a clean-up is the drawing's own (RDKit's reading of the molblock); the
stereo of a match3d is the computed structure's (`AssignStereochemistryFrom3D`). Each centre
gets one wedge or hash on the single bond that scores best (acyclic +4, terminal neighbour +2,
neighbour not a centre +3; with depth, +2 |dz| / L and +3 when the mark follows the depth, a
wedge in front or a hash behind, else -3); `Chem.WedgeBond` gives the mark's kind. RDKit reads
the result back and the canonical isomeric SMILES must equal the target's; otherwise RDKit's
own `WedgeMolBonds` is tried, and a remaining difference is reported as `stereo: mismatch`.

RDKit's global state is never toggled: `forceRDKit=True` everywhere (`depict.py` sets the
process-wide CoordGen preference), CoordGen is called by name, and the ring templates are
RDKit's defaults (`RDKIT_TEMPLATE_LOCK` guards the one test that loads another set).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Geometry import Point3D

from ksmech.serve import depiction_metrics as M
from ksmech.serve import depiction_templates as T
from ksmech.serve.errors import Code, ServeError

DEPICTION_VERSION = 1
L = M.L
MAX_ATOMS = 250
BUDGET_S = 6.0  # optional candidates are skipped once a request has spent this long
RESIDUAL_WEIGHT = 20.0
STEREO_PENALTY = 50.0
TIE = 0.5  # a later candidate must beat an earlier one by this much
COORDGEN_TIE = 2.0  # CoordGen must beat RDKit's templates by this much (the gallery favours them)
PERSPECTIVE_SLACK = 15.0  # a perspective drawing may score this much worse than the textbook one
TEMPLATE_UPRIGHT_ABOVE = 0.8  # a template that fits the view worse than this stays upright
PERSPECTIVE_MAX_ATOMS = 80  # no 3D structure (and so no perspective) beyond this many atoms
VIEW_CONVENTION = "3dmol-rotationGroup"
_BOND_CODE = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


@dataclass
class Candidate:
    method: str
    xy: np.ndarray
    depth: np.ndarray | None = None
    template: str | None = None
    residual: float | None = None
    kind: str = "flat"  # flat | template | view (depiction_metrics.defects)
    defects: M.Defects | None = None
    wedges: dict[int, tuple[int, str]] = field(default_factory=dict)
    stereo: str = "none"
    smiles: str = ""
    molblock: str = ""

    @property
    def total(self) -> float:
        d = self.defects.score if self.defects else 0.0
        r = RESIDUAL_WEIGHT * self.residual if self.residual is not None else 0.0
        return d + r + (STEREO_PENALTY if self.stereo == "mismatch" else 0.0)


# ---- input


def parse_molblock(text: str | None) -> tuple[Chem.Mol, Chem.Mol]:
    """(sanitized mol, unsanitized mol with the file's bond orders), drawn H atoms kept."""
    if text is None or not text.strip():
        raise ServeError(Code.EMPTY_INPUT, "the molblock is empty", detail={"field": "molblock"})
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
        raw = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)
    if mol is None or raw is None:
        raise ServeError(Code.INVALID_MOLFILE, "RDKit cannot read the molblock")
    n = mol.GetNumAtoms()
    if n == 0:
        raise ServeError(Code.EMPTY_INPUT, "the molblock has no atoms")
    if n > MAX_ATOMS:
        msg = f"{n} atoms; the 2D depiction takes at most {MAX_ATOMS}"
        raise ServeError(Code.LIMIT_EXCEEDED, msg, detail={"n_atoms": n, "max": MAX_ATOMS})
    if any(a.HasQuery() or a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ServeError(Code.INVALID_MOLFILE, "query atoms are not molecules")
    return mol, raw


def view_matrix(q) -> np.ndarray:
    """R(q) of a quaternion [x, y, z, w] (normalized): v = R p, the same rotation as
    projection.ts rotateByQuat (v' = q v q*)."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-9:
        raise ServeError(Code.REQUEST_INVALID, "the view quaternion is zero or not finite")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---- layouts


def _positions(mol: Chem.Mol) -> np.ndarray:
    return mol.GetConformer().GetPositions()[:, :2].copy()


def normalized(mol: Chem.Mol, xy: np.ndarray) -> tuple[np.ndarray, float]:
    """(xy scaled so the median bond is L and centred on the origin, the factor)."""
    xy = np.asarray(xy, dtype=float)
    ls = [np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    med = float(np.median(ls)) if ls else L
    k = L / med if med > 1e-6 else 1.0
    return (xy - xy.mean(axis=0)) * k, k


def rdkit_layout(mol: Chem.Mol) -> np.ndarray:
    m = Chem.Mol(mol)
    with T.RDKIT_TEMPLATE_LOCK, rdBase.BlockLogs():
        rdDepictor.Compute2DCoords(m, forceRDKit=True, useRingTemplates=True)
    return normalized(mol, _positions(m))[0]


def coordgen_layout(mol: Chem.Mol) -> np.ndarray:
    m = Chem.Mol(mol)
    with rdBase.BlockLogs():
        rdCoordGen.AddCoords(m)
    return normalized(mol, _positions(m))[0]


def textbook_candidates(mol: Chem.Mol) -> list[Candidate]:
    out = []
    for method, fn in (("rdkit-templates", rdkit_layout), ("coordgen", coordgen_layout)):
        try:
            out.append(Candidate(method, fn(mol)))
        except (RuntimeError, ValueError) as exc:  # an RDKit depiction error
            out.append(Candidate(method, np.zeros((mol.GetNumAtoms(), 2)), stereo=f"failed: {exc}"))
    return [c for c in out if not c.stereo.startswith("failed")]


def perspective_candidate(mol: Chem.Mol, xyz: np.ndarray | None, method: str) -> Candidate | None:
    """The template layout as a candidate; None without a fitting template or when RDKit
    fails to lay out around it (an optional candidate never fails the request)."""
    try:
        lay = T.perspective_layout(mol, xyz)
    except (RuntimeError, ValueError):
        lay = None
    if lay is None:
        return None
    xy, k = normalized(mol, lay.xy)
    return Candidate(
        f"{method}:{lay.template}",
        xy,
        depth=(lay.depth - lay.depth.mean()) * k,
        template=lay.template,
        kind="template",
    )


# ---- wedges and the molblock


def _centres(mol: Chem.Mol) -> list[int]:
    with rdBase.BlockLogs():
        found = Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
    return [int(c) for c, _ in found]


def _with_coords(mol: Chem.Mol, xy: np.ndarray) -> Chem.Mol:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y) in enumerate(xy):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    m.AddConformer(conf, assignId=True)
    return m


def _mark_kind(m: Chem.Mol, bond: Chem.Bond, centre: int) -> str | None:
    bond.SetBondDir(Chem.BondDir.NONE)
    Chem.WedgeBond(bond, centre, m.GetConformer())
    d = bond.GetBondDir()
    bond.SetBondDir(Chem.BondDir.NONE)
    return {Chem.BondDir.BEGINWEDGE: "wedge", Chem.BondDir.BEGINDASH: "hash"}.get(d)


def _mark_score(m, bond, c, n, kind, centres, depth) -> float:
    score = 0.0 if bond.IsInRing() else 4.0
    score += 2.0 if m.GetAtomWithIdx(n).GetDegree() == 1 else 0.0
    score += 0.0 if n in centres else 3.0
    if depth is not None:
        dz = float(depth[n] - depth[c])
        score += 2.0 * abs(dz) / L
        if abs(dz) > 1e-6:
            score += 3.0 if (dz > 0) == (kind == "wedge") else -3.0
    return score


def assign_wedges(
    mol: Chem.Mol, xy: np.ndarray, depth: np.ndarray | None
) -> dict[int, tuple[int, str]]:
    """{bond index: (narrow-end atom, "wedge" | "hash")}, one mark per stereocentre."""
    m = _with_coords(mol, xy)
    centres = set(_centres(m))
    taken: dict[int, tuple[int, str]] = {}
    for c in sorted(centres):
        best = None
        for bond in m.GetAtomWithIdx(c).GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE or bond.GetIdx() in taken:
                continue
            kind = _mark_kind(m, bond, c)
            if kind is None:
                continue
            n = bond.GetOtherAtomIdx(c)
            s = _mark_score(m, bond, c, n, kind, centres, depth)
            if best is None or s > best[0]:
                best = (s, bond.GetIdx(), kind)
        if best is not None:
            taken[best[1]] = (c, best[2])
    return taken


def rdkit_wedges(mol: Chem.Mol, xy: np.ndarray) -> dict[int, tuple[int, str]]:
    m = _with_coords(mol, xy)
    for b in m.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    Chem.WedgeMolBonds(m, m.GetConformer())
    kinds = {Chem.BondDir.BEGINWEDGE: "wedge", Chem.BondDir.BEGINDASH: "hash"}
    return {
        b.GetIdx(): (b.GetBeginAtomIdx(), kinds[b.GetBondDir()])
        for b in m.GetBonds()
        if b.GetBondDir() in kinds
    }


def write_molblock(raw: Chem.Mol, xy: np.ndarray, wedges: dict[int, tuple[int, str]]) -> str:
    """V2000 in the input's atom and bond order and bond orders; wedges narrow end first."""
    n, nb = raw.GetNumAtoms(), raw.GetNumBonds()
    lines = ["", "  ksmech-depict2d   2D", "", f"{n:3d}{nb:3d}  0  0  0  0  0  0  0  0999 V2000"]
    for a, (x, y) in zip(raw.GetAtoms(), xy, strict=True):
        lines.append(f"{x:10.4f}{y:10.4f}{0.0:10.4f} {a.GetSymbol():<3} 0" + "  0" * 11)
    for b in raw.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        st = 0
        if b.GetIdx() in wedges:
            narrow, kind = wedges[b.GetIdx()]
            st = 1 if kind == "wedge" else 6
            if narrow != i:
                i, j = j, i
        lines.append(f"{i + 1:3d}{j + 1:3d}{_BOND_CODE.get(b.GetBondType(), 1):3d}{st:3d}")
    charged = [(a.GetIdx() + 1, a.GetFormalCharge()) for a in raw.GetAtoms() if a.GetFormalCharge()]
    for k in range(0, len(charged), 8):
        chunk = charged[k : k + 8]
        lines.append(f"M  CHG{len(chunk):3d}" + "".join(f"{i:4d}{c:4d}" for i, c in chunk))
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def molblock_smiles(molblock: str) -> str:
    """Canonical isomeric SMILES RDKit reads from a molblock ('' when it refuses it)."""
    with rdBase.BlockLogs():
        m = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
    return Chem.MolToSmiles(m) if m is not None else ""


def finish(c: Candidate, mol: Chem.Mol, raw: Chem.Mol, target: str) -> Candidate:
    """Wedges, molblock, SMILES, stereo outcome and defects of a candidate."""
    has_stereo = bool(_centres(mol)) or "/" in target or "\\" in target
    for wedges in (assign_wedges(mol, c.xy, c.depth), None):
        w = rdkit_wedges(mol, c.xy) if wedges is None else wedges
        c.wedges, c.molblock = w, write_molblock(raw, c.xy, w)
        c.smiles = molblock_smiles(c.molblock)
        if c.smiles == target:
            break
    if not has_stereo:
        c.stereo = "none" if c.smiles == target else "mismatch"
    else:
        c.stereo = "kept" if c.smiles == target else "mismatch"
    c.defects = M.defects(mol, c.xy, kind=c.kind)
    return c


def _pick(cands: list[Candidate]) -> Candidate:
    best = cands[0]
    for c in cands[1:]:
        if c.total < best.total - (COORDGEN_TIE if c.method == "coordgen" else TIE):
            best = c
    return best


def _result(best: Candidate, cands: list[Candidate], mode: str, style: str, notes: list[str]):
    return {
        "mode": mode,
        "style": style,
        "method": best.method,
        "template": best.template,
        "molblock": best.molblock,
        "coords": [[round(float(x), 4), round(float(y), 4)] for x, y in best.xy],
        "depth": None if best.depth is None else [round(float(z), 4) for z in best.depth],
        "wedges": [
            {"bond": int(k), "narrow": int(a), "kind": kind}
            for k, (a, kind) in sorted(best.wedges.items())
        ],
        "smiles": best.smiles,
        "stereo": best.stereo,
        "view_residual": None if best.residual is None else round(best.residual, 4),
        "defects": best.defects.as_dict() if best.defects else None,
        "candidates": [
            {
                "method": c.method,
                "total": round(c.total, 3),
                "score": c.defects.score if c.defects else None,
                "residual": None if c.residual is None else round(c.residual, 4),
                "stereo": c.stereo,
            }
            for c in cands
        ],
        "notes": notes,
        "version": DEPICTION_VERSION,
    }


def clean(mol: Chem.Mol, raw: Chem.Mol, style: str) -> dict:
    """The clean-up of a drawing (module docstring); stereo = the drawing's own."""
    target = Chem.MolToSmiles(mol)
    notes: list[str] = []
    cands = [finish(c, mol, raw, target) for c in textbook_candidates(mol)]
    if not cands:
        raise ServeError(Code.DEPICTION_FAILED, "RDKit could not lay out the drawing")
    best = _pick(cands)
    if style == "perspective":
        xyz = T.conformer3d(mol) if mol.GetNumAtoms() <= PERSPECTIVE_MAX_ATOMS else None
        pc = perspective_candidate(mol, xyz, "perspective")
        if pc is None:
            notes.append("no perspective template fits this molecule: textbook layout")
        else:
            finish(pc, mol, raw, target)
            cands.insert(0, pc)
            if pc.total <= best.total + PERSPECTIVE_SLACK:
                best = pc
            else:
                notes.append(f"the {pc.template} drawing has more defects: textbook layout")
    return _result(best, cands, "clean", style, notes)


def _turned(a: np.ndarray, b: np.ndarray, theta: float, spread: float):
    c, s = np.cos(theta), np.sin(theta)
    out = a @ np.array([[c, s], [-s, c]])
    return out, float(np.sqrt(((out - b) ** 2).sum(axis=1).mean())) / spread


def align_to(
    xy: np.ndarray, target: np.ndarray, mirror: bool = True, upright_above: float | None = None
) -> tuple[np.ndarray, float]:
    """(xy turned, and mirrored when allowed and that fits better, onto `target` about the
    centroids, no scaling; the RMS residual over the target's RMS spread). A perspective
    drawing is never mirrored (its picture would show the enantiomer); with `upright_above`,
    a turn whose residual is still above it is dropped (the view does not resemble the
    picture, and a chair or a norbornane turned at random reads worse than upright)."""
    a = xy - xy.mean(axis=0)
    b = target - target.mean(axis=0)
    spread = float(np.sqrt((b**2).sum(axis=1).mean())) or 1.0
    best = None
    for flip in (np.eye(2), np.diag([-1.0, 1.0]))[: 2 if mirror else 1]:
        am = a @ flip
        # the turn that best maps am onto b: angle of sum(am x b), sum(am . b)
        cross = float((am[:, 0] * b[:, 1] - am[:, 1] * b[:, 0]).sum())
        theta = float(np.arctan2(cross, float((am * b).sum())))
        out, res = _turned(am, b, theta, spread)
        if upright_above is not None and res > upright_above:
            out, res = _turned(am, b, 0.0, spread)
        if best is None or res < best[1]:
            best = (out, res)
    return best  # type: ignore[return-value]


def _drop_cage_stereo(ref: Chem.Mol) -> None:
    """Keep the 3D-read tetrahedral stereo only where it means something: CIP R / S, or r / s
    on an atom with at most two ring bonds (cis / trans on a ring). A bridgehead's r / s or an
    unlabelled tag (adamantane, cubane) is fixed by the cage and would only add wedges."""
    with rdBase.BlockLogs():
        found = Chem.FindMolChiralCenters(
            ref, includeUnassigned=True, useLegacyImplementation=False
        )
    label = {int(i): lab for i, lab in found}
    changed = False
    for a in ref.GetAtoms():
        if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        lab = label.get(a.GetIdx(), "?")
        ring_bonds = sum(b.IsInRing() for b in a.GetBonds())
        if lab in ("R", "S") or (lab in ("r", "s") and ring_bonds <= 2):
            continue
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        changed = True
    if changed:
        Chem.AssignStereochemistry(ref, cleanIt=True, force=True)


def _reference(mol: Chem.Mol, view: np.ndarray) -> Chem.Mol:
    """mol with a 3D conformer at the view-frame coordinates and its stereo from 3D."""
    ref = Chem.Mol(mol)
    ref.RemoveAllConformers()
    conf = Chem.Conformer(ref.GetNumAtoms())
    for i, p in enumerate(view):
        conf.SetAtomPosition(i, Point3D(*(float(v) for v in p)))
    conf.Set3D(True)
    ref.AddConformer(conf, assignId=True)
    with rdBase.BlockLogs():
        Chem.AssignStereochemistryFrom3D(ref)
    _drop_cage_stereo(ref)
    return ref


def view_frame(mol: Chem.Mol, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the projection x, y and the depth z) of view-frame coordinates, centred and scaled so
    the median 3D bond is L: a uniform scale of the 3D picture, foreshortening kept."""
    v, _ = normalized(mol, view)
    return v[:, :2], v[:, 2]


def _projected_distmat(xy: np.ndarray) -> np.ndarray:
    i, j = np.tril_indices(len(xy), -1)  # RDKit's order: d(i, j) at i (i - 1) / 2 + j, i > j
    return np.linalg.norm(xy[i] - xy[j], axis=1)


def _match3d_layouts(mol, ref, proj, ref_view, style, t0) -> list[Candidate]:
    out = []
    m = Chem.Mol(mol)
    with rdBase.BlockLogs():
        rdDepictor.GenerateDepictionMatching3DStructure(m, ref, forceRDKit=True)
    out.append(Candidate("match3d-rdkit", normalized(mol, _positions(m))[0]))
    if mol.GetNumAtoms() > 2:
        m = Chem.Mol(mol)
        with rdBase.BlockLogs():
            rdDepictor.Compute2DCoordsMimicDistmat(
                m, _projected_distmat(proj), canonOrient=False, forceRDKit=True
            )
        out.append(Candidate("match3d-projected", normalized(mol, _positions(m))[0]))
    try:
        lay = T.view_layout(mol, ref_view)
    except (RuntimeError, ValueError):  # an optional candidate never fails the request
        lay = None
    if lay is not None:
        xy = lay.xy - lay.xy.mean(axis=0)
        out.append(Candidate("match3d-view", xy, depth=lay.depth, kind="view"))
    if time.monotonic() - t0 < BUDGET_S:
        out.append(Candidate("match3d-textbook", rdkit_layout(mol)))
    if style == "perspective" and time.monotonic() - t0 < BUDGET_S:
        pc = perspective_candidate(mol, ref_view, "match3d-perspective")
        if pc is not None:
            out.append(pc)
    return out


def match3d(mol: Chem.Mol, raw: Chem.Mol, ref_xyz: np.ndarray, quat, style: str) -> dict:
    """The drawing of the computed structure `ref_xyz` (Angstrom, sketch atom order) as the 3D
    viewer shows it (module docstring); stereo = the structure's."""
    t0 = time.monotonic()
    ref_xyz = np.asarray(ref_xyz, dtype=float).reshape(-1, 3)
    if ref_xyz.shape[0] != mol.GetNumAtoms():
        msg = f"{ref_xyz.shape[0]} reference atoms for {mol.GetNumAtoms()} drawn atoms"
        raise ServeError(Code.MAPPING_FAILED, msg)
    view = (ref_xyz - ref_xyz.mean(axis=0)) @ view_matrix(quat).T
    ref = _reference(mol, view)
    work = Chem.Mol(ref)
    work.RemoveAllConformers()
    proj, depth = view_frame(mol, view)
    try:
        cands = _match3d_layouts(work, ref, proj, view, style, t0)
    except (RuntimeError, ValueError) as exc:
        msg = f"RDKit could not lay out the drawing: {exc}"
        raise ServeError(Code.DEPICTION_FAILED, msg) from exc
    target = Chem.MolToSmiles(work)
    for c in cands:
        upright = TEMPLATE_UPRIGHT_ABOVE if c.kind == "template" else None
        c.xy, c.residual = align_to(c.xy, proj, mirror=c.kind == "flat", upright_above=upright)
        if c.depth is None:
            c.depth = depth
        finish(c, work, raw, target)
    return _result(_pick(cands), cands, "match3d", style, [])
