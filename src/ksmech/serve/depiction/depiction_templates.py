"""Perspective ring templates of the sketcher's 2D depiction (`depiction2d`, style "perspective").

The library is two files under `serve/data/`:

- `ring_templates.smi`: one template per line in the format RDKit's
  `rdDepictor.SetRingSystemTemplates` reads (a CXSMILES of one ring system, generic `*` atoms
  and `~` bonds, 2D coordinates, then the name). `validate_with_rdkit` loads it into RDKit to
  prove the format; the test suite runs it.
- `ring_templates.json`: per name, the title, the examples it must be used for, its source and
  drawing convention, the depth and the 3D shape (`fit`) of every template atom (same units and
  atom order as the .smi coordinates: x right, y up, z towards the viewer), the bonds drawn
  short on purpose, the number of crossings the picture has, optional hand-set slot directions,
  and the derivation of `fit` (`depiction_gallery.derive_template`).

The drawing and the shape are kept apart. The .smi coordinates are a hand-made perspective
picture in the textbook style (Clayden, Eliel): every bond about one length (bonds receding
from the viewer may be drawn short on purpose, `foreshortened`), no two bonds at an atom closer
than ~40 deg or straighter than ~155 deg, no atom near a bond it is not on. `fit` is the
parent's MMFF structure turned to the orientation whose orthographic projection best matches
the picture (`depiction_gallery.derive_template`), in the same atom order and drawing units;
`depth` is its z, so the front / back of every atom (and of each crossing, which the sketcher
breaks) agrees with the picture. Matching uses `fit` (a real 3D shape); the exocyclic slots
(axial / equatorial, exo / endo) are the tetrahedral directions of `fit`, drawn through a
local linear map from `fit` onto the picture, kept at least SLOT_MIN_DEG from the other bonds
of the atom, optionally replaced by a hand-set direction (`slot_overrides`), and snapped onto
the vertical or a ring-bond direction when within SNAP_DEG (axial bonds vertical, equatorial
bonds parallel to ring bonds).

The templates are applied here, not by RDKit: RDKit's own template set is process-global
(switching it costs ~0.4 s) and RDKit never uses a template for a monocycle, so the chair could
not work. `perspective_layout` instead
1. matches each template (largest first) on the drawn molecule, keeping only matches whose atom
   set is exactly one ring system (rings sharing two or more atoms; spiro rings are separate);
2. fits the template's 3D shape (`fit`) to a 3D structure of the molecule (an ETKDG + MMFF
   conformer for a clean-up, the record for a 3D->2D), mirror image allowed, and keeps the best
   match if its RMSD is below `FIT_RMSD_MAX` bond lengths: a template is a conformation, so
   benzene never becomes a chair and cis- and trans-decalin pick their own drawing;
3. puts each exocyclic neighbour that is in no ring on the slot that its 3D bond points to, at
   one bond length along the slot's drawn direction (a neighbour in a ring, spiro or a ring
   substituent, is placed by RDKit: locked on a slot it squeezes its own ring);
4. locks the template atoms and these neighbours (`Compute2DCoords(coordMap=...)`, RDKit's
   depictor, `forceRDKit=True` so the global CoordGen preference never matters) and lets RDKit
   lay out the rest.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Geometry import Point2D

DATA = Path(__file__).with_name("data")
SMI_PATH = DATA / "ring_templates.smi"
META_PATH = DATA / "ring_templates.json"
BOND = 1.5  # drawing units of one bond (RDKit's depiction bond length)
FIT_RMSD_MAX = 0.12  # template-to-structure RMSD, in bond lengths
FIT_TIE = 0.02  # fits this close to the best RMSD are ranked by how they place substituents
EMBED_SEED = 0xF00D
EMBED_TIMEOUT_S = 3
EMBED_CONFS = 4
SLOT_HALF_ANGLE = np.deg2rad(109.47 / 2)
MIN_SLOT_XY = 0.3  # a slot drawn shorter than this (pointing at the viewer) is not used
SNAP_DEG = 8.0  # a drawn slot this close to vertical or to a ring bond is put on it
SLOT_MIN_DEG = 38.0  # a drawn slot is turned (by at most SLOT_TURN_DEG) to clear the atom's bonds
SLOT_MAX_DEG = 160.0  # ... and kept this far from straight through the atom
SLOT_TURN_DEG = 60.0
LOCAL_MAP_REG = 0.3  # pull of each atom's fit->picture map towards the global one

# RDKit's ring templates are global; anything that loads another set holds this lock and puts
# the default set back before releasing it (depiction2d holds it around useRingTemplates).
RDKIT_TEMPLATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Slot:
    """An exocyclic direction of a template atom: `u` the 3D unit vector (fit frame), `d` the
    drawn unit direction, `drawn` False when it points at the viewer (no locked position)."""

    u: np.ndarray
    d: np.ndarray
    drawn: bool


@dataclass(frozen=True)
class Template:
    name: str
    title: str
    query: Chem.Mol  # generic ring-system query, atom order of the .smi line
    xyz: np.ndarray  # (n, 3) drawing coordinates: x, y from the .smi, z from the JSON depth
    fit: np.ndarray  # (n, 3) the parent's 3D structure turned to the picture (drawing units)
    bonds: tuple[tuple[int, int], ...]
    slots: tuple[tuple[Slot, ...], ...]  # per atom
    examples: tuple[str, ...]
    source: str
    convention: str
    foreshortened: frozenset[tuple[int, int]] = frozenset()  # bonds drawn short on purpose
    crossings: int = 0  # bond crossings of the picture (drawn broken by the sketcher)

    @property
    def size(self) -> int:
        return int(self.query.GetNumAtoms())


@dataclass(frozen=True)
class Fit:
    template: Template
    match: tuple[int, ...]  # molecule atom of template atom k
    mirrored: bool
    rotation: np.ndarray  # template frame -> structure frame (proper rotation)
    rmsd: float  # in bond lengths


@dataclass(frozen=True)
class PerspectiveLayout:
    xy: np.ndarray  # (n_atoms, 2) drawing coordinates
    depth: np.ndarray  # (n_atoms,) + towards the viewer
    template: str
    rmsd: float
    short: frozenset[tuple[int, int]] = frozenset()  # foreshortened bonds (molecule atoms)


# ---- library


def _smi_lines(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{path}: expected '<cxsmiles> |coords| <name>', got {line!r}")
        out.append((" ".join(parts[:2]), parts[2]))
    return out


def _template(cx: str, name: str, meta: dict) -> Template:
    with rdBase.BlockLogs():
        query = Chem.MolFromSmarts(cx)
    if query is None or query.GetNumConformers() != 1:
        raise ValueError(f"template {name}: unreadable CXSMILES {cx!r}")
    xy = query.GetConformer().GetPositions()[:, :2]
    depth = np.asarray(meta["depth"], dtype=float)
    fit = np.asarray(meta["fit"], dtype=float)
    if depth.shape != (len(xy),) or fit.shape != (len(xy), 3):
        raise ValueError(f"template {name}: depth / fit do not match its {len(xy)} atoms")
    bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in query.GetBonds())
    overrides = {int(k): v for k, v in meta.get("slot_overrides", {}).items()}
    return Template(
        name=name,
        title=meta["title"],
        query=query,
        xyz=np.column_stack([xy, depth]),
        fit=fit,
        bonds=bonds,
        slots=slot_table(xy, fit, bonds, overrides, float(meta.get("snap_deg", SNAP_DEG))),
        examples=tuple(meta.get("examples", ())),
        source=meta["source"],
        convention=meta["convention"],
        foreshortened=frozenset(tuple(sorted(b)) for b in meta.get("foreshortened", ())),
        crossings=int(meta.get("crossings", 0)),
    )


@lru_cache(maxsize=4)
def load_library(smi_path: Path = SMI_PATH, meta_path: Path = META_PATH) -> tuple[Template, ...]:
    """The templates, largest first (loaded once per path pair)."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))["templates"]
    lines = _smi_lines(Path(smi_path))
    names = [n for _, n in lines]
    if sorted(names) != sorted(meta):
        raise ValueError(f"template names differ: .smi {sorted(names)} vs .json {sorted(meta)}")
    out = [_template(cx, name, meta[name]) for cx, name in lines]
    return tuple(sorted(out, key=lambda t: -t.size))


def validate_with_rdkit(smi_path: Path = SMI_PATH) -> None:
    """Load the file into RDKit's own template set (DepictException -> ValueError on a bad
    template), then restore RDKit's defaults."""
    with RDKIT_TEMPLATE_LOCK:
        try:
            rdDepictor.SetRingSystemTemplates(str(smi_path))
        finally:
            rdDepictor.LoadDefaultRingSystemTemplates()


# ---- geometry helpers


def ring_systems(mol: Chem.Mol) -> list[frozenset[int]]:
    """Atom sets of the ring systems: rings sharing two or more atoms are merged."""
    systems: list[set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for s in [s for s in systems if len(s & merged) >= 2]:
            merged |= s
            systems.remove(s)
        systems.append(merged)
    return [frozenset(s) for s in systems]


def conformer3d(mol: Chem.Mol) -> np.ndarray | None:
    """(n_atoms, 3) Angstrom of the lowest-energy of EMBED_CONFS srETKDGv3 + MMFF structures of
    `mol` (its own atoms; the H that AddHs appends are dropped), chirality from the molecule's
    tags; None when embedding fails. Plain ETKDGv3 often gives only twist-boats (all six
    structures of cis-4-tert-butylcyclohexanol tried); its small-ring variant samples the chairs."""
    mh = Chem.AddHs(mol)
    params = AllChem.srETKDGv3()
    params.randomSeed = EMBED_SEED
    params.timeout = EMBED_TIMEOUT_S
    with rdBase.BlockLogs():
        ids = list(AllChem.EmbedMultipleConfs(mh, EMBED_CONFS, params))
        if not ids:
            return None
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=200)
            energies = [e if conv != -1 else np.inf for conv, e in res]
        except (ValueError, RuntimeError):  # no MMFF parameters: keep the ETKDG structures
            energies = [0.0] * len(ids)
    best = ids[int(np.argmin(energies))]
    return mh.GetConformer(best).GetPositions()[: mol.GetNumAtoms()]


def _similarity_fit(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(R proper, scale, rms) with q ~ scale * R p (both centred)."""
    h = p.T @ q
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    corr = np.diag([1.0, 1.0, d])
    r = vt.T @ corr @ u.T
    scale = float((s * np.diag(corr)).sum() / max((p**2).sum(), 1e-12))
    resid = q - scale * (p @ r.T)
    return r, scale, float(np.sqrt((resid**2).sum(axis=1).mean()))


MIRROR = np.diag([-1.0, 1.0, 1.0])


def template_xyz(t: Template, mirrored: bool) -> np.ndarray:
    """The picture (x, y, depth), mirrored left-right for the enantiomer."""
    return t.xyz @ MIRROR if mirrored else t.xyz


def template_fit(t: Template, mirrored: bool) -> np.ndarray:
    """The 3D shape matched against a structure, mirrored like the picture."""
    return t.fit @ MIRROR if mirrored else t.fit


def _fit_match(t: Template, match: tuple[int, ...], xyz: np.ndarray, bond3d: float) -> Fit:
    q = xyz[list(match)]
    q = q - q.mean(axis=0)
    best = None
    for mirrored in (False, True):
        p = template_fit(t, mirrored)
        p = p - p.mean(axis=0)
        r, scale, rms = _similarity_fit(p, q)
        if scale <= 0:
            continue
        fit = Fit(t, match, mirrored, r, rms / bond3d)
        if best is None or fit.rmsd < best.rmsd:
            best = fit
    return best  # type: ignore[return-value]


def _mean_bond(mol: Chem.Mol, xyz: np.ndarray) -> float:
    d = [np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    return float(np.mean(d)) if d else 1.5


def best_fit(mol: Chem.Mol, xyz: np.ndarray, library: tuple[Template, ...]) -> Fit | None:
    """The best template fit on a whole ring system (largest system first, then lowest RMSD),
    or None when no template fits within FIT_RMSD_MAX. A symmetric template fits in several
    ways equally well (which bridge of a norbornane is in front, which bridgeheads of an
    adamantane carry the substituents): among the fits within FIT_TIE of the best, the one
    whose substituents land on the clearest slots wins (`_crowding`)."""
    systems = set(ring_systems(mol))
    bond3d = _mean_bond(mol, xyz)
    fits: list[Fit] = []
    for t in library:
        if fits and t.size < fits[0].template.size:
            break  # the library is sorted largest first
        with rdBase.BlockLogs():
            matches = mol.GetSubstructMatches(t.query, uniquify=False, maxMatches=2000)
        for match in matches:
            if frozenset(match) not in systems:
                continue
            fit = _fit_match(t, tuple(int(i) for i in match), xyz, bond3d)
            if fit is not None and fit.rmsd <= FIT_RMSD_MAX:
                fits.append(fit)
    if not fits:
        return None
    best = min(f.rmsd for f in fits)
    close = [f for f in fits if f.rmsd <= best + FIT_TIE]
    return min(close, key=lambda f: (_crowding(mol, xyz, f), f.rmsd))


def _crowding(mol: Chem.Mol, xyz: np.ndarray, fit: Fit) -> float:
    """How badly a fit would place the substituents: 1 per exocyclic neighbour it cannot put
    on a drawn slot, 0.3 per one drawn behind the ring (depth below the template's mean), 0.5
    per ring double bond drawn behind, and
    for each placed one the shortfall of its clearance from the other locked atoms (under
    0.8 bond) and bonds (under 0.5 bond)."""
    pos = _locked_positions(mol, xyz, fit)
    ring = set(fit.match)
    score = 0.0
    zmean = float(np.mean([pos[a][2] for a in ring]))
    locked_bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in pos and b.GetEndAtomIdx() in pos
    ]
    for b in mol.GetBonds():  # a double bond inside the cage, behind, reads poorly
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring and j in ring and b.GetBondType() == Chem.BondType.DOUBLE:
            score += 0.5 * float(pos[i][2] + pos[j][2] < 2 * zmean)
    for a in ring:
        for s in _exocyclic(mol, a, ring):
            if s not in pos:
                score += 1.0
                continue
            score += 0.3 * float(pos[s][2] < zmean)
            p = pos[s][:2]
            for o, q in pos.items():
                if o not in (s, a):
                    score += max(0.0, 0.8 - float(np.linalg.norm(p - q[:2])) / BOND)
            for i, j in locked_bonds:
                if s not in (i, j):
                    d = _segment_distance(p, pos[i][:2], pos[j][:2]) / BOND
                    score += max(0.0, 0.5 - d)
    return score


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


# ---- slots and placement


def _neighbours(n: int, bonds: tuple[tuple[int, int], ...]) -> list[list[int]]:
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def slots(xyz: np.ndarray, bonds: tuple[tuple[int, int], ...]) -> list[list[np.ndarray]]:
    """Unit exocyclic directions of every atom of a 3D ring system: two for an atom with two
    ring neighbours (tetrahedral), one for three, none for four."""
    out: list[list[np.ndarray]] = []
    for k, ns in enumerate(_neighbours(len(xyz), bonds)):
        us = [(xyz[n] - xyz[k]) / np.linalg.norm(xyz[n] - xyz[k]) for n in ns]
        out.append(_tetrahedral(us))
    return out


def _tetrahedral(us: list[np.ndarray]) -> list[np.ndarray]:
    total = np.sum(us, axis=0) if us else np.zeros(3)
    if len(us) == 3 or (len(us) == 2 and np.linalg.norm(np.cross(us[0], us[1])) < 1e-6):
        n = np.linalg.norm(total)
        return [-total / n] if n > 1e-6 else []
    if len(us) != 2:
        return []
    b = -total / np.linalg.norm(total)
    n = np.cross(us[0], us[1])
    n /= np.linalg.norm(n)
    c, s = np.cos(SLOT_HALF_ANGLE), np.sin(SLOT_HALF_ANGLE)
    return [b * c + n * s, b * c - n * s]


def _linear_map(x: np.ndarray, y: np.ndarray, prior: np.ndarray, reg: float) -> np.ndarray:
    """The 2x3 map M minimising |x M^T - y|^2 + reg |M - prior|^2 (rows of x: 3D vectors)."""
    a = x.T @ x + reg * np.eye(3)
    return np.linalg.solve(a, x.T @ y + reg * prior.T).T


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def _clearance_deg(d: np.ndarray, others: list[np.ndarray]) -> float:
    """How far (deg) d stays inside [SLOT_MIN_DEG, SLOT_MAX_DEG] from every direction in
    `others` (negative when outside)."""
    angles = [float(np.degrees(np.arccos(np.clip(d @ o, -1.0, 1.0)))) for o in others]
    return min((min(a - SLOT_MIN_DEG, SLOT_MAX_DEG - a) for a in angles), default=90.0)


def _clear_turn(d: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """d turned by the smallest angle (at most SLOT_TURN_DEG) that puts it between
    SLOT_MIN_DEG and SLOT_MAX_DEG from every direction in `others`; the clearest turn when
    none does."""
    base = np.arctan2(d[1], d[0])
    best, best_gap = d, None
    for step in np.deg2rad(np.arange(0.0, SLOT_TURN_DEG + 0.5, 1.0)):
        for t in (base + step, base - step):
            c = np.array([np.cos(t), np.sin(t)])
            gap = _clearance_deg(c, others)
            if gap >= 0:
                return c
            if best_gap is None or gap > best_gap:
                best, best_gap = c, gap
    return best


def slot_table(
    xy: np.ndarray,
    fit: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    overrides: dict,
    snap_deg: float = SNAP_DEG,
) -> tuple[tuple[Slot, ...], ...]:
    """The exocyclic slots of every template atom (module docstring): 3D directions from
    `fit`, drawn through each atom's fit->picture map, cleared, snapped (within `snap_deg`,
    larger for the stylised chairs), overridden."""
    nbrs = _neighbours(len(xy), bonds)
    dx = np.array([fit[b] - fit[a] for a, b in bonds])
    dy = np.array([xy[b] - xy[a] for a, b in bonds])
    prior = _linear_map(dx, dy, np.zeros((2, 3)), 1e-9)
    lines = _bond_directions(xy, bonds)
    out = []
    for k, ns in enumerate(nbrs):
        m = _linear_map(fit[ns] - fit[k], xy[ns] - xy[k], prior, LOCAL_MAP_REG)
        ring = [_unit(xy[n] - xy[k]) for n in ns]
        placed: list[Slot] = []
        for u in sorted(slots(fit, bonds)[k], key=lambda v: -np.hypot(v[0], v[1])):
            d = _clear_turn(_unit(m @ u), ring + [s.d for s in placed if s.drawn])
            snapped = _snap(d, lines, snap_deg)
            if _clearance_deg(snapped, ring + [s.d for s in placed if s.drawn]) >= 0:
                d = snapped
            placed.append(Slot(u, d, bool(np.hypot(u[0], u[1]) >= MIN_SLOT_XY)))
        out.append(_override(placed, overrides.get(k, ())))
    return tuple(out)


def _override(placed: list[Slot], wants) -> tuple[Slot, ...]:
    """The slots with their drawn directions replaced by the hand-set ones, each given to a
    different slot so that the total turn is smallest."""
    ws = [_unit(np.asarray(w, dtype=float)) for w in wants]
    if not ws:
        return tuple(placed)
    j = _assign(ws, [s.d for s in placed])
    out = list(placed)
    for w, k in zip(ws, j, strict=True):
        if k is not None:
            out[k] = Slot(placed[k].u, w, True)
    return tuple(out)


def _assign(dirs: list[np.ndarray], cand: list[np.ndarray]) -> list[int | None]:
    """Slot index of each direction (best total cosine, each slot once); None when there are
    more directions than slots."""
    if not cand:
        return [None] * len(dirs)
    if len(dirs) == 1:
        return [int(np.argmax([float(d @ dirs[0]) for d in cand]))]
    if len(dirs) == 2 and len(cand) == 2:
        keep = float(dirs[0] @ cand[0] + dirs[1] @ cand[1])
        swap = float(dirs[0] @ cand[1] + dirs[1] @ cand[0])
        return [0, 1] if keep >= swap else [1, 0]
    return [None] * len(dirs)


def _exocyclic(mol: Chem.Mol, atom: int, ring: set[int]) -> list[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(atom).GetNeighbors() if n.GetIdx() not in ring]


def _locked_positions(mol: Chem.Mol, xyz: np.ndarray, fit: Fit) -> dict[int, np.ndarray]:
    """Drawing (x, y, z) of the template atoms and of their exocyclic neighbours."""
    t = fit.template
    txyz = template_xyz(t, fit.mirrored)
    flip = np.array([-1.0, 1.0, 1.0]) if fit.mirrored else np.ones(3)
    pos = {a: txyz[k] for k, a in enumerate(fit.match)}
    ring = set(fit.match)
    for k, a in enumerate(fit.match):
        subs = _exocyclic(mol, a, ring)
        if not subs:
            continue
        cand = t.slots[k]
        dirs = [_unit(fit.rotation.T @ (xyz[s] - xyz[a])) for s in subs]
        for s, j in zip(subs, _assign(dirs, [c.u * flip for c in cand]), strict=True):
            if j is None or s in pos or mol.GetAtomWithIdx(s).IsInRing() or not cand[j].drawn:
                continue  # a ring neighbour (spiro, a ring substituent) is left to RDKit
            d, uz = cand[j].d * flip[:2], cand[j].u[2]
            pos[s] = txyz[k] + np.array([BOND * d[0], BOND * d[1], BOND * uz])
    return pos


def _bond_directions(xy: np.ndarray, bonds: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    """Vertical plus the drawn direction of every template bond (unit 2D vectors)."""
    return [np.array([0.0, 1.0])] + [_unit(xy[b, :2] - xy[a, :2]) for a, b in bonds]


def _snap(d: np.ndarray, lines: list[np.ndarray], deg: float = SNAP_DEG) -> np.ndarray:
    """d turned onto the nearest line (either sense) within `deg`: axial bonds exactly
    vertical and equatorial bonds exactly parallel to ring bonds, as chemists draw them."""
    best, cos = d, np.cos(np.deg2rad(deg))
    for line in lines:
        c = float(d @ line)
        if abs(c) > cos:
            best, cos = (line if c > 0 else -line), abs(c)
    return best


def _propagate_depth(mol: Chem.Mol, known: dict[int, float]) -> np.ndarray:
    """Depth of every atom: known values, else the depth of the nearest known atom (BFS)."""
    depth = np.zeros(mol.GetNumAtoms())
    seen = dict(known)
    frontier = list(known)
    while frontier:
        nxt = []
        for a in frontier:
            for n in mol.GetAtomWithIdx(a).GetNeighbors():
                if n.GetIdx() not in seen:
                    seen[n.GetIdx()] = seen[a]
                    nxt.append(n.GetIdx())
        frontier = nxt
    for a, z in seen.items():
        depth[a] = z
    return depth


def perspective_layout(
    mol: Chem.Mol, xyz: np.ndarray | None, library: tuple[Template, ...] | None = None
) -> PerspectiveLayout | None:
    """The perspective drawing of `mol` (atoms in its own order) from the 3D structure `xyz`,
    or None when no template fits (see the module docstring)."""
    if xyz is None:
        return None
    fit = best_fit(mol, xyz, load_library() if library is None else library)
    if fit is None:
        return None
    pos = _locked_positions(mol, xyz, fit)
    cmap = {a: Point2D(float(p[0]), float(p[1])) for a, p in pos.items()}
    work = Chem.Mol(mol)
    with rdBase.BlockLogs():
        rdDepictor.Compute2DCoords(work, coordMap=cmap, canonOrient=False, forceRDKit=True)
    xy = work.GetConformer().GetPositions()[:, :2]
    locked = np.array(list(pos))
    shift = np.array([[p[0], p[1]] for p in pos.values()]) - xy[locked]
    xy = xy + shift.mean(axis=0)  # RDKit keeps the locked frame; guard against a translation
    depth = _propagate_depth(mol, {a: float(p[2]) for a, p in pos.items()})
    m = fit.match
    short = frozenset(tuple(sorted((m[a], m[b]))) for a, b in fit.template.foreshortened)
    return PerspectiveLayout(xy, depth, fit.template.name, fit.rmsd, short)


def view_layout(mol: Chem.Mol, view: np.ndarray) -> PerspectiveLayout | None:
    """The ring systems drawn as the 3D view shows them: every ring atom locked at its
    projected position (view-frame x, y, scaled so the mean 3D bond is BOND), each exocyclic
    neighbour in no ring one bond length along its projected bond (unless that bond points at
    the viewer), the rest laid out by RDKit; the depth is the view's. None without rings. The
    view acts as a template made on the spot, so a 3D->2D of a cage keeps its picture."""
    systems = ring_systems(mol)
    if not systems:
        return None
    v = (view - view.mean(axis=0)) * (BOND / _mean_bond(mol, view))
    ring = set().union(*systems)
    pos = {a: v[a] for a in ring}
    for a in ring:
        for s in _exocyclic(mol, a, ring):
            d = v[s] - v[a]
            dxy = float(np.hypot(d[0], d[1]))
            if s in pos or mol.GetAtomWithIdx(s).IsInRing() or dxy < MIN_SLOT_XY * BOND:
                continue
            pos[s] = v[a] + np.array([BOND * d[0] / dxy, BOND * d[1] / dxy, d[2]])
    cmap = {a: Point2D(float(p[0]), float(p[1])) for a, p in pos.items()}
    work = Chem.Mol(mol)
    with rdBase.BlockLogs():
        rdDepictor.Compute2DCoords(work, coordMap=cmap, canonOrient=False, forceRDKit=True)
    xy = work.GetConformer().GetPositions()[:, :2]
    locked = np.array(list(pos))
    xy = xy + (np.array([[p[0], p[1]] for p in pos.values()]) - xy[locked]).mean(axis=0)
    return PerspectiveLayout(xy, v[:, 2].copy(), "view", 0.0)
