"""Coordination-aware start geometries for transition-metal complexes (workbench adapter).

The standard path (`qm.embed`: ETKDGv3, then MMFF or UFF) knows nothing about metal centres:
MMFF has no transition-metal parameters and UFF types no charged metal (raw ETKDG coordinates
are kept), ETKDG embeds every four-coordinate metal tetrahedral (also square-planar Pt(II)), and
it fails on five- and six-coordinate metals without SMILES stereo and on ferrocene. GFN2 then
starts from the wrong polyhedron, loses the isomer or breaks a bond (docs/decisions.md,
2026-10-25). This module builds the start geometry from the coordination sphere instead:

1. **Sites** (`find_sites`): the metal's neighbours, dative or covalent, with neighbours that
   are bonded to each other grouped into one eta-n site (a Cp ring, an alkene).
2. **Polyhedron** (`choose_polyhedron`): by the number of sites; four-coordinate d8 Pd(II),
   Pt(II), Au(III), Rh(I), Ir(I) (and Ni(II) with C/P donors or only N donors) is square planar,
   any other four-coordinate centre tetrahedral; five-coordinate d6 is square pyramidal, other
   five-coordinate centres trigonal bipyramidal. SMILES stereo (@SP, @TB, @OH) decides.
3. **Isomer** (`assign_slots`): SMILES stereo when given (cis/trans, fac/mer; the handedness of
   non-tetrahedral centres is not enforced, as RDKit's embedder does not either); otherwise the
   stated default: chelate donors cis, and as many pairs of like ligands trans as possible
   (trans-MA4B2, mer-MA3B3, trans-MA2B2).
4. **Embedding** (`build_start_geometry`): donors at covalent-radius distances along the
   polyhedron vertices (eta-n rings as regular polygons across the other sites), the ligands
   embedded by ETKDG with the metal-ligand bonds removed, the template atoms in the coordMap
   and their pair distances in the bounds matrix (inter-ligand van der Waals floors capped),
   then each ligand moved rigidly onto its donors, each monodentate ligand turned so its
   lone-pair direction points at the metal, and eta-n substituents laid in the ring plane.

Not handled (MetalTemplateError, the standard embed is used and the record carries
METAL_TEMPLATE_SKIPPED): more than one metal, more than one fragment, more than six sites.
Open-shell centres are rejected before any compute (`preflight_problems`).

The GFN2 optimization that follows (`serve.metal_qm`) checks metal-ligand bonds with their own
tolerance. Records it writes carry the method `<standard method>+metal-geom-v1`
(`metal_method`), so they never collide with the QM batch's records. Pure RDKit and numpy: no
xTB here.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

log = logging.getLogger(__name__)

METAL_GEOM_TAG = "+metal-geom-v1"
TRANSITION_METALS = frozenset([*range(21, 31), *range(39, 49), 57, *range(72, 81)])
# A metal-ligand bond is broken beyond ML_COV_FACTOR x the covalent radii sum; walls on it
# start at ML_WALL_FACTOR. Kept equal to the organic 1.3 / 1.2 on the 2026-10-25 test set:
# GFN2 M-L bonds of the correct geometries lie at 0.81-1.03 x the (Alvarez) radii sum, and the
# one dissociation seen (a Pt-Cl at 1.38x from a tetrahedral start) must still be caught.
ML_COV_FACTOR = 1.3
ML_WALL_FACTOR = 1.2
TRANS_DEG = 150.0  # two sites at least this far apart are trans
STRONG_FIELD_Z = frozenset({1, 6, 14, 15})  # H, C, Si, P donors
_ELECTROPOSITIVE = frozenset({3, 4, 11, 12, 13, 19, 20, 37, 38, 55, 56})  # Li Be Na Mg Al K ...
_D8_SQUARE_PLANAR = frozenset({27, 29, 45, 46, 47, 77, 78, 79})  # Co Cu Rh Pd Ag Ir Pt Au
_SP_CARBON_R = 0.69  # Alvarez radius of an sp carbon (CO, CN-), shorter than RDKit's 0.76
_ETA_BOND = 1.40  # C-C of an eta-n ligand (Cp 1.42, benzene 1.40, alkene 1.40)
_ML_SHRINK = 0.05  # M-L start distance = r_M + r_L - this
_GOOD_CONTACT = 1.5  # `contact_ratio` below this (a clash) tries the next ETKDG seed
_N_SEEDS = 8
_MAX_FIT_DEV = 0.3  # A: a ligand whose donors miss the template by more is refused
_METAL_LOWER = 2.2  # A: cap on the ETKDG lower bounds of the (isolated) metal atom

_S3 = np.sqrt(3.0) / 2.0
_SPY = np.radians(100.0)  # apical-basal angle of the square pyramid
POLYHEDRA: dict[str, np.ndarray] = {
    "terminal": np.array([[0.0, 0.0, 1.0]]),
    "linear": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    "trigonal": np.array([[1.0, 0.0, 0.0], [-0.5, _S3, 0.0], [-0.5, -_S3, 0.0]]),
    "tetrahedral": np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0),
    "square_planar": np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]]),
    "tbp": np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [-0.5, _S3, 0], [-0.5, -_S3, 0]]),
    "square_pyramidal": np.array(
        [[0, 0, 1.0]]
        + [
            [np.sin(_SPY) * np.cos(a), np.sin(_SPY) * np.sin(a), np.cos(_SPY)]
            for a in np.radians([0.0, 90.0, 180.0, 270.0])
        ]
    ),
    "octahedral": np.array(
        [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    ),
}
BY_CN: dict[int, tuple[str, ...]] = {
    1: ("terminal",),
    2: ("linear",),
    3: ("trigonal",),
    4: ("tetrahedral", "square_planar"),
    5: ("tbp", "square_pyramidal"),
    6: ("octahedral",),
}
# RDKit's non-tetrahedral stereo: neighbour k (bond order on the atom) -> vertex of POLYHEDRA.
# Derived by embedding proxy complexes with six distinct ligands and aligning them to the ideal
# polyhedra (tests/test_serve_metal_geometry.py re-derives them); they fix which neighbours are
# trans (OpenSMILES @SP1 U, @SP2 4, @SP3 Z; @TBn and @OHn axes), not the handedness.
_SP_TABLE = {1: (0, 1, 2, 3), 2: (0, 2, 1, 3), 3: (0, 1, 3, 2)}
_TB_TABLE = {
    1: (0, 2, 4, 3, 1), 2: (0, 2, 4, 3, 1), 3: (0, 2, 4, 1, 3), 4: (0, 2, 4, 1, 3),
    5: (0, 2, 1, 3, 4), 6: (0, 2, 1, 3, 4), 7: (0, 1, 2, 4, 3), 8: (0, 1, 2, 4, 3),
    9: (2, 0, 4, 3, 1), 10: (2, 0, 3, 1, 4), 11: (2, 0, 4, 3, 1), 12: (2, 0, 3, 1, 4),
    13: (2, 0, 1, 4, 3), 14: (2, 0, 1, 4, 3), 15: (2, 3, 0, 4, 1), 16: (2, 3, 0, 1, 4),
    17: (2, 3, 4, 0, 1), 18: (2, 3, 4, 0, 1), 19: (2, 3, 0, 1, 4), 20: (2, 3, 0, 4, 1),
}  # fmt: skip
_OH_TABLE = {
    1: (0, 2, 4, 3, 5, 1), 2: (0, 2, 4, 3, 5, 1), 3: (0, 2, 4, 3, 1, 5),
    4: (0, 2, 4, 5, 3, 1), 5: (0, 2, 5, 4, 1, 3), 6: (0, 2, 5, 1, 3, 4),
    7: (0, 2, 4, 1, 5, 3), 8: (0, 2, 3, 4, 5, 1), 9: (0, 2, 3, 4, 1, 5),
    10: (0, 2, 3, 4, 5, 1), 11: (0, 2, 3, 4, 1, 5), 12: (0, 2, 3, 1, 4, 5),
    13: (0, 2, 3, 1, 4, 5), 14: (0, 2, 4, 5, 3, 1), 15: (0, 2, 5, 4, 1, 3),
    16: (0, 2, 4, 3, 1, 5), 17: (0, 2, 4, 1, 5, 3), 18: (0, 2, 5, 1, 3, 4),
    19: (0, 2, 1, 5, 3, 4), 20: (0, 2, 1, 5, 4, 3), 21: (0, 2, 1, 3, 4, 5),
    22: (0, 2, 1, 3, 4, 5), 23: (0, 2, 1, 5, 4, 3), 24: (0, 2, 1, 5, 3, 4),
    25: (0, 1, 2, 4, 3, 5), 26: (0, 1, 2, 5, 4, 3), 27: (0, 1, 2, 3, 5, 4),
    28: (0, 1, 2, 3, 5, 4), 29: (0, 1, 2, 5, 4, 3), 30: (0, 1, 2, 4, 3, 5),
}  # fmt: skip
_CT = Chem.ChiralType
STEREO_TABLES = {
    _CT.CHI_SQUAREPLANAR: ("square_planar", _SP_TABLE),
    _CT.CHI_TRIGONALBIPYRAMIDAL: ("tbp", _TB_TABLE),
    _CT.CHI_OCTAHEDRAL: ("octahedral", _OH_TABLE),
}


class MetalTemplateError(Exception):
    """No template geometry for this complex (the caller falls back to the standard embed)."""


# ---- method tag


def metal_method(base: str) -> str:
    """The method string of a record built on the metal path."""
    return base if base.endswith(METAL_GEOM_TAG) else base + METAL_GEOM_TAG


@lru_cache(maxsize=4096)
def key_has_metal(key: str) -> bool:
    """Whether the SMILES holds a transition-metal atom (False when it does not parse)."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(key, sanitize=False)
    return mol is not None and any(a.GetAtomicNum() in TRANSITION_METALS for a in mol.GetAtoms())


def method_for(key: str, base: str) -> str:
    """`base` + the metal tag for a GFN2 method and a key with a transition metal; else `base`
    unchanged (non-metal molecules and the synthetic method keep the standard path)."""
    if not base.startswith("GFN2-xTB") or not key_has_metal(key):
        return base
    return metal_method(base)


def metal_atoms(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in TRANSITION_METALS]


def is_metal_bond(mol: Chem.Mol, i: int, j: int) -> bool:
    zs = (mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum())
    return any(z in TRANSITION_METALS for z in zs)


# ---- sites, oxidation state, d count


@dataclass(frozen=True)
class Site:
    atoms: tuple[int, ...]  # one donor, or the eta-n atoms in ring / chain order

    @property
    def hapto(self) -> int:
        return len(self.atoms)


def _walk(mol: Chem.Mol, group: list[int]) -> tuple[int, ...]:
    """The atoms of an eta-n group in ring or chain order (starting at a chain end)."""
    inside = set(group)
    nbrs = {
        i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in inside]
        for i in group
    }
    start = min(group, key=lambda i: (len(nbrs[i]), group.index(i)))
    order, prev = [start], None
    while len(order) < len(group):
        nxt = [n for n in nbrs[order[-1]] if n != prev and n not in order]
        if not nxt:
            return tuple(group)  # branched: keep the bond order
        prev = order[-1]
        order.append(min(nxt, key=group.index))
    return tuple(order)


def find_sites(mol: Chem.Mol, m: int) -> list[Site]:
    """The coordination sites of metal `m`, in the order of its bonds (the SMILES stereo order).
    Neighbours bonded to each other form one eta-n site; metal-metal bonds are not sites."""
    atom = mol.GetAtomWithIdx(m)
    nbrs = [b.GetOtherAtomIdx(m) for b in atom.GetBonds()]
    nbrs = [n for n in nbrs if mol.GetAtomWithIdx(n).GetAtomicNum() not in TRANSITION_METALS]
    parent = {n: n for n in nbrs}

    def root(n: int) -> int:
        while parent[n] != n:
            n = parent[n]
        return n

    for a, b in itertools.combinations(nbrs, 2):
        if mol.GetBondBetweenAtoms(a, b) is not None:
            parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for n in nbrs:
        groups.setdefault(root(n), []).append(n)
    return [Site(_walk(mol, g) if len(g) > 2 else tuple(g)) for g in groups.values()]


def oxidation_state(mol: Chem.Mol, m: int) -> int:
    """Formal charge plus the order of every covalent bond to a non-metal: dative bonds give 0,
    so `N->[Co+3]` is Co(III) and `Cl[Pt-2](Cl)(Cl)Cl` Pt(II). A double bond to carbon counts 0
    (a carbene or M=C=O drawn as L-type), so a Grubbs carbene `[Ru](=C)` stays Ru(II); a bond
    to an alkali, alkaline-earth metal or Al counts -1 (the metal is the electronegative end)."""
    atom = mol.GetAtomWithIdx(m)
    ox = atom.GetFormalCharge() + atom.GetNumExplicitHs() + atom.GetNumImplicitHs()
    for b in atom.GetBonds():
        other = b.GetOtherAtom(atom)
        if b.GetBondType() == Chem.BondType.DATIVE or other.GetAtomicNum() in TRANSITION_METALS:
            continue
        order = b.GetBondTypeAsDouble()
        if order == 2.0 and other.GetAtomicNum() == 6:
            continue
        sign = -1 if other.GetAtomicNum() in _ELECTROPOSITIVE else 1  # Cu-Mg: a cuprate
        ox += sign * int(round(order))
    return ox


def group_number(z: int) -> int | None:
    """IUPAC group of a d-block element (La counts as group 3)."""
    for lo, hi, base in ((21, 30, 18), (39, 48, 36), (72, 80, 68)):
        if lo <= z <= hi:
            return z - base
    return 3 if z == 57 else None


def d_count(mol: Chem.Mol, m: int) -> int | None:
    group = group_number(mol.GetAtomWithIdx(m).GetAtomicNum())
    if group is None:
        return None
    d = group - oxidation_state(mol, m)
    return d if 0 <= d <= 10 else None


def _strong_field(mol: Chem.Mol, sites: list[Site]) -> bool:
    """A C, P, Si or H donor, or an eta-n ligand: an organometallic ligand field."""
    return any(
        s.hapto > 1 or mol.GetAtomWithIdx(s.atoms[0]).GetAtomicNum() in STRONG_FIELD_Z
        for s in sites
    )


# ---- polyhedron and isomer


def choose_polyhedron(
    mol: Chem.Mol, m: int, sites: list[Site], override: str | None = None
) -> str | None:
    """The ideal polyhedron of metal `m` (None for more than six sites)."""
    n = len(sites)
    if override is not None:
        if override not in BY_CN.get(n, ()):
            raise ValueError(f"{override!r} does not fit {n} sites (choose from {BY_CN.get(n)})")
        return override
    if n not in BY_CN:
        return None
    tag = mol.GetAtomWithIdx(m).GetChiralTag()
    if tag in STEREO_TABLES and STEREO_TABLES[tag][0] in BY_CN[n]:
        return STEREO_TABLES[tag][0]
    d = d_count(mol, m)
    z = mol.GetAtomWithIdx(m).GetAtomicNum()
    if n == 4:
        donors = [mol.GetAtomWithIdx(s.atoms[0]).GetAtomicNum() for s in sites]
        nickel_sp = z == 28 and (_strong_field(mol, sites) or set(donors) == {7})
        return (
            "square_planar" if d == 8 and (z in _D8_SQUARE_PLANAR or nickel_sp) else "tetrahedral"
        )
    if n == 5:
        return "square_pyramidal" if d == 6 else "tbp"
    return BY_CN[n][0]


def _stereo_slots(mol: Chem.Mol, m: int, sites: list[Site], poly: str) -> tuple[int, ...] | None:
    """Vertices from the SMILES stereo of metal `m`, or None when it gives none."""
    atom = mol.GetAtomWithIdx(m)
    tag = atom.GetChiralTag()
    if any(s.hapto > 1 for s in sites) or len(sites) != atom.GetDegree():
        return None
    if tag in STEREO_TABLES and STEREO_TABLES[tag][0] == poly:
        perm = (
            atom.GetUnsignedProp("_chiralPermutation") if atom.HasProp("_chiralPermutation") else 0
        )
        return STEREO_TABLES[tag][1].get(int(perm) or 1)
    if poly == "tetrahedral" and tag == _CT.CHI_TETRAHEDRAL_CCW:
        return (0, 1, 2, 3)  # looking from neighbour 0, neighbours 1-3 anticlockwise
    if poly == "tetrahedral" and tag == _CT.CHI_TETRAHEDRAL_CW:
        return (0, 1, 3, 2)
    return None


def _fragment_of(mol: Chem.Mol) -> list[int]:
    """Ligand id of every atom: connected components once the metal atoms are removed."""
    label = [-1] * mol.GetNumAtoms()
    for start in range(mol.GetNumAtoms()):
        if label[start] >= 0 or mol.GetAtomWithIdx(start).GetAtomicNum() in TRANSITION_METALS:
            continue
        stack = [start]
        label[start] = start
        while stack:
            for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                i = nb.GetIdx()
                if label[i] < 0 and nb.GetAtomicNum() not in TRANSITION_METALS:
                    label[i] = start
                    stack.append(i)
    return label


def assign_slots(
    mol: Chem.Mol, m: int, sites: list[Site], poly: str
) -> tuple[tuple[int, ...], str]:
    """(vertex of each site, source): SMILES stereo when present, else the default isomer
    (chelate donors cis, then the most pairs of like ligands trans; ties: the first
    permutation)."""
    slots = _stereo_slots(mol, m, sites, poly)
    if slots is not None:
        return slots, "smiles-stereo"
    verts = POLYHEDRA[poly]
    frag = _fragment_of(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    sig = [tuple(sorted(ranks[a] for a in s.atoms)) for s in sites]
    pairs = list(itertools.combinations(range(len(sites)), 2))
    chelate = [(i, j) for i, j in pairs if frag[sites[i].atoms[0]] == frag[sites[j].atoms[0]]]
    like = [(i, j) for i, j in pairs if sig[i] == sig[j]]
    best, best_score = None, None
    for perm in itertools.permutations(range(len(verts)), len(sites)):
        score = 100 * _n_trans(verts, perm, chelate) - _n_trans(verts, perm, like)
        if best_score is None or score < best_score:
            best, best_score = perm, score
    return tuple(best), "default"


def _n_trans(verts: np.ndarray, perm: tuple[int, ...], pairs: list[tuple[int, int]]) -> int:
    cos_trans = np.cos(np.radians(TRANS_DEG))
    return sum(int(verts[perm[i]] @ verts[perm[j]] < cos_trans) for i, j in pairs)


@dataclass(frozen=True)
class CentrePlan:
    metal: int
    sites: tuple[Site, ...]
    oxidation: int
    d_count: int | None
    polyhedron: str
    slots: tuple[int, ...]  # vertex of POLYHEDRA[polyhedron] for each site
    isomer_source: str  # "smiles-stereo" | "default"

    def describe(self) -> str:
        d = "?" if self.d_count is None else self.d_count
        return f"{self.polyhedron}(CN{len(self.sites)},d{d},{self.isomer_source})"


def plan_centre(mol: Chem.Mol, m: int, override: str | None = None) -> CentrePlan:
    """Sites, polyhedron and isomer of metal `m`; MetalTemplateError without a template."""
    sites = find_sites(mol, m)
    if not sites:
        raise MetalTemplateError(f"metal atom {m} has no ligands")
    poly = choose_polyhedron(mol, m, sites, override)
    if poly is None:
        raise MetalTemplateError(f"metal atom {m} has {len(sites)} sites (templates: 1-6)")
    slots, source = assign_slots(mol, m, sites, poly)
    return CentrePlan(
        m, tuple(sites), oxidation_state(mol, m), d_count(mol, m), poly, slots, source
    )


def plan_complex(mol: Chem.Mol, overrides: Mapping[int, str] | None = None) -> list[CentrePlan]:
    """One plan per metal centre. Only mononuclear complexes have a template: two metals (a
    bridging ligand, a metal-metal bond, or two centres on one ligand) is MetalTemplateError."""
    metals = metal_atoms(mol)
    if len(Chem.GetMolFrags(mol)) > 1:
        raise MetalTemplateError("more than one fragment (templates: one connected complex)")
    if len(metals) != 1:
        raise MetalTemplateError(f"{len(metals)} metal centres (templates: mononuclear only)")
    return [plan_centre(mol, metals[0], (overrides or {}).get(metals[0]))]


# ---- spin: the pipeline is closed-shell


def _aromatic_n(mol: Chem.Mol, sites: list[Site]) -> int:
    return sum(
        1
        for s in sites
        if s.hapto == 1
        and mol.GetAtomWithIdx(s.atoms[0]).GetAtomicNum() == 7
        and mol.GetAtomWithIdx(s.atoms[0]).GetIsAromatic()
    )


def open_shell_reason(mol: Chem.Mol, plan: CentrePlan) -> str | None:
    """Why the centre's ground state is likely open-shell although its electron count is even,
    or None. Ligand-field rule for Werner complexes (no C, P, Si, H or eta-n donor):
    octahedral d2, d4, d8 always have unpaired electrons, octahedral d6 of a 3d metal is high
    spin (except Co(III) and six aromatic N donors), tetrahedral d2-d8 is high spin."""
    sites = list(plan.sites)
    d = plan.d_count
    if d is None or d % 2 or d in (0, 10) or _strong_field(mol, sites):
        return None
    z = mol.GetAtomWithIdx(plan.metal).GetAtomicNum()
    sym = Chem.GetPeriodicTable().GetElementSymbol(z)
    what = f"{plan.polyhedron} d{d} {sym}({plan.oxidation})"
    if plan.polyhedron == "octahedral" and d in (2, 4, 8):
        return f"{what} has unpaired electrons in any ligand field"
    if plan.polyhedron == "octahedral" and d == 6 and 21 <= z <= 30:
        if z == 27 and plan.oxidation >= 3 or _aromatic_n(mol, sites) >= 6:
            return None
        return f"{what} with weak-field ligands is high spin (S=2)"
    if plan.polyhedron == "tetrahedral":
        return f"{what} is high spin"
    return None


def preflight_problems(mol: Chem.Mol) -> list[str]:
    """Reasons the metal path cannot give a closed-shell geometry of `mol` (empty: none)."""
    if not metal_atoms(mol):
        return []
    try:
        plans = plan_complex(mol)
    except (MetalTemplateError, ValueError):
        return []  # no template: the standard embedding is used, nothing to reject here
    reasons = [open_shell_reason(mol, p) for p in plans]
    return [
        f"likely open-shell ground state: {r}; the pipeline is closed-shell" for r in reasons if r
    ]


# ---- donor positions


def _radius(mol: Chem.Mol, i: int) -> float:
    atom = mol.GetAtomWithIdx(i)
    z = atom.GetAtomicNum()
    triple = any(b.GetBondType() == Chem.BondType.TRIPLE for b in atom.GetBonds())
    return _SP_CARBON_R if z == 6 and triple else Chem.GetPeriodicTable().GetRcovalent(z)


def ml_distance(mol: Chem.Mol, m: int, donor: int) -> float:
    """Start M-L distance: the covalent radii sum, an sp carbon taken at 0.69 A, less 0.05 A."""
    return _radius(mol, m) + _radius(mol, donor) - _ML_SHRINK


def _perpendicular(u: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, ref)
    return v / np.linalg.norm(v)


def _eta_layout(site: Site, mol: Chem.Mol) -> np.ndarray:
    """In-plane offsets (n, 2) of an eta-n group: a regular polygon for a ring, consecutive
    vertices of a hexagon (recentred) for an open chain such as an alkene or an allyl."""
    n = site.hapto
    first, last = site.atoms[0], site.atoms[-1]
    ring = n >= 3 and mol.GetBondBetweenAtoms(first, last) is not None
    k = n if ring else 6
    radius = _ETA_BOND / (2 * np.sin(np.pi / k))
    ang = 2 * np.pi * np.arange(n) / k
    pts = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    pts -= pts.mean(axis=0)
    if not ring:  # the chain's end-to-end vector along the in-plane axis (x)
        dx, dy = pts[-1] - pts[0]
        c, s = np.array([dx, dy]) / np.hypot(dx, dy)
        pts = pts @ np.array([[c, -s], [s, c]])
    return pts


def _eta_axis(u: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """The in-plane axis of an eta-n ligand on direction `u`: of 12 perpendiculars, the one
    most nearly perpendicular to the other sites (an alkene stands across the square plane,
    as in Zeise's salt)."""
    p = _perpendicular(u)
    q = np.cross(u, p)
    best, best_score = p, None
    for theta in np.linspace(0.0, np.pi, 12, endpoint=False):
        e = np.cos(theta) * p + np.sin(theta) * q
        score = max((abs(float(e @ v)) for v in others), default=0.0)
        if best_score is None or score < best_score - 1e-9:
            best, best_score = e, score
    return best


def site_positions(mol: Chem.Mol, plan: CentrePlan) -> dict[int, np.ndarray]:
    """Template coordinates of the metal (origin) and every site atom."""
    out = {plan.metal: np.zeros(3)}
    verts = POLYHEDRA[plan.polyhedron]
    for site, slot in zip(plan.sites, plan.slots, strict=True):
        u = verts[slot] / np.linalg.norm(verts[slot])
        if site.hapto == 1:
            out[site.atoms[0]] = u * ml_distance(mol, plan.metal, site.atoms[0])
            continue
        offsets = _eta_layout(site, mol)
        d_mc = float(np.mean([ml_distance(mol, plan.metal, a) for a in site.atoms])) + _ML_SHRINK
        r = float(np.sqrt((offsets**2).sum(axis=1)).mean())
        h = float(np.sqrt(max(d_mc**2 - r**2, 1.2**2)))
        others = [verts[k] / np.linalg.norm(verts[k]) for k in plan.slots if k != slot]
        e1 = _eta_axis(u, others)
        e2 = np.cross(u, e1)
        for a, (x, y) in zip(site.atoms, offsets, strict=True):
            out[a] = h * u + x * e1 + y * e2
    return out


# ---- embedding around the fixed donors


def _stripped(mol: Chem.Mol) -> Chem.Mol:
    """`mol` without its metal-ligand bonds (same atom order), metals as isolated atoms."""
    rw = Chem.RWMol(mol)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if is_metal_bond(mol, i, j):
            rw.RemoveBond(i, j)
    for m in metal_atoms(mol):
        rw.GetAtomWithIdx(m).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(out)
    return out


def _bounds(mol: Chem.Mol, stripped: Chem.Mol, fixed: dict[int, np.ndarray]) -> np.ndarray | None:
    """ETKDG bounds of the stripped molecule with every fixed pair set to its template
    distance and the metal's lower bounds capped at `_METAL_LOWER`: the van der Waals floors
    of the isolated metal (3.2-3.9 A) would forbid the M-L distances and the atoms behind the
    donors. The floors between ligands stay, so ligands do not interpenetrate. None when
    smoothing fails."""
    from rdkit import DistanceGeometry
    from rdkit.Chem import rdDistGeom

    with rdBase.BlockLogs():  # the UFF typer behind the 1-2 bounds warns on every metal
        bm = rdDistGeom.GetMoleculeBoundsMatrix(stripped)
    for m in metal_atoms(mol):
        for i in range(mol.GetNumAtoms()):
            if i != m:
                a, b = min(i, m), max(i, m)
                bm[b, a] = min(bm[b, a], _METAL_LOWER)
    for i, j in itertools.combinations(sorted(fixed), 2):
        d = float(np.linalg.norm(fixed[i] - fixed[j]))
        bm[i, j], bm[j, i] = d + 0.01, max(d - 0.01, 0.0)
    return bm if DistanceGeometry.DoTriangleSmoothing(bm) else None


def _embed(mol: Chem.Mol, stripped: Chem.Mol, fixed: dict[int, np.ndarray], seed: int):
    """ETKDGv3 of the stripped molecule as one piece with the template atoms fixed: first on
    the adjusted bounds (`_bounds`), else on RDKit's own with smoothing failures ignored."""
    cmap = {i: Point3D(*(float(c) for c in p)) for i, p in fixed.items()}
    bm = _bounds(mol, stripped, fixed)
    for use_bounds in (True, False):
        if use_bounds and bm is None:
            continue
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.numThreads = 1
        params.embedFragmentsSeparately = False
        params.SetCoordMap(cmap)
        if use_bounds:
            params.SetBoundsMat(bm)
        else:
            params.useRandomCoords = True
            params.ignoreSmoothingFailures = True
        with rdBase.BlockLogs():
            cid = AllChem.EmbedMolecule(stripped, params)
        if cid >= 0:
            return stripped.GetConformer(cid).GetPositions()
    return None


def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rotation, centroid of a, centroid of b) of the proper rotation moving rows `a` onto
    rows `b`: x -> (x - ca) @ rot + cb."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt, ca, cb


def _rigid_fit(raw, members, anchors, metals, fixed) -> np.ndarray | None:
    """The ligand `members` rotated onto its template atoms: with the metal first (ETKDG put
    it at its template distances too, which fixes the side of a chelate), then without it (an
    eta-n ring embedded with the other ring direction is an in-plane flip). None when
    neither fit is within `_MAX_FIT_DEV`."""
    target = np.array([fixed[i] for i in anchors])
    for ref in (anchors + [m for m in metals if m in fixed], anchors):
        rot, ca, cb = _kabsch(raw[ref], np.array([fixed[i] for i in ref]))
        placed = (raw[members] - ca) @ rot + cb
        sel = [members.index(i) for i in anchors]
        if float(np.abs(placed[sel] - target).max()) <= _MAX_FIT_DEV:
            return placed
    return None


def _place_fragments(mol: Chem.Mol, raw: np.ndarray, fixed: dict[int, np.ndarray]):
    """Each ligand moved rigidly onto its template atoms (a monodentate ligand translated so
    its donor sits on its vertex, a chelate or eta-n ligand rotated onto its donors), the
    template atoms then set exactly; the metal at its template position. Rigid moves keep
    every ligand's internal geometry from ETKDG. None when a ligand misses its template by
    more than `_MAX_FIT_DEV` (e.g. a tridentate embedded as the mirror image)."""
    frag = _fragment_of(mol)
    out = raw.copy()
    for f in sorted({frag[i] for i in fixed if frag[i] >= 0}):
        members = [i for i, g in enumerate(frag) if g == f]
        anchors = [i for i in members if i in fixed]
        target = np.array([fixed[i] for i in anchors])
        if len(anchors) == 1:
            out[members] = raw[members] + (target[0] - raw[anchors[0]])
        else:
            placed = _rigid_fit(raw, members, anchors, metal_atoms(mol), fixed)
            if placed is None:
                return None
            out[members] = placed
        out[anchors] = target
    for m in metal_atoms(mol):
        out[m] = fixed.get(m, out[m])
    return out


def _rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation matrix taking unit vector `a` onto unit vector `b`."""
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:
        p = _perpendicular(a)
        return 2.0 * np.outer(p, p) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def _point_monodentate(mol, xyz, metal, donor, members, others) -> np.ndarray:
    """Turn one monodentate ligand about its donor so that the donor's lone-pair direction
    (away from its ligand neighbours) points at the metal, then spin it about the M-L axis to
    the angle (of 12) with the largest closest contact to the rest of the complex."""
    centre = xyz[donor]
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors() if n.GetIdx() in members]
    target = xyz[metal] - centre
    target /= np.linalg.norm(target)
    vecs = [(xyz[n] - centre) / max(np.linalg.norm(xyz[n] - centre), 1e-9) for n in nbrs]
    lone = -np.sum(vecs, axis=0) if vecs else target
    lone = target if np.linalg.norm(lone) < 1e-3 else lone / np.linalg.norm(lone)
    idx = sorted(members)
    rel = (xyz[idx] - centre) @ _rotation(lone, target).T
    moving = [k for k, i in enumerate(idx) if i != donor]
    best, best_gap = rel, -1.0
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        trial = rel @ _axis_rotation(target, float(theta)).T
        diff = centre + trial[moving][:, None] - xyz[others][None]
        gap = float(np.min(np.linalg.norm(diff, axis=2))) if moving else 0.0
        if gap > best_gap + 1e-6:
            best, best_gap = trial, gap
    out = xyz.copy()
    out[idx] = centre + best
    return out


def _orient_monodentates(mol: Chem.Mol, plans: list[CentrePlan], xyz: np.ndarray) -> np.ndarray:
    frag = _fragment_of(mol)
    for plan in plans:
        site_frags = [frag[s.atoms[0]] for s in plan.sites]
        for site, f in zip(plan.sites, site_frags, strict=True):
            members = {i for i, g in enumerate(frag) if g == f}
            if site.hapto > 1 or site_frags.count(f) > 1 or len(members) == 1:
                continue
            others = [i for i in range(len(frag)) if i not in members]
            xyz = _point_monodentate(mol, xyz, plan.metal, site.atoms[0], members, others)
    return xyz


def _branch(mol: Chem.Mol, root: int, blocked: set[int]) -> list[int]:
    """Atoms reachable from `root` without passing through `blocked`."""
    seen, stack = {root}, [root]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            i = nb.GetIdx()
            if i not in seen and i not in blocked:
                seen.add(i)
                stack.append(i)
    return sorted(seen)


def _place_eta_substituents(mol: Chem.Mol, plan: CentrePlan, xyz: np.ndarray) -> np.ndarray:
    """Put the substituents of every eta-n atom in the ligand plane, pointing out of the ring
    (one substituent: radially; two: at +-60 degrees), tilted slightly away from the metal.
    ETKDG, which never saw the metal, may bend a ring C-H towards it."""
    out = xyz.copy()
    for site in plan.sites:
        if site.hapto < 2:
            continue
        inside = set(site.atoms) | {plan.metal}
        centre = out[list(site.atoms)].mean(axis=0)
        away = centre - out[plan.metal]
        away /= np.linalg.norm(away)
        for a in site.atoms:
            subs = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()]
            subs = [i for i in subs if i not in inside]
            radial = out[a] - centre
            radial -= (radial @ away) * away
            if not subs or len(subs) > 2 or np.linalg.norm(radial) < 0.1:
                continue
            radial /= np.linalg.norm(radial)
            turns = [0.0] if len(subs) == 1 else [np.pi / 3, -np.pi / 3]
            for sub, turn in zip(subs, turns, strict=True):
                want = _axis_rotation(away, turn) @ radial + 0.1 * away
                want /= np.linalg.norm(want)
                have = (out[sub] - out[a]) / np.linalg.norm(out[sub] - out[a])
                idx = _branch(mol, sub, inside | {a})
                out[idx] = out[a] + (out[idx] - out[a]) @ _rotation(have, want).T
    return out


def closest_contact(mol: Chem.Mol, xyz: np.ndarray) -> float:
    """The closest approach of two atoms that are neither bonded nor share a neighbour (with
    the metal-ligand bonds counted): a start geometry with a value below ~1 A has a clash."""
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    topo = Chem.GetDistanceMatrix(mol)
    far = topo > 2
    return float(dist[far].min()) if far.any() else float("inf")


def contact_ratio(mol: Chem.Mol, xyz: np.ndarray) -> float:
    """The smallest d / (r_cov,i + r_cov,j) over atom pairs more than two bonds apart (more
    than three for the metal, whose 1-4 partners across a chelate ring sit at ~3 A): about 2
    for ordinary contacts, below `_GOOD_CONTACT` a clash (a C...O at 1.6 A is 1.13, an H...H
    at 1.8 A is 2.9)."""
    r = np.array([Chem.GetPeriodicTable().GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
    topo = Chem.GetDistanceMatrix(mol)
    far = topo > 2
    metals = metal_atoms(mol)
    far[metals] &= topo[metals] > 3
    far[:, metals] &= topo[:, metals] > 3
    if not far.any():
        return float("inf")
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    return float((dist / (r[:, None] + r[None]))[far].min())


def build_start_geometry(
    mol: Chem.Mol, overrides: Mapping[int, str] | None = None, seed: int = 0
) -> tuple[np.ndarray, list[CentrePlan]]:
    """Start coordinates (n, 3) in Angstrom for `mol` (explicit H, `prepare_mol` order) and
    the centre plans. Raises MetalTemplateError when no template applies or ETKDG fails."""
    plans = plan_complex(mol, overrides)
    fixed: dict[int, np.ndarray] = {}
    for plan in plans:
        fixed |= site_positions(mol, plan)
    stripped = _stripped(mol)
    best, best_gap = None, -1.0
    for k in range(_N_SEEDS):
        raw = _embed(mol, stripped, fixed, seed + k)
        if raw is None:
            continue
        fitted = _place_fragments(mol, raw, fixed)
        if fitted is None:
            continue
        xyz = _orient_monodentates(mol, plans, fitted)
        for plan in plans:
            xyz = _place_eta_substituents(mol, plan, xyz)
        gap = contact_ratio(mol, xyz)
        if gap > best_gap:
            best, best_gap = xyz, gap
        if gap >= _GOOD_CONTACT:
            break
    if best is None:
        raise MetalTemplateError("ETKDG found no ligand embedding around the fixed donors")
    return best, plans


def set_conformer(mol: Chem.Mol, xyz: np.ndarray) -> int:
    """Replace the conformers of `mol` by one (id 0) with coordinates `xyz`."""
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetId(0)
    for i, p in enumerate(np.asarray(xyz, dtype=float)):
        conf.SetAtomPosition(i, Point3D(*p.tolist()))
    mol.AddConformer(conf, assignId=False)
    return 0


# ---- analysis of a geometry


def site_vectors(mol: Chem.Mol, metal: int, sites: tuple[Site, ...], xyz: np.ndarray) -> np.ndarray:
    """Unit vectors from the metal to each site (an eta-n site: to its centroid)."""
    vecs = np.array([xyz[list(s.atoms)].mean(axis=0) - xyz[metal] for s in sites])
    return vecs / np.linalg.norm(vecs, axis=1)[:, None]


def _angles(vecs: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(vecs @ vecs.T, -1.0, 1.0)))


def assignment_fit(vecs: np.ndarray, name: str) -> tuple[float, tuple[int, ...]]:
    """(RMS deviation of the L-M-L angles in degrees, vertex of each site) of the best match
    of the site vectors to polyhedron `name`."""
    n = len(vecs)
    got, ideal = _angles(vecs), _angles(POLYHEDRA[name])
    iu = np.triu_indices(n, 1)
    return min(
        (float(np.sqrt(np.mean((got[iu] - ideal[np.ix_(p, p)][iu]) ** 2))), p)
        for p in itertools.permutations(range(n))
    )


def fit_polyhedra(vecs: np.ndarray) -> list[tuple[str, float]]:
    """(polyhedron, angle RMS in degrees) for every ideal polyhedron of this coordination
    number, best first."""
    out = [(name, assignment_fit(vecs, name)[0]) for name in BY_CN.get(len(vecs), ())]
    return sorted(out, key=lambda t: t[1])


def fitted_trans_pairs(vecs: np.ndarray, name: str) -> list[tuple[int, int]]:
    """The trans pairs of the best vertex assignment to `name`: robust to distortion (the
    Cl-Ru-Cl of a Grubbs-type complex bends to ~147 degrees and is still trans)."""
    return trans_pairs(POLYHEDRA[name][list(assignment_fit(vecs, name)[1])])


def trans_pairs(vecs: np.ndarray) -> list[tuple[int, int]]:
    ang = _angles(vecs)
    return [(i, j) for i, j in itertools.combinations(range(len(vecs)), 2) if ang[i, j] > TRANS_DEG]


def plan_trans_pairs(plan: CentrePlan) -> list[tuple[int, int]]:
    verts = POLYHEDRA[plan.polyhedron][list(plan.slots)]
    return trans_pairs(verts / np.linalg.norm(verts, axis=1)[:, None])


def check_geometry(mol: Chem.Mol, plan: CentrePlan, xyz: np.ndarray) -> list[str]:
    """Warnings for a final geometry against its plan: METAL_POLYHEDRON_CHANGED (another
    polyhedron fits better) and METAL_ISOMER_CHANGED (the best vertex assignment has other
    trans pairs)."""
    vecs = site_vectors(mol, plan.metal, plan.sites, xyz)
    out = []
    fits = fit_polyhedra(vecs)
    if fits and fits[0][0] != plan.polyhedron:
        out.append("METAL_POLYHEDRON_CHANGED")
    elif fitted_trans_pairs(vecs, plan.polyhedron) != plan_trans_pairs(plan):
        out.append("METAL_ISOMER_CHANGED")
    return out
