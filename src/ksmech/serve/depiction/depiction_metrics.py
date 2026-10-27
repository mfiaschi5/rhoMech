"""Layout defects of a 2D depiction (drawing units, a bond is `L` = 1.5), one score per layout.

Counted, with the bond length L:
- `close_contacts`: pairs of atoms not bonded to each other closer than 0.5 L, and among them
  `coincident`: closer than 0.25 L (one atom hides the other);
- `label_overlaps`: a labelled atom (not a plain carbon) closer than 0.8 L to an atom it is
  not bonded to (its symbol box would touch that atom or its label);
- `atom_on_bond`: an atom within 0.2 L of a bond it is not on (strictly inside the segment);
- `crossings`: bonds that cross without sharing an atom;
- `small_angles`: two bonds at one atom under 35 degrees apart;
- `inside_ring`: an atom in no ring drawn inside the polygon of a ring (SSSR), e.g. the methyls
  of a bridge drawn inside the ring it bridges;
- `squashed_rings`: rings of up to 8 atoms whose drawn shape is under 0.6 as wide as it is long
  (the ratio of the principal spreads of its atoms: a regular polygon has 1, a flat ring seen
  53 degrees from face-on 0.6);
- `bond_length_rms`: RMS of (length / L - 1) over the bonds.

`score` = 10 label overlaps + 8 contacts + 15 coincident + 6 atom-on-bond + 10 inside-ring
+ w_x crossings + w_s squashed rings + 2 small angles + 20 bond RMS, by the kind of drawing:
flat (a textbook layout: w_x 3, w_s 0, as only a bridged ring is drawn narrow there and that is
the convention), template (a perspective template, whose crossings are meant and drawn broken
and whose rings are drawn at a slant on purpose: w_x 1, w_s 0) or view (the 3D view used as
the template, where a slanted aromatic ring is just squashed: w_x 1.5, w_s 6). The weights
were set by comparing the gallery's drawings (`ksmech-serve gallery`): a drawing with a lower
score never looked worse there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem

L = 1.5
CONTACT = 0.5 * L
COINCIDENT = 0.25 * L
SQUASHED = 0.6
LABEL = 0.8 * L
ON_BOND = 0.2 * L
SMALL_ANGLE_DEG = 35.0


WEIGHTS = {"flat": (3.0, 0.0), "template": (1.0, 0.0), "view": (1.5, 6.0)}  # (w_x, w_s)


@dataclass(frozen=True)
class Defects:
    close_contacts: int
    coincident: int
    label_overlaps: int
    atom_on_bond: int
    crossings: int
    small_angles: int
    inside_ring: int
    squashed_rings: int
    bond_length_rms: float
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


def _labelled(a: Chem.Atom) -> bool:
    return a.GetAtomicNum() != 6 or a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() > 0


def _pairs(mol: Chem.Mol, xy: np.ndarray) -> tuple[int, int, int]:
    bonded = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
    bonded |= {(j, i) for i, j in bonded}
    lab = [_labelled(a) for a in mol.GetAtoms()]
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    contacts = labels = coincident = 0
    n = len(xy)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            if d[i, j] < CONTACT:
                contacts += 1
                coincident += int(d[i, j] < COINCIDENT)
            elif (lab[i] or lab[j]) and d[i, j] < LABEL:
                labels += 1
    return contacts, labels, coincident


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ab = b - a
    t = float(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12))
    return float(np.linalg.norm(p - (a + np.clip(t, 0, 1) * ab))), t


def _on_bond(mol: Chem.Mol, xy: np.ndarray) -> int:
    count = 0
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for k in range(len(xy)):
            if k in (i, j):
                continue
            d, t = _segment_distance(xy[k], xy[i], xy[j])
            if d < ON_BOND and 0.05 < t < 0.95:
                count += 1
    return count


def _cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def crossing_pairs(mol: Chem.Mol, xy: np.ndarray) -> list[tuple[int, int]]:
    """Bond index pairs that cross (no shared atom)."""
    bonds = [(b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    out = []
    for x in range(len(bonds)):
        for y in range(x + 1, len(bonds)):
            k1, a, b = bonds[x]
            k2, c, d = bonds[y]
            if {a, b} & {c, d}:
                continue
            if _cross(xy[a], xy[b], xy[c], xy[d]):
                out.append((k1, k2))
    return out


def _small_angles(mol: Chem.Mol, xy: np.ndarray) -> int:
    count = 0
    lim = np.cos(np.deg2rad(SMALL_ANGLE_DEG))
    for a in mol.GetAtoms():
        vs = [xy[n.GetIdx()] - xy[a.GetIdx()] for n in a.GetNeighbors()]
        vs = [v / (np.linalg.norm(v) or 1.0) for v in vs]
        for x in range(len(vs)):
            for y in range(x + 1, len(vs)):
                if float(vs[x] @ vs[y]) > lim:
                    count += 1
    return count


def _bond_rms(mol: Chem.Mol, xy: np.ndarray) -> float:
    ls = [np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    return float(np.sqrt(np.mean((np.array(ls) / L - 1.0) ** 2))) if ls else 0.0


def _inside(p: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd rule: is point p strictly inside the polygon?"""
    inside = False
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x > p[0]:
                inside = not inside
    return inside


def _inside_ring(mol: Chem.Mol, xy: np.ndarray) -> int:
    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    in_ring = {a for r in rings for a in r}
    loose = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in in_ring]
    count = 0
    for ring in rings:
        poly = xy[list(ring)]
        count += sum(_inside(xy[a], poly) for a in loose)
    return count


def _squashed(mol: Chem.Mol, xy: np.ndarray) -> int:
    count = 0
    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        n = len(ring)
        if n > 8:
            continue
        p = xy[ring] - xy[ring].mean(axis=0)
        ev = np.linalg.eigvalsh(p.T @ p / n)
        count += int(np.sqrt(max(ev[0], 0.0) / max(ev[1], 1e-12)) < SQUASHED)
    return count


def defects(mol: Chem.Mol, xy: np.ndarray, kind: str = "flat") -> Defects:
    """The defects of `xy` (atoms in mol order, scaled so a bond is about L); `kind` picks the
    crossing and squashed-ring weights (module docstring)."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    contacts, labels, coincident = _pairs(mol, xy)
    on_bond = _on_bond(mol, xy)
    crossings = len(crossing_pairs(mol, xy))
    small = _small_angles(mol, xy)
    inside = _inside_ring(mol, xy)
    squashed = _squashed(mol, xy)
    rms = _bond_rms(mol, xy)
    w_cross, w_squash = WEIGHTS[kind]
    score = 10 * labels + 8 * contacts + 15 * coincident + 6 * on_bond + 10 * inside
    score += w_cross * crossings + w_squash * squashed + 2 * small + 20 * rms
    return Defects(
        close_contacts=contacts,
        coincident=coincident,
        label_overlaps=labels,
        atom_on_bond=on_bond,
        crossings=crossings,
        small_angles=small,
        inside_ring=inside,
        squashed_rings=squashed,
        bond_length_rms=round(rms, 4),
        score=round(score, 3),
    )


# ---- drawing rules of the perspective templates (checked by the tests and the gallery)

RULE_LENGTH = 0.15  # every bond within 15% of the median, except bonds drawn short on purpose
RULE_SHORT_MIN = 0.55  # a deliberately foreshortened bond is still at least this long (x L)
RULE_ANGLE_MIN = 30.0  # degrees between two bonds at one atom
RULE_ANGLE_MAX = 165.0  # straighter reads as one line (RDKit even prints a "C" there)
RULE_CLEARANCE = 0.4  # an atom at least this far (x L) from any bond it is not on
RULE_SEPARATION = 0.6  # two atoms at least this far apart (x L)


@dataclass(frozen=True)
class RuleReport:
    length: tuple[float, float]  # min, max of length / median over the regular bonds
    short: float  # the shortest foreshortened bond / median (1.0 without any)
    angle: tuple[float, float]  # min, max angle (deg) between two bonds at one atom
    clearance: float  # min atom-to-bond distance / L (bonds the atom is not on)
    separation: float  # min atom-atom distance / L
    crossings: int

    def violations(self) -> list[str]:
        out = []
        if self.length[0] < 1 - RULE_LENGTH or self.length[1] > 1 + RULE_LENGTH:
            out.append(f"bond lengths {self.length[0]:.2f}..{self.length[1]:.2f}")
        if self.short < RULE_SHORT_MIN:
            out.append(f"foreshortened bond {self.short:.2f}")
        if self.angle[0] < RULE_ANGLE_MIN or self.angle[1] > RULE_ANGLE_MAX:
            out.append(f"angles {self.angle[0]:.0f}..{self.angle[1]:.0f}")
        if self.clearance < RULE_CLEARANCE:
            out.append(f"atom-bond clearance {self.clearance:.2f}")
        if self.separation < RULE_SEPARATION:
            out.append(f"atom separation {self.separation:.2f}")
        return out


def _angle_range(xy: np.ndarray, nbrs: list[list[int]]) -> tuple[float, float]:
    """(smallest angle at any atom, straightest angle at an atom with two or three bonds; a
    four-bond atom drawn as a cross is the convention)."""
    lo, hi = 180.0, 0.0
    for k, ns in enumerate(nbrs):
        us = [(xy[n] - xy[k]) / (np.linalg.norm(xy[n] - xy[k]) or 1.0) for n in ns]
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a = float(np.degrees(np.arccos(np.clip(us[i] @ us[j], -1.0, 1.0))))
                lo = min(lo, a)
                hi = max(hi, a) if len(us) <= 3 else hi
    return (lo, hi) if hi > 0 else (lo if lo < 180 else 90.0, 90.0)


def rule_report(xy, bonds, short=frozenset()) -> RuleReport:
    """The template drawing rules on points `xy` (drawing units) joined by `bonds` (index
    pairs); `short` holds the sorted pairs drawn short on purpose."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    bonds = [tuple(int(v) for v in b) for b in bonds]
    nbrs: list[list[int]] = [[] for _ in xy]
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    lens = {b: float(np.linalg.norm(xy[b[0]] - xy[b[1]])) for b in bonds}
    reg = [v for b, v in lens.items() if tuple(sorted(b)) not in short]
    med = float(np.median(reg)) if reg else L
    sh = [v / med for b, v in lens.items() if tuple(sorted(b)) in short]
    clear = min(
        (
            _segment_distance(xy[k], xy[a], xy[b])[0]
            for a, b in bonds
            for k in range(len(xy))
            if k not in (a, b)
        ),
        default=9 * L,
    )
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 9 * L
    xs = sum(
        _cross(xy[a], xy[b], xy[c], xy[e])
        for i, (a, b) in enumerate(bonds)
        for c, e in bonds[i + 1 :]
        if not {a, b} & {c, e}
    )
    return RuleReport(
        length=(round(min(reg) / med, 3), round(max(reg) / med, 3)) if reg else (1.0, 1.0),
        short=round(min(sh), 3) if sh else 1.0,
        angle=tuple(round(v, 1) for v in _angle_range(xy, nbrs)),
        clearance=round(clear / L, 3),
        separation=round(float(d.min()) / L, 3),
        crossings=int(xs),
    )


def mol_rule_report(mol: Chem.Mol, xy, short=frozenset()) -> RuleReport:
    """rule_report of a laid-out molecule (atom order of mol)."""
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return rule_report(xy, bonds, short)
