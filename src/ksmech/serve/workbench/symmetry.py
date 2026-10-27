"""Schoenflies point group of a 3D structure (atomic numbers + coordinates in Angstrom), plain
numpy plus scipy's kd-tree and assignment solver. Display only (the workbench).

Tolerance semantics: an operation is a symmetry of the structure when it maps every atom to
within `tolerance` Angstrom of a distinct atom of the same element (a one-to-one matching).
Each operation is the best orthogonal fit (Kabsch, determinant fixed) to the atom permutation
it induces, so its deviation is the largest residual of that fit, and every operation of the
reported group, products included, passes the test.

The algorithm:
1. Centre on the centre of mass; classify the inertia tensor (atom, linear, spherical,
   symmetric or asymmetric top). A linear structure (every atom within the tolerance of the
   axis of least inertia) is C∞v or D∞h by an inversion test.
2. Candidate directions: the principal axes, the principal axes and centroids of each set of
   symmetry-equivalent-atom candidates (same element, same distance from the centre within the
   tolerance), the atom directions, and per pair of such atoms the midpoint, the difference and
   the normal of the plane through the centre (sets above SMALL_SET atoms use only the pairs
   with their first ANCHORS atoms; the closure below recovers the rest).
3. For each direction: C_n (n = 2..12, bounded by the largest set), the mirror with that normal
   and S_2n; plus the inversion. A kd-tree on (x, y, z, 1000 Z) rejects most candidates on a few
   probe atoms; the survivors are matched one-to-one (Hungarian per element when the nearest
   neighbours collide), refitted and kept when within the tolerance.
4. The group: greedy closure of the kept operations, best deviation first, from a few seeds; a
   generator whose closure has an operation outside the tolerance (or more than MAX_ORDER) is
   skipped. Elements are keyed by (determinant, permutation), unique for a non-linear structure.
5. The decision tree on the group: several C_n with n >= 3 give T, Td, Th, O, Oh, I, Ih; no axis
   gives C1, Cs, Ci; otherwise the principal axis C_n with n perpendicular C2 gives Dn, Dnh, Dnd,
   else Cn, Cnh, Cnv, S2n. The group order is checked against the name.
6. The symmetrized structure: the group average of every atom's images, refitted three times;
   its RMS and largest displacement from the input are reported, with the equivalent-atom
   classes (orbits) and each element's atoms (within the tolerance of the axis, plane or centre).
A hint repeats the search at the looser tolerances of HINT_TOLERANCES and reports the highest
group found ("would be Td at 0.3 Å").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DEFAULT_TOLERANCE = 0.1
MIN_TOLERANCE = 0.01
MAX_TOLERANCE = 0.5
MAX_ATOMS = 1000
HINT_MAX_ATOMS = 300
HINT_TOLERANCES = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
MAX_ORDER = 240  # Ih has 120; D12h 48
MAX_AXIS_ORDER = 12
SMALL_SET = 16
ANCHORS = 3
N_PROBES = 8
N_SEEDS = 6
DIRECTION_DEDUPE_COS = math.cos(math.radians(2.0))
AXIS_MERGE_COS = math.cos(math.radians(4.0))
PERP_SIN = math.sin(math.radians(5.0))
_Z_SCALE = 1000.0

EXPECTED_ORDER = {"T": 12, "Td": 24, "Th": 24, "O": 24, "Oh": 48, "I": 60, "Ih": 120}


class SymmetryError(ValueError):
    """`code` is a serve reason code (REQUEST_INVALID, LIMIT_EXCEEDED)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Op:
    """R x_i ~ x_perm[i] for every atom i (centred frame)."""

    R: np.ndarray
    perm: np.ndarray
    det: int
    dev: float

    @property
    def key(self) -> tuple:
        return (self.det, self.perm.tobytes())


@dataclass
class Frame:
    z: np.ndarray
    x: np.ndarray  # centred coordinates
    com: np.ndarray
    masses: np.ndarray
    tol: float
    tree: cKDTree


# ---- geometry helpers


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _masses(z: np.ndarray) -> np.ndarray:
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    return np.array([table.GetAtomicWeight(int(zi)) or 1.0 for zi in z], dtype=float)


def make_frame(z, xyz, tol: float) -> Frame:
    z = np.asarray(z, dtype=int).reshape(-1)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    m = _masses(z)
    com = (m[:, None] * xyz).sum(0) / m.sum()
    x = xyz - com
    tree = cKDTree(np.c_[x, z * _Z_SCALE])
    return Frame(z=z, x=x, com=com, masses=m, tol=float(tol), tree=tree)


def inertia(f: Frame) -> tuple[np.ndarray, np.ndarray]:
    """(moments ascending, amu Å², axes as columns)."""
    x, m = f.x, f.masses
    r2 = (x * x).sum(1)
    t = np.eye(3) * (m * r2).sum() - (m[:, None, None] * x[:, :, None] * x[:, None, :]).sum(0)
    w, v = np.linalg.eigh(t)
    return w, v


def top_kind(f: Frame, moments: np.ndarray) -> str:
    """Rotor class; two moments are equal within 2 tol sqrt(M I) (one atom moved by tol)."""
    if len(f.z) == 1:
        return "atom"
    scale = 2.0 * f.tol * math.sqrt(f.masses.sum() * max(moments[-1], 1e-12))
    if moments[0] <= scale:
        return "linear"
    eq01 = moments[1] - moments[0] <= scale
    eq12 = moments[2] - moments[1] <= scale
    if eq01 and eq12:
        return "spherical"
    return "symmetric" if eq01 or eq12 else "asymmetric"


def sea_sets(f: Frame) -> list[np.ndarray]:
    """Candidate symmetry-equivalent sets: same element, |r| chained within the tolerance."""
    r = np.linalg.norm(f.x, axis=1)
    out = []
    for zi in np.unique(f.z):
        idx = np.flatnonzero(f.z == zi)
        idx = idx[np.argsort(r[idx])]
        start = 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or r[idx[k]] - r[idx[k - 1]] > f.tol:
                out.append(idx[start:k])
                start = k
    return out


# ---- candidate directions and operations


def _pairs(idx: np.ndarray) -> list[tuple[int, int]]:
    ids = [int(i) for i in idx]
    n_anchor = len(ids) if len(ids) <= SMALL_SET else ANCHORS
    return [(ids[i], ids[j]) for i in range(n_anchor) for j in range(i + 1, len(ids))]


def _set_directions(f: Frame, idx: np.ndarray) -> list[np.ndarray]:
    x, tol = f.x[idx], f.tol
    out: list[np.ndarray] = []
    c = x.mean(0)
    if np.linalg.norm(c) > tol:
        out.append(c)
    if len(idx) >= 3:
        out.extend(np.linalg.eigh((x - c).T @ (x - c))[1].T)
    pairs = np.asarray(_pairs(idx), dtype=int).reshape(-1, 2)
    ra, rb = f.x[pairs[:, 0]], f.x[pairs[:, 1]]
    mid, diff, cross = (ra + rb) / 2, ra - rb, np.cross(ra, rb)
    norm = np.linalg.norm
    out.extend(mid[norm(mid, axis=1) > tol / 2])
    out.extend(diff[norm(diff, axis=1) > tol])
    out.extend(cross[norm(cross, axis=1) > 0.1 * norm(ra, axis=1) * norm(rb, axis=1)])
    return out


def dedupe_directions(vs: list[np.ndarray] | np.ndarray, cos_max: float) -> np.ndarray:
    """Unit vectors, sign-insensitive, none within acos(cos_max) of an earlier one."""
    v = np.asarray(vs, dtype=float).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1)
    v = v[n > 1e-9] / n[n > 1e-9, None]
    kept = np.empty_like(v)
    nk = 0
    for u in v:
        if nk and np.max(np.abs(kept[:nk] @ u)) > cos_max:
            continue
        kept[nk] = u
        nk += 1
    return kept[:nk]


def candidate_directions(f: Frame, sets: list[np.ndarray], axes: np.ndarray) -> np.ndarray:
    vs: list[np.ndarray] = list(axes.T)
    r = np.linalg.norm(f.x, axis=1)
    vs.extend(f.x[r > f.tol])
    for idx in sets:
        if len(idx) >= 2:
            vs.extend(_set_directions(f, idx))
    return dedupe_directions(vs, DIRECTION_DEDUPE_COS)


def _rotations(dirs: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Rodrigues rotations, (directions, angles, 3, 3)."""
    u = dirs
    k = np.zeros((len(u), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -u[:, 2], u[:, 1], -u[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = u[:, 2], -u[:, 1], u[:, 0]
    kk = k @ k
    s, c = np.sin(thetas)[None, :, None, None], np.cos(thetas)[None, :, None, None]
    return np.eye(3) + s * k[:, None] + (1 - c) * kk[:, None]


def candidate_matrices(dirs: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """(matrices, determinants): the inversion, then per direction the mirror, C_n and S_2n
    (n = 2..nmax)."""
    ns = np.arange(2, nmax + 1)
    sigma = np.eye(3) - 2.0 * dirs[:, :, None] * dirs[:, None, :]
    c_n = _rotations(dirs, 2 * np.pi / ns)
    s_2n = sigma[:, None] @ _rotations(dirs, np.pi / ns)
    mats = np.concatenate([-np.eye(3)[None], sigma, c_n.reshape(-1, 3, 3), s_2n.reshape(-1, 3, 3)])
    dets = np.r_[
        -1,
        -np.ones(len(dirs)),
        np.ones(c_n.shape[0] * c_n.shape[1]),
        -np.ones(s_2n.shape[0] * s_2n.shape[1]),
    ]
    return mats, dets.astype(int)


def _within(f: Frame, y: np.ndarray, zs: np.ndarray, bound: float) -> np.ndarray:
    """Distance from each point y (with element zs) to the nearest same-element atom."""
    d, _ = f.tree.query(np.c_[y, zs * _Z_SCALE], distance_upper_bound=bound)
    return d


def _probe_filter(f: Frame, mats: np.ndarray, bound: float) -> np.ndarray:
    r = np.linalg.norm(f.x, axis=1)
    probes = np.argsort(-r)[: min(N_PROBES, len(r))]
    keep = np.ones(len(mats), dtype=bool)
    for start in range(0, len(mats), 512):
        m = mats[start : start + 512]
        y = np.einsum("bij,kj->bki", m, f.x[probes]).reshape(-1, 3)
        d = _within(f, y, np.tile(f.z[probes], len(m)), bound).reshape(len(m), -1)
        keep[start : start + len(m)] = np.isfinite(d).all(1)
    return keep


def match(f: Frame, y: np.ndarray, bound: float) -> np.ndarray | None:
    """The one-to-one same-element permutation with y_i ~ x_perm[i], every pair within
    `bound`; None when there is none (nearest neighbours first, Hungarian on a collision)."""
    d, j = f.tree.query(np.c_[y, f.z * _Z_SCALE], distance_upper_bound=bound)
    if not np.isfinite(d).all():
        return None
    if len(np.unique(j)) == len(j):
        return j.astype(int)
    perm = np.empty(len(y), dtype=int)
    for zi in np.unique(f.z):
        idx = np.flatnonzero(f.z == zi)
        cost = ((y[idx][:, None, :] - f.x[idx][None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(cost)
        if np.sqrt(cost[rows, cols].max()) > bound:
            return None
        perm[idx[rows]] = idx[cols]
    return perm


def fit(f: Frame | np.ndarray, perm: np.ndarray, det: int) -> tuple[np.ndarray, float]:
    """The orthogonal R with det(R) = det minimizing sum |R x_i - x_perm[i]|^2 over the frame's
    (or the given) coordinates, and the largest residual."""
    x = f.x if isinstance(f, Frame) else f
    h = x.T @ x[perm]
    u, _, vt = np.linalg.svd(h)
    d = np.diag([1.0, 1.0, 1.0])
    if np.linalg.det(vt.T @ u.T) * det < 0:
        d[2, 2] = -1.0
    r = vt.T @ d @ u.T
    dev = float(np.linalg.norm(x @ r.T - x[perm], axis=1).max()) if len(perm) else 0.0
    return r, dev


def identity_op(n: int) -> Op:
    return Op(R=np.eye(3), perm=np.arange(n), det=1, dev=0.0)


def find_operations(f: Frame, dirs: np.ndarray, nmax: int) -> list[Op]:
    """Every candidate operation within the tolerance after its refit, best first, one per
    (determinant, permutation)."""
    mats, dets = candidate_matrices(dirs, nmax)
    bound = 2.0 * f.tol  # a candidate direction is only approximate before the refit
    keep = _probe_filter(f, mats, bound)
    found: dict[tuple, Op] = {}
    for k in np.flatnonzero(keep):
        mat, det = mats[k], int(dets[k])
        perm = match(f, f.x @ mat.T, bound)
        if perm is None:
            continue
        r, dev = fit(f, perm, det)
        if dev > f.tol:
            continue
        op = Op(R=r, perm=perm, det=det, dev=dev)
        if op.key not in found or found[op.key].dev > dev:
            found[op.key] = op
    return sorted(found.values(), key=lambda o: o.dev)


# ---- the group


def _product(f: Frame, g: Op, a: Op) -> Op | None:
    """g after a, refitted; None when outside the tolerance."""
    perm = g.perm[a.perm]
    det = g.det * a.det
    r, dev = fit(f, perm, det)
    return None if dev > f.tol else Op(R=r, perm=perm, det=det, dev=dev)


def closure(f: Frame, group: dict[tuple, Op], gens: list[Op]) -> dict[tuple, Op] | None:
    """The group generated by `group` and `gens`; None when a product leaves the tolerance or
    the order passes MAX_ORDER."""
    elems = dict(group)
    frontier = list(elems.values())
    while frontier:
        new: list[Op] = []
        for a in frontier:
            for g in gens:
                key = (g.det * a.det, g.perm[a.perm].tobytes())
                if key in elems:
                    continue
                p = _product(f, g, a)
                if p is None or len(elems) >= MAX_ORDER:
                    return None
                elems[key] = p
                new.append(p)
        frontier = new
    return elems


def greedy_group(f: Frame, ops: list[Op], seed: Op | None) -> dict[tuple, Op]:
    e = identity_op(len(f.z))
    group: dict[tuple, Op] = {e.key: e}
    gens: list[Op] = []
    for op in ([seed] if seed is not None else []) + ops:
        if op.key in group:
            continue
        trial = closure(f, group, [*gens, op])
        if trial is not None:
            group, gens = trial, [*gens, op]
    return group


def best_group(f: Frame, ops: list[Op]) -> dict[tuple, Op]:
    """The largest group among greedy closures seeded by each of the first N_SEEDS
    operations (ties: the smaller largest deviation)."""
    best: dict[tuple, Op] = {}
    best_dev = math.inf
    for seed in [None, *ops[:N_SEEDS]]:
        g = greedy_group(f, ops, seed)
        dev = max(o.dev for o in g.values())
        if len(g) > len(best) or (len(g) == len(best) and dev < best_dev):
            best, best_dev = g, dev
        if all(o.key in best for o in ops):
            break
    return best


# ---- symmetry elements


@dataclass
class Element:
    kind: str  # "C" (proper axis), "S" (improper axis), "sigma" (mirror), "i" (inversion)
    order: int | None
    vector: np.ndarray | None  # axis or plane normal (unit), centred frame
    ops: list[Op]
    label: str = ""
    principal: bool = False
    atoms: tuple[int, ...] = ()


def op_angle(op: Op) -> float:
    """Rotation angle of a proper op; for an improper one, phi of sigma_h C(phi)."""
    tr = float(np.trace(op.R))
    c = (tr - 1.0) / 2.0 if op.det == 1 else (tr + 1.0) / 2.0
    return math.acos(min(1.0, max(-1.0, c)))


def op_axis(op: Op) -> np.ndarray:
    w, v = np.linalg.eig(op.R)
    k = int(np.argmin(np.abs(w - float(op.det))))
    return _unit(np.real(v[:, k]))


def element_order(theta: float) -> int:
    """Smallest k with k theta a multiple of 2 pi (within 0.03 turns)."""
    turns = theta / (2 * math.pi)
    for k in range(1, 2 * MAX_AXIS_ORDER + 1):
        if abs(k * turns - round(k * turns)) < 0.03:
            return k
    return 0


def _merge(elems: list[Element], kind: str, u: np.ndarray, n: int | None, op: Op) -> None:
    for e in elems:
        if e.kind == kind and abs(float(e.vector @ u)) > AXIS_MERGE_COS:
            e.ops.append(op)
            if n is not None and (e.order or 0) < n:
                e.order = n
            return
    elems.append(Element(kind=kind, order=n, vector=u, ops=[op]))


def collect_elements(group: dict[tuple, Op]) -> list[Element]:
    """Proper axes (order = the largest element order about them), improper axes S_n (n >= 3),
    mirrors and the inversion centre."""
    out: list[Element] = []
    for op in group.values():
        theta = op_angle(op)
        if op.det == 1:
            if theta > 1e-3:
                _merge(out, "C", op_axis(op), element_order(theta), op)
        elif theta < 0.1:
            _merge(out, "sigma", op_axis(op), None, op)
        elif theta > math.pi - 0.1:
            out.append(Element(kind="i", order=None, vector=None, ops=[op]))
        else:
            n = round(2 * math.pi / theta)
            if abs(2 * math.pi / theta - n) < 0.15 and n >= 3:
                _merge(out, "S", op_axis(op), n, op)
    return out


def _parallel(a: Element, b: np.ndarray) -> bool:
    return abs(float(a.vector @ b)) > AXIS_MERGE_COS


def _perpendicular(a: Element, b: np.ndarray) -> bool:
    return abs(float(a.vector @ b)) < PERP_SIN


def _has_c2(e: Element) -> bool:
    return any(abs(op_angle(o) - math.pi) < 0.1 for o in e.ops)


def choose_principal(axes: list[Element], improper: list[Element]) -> Element:
    """Highest order; ties: an S_2n along it, then the most perpendicular C2 axes."""
    top = max(a.order for a in axes)

    def rank(a: Element) -> tuple:
        s2n = any(s.order == 2 * a.order and _parallel(s, a.vector) for s in improper)
        perp = sum(1 for b in axes if b is not a and _perpendicular(b, a.vector) and _has_c2(b))
        return (s2n, perp)

    return max((a for a in axes if a.order == top), key=rank)


def name_group(elems: list[Element]) -> tuple[str, Element | None]:
    """The Schoenflies symbol and the principal axis (None for the cubic, icosahedral and
    axis-free groups)."""
    axes = [e for e in elems if e.kind == "C"]
    mirrors = [e for e in elems if e.kind == "sigma"]
    improper = [e for e in elems if e.kind == "S"]
    inv = any(e.kind == "i" for e in elems)
    if sum(1 for a in axes if a.order >= 3) >= 2:
        orders = {a.order for a in axes}
        if 5 in orders:
            return ("Ih" if inv else "I"), None
        if 4 in orders:
            return ("Oh" if inv else "O"), None
        return ("Th" if inv else "Td" if mirrors else "T"), None
    if not axes:
        return ("Cs" if mirrors else "Ci" if inv else "C1"), None
    p = choose_principal(axes, improper)
    n = p.order
    perp_c2 = [a for a in axes if a is not p and _perpendicular(a, p.vector) and _has_c2(a)]
    sigma_h = any(_parallel(m, p.vector) for m in mirrors)
    sigma_v = sum(1 for m in mirrors if _perpendicular(m, p.vector))
    if len(perp_c2) >= n:
        return (f"D{n}h" if sigma_h else f"D{n}d" if sigma_v >= n else f"D{n}"), p
    if sigma_h:
        return f"C{n}h", p
    if sigma_v >= n:
        return f"C{n}v", p
    if any(s.order == 2 * n and _parallel(s, p.vector) for s in improper):
        return f"S{2 * n}", p
    return f"C{n}", p


def expected_order(name: str) -> int | None:
    if name in EXPECTED_ORDER:
        return EXPECTED_ORDER[name]
    if name in ("C1",):
        return 1
    if name in ("Cs", "Ci"):
        return 2
    family, rest = name[0], name[1:]
    digits = "".join(ch for ch in rest if ch.isdigit())
    if not digits:
        return None
    n, suffix = int(digits), rest[len(digits) :]
    if family == "S":
        return n
    if family == "C":
        return n if suffix == "" else 2 * n
    return 2 * n if suffix == "" else 4 * n


def conjugacy_classes(group: dict[tuple, Op]) -> dict[tuple, int]:
    ops = list(group.values())
    inverse = {o.key: np.argsort(o.perm) for o in ops}
    cls: dict[tuple, int] = {}
    for g in ops:
        if g.key in cls:
            continue
        c = len(set(cls.values()))
        for h in ops:
            cls[(g.det, h.perm[g.perm[inverse[h.key]]].tobytes())] = c
    return cls


def _on_element(f: Frame, e: Element) -> tuple[int, ...]:
    x = f.x
    if e.kind == "i":
        d = np.linalg.norm(x, axis=1)
    elif e.kind == "sigma":
        d = np.abs(x @ e.vector)
    else:
        d = np.linalg.norm(x - np.outer(x @ e.vector, e.vector), axis=1)
    return tuple(int(i) for i in np.flatnonzero(d <= f.tol))


def _class_of(e: Element, classes: dict[tuple, int]) -> int:
    ops = [o for o in e.ops if e.kind != "C" or abs(op_angle(o) - math.pi) < 0.1] or e.ops
    return min(classes.get(o.key, 0) for o in ops)


def _split(elems: list[Element], classes: dict[tuple, int], labels: tuple[str, ...]) -> None:
    """Label conjugacy classes in order of the atoms they hold (most first)."""
    by: dict[int, list[Element]] = {}
    for e in elems:
        by.setdefault(_class_of(e, classes), []).append(e)
    order = sorted(by, key=lambda c: (-sum(len(e.atoms) for e in by[c]), c))
    for k, c in enumerate(order):
        for e in by[c]:
            e.label = labels[min(k, len(labels) - 1)]


def _label_mirrors(name: str, mirrors: list[Element], p: Element | None, axes, classes) -> None:
    if name in ("Cs", "Ih"):
        labels = {id(m): "σ" for m in mirrors}
    elif name in ("Th", "Td"):
        labels = {id(m): "σh" if name == "Th" else "σd" for m in mirrors}
    elif name == "Oh":
        c4 = [a for a in axes if a.order == 4]
        labels = {id(m): "σh" if any(_parallel(a, m.vector) for a in c4) else "σd" for m in mirrors}
    else:
        labels = {id(m): "σh" for m in mirrors if p is not None and _parallel(m, p.vector)}
        vertical = [m for m in mirrors if id(m) not in labels]
        if name.endswith("d"):
            for m in vertical:
                m.label = "σd"
        else:
            n = p.order if p is not None else 1
            _split(vertical, classes, ("σv", "σv′") if n == 2 else ("σv", "σd"))
        labels.update({id(m): m.label for m in vertical})
    for m in mirrors:
        m.label = labels[id(m)]


def label_elements(f: Frame, name: str, p: Element | None, elems: list[Element], group) -> None:
    classes = conjugacy_classes(group)
    for e in elems:
        e.atoms = _on_element(f, e)
        e.principal = e is p
        e.label = {"C": f"C{e.order}", "S": f"S{e.order}", "i": "i"}.get(e.kind, "σ")
    axes = [e for e in elems if e.kind == "C"]
    if p is not None and name.startswith("D"):
        perp = [a for a in axes if a is not p and _perpendicular(a, p.vector) and a.order == 2]
        _split(perp, classes, ("C2′", "C2″"))
    _label_mirrors(name, [e for e in elems if e.kind == "sigma"], p, axes, classes)


_KIND_RANK = {"C": 0, "S": 1, "sigma": 2, "i": 3}
_MIRROR_RANK = {"σh": 0, "σv": 1, "σv′": 2, "σd": 3, "σ": 4}


def sort_elements(elems: list[Element]) -> list[Element]:
    def key(e: Element) -> tuple:
        return (
            _KIND_RANK[e.kind],
            not e.principal,
            -(e.order or 0),
            _MIRROR_RANK.get(e.label, 5),
            e.label,
        )

    return sorted(elems, key=key)


# ---- classes and the symmetrized structure


def orbits(n: int, ops: list[Op]) -> list[list[int]]:
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for op in ops:
        for i, j in enumerate(op.perm):
            a, b = root(i), root(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(root(i), []).append(i)
    return sorted(out.values(), key=lambda c: c[0])


def symmetrize(f: Frame, ops: list[Op], rounds: int = 3) -> np.ndarray:
    """The group average of every atom's images (centred frame), the operations refitted to
    the averaged structure between rounds."""
    y = f.x.copy()
    mats = [o.R for o in ops]
    for _ in range(rounds):
        acc = np.zeros_like(y)
        for o, r in zip(ops, mats, strict=True):
            acc += y[o.perm] @ r  # R^T x_perm[i], as rows
        y = acc / len(ops)
        mats = [fit(y, o.perm, o.det)[0] for o in ops]
    return y


# ---- the search


@dataclass
class Search:
    frame: Frame
    moments: np.ndarray
    top: str
    name: str
    order: int | None  # None: an infinite group
    elements: list[Element]
    symmetrized: np.ndarray  # centred frame
    classes: list[list[int]]
    notes: list[str]


def _perpendicular_to(u: np.ndarray) -> np.ndarray:
    trial = np.eye(3)[int(np.argmin(np.abs(u)))]
    return _unit(np.cross(u, trial))


def _linear(f: Frame, u: np.ndarray, moments: np.ndarray, top: str) -> Search | None:
    """C∞v or D∞h when every atom lies within the tolerance of the axis u, else None."""
    s = f.x @ u
    if len(f.z) < 2 or np.linalg.norm(f.x - np.outer(s, u), axis=1).max() > f.tol:
        return None
    perm = match(f, -f.x, f.tol)
    every = tuple(range(len(f.z)))
    elems = [Element("C", None, u, [], "C∞", True, every)]
    elems.append(Element("sigma", None, _perpendicular_to(u), [], "∞σv", False, every))
    if perm is None:
        return Search(f, moments, top, "C∞v", None, elems, np.outer(s, u), [[i] for i in every], [])
    inv = Op(R=-np.eye(3), perm=perm, det=-1, dev=float(np.abs(s + s[perm]).max()))
    perp = _perpendicular_to(u)
    for e in (
        Element("C", 2, perp, [inv], "∞C2′"),
        Element("sigma", None, u, [inv], "σh"),
        Element("i", None, None, [inv], "i"),
    ):
        e.atoms = _on_element(f, e)
        elems.append(e)
    y = np.outer((s - s[perm]) / 2.0, u)
    return Search(f, moments, top, "D∞h", None, elems, y, orbits(len(f.z), [inv]), [])


def search(z, xyz, tol: float) -> Search:
    f = make_frame(z, xyz, tol)
    moments, axes = inertia(f)
    top = top_kind(f, moments)
    if top == "atom":
        note = "a single atom: the full rotation-reflection group"
        return Search(f, moments, top, "Kh", None, [], f.x.copy(), [[0]], [note])
    lin = _linear(f, axes[:, 0], moments, top)
    if lin is not None:
        return lin
    sets = sea_sets(f)
    nmax = min(MAX_AXIS_ORDER, max(2, max(len(s) for s in sets)))
    ops = find_operations(f, candidate_directions(f, sets, axes), nmax)
    group = best_group(f, ops)
    elems = collect_elements(group)
    name, principal = name_group(elems)
    label_elements(f, name, principal, elems, group)
    notes = []
    want = expected_order(name)
    if want is not None and want != len(group):
        notes.append(f"{name} has order {want} but the operations found form {len(group)}")
    gops = list(group.values())
    y = symmetrize(f, gops)
    return Search(
        f, moments, top, name, len(group), sort_elements(elems), y, orbits(len(f.z), gops), notes
    )


# ---- the public entry point


def check_input(z, xyz, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    if not (MIN_TOLERANCE <= float(tolerance) <= MAX_TOLERANCE):
        msg = f"tolerance {tolerance} outside {MIN_TOLERANCE}..{MAX_TOLERANCE} Angstrom"
        raise SymmetryError("REQUEST_INVALID", msg)
    z = np.asarray(z, dtype=int).reshape(-1)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(z) == 0 or len(z) != len(xyz):
        raise SymmetryError("REQUEST_INVALID", f"{len(z)} atomic numbers for {len(xyz)} positions")
    if len(z) > MAX_ATOMS:
        msg = f"{len(z)} atoms: the point-group search takes at most {MAX_ATOMS}"
        raise SymmetryError("LIMIT_EXCEEDED", msg)
    if not np.isfinite(xyz).all():
        raise SymmetryError("REQUEST_INVALID", "non-finite coordinates")
    return z, xyz


def _rank(order: int | None) -> float:
    return math.inf if order is None else float(order)


def looser_hint(z, xyz, tol: float, order: int | None) -> dict | None:
    """The highest group at a looser tolerance of HINT_TOLERANCES (its smallest such
    tolerance), when it is higher than `order`."""
    best: dict | None = None
    for t in HINT_TOLERANCES:
        if t <= tol + 1e-9:
            continue
        s = search(z, xyz, t)
        if _rank(s.order) > _rank(best["order"] if best else order):
            best = {"point_group": s.name, "order": s.order, "tolerance": t}
    return best


def _signed(v: np.ndarray) -> list[float]:
    k = int(np.argmax(np.abs(v)))
    return [float(c) for c in (v if v[k] >= 0 else -v)]


def _element_out(e: Element, k: int, com: np.ndarray) -> dict:
    return {
        "id": f"e{k}",
        "kind": e.kind,
        "label": e.label,
        "order": e.order,
        "vector": None if e.vector is None else _signed(e.vector),
        "point": [float(c) for c in com],
        "atoms": list(e.atoms),
        "deviation_ang": max((o.dev for o in e.ops), default=0.0),
        "principal": e.principal,
        "n_operations": len(e.ops),
    }


def _symbol(z: int) -> str:
    from rdkit import Chem

    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


def point_group(z, xyz, tolerance: float = DEFAULT_TOLERANCE, *, hint: bool = True) -> dict:
    """The point group of a structure: `PointGroupOut`'s fields other than molecule_id and key
    (coordinates and vectors in the input frame). Raises `SymmetryError`."""
    z, xyz = check_input(z, xyz, tolerance)
    s = search(z, xyz, float(tolerance))
    f = s.frame
    disp = np.linalg.norm(f.x - s.symmetrized, axis=1)
    notes = list(s.notes)
    looser = None
    if hint and len(z) <= HINT_MAX_ATOMS:
        looser = looser_hint(z, xyz, float(tolerance), s.order)
    elif hint:
        notes.append(f"no looser-tolerance hint above {HINT_MAX_ATOMS} atoms")
    return {
        "tolerance": float(tolerance),
        "point_group": s.name,
        "order": s.order,
        "top": s.top,
        "moments": [float(m) for m in s.moments],
        "n_atoms": len(z),
        "centre": [float(c) for c in f.com],
        "elements": [_element_out(e, k, f.com) for k, e in enumerate(s.elements)],
        "classes": [{"element": _symbol(z[c[0]]), "atoms": c} for c in s.classes],
        "rms_ang": float(np.sqrt((disp**2).mean())),
        "max_deviation_ang": float(disp.max()),
        "symmetrized_xyz": [[float(c) for c in p] for p in s.symmetrized + f.com],
        "looser": looser,
        "notes": notes,
    }
