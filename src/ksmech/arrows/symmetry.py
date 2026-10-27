"""Symmetry-equivalent edits: the label class closed under the automorphisms of the reactant pool
(spec v9, "Edit labels").

A dataset step names its atoms by atom maps, so a step on one of several equivalent atoms (one
H of H3O+ or of a methyl, an atom of either of two identical molecules) labels only the one it
drew. The model reads the pool, not the maps: equivalent atoms get equal scores, so the
probability splits among them, a top-1 hit is a coin toss, and the loss keeps a floor of log k on
such a step. On the PMechDB molecule split that was 21.5% of the val samples, with top-1 18%
there against 71% elsewhere (docs/decisions.md, 2026-10-26). The class is therefore closed:
every automorphism of the pool's Lewis structure on the token atoms (elements, formal charges,
bond orders, every H an atom) maps each variant to an equivalent one, and the images join the
class with their anchors. The structure is a drawing's, so a Kekule ring or a localized charge
breaks the symmetry it breaks in B, as it does for the label.

Automorphisms act on the pool with every H that has a single heavy neighbour folded into it
(`_fold`). Only where they send the label's atoms matters, so `placements` enumerates the
distinct images of those atoms, by backtracking within RDKit's symmetry classes (canonical ranks
without tie breaking), each checked by completing it once to a full automorphism; at most
MAX_AUTOMORPHISMS placements and MAX_STEPS candidate checks. A variant's folded H's then go to
every choice of distinct H's on the image of their heavy atom, since H's on one atom are always
interchangeable. A class keeps at most MAX_VARIANTS members; a cut-short closure is
flagged on the label (`symmetry_capped`).
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
from rdkit import Chem, rdBase

from ksmech.arrows.delta_be import EditLabel, ReactionMatrix

MAX_AUTOMORPHISMS = 1000  # placements of the label's atoms
MAX_STEPS = 200_000  # candidate checks per enumeration
MAX_VARIANTS = 32
_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def lewis_mol(z: np.ndarray, charges: np.ndarray, bonds: np.ndarray) -> Chem.Mol:
    """The Lewis structure on the token atoms as an unsanitized RDKit molecule: atom k is token
    atom k, every H an atom, no implicit H; ring info found."""
    rw = Chem.RWMol()
    for el, q in zip(z, charges, strict=True):
        atom = Chem.Atom(int(el))
        atom.SetFormalCharge(int(q))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for i, j, o in bonds:
        rw.AddBond(int(i), int(j), _TYPES[int(o)])
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def symmetry_classes(z: np.ndarray, charges: np.ndarray, bonds: np.ndarray) -> list[int]:
    """RDKit's canonical rank of every atom without tie breaking: atoms an automorphism may
    swap share one (elements, charges, bond orders; every H an atom, no stereo)."""
    mol = lewis_mol(z, charges, bonds)
    with rdBase.BlockLogs():
        return list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))


def _fold(z: np.ndarray, bonds: np.ndarray) -> tuple[dict, dict]:
    """(node -> {neighbour node: bond order}, node -> its folded H's): an H with a single
    neighbour that is not an H is folded into it; every other atom is a node."""
    nbrs: dict[int, dict[int, int]] = {a: {} for a in range(len(z))}
    for i, j, o in bonds:
        nbrs[int(i)][int(j)] = nbrs[int(j)][int(i)] = int(o)
    folded = {
        a: next(iter(n))
        for a, n in nbrs.items()
        if z[a] == 1 and len(n) == 1 and z[next(iter(n))] != 1
    }
    nodes = {
        a: {b: o for b, o in n.items() if b not in folded}
        for a, n in nbrs.items()
        if a not in folded
    }
    hs: dict[int, list[int]] = {a: [] for a in nodes}
    for h, a in sorted(folded.items()):
        hs[a].append(h)
    return nodes, hs


def _order(nodes: dict, support: set[int]) -> list[int]:
    """The support first, then every other node breadth-first from it, then the other
    components: each node after the support has a mapped neighbour when it is reached."""
    order = sorted(support)
    seen = set(order)

    def bfs(queue: list[int]) -> None:
        while queue:
            v = queue.pop(0)
            for w in sorted(nodes[v]):
                if w not in seen:
                    seen.add(w)
                    order.append(w)
                    queue.append(w)

    bfs(list(order))
    for start in sorted(nodes):
        if start not in seen:
            seen.add(start)
            order.append(start)
            bfs([start])
    return order


def placements(nodes: dict, rank: list[int], support: set[int]) -> tuple[list[dict], bool]:
    """(the distinct maps of the `support` nodes that extend to an automorphism of the node
    graph, the identity first; whether the search was cut short). Only where the support goes
    matters to the images of a label, so the rest of the graph is completed once per placement,
    not enumerated: a pool with many symmetric groups far from the step stays cheap."""
    order = _order(nodes, support)
    by_rank: dict[int, list[int]] = {}
    for v in order:
        by_rank.setdefault(rank[v], []).append(v)
    out: list[dict[int, int]] = []
    sigma: dict[int, int] = {}
    used: set[int] = set()
    budget = [MAX_STEPS]

    def fits(v: int, c: int) -> bool:
        mapped = [(u, o) for u, o in nodes[v].items() if u in sigma]
        if sum(w in used for w in nodes[c]) != len(mapped):
            return False
        return all(nodes[c].get(sigma[u]) == o for u, o in mapped)

    def extend(k: int, completing: bool) -> bool:
        """True when the mapping reached a full automorphism (completing: stop there)."""
        if k == len(support) and not completing:
            if extend(k, True):
                out.append({v: sigma[v] for v in support})
            return False
        if k == len(order):
            return True
        v = order[k]
        for c in sorted(by_rank[rank[v]], key=lambda c: c != v):  # the identity first
            budget[0] -= 1
            if budget[0] <= 0 or len(out) >= MAX_AUTOMORPHISMS:
                return False
            if c in used or not fits(v, c):
                continue
            sigma[v] = c
            used.add(c)
            done = extend(k + 1, completing)
            del sigma[v]
            used.discard(c)
            if done and completing:
                return True
        return False

    extend(0, False)
    return out, budget[0] <= 0 or len(out) >= MAX_AUTOMORPHISMS


def _images(variant: ReactionMatrix, anchors: tuple, sigma: dict, hs: dict, home: dict):
    """Every (variant, anchors) that `sigma` and a choice of distinct H's on the images of the
    folded H's' heavy atoms make of `variant`."""
    atoms = {a for i, j, _ in variant.bonds for a in (i, j)} | {i for i, _ in variant.lone}
    base = {a: sigma[a] for a in atoms if a in sigma}
    groups: dict[int, list[int]] = {}
    for h in sorted(atoms - base.keys()):
        groups.setdefault(home[h], []).append(h)
    heavy = sorted(groups)
    choices = [itertools.permutations(hs[sigma[a]], len(groups[a])) for a in heavy]
    for combo in itertools.product(*choices):
        index = dict(base)
        for a, picked in zip(heavy, combo, strict=True):
            index.update(zip(groups[a], picked, strict=True))
        image_anchors = tuple(sorted((*sorted((index[i], index[j])), s) for i, j, s in anchors))
        yield variant.mapped(index), image_anchors


def symmetric_closure(
    label: EditLabel, z: np.ndarray, charges: np.ndarray, bonds: np.ndarray
) -> EditLabel:
    """`label` (on the global atoms of a pool with elements `z`, formal charges `charges` and
    Lewis bonds `bonds`) with every image of its variants under the pool's automorphisms
    added, sorted, at most MAX_VARIANTS; `n_symmetric` counts the added members and
    `symmetry_capped` flags a closure cut short."""
    nodes, hs = _fold(z, bonds)
    home = {h: a for a, folded in hs.items() for h in folded}
    atoms = {a for v in label.variants for i, j, _ in v.bonds for a in (i, j)}
    atoms |= {i for v in label.variants for i, _ in v.lone}
    support = {home.get(a, a) for a in atoms}
    sigmas, capped = placements(nodes, symmetry_classes(z, charges, bonds), support)
    members = dict(zip(label.variants, label.anchors, strict=True))
    for sigma in sigmas:
        for variant, anchors in zip(label.variants, label.anchors, strict=True):
            for image, image_anchors in _images(variant, anchors, sigma, hs, home):
                members.setdefault(image, image_anchors)
    if len(members) > MAX_VARIANTS:
        keep = list(label.variants) + sorted(set(members) - set(label.variants))
        members = {v: members[v] for v in keep[:MAX_VARIANTS]}
        capped = True
    variants = tuple(sorted(members))
    return replace(
        label,
        variants=variants,
        anchors=tuple(members[v] for v in variants),
        n_symmetric=len(variants) - len(label.variants),
        symmetry_capped=capped,
    )
