"""The reactant's Lewis structure on the token atoms of a sample, and the check of a reaction
matrix against it (spec v8, "Edit decoder").

The structure is the reactant side as drawn, in its Kekulé form: the edit labels are computed
on it (`arrows.delta_be`). The token atoms are the QM record's, which carries every H as an
atom; an H the drawing leaves implicit is bonded (order 1) to the record atom of its heavy atom.
`reactant_lewis` gives the bonds (i < j, order), the lone pairs of every atom and the Wiberg
index of every bond; `graph_distances` the bond-graph distances (-1 between molecules);
`check_edit` whether every variant of an edit label is a valid edit of that structure and every
anchor one the decoder's masks allow.
"""

from __future__ import annotations

import numpy as np

from ksmech.arrows.apply import nonbonding_electrons
from ksmech.arrows.delta_be import EditLabel, _order
from ksmech.arrows.reactants import ReactantTokens

MAX_ORDER = 3  # a forming anchor keeps the bond order at most 3 (spec v8, Anchor)


class MissingBondIndex(ValueError):
    """A bond of the reactant has no Wiberg index in its fragment's sidecar."""


def _drawn_bonds(rt: ReactantTokens) -> dict[tuple[int, int], int]:
    out = {}
    for b in rt.mol.GetBonds():
        i, j = rt.global_atom[b.GetBeginAtomIdx()], rt.global_atom[b.GetEndAtomIdx()]
        out[(min(i, j), max(i, j))] = _order(rt.mol, b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return out


def _implicit_h_bonds(rt: ReactantTokens) -> dict[tuple[int, int], int]:
    """Single bonds from every record H the drawing leaves implicit to its neighbour."""
    out = {}
    for f, frag in enumerate(rt.fragments):
        off = int(rt.atom_offsets[f])
        drawn = set(frag.record_of.values())
        for b in frag.record.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in drawn and j in drawn:
                continue  # the drawing decides the bonds between drawn atoms
            h = j if i in drawn else i
            if frag.record.GetAtomWithIdx(h).GetAtomicNum() != 1:
                raise ValueError(f"{frag.key}: record atom {h} is neither drawn nor an H")
            out[(off + min(i, j), off + max(i, j))] = 1
    return out


def reactant_lewis(rt: ReactantTokens) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bonds (E, 3) int16 rows (i, j, order) with i < j, lone pairs (A,) int8, Wiberg index
    (E,) float16 of every bond) on the global atoms of `rt`. Raises MissingBondIndex when a
    bond has no Wiberg index in its fragment's sidecar."""
    rows = {**_implicit_h_bonds(rt), **_drawn_bonds(rt)}
    bonds = np.array([(i, j, o) for (i, j), o in sorted(rows.items())], dtype=np.int16)
    bonds = bonds.reshape(-1, 3)
    lone = np.zeros(rt.n_atoms, dtype=np.int8)  # implicit H atoms carry none
    for m, g in rt.global_atom.items():
        lone[g] = nonbonding_electrons(rt.mol.GetAtomWithIdx(m)) // 2
    wiberg = np.zeros(len(bonds), dtype=np.float16)
    for k, (i, j, _) in enumerate(bonds):
        w = rt.bond_index(int(i), int(j))
        if w is None:
            raise MissingBondIndex(f"no Wiberg index for the bond {int(i)}-{int(j)}")
        wiberg[k] = w
    return bonds, lone, wiberg


def graph_distances(bonds: np.ndarray, n_atoms: int) -> np.ndarray:
    """(A, A) int16 bond-graph distances, -1 between atoms of different molecules."""
    adj = np.zeros((n_atoms, n_atoms), dtype=np.uint8)
    adj[bonds[:, 0], bonds[:, 1]] = adj[bonds[:, 1], bonds[:, 0]] = 1
    dist = np.full((n_atoms, n_atoms), -1, dtype=np.int16)
    np.fill_diagonal(dist, 0)
    reached = np.eye(n_atoms, dtype=bool)
    frontier, d = reached.copy(), 0
    while frontier.any():
        d += 1
        frontier = ((frontier.astype(np.uint8) @ adj) > 0) & ~reached
        dist[frontier] = d
        reached |= frontier
    return dist


def check_edit(label: EditLabel, bonds: np.ndarray, lone_pairs: np.ndarray) -> str | None:
    """Why `label` is not a valid edit of the structure (None when it is): every variant
    conserves the electrons (CHARGE), keeps every bond order in [0, 3] (BOND_ORDER) and every
    lone-pair count non-negative (LONE_PAIRS); every forming anchor keeps its order at most 3
    and every breaking anchor is a bond (ANCHOR_MASK)."""
    order = {(int(i), int(j)): int(o) for i, j, o in bonds}
    for variant, anchors in zip(label.variants, label.anchors, strict=True):
        if sum(d for _, d in variant.lone) + 2 * sum(d for *_, d in variant.bonds) != 0:
            return "CHARGE"
        if any(not 0 <= order.get((i, j), 0) + d <= MAX_ORDER for i, j, d in variant.bonds):
            return "BOND_ORDER"
        if any(2 * int(lone_pairs[i]) + d < 0 for i, d in variant.lone):
            return "LONE_PAIRS"
        for i, j, kind in anchors:
            o = order.get((i, j), 0)
            if (kind == "form" and o + 1 > MAX_ORDER) or (kind == "break" and o == 0):
                return "ANCHOR_MASK"
    return None
