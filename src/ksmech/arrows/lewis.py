"""The reactant's Lewis structure on the token atoms of a sample, and the check of a reaction
matrix against it (spec v8, "Edit decoder").

The structure is the reactant side as drawn, in its Kekulé form: the edit labels are computed
on it (`arrows.delta_be`). The token atoms are the QM record's, which carries every H as an
atom; an H the drawing leaves implicit is bonded (order 1) to the record atom of its heavy atom.
`reactant_lewis` gives the bonds (i < j, order), the lone pairs of every atom, the Wiberg index
of every bond, and (open-shell plan P6) the non-bonding electrons N of every atom (B_ii; the
lone pairs are N // 2) and its unpaired count u (`data.spin.spin_of_drawing`: the odd-N atoms,
raised where the species table holds a species as high spin); `lewis_bonds`, `lewis_nonbonding`
and `lewis_unpaired` the same for another resonance drawing of the side (`arrows.drawings`);
`pair_bond_orders` the pair feature `bond_order` of the orbital pairs on a structure;
`graph_distances` the bond-graph distances (-1 between molecules); `check_edit` whether every
variant of an edit label is a valid edit of that structure and every anchor one the decoder's
masks allow.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from rdkit import Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ksmech.arrows.apply import nonbonding_electrons
from ksmech.arrows.delta_be import EditLabel, _order
from ksmech.arrows.reactants import ReactantTokens
from ksmech.data.spin import spin_of_drawing

MAX_ORDER = 3  # a forming anchor keeps the bond order at most 3 (spec v8, Anchor)


class MissingBondIndex(ValueError):
    """A bond of the reactant has no Wiberg index in its fragment's sidecar."""


def _drawn_bonds(rt: ReactantTokens, mol: Chem.Mol) -> dict[tuple[int, int], int]:
    out = {}
    for b in mol.GetBonds():
        i, j = rt.global_atom[b.GetBeginAtomIdx()], rt.global_atom[b.GetEndAtomIdx()]
        out[(min(i, j), max(i, j))] = _order(mol, b.GetBeginAtomIdx(), b.GetEndAtomIdx())
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


def lewis_bonds(rt: ReactantTokens, mol: Chem.Mol | None = None) -> np.ndarray:
    """(E, 3) int16 rows (i, j, order), i < j, on the global atoms of `rt`: the bonds of `mol`
    (a drawing of `rt.mol` on its atom indices; `rt.mol` itself by default) and the single
    bonds of the H atoms it leaves implicit."""
    rows = {**_implicit_h_bonds(rt), **_drawn_bonds(rt, rt.mol if mol is None else mol)}
    bonds = np.array([(i, j, o) for (i, j), o in sorted(rows.items())], dtype=np.int16)
    return bonds.reshape(-1, 3)


def lewis_nonbonding(rt: ReactantTokens, mol: Chem.Mol | None = None) -> np.ndarray:
    """(A,) int8 non-bonding electrons N of every global atom of `rt` in the drawing `mol`
    (`rt.mol` by default); the implicit H atoms carry none."""
    mol = rt.mol if mol is None else mol
    n = np.zeros(rt.n_atoms, dtype=np.int8)
    for m, g in rt.global_atom.items():
        n[g] = nonbonding_electrons(mol.GetAtomWithIdx(m))
    return n


def lewis_lone_pairs(rt: ReactantTokens, mol: Chem.Mol | None = None) -> np.ndarray:
    """(A,) int8 lone pairs (N // 2) of every global atom of `rt` in the drawing `mol`."""
    return lewis_nonbonding(rt, mol) // 2


def lewis_unpaired(rt: ReactantTokens, mol: Chem.Mol | None = None) -> np.ndarray:
    """(A,) int8 unpaired count of every global atom of `rt` in the drawing `mol` (`rt.mol` by
    default): `spin.spin_of_drawing` of each drawn fragment on its N; the implicit H atoms
    carry none."""
    mol = rt.mol if mol is None else mol
    u = np.zeros(rt.n_atoms, dtype=np.int8)
    mapping: list = []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping)
    for atoms, frag in zip(mapping, frags, strict=True):
        n = [nonbonding_electrons(mol.GetAtomWithIdx(m)) for m in atoms]
        for m, x in zip(atoms, spin_of_drawing(frag, n)[1], strict=True):
            if m in rt.global_atom:
                u[rt.global_atom[m]] = x
    return u


class LewisStructure(NamedTuple):
    """The reactant's Lewis structure on the global atoms (`reactant_lewis`)."""

    bonds: np.ndarray  # (E, 3) int16 rows (i, j, order), i < j
    lone_pairs: np.ndarray  # (A,) int8, N // 2
    wiberg: np.ndarray  # (E,) float16
    nonbonding: np.ndarray  # (A,) int8 N
    unpaired: np.ndarray  # (A,) int8 u


def reactant_lewis(rt: ReactantTokens) -> LewisStructure:
    """The Lewis structure of the side as drawn on the global atoms of `rt`: bonds, lone pairs,
    the Wiberg index of every bond, N and u of every atom. Raises MissingBondIndex when a bond
    has no Wiberg index in its fragment's sidecar."""
    bonds, nonbonding = lewis_bonds(rt), lewis_nonbonding(rt)
    wiberg = np.zeros(len(bonds), dtype=np.float16)
    for k, (i, j, _) in enumerate(bonds):
        w = rt.bond_index(int(i), int(j))
        if w is None:
            raise MissingBondIndex(f"no Wiberg index for the bond {int(i)}-{int(j)}")
        wiberg[k] = w
    return LewisStructure(bonds, nonbonding // 2, wiberg, nonbonding, lewis_unpaired(rt))


def pair_bond_orders(
    bonds: np.ndarray, incidence: np.ndarray, occupied: np.ndarray, n_atoms: int
) -> np.ndarray:
    """(n_occ, n_virt) float32 the pair feature `bond_order` on the structure `bonds`: the
    largest bond order between a centre of the occupied token and a centre of the virtual one
    (`physics.stage_a`, which reads it from the record's own Kekule structure). The centres are
    the rows of `incidence` (orbital j, atom g, weight); atoms of two molecules share no bond."""
    centre = np.zeros((len(occupied), n_atoms), dtype=np.float32)
    centre[incidence[:, 0].astype(np.int64), incidence[:, 1].astype(np.int64)] = 1.0
    order = np.zeros((n_atoms, n_atoms), dtype=np.int8)
    i, j = bonds[:, 0].astype(np.int64), bonds[:, 1].astype(np.int64)
    order[i, j] = order[j, i] = bonds[:, 2]
    occ, virt = centre[occupied], centre[~occupied]
    out = np.zeros((len(occ), len(virt)), dtype=np.float32)
    for k in range(1, MAX_ORDER + 1):
        out[occ @ (order >= k).astype(np.float32) @ virt.T > 0] = k
    return out


def graph_distances(bonds: np.ndarray, n_atoms: int) -> np.ndarray:
    """(A, A) int16 bond-graph distances, -1 between atoms of different molecules (breadth-first
    search per atom: an integer matrix product per level took 0.8 s on a 411-atom pool)."""
    i, j = bonds[:, 0].astype(np.int64), bonds[:, 1].astype(np.int64)
    adj = csr_matrix((np.ones(len(i)), (i, j)), shape=(n_atoms, n_atoms))
    d = shortest_path(adj, directed=False, unweighted=True)
    return np.where(np.isfinite(d), d, -1).astype(np.int16)


def check_edit(
    label: EditLabel, bonds: np.ndarray, nonbonding: np.ndarray, open_shell: bool = False
) -> str | None:
    """Why `label` is not a valid edit of the structure (None when it is): every variant is an
    elementary step of the source's rule (NOT_ELEMENTARY, `ReactionMatrix.is_elementary`;
    `open_shell` is `domain.open_shell_labels`), conserves the electrons (CHARGE), keeps every
    bond order in [0, 3] (BOND_ORDER) and every atom's non-bonding electrons N + d
    non-negative (LONE_PAIRS); every forming anchor keeps its order at most 3 and every
    breaking anchor is a bond (ANCHOR_MASK). `nonbonding` is N of every atom."""
    order = {(int(i), int(j)): int(o) for i, j, o in bonds}
    for variant, anchors in zip(label.variants, label.anchors, strict=True):
        if not variant.is_elementary(open_shell):
            return "NOT_ELEMENTARY"
        if sum(d for _, d in variant.lone) + 2 * sum(d for *_, d in variant.bonds) != 0:
            return "CHARGE"
        if any(not 0 <= order.get((i, j), 0) + d <= MAX_ORDER for i, j, d in variant.bonds):
            return "BOND_ORDER"
        if any(int(nonbonding[i]) + d < 0 for i, d in variant.lone):
            return "LONE_PAIRS"
        for i, j, kind in anchors:
            o = order.get((i, j), 0)
            if (kind == "form" and o + 1 > MAX_ORDER) or (kind == "break" and o == 0):
                return "ANCHOR_MASK"
    return None
