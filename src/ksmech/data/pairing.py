"""The drawing a species' spin decision asks for (open-shell plan P8, step 3): a species the
species table decides singlet but drawn with unpaired electrons (odd-N atoms) is redrawn as a
closed-shell Lewis structure (PAIRED_DRAWING: [N]=[N] -> N#N, [O]O[O] -> [O-][O+]=O, the
Criegee [CH2]O[O] -> C=[O+][O-]), and a Hund-case species drawn paired is redrawn unpaired
(HIGH_SPIN_DRAWING: O=O -> [O][O]).

`redraw` searches the structures on the drawing's sigma skeleton, with its H and total charge,
whose number of odd-N atoms is the target (0 for a singlet), changing only the orders of the
bonds that touch a drawn radical atom (each by at most 1) and the formal charges of those atoms
and their neighbours (`_region`), and reachable
by electron moves between bonded atoms: the changed bonds form components, an atom whose N
changes joins a component it lies on or is bonded to, and each component keeps its electrons
(`_reachable`). A charge that would jump between two atoms no changed bond joins (p-benzyne
into a zwitterion) is not a move, so a distant biradical has no paired drawing. Among the
candidates the first by (most complete octets, fewest formal charges,
the charges on the more electronegative atoms (least sum q x chi, Pauling), fewest changed
bonds, atom order) wins: the Criegee intermediate is C=[O+][O-], not [CH2-][O+]=O. Every
number comes from the drawing: no list of species.
"""

from __future__ import annotations

import itertools

from rdkit import Chem

from ksmech.arrows.apply import ORDER_TO_TYPE, TYPE_TO_ORDER, nonbonding_electrons

# Pauling electronegativities of the main-group elements a drawing may hold
PAULING = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58,
           17: 3.16, 34: 2.55, 35: 2.96, 53: 2.66}  # fmt: skip
MAX_REGION_ATOMS = 8  # atoms whose charge the search may change
MAX_REGION_BONDS = 8  # bonds whose order it may change


def _n(atom: Chem.Atom, charge: int, order_sum: int) -> int:
    v = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
    return v - charge - order_sum - atom.GetTotalNumHs()


def _region(mol: Chem.Mol, target_odd: int) -> tuple[list[int], list[int]]:
    """(atoms, bonds) the search may change: the odd-N heavy atoms with their heavy neighbours,
    and the bonds that touch an odd-N atom (so a radical centre pairs with a bonded partner or
    through one neighbour, never by rearranging a ring between two distant centres). A drawing
    with no odd-N atom redrawn unpaired (`target_odd` > 0) opens its whole heavy skeleton."""
    odd = [a.GetIdx() for a in mol.GetAtoms() if nonbonding_electrons(a) % 2]
    if not odd and target_odd:
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        bonds = [
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1
        ]
        return heavy, bonds
    atoms = set(odd)
    for i in odd:
        atoms |= {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() > 1}
    bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if (b.GetBeginAtomIdx() in odd or b.GetEndAtomIdx() in odd)
        and b.GetBeginAtom().GetAtomicNum() > 1
        and b.GetEndAtom().GetAtomicNum() > 1
    ]
    return sorted(atoms), bonds


def _reachable(mol: Chem.Mol, d_order: dict[int, int], d_n: dict[int, int]) -> bool:
    """The change (bond index -> d order, atom -> dN) moves electrons between bonded atoms
    only (module docstring)."""
    parent: dict[int, int] = {}

    def root(a: int) -> int:
        while parent.setdefault(a, a) != a:
            a = parent[a]
        return a

    changed = [mol.GetBondWithIdx(k) for k, d in d_order.items() if d]
    for b in changed:
        parent[root(b.GetBeginAtomIdx())] = root(b.GetEndAtomIdx())
    on_bond = set(parent)
    for a, d in d_n.items():
        if d and a not in on_bond:
            near = [
                n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in on_bond
            ]
            if not near:
                return False
            parent[root(a)] = root(near[0])
    total: dict[int, int] = {}
    for b in changed:
        r = root(b.GetBeginAtomIdx())
        total[r] = total.get(r, 0) + 2 * d_order[b.GetIdx()]
    for a, d in d_n.items():
        if d:
            total[root(a)] = total.get(root(a), 0) + d
    return not any(total.values())


def _octet_ok(atom: Chem.Atom, n: int, order_sum: int) -> bool:
    electrons = n + 2 * (order_sum + atom.GetTotalNumHs())
    return electrons <= 8 if atom.GetAtomicNum() <= 10 else True


def _candidates(mol: Chem.Mol, atoms: list[int], bonds: list[int], target_odd: int):
    """Yield (score, orders, charges) of every structure the search admits."""
    drawn_order = {k: TYPE_TO_ORDER.get(mol.GetBondWithIdx(k).GetBondType(), 1) for k in bonds}
    choices = [
        [o for o in (drawn_order[k] - 1, drawn_order[k], drawn_order[k] + 1) if 1 <= o <= 3]
        for k in bonds
    ]
    q_total = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)
    n_drawn = {i: nonbonding_electrons(mol.GetAtomWithIdx(i)) for i in atoms}
    others = sum(nonbonding_electrons(a) % 2 for a in mol.GetAtoms() if a.GetIdx() not in atoms)
    for orders in itertools.product(*choices):
        order = dict(zip(bonds, orders, strict=True))
        yield from _charge_choices(mol, atoms, order, q_total, n_drawn, target_odd - others)


def _order_sum(mol: Chem.Mol, i: int, order: dict[int, int]) -> int:
    return sum(
        order.get(b.GetIdx(), TYPE_TO_ORDER.get(b.GetBondType(), 1))
        for b in mol.GetAtomWithIdx(i).GetBonds()
    )


def _charge_choices(mol, atoms, order, q_total, n_drawn, target_odd):
    per_atom = []
    for i in atoms:
        atom, s = mol.GetAtomWithIdx(i), _order_sum(mol, i, order)
        opts = [(q, _n(atom, q, s)) for q in (-1, 0, 1)]
        per_atom.append([(q, n) for q, n in opts if n >= 0 and _octet_ok(atom, n, s)])
    for picked in itertools.product(*per_atom):
        if sum(q for q, _ in picked) != q_total or sum(n % 2 for _, n in picked) != target_odd:
            continue
        d_n = {i: n - n_drawn[i] for i, (_, n) in zip(atoms, picked, strict=True)}
        d_order = {k: o - TYPE_TO_ORDER.get(mol.GetBondWithIdx(k).GetBondType(), 1)
                   for k, o in order.items()}  # fmt: skip
        if not any(d_n.values()) and not any(d_order.values()):
            continue
        if not _reachable(mol, d_order, d_n):
            continue
        octets = sum(
            _octet_complete(mol.GetAtomWithIdx(i), n, _order_sum(mol, i, order))
            for i, (_, n) in zip(atoms, picked, strict=True)
        )
        charged = sum(q != 0 for q, _ in picked)
        polar = sum(
            q * _chi(mol.GetAtomWithIdx(i)) for i, (q, _) in zip(atoms, picked, strict=True)
        )
        score = (-octets, charged, round(polar, 6), sum(1 for d in d_order.values() if d))
        yield score, order, {i: q for i, (q, _) in zip(atoms, picked, strict=True)}


def _chi(atom: Chem.Atom) -> float:
    return PAULING.get(atom.GetAtomicNum(), 2.0)


def _octet_complete(atom: Chem.Atom, n: int, order_sum: int) -> bool:
    return n + 2 * (order_sum + atom.GetTotalNumHs()) == (2 if atom.GetAtomicNum() <= 2 else 8)


def redraw(mol: Chem.Mol, target_unpaired: int) -> Chem.Mol | None:
    """`mol` (one fragment, mapped or not) redrawn with `target_unpaired` odd-N atoms (module
    docstring), or None when no structure the search admits has them (a distant biradical,
    a region too large to search, a fragment with a metal)."""
    from ksmech.data.domain import METALS  # domain reads this module

    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    if any(a.GetAtomicNum() in METALS for a in m.GetAtoms()):
        return None  # a metal's spin is the metal table's (P4), never a redrawing
    atoms, bonds = _region(m, target_unpaired)
    if not atoms or len(atoms) > MAX_REGION_ATOMS or len(bonds) > MAX_REGION_BONDS:
        return None
    best = min(_candidates(m, atoms, bonds, target_unpaired), key=lambda c: c[0], default=None)
    if best is None:
        return None
    _, order, charges = best
    rw = Chem.RWMol(m)
    for k, o in order.items():
        rw.GetBondWithIdx(k).SetBondType(ORDER_TO_TYPE[o])
    for i, q in charges.items():
        rw.GetAtomWithIdx(i).SetFormalCharge(q)
    for a in rw.GetAtoms():
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(mol.GetAtomWithIdx(a.GetIdx()).GetTotalNumHs())
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    for a in out.GetAtoms():
        a.SetNumRadicalElectrons(nonbonding_electrons(a) % 2)
    Chem.SanitizeMol(out)
    return out
